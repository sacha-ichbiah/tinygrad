"""
TinyPhysics - A Compiler for the Laws of Physics

Core Philosophy:
- Physics = Energy + Geometry
- User defines ONLY the Hamiltonian H(q, p)
- Forces are derived automatically via autograd: F = -dH/dq
- The Poisson structure transforms gradients into equations of motion:
    dq/dt = +dH/dp
    dp/dt = -dH/dq

This is the "native language" of physics on AI hardware.
"""

import math
from tinygrad.engine.jit import TinyJit, JitError
from tinygrad.engine.realize import capturing
from tinygrad.helpers import getenv, CAPTURING
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import dtypes, AddrSpace, PtrDType
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import AxisType, GroupOp, KernelInfo, Ops, UOp, PatternMatcher, UPat, graph_rewrite, resolve
from tinygrad.uop.symbolic import symbolic
from typing import Callable
from dataclasses import dataclass
import time

# ============================================================================
# CORE: HAMILTONIAN MECHANICS VIA AUTOGRAD
# ============================================================================

_grad_H_jit_cache: dict[tuple, TinyJit] = {}
_grad_LP_jit_cache: dict[tuple, TinyJit] = {}
_implicit_mid_buf_cache: dict[tuple[str, tuple[int, ...], object], tuple[Tensor, Tensor, Tensor, Tensor]] = {}
_implicit_iters_cache: dict[tuple[float, int, int], int] = {}


class SymplecticPolicy:
    def __init__(self, accuracy: str = "balanced", scan: bool = True, tune: bool = False,
                 max_unroll: int = 16, min_unroll: int = 2, drift_target: float | None = None,
                 budget_ms: float | None = None):
        self.accuracy = accuracy
        self.scan = scan
        self.tune = tune
        self.max_unroll = max_unroll
        self.min_unroll = min_unroll
        self.drift_target = drift_target
        self.budget_ms = budget_ms
        self._unroll_cache: dict[tuple, int] = {}
        self._last_report: dict[tuple, dict] = {}
        self._vec_cache: dict[tuple, int] = {}
        self._proj_cache: dict[tuple, int] = {}

    def _record(self, steps: int, shape: tuple[int, ...], device: str, extra: dict):
        key = (steps, shape, device, self.accuracy, self.scan, self.tune, self.max_unroll, self.min_unroll)
        if key in self._last_report:
            self._last_report[key].update(extra)
        else:
            self._last_report[key] = extra

    def _heuristic_unroll(self, steps: int, shape: tuple[int, ...], device: str) -> int:
        numel = 1
        for s in shape:
            try:
                numel *= int(s)
            except Exception:
                numel *= 1
        is_cpu = "CPU" in device.upper()
        if numel <= 1024:
            base = 16 if is_cpu else 8
        elif numel <= 16384:
            base = 8 if is_cpu else 4
        else:
            base = 4 if is_cpu else 2
        base = max(self.min_unroll, min(self.max_unroll, base))
        if steps % base != 0:
            for u in [16, 8, 4, 2]:
                if u < self.min_unroll or u > self.max_unroll: continue
                if steps % u == 0:
                    return u
        return base

    def choose_unroll(self, steps: int, shape: tuple[int, ...], device: str,
                      candidates: list[int] | None = None, step_factory: Callable | None = None,
                      op_cost: int | None = None) -> int:
        key = (steps, shape, device, self.accuracy, self.scan, self.tune, self.max_unroll, self.min_unroll, op_cost)
        cached = self._unroll_cache.get(key)
        if cached is not None:
            return cached
        if candidates is None:
            candidates = [2, 4, 8, 16]
        candidates = [u for u in candidates if self.min_unroll <= u <= self.max_unroll and steps % u == 0]
        if self.tune and step_factory is not None and candidates:
            best = None
            best_t = None
            trial_steps = min(steps, int(getenv("TINYGRAD_PHYSICS_TUNE_STEPS", 200)))
            for u in candidates:
                step = step_factory(u)
                start = time.perf_counter()
                for _ in range(trial_steps // u):
                    step()
                elapsed = time.perf_counter() - start
                if best_t is None or elapsed < best_t:
                    best_t = elapsed
                    best = u
            if best is not None:
                self._unroll_cache[key] = best
                self._last_report[key] = {"mode": "tune", "unroll": best, "trial_steps": trial_steps}
                return best
        chosen = self._heuristic_unroll(steps, shape, device)
        if op_cost is not None:
            if op_cost > 512:
                chosen = min(chosen, 2)
            elif op_cost > 256:
                chosen = min(chosen, 4)
        self._unroll_cache[key] = chosen
        self._last_report[key] = {"mode": "heuristic", "unroll": chosen, "op_cost": op_cost}
        return chosen

    def should_scan(self, steps: int, shape: tuple[int, ...], device: str) -> bool:
        if not self.scan:
            return False
        if steps < self.min_unroll:
            return False
        return True

    def projection_stride(self, constraint: str) -> int:
        strict_drift = self.drift_target is not None and self.drift_target <= 1e-6
        if self.accuracy in ("precise", "ultra_precise") or strict_drift:
            return 1
        if self.accuracy == "fast":
            return 4
        return 2

    def choose_vector_width(self, shape: tuple[int, ...], device: str, vec_dim: int) -> int:
        key = (shape, device, vec_dim, self.accuracy, self.drift_target)
        cached = self._vec_cache.get(key)
        if cached is not None:
            return cached
        if not shape or shape[-1] != vec_dim:
            self._vec_cache[key] = 1
            return 1
        if "CPU" not in device.upper():
            self._vec_cache[key] = 1
            return 1
        total = 1
        for s in shape:
            try:
                total *= int(s)
            except Exception:
                total *= 1
        if total % (8 * vec_dim) == 0:
            self._vec_cache[key] = 8
            return 8
        if total % (4 * vec_dim) == 0:
            self._vec_cache[key] = 4
            return 4
        if total % (2 * vec_dim) == 0:
            self._vec_cache[key] = 2
            return 2
        self._vec_cache[key] = 1
        return 1

    def choose_projection_stride(self, steps: int, shape: tuple[int, ...], device: str, constraint: str,
                                 state: tuple[Tensor, ...], step_fn: Callable, constraint_fn: Callable,
                                 candidates: tuple[int, ...] = (1, 2, 4, 8)) -> int:
        if not self.tune and not getenv("TINYGRAD_PHYSICS_PROJ_TUNE", 0):
            stride = self.projection_stride(constraint)
            key = (constraint, steps, shape, device, self.accuracy, self.drift_target, candidates)
            self._proj_cache[key] = stride
            return stride
        key = (constraint, steps, shape, device, self.accuracy, self.drift_target, candidates)
        cached = self._proj_cache.get(key)
        if cached is not None:
            return cached
        strict_drift = self.drift_target is not None and self.drift_target <= 1e-6
        if self.accuracy in ("precise", "ultra_precise") or strict_drift:
            target = 1e-6
        elif self.accuracy == "fast":
            target = 1e-3
        else:
            target = 1e-4
        probe_steps = min(steps, int(getenv("TINYGRAD_PHYSICS_PROJ_STEPS", 200)))
        best = candidates[0]
        for stride in sorted(candidates, reverse=True):
            s = tuple(x.detach() for x in state)
            max_drift = 0.0
            for i in range(probe_steps):
                s = step_fn(s, i, stride)
                s = tuple(x.detach().realize() for x in s)
                val = constraint_fn(s)
                try:
                    v = float(val.numpy())
                except Exception:
                    v = float(val)
                drift = abs(v - 1.0)
                if drift > max_drift:
                    max_drift = drift
                if max_drift > target:
                    break
            if max_drift <= target:
                best = stride
                break
        self._proj_cache[key] = best
        return best

    def adjust_dt(self, steps: int, shape: tuple[int, ...], device: str, state: tuple[Tensor, ...],
                  make_step: Callable[[float], Callable], energy_fn: Callable, drift_env_key: str) -> tuple[float, float | None]:
        if self.drift_target is None or not getenv(drift_env_key, 1):
            return 1.0, None
        probe_steps = min(steps, int(getenv("TINYGRAD_PHYSICS_DRIFT_STEPS", 200)))
        max_adjust = int(getenv("TINYGRAD_PHYSICS_DRIFT_ADJUSTS", 3))
        dt_scale = 1.0
        drift = None
        for _ in range(max_adjust + 1):
            step_fn = make_step(dt_scale)
            drift = _energy_drift_probe(step_fn, energy_fn, state, probe_steps)
            if drift <= self.drift_target:
                break
            dt_scale *= 0.5
        if drift is not None:
            self._record(steps, shape, device, {"dt_scale": dt_scale, "drift": drift, "probe_steps": probe_steps})
        return dt_scale, drift

    def report(self, steps: int, shape: tuple[int, ...], device: str) -> dict | None:
        key = (steps, shape, device, self.accuracy, self.scan, self.tune, self.max_unroll, self.min_unroll)
        return self._last_report.get(key)

    def maybe_print_report(self, steps: int, shape: tuple[int, ...], device: str):
        if getenv("TINYGRAD_PHYSICS_REPORT", 0):
            rep = self.report(steps, shape, device)
            if rep is not None:
                print(f"Policy: {rep}")


def compile_system(kind: str, *, H=None, policy: SymplecticPolicy | None = None,
                   integrator: str = "auto", **kwargs):
    if policy is None:
        from tinygrad.physics_profile import get_default_profile
        policy = get_default_profile().policy
    kind = kind.lower()
    if kind in ("canonical", "hamiltonian"):
        if H is None:
            raise ValueError("Hamiltonian H is required for canonical systems")
        return HamiltonianSystem(H, integrator=integrator, policy=policy)
    if kind in ("so3", "lie_poisson"):
        if H is None:
            raise ValueError("Hamiltonian H is required for Lie-Poisson systems")
        return LiePoissonSystem(H, algebra="so3", integrator=integrator, policy=policy)
    if kind in ("rigid_body", "so3_rigid"):
        I = kwargs.get("I")
        if I is None:
            raise ValueError("Inertia tensor I is required for rigid_body systems")
        return RigidBodySystem(I, integrator=integrator, policy=policy)
    if kind in ("heavy_top", "e3"):
        Htop = kwargs.get("hamiltonian", H)
        if Htop is None:
            I1 = kwargs.get("I1")
            I2 = kwargs.get("I2")
            I3 = kwargs.get("I3")
            mgl = kwargs.get("mgl")
            if None in (I1, I2, I3, mgl):
                raise ValueError("Heavy top requires hamiltonian or I1/I2/I3/mgl")
            Htop = HeavyTopHamiltonian(I1, I2, I3, mgl, dtype=kwargs.get("dtype", dtypes.float64))
        dt = kwargs.get("dt", 0.001)
        return HeavyTopIntegrator(Htop, dt, policy=policy)
    if kind in ("satellite_control", "control"):
        I_inv = kwargs.get("I_inv")
        control = kwargs.get("control")
        dt = kwargs.get("dt")
        if I_inv is None or control is None or dt is None:
            raise ValueError("satellite_control requires I_inv, control, and dt")
        return SatelliteControlIntegrator(I_inv, control, dt, policy=policy)
    raise ValueError(f"Unknown system kind: {kind}")


def simulate_hamiltonian(H, q: Tensor, p: Tensor, dt: float, steps: int,
                         policy: SymplecticPolicy | None = None,
                         record_every: int = 1) -> tuple[Tensor, Tensor, list]:
    system = compile_system("canonical", H=H, policy=policy, integrator="auto")
    return system.evolve(q, p, dt=dt, steps=steps, record_every=record_every, scan=None, unroll=None, policy=policy)


@dataclass
class StructureInfo:
    algebra: str
    separable: bool
    coupled_reduce: bool = False
    uses_sin: bool = False
    constraints: tuple[str, ...] = ()


def _energy_drift_probe(step_fn, energy_fn, state: tuple[Tensor, ...], steps: int) -> float:
    s = tuple(x.detach() for x in state)
    E0 = energy_fn(*s)
    for _ in range(steps):
        out = step_fn(*s)
        if isinstance(out, tuple):
            s = out
        else:
            s = (out,)
    E1 = energy_fn(*s)
    if E0 == 0:
        return abs(E1 - E0)
    return abs(E1 - E0) / abs(E0)

def _count_uops_limit(root: UOp, limit: int) -> int:
    seen: set[int] = set()
    stack = [root]
    count = 0
    while stack:
        u = stack.pop()
        if not isinstance(u, UOp): continue
        uid = id(u)
        if uid in seen: continue
        seen.add(uid)
        count += 1
        if count >= limit: return count
        stack.extend(u.src)
    return count


def _uop_depends_on(root: UOp, target: UOp) -> bool:
    if root is target:
        return True
    seen: set[int] = set()
    stack = [root]
    while stack:
        u = stack.pop()
        if not isinstance(u, UOp):
            continue
        uid = id(u)
        if uid in seen:
            continue
        seen.add(uid)
        if u is target:
            return True
        stack.extend(u.src)
    return False


def _uop_is_sym(u: UOp, sym: UOp) -> bool:
    return u is sym or (u.op is Ops.CAST and u.src[0] is sym)


def _uop_is_mul_sym_sym(u: UOp, sym: UOp) -> bool:
    return u.op is Ops.MUL and _uop_is_sym(u.src[0], sym) and _uop_is_sym(u.src[1], sym)


def _uop_is_reduce_sum(u: UOp, sym: UOp) -> bool:
    if u.op is not Ops.REDUCE_AXIS:
        return False
    src = u.src[0]
    return _uop_is_mul_sym_sym(src, sym)


def _uop_is_sqrt_of(u: UOp, sym: UOp) -> bool:
    if u.op is Ops.SQRT:
        return _uop_is_reduce_sum(u.src[0], sym)
    if u.op is Ops.POW and u.arg == 0.5:
        return _uop_is_reduce_sum(u.src[0], sym)
    return False


def _uop_is_rsqrt_of(u: UOp, sym: UOp) -> bool:
    if u.op is Ops.RSQRT:
        return _uop_is_reduce_sum(u.src[0], sym)
    return False


def _detect_unit_norm(root: UOp, sym: UOp) -> bool:
    for u in root.toposort():
        if u.op in (Ops.FDIV, Ops.IDIV):
            if _uop_depends_on(u.src[0], sym) and _uop_is_sqrt_of(u.src[1], sym):
                return True
        if u.op is Ops.MUL:
            if _uop_depends_on(u.src[0], sym) and _uop_is_rsqrt_of(u.src[1], sym):
                return True
            if _uop_depends_on(u.src[1], sym) and _uop_is_rsqrt_of(u.src[0], sym):
                return True
    return False


def _grad_H_key(q: Tensor, p: Tensor, H_func) -> tuple:
    return (H_func, q.shape, p.shape, q.dtype, p.dtype, q.device, p.device)


def _lower_reduce_axis(ctx: dict, x: UOp) -> UOp:
  ranges = ctx["ranges"]
  reduce_counter = ctx["reduce_counter"]
  axis = x.arg[1]
  full_reduce = len(axis) == len(ranges) and all(i in axis for i in range(len(ranges)))
  reduce_ranges = []
  for i in axis:
    reduce_ranges.append(UOp.range(ranges[i].src[0], reduce_counter[0], AxisType.REDUCE))
    reduce_counter[0] += 1
  reduce_map = {ax: rr for ax, rr in zip(axis, reduce_ranges)}
  if full_reduce:
    range_args = tuple(r.arg for r in ranges)
    range_arg_to_axis = {r.arg: i for i, r in enumerate(ranges)}
    def reindex_reduce_full(ctx: dict, uop: UOp) -> UOp|None:
      if uop.op is not Ops.INDEX or uop.arg != "value": return None
      idx = []
      changed = False
      for s in uop.src[1:]:
        if s.op is Ops.RANGE and s.arg in ctx["range_arg_to_axis"]:
          idx.append(ctx["reduce_map"].get(ctx["range_arg_to_axis"][s.arg], s))
          changed = True
        else:
          idx.append(s)
      idx = tuple(idx)
      if not changed: return None
      return uop.src[0].vindex(*idx, dtype=uop.dtype)
    src0 = graph_rewrite(
      x.src[0],
      PatternMatcher([(UPat(Ops.INDEX, name="uop"), reindex_reduce_full)], compiled=False),
      ctx={"ranges": ranges, "range_args": range_args, "range_arg_to_axis": range_arg_to_axis, "reduce_map": reduce_map},
      name="reduce_full_reindex",
    )
    replace_map = {ranges[i].arg[0]: reduce_map[i] for i in axis}
    def replace_range(ctx: dict, uop: UOp) -> UOp|None:
      if uop.op is not Ops.RANGE or uop.arg[-1] != AxisType.LOOP: return None
      rr = ctx["replace_map"].get(uop.arg[0])
      if rr is None: return None
      return rr
    src0 = graph_rewrite(
      src0,
      PatternMatcher([(UPat(Ops.RANGE, name="uop"), replace_range)], compiled=False),
      ctx={"replace_map": replace_map},
      name="reduce_full_range",
    )
  else:
    range_args = tuple(r.arg for r in ranges)
    def reindex_reduce(ctx: dict, uop: UOp) -> UOp|None:
      if uop.op is not Ops.INDEX or uop.arg != "value": return None
      if len(uop.src[1:]) != len(ctx["ranges"]): return None
      if tuple(s.arg for s in uop.src[1:]) != ctx["range_args"]: return None
      idx = tuple(ctx["reduce_map"].get(i, r) for i, r in enumerate(uop.src[1:]))
      if idx == uop.src[1:]: return None
      return uop.src[0].vindex(*idx, dtype=uop.dtype)
    src0 = graph_rewrite(
      x.src[0],
      PatternMatcher([(UPat(Ops.INDEX, name="uop"), reindex_reduce)], compiled=False),
      ctx={"ranges": ranges, "range_args": range_args, "reduce_map": reduce_map},
      name="reduce_reindex",
    )
  reduced = UOp(Ops.REDUCE, x.dtype, src=(src0,)+tuple(reduce_ranges), arg=x.arg[0])
  if full_reduce and reduced.dtype.count > 1:
    scalar = reduced.gep(0)
    for i in range(1, reduced.dtype.count):
      scalar = scalar + reduced.gep(i)
    reduced = scalar.vectorize(*([scalar] * (reduced.dtype.count - 1)))
  if full_reduce:
    idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(ranges)))
  else:
    idx = tuple(UOp.const(dtypes.index, 0) if i in axis else ranges[i] for i in range(len(ranges)))
  return reduced.vindex(*idx)

def _reindex_reduce_input(ctx: dict, x: UOp) -> UOp|None:
  if x.op is not Ops.REDUCE: return None
  reduce_ranges = x.src[1:]
  ranges = ctx.get("ranges")
  if ranges is None or len(reduce_ranges) != len(ranges): return None
  reduce_map = {loop_r.arg[0]: red_r for loop_r, red_r in zip(ranges, reduce_ranges)}
  def replace_range(ctx: dict, uop: UOp) -> UOp|None:
    if uop.op is not Ops.RANGE or uop.arg[-1] != AxisType.LOOP: return None
    repl = ctx["reduce_map"].get(uop.arg[0])
    if repl is None: return None
    return repl
  src0 = graph_rewrite(
    x.src[0],
    PatternMatcher([(UPat(Ops.RANGE, name="uop"), replace_range)], compiled=False),
    ctx={"reduce_map": reduce_map},
    name="reduce_input_reindex",
  )
  if src0 is x.src[0]: return None
  return x.replace(src=(src0,)+reduce_ranges)


def _const_to_vindex(ctx: dict, x: UOp) -> UOp|None:
    base = x.base
    if base.op not in (Ops.CONST, Ops.VCONST): return None
    dev = ctx.get("device")
    if dev is not None and (len(base.src) == 0 or base.src[0].op is not Ops.DEVICE):
        base = UOp.const(base.dtype, base.arg, device=dev)
    shape = x.shape
    ranges = ctx["ranges"]
    if any(r.op is not Ops.RANGE for r in ranges): return None
    if shape is None or len(shape) != len(ranges): return None
    idx = tuple(UOp.const(dtypes.index, 0) if resolve(s == 1) else ranges[i] for i, s in enumerate(shape))
    return base.vindex(*idx, dtype=x.dtype)


def _broadcast_value_index(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    src = x.src[0]
    if src.op is not Ops.INDEX or src.arg != "value": return None
    if src.src[0].op in (Ops.REDUCE, Ops.REDUCE_AXIS): return None
    try:
        shape = x.shape
    except Exception:
        try:
            shape = x.marg
        except Exception:
            return None
    ranges = ctx["ranges"]
    if any(r.op is not Ops.RANGE for r in ranges): return None
    if shape is None or len(shape) != len(ranges): return None
    idx = tuple(UOp.const(dtypes.index, 0) if resolve(s == 1) else ranges[i] for i, s in enumerate(shape))
    base = src.src[0]
    dev = ctx.get("device")
    if dev is not None and base.op in (Ops.CONST, Ops.VCONST) and (len(base.src) == 0 or base.src[0].op is not Ops.DEVICE):
        base = UOp.const(base.dtype, base.arg, device=dev)
    return base.vindex(*idx, dtype=x.dtype)


def _drop_scalar_value_index(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    src = x.src[0]
    if src.op is not Ops.INDEX or src.arg != "value": return None
    shape_src = x.src[1:]
    if len(shape_src) == 0: return src
    def is_one(u: UOp) -> bool:
        if u.op is Ops.CONST and u.arg == 1: return True
        if u.op is Ops.VCONST and all(v == 1 for v in u.arg): return True
        if u.op is Ops.VECTORIZE and u.dtype == dtypes.index.vec(0): return True
        return False
    if all(is_one(s) for s in shape_src): return src
    return None

def _broadcast_scalar_index(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    src = x.src[0]
    if src.op is not Ops.INDEX or src.arg is not None: return None
    if src.src[0].op in (Ops.REDUCE, Ops.REDUCE_AXIS): return None
    if src.src[0].op is Ops.DEFINE_REG: return None
    if src.src[0].op is Ops.AFTER and src.src[0].src[0].op is Ops.DEFINE_REG: return None
    if not all(s.op is Ops.CONST and s.arg == 0 for s in src.src[1:]): return None
    try:
        shape = x.shape
    except Exception:
        try:
            shape = x.marg
        except Exception:
            return None
    ranges = ctx["ranges"]
    if any(r.op is not Ops.RANGE for r in ranges): return None
    if shape is None or len(shape) != len(ranges): return None
    idx = tuple(UOp.const(dtypes.index, 0) if resolve(s == 1) else ranges[i] for i, s in enumerate(shape))
    return src.src[0].vindex(*idx, dtype=x.dtype)


def _broadcast_scalar_base(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    base = x.base
    if base.op is not Ops.INDEX: return None
    if base.src[0].op in (Ops.REDUCE, Ops.REDUCE_AXIS): return None
    if base.src[0].op is Ops.DEFINE_REG: return None
    if base.src[0].op is Ops.AFTER and base.src[0].src[0].op is Ops.DEFINE_REG: return None
    if base.arg == "value":
        try:
            if base.size != 1: return None
        except Exception:
            return None
    elif base.arg is None:
        if not all(s.op is Ops.CONST and s.arg == 0 for s in base.src[1:]): return None
    else:
        return None
    try:
        shape = x.shape
    except Exception:
        try:
            shape = x.marg
        except Exception:
            return None
    ranges = ctx["ranges"]
    if any(r.op is not Ops.RANGE for r in ranges): return None
    if shape is None or len(shape) != len(ranges): return None
    idx = tuple(UOp.const(dtypes.index, 0) if resolve(s == 1) else ranges[i] for i, s in enumerate(shape))
    return base.src[0].vindex(*idx, dtype=x.dtype)

def _broadcast_scalar_alu(ctx: dict, uop: UOp) -> UOp|None:
    if uop.op not in GroupOp.Binary: return None
    try:
        shapes = [s.shape for s in uop.src]
    except Exception:
        return None
    if any(s is None for s in shapes): return None
    def shape_prod(s: tuple[int, ...]) -> int:
        return math.prod(s) if len(s) else 1
    target = max((s for s in shapes if s is not None), key=shape_prod, default=None)
    if target is None or target == (): return None
    new_src = []
    changed = False
    for s, shape in zip(uop.src, shapes):
        is_scalar = shape == () or (len(shape) == len(target) and all(x == 1 for x in shape))
        if shape == target: 
            new_src.append(s)
            continue
        if is_scalar and target != ():
            try:
                s = s.forced_reshape(tuple(1 for _ in range(len(target))))
                s = s._mop(Ops.EXPAND, target, same_shape_noop=True)
                changed = True
            except Exception:
                return None
        new_src.append(s)
    if not changed: return None
    return UOp(uop.op, uop.dtype, tuple(new_src), uop.arg, uop.tag)


def _drop_reg_expand(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    src = x.src[0]
    if src.op is not Ops.INDEX or src.arg is not None: return None
    base = src.src[0]
    if base.op is Ops.DEFINE_REG: return src
    if base.op is Ops.AFTER and base.src[0].op is Ops.DEFINE_REG: return src
    return None

def _drop_reg_expand_deep(ctx: dict, x: UOp) -> UOp|None:
    if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
    base = x.base
    if base.op is not Ops.INDEX: return None
    src0 = base.src[0]
    if src0.op is Ops.DEFINE_REG: return base
    if src0.op is Ops.AFTER and src0.src[0].op is Ops.DEFINE_REG: return base
    return None

def _drop_const_reshape(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  base = x.base
  if base.op not in (Ops.CONST, Ops.VCONST): return None
  try:
    shape = x.shape
  except Exception:
    return None
  if shape is None: return None
  if not all(resolve(s == 1) for s in shape): return None
  return base

def _drop_const_expand_any(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  base = x.base
  if base.op not in (Ops.CONST, Ops.VCONST): return None
  return base

def _hreduce_vector(uop: UOp, op: Ops) -> UOp:
  if uop.dtype.count <= 1:
    return uop
  scalar = uop.gep(0)
  for i in range(1, uop.dtype.count):
    scalar = scalar.alu(op, uop.gep(i))
  return scalar

def _drop_empty_reshape(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  try:
    shape = x.shape
  except Exception:
    return None
  if shape is None: return None
  if len(shape) == 0: return x.src[0]
  if shape == (1,): return x.src[0]
  return None

def _drop_zero_vec_shape(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  if any(s.op is Ops.VECTORIZE and s.dtype.count == 0 for s in x.src[1:]):
    return x.src[0]
  return None

def _const_to_vec(ctx: dict, x: UOp) -> UOp|None:
  if x.dtype.count != 1 or x.dtype.scalar() not in dtypes.floats: return None
  return UOp.const(x.dtype.vec(ctx["vector_width"]), x.arg)

def _strip_device_const(ctx: dict, c: UOp) -> UOp|None:
  if c.op not in (Ops.CONST, Ops.VCONST): return None
  if len(c.src) == 0 or c.src[0].op is not Ops.DEVICE: return None
  return UOp.const(c.dtype, c.arg)

def _add_device_const(ctx: dict, c: UOp) -> UOp|None:
  if c.op not in (Ops.CONST, Ops.VCONST): return None
  if len(c.src) != 0 and c.src[0].op is Ops.DEVICE: return None
  dev = ctx.get("device")
  if dev is None: return None
  return UOp.const(c.dtype, c.arg, device=dev)

def _drop_vector_expand(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  if x.src[0].dtype.count != ctx["vector_width"]: return None
  return x.src[0]

def _cse_uops(sink: UOp) -> UOp:
  cse: dict[tuple, UOp] = {}
  cache: dict[UOp, UOp] = {}
  def visit(u: UOp) -> UOp:
    src = tuple(cache[s] for s in u.src)
    if src != u.src: u = u.replace(src=src)
    if u.op not in GroupOp.ALU and u.op not in {
      Ops.CONST, Ops.VCONST, Ops.CAST, Ops.BITCAST, Ops.GEP, Ops.WHERE, Ops.VECTORIZE, Ops.REDUCE,
    }:
      return u
    key = (u.op, u.arg, u.dtype, src)
    ret = cse.get(key)
    if ret is not None: return ret
    cse[key] = u
    return u
  return sink.topovisit(visit, cache)


def _grad_H_compute(q: Tensor, p: Tensor, H_func) -> tuple[Tensor, Tensor]:
    q_grad = q.detach().requires_grad_(True)
    p_grad = p.detach().requires_grad_(True)

    H = H_func(q_grad, p_grad)
    H.backward()

    dHdq = q_grad.grad.detach() if q_grad.grad is not None else q * 0
    dHdp = p_grad.grad.detach() if p_grad.grad is not None else p * 0
    return dHdq, dHdp


def _grad_H(q: Tensor, p: Tensor, H_func) -> tuple[Tensor, Tensor]:
    """
    Compute gradients of Hamiltonian using tinygrad autograd.

    This is the ONLY place where forces are computed - derived from energy.

    The Poisson structure:
        dq/dt = +dH/dp  (velocity)
        dp/dt = -dH/dq  (force = negative gradient of energy)
    """
    if not getenv("TINYGRAD_GRADH_JIT", 1):
        return _grad_H_compute(q, p, H_func)
    if capturing:
        return _grad_H_compute(q, p, H_func)

    q_use = q.contiguous().realize()
    p_use = p.contiguous().realize()
    key = _grad_H_key(q_use, p_use, H_func)
    jit = _grad_H_jit_cache.get(key)
    if jit is None:
        def grad_fn(q_in: Tensor, p_in: Tensor) -> tuple[Tensor, Tensor]:
            return _grad_H_compute(q_in, p_in, H_func)
        jit = _grad_H_jit_cache.setdefault(key, TinyJit(grad_fn))

    try:
        return jit(q_use, p_use)
    except JitError:
        _grad_H_jit_cache.pop(key, None)
        return _grad_H_compute(q_use, p_use, H_func)


# ============================================================================
# SYMPLECTIC INTEGRATORS
# ============================================================================

def _grad_LP_compute(z: Tensor, H_func) -> Tensor:
    z_grad = z.detach().requires_grad_(True)
    H = H_func(z_grad)
    H.backward()
    return z_grad.grad.detach() if z_grad.grad is not None else z * 0


def _grad_LP(z: Tensor, H_func) -> Tensor:
    if not getenv("TINYGRAD_GRADH_JIT", 1):
        return _grad_LP_compute(z, H_func)
    if capturing:
        return _grad_LP_compute(z, H_func)
    z_use = z.contiguous().realize()
    key = (H_func, z_use.device, z_use.shape, z_use.dtype)
    jit = _grad_LP_jit_cache.get(key)
    if jit is None:
        def grad_fn(z_in: Tensor) -> Tensor:
            return _grad_LP_compute(z_in, H_func)
        jit = _grad_LP_jit_cache.setdefault(key, TinyJit(grad_fn))
    try:
        return jit(z_use)
    except JitError:
        _grad_LP_jit_cache.pop(key, None)
        return _grad_LP_compute(z_use, H_func)

def symplectic_euler(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """1st-order Symplectic Euler (Kick-Drift). Order: O(dt)"""
    dHdq, _ = _grad_H(q, p, H_func)
    p_new = p - dt * dHdq
    _, dHdp = _grad_H(q, p_new, H_func)
    q_new = q + dt * dHdp
    return q_new.realize(), p_new.realize()


def leapfrog(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """2nd-order Leapfrog / Störmer-Verlet. Order: O(dt²), Symplectic, Time-reversible"""
    dHdq_1, _ = _grad_H(q, p, H_func)
    p_half = p - (0.5 * dt) * dHdq_1
    _, dHdp = _grad_H(q, p_half, H_func)
    q_new = q + dt * dHdp
    dHdq_2, _ = _grad_H(q_new, p_half, H_func)
    p_new = p_half - (0.5 * dt) * dHdq_2
    return q_new.realize(), p_new.realize()


# Yoshida 4th-order coefficients
_W0 = -2**(1/3) / (2 - 2**(1/3))
_W1 = 1 / (2 - 2**(1/3))


def yoshida4(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """4th-order Yoshida symplectic integrator. Order: O(dt⁴)"""
    def substep(q, p, h):
        dHdq_1, _ = _grad_H(q, p, H_func)
        p_half = p - (0.5 * h) * dHdq_1
        _, dHdp = _grad_H(q, p_half, H_func)
        q_new = q + h * dHdp
        dHdq_2, _ = _grad_H(q_new, p_half, H_func)
        p_new = p_half - (0.5 * h) * dHdq_2
        return q_new, p_new

    q, p = substep(q, p, _W1 * dt)
    q, p = substep(q, p, _W0 * dt)
    q, p = substep(q, p, _W1 * dt)
    return q.realize(), p.realize()


def implicit_midpoint(q: Tensor, p: Tensor, H_func, dt: float = 0.01,
                      tol: float = 1e-10, max_iter: int = 10) -> tuple[Tensor, Tensor]:
    """Implicit Midpoint Rule - truly symplectic for ANY Hamiltonian. Order: O(dt²)"""
    dHdq, dHdp = _grad_H(q, p, H_func)
    half_dt = 0.5 * dt
    q_mid = q + half_dt * dHdp
    p_mid = p - half_dt * dHdq

    fixed_iters = _implicit_fixed_iters(dt, max_iter)
    if fixed_iters > 0:
        for _ in range(fixed_iters):
            dHdq_mid, dHdp_mid = _grad_H(q_mid, p_mid, H_func)
            q_mid = q + half_dt * dHdp_mid
            p_mid = p - half_dt * dHdq_mid
        q_next = q + dt * dHdp_mid
        p_next = p - dt * dHdq_mid
    else:
        q_next = q + dt * dHdp
        p_next = p - dt * dHdq
        for _ in range(max_iter):
            dHdq_mid, dHdp_mid = _grad_H(q_mid, p_mid, H_func)
            q_new = q + dt * dHdp_mid
            p_new = p - dt * dHdq_mid
            diff_q = (q_new - q_next).abs().max().numpy()
            diff_p = (p_new - p_next).abs().max().numpy()
            if diff_q < tol and diff_p < tol:
                break
            q_next, p_next = q_new, p_new
            q_mid = q + half_dt * dHdp_mid
            p_mid = p - half_dt * dHdq_mid

    return q_next.realize(), p_next.realize()


def _implicit_fixed_iters(dt: float, max_iter: int) -> int:
    key = (round(dt, 12), max_iter, int(getenv("TINYGRAD_IMPLICIT_HEURISTIC", 1)))
    cached = _implicit_iters_cache.get(key)
    if cached is not None:
        return cached
    fixed_iters = getenv("TINYGRAD_IMPLICIT_ITERS", 0)
    if fixed_iters <= 0:
        if CAPTURING and len(capturing):
            fixed_iters = max_iter
        elif getenv("TINYGRAD_IMPLICIT_HEURISTIC", 1):
            if dt <= 0.0025:
                fixed_iters = 2
            elif dt <= 0.005:
                fixed_iters = 3
            elif dt <= 0.01:
                fixed_iters = 3
            elif dt <= 0.02:
                fixed_iters = 6
            else:
                fixed_iters = 8
    _implicit_iters_cache[key] = fixed_iters
    return fixed_iters


def implicit_midpoint_fixed(q: Tensor, p: Tensor, H_func, dt: float, iters: int) -> tuple[Tensor, Tensor]:
    dHdq, dHdp = _grad_H(q, p, H_func)
    half_dt = 0.5 * dt
    q_mid = q + half_dt * dHdp
    p_mid = p - half_dt * dHdq
    for _ in range(iters):
        dHdq_mid, dHdp_mid = _grad_H(q_mid, p_mid, H_func)
        q_mid = q + half_dt * dHdp_mid
        p_mid = p - half_dt * dHdq_mid
    q_next = q + dt * dHdp_mid
    p_next = p - dt * dHdq_mid
    return q_next, p_next


def implicit_midpoint_inplace(q: Tensor, p: Tensor, H_func, dt: float = 0.01,
                              tol: float = 1e-10, max_iter: int = 10) -> tuple[Tensor, Tensor]:
    fixed_iters = _implicit_fixed_iters(dt, max_iter)
    if fixed_iters <= 0:
        q_next, p_next = implicit_midpoint(q, p, H_func, dt=dt, tol=tol, max_iter=max_iter)
        q.assign(q_next)
        p.assign(p_next)
        return q, p

    buf_key = (q.device, q.shape, q.dtype)
    buf = _implicit_mid_buf_cache.get(buf_key, (None, None, None, None))
    q_mid_buf, p_mid_buf, q_mid_tmp, p_mid_tmp = buf
    if q_mid_buf is None or p_mid_buf is None or q_mid_buf.shape != q.shape or p_mid_buf.shape != p.shape:
        q_mid_buf = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_mid_buf = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        q_mid_tmp = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_mid_tmp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        _implicit_mid_buf_cache[buf_key] = (q_mid_buf, p_mid_buf, q_mid_tmp, p_mid_tmp)

    dHdq, dHdp = _grad_H(q, p, H_func)
    half_dt = 0.5 * dt
    q_mid_buf.assign(q + half_dt * dHdp)
    p_mid_buf.assign(p - half_dt * dHdq)
    for _ in range(fixed_iters):
        dHdq_mid, dHdp_mid = _grad_H(q_mid_buf, p_mid_buf, H_func)
        q_mid_buf.assign(q + half_dt * dHdp_mid)
        p_mid_buf.assign(p - half_dt * dHdq_mid)
    q.assign(q + dt * dHdp_mid)
    p.assign(p - dt * dHdq_mid)
    return q, p


def implicit_midpoint_into(q: Tensor, p: Tensor, q_out: Tensor, p_out: Tensor, H_func, dt: float = 0.01,
                           tol: float = 1e-10, max_iter: int = 10) -> tuple[Tensor, Tensor]:
    fixed_iters = _implicit_fixed_iters(dt, max_iter)
    if fixed_iters <= 0:
        q_next, p_next = implicit_midpoint(q, p, H_func, dt=dt, tol=tol, max_iter=max_iter)
        q_out = _assign_or_replace(q_out, q_next)
        p_out = _assign_or_replace(p_out, p_next)
        return q_out, p_out

    buf_key = (q.device, q.shape, q.dtype)
    buf = _implicit_mid_buf_cache.get(buf_key, (None, None, None, None))
    q_mid_buf, p_mid_buf, q_mid_tmp, p_mid_tmp = buf
    if q_mid_buf is None or p_mid_buf is None or q_mid_buf.shape != q.shape or p_mid_buf.shape != p.shape:
        q_mid_buf = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_mid_buf = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        q_mid_tmp = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_mid_tmp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        _implicit_mid_buf_cache[buf_key] = (q_mid_buf, p_mid_buf, q_mid_tmp, p_mid_tmp)

    dHdq, dHdp = _grad_H(q, p, H_func)
    half_dt = 0.5 * dt
    q_mid_buf.assign(q + half_dt * dHdp)
    p_mid_buf.assign(p - half_dt * dHdq)
    for _ in range(fixed_iters):
        dHdq_mid, dHdp_mid = _grad_H(q_mid_buf, p_mid_buf, H_func)
        q_mid_buf.assign(q + half_dt * dHdp_mid)
        p_mid_buf.assign(p - half_dt * dHdq_mid)
    q_out = _assign_or_replace(q_out, q + dt * dHdp_mid)
    p_out = _assign_or_replace(p_out, p - dt * dHdq_mid)
    return q_out, p_out


def _assign_or_replace(dst: Tensor, src: Tensor) -> Tensor:
    if getenv("TINYGRAD_IMPLICIT_INTO_REPLACE", 0):
        return dst.replace(src)
    return dst.assign(src)


# ============================================================================
# HAMILTONIAN SYSTEM - The "Compiler" Interface
# ============================================================================

class HamiltonianSystem:
    """
    A physics system defined purely by its Hamiltonian.

    Example:
        def H(q, p): return 0.5 * (p**2).sum() + 0.5 * (q**2).sum()
        system = HamiltonianSystem(H)
        q, p = system.step(q, p, dt=0.01)
    """
    INTEGRATORS = {
        "euler": symplectic_euler,
        "leapfrog": leapfrog,
        "yoshida4": yoshida4,
        "implicit": implicit_midpoint,
    }

    def __init__(self, H_func, integrator: str = "auto", policy: SymplecticPolicy | None = None):
        self.H = H_func
        if integrator != "auto" and integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self.integrator_name = integrator
        self._step = self.INTEGRATORS[integrator] if integrator != "auto" else None
        if policy is None:
            from tinygrad.physics_profile import get_default_profile
            policy = get_default_profile().policy
        self.policy = policy
        self._jit_step = TinyJit(self.step)
        self._jit_step_inplace = TinyJit(self.step_inplace)
        self._scan_kernel_cache: dict[tuple[float, int, int, int, bool, str, tuple[int, ...], object], Callable] = {}
        self._scan_kernel_coupled_cache: dict[tuple[float, int, int, int, str, tuple[int, ...], object], Callable] = {}
        self._scan_kernel_tune_cache: dict[tuple[float, int, str, tuple[int, ...], object], tuple[int, int]] = {}
        self._ho_coeff_cache: dict[tuple[object, str, object], tuple[float, float, float, float]|None] = {}
        self._ho_mode_cache: dict[tuple[float, int, str, tuple[int, ...], object], bool] = {}
        self._ho_vec_cache: dict[tuple[float, int, str, tuple[int, ...], object], int] = {}
        self._scan_vec_cache: dict[tuple[float, int, int, str, tuple[int, ...], object], int] = {}
        self._ho_trig_cache: dict[tuple[float, int, str, object, float, float], tuple[float, float, float, float]] = {}
        self._ho_shift_cache: dict[tuple[str, object, float, float, float, float], tuple[float, float]] = {}
        self._scan_vec_shape_cache: dict[tuple[str, object, int], int] = {}
        self._step_kernel_coupled_cache: dict[tuple[float, str, tuple[int, ...], object], Callable] = {}
        self._grad_kernel_coupled_cache: dict[tuple[str, tuple[int, ...], object], Callable] = {}
        self._update_kernel_coupled_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._update_kernel_coupled_p_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._update_kernel_coupled_fused_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._update_kernel_coupled_p_fused_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._update_kernel_coupled_fused_ok: dict[tuple[float, str, tuple[int, ...], object, int], bool] = {}
        self._scan_kernel_coupled_tune_cache: dict[tuple[float, int, str, tuple[int, ...], object], int] = {}
        self._fused_kernel_coupled_tune_cache: dict[tuple[float, int, str, tuple[int, ...], object], tuple[int, int]] = {}
        self._fused_reduce_vec_tune_cache: dict[tuple[float, str, tuple[int, ...], object], int] = {}
        self._reduce_kernel_coupled_cache: dict[tuple[UOp, str, tuple[int, ...], object], Callable] = {}
        self._reduce_kernel_coupled_multi_cache: dict[tuple[tuple[UOp, ...], str, tuple[int, ...], object, int], Callable] = {}
        self._reduce_kernel_coupled_qnew_cache: dict[tuple[tuple[UOp, ...], object, str, tuple[int, ...], object, float, int], Callable] = {}
        self._reduce_kernel_coupled_tune_cache: dict[tuple[str, tuple[int, ...], object], int] = {}
        self._reduce_kernel_coupled_unroll_tune_cache: dict[tuple[str, tuple[int, ...], object, int], int] = {}
        self._reduce_kernel_coupled_tune_both_cache: dict[tuple[str, tuple[int, ...], object], tuple[int, int]] = {}
        self._reduce_acc_buf_cache_dHdq: dict[tuple[str, object, int], list[Tensor]] = {}
        self._reduce_acc_buf_cache_dHdp: dict[tuple[str, object, int], list[Tensor]] = {}
        self._scan_tmp_buf_cache: dict[tuple[str, object, tuple[int, ...]], tuple[Tensor, Tensor]] = {}
        self._elem_kernel_coupled_cache: dict[tuple[object, str, tuple[int, ...], object, str, float], Callable] = {}
        self._grad_uop_cache: dict[tuple[object, str, tuple[int, ...], object], tuple[UOp, UOp, UOp, UOp]] = {}
        self._grad_kernel_coupled_vec_cache: dict[tuple[str, tuple[int, ...], object, int], Callable] = {}
        self._grad_kernel_coupled_vec_ok: dict[tuple[str, tuple[int, ...], object, int], bool] = {}
        self._two_phase_vec_perf_cache: dict[tuple[str, tuple[int, ...], object, int, float], bool] = {}
        self._coupled_sin_cache: dict[tuple[str, tuple[int, ...], object], bool] = {}
        self._implicit_mid_init_cache: dict[tuple[float, str, tuple[int, ...], object], Callable] = {}
        self._implicit_mid_iter_cache: dict[tuple[float, str, tuple[int, ...], object], Callable] = {}
        self._implicit_mid_final_cache: dict[tuple[float, str, tuple[int, ...], object], Callable] = {}
        self._implicit_mid_full_cache: dict[tuple[float, int, int, str, tuple[int, ...], object], Callable] = {}
        self._implicit_mid_full_ok: dict[tuple[float, int, int, str, tuple[int, ...], object], bool] = {}
        self._structure_cache: dict[tuple[str, tuple[int, ...], object], StructureInfo] = {}
        self._integrator_auto_cache: dict[tuple[str, tuple[int, ...], object, str], str] = {}

    def step(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        name = self._select_integrator_name(q, p)
        return self.INTEGRATORS[name](q, p, self.H, dt)

    def step_inplace(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        """In-place step for scan loops; updates q and p buffers via assign."""
        if self._select_integrator_name(q, p) == "implicit":
            return self._implicit_midpoint_inplace_fast(q, p, dt)
        q_new, p_new = self.step(q, p, dt)
        q.assign(q_new)
        p.assign(p_new)
        return q, p

    def energy(self, q: Tensor, p: Tensor) -> float:
        return float(self.H(q, p).numpy())

    def analyze_structure(self, q: Tensor, p: Tensor) -> StructureInfo:
        key = (q.device, q.shape, q.dtype)
        cached = self._structure_cache.get(key)
        if cached is not None:
            return cached
        try:
            q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(q.device, q.shape, q.dtype)
            dHdq_dep_p = _uop_depends_on(dHdq_uop, p_sym_uop)
            dHdp_dep_q = _uop_depends_on(dHdp_uop, q_sym_uop)
            separable = not (dHdq_dep_p or dHdp_dep_q)
            coupled_reduce = any(u.op is Ops.REDUCE_AXIS for u in dHdq_uop.toposort()) or any(u.op is Ops.REDUCE_AXIS for u in dHdp_uop.toposort())
            uses_sin = any(u.op is Ops.SIN for u in dHdq_uop.toposort()) or any(u.op is Ops.SIN for u in dHdp_uop.toposort())
            q_sym = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype, requires_grad=True)
            p_sym = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype, requires_grad=True)
            H_uop = self.H(q_sym, p_sym).uop
            constraints = []
            if not separable:
                constraints.append("coupled")
            if _detect_unit_norm(H_uop, q_sym.uop):
                constraints.append("unit_norm_q")
            if _detect_unit_norm(H_uop, p_sym.uop):
                constraints.append("unit_norm_p")
            info = StructureInfo(
                algebra="canonical",
                separable=separable,
                coupled_reduce=coupled_reduce,
                uses_sin=uses_sin,
                constraints=tuple(constraints),
            )
        except Exception:
            info = StructureInfo(algebra="canonical", separable=False, coupled_reduce=False, uses_sin=False, constraints=())
        self._structure_cache[key] = info
        return info

    def _select_integrator_name(self, q: Tensor, p: Tensor) -> str:
        if self.integrator_name != "auto":
            return self.integrator_name
        key = (q.device, q.shape, q.dtype, self.policy.accuracy)
        cached = self._integrator_auto_cache.get(key)
        if cached is not None:
            return cached
        info = self.analyze_structure(q, p)
        strict_drift = self.policy.drift_target is not None and self.policy.drift_target <= 1e-6
        if info.separable:
            if self.policy.accuracy in ("precise", "ultra_precise") or strict_drift:
                name = "yoshida4"
            else:
                name = "leapfrog"
        else:
            if self.policy.accuracy == "fast":
                name = "leapfrog"
            else:
                name = "implicit"
        self._integrator_auto_cache[key] = name
        return name

    def _implicit_midpoint_inplace_fast(self, q: Tensor, p: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        fixed_iters = _implicit_fixed_iters(dt, 10)
        if fixed_iters <= 0 or self._coupled_has_reduce(q.device, q.shape, q.dtype):
            return implicit_midpoint_inplace(q, p, self.H, dt)
        vec = getenv("TINYGRAD_IMPLICIT_VEC", 0)
        if vec <= 0:
            vec = 4 if q.shape and q.shape[-1] % 4 == 0 else 1
        if vec > 1 and q.shape and q.shape[-1] % vec != 0:
            vec = 1
        key = (dt, fixed_iters, vec, q.device, q.shape, q.dtype)
        kernel = self._implicit_mid_full_cache.get(key)
        if kernel is None:
            kernel = self._build_implicit_mid_full_kernel(dt, fixed_iters, q.device, q.shape, q.dtype, vec)
            self._implicit_mid_full_cache[key] = kernel
        ok = self._implicit_mid_full_ok.get(key)
        if ok is None:
            try:
                q_probe = q.detach().clone().realize()
                p_probe = p.detach().clone().realize()
                out = Tensor.custom_kernel(q_probe, p_probe, q_probe, p_probe, fxn=kernel)
                Tensor.realize(out[2], out[3])
                ok = True
            except Exception:
                ok = False
            self._implicit_mid_full_ok[key] = ok
        if not ok:
            return self._implicit_midpoint_inplace_kernels(q, p, dt, fixed_iters)
        out = Tensor.custom_kernel(q, p, q, p, fxn=kernel)
        Tensor.realize(out[2], out[3])
        return q, p

    def _implicit_midpoint_into_fast(self, q: Tensor, p: Tensor, q_out: Tensor, p_out: Tensor, dt: float) -> tuple[Tensor, Tensor]:
        fixed_iters = _implicit_fixed_iters(dt, 10)
        if fixed_iters <= 0 or self._coupled_has_reduce(q.device, q.shape, q.dtype):
            return implicit_midpoint_into(q, p, q_out, p_out, self.H, dt)
        vec = getenv("TINYGRAD_IMPLICIT_VEC", 0)
        if vec <= 0:
            vec = 4 if q.shape and q.shape[-1] % 4 == 0 else 1
        if vec > 1 and q.shape and q.shape[-1] % vec != 0:
            vec = 1
        key = (dt, fixed_iters, vec, q.device, q.shape, q.dtype)
        kernel = self._implicit_mid_full_cache.get(key)
        if kernel is None:
            kernel = self._build_implicit_mid_full_kernel(dt, fixed_iters, q.device, q.shape, q.dtype, vec)
            self._implicit_mid_full_cache[key] = kernel
        ok = self._implicit_mid_full_ok.get(key)
        if ok is None:
            try:
                q_probe = q.detach().clone().realize()
                p_probe = p.detach().clone().realize()
                q_tmp = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
                p_tmp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                out = Tensor.custom_kernel(q_probe, p_probe, q_tmp, p_tmp, fxn=kernel)
                Tensor.realize(out[2], out[3])
                ok = True
            except Exception:
                ok = False
            self._implicit_mid_full_ok[key] = ok
        if not ok:
            return self._implicit_midpoint_into_kernels(q, p, q_out, p_out, dt, fixed_iters)
        out = Tensor.custom_kernel(q, p, q_out, p_out, fxn=kernel)
        Tensor.realize(out[2], out[3])
        return out[2], out[3]

    def _implicit_midpoint_inplace_kernels(self, q: Tensor, p: Tensor, dt: float, iters: int) -> tuple[Tensor, Tensor]:
        key = (dt, q.device, q.shape, q.dtype)
        init_kernel = self._implicit_mid_init_cache.get(key)
        if init_kernel is None:
            init_kernel = self._build_implicit_mid_init_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_init_cache[key] = init_kernel
        iter_kernel = self._implicit_mid_iter_cache.get(key)
        if iter_kernel is None:
            iter_kernel = self._build_implicit_mid_iter_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_iter_cache[key] = iter_kernel
        final_kernel = self._implicit_mid_final_cache.get(key)
        if final_kernel is None:
            final_kernel = self._build_implicit_mid_final_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_final_cache[key] = final_kernel

        buf_key = (q.device, q.shape, q.dtype)
        buf = _implicit_mid_buf_cache.get(buf_key, (None, None, None, None))
        q_mid_a, p_mid_a, q_mid_b, p_mid_b = buf
        if q_mid_a is None or p_mid_a is None or q_mid_a.shape != q.shape or p_mid_a.shape != p.shape:
            q_mid_a = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_mid_a = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            q_mid_b = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_mid_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            _implicit_mid_buf_cache[buf_key] = (q_mid_a, p_mid_a, q_mid_b, p_mid_b)

        out = Tensor.custom_kernel(q, p, q_mid_a, p_mid_a, fxn=init_kernel)
        Tensor.realize(out[2], out[3])
        q_mid, p_mid = out[2], out[3]
        for _ in range(max(iters - 1, 0)):
            out = Tensor.custom_kernel(q, p, q_mid, p_mid, q_mid_b, p_mid_b, fxn=iter_kernel)
            Tensor.realize(out[4], out[5])
            q_mid, p_mid = out[4], out[5]
        out = Tensor.custom_kernel(q, p, q_mid, p_mid, q, p, fxn=final_kernel)
        Tensor.realize(out[4], out[5])
        return q, p

    def _implicit_midpoint_into_kernels(self, q: Tensor, p: Tensor, q_out: Tensor, p_out: Tensor,
                                        dt: float, iters: int) -> tuple[Tensor, Tensor]:
        key = (dt, q.device, q.shape, q.dtype)
        init_kernel = self._implicit_mid_init_cache.get(key)
        if init_kernel is None:
            init_kernel = self._build_implicit_mid_init_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_init_cache[key] = init_kernel
        iter_kernel = self._implicit_mid_iter_cache.get(key)
        if iter_kernel is None:
            iter_kernel = self._build_implicit_mid_iter_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_iter_cache[key] = iter_kernel
        final_kernel = self._implicit_mid_final_cache.get(key)
        if final_kernel is None:
            final_kernel = self._build_implicit_mid_final_kernel(dt, q.device, q.shape, q.dtype)
            self._implicit_mid_final_cache[key] = final_kernel

        buf_key = (q.device, q.shape, q.dtype)
        buf = _implicit_mid_buf_cache.get(buf_key, (None, None, None, None))
        q_mid_a, p_mid_a, q_mid_b, p_mid_b = buf
        if q_mid_a is None or p_mid_a is None or q_mid_a.shape != q.shape or p_mid_a.shape != p.shape:
            q_mid_a = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_mid_a = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            q_mid_b = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_mid_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            _implicit_mid_buf_cache[buf_key] = (q_mid_a, p_mid_a, q_mid_b, p_mid_b)

        out = Tensor.custom_kernel(q, p, q_mid_a, p_mid_a, fxn=init_kernel)
        Tensor.realize(out[2], out[3])
        q_mid, p_mid = out[2], out[3]
        for _ in range(max(iters - 1, 0)):
            out = Tensor.custom_kernel(q, p, q_mid, p_mid, q_mid_b, p_mid_b, fxn=iter_kernel)
            Tensor.realize(out[4], out[5])
            q_mid, p_mid = out[4], out[5]
        out = Tensor.custom_kernel(q, p, q_mid, p_mid, q_out, p_out, fxn=final_kernel)
        Tensor.realize(out[4], out[5])
        return out[4], out[5]

    def evolve(self, q: Tensor, p: Tensor, dt: float, steps: int,
               record_every: int = 1, policy: SymplecticPolicy | None = None,
               scan: bool | None = None, unroll: int | None = None) -> tuple[Tensor, Tensor, list]:
        policy = policy or self.policy
        info = self.analyze_structure(q, p)
        use_scan = policy.should_scan(steps, q.shape, q.device) if scan is None else scan
        integrator = self._select_integrator_name(q, p)
        if integrator != "leapfrog" or info.constraints:
            use_scan = False
        op_cost = None
        if getenv("TINYGRAD_PHYSICS_UNROLL_COST", 1):
            try:
                op_cost = _count_uops_limit(self.H(q, p).uop, 512)
            except Exception:
                op_cost = None
        def make_step(scale: float):
            return lambda q_in, p_in: self._jit_step(q_in, p_in, dt * scale)
        def energy_fn(q_in: Tensor, p_in: Tensor):
            return float(self.H(q_in, p_in).numpy())
        dt_scale, _ = policy.adjust_dt(steps, q.shape, q.device, (q, p), make_step, energy_fn, "TINYGRAD_PHYSICS_DRIFT_HAMIL")
        if dt_scale != 1.0:
            dt *= dt_scale
        if unroll is None and use_scan:
            unroll = policy.choose_unroll(steps, q.shape, q.device, op_cost=op_cost)
        if use_scan and unroll is not None:
            if record_every % unroll != 0:
                record_every = unroll
            out = self.evolve_unrolled(q, p, dt, steps, unroll, record_every=record_every)
            policy.maybe_print_report(steps, q.shape, q.device)
            return out
        q_history: list[Tensor] = []
        p_history: list[Tensor] = []

        step = self._jit_step if not (q.requires_grad or p.requires_grad) else self.step
        for i in range(steps):
            if i % record_every == 0:
                q_history.append(q.detach())
                p_history.append(p.detach())
            q, p = step(q, p, dt)

        q_history.append(q.detach())
        p_history.append(p.detach())

        history = []
        for q_t, p_t in zip(q_history, p_history):
            q_np = q_t.numpy().copy()
            p_np = p_t.numpy().copy()
            e = float(self.H(q_t, p_t).numpy())
            history.append((q_np, p_np, e))

        policy.maybe_print_report(steps, q.shape, q.device)
        return q, p, history


    def evolve_scan(self, q: Tensor, p: Tensor, dt: float, steps: int, scan: int = 8,
                    record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        """Higher-level scan API: unrolls multiple steps while preserving record cadence."""
        if scan <= 1:
            return self.evolve(q, p, dt, steps, record_every=record_every)
        if q.requires_grad or p.requires_grad:
            return self.evolve(q, p, dt, steps, record_every=record_every)
        scan_integrator = self.integrator_name
        if scan_integrator == "auto":
            scan_integrator = self._select_integrator_name(q, p)
        if scan_integrator == "leapfrog" and scan >= steps and record_every >= steps and q.shape == p.shape:
            return self.evolve_scan_kernel(q, p, dt, steps)

        q_history: list[Tensor] = []
        p_history: list[Tensor] = []
        step_single = self._jit_step
        q_start = q.detach()
        p_start = p.detach()
        step_scanned = self._jit_step_inplace

        if step_scanned.captured is None:
            q_tmp = q.detach()
            p_tmp = p.detach()
            step_scanned(q_tmp, p_tmp, dt)
            step_scanned(q_tmp, p_tmp, dt)

        next_record = 0
        i = 0
        while i < steps:
            if i == next_record:
                q_history.append(q.detach())
                p_history.append(p.detach())
                next_record += record_every

            remaining = steps - i
            to_record = next_record - i if next_record <= steps else remaining
            if remaining < scan or to_record < scan:
                q, p = step_single(q, p, dt)
                i += 1
            else:
                step_scanned.call_repeat(q, p, dt, repeat=scan)
                i += scan

        q_history.append(q.detach())
        p_history.append(p.detach())

        history = []
        for q_t, p_t in zip(q_history, p_history):
            q_np = q_t.numpy().copy()
            p_np = p_t.numpy().copy()
            e = float(self.H(q_t, p_t).numpy())
            history.append((q_np, p_np, e))

        return q, p, history

    def evolve_scan_kernel(self, q: Tensor, p: Tensor, dt: float, steps: int, coupled: bool = False,
                           unroll_steps: int = 1, vector_width: int = 1, inplace: bool = False,
                           coupled_fused: bool = False, double_buffer: bool = False, scan_tune: bool = False,
                           tune_unrolls: tuple[int, ...] = (1, 2, 4, 8)) -> tuple[Tensor, Tensor, list]:
        """Single-kernel scan for leapfrog with static steps (elementwise H)."""
        integrator = self.integrator_name
        if integrator == "auto":
            integrator = self._select_integrator_name(q, p)
        if integrator != "leapfrog":
            raise ValueError("scan kernel only supports leapfrog")
        if q.requires_grad or p.requires_grad:
            raise ValueError("scan kernel does not support gradients")
        if q.shape != p.shape:
            raise ValueError("scan kernel requires q and p with the same shape")
        if q.dtype != p.dtype:
            raise ValueError("scan kernel requires q and p with the same dtype")
        if coupled:
            if coupled_fused:
                return self.evolve_scan_kernel_coupled_fused(
                    q, p, dt, steps, unroll_steps=unroll_steps, vector_width=vector_width,
                    double_buffer=double_buffer, scan_tune=scan_tune, tune_unrolls=tune_unrolls)
            return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and q.numel() % vector_width != 0:
            raise ValueError("vector_width must divide the number of elements")
        if vector_width > 1 and q.numel() % vector_width != 0:
            raise ValueError("vector_width must divide the number of elements")
        if not scan_tune and getenv("TINYGRAD_SCAN_UNROLL_HEURISTIC", 1):
            if steps >= 1024 and q.numel() >= 1024:
                guess = 8
            elif steps >= 256:
                guess = 4
            elif steps >= 64:
                guess = 2
            else:
                guess = 1
            for u in (guess, 4, 2, 1):
                if steps % u == 0:
                    unroll_steps = u
                    break
        q = q.contiguous().realize()
        p = p.contiguous().realize()

        if q.device != p.device:
            raise ValueError("scan kernel requires q and p on the same device")

        q_start = q.detach()
        p_start = p.detach()
        ho_coeffs = self._get_ho_coeffs(q.device, q.dtype)
        if scan_tune:
            key = (dt, steps, q.device, q.shape, q.dtype)
            cached = self._scan_kernel_tune_cache.get(key)
            if cached is not None:
                unroll_steps, vector_width = cached
            else:
                candidates_unroll = [u for u in tune_unrolls if u >= 1 and steps % u == 0]
                if not candidates_unroll:
                    candidates_unroll = [1]
                if ho_coeffs is not None:
                    candidates_unroll = [unroll_steps]
                candidates_vec = [vector_width] if vector_width > 1 else [1, 2, 4, 8]
                candidates_vec = [v for v in candidates_vec if q.numel() % v == 0]
                if not candidates_vec:
                    candidates_vec = [1]
                best_u, best_v = unroll_steps, vector_width
                best_time = float("inf")
                for v in candidates_vec:
                    for u in candidates_unroll:
                        key_u = (dt, steps, u, v, False, q.device, q.shape, q.dtype)
                        try:
                            kernel = self._scan_kernel_cache.get(key_u)
                            if kernel is None:
                                kernel = self._build_leapfrog_scan_kernel(
                                    dt, steps, u, v, q.device, q.shape, q.dtype, ho_closed_form=False)
                                self._scan_kernel_cache[key_u] = kernel
                            q_tmp = q_start.clone().realize()
                            p_tmp = p_start.clone().realize()
                            start = time.perf_counter()
                            q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                            Tensor.realize(q_out, p_out)
                            elapsed = time.perf_counter() - start
                        except Exception:
                            continue
                        if elapsed < best_time:
                            best_time = elapsed
                            best_u, best_v = u, v
                if best_time == float("inf"):
                    best_u, best_v = unroll_steps, 1
                self._scan_kernel_tune_cache[key] = (best_u, best_v)
                unroll_steps, vector_width = best_u, best_v

        if vector_width == 1 and getenv("TINYGRAD_SCAN_VEC_SHAPE_AUTO", 1):
            key = (q.device, q.dtype, q.numel())
            cached = self._scan_vec_shape_cache.get(key)
            if cached is not None:
                vector_width = cached
            else:
                candidates_vec = [1, 2, 4, 8]
                candidates_vec = [v for v in candidates_vec if q.numel() % v == 0]
                if not candidates_vec:
                    candidates_vec = [1]
                best_v = 1
                best_time = float("inf")
                for v in candidates_vec:
                    key_u = (dt, steps, unroll_steps, v, False, q.device, q.shape, q.dtype)
                    try:
                        kernel = self._scan_kernel_cache.get(key_u)
                        if kernel is None:
                            kernel = self._build_leapfrog_scan_kernel(
                                dt, steps, unroll_steps, v, q.device, q.shape, q.dtype, ho_closed_form=False)
                            self._scan_kernel_cache[key_u] = kernel
                        q_tmp = q_start.clone().realize()
                        p_tmp = p_start.clone().realize()
                        start = time.perf_counter()
                        q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                        Tensor.realize(q_out, p_out)
                        elapsed = time.perf_counter() - start
                    except Exception:
                        continue
                    if elapsed < best_time:
                        best_time = elapsed
                        best_v = v
                self._scan_vec_shape_cache[key] = best_v
                vector_width = best_v

        if ho_coeffs is None and getenv("TINYGRAD_SCAN_VEC_SHAPE_ALWAYS", 0):
            key = (q.device, q.dtype, q.numel())
            cached = self._scan_vec_shape_cache.get(key)
            if cached is not None:
                vector_width = cached

        if vector_width == 1 and getenv("TINYGRAD_SCAN_VEC_AUTO", 0):
            key = (dt, steps, unroll_steps, q.device, q.shape, q.dtype)
            cached = self._scan_vec_cache.get(key)
            if cached is not None:
                vector_width = cached
            else:
                candidates_vec = [1, 2, 4, 8]
                candidates_vec = [v for v in candidates_vec if q.numel() % v == 0]
                if not candidates_vec:
                    candidates_vec = [1]
                best_v = 1
                best_time = float("inf")
                for v in candidates_vec:
                    key_u = (dt, steps, unroll_steps, v, False, q.device, q.shape, q.dtype)
                    try:
                        kernel = self._scan_kernel_cache.get(key_u)
                        if kernel is None:
                            kernel = self._build_leapfrog_scan_kernel(
                                dt, steps, unroll_steps, v, q.device, q.shape, q.dtype, ho_closed_form=False)
                            self._scan_kernel_cache[key_u] = kernel
                        q_tmp = q_start.clone().realize()
                        p_tmp = p_start.clone().realize()
                        start = time.perf_counter()
                        q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                        Tensor.realize(q_out, p_out)
                        elapsed = time.perf_counter() - start
                    except Exception:
                        continue
                    if elapsed < best_time:
                        best_time = elapsed
                        best_v = v
                self._scan_vec_cache[key] = best_v
                vector_width = best_v

        if vector_width == 1 and getenv("TINYGRAD_HO_VEC_AUTO", 0):
            if ho_coeffs is not None:
                key = (dt, steps, q.device, q.shape, q.dtype)
                cached = self._ho_vec_cache.get(key)
                if cached is not None:
                    vector_width = cached
                else:
                    candidates_vec = [1, 2, 4, 8]
                    candidates_vec = [v for v in candidates_vec if q.numel() % v == 0]
                    if not candidates_vec:
                        candidates_vec = [1]
                    best_v = 1
                    best_time = float("inf")
                    for v in candidates_vec:
                        key_u = (dt, steps, unroll_steps, v, False, q.device, q.shape, q.dtype)
                        kernel = self._scan_kernel_cache.get(key_u)
                        if kernel is None:
                            kernel = self._build_leapfrog_scan_kernel(
                                dt, steps, unroll_steps, v, q.device, q.shape, q.dtype, ho_closed_form=False)
                            self._scan_kernel_cache[key_u] = kernel
                        q_tmp = q_start.clone().realize()
                        p_tmp = p_start.clone().realize()
                        start = time.perf_counter()
                        q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                        Tensor.realize(q_out, p_out)
                        elapsed = time.perf_counter() - start
                        if elapsed < best_time:
                            best_time = elapsed
                            best_v = v
                    self._ho_vec_cache[key] = best_v
                    vector_width = best_v

        ho_closed_form = False
        if ho_coeffs is not None:
            force = getenv("TINYGRAD_HO_CLOSED_FORM", -1)
            if force in (0, 1):
                ho_closed_form = bool(force)
            else:
                max_steps = getenv("TINYGRAD_HO_TUNE_MAX_STEPS", 512)
                if steps >= max_steps:
                    ho_closed_form = True
                    self._ho_mode_cache[(dt, steps, q.device, q.shape, q.dtype)] = True
                else:
                    key = (dt, steps, q.device, q.shape, q.dtype)
                    cached = self._ho_mode_cache.get(key)
                    if cached is not None:
                        ho_closed_form = cached
                    else:
                        ho_closed_form = True
                        best_cf = True
                        best_time = float("inf")
                        for cf in (True, False):
                            key_u = (dt, steps, unroll_steps, vector_width, cf, q.device, q.shape, q.dtype)
                            kernel = self._scan_kernel_cache.get(key_u)
                            if kernel is None:
                                kernel = self._build_leapfrog_scan_kernel(
                                    dt, steps, unroll_steps, vector_width, q.device, q.shape, q.dtype, ho_closed_form=cf)
                                self._scan_kernel_cache[key_u] = kernel
                            q_tmp = q_start.clone().realize()
                            p_tmp = p_start.clone().realize()
                            start = time.perf_counter()
                            q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                            Tensor.realize(q_out, p_out)
                            elapsed = time.perf_counter() - start
                            if elapsed < best_time:
                                best_time = elapsed
                                best_cf = cf
                        self._ho_mode_cache[key] = best_cf

        key = (dt, steps, unroll_steps, vector_width, ho_closed_form, q.device, q.shape, q.dtype)
        kernel = self._scan_kernel_cache.get(key)
        if kernel is None:
            kernel = self._build_leapfrog_scan_kernel(
                dt, steps, unroll_steps, vector_width, q.device, q.shape, q.dtype, ho_closed_form=ho_closed_form)
            self._scan_kernel_cache[key] = kernel

        q_out, p_out = Tensor.custom_kernel(q, p, fxn=kernel)[:2]
        Tensor.realize(q_out, p_out)
        if inplace:
            q.assign(q_out)
            p.assign(p_out)
            Tensor.realize(q, p)
        else:
            q, p = q_out, p_out

        history = []
        history.append((q_start.numpy().copy(), p_start.numpy().copy(), self.energy(q_start, p_start)))
        history.append((q.numpy().copy(), p.numpy().copy(), self.energy(q, p)))
        return q, p, history

    def evolve_scan_kernel_coupled(self, q: Tensor, p: Tensor, dt: float, steps: int, unroll_steps: int = 1) -> tuple[Tensor, Tensor, list]:
        """Static-step scan for coupled Hamiltonians (multi-kernel, slower)."""
        integrator = self.integrator_name
        if integrator == "auto":
            integrator = self._select_integrator_name(q, p)
        if integrator != "leapfrog":
            raise ValueError("scan kernel only supports leapfrog")
        if q.requires_grad or p.requires_grad:
            raise ValueError("scan kernel does not support gradients")
        if q.shape != p.shape:
            raise ValueError("scan kernel requires q and p with the same shape")
        if q.dtype != p.dtype:
            raise ValueError("scan kernel requires q and p with the same dtype")
        q = q.contiguous().realize()
        p = p.contiguous().realize()
        if q.device != p.device:
            raise ValueError("scan kernel requires q and p on the same device")

        q_start = q.detach()
        p_start = p.detach()
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps")

        step_scanned = self._jit_step_inplace
        if unroll_steps > 1:
            step_scanned = self.compile_unrolled_step_inplace(dt, unroll_steps)
        if step_scanned.captured is None:
            q_tmp = Tensor(q.numpy().copy(), device=q.device)
            p_tmp = Tensor(p.numpy().copy(), device=p.device)
            step_scanned(q_tmp, p_tmp, dt)
            step_scanned(q_tmp, p_tmp, dt)
        step_scanned.call_repeat(q, p, dt, repeat=steps // unroll_steps)
        Tensor.realize(q, p)

        history = []
        history.append((q_start.numpy().copy(), p_start.numpy().copy(), self.energy(q_start, p_start)))
        history.append((q.detach().numpy().copy(), p.detach().numpy().copy(), self.energy(q, p)))
        return q, p, history

    def evolve_scan_kernel_coupled_fused(self, q: Tensor, p: Tensor, dt: float, steps: int,
                                         unroll_steps: int = 1, vector_width: int = 1, double_buffer: bool = False,
                                         scan_tune: bool = False,
                                         tune_unrolls: tuple[int, ...] = (1, 2, 4, 8)) -> tuple[Tensor, Tensor, list]:
        """Single-kernel scan for coupled H."""
        integrator = self.integrator_name
        if integrator == "auto":
            integrator = self._select_integrator_name(q, p)
        if integrator != "leapfrog":
            raise ValueError("scan kernel only supports leapfrog")
        if q.requires_grad or p.requires_grad:
            raise ValueError("scan kernel does not support gradients")
        if q.shape != p.shape:
            raise ValueError("scan kernel requires q and p with the same shape")
        if q.dtype != p.dtype:
            raise ValueError("scan kernel requires q and p with the same dtype")
        q = q.contiguous().realize()
        p = p.contiguous().realize()

        if q.device != p.device:
            raise ValueError("scan kernel requires q and p on the same device")

        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and q.shape and q.shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if double_buffer and scan_tune:
            raise ValueError("scan_tune does not support double_buffer")
        if double_buffer and unroll_steps != 1:
            raise ValueError("double_buffer does not support unroll_steps > 1")
        q_start = q.detach()
        p_start = p.detach()
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(q.device, q.shape, q.dtype)
        dHdq_reduce_nodes = [u for u in dHdq_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        dHdp_reduce_nodes = [u for u in dHdp_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        if not dHdq_reduce_nodes and not dHdp_reduce_nodes:
            uses_sin = any(u.op is Ops.SIN for u in dHdq_uop.toposort()) or any(u.op is Ops.SIN for u in dHdp_uop.toposort())
            if uses_sin:
                return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
        if not dHdq_reduce_nodes and not dHdp_reduce_nodes:
            q_cur, p_cur = q, p
            for _ in range(steps):
                q_cur, p_cur = self.step(q_cur, p_cur, dt)
            return q_cur, p_cur
        if dHdq_reduce_nodes and dHdp_reduce_nodes:
            return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
        if not double_buffer:
            if vector_width > 1 and not getenv("TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL", 0):
                if self._coupled_has_reduce(q.device, q.shape, q.dtype):
                    q_out, p_out = self._evolve_scan_kernel_coupled_split(
                        q, p, dt, steps, unroll_steps=unroll_steps, vector_width=vector_width)
                elif self._should_use_two_phase_vec(q, p, dt, steps, vector_width):
                    q_out, p_out = self._evolve_coupled_two_phase_vec(q, p, dt, steps, vector_width)
                else:
                    if self._coupled_has_sin(q.device, q.shape, q.dtype):
                        return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
                    q_out, p_out = self._evolve_scan_kernel_coupled_split(
                        q, p, dt, steps, unroll_steps=unroll_steps, vector_width=vector_width)
            else:
                try:
                    q_out, p_out = self._evolve_scan_kernel_coupled_split(
                        q, p, dt, steps, unroll_steps=unroll_steps, vector_width=vector_width)
                except Exception as e:
                    if getenv("TINYGRAD_COUPLED_FUSED_FALLBACK", 1):
                        return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
                    raise e
            Tensor.realize(q_out, p_out)
            history = []
            history.append((q_start.numpy().copy(), p_start.numpy().copy(), self.energy(q_start, p_start)))
            history.append((q_out.numpy().copy(), p_out.numpy().copy(), self.energy(q_out, p_out)))
            return q_out, p_out, history
        if scan_tune:
            key = (dt, steps, q.device, q.shape, q.dtype, vector_width)
            unroll_steps = self._scan_kernel_coupled_tune_cache.get(key, 0)
            if unroll_steps == 0:
                best_elapsed = float("inf")
                best_unroll = 1
                for unroll in tune_unrolls:
                    if unroll < 1 or steps % unroll != 0:
                        continue
                    key_u = (dt, steps, unroll, vector_width, q.device, q.shape, q.dtype)
                    kernel = self._scan_kernel_coupled_cache.get(key_u)
                    if kernel is None:
                        kernel = self._build_leapfrog_scan_kernel_coupled(dt, steps, q.device, q.shape, q.dtype,
                                                                          unroll_steps=unroll, vector_width=vector_width)
                        self._scan_kernel_coupled_cache[key_u] = kernel
                    q_tmp = q_start.clone().realize()
                    p_tmp = p_start.clone().realize()
                    start = time.perf_counter()
                    q_out, p_out = Tensor.custom_kernel(q_tmp, p_tmp, fxn=kernel)[:2]
                    Tensor.realize(q_out, p_out)
                    elapsed = time.perf_counter() - start
                    if elapsed < best_elapsed:
                        best_elapsed = elapsed
                        best_unroll = unroll
                self._scan_kernel_coupled_tune_cache[key] = best_unroll
                unroll_steps = best_unroll
        if double_buffer:
            key = (dt, q.device, q.shape, q.dtype)
            kernel = self._step_kernel_coupled_cache.get(key)
            if kernel is None:
                kernel = self._build_leapfrog_step_kernel_coupled(dt, q.device, q.shape, q.dtype)
                self._step_kernel_coupled_cache[key] = kernel
            q_a, p_a = q, p
            q_b = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            for _ in range(steps):
                out = Tensor.custom_kernel(q_a, p_a, q_b, p_b, fxn=kernel)
                q_next, p_next = out[2], out[3]
                Tensor.realize(q_next, p_next)
                q_a, p_a, q_b, p_b = q_next, p_next, q_a, p_a
            q_out, p_out = q_a, p_a
        else:
            if vector_width > 1 and unroll_steps != 1:
                raise ValueError("vector_width with coupled_fused requires unroll_steps=1")
            key = (dt, steps, unroll_steps, vector_width, q.device, q.shape, q.dtype)
            kernel = self._scan_kernel_coupled_cache.get(key)
            if kernel is None:
                try:
                    kernel = self._build_leapfrog_scan_kernel_coupled(dt, steps, q.device, q.shape, q.dtype,
                                                                      unroll_steps=unroll_steps, vector_width=vector_width)
                except ValueError:
                    if self._coupled_has_reduce(q.device, q.shape, q.dtype):
                        if vector_width > 1:
                            if self._should_use_two_phase_vec(q, p, dt, steps, vector_width):
                                q_out, p_out = self._evolve_coupled_two_phase_vec(q, p, dt, steps, vector_width)
                            else:
                                return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
                            Tensor.realize(q_out, p_out)
                            history = []
                            history.append((q_start.numpy().copy(), p_start.numpy().copy(), self.energy(q_start, p_start)))
                            history.append((q_out.numpy().copy(), p_out.numpy().copy(), self.energy(q_out, p_out)))
                            return q_out, p_out, history
                        return self.evolve_scan_kernel_coupled(q, p, dt, steps, unroll_steps=unroll_steps)
                    raise
                self._scan_kernel_coupled_cache[key] = kernel
            q_out, p_out = Tensor.custom_kernel(q, p, fxn=kernel)[:2]
        Tensor.realize(q_out, p_out)

        history = []
        history.append((q_start.numpy().copy(), p_start.numpy().copy(), self.energy(q_start, p_start)))
        history.append((q_out.numpy().copy(), p_out.numpy().copy(), self.energy(q_out, p_out)))
        return q_out, p_out, history

    def _coupled_has_reduce(self, device: str, shape: tuple[int, ...], dtype) -> bool:
        _, _, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        return any(u.op is Ops.REDUCE_AXIS for u in dHdq_uop.toposort()) or any(u.op is Ops.REDUCE_AXIS for u in dHdp_uop.toposort())

    def _coupled_has_sin(self, device: str, shape: tuple[int, ...], dtype) -> bool:
        key = (device, shape, dtype)
        cached = self._coupled_sin_cache.get(key)
        if cached is not None:
            return cached
        _, _, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        has_sin = any(u.op is Ops.SIN for u in dHdq_uop.toposort()) or any(u.op is Ops.SIN for u in dHdp_uop.toposort())
        self._coupled_sin_cache[key] = has_sin
        return has_sin

    def _should_use_two_phase_vec(self, q: Tensor, p: Tensor, dt: float, steps: int, vector_width: int) -> bool:
        if vector_width <= 1:
            return False
        if getenv("TINYGRAD_COUPLED_TWO_PHASE_FORCE", 0):
            return True
        if getenv("TINYGRAD_COUPLED_TWO_PHASE_DISABLE", 0):
            return False
        if not self._coupled_has_sin(q.device, q.shape, q.dtype):
            return True
        key = (q.device, q.shape, q.dtype, vector_width, dt)
        cached = self._two_phase_vec_perf_cache.get(key)
        if cached is not None:
            return cached
        probe_steps = min(steps, 8)
        q_base = q.detach().clone().realize()
        p_base = p.detach().clone().realize()
        start = time.perf_counter()
        q_out, p_out = self.evolve_scan_kernel_coupled(q_base, p_base, dt, probe_steps, unroll_steps=1)[:2]
        Tensor.realize(q_out, p_out)
        base_elapsed = time.perf_counter() - start
        q_vec = q.detach().clone().realize()
        p_vec = p.detach().clone().realize()
        start = time.perf_counter()
        q_out, p_out = self._evolve_coupled_two_phase_vec(q_vec, p_vec, dt, probe_steps, vector_width)
        Tensor.realize(q_out, p_out)
        vec_elapsed = time.perf_counter() - start
        use_vec = vec_elapsed < base_elapsed
        self._two_phase_vec_perf_cache[key] = use_vec
        return use_vec

    def _build_coupled_grad_kernel(self, device: str, shape: tuple[int, ...], dtype) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, dHdq_out: UOp, dHdp_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape)]
            reduce_counter = [len(ranges) + 1]
            def grad_uop(q_uop: UOp, p_uop: UOp) -> tuple[UOp, UOp]:
                sub = {q_sym_uop: q_uop, p_sym_uop: p_uop}
                dHdq = dHdq_uop.substitute(sub)
                dHdp = dHdp_uop.substitute(sub)
                return dHdq, dHdp

            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            dHdq, dHdp = grad_uop(q_val, p_val)
            store_dHdq = dHdq_out.index(*ranges, ptr=True).store(dHdq)
            store_dHdp = dHdp_out.index(*ranges, ptr=True).store(dHdp)
            kernel_sink = UOp.group(store_dHdq, store_dHdp).end(*ranges).sink(
                arg=KernelInfo(name="coupled_grad", opts_to_apply=()))
            lower_reduce_axis = PatternMatcher([
                (UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis),
            ], compiled=False)
            const_to_vindex = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex),
            ], compiled=False)
            drop_scalar_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index),
            ], compiled=False)
            drop_const_reshape = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape),
            ], compiled=False)
            broadcast_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index),
            ], compiled=False)
            broadcast_scalar_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index),
            ], compiled=False)
            broadcast_scalar_base = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base),
            ], compiled=False)
            kernel_sink = graph_rewrite(
                kernel_sink,
                lower_reduce_axis,
                ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                name="lower_reduce_axis",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                const_to_vindex,
                ctx={"ranges": ranges},
                name="const_to_vindex",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_scalar_value_index,
                ctx={},
                name="drop_scalar_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_const_reshape,
                ctx={},
                name="drop_const_reshape",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_value_index,
                ctx={"ranges": ranges},
                name="broadcast_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_index,
                ctx={"ranges": ranges},
                name="broadcast_scalar_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_base,
                ctx={"ranges": ranges},
                name="broadcast_scalar_base",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                ctx={"ranges": ranges},
                name="reindex_reduce_input",
            )
            kernel_sink = graph_rewrite(kernel_sink, strip_device_consts, name="strip_device_consts")
            return _cse_uops(kernel_sink)

        return kernel

    def _build_coupled_grad_kernel_vec(self, device: str, shape: tuple[int, ...], dtype,
                                       vector_width: int) -> Callable:
        if vector_width < 2:
            raise ValueError("vector_width must be >= 2 for grad vec")
        if shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        const_vec = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="x"), _const_to_vec),
        ])
        drop_vector_expand = PatternMatcher([
            (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_vector_expand),
        ])
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, dHdq_out: UOp, dHdp_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
            ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            base = ranges[-1] * UOp.const(dtypes.index, vector_width)
            q_ptr = q.index(*ranges[:-1], base, ptr=True)
            p_ptr = p.index(*ranges[:-1], base, ptr=True)
            vec_dtype = q.dtype.base.vec(vector_width)
            q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
            p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            dHdq = dHdq_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdp = dHdp_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdq = graph_rewrite(dHdq, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
            dHdp = graph_rewrite(dHdp, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
            dHdq = graph_rewrite(dHdq, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
            dHdp = graph_rewrite(dHdp, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
            dHdq = graph_rewrite(dHdq, strip_device_consts, name="strip_device_consts")
            dHdp = graph_rewrite(dHdp, strip_device_consts, name="strip_device_consts")
            if dHdq.dtype.count == 1:
                dHdq = dHdq.broadcast(vector_width)
            if dHdp.dtype.count == 1:
                dHdp = dHdp.broadcast(vector_width)
            dHdq_ptr = dHdq_out.index(*ranges[:-1], base, ptr=True)
            dHdp_ptr = dHdp_out.index(*ranges[:-1], base, ptr=True)
            store_dHdq = dHdq_ptr.cast(vec_dtype.ptr(size=dHdq_ptr.dtype.size, addrspace=dHdq_ptr.dtype.addrspace)).store(dHdq)
            store_dHdp = dHdp_ptr.cast(vec_dtype.ptr(size=dHdp_ptr.dtype.size, addrspace=dHdp_ptr.dtype.addrspace)).store(dHdp)
            return UOp.group(store_dHdq, store_dHdp).end(*ranges).sink(arg=KernelInfo(name="coupled_grad_vec", opts_to_apply=()))

        return kernel

    def _build_coupled_reduce_kernel(self, reduce_uop: UOp, device: str, shape: tuple[int, ...], dtype,
                                     vector_width: int = 1, reduce_unroll: int = 1) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        axis = reduce_uop.arg[1]
        if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
            raise ValueError("split reduce kernel only supports full reductions")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if reduce_unroll < 1:
            raise ValueError("reduce_unroll must be >= 1")

        def kernel(q: UOp, p: UOp, out: UOp) -> UOp:
            use_vec = vector_width > 1
            if use_vec:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
                ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                q_ptr = q.index(*ranges[:-1], base, ptr=True)
                p_ptr = p.index(*ranges[:-1], base, ptr=True)
                vec_dtype = q.dtype.base.vec(vector_width)
                q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            else:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
                q_val = q.vindex(*ranges)
                p_val = p.vindex(*ranges)
            reduce_counter = [len(ranges) + 1]
            red = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                name="lower_reduce_axis",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                ctx={"ranges": ranges},
                name="const_to_vindex",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                ctx={},
                name="drop_scalar_value_index",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape)], compiled=False),
                ctx={},
                name="drop_const_reshape",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_value_index",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_scalar_index",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_scalar_base",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                ctx={"ranges": ranges},
                name="reindex_reduce_input",
            )
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                ctx={},
                name="strip_device_consts",
            )
            if use_vec and red.dtype.count > 1:
                red = _hreduce_vector(red, reduce_uop.arg[0])
            reduce_ranges = [r for r in red.ranges if r.arg[-1] == AxisType.REDUCE]
            store = out.index(UOp.const(dtypes.index, 0), ptr=True).store(red)
            opts = ()
            if reduce_unroll > 1 and reduce_ranges:
                opts = (Opt(OptOps.UNROLL, reduce_ranges[-1].arg[0], reduce_unroll),)
            return store.end(*reduce_ranges).sink(arg=KernelInfo(name="coupled_reduce", opts_to_apply=opts))

        return kernel

    def _build_coupled_reduce_kernel_multi(self, reduce_uops: tuple[UOp, ...], device: str, shape: tuple[int, ...], dtype,
                                           vector_width: int = 1, reduce_unroll: int = 1) -> Callable:
        if len(reduce_uops) == 0:
            raise ValueError("reduce_uops must be non-empty")
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        for reduce_uop in reduce_uops:
            axis = reduce_uop.arg[1]
            if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
                raise ValueError("split reduce kernel only supports full reductions")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if reduce_unroll < 1:
            raise ValueError("reduce_unroll must be >= 1")

        def kernel(q: UOp, p: UOp, *outs: UOp) -> UOp:
            def rewrite_elem(uop: UOp, ranges: list[UOp]) -> UOp:
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape)], compiled=False),
                    ctx={},
                    name="drop_const_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                return uop

            any_nested_reduce = any(u.op is Ops.REDUCE_AXIS for red in reduce_uops for u in red.src[0].toposort())
            if not any_nested_reduce and len(reduce_uops) > 1:
                use_vec = vector_width > 1
                if use_vec:
                    reduce_ranges = [UOp.range(s, i+1, AxisType.REDUCE) for i, s in enumerate(shape[:-1])]
                    reduce_ranges.append(UOp.range(shape[-1] // vector_width, len(shape), AxisType.REDUCE))
                    base = reduce_ranges[-1] * UOp.const(dtypes.index, vector_width)
                    q_ptr = q.index(*reduce_ranges[:-1], base, ptr=True)
                    p_ptr = p.index(*reduce_ranges[:-1], base, ptr=True)
                    vec_dtype = q.dtype.base.vec(vector_width)
                    q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                    p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
                else:
                    reduce_ranges = [UOp.range(s, i+1, AxisType.REDUCE) for i, s in enumerate(shape)]
                    q_val = q.vindex(*reduce_ranges)
                    p_val = p.vindex(*reduce_ranges)
                reduce_ranges_used = reduce_ranges
                opts = ()
                if reduce_unroll > 1 and reduce_ranges_used:
                    opts = (Opt(OptOps.UNROLL, reduce_ranges_used[-1].arg[0], reduce_unroll),)
                op_groups: dict[Ops, list[int]] = {}
                for i, red in enumerate(reduce_uops):
                    op_groups.setdefault(red.arg[0], []).append(i)
                stores = []
                idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(reduce_ranges)))
                expr_cache: dict[bytes, UOp] = {}
                for op_kind, group in op_groups.items():
                    exprs = []
                    for gi in group:
                        red = reduce_uops[gi]
                        expr = red.src[0].substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                        expr = rewrite_elem(expr, reduce_ranges)
                        if expr.dtype.count > 1:
                            expr = _hreduce_vector(expr, op_kind)
                        cached = expr_cache.get(expr.key)
                        if cached is None:
                            expr_cache[expr.key] = expr
                        else:
                            expr = cached
                        exprs.append(expr)
                    vec_expr = exprs[0].vectorize(*exprs[1:]) if len(exprs) > 1 else exprs[0]
                    red = UOp(Ops.REDUCE, vec_expr.dtype, src=(vec_expr,)+tuple(reduce_ranges), arg=op_kind)
                    red_val = red.vindex(*idx)
                    for j, gi in enumerate(group):
                        out = outs[gi]
                        stores.append(out.index(UOp.const(dtypes.index, 0), ptr=True).store(red_val.gep(j)).end(*reduce_ranges_used))
                return UOp.group(*stores).sink(arg=KernelInfo(name="coupled_reduce_multi", opts_to_apply=opts))

            use_vec = vector_width > 1
            if use_vec:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
                ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                q_ptr = q.index(*ranges[:-1], base, ptr=True)
                p_ptr = p.index(*ranges[:-1], base, ptr=True)
                vec_dtype = q.dtype.base.vec(vector_width)
                q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            else:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
                q_val = q.vindex(*ranges)
                p_val = p.vindex(*ranges)
            reduce_counter = [len(ranges) + 1]
            reduce_ranges = [
                UOp.range(ranges[i].src[0], reduce_counter[0] + i, AxisType.REDUCE) for i in range(len(ranges))
            ]
            range_args = tuple(r.arg for r in ranges)
            range_arg_to_axis = {r.arg: i for i, r in enumerate(ranges)}
            reduce_map = {i: reduce_ranges[i] for i in range(len(ranges))}
            def drop_value_index(ctx: dict, uop: UOp) -> UOp|None:
                if uop.op is not Ops.INDEX or uop.arg != "value": return None
                src0 = uop.src[0]
                if src0 not in (ctx["q_val"], ctx["p_val"]): return None
                if len(uop.src[1:]) != len(ctx["ranges"]): return None
                for s, r in zip(uop.src[1:], ctx["ranges"]):
                    if s is not r: return None
                return src0

            def lower_full_reduce_shared(red: UOp) -> UOp:
                def reindex_reduce_full(ctx: dict, uop: UOp) -> UOp|None:
                    if uop.op is not Ops.INDEX or uop.arg != "value": return None
                    idx = []
                    changed = False
                    for s in uop.src[1:]:
                        if s.op is Ops.RANGE and s.arg in ctx["range_arg_to_axis"]:
                            idx.append(ctx["reduce_map"].get(ctx["range_arg_to_axis"][s.arg], s))
                            changed = True
                        else:
                            idx.append(s)
                    idx = tuple(idx)
                    if not changed: return None
                    return uop.src[0].vindex(*idx, dtype=uop.dtype)

                def replace_range(ctx: dict, uop: UOp) -> UOp|None:
                    if uop.op is not Ops.RANGE or uop.arg[-1] != AxisType.LOOP: return None
                    rr = ctx["replace_map"].get(uop.arg[0])
                    if rr is None: return None
                    return rr

                src0 = graph_rewrite(
                    red.src[0],
                    PatternMatcher([(UPat(Ops.INDEX, name="uop"), reindex_reduce_full)], compiled=False),
                    ctx={"ranges": ranges, "range_args": range_args, "range_arg_to_axis": range_arg_to_axis, "reduce_map": reduce_map},
                    name="reduce_full_reindex_shared",
                )
                replace_map = {ranges[i].arg[0]: reduce_map[i] for i in range(len(ranges))}
                src0 = graph_rewrite(
                    src0,
                    PatternMatcher([(UPat(Ops.RANGE, name="uop"), replace_range)], compiled=False),
                    ctx={"replace_map": replace_map},
                    name="reduce_full_range_shared",
                )
                reduced = UOp(Ops.REDUCE, red.dtype, src=(src0,)+tuple(reduce_ranges), arg=red.arg[0])
                if reduced.dtype.count > 1:
                    scalar = _hreduce_vector(reduced, red.arg[0])
                    reduced = scalar.vectorize(*([scalar] * (reduced.dtype.count - 1)))
                idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(ranges)))
                return reduced.vindex(*idx)

            stores = []
            opts = ()
            groupable = []
            groupable_idx = set()
            for i, red in enumerate(reduce_uops):
                if any(u.op is Ops.REDUCE_AXIS for u in red.src[0].toposort()): continue
                if red.arg[1] is None or len(red.arg[1]) != len(shape): continue
                if not all(j in red.arg[1] for j in range(len(shape))): continue
                groupable.append((i, red))
                groupable_idx.add(i)
            if len(groupable) > 1:
                expr_cache: dict[bytes, UOp] = {}
                if use_vec:
                    reduce_ranges_group = [UOp.range(s, i+1, AxisType.REDUCE) for i, s in enumerate(shape[:-1])]
                    reduce_ranges_group.append(UOp.range(shape[-1] // vector_width, len(shape), AxisType.REDUCE))
                    base = reduce_ranges_group[-1] * UOp.const(dtypes.index, vector_width)
                    q_ptr = q.index(*reduce_ranges_group[:-1], base, ptr=True)
                    p_ptr = p.index(*reduce_ranges_group[:-1], base, ptr=True)
                    vec_dtype = q.dtype.base.vec(vector_width)
                    q_val_red = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                    p_val_red = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
                else:
                    reduce_ranges_group = [UOp.range(s, i+1, AxisType.REDUCE) for i, s in enumerate(shape)]
                    q_val_red = q.vindex(*reduce_ranges_group)
                    p_val_red = p.vindex(*reduce_ranges_group)
                idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(reduce_ranges_group)))
                op_groups: dict[Ops, list[tuple[int, UOp]]] = {}
                for i, red in groupable:
                    op_groups.setdefault(red.arg[0], []).append((i, red))
                for op_kind, group in op_groups.items():
                    exprs = []
                    for _, red in group:
                        expr = red.src[0].substitute({q_sym_uop: q_val_red, p_sym_uop: p_val_red})
                        expr = rewrite_elem(expr, reduce_ranges_group)
                        if expr.dtype.count > 1:
                            expr = _hreduce_vector(expr, op_kind)
                        cached = expr_cache.get(expr.key)
                        if cached is None:
                            expr_cache[expr.key] = expr
                        else:
                            expr = cached
                        exprs.append(expr)
                    vec_expr = exprs[0].vectorize(*exprs[1:]) if len(exprs) > 1 else exprs[0]
                    red = UOp(Ops.REDUCE, vec_expr.dtype, src=(vec_expr,)+tuple(reduce_ranges_group), arg=op_kind)
                    red_val = red.vindex(*idx)
                    for j, (i, _) in enumerate(group):
                        out = outs[i]
                        stores.append(out.index(UOp.const(dtypes.index, 0), ptr=True).store(red_val.gep(j)).end(*reduce_ranges_group))
                if not opts and reduce_unroll > 1 and reduce_ranges_group:
                    opts = (Opt(OptOps.UNROLL, reduce_ranges_group[-1].arg[0], reduce_unroll),)
            for i, (reduce_uop, out) in enumerate(zip(reduce_uops, outs, strict=True)):
                if i in groupable_idx: continue
                red = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat(Ops.INDEX, name="uop"), drop_value_index)], compiled=False),
                    ctx={"ranges": ranges, "q_val": q_val, "p_val": p_val},
                    name="drop_value_index",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), lambda ctx, x: lower_full_reduce_shared(x))], compiled=False),
                    ctx={},
                    name="lower_reduce_axis_shared",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges},
                    name="const_to_vindex",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape)], compiled=False),
                    ctx={},
                    name="drop_const_reshape",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_value_index",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                    ctx={"ranges": ranges},
                    name="reindex_reduce_input",
                )
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                reduce_ranges_used = [r for r in red.ranges if r.arg[-1] == AxisType.REDUCE]
                if not opts and reduce_unroll > 1 and reduce_ranges_used:
                    opts = (Opt(OptOps.UNROLL, reduce_ranges_used[-1].arg[0], reduce_unroll),)
                stores.append(out.index(UOp.const(dtypes.index, 0), ptr=True).store(red).end(*reduce_ranges_used))
            return UOp.group(*stores).sink(arg=KernelInfo(name="coupled_reduce_multi", opts_to_apply=opts))

        return kernel

    def _build_coupled_reduce_kernel_qnew(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                          dHdq_elem_uop: UOp, dHdp_elem_uop: UOp, reduce_uop: UOp,
                                          placeholders_dHdq: list[UOp]) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        axis = reduce_uop.arg[1]
        if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
            raise ValueError("split reduce kernel only supports full reductions")

        def kernel(q: UOp, p: UOp, acc0: UOp, out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            reduce_counter = [len(ranges) + 1]
            def rewrite_elem(uop: UOp) -> UOp:
                def squeeze_unit_alu(ctx: dict, x: UOp) -> UOp|None:
                    if x.op not in GroupOp.ALU: return None
                    if any(s.op is Ops.RANGE for s in x.src): return None
                    try:
                        if x.shape != (1,): return None
                    except Exception:
                        return None
                    for s in x.src:
                        try:
                            if s.shape not in ((), (1,)): return None
                        except Exception:
                            return None
                    return x.forced_reshape(())
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                    ctx={},
                    name="drop_reg_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand_deep)], compiled=False),
                    ctx={},
                    name="drop_reg_expand_deep",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                    ctx={},
                    name="drop_reg_expand",
                )
                return uop
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            acc0_val = acc0.vindex(UOp.const(dtypes.index, 0))
            dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            if placeholders_dHdq:
                acc_loads = {placeholders_dHdq[i]: acc0_val for i in range(len(placeholders_dHdq))}
                dHdq_1 = dHdq_1.substitute(acc_loads, name="reduce_placeholders")
            dHdq_1 = rewrite_elem(dHdq_1)
            half_dt = UOp.const(dHdq_1.dtype, 0.5 * dt, shape=tuple(r.vmax + 1 for r in ranges))
            neg_one = UOp.const(dHdq_1.dtype, -1.0, shape=tuple(r.vmax + 1 for r in ranges))
            p_half = p_val + (half_dt * dHdq_1) * neg_one
            dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
            dHdp = rewrite_elem(dHdp)
            dt_uop = UOp.const(dHdp.dtype, dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_new = q_val + dt_uop * dHdp
            red = reduce_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                name="lower_reduce_axis",
            )
            red = rewrite_elem(red)
            red = graph_rewrite(
                red,
                PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                ctx={"ranges": ranges},
                name="reindex_reduce_input",
            )
            reduce_ranges = [r for r in red.ranges if r.arg[-1] == AxisType.REDUCE]
            store = out.index(UOp.const(dtypes.index, 0), ptr=True).store(red)
            return store.end(*reduce_ranges).sink(arg=KernelInfo(name="coupled_reduce_qnew", opts_to_apply=()))

        return kernel

    def _build_coupled_reduce_kernel_qnew_multi(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                                dHdp_elem_uop: UOp, reduce_uops: tuple[UOp, ...],
                                                placeholders_dHdp: list[UOp], vector_width: int = 1,
                                                reduce_unroll: int = 1) -> Callable:
        if len(reduce_uops) == 0:
            raise ValueError("reduce_uops must be non-empty")
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        for reduce_uop in reduce_uops:
            axis = reduce_uop.arg[1]
            if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
                raise ValueError("split reduce kernel only supports full reductions")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if reduce_unroll < 1:
            raise ValueError("reduce_unroll must be >= 1")

        def kernel(q: UOp, p_half: UOp, *accs_and_out: UOp) -> UOp:
            outs = accs_and_out[-len(reduce_uops):]
            accs = accs_and_out[:-len(reduce_uops)]
            accs_dHdp = accs
            use_vec = vector_width > 1
            if use_vec:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
                ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                q_ptr = q.index(*ranges[:-1], base, ptr=True)
                p_ptr = p_half.index(*ranges[:-1], base, ptr=True)
                vec_dtype = q.dtype.base.vec(vector_width)
                q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                p_half_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            else:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
                q_val = q.vindex(*ranges)
                p_half_val = p_half.vindex(*ranges)
            reduce_counter = [len(ranges) + 1]
            def rewrite_elem(uop: UOp) -> UOp:
                def squeeze_unit_alu(ctx: dict, x: UOp) -> UOp|None:
                    if x.op not in GroupOp.ALU: return None
                    if any(s.op is Ops.RANGE for s in x.src): return None
                    try:
                        if x.shape != (1,): return None
                    except Exception:
                        return None
                    for s in x.src:
                        try:
                            if s.shape not in ((), (1,)): return None
                        except Exception:
                            return None
                    return x.forced_reshape(())
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                return uop
            dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half_val})
            if placeholders_dHdp:
                acc_loads = {
                    placeholders_dHdp[i]: accs_dHdp[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdp))
                }
                dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
            dHdp = rewrite_elem(dHdp)
            dt_uop = UOp.const(dHdp.dtype, dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_new = q_val + dt_uop * dHdp
            stores = []
            opts = ()
            for reduce_uop, out in zip(reduce_uops, outs, strict=True):
                red = reduce_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half_val})
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                    ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                    name="lower_reduce_axis",
                )
                red = rewrite_elem(red)
                red = graph_rewrite(
                    red,
                    PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                    ctx={"ranges": ranges},
                    name="reindex_reduce_input",
                )
                reduce_ranges = [r for r in red.ranges if r.arg[-1] == AxisType.REDUCE]
                if not opts and reduce_unroll > 1 and reduce_ranges:
                    opts = (Opt(OptOps.UNROLL, reduce_ranges[-1].arg[0], reduce_unroll),)
                stores.append(out.index(UOp.const(dtypes.index, 0), ptr=True).store(red).end(*reduce_ranges))
            return UOp.group(*stores).sink(arg=KernelInfo(name="coupled_reduce_qnew_multi", opts_to_apply=opts))

        return kernel

    def _build_coupled_elem_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                   elem_uop: UOp, placeholders: list[UOp], op_name: str) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)

        def kernel(q: UOp, p: UOp, *accs_and_out: UOp) -> UOp:
            out = accs_and_out[-1]
            accs = accs_and_out[:-1]
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            def ensure_broadcast(uop: UOp, target: UOp) -> UOp:
                try:
                    tshape = target.shape
                except Exception:
                    return uop
                if tshape is None: return uop
                try:
                    ushape = uop.shape
                except Exception:
                    return uop
                if ushape == tshape: return uop
                try:
                    if ushape in ((), (1,)) and len(tshape) > 0:
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(tshape))))
                        return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                    if ushape is None or len(ushape) != len(tshape):
                        return uop
                    return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                except Exception:
                    return uop
            expr = elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            if placeholders:
                acc_loads = {
                    placeholders[i]: accs[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders))
                }
                expr = expr.substitute(acc_loads, name="reduce_placeholders")
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                ctx={"ranges": ranges},
                name="const_to_vindex",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                ctx={},
                name="drop_scalar_value_index",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape)], compiled=False),
                ctx={},
                name="drop_const_reshape",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_value_index",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_scalar_index",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                ctx={"ranges": ranges},
                name="broadcast_scalar_base",
            )
            expr = graph_rewrite(
                expr,
                PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                ctx={},
                name="strip_device_consts",
            )
            shape_vals = tuple(r.vmax + 1 for r in ranges)
            if op_name in ("p_half", "p_new"):
                expr = ensure_broadcast(expr, p_val)
                half_dt = UOp.const(expr.dtype, 0.5 * dt, shape=shape_vals)
                neg_one = UOp.const(expr.dtype, -1.0, shape=shape_vals)
                out_val = p_val + (half_dt * expr) * neg_one
            else:
                expr = ensure_broadcast(expr, q_val)
                dt_uop = UOp.const(expr.dtype, dt, shape=shape_vals)
                out_val = q_val + dt_uop * expr
            store = out.index(*ranges, ptr=True).store(out_val)
            return store.end(*ranges).sink(arg=KernelInfo(name=f"coupled_{op_name}", opts_to_apply=()))

        return kernel

    def _build_coupled_elem_kernel_qp_new(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                          dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                          placeholders_dHdp: list[UOp]) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)

        def kernel(q: UOp, p: UOp, *accs_and_out: UOp) -> UOp:
            q_out = accs_and_out[-2]
            p_out = accs_and_out[-1]
            accs_dHdp = accs_and_out[:-2]
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            def broadcast_scalar_const(ctx: dict, uop: UOp) -> UOp|None:
                if uop.op not in GroupOp.Binary: return None
                def maybe_broadcast(c: UOp, other: UOp) -> UOp|None:
                    if c.op not in (Ops.CONST, Ops.VCONST): return None
                    try:
                        cshape = c.shape
                    except Exception:
                        return None
                    if cshape not in ((), (1,)): return None
                    try:
                        oshape = other.shape
                    except Exception:
                        return None
                    if oshape is None or oshape == cshape or oshape == (1,): return None
                    return UOp.const(c.dtype, c.arg, shape=oshape)
                a, b = uop.src
                na = maybe_broadcast(a, b)
                if na is None: na = a
                nb = maybe_broadcast(b, a)
                if nb is None: nb = b
                if na is a and nb is b: return None
                return UOp(uop.op, uop.dtype, (na, nb), uop.arg, uop.tag)
            def rewrite_elem(uop: UOp) -> UOp:
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_rewrite_enter", flush=True)
                def squeeze_unit_alu(ctx: dict, x: UOp) -> UOp|None:
                    if x.op not in GroupOp.ALU: return None
                    if any(s.op is Ops.RANGE for s in x.src): return None
                    try:
                        if x.shape != (1,): return None
                    except Exception:
                        return None
                    for s in x.src:
                        try:
                            if s.shape not in ((), (1,)): return None
                        except Exception:
                            return None
                    return x.forced_reshape(())
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="const_to_vindex",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_const_to_vindex", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_drop_scalar_value_index", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_drop_empty_reshape", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_drop_zero_vec_shape", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="broadcast_value_index",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_broadcast_value_index", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_broadcast_scalar_index", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_broadcast_scalar_base", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                    ctx={},
                    name="drop_reg_expand",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_drop_reg_expand", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                    ctx={},
                    name="broadcast_scalar_alu",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_broadcast_scalar_alu", flush=True)
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), broadcast_scalar_const)], compiled=False),
                    ctx={"device": device},
                    name="broadcast_scalar_const",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.ALU, name="x"), squeeze_unit_alu)], compiled=False),
                    ctx={},
                    name="squeeze_unit_alu",
                )
                if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                    print("qp_new_after_broadcast_scalar_const", flush=True)
                return uop
            def ensure_broadcast(uop: UOp, target: UOp) -> UOp:
                try:
                    tshape = target.shape
                except Exception:
                    return uop
                if tshape is None: return uop
                try:
                    ushape = uop.shape
                except Exception:
                    return uop
                if ushape == tshape: return uop
                try:
                    if ushape in ((), (1,)) and len(tshape) > 0:
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(tshape))))
                        return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                    if ushape is None or len(ushape) != len(tshape):
                        return uop
                    return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                except Exception:
                    return uop
            def squeeze_unit(uop: UOp) -> UOp:
                try:
                    if uop.shape == (1,): return uop.forced_reshape(())
                except Exception:
                    return uop
                return uop
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdq_1 = rewrite_elem(dHdq_1)
            dHdq_1 = squeeze_unit(dHdq_1)
            dHdq_1 = ensure_broadcast(dHdq_1, p_val)
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_after_dHdq_1", flush=True)
            half_dt = UOp.const(dHdq_1.dtype, 0.5 * dt, shape=tuple(r.vmax + 1 for r in ranges))
            neg_one = UOp.const(dHdq_1.dtype, -1.0, shape=tuple(r.vmax + 1 for r in ranges))
            p_half = p_val + (half_dt * dHdq_1) * neg_one
            dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
            if placeholders_dHdp:
                acc_loads = {
                    placeholders_dHdp[i]: accs_dHdp[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdp))
                }
                dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
            dHdp = rewrite_elem(dHdp)
            dHdp = squeeze_unit(dHdp)
            dHdp = ensure_broadcast(dHdp, q_val)
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_after_dHdp", flush=True)
            dt_uop = UOp.const(dHdp.dtype, dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_new = q_val + dt_uop * dHdp
            dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
            dHdq_2 = rewrite_elem(dHdq_2)
            dHdq_2 = squeeze_unit(dHdq_2)
            dHdq_2 = ensure_broadcast(dHdq_2, p_half)
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_after_dHdq_2", flush=True)
            p_new = p_half + (half_dt * dHdq_2) * neg_one
            store_q = q_out.index(*ranges, ptr=True).store(q_new)
            store_p = p_out.index(*ranges, ptr=True).store(p_new)
            sink = UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="coupled_qp_new", opts_to_apply=()))
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_before_sink_rewrites", flush=True)
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                ctx={},
                name="broadcast_scalar_alu_sink",
            )
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_after_broadcast_scalar_alu_sink", flush=True)
            sink = _cse_uops(sink)
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                limit = getenv("TINYGRAD_COUPLED_FUSED_DEBUG_LIMIT", 20000)
                count = _count_uops_limit(sink, limit)
                print(f"fused_qp_reduce_uops~{count}")
                raise RuntimeError("fused debug stop")
            return sink

        return kernel

    def _build_coupled_elem_kernel_qp_update(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                             dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                             placeholders_dHdq: list[UOp], placeholders_dHdp: list[UOp]) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)

        def kernel(q: UOp, p_half: UOp, *accs_and_out: UOp) -> UOp:
            q_out = accs_and_out[-2]
            p_out = accs_and_out[-1]
            accs = accs_and_out[:-2]
            accs_dHdp = accs[:len(placeholders_dHdp)]
            accs_dHdq_2 = accs[len(placeholders_dHdp):]
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            def broadcast_scalar_const(ctx: dict, uop: UOp) -> UOp|None:
                if uop.op not in GroupOp.Binary: return None
                def maybe_broadcast(c: UOp, other: UOp) -> UOp|None:
                    if c.op not in (Ops.CONST, Ops.VCONST): return None
                    try:
                        cshape = c.shape
                    except Exception:
                        return None
                    if cshape not in ((), (1,)): return None
                    try:
                        oshape = other.shape
                    except Exception:
                        return None
                    if oshape is None or oshape == cshape: return None
                    return UOp.const(c.dtype, c.arg, shape=oshape)
                a, b = uop.src
                na = maybe_broadcast(a, b)
                if na is None: na = a
                nb = maybe_broadcast(b, a)
                if nb is None: nb = b
                if na is a and nb is b: return None
                return UOp(uop.op, uop.dtype, (na, nb), uop.arg, uop.tag)
            def rewrite_elem(uop: UOp) -> UOp:
                def squeeze_unit_alu(ctx: dict, x: UOp) -> UOp|None:
                    if x.op not in GroupOp.ALU: return None
                    try:
                        if x.shape == (1,): return x.forced_reshape(())
                    except Exception:
                        return None
                    return None
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                    ctx={},
                    name="drop_reg_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                    ctx={},
                    name="broadcast_scalar_alu",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), broadcast_scalar_const)], compiled=False),
                    ctx={"device": device},
                    name="broadcast_scalar_const",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.ALU, name="x"), squeeze_unit_alu)], compiled=False),
                    ctx={},
                    name="squeeze_unit_alu",
                )
                return uop
            def ensure_broadcast(uop: UOp, target: UOp) -> UOp:
                try:
                    tshape = target.shape
                except Exception:
                    return uop
                if tshape is None: return uop
                try:
                    ushape = uop.shape
                except Exception:
                    return uop
                if ushape == tshape: return uop
                try:
                    if ushape in ((), (1,)) and len(tshape) > 0:
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(tshape))))
                        return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                    if ushape is None or len(ushape) != len(tshape):
                        return uop
                    return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                except Exception:
                    return uop
            q_val = q.vindex(*ranges)
            p_half_val = p_half.vindex(*ranges)
            dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half_val})
            if placeholders_dHdp:
                acc_loads = {
                    placeholders_dHdp[i]: accs_dHdp[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdp))
                }
                dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
            dHdp = rewrite_elem(dHdp)
            dHdp = ensure_broadcast(dHdp, q_val)
            shape_vals = tuple(r.vmax + 1 for r in ranges)
            dt_uop = UOp.const(dHdp.dtype, dt, shape=shape_vals)
            q_new = q_val + dt_uop * dHdp
            dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half_val})
            if placeholders_dHdq:
                acc_loads = {
                    placeholders_dHdq[i]: accs_dHdq_2[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdq))
                }
                dHdq_2 = dHdq_2.substitute(acc_loads, name="reduce_placeholders")
            dHdq_2 = rewrite_elem(dHdq_2)
            dHdq_2 = ensure_broadcast(dHdq_2, p_half_val)
            half_dt = UOp.const(dHdq_2.dtype, 0.5 * dt, shape=shape_vals)
            neg_one = UOp.const(dHdq_2.dtype, -1.0, shape=shape_vals)
            p_new = p_half_val + (half_dt * dHdq_2) * neg_one
            store_q = q_out.index(*ranges, ptr=True).store(q_new)
            store_p = p_out.index(*ranges, ptr=True).store(p_new)
            sink = UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="coupled_qp_update", opts_to_apply=()))
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                ctx={},
                name="broadcast_scalar_alu_sink",
            )
            if getenv("TINYGRAD_COUPLED_FUSED_DEBUG", 0):
                print("qp_new_before_cse", flush=True)
            return _cse_uops(sink)

        return kernel

    def _build_coupled_elem_kernel_qp_update_from_qp(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                                     dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                                     placeholders_dHdq: list[UOp], placeholders_dHdp: list[UOp]) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)

        def kernel(q: UOp, p: UOp, *accs_and_out: UOp) -> UOp:
            q_out = accs_and_out[-2]
            p_out = accs_and_out[-1]
            accs = accs_and_out[:-2]
            accs_dHdq = accs[:len(placeholders_dHdq)]
            accs_dHdp = accs[len(placeholders_dHdq):len(placeholders_dHdq) + len(placeholders_dHdp)]
            accs_dHdq_2 = accs[len(placeholders_dHdq) + len(placeholders_dHdp):]
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            def broadcast_scalar_const(ctx: dict, uop: UOp) -> UOp|None:
                if uop.op not in GroupOp.Binary: return None
                def maybe_broadcast(c: UOp, other: UOp) -> UOp|None:
                    if c.op not in (Ops.CONST, Ops.VCONST): return None
                    try:
                        cshape = c.shape
                    except Exception:
                        return None
                    if cshape not in ((), (1,)): return None
                    try:
                        oshape = other.shape
                    except Exception:
                        return None
                    if oshape is None or oshape == cshape: return None
                    return UOp.const(c.dtype, c.arg, shape=oshape)
                a, b = uop.src
                na = maybe_broadcast(a, b)
                if na is None: na = a
                nb = maybe_broadcast(b, a)
                if nb is None: nb = b
                if na is a and nb is b: return None
                return UOp(uop.op, uop.dtype, (na, nb), uop.arg, uop.tag)
            def rewrite_elem(uop: UOp) -> UOp:
                def squeeze_unit_alu(ctx: dict, x: UOp) -> UOp|None:
                    if x.op not in GroupOp.ALU: return None
                    try:
                        if x.shape == (1,): return x.forced_reshape(())
                    except Exception:
                        return None
                    return None
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges, "device": device},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                    ctx={},
                    name="drop_reg_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                    ctx={},
                    name="broadcast_scalar_alu",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), broadcast_scalar_const)], compiled=False),
                    ctx={"device": device},
                    name="broadcast_scalar_const",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.ALU, name="x"), squeeze_unit_alu)], compiled=False),
                    ctx={},
                    name="squeeze_unit_alu",
                )
                return uop
            def ensure_broadcast(uop: UOp, target: UOp) -> UOp:
                try:
                    tshape = target.shape
                except Exception:
                    return uop
                if tshape is None: return uop
                try:
                    ushape = uop.shape
                except Exception:
                    return uop
                if ushape == tshape: return uop
                try:
                    if ushape in ((), (1,)) and len(tshape) > 0:
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(tshape))))
                        return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                    if ushape is None or len(ushape) != len(tshape):
                        return uop
                    return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                except Exception:
                    return uop
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            if placeholders_dHdq:
                acc_loads = {
                    placeholders_dHdq[i]: accs_dHdq[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdq))
                }
                dHdq_1 = dHdq_1.substitute(acc_loads, name="reduce_placeholders")
            dHdq_1 = rewrite_elem(dHdq_1)
            dHdq_1 = ensure_broadcast(dHdq_1, p_val)
            half_dt = UOp.const(dHdq_1.dtype, 0.5 * dt, shape=tuple(r.vmax + 1 for r in ranges))
            neg_one = UOp.const(dHdq_1.dtype, -1.0, shape=tuple(r.vmax + 1 for r in ranges))
            p_half = p_val + (half_dt * dHdq_1) * neg_one
            dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
            if placeholders_dHdp:
                acc_loads = {
                    placeholders_dHdp[i]: accs_dHdp[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdp))
                }
                dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
            dHdp = rewrite_elem(dHdp)
            dHdp = ensure_broadcast(dHdp, q_val)
            dt_uop = UOp.const(dHdp.dtype, dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_new = q_val + dt_uop * dHdp
            dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
            if placeholders_dHdq:
                acc_loads = {
                    placeholders_dHdq[i]: accs_dHdq_2[i].vindex(UOp.const(dtypes.index, 0))
                    for i in range(len(placeholders_dHdq))
                }
                dHdq_2 = dHdq_2.substitute(acc_loads, name="reduce_placeholders")
            dHdq_2 = rewrite_elem(dHdq_2)
            dHdq_2 = ensure_broadcast(dHdq_2, p_half)
            p_new = p_half + (half_dt * dHdq_2) * neg_one
            store_q = q_out.index(*ranges, ptr=True).store(q_new)
            store_p = p_out.index(*ranges, ptr=True).store(p_new)
            sink = UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="coupled_qp_update_qp", opts_to_apply=()))
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                ctx={},
                name="broadcast_scalar_alu_sink",
            )
            return _cse_uops(sink)

        return kernel

    def _build_coupled_kernel_qp_new_with_reduce(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                                 dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                                 dHdq_reduce_uops: list[UOp], dHdp_reduce_uops: list[UOp],
                                                 placeholders_dHdq: list[UOp], placeholders_dHdp: list[UOp],
                                                 unroll_steps: int = 1, vector_width: int = 1,
                                                 reduce_unroll: int = 1) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        if placeholders_dHdq and len(placeholders_dHdq) != len(dHdq_reduce_uops):
            raise ValueError("placeholders_dHdq must match dHdq_reduce_uops length")
        if placeholders_dHdp and len(placeholders_dHdp) != len(dHdp_reduce_uops):
            raise ValueError("placeholders_dHdp must match dHdp_reduce_uops length")
        for reduce_uop in dHdq_reduce_uops:
            axis = reduce_uop.arg[1]
            if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
                raise ValueError("split reduce kernel only supports full reductions")
        for reduce_uop in dHdp_reduce_uops:
            axis = reduce_uop.arg[1]
            if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
                raise ValueError("split reduce kernel only supports full reductions")
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if reduce_unroll < 1:
            raise ValueError("reduce_unroll must be >= 1")

        def kernel(q: UOp, p: UOp, q_out: UOp, p_out: UOp) -> UOp:
            use_vec = vector_width > 1
            if use_vec:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
                ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            else:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            reduce_counter = [len(ranges) + 1]
            acc_dtype = q.dtype.base
            acc0_regs = [
                UOp(Ops.DEFINE_REG, acc_dtype.ptr(size=unroll_steps, addrspace=AddrSpace.REG), arg=i)
                for i in range(len(dHdq_reduce_uops))
            ]
            acc1_regs = [
                UOp(Ops.DEFINE_REG, acc_dtype.ptr(size=unroll_steps, addrspace=AddrSpace.REG), arg=i+len(dHdq_reduce_uops))
                for i in range(len(dHdq_reduce_uops))
            ]
            acc_hdp_regs = [
                UOp(Ops.DEFINE_REG, acc_dtype.ptr(size=unroll_steps, addrspace=AddrSpace.REG), arg=i+len(dHdq_reduce_uops)*2)
                for i in range(len(dHdp_reduce_uops))
            ]
            def broadcast_scalar_const(ctx: dict, uop: UOp) -> UOp|None:
                if uop.op not in GroupOp.Binary: return None
                def maybe_broadcast(c: UOp, other: UOp) -> UOp|None:
                    if c.op not in (Ops.CONST, Ops.VCONST): return None
                    try:
                        cshape = c.shape
                    except Exception:
                        return None
                    if cshape not in ((), (1,)): return None
                    try:
                        oshape = other.shape
                    except Exception:
                        return None
                    if oshape is None or oshape == cshape: return None
                    return UOp.const(c.dtype, c.arg, shape=oshape)
                a, b = uop.src
                na = maybe_broadcast(a, b)
                if na is None: na = a
                nb = maybe_broadcast(b, a)
                if nb is None: nb = b
                if na is a and nb is b: return None
                return UOp(uop.op, uop.dtype, (na, nb), uop.arg, uop.tag)
            def rewrite_elem(uop: UOp, ranges_override: list[UOp]|None = None) -> UOp:
                ranges_use = ranges_override if ranges_override is not None else ranges
                def strip_device_const_noptr(ctx: dict, c: UOp) -> UOp|None:
                    if c.op not in (Ops.CONST, Ops.VCONST): return None
                    if len(c.src) == 0 or c.src[0].op is not Ops.DEVICE: return None
                    if isinstance(c.dtype, PtrDType): return None
                    return UOp.const(c.dtype, c.arg)
                def const_ptr_index_to_const(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.INDEX or x.arg != "value": return None
                    base = x.src[0]
                    if base.op is not Ops.CONST: return None
                    if not isinstance(base.dtype, PtrDType): return None
                    return UOp.const(x.dtype, base.arg)
                def drop_redundant_expand(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.EXPAND: return None
                    base = x.src[0]
                    if base.op is not Ops.RESHAPE: return None
                    if base.marg != (1,): return None
                    if base.src[0].op in GroupOp.ALU: return base.src[0]
                    return None
                def drop_empty_expand(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.EXPAND: return None
                    try:
                        shp = x.shape
                    except Exception:
                        return x.src[0]
                    if shp is None: return x.src[0]
                    return None
                def drop_unit_expand(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.EXPAND: return None
                    try:
                        if x.marg == (1,): return x.src[0]
                    except Exception:
                        return None
                    return None
                def normalize_expand_arg(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.EXPAND: return None
                    if x.arg is not None: return None
                    if len(x.src) != 2: return None
                    try:
                        marg = x.marg
                    except Exception:
                        return None
                    try:
                        if len(marg) == 1 and x.src[0].shape == ():
                            return x.src[0].forced_reshape((1,))._mop(Ops.EXPAND, marg, same_shape_noop=True)
                        if x.src[0].shape == ():
                            return x.src[0].forced_reshape(tuple(1 for _ in range(len(marg))))._mop(
                                Ops.EXPAND, marg, same_shape_noop=True)
                    except Exception:
                        pass
                    return x.src[0]._mop(Ops.EXPAND, marg, same_shape_noop=True)
                def drop_bad_reshape(ctx: dict, x: UOp) -> UOp|None:
                    if x.op is not Ops.RESHAPE: return None
                    if x.marg != (1,): return None
                    return x.src[0]
                if any(r.op is not Ops.RANGE for r in ranges_use):
                    uop = graph_rewrite(
                        uop,
                        PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_expand_any)], compiled=False),
                        ctx={},
                        name="drop_const_expand_any",
                    )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.INDEX, name="x"), const_ptr_index_to_const)], compiled=False),
                    ctx={},
                    name="const_ptr_index_to_const",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index)], compiled=False),
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.RESHAPE, name="x"), drop_bad_reshape)], compiled=False),
                    ctx={},
                    name="drop_bad_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_empty_reshape)], compiled=False),
                    ctx={},
                    name="drop_empty_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), drop_redundant_expand)], compiled=False),
                    ctx={},
                    name="drop_redundant_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), drop_empty_expand)], compiled=False),
                    ctx={},
                    name="drop_empty_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), drop_unit_expand)], compiled=False),
                    ctx={},
                    name="drop_unit_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), normalize_expand_arg)], compiled=False),
                    ctx={},
                    name="normalize_expand_arg",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), fix_scalar_expand)], compiled=False),
                    ctx={},
                    name="fix_scalar_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.EXPAND, name="x"), fix_reshape_expand)], compiled=False),
                    ctx={},
                    name="fix_reshape_expand",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_zero_vec_shape)], compiled=False),
                    ctx={},
                    name="drop_zero_vec_shape",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index)], compiled=False),
                    ctx={"ranges": ranges_use},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index)], compiled=False),
                    ctx={"ranges": ranges_use},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base)], compiled=False),
                    ctx={"ranges": ranges_use},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                    ctx={},
                    name="broadcast_scalar_alu",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), strip_device_const_noptr)], compiled=False),
                    ctx={},
                    name="strip_device_consts",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand_deep)], compiled=False),
                    ctx={},
                    name="drop_reg_expand_deep_post",
                )
                return uop

            def ensure_broadcast(uop: UOp, target: UOp) -> UOp:
                try:
                    tshape = target.shape
                except Exception:
                    return uop
                if tshape is None: return uop
                try:
                    ushape = uop.shape
                except Exception:
                    return uop
                if ushape == tshape: return uop
                try:
                    if ushape in ((), (1,)) and len(tshape) > 0:
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(tshape))))
                        return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                    if ushape is None or len(ushape) != len(tshape):
                        return uop
                    return uop._mop(Ops.EXPAND, tshape, same_shape_noop=True)
                except Exception:
                    return uop
            def fix_scalar_expand(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.EXPAND: return None
                try:
                    base_shape = x.base._shape
                except Exception:
                    return None
                if base_shape != (): return None
                marg = x.marg
                if marg is None: return None
                try:
                    reshaped = x.base.forced_reshape(tuple(1 for _ in range(len(marg))))
                    return reshaped._mop(Ops.EXPAND, marg, same_shape_noop=True)
                except Exception:
                    return None
            def fix_reshape_expand(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.EXPAND: return None
                src0 = x.src[0]
                if src0.op is not Ops.RESHAPE: return None
                try:
                    base_shape = src0.src[0]._shape
                except Exception:
                    return None
                if base_shape != (): return None
                marg = x.marg
                if marg is None: return None
                try:
                    reshaped = src0.src[0].forced_reshape(tuple(1 for _ in range(len(marg))))
                    return reshaped._mop(Ops.EXPAND, marg, same_shape_noop=True)
                except Exception:
                    return None
            def broadcast_scalar(uop: UOp) -> UOp:
                if shape_vals is None: return uop
                try:
                    if uop.shape in ((), (1,)):
                        uop = uop.forced_reshape(tuple(1 for _ in range(len(shape_vals))))
                        return uop._mop(Ops.EXPAND, shape_vals, same_shape_noop=True)
                except Exception:
                    return uop
                return uop
            if use_vec:
                base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                q_ptr = q.index(*ranges[:-1], base, ptr=True)
                p_ptr = p.index(*ranges[:-1], base, ptr=True)
                vec_dtype = q.dtype.base.vec(vector_width)
                q_vec_ptr = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                p_vec_ptr = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                q_val = q_vec_ptr.load()
                p_val = p_vec_ptr.load()
                q_vec = q_val
                p_vec = p_val
                q_lanes = [q_val.gep(i) for i in range(vector_width)]
                p_lanes = [p_val.gep(i) for i in range(vector_width)]
            else:
                q_val = q.vindex(*ranges)
                p_val = p.vindex(*ranges)
            shape_vals = None if use_vec else tuple(r.vmax + 1 for r in ranges)
            const_dtype = q_lanes[0].dtype if use_vec else q.dtype
            if use_vec:
                half_dt = UOp.const(const_dtype, 0.5 * dt, shape=shape_vals)
                neg_one = UOp.const(const_dtype, -1.0, shape=shape_vals)
                dt_uop = UOp.const(const_dtype, dt, shape=shape_vals)
            else:
                half_dt = UOp.const(const_dtype, 0.5 * dt).vindex(*ranges)
                neg_one = UOp.const(const_dtype, -1.0).vindex(*ranges)
                dt_uop = UOp.const(const_dtype, dt).vindex(*ranges)
            reduce_unroll_axis = None
            for step in range(unroll_steps):
                acc0_vals = []
                acc0_idx = UOp.const(dtypes.index, step)
                for i, reduce_uop in enumerate(dHdq_reduce_uops):
                    if use_vec:
                        red0 = reduce_uop.substitute({q_sym_uop: q_vec, p_sym_uop: p_vec})
                    else:
                        red0 = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    red0 = graph_rewrite(
                        red0,
                        PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                        ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                        name="lower_reduce_axis",
                    )
                    red0 = graph_rewrite(
                        red0,
                        PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                        ctx={},
                        name="broadcast_scalar_alu_red0",
                    )
                    if use_vec and red0.dtype.count > 1:
                        red0 = _hreduce_vector(red0, reduce_uop.arg[0])
                    reduce_ranges0 = [r for r in red0.ranges if r.arg[-1] == AxisType.REDUCE]
                    if reduce_unroll_axis is None and reduce_ranges0:
                        reduce_unroll_axis = reduce_ranges0[-1].arg[0]
                    acc0_store = acc0_regs[i].index(acc0_idx).store(red0).end(*reduce_ranges0)
                    acc0_val = acc0_regs[i].after(acc0_store).vindex(acc0_idx)
                    if not use_vec:
                        try:
                            if acc0_val.shape == (1,): acc0_val = acc0_val.forced_reshape(())
                        except Exception:
                            pass
                        acc0_val = ensure_broadcast(acc0_val, p_val)
                        acc0_val = broadcast_scalar(acc0_val)
                    acc0_vals.append(acc0_val)
                if use_vec:
                    p_half_lanes = []
                    for i in range(vector_width):
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width) + UOp.const(dtypes.index, i)
                        ranges_lane = list(ranges[:-1]) + [base]
                        dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_lanes[i], p_sym_uop: p_lanes[i]})
                        if placeholders_dHdq:
                            acc_loads = {placeholders_dHdq[j]: acc0_vals[j] for j in range(len(placeholders_dHdq))}
                            dHdq_1 = dHdq_1.substitute(acc_loads, name="reduce_placeholders")
                        dHdq_1 = rewrite_elem(dHdq_1, ranges_lane)
                        dHdq_1 = ensure_broadcast(dHdq_1, p_lanes[i])
                        p_half = p_lanes[i] + (half_dt * dHdq_1) * neg_one
                        p_half_lanes.append(p_half)
                    p_half_vec = p_half_lanes[0].vectorize(*p_half_lanes[1:])
                else:
                    dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    if placeholders_dHdq:
                        acc_loads = {placeholders_dHdq[j]: acc0_vals[j] for j in range(len(placeholders_dHdq))}
                        dHdq_1 = dHdq_1.substitute(acc_loads, name="reduce_placeholders")
                    dHdq_1 = rewrite_elem(dHdq_1)
                    dHdq_1 = ensure_broadcast(dHdq_1, p_val)
                    p_half = p_val + (half_dt * dHdq_1) * neg_one
                acc_hdp_vals = []
                if dHdp_reduce_uops:
                    acc_hdp_idx = UOp.const(dtypes.index, step)
                    for i, reduce_uop in enumerate(dHdp_reduce_uops):
                        if use_vec:
                            red_hdp = reduce_uop.substitute({q_sym_uop: q_vec, p_sym_uop: p_half_vec})
                        else:
                            red_hdp = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
                        red_hdp = graph_rewrite(
                            red_hdp,
                            PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                            ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                            name="lower_reduce_axis",
                        )
                        red_hdp = graph_rewrite(
                            red_hdp,
                            PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                            ctx={},
                            name="broadcast_scalar_alu_red_hdp",
                        )
                        if use_vec and red_hdp.dtype.count > 1:
                            red_hdp = _hreduce_vector(red_hdp, reduce_uop.arg[0])
                        reduce_ranges_hdp = [r for r in red_hdp.ranges if r.arg[-1] == AxisType.REDUCE]
                        if reduce_unroll_axis is None and reduce_ranges_hdp:
                            reduce_unroll_axis = reduce_ranges_hdp[-1].arg[0]
                        acc_hdp_store = acc_hdp_regs[i].index(acc_hdp_idx).store(red_hdp).end(*reduce_ranges_hdp)
                        acc_hdp_val = acc_hdp_regs[i].after(acc_hdp_store).vindex(acc_hdp_idx)
                        if not use_vec:
                            try:
                                if acc_hdp_val.shape == (1,): acc_hdp_val = acc_hdp_val.forced_reshape(())
                            except Exception:
                                pass
                            acc_hdp_val = ensure_broadcast(acc_hdp_val, p_val)
                            acc_hdp_val = broadcast_scalar(acc_hdp_val)
                        acc_hdp_vals.append(acc_hdp_val)
                if use_vec:
                    q_new_lanes = []
                    for i in range(vector_width):
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width) + UOp.const(dtypes.index, i)
                        ranges_lane = list(ranges[:-1]) + [base]
                        dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_lanes[i], p_sym_uop: p_half_lanes[i]})
                        if placeholders_dHdp:
                            acc_loads = {placeholders_dHdp[j]: acc_hdp_vals[j] for j in range(len(placeholders_dHdp))}
                            dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
                        dHdp = rewrite_elem(dHdp, ranges_lane)
                        dHdp = ensure_broadcast(dHdp, q_lanes[i])
                        q_new = q_lanes[i] + dt_uop * dHdp
                        q_new_lanes.append(q_new)
                    q_new_vec = q_new_lanes[0].vectorize(*q_new_lanes[1:])
                else:
                    dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
                    if placeholders_dHdp:
                        acc_loads = {placeholders_dHdp[j]: acc_hdp_vals[j] for j in range(len(placeholders_dHdp))}
                        dHdp = dHdp.substitute(acc_loads, name="reduce_placeholders")
                    dHdp = rewrite_elem(dHdp)
                    dHdp = ensure_broadcast(dHdp, q_val)
                    q_new = q_val + dt_uop * dHdp
                acc1_vals = []
                acc1_idx = UOp.const(dtypes.index, step)
                for i, reduce_uop in enumerate(dHdq_reduce_uops):
                    red1 = reduce_uop.substitute({
                        q_sym_uop: q_new_vec if use_vec else q_new,
                        p_sym_uop: p_half_vec if use_vec else p_half,
                    })
                    red1 = graph_rewrite(
                        red1,
                        PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                        ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                        name="lower_reduce_axis",
                    )
                    red1 = graph_rewrite(
                        red1,
                        PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                        ctx={},
                        name="broadcast_scalar_alu_red1",
                    )
                    if use_vec and red1.dtype.count > 1:
                        red1 = _hreduce_vector(red1, reduce_uop.arg[0])
                    reduce_ranges1 = [r for r in red1.ranges if r.arg[-1] == AxisType.REDUCE]
                    if reduce_unroll_axis is None and reduce_ranges1:
                        reduce_unroll_axis = reduce_ranges1[-1].arg[0]
                    acc1_store = acc1_regs[i].index(acc1_idx).store(red1).end(*reduce_ranges1)
                    acc1_val = acc1_regs[i].after(acc1_store).vindex(acc1_idx)
                    if not use_vec:
                        try:
                            if acc1_val.shape == (1,): acc1_val = acc1_val.forced_reshape(())
                        except Exception:
                            pass
                        acc1_val = ensure_broadcast(acc1_val, p_val)
                        acc1_val = broadcast_scalar(acc1_val)
                    acc1_vals.append(acc1_val)
                if use_vec:
                    p_new_lanes = []
                    for i in range(vector_width):
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width) + UOp.const(dtypes.index, i)
                        ranges_lane = list(ranges[:-1]) + [base]
                        dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new_lanes[i], p_sym_uop: p_half_lanes[i]})
                        if placeholders_dHdq:
                            acc_loads = {placeholders_dHdq[j]: acc1_vals[j] for j in range(len(placeholders_dHdq))}
                            dHdq_2 = dHdq_2.substitute(acc_loads, name="reduce_placeholders")
                        dHdq_2 = rewrite_elem(dHdq_2, ranges_lane)
                        dHdq_2 = ensure_broadcast(dHdq_2, p_half_lanes[i])
                        p_new = p_half_lanes[i] + (half_dt * dHdq_2) * neg_one
                        p_new_lanes.append(p_new)
                    q_lanes = q_new_lanes
                    p_lanes = p_new_lanes
                    q_vec = q_new_vec
                    p_vec = p_new_lanes[0].vectorize(*p_new_lanes[1:])
                else:
                    dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
                    if placeholders_dHdq:
                        acc_loads = {placeholders_dHdq[j]: acc1_vals[j] for j in range(len(placeholders_dHdq))}
                        dHdq_2 = dHdq_2.substitute(acc_loads, name="reduce_placeholders")
                    dHdq_2 = rewrite_elem(dHdq_2)
                    dHdq_2 = ensure_broadcast(dHdq_2, p_half)
                    p_new = p_half + (half_dt * dHdq_2) * neg_one
                    q_val, p_val = q_new, p_new
            if use_vec:
                q_val = q_lanes[0].vectorize(*q_lanes[1:])
                p_val = p_lanes[0].vectorize(*p_lanes[1:])
                q_out_ptr = q_out.index(*ranges[:-1], base, ptr=True)
                p_out_ptr = p_out.index(*ranges[:-1], base, ptr=True)
                q_out_vec_ptr = q_out_ptr.cast(vec_dtype.ptr(size=q_out_ptr.dtype.size, addrspace=q_out_ptr.dtype.addrspace))
                p_out_vec_ptr = p_out_ptr.cast(vec_dtype.ptr(size=p_out_ptr.dtype.size, addrspace=p_out_ptr.dtype.addrspace))
                store_q = q_out_vec_ptr.store(q_val)
                store_p = p_out_vec_ptr.store(p_val)
            else:
                store_q = q_out.index(*ranges, ptr=True).store(q_val)
                store_p = p_out.index(*ranges, ptr=True).store(p_val)
            opts_to_apply = ()
            if reduce_unroll_axis is not None and reduce_unroll > 1:
                opts_to_apply = (Opt(OptOps.UNROLL, reduce_unroll_axis, reduce_unroll),)
            sink = UOp.group(store_q, store_p).end(*ranges).sink(
                arg=KernelInfo(name=f"coupled_qp_new_with_reduce_{unroll_steps}", opts_to_apply=opts_to_apply))
            def strip_const_ptr_index(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.INDEX: return None
                base = x.src[0]
                if base.op is not Ops.CONST: return None
                if not isinstance(base.dtype, PtrDType): return None
                if x.arg == "value": return UOp.const(x.dtype, base.arg)
                if x.arg is None: return base
                return None
            def drop_bad_reshape_sink(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.RESHAPE: return None
                if x.marg != (1,): return None
                return x.src[0]
            def drop_empty_expand_sink(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.EXPAND: return None
                try:
                    shp = x.shape
                except Exception:
                    return x.src[0]
                if shp is None: return x.src[0]
                return None
            def drop_unit_expand_sink(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.EXPAND: return None
                try:
                    if x.marg == (1,): return x.src[0]
                except Exception:
                    return None
                return None
            def normalize_expand_arg_sink(ctx: dict, x: UOp) -> UOp|None:
                if x.op is not Ops.EXPAND: return None
                if x.arg is not None: return None
                if len(x.src) != 2: return None
                try:
                    marg = x.marg
                except Exception:
                    return None
                try:
                    if len(marg) == 1 and x.src[0].shape == ():
                        return x.src[0].forced_reshape((1,))._mop(Ops.EXPAND, marg, same_shape_noop=True)
                    if x.src[0].shape == ():
                        return x.src[0].forced_reshape(tuple(1 for _ in range(len(marg))))._mop(
                            Ops.EXPAND, marg, same_shape_noop=True)
                except Exception:
                    pass
                return x.src[0]._mop(Ops.EXPAND, marg, same_shape_noop=True)
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.INDEX, name="x"), strip_const_ptr_index)], compiled=False),
                ctx={},
                name="strip_const_ptr_index",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.RESHAPE, name="x"), drop_bad_reshape_sink)], compiled=False),
                ctx={},
                name="drop_bad_reshape_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.EXPAND, name="x"), drop_empty_expand_sink)], compiled=False),
                ctx={},
                name="drop_empty_expand_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.EXPAND, name="x"), drop_unit_expand_sink)], compiled=False),
                ctx={},
                name="drop_unit_expand_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.EXPAND, name="x"), normalize_expand_arg_sink)], compiled=False),
                ctx={},
                name="normalize_expand_arg_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.EXPAND, name="x"), fix_scalar_expand)], compiled=False),
                ctx={},
                name="fix_scalar_expand_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.EXPAND, name="x"), fix_reshape_expand)], compiled=False),
                ctx={},
                name="fix_reshape_expand_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(GroupOp.Binary, name="uop"), _broadcast_scalar_alu)], compiled=False),
                ctx={},
                name="broadcast_scalar_alu_sink",
            )
            sink = graph_rewrite(
                sink,
                PatternMatcher([(UPat(Ops.RESHAPE, name="x"), drop_bad_reshape_sink)], compiled=False),
                ctx={},
                name="drop_bad_reshape_sink_post",
            )
            return _cse_uops(sink)

        return kernel

    def _evolve_scan_kernel_coupled_split(self, q: Tensor, p: Tensor, dt: float, steps: int,
                                          unroll_steps: int = 1, vector_width: int = 1) -> tuple[Tensor, Tensor]:
        if any(u.op is Ops.SIN for u in self.H(q, p).uop.toposort()):
            return self.evolve_scan_kernel_coupled(q, p, dt, steps)[0:2]
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(q.device, q.shape, q.dtype)
        dHdq_reduce_nodes = [u for u in dHdq_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        dHdp_reduce_nodes = [u for u in dHdp_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        if vector_width == 1 and q.device == "CPU" and q.shape:
            if q.shape[-1] % 4 == 0:
                vector_width = 4
            elif q.shape[-1] % 2 == 0:
                vector_width = 2
        reduce_placeholders_dHdq = {red: UOp.const(red.dtype, 0, q.device) for red in dHdq_reduce_nodes}
        reduce_placeholders_dHdp = {red: UOp.const(red.dtype, 0, q.device) for red in dHdp_reduce_nodes}
        if reduce_placeholders_dHdq or reduce_placeholders_dHdp:
            def replace_reduce(ctx: dict, uop: UOp) -> UOp|None:
                repl = ctx["reduce_placeholders"].get(uop)
                if repl is None: return None
                return repl
            reduce_pm = PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="uop"), replace_reduce)], compiled=False)
            dHdq_elem_uop = graph_rewrite(
                dHdq_uop, reduce_pm, ctx={"reduce_placeholders": reduce_placeholders_dHdq}, name="replace_reduce")
            dHdp_elem_uop = graph_rewrite(
                dHdp_uop, reduce_pm, ctx={"reduce_placeholders": reduce_placeholders_dHdp}, name="replace_reduce")
        else:
            dHdq_elem_uop = dHdq_uop
            dHdp_elem_uop = dHdp_uop

        dHdq_placeholders = [reduce_placeholders_dHdq[r] for r in dHdq_reduce_nodes]
        dHdp_placeholders = [reduce_placeholders_dHdp[r] for r in dHdp_reduce_nodes]

        use_fused_qp = bool(dHdq_reduce_nodes or dHdp_reduce_nodes)
        if use_fused_qp and vector_width == 1 and q.device == "CPU" and q.shape and getenv("TINYGRAD_COUPLED_FUSED_VEC_TUNE_AUTO", 1):
            tune_key = (dt, q.device, q.shape, q.dtype)
            cached = self._fused_reduce_vec_tune_cache.get(tune_key)
            if cached is not None:
                vector_width = cached
            else:
                cand_vw = [1, 2, 4, 8]
                cand_vw = [v for v in cand_vw if q.shape[-1] % v == 0]
                if not cand_vw:
                    cand_vw = [1]
                best_vw = 1
                best_time = float("inf")
                for vw in cand_vw:
                    key_qp = (self.H, q.device, q.shape, q.dtype, "qp_new_reduce", dt, 1,
                              vw, 1, tuple(dHdq_reduce_nodes), tuple(dHdp_reduce_nodes))
                    kernel = self._elem_kernel_coupled_cache.get(key_qp)
                    if kernel is None:
                        kernel = self._build_coupled_kernel_qp_new_with_reduce(
                            dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                            dHdq_reduce_nodes, dHdp_reduce_nodes, dHdq_placeholders, dHdp_placeholders,
                            unroll_steps=1, vector_width=vw, reduce_unroll=1)
                        self._elem_kernel_coupled_cache[key_qp] = kernel
                    q_tmp = q.detach().clone().realize()
                    p_tmp = p.detach().clone().realize()
                    q_out = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
                    p_out = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                    start = time.perf_counter()
                    out = Tensor.custom_kernel(q_tmp, p_tmp, q_out, p_out, fxn=kernel)
                    Tensor.realize(out[2], out[3])
                    elapsed = time.perf_counter() - start
                    if elapsed < best_time:
                        best_time = elapsed
                        best_vw = vw
                vector_width = best_vw
                self._fused_reduce_vec_tune_cache[tune_key] = best_vw
        reduce_unroll = 1
        reduce_unroll_tuned = False
        tune_fused = use_fused_qp and (getenv("TINYGRAD_COUPLED_FUSED_TUNE", 0) or
                                       (q.device == "CPU" and getenv("TINYGRAD_COUPLED_FUSED_TUNE_AUTO", 1)))
        if tune_fused:
            tune_key = (dt, steps, q.device, q.shape, q.dtype)
            cached = self._fused_kernel_coupled_tune_cache.get(tune_key)
            if cached is not None:
                if len(cached) == 2:
                    vector_width, unroll_steps = cached
                else:
                    vector_width, unroll_steps, reduce_unroll = cached
                    reduce_unroll_tuned = True
            else:
                cand_vw = [1, 2, 4, 8]
                if vector_width > 1:
                    max_vw = vector_width
                else:
                    max_vw = max(cand_vw)
                cand_vw = [v for v in cand_vw if v <= max_vw and (not q.shape or q.shape[-1] % v == 0)]
                if vector_width > 1 and vector_width not in cand_vw and (not q.shape or q.shape[-1] % vector_width == 0):
                    cand_vw.append(vector_width)
                cand_unroll = [1, 2, 4, 8, 16]
                if unroll_steps > 1:
                    max_unroll = unroll_steps
                else:
                    max_unroll = max(cand_unroll)
                cand_unroll = [u for u in cand_unroll if u <= max_unroll and steps % u == 0]
                if unroll_steps > 1 and unroll_steps not in cand_unroll and steps % unroll_steps == 0:
                    cand_unroll.append(unroll_steps)
                if not cand_vw:
                    cand_vw = [vector_width]
                if not cand_unroll:
                    cand_unroll = [unroll_steps]
                best_time = float("inf")
                best = (vector_width, unroll_steps, reduce_unroll)
                for vw in cand_vw:
                    for unroll in cand_unroll:
                        cand_ru = [1, 2, 4, 8]
                        if q.shape:
                            last_extent = q.shape[-1] // vw if vw > 0 else q.shape[-1]
                            cand_ru = [r for r in cand_ru if r <= last_extent]
                        if not cand_ru:
                            cand_ru = [1]
                        for ru in cand_ru:
                            key_qp = (self.H, q.device, q.shape, q.dtype, "qp_new_reduce", dt, unroll,
                                      vw, ru, tuple(dHdq_reduce_nodes), tuple(dHdp_reduce_nodes))
                            kernel = self._elem_kernel_coupled_cache.get(key_qp)
                            if kernel is None:
                                kernel = self._build_coupled_kernel_qp_new_with_reduce(
                                    dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                                    dHdq_reduce_nodes, dHdp_reduce_nodes, dHdq_placeholders, dHdp_placeholders,
                                    unroll_steps=unroll, vector_width=vw, reduce_unroll=ru)
                                self._elem_kernel_coupled_cache[key_qp] = kernel
                            q_tmp = q.detach().clone().realize()
                            p_tmp = p.detach().clone().realize()
                            q_out = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
                            p_out = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                            start = time.perf_counter()
                            out = Tensor.custom_kernel(q_tmp, p_tmp, q_out, p_out, fxn=kernel)
                            Tensor.realize(out[2], out[3])
                            elapsed = time.perf_counter() - start
                            if elapsed < best_time:
                                best_time = elapsed
                                best = (vw, unroll, ru)
                vector_width, unroll_steps, reduce_unroll = best
                reduce_unroll_tuned = True
                self._fused_kernel_coupled_tune_cache[tune_key] = best
        reduce_vector_width = 1
        if vector_width > 1 and q.shape and q.shape[-1] % vector_width == 0:
            reduce_vector_width = vector_width
        if (dHdq_reduce_nodes or dHdp_reduce_nodes) and q.shape and not reduce_unroll_tuned and \
           not getenv("TINYGRAD_COUPLED_REDUCE_TUNE", 0) and \
           not getenv("TINYGRAD_COUPLED_REDUCE_TUNE_BOTH", 0) and not getenv("TINYGRAD_COUPLED_REDUCE_UNROLL_TUNE", 0):
            last_extent = q.shape[-1] // reduce_vector_width if reduce_vector_width > 0 else q.shape[-1]
            if last_extent >= 16384:
                reduce_unroll = 8
            elif last_extent >= 4096:
                reduce_unroll = 4
        if getenv("TINYGRAD_COUPLED_REDUCE_TUNE_BOTH", 0) and (dHdq_reduce_nodes or dHdp_reduce_nodes):
            tune_key = (q.device, q.shape, q.dtype)
            cached = self._reduce_kernel_coupled_tune_both_cache.get(tune_key)
            if cached is not None:
                reduce_vector_width, reduce_unroll = cached
            else:
                cand_vw = [1, 2, 4, 8]
                cand_vw = [c for c in cand_vw if q.shape and q.shape[-1] % c == 0]
                if not cand_vw:
                    cand_vw = [1]
                cand_ru = [1, 2, 4, 8]
                cand_ru = [u for u in cand_ru if u >= 1]
                best = (reduce_vector_width, reduce_unroll)
                best_time = float("inf")
                for vw in cand_vw:
                    for ru in cand_ru:
                        kernel = None
                        kernel_hdp = None
                        if dHdq_reduce_nodes:
                            key = (tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, vw, ru)
                            kernel = self._reduce_kernel_coupled_multi_cache.get(key)
                            if kernel is None:
                                kernel = self._build_coupled_reduce_kernel_multi(
                                    tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, vector_width=vw, reduce_unroll=ru)
                                self._reduce_kernel_coupled_multi_cache[key] = kernel
                        if dHdp_reduce_nodes:
                            key = (tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, vw, ru)
                            kernel_hdp = self._reduce_kernel_coupled_multi_cache.get(key)
                            if kernel_hdp is None:
                                kernel_hdp = self._build_coupled_reduce_kernel_multi(
                                    tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, vector_width=vw, reduce_unroll=ru)
                                self._reduce_kernel_coupled_multi_cache[key] = kernel_hdp
                        start = time.perf_counter()
                        if kernel is not None:
                            tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdq_reduce_nodes]
                            Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel)[2:])
                        if kernel_hdp is not None:
                            tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdp_reduce_nodes]
                            Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel_hdp)[2:])
                        elapsed = time.perf_counter() - start
                        if elapsed < best_time:
                            best_time = elapsed
                            best = (vw, ru)
                reduce_vector_width, reduce_unroll = best
                self._reduce_kernel_coupled_tune_both_cache[tune_key] = best
        if not getenv("TINYGRAD_COUPLED_REDUCE_TUNE_BOTH", 0) and getenv("TINYGRAD_COUPLED_REDUCE_TUNE", 0):
            tune_key = (q.device, q.shape, q.dtype)
            cached = self._reduce_kernel_coupled_tune_cache.get(tune_key)
            if cached is not None:
                reduce_vector_width = cached
            else:
                candidates = [1, 2, 4, 8]
                candidates = [c for c in candidates if q.shape and q.shape[-1] % c == 0]
                best_vw = 1
                best_time = float("inf")
                for vw in candidates:
                    if dHdq_reduce_nodes:
                        key = (tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, vw, 1)
                        kernel = self._reduce_kernel_coupled_multi_cache.get(key)
                        if kernel is None:
                            kernel = self._build_coupled_reduce_kernel_multi(
                                tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, vector_width=vw, reduce_unroll=1)
                            self._reduce_kernel_coupled_multi_cache[key] = kernel
                    else:
                        kernel = None
                    if dHdp_reduce_nodes:
                        key = (tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, vw, 1)
                        kernel_hdp = self._reduce_kernel_coupled_multi_cache.get(key)
                        if kernel_hdp is None:
                            kernel_hdp = self._build_coupled_reduce_kernel_multi(
                                tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, vector_width=vw, reduce_unroll=1)
                            self._reduce_kernel_coupled_multi_cache[key] = kernel_hdp
                    else:
                        kernel_hdp = None
                    start = time.perf_counter()
                    if kernel is not None:
                        tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdq_reduce_nodes]
                        Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel)[2:])
                    if kernel_hdp is not None:
                        tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdp_reduce_nodes]
                        Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel_hdp)[2:])
                    elapsed = time.perf_counter() - start
                    if elapsed < best_time:
                        best_time = elapsed
                        best_vw = vw
                self._reduce_kernel_coupled_tune_cache[tune_key] = best_vw
                reduce_vector_width = best_vw
        if not getenv("TINYGRAD_COUPLED_REDUCE_TUNE_BOTH", 0) and getenv("TINYGRAD_COUPLED_REDUCE_UNROLL_TUNE", 0):
            tune_key = (q.device, q.shape, q.dtype, reduce_vector_width)
            cached = self._reduce_kernel_coupled_unroll_tune_cache.get(tune_key)
            if cached is not None:
                reduce_unroll = cached
            else:
                candidates = [1, 2, 4, 8]
                if q.shape:
                    last_extent = q.shape[-1] // reduce_vector_width
                    candidates = [c for c in candidates if c <= last_extent]
                if not candidates:
                    candidates = [1]
                best_ru = 1
                best_time = float("inf")
                for ru in candidates:
                    if dHdq_reduce_nodes:
                        key = (tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, reduce_vector_width, ru)
                        kernel = self._reduce_kernel_coupled_multi_cache.get(key)
                        if kernel is None:
                            kernel = self._build_coupled_reduce_kernel_multi(
                                tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype,
                                vector_width=reduce_vector_width, reduce_unroll=ru)
                            self._reduce_kernel_coupled_multi_cache[key] = kernel
                    else:
                        kernel = None
                    if dHdp_reduce_nodes:
                        key = (tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, reduce_vector_width, ru)
                        kernel_hdp = self._reduce_kernel_coupled_multi_cache.get(key)
                        if kernel_hdp is None:
                            kernel_hdp = self._build_coupled_reduce_kernel_multi(
                                tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype,
                                vector_width=reduce_vector_width, reduce_unroll=ru)
                            self._reduce_kernel_coupled_multi_cache[key] = kernel_hdp
                    else:
                        kernel_hdp = None
                    start = time.perf_counter()
                    if kernel is not None:
                        tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdq_reduce_nodes]
                        Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel)[2:])
                    if kernel_hdp is not None:
                        tmp_outs = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdp_reduce_nodes]
                        Tensor.realize(*Tensor.custom_kernel(q, p, *tmp_outs, fxn=kernel_hdp)[2:])
                    elapsed = time.perf_counter() - start
                    if elapsed < best_time:
                        best_time = elapsed
                        best_ru = ru
                self._reduce_kernel_coupled_unroll_tune_cache[tune_key] = best_ru
                reduce_unroll = best_ru
        kernel_reduce_dHdq = None
        if dHdq_reduce_nodes:
            key = (tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype, reduce_vector_width, reduce_unroll)
            kernel_reduce_dHdq = self._reduce_kernel_coupled_multi_cache.get(key)
            if kernel_reduce_dHdq is None:
                kernel_reduce_dHdq = self._build_coupled_reduce_kernel_multi(
                    tuple(dHdq_reduce_nodes), q.device, q.shape, q.dtype,
                    vector_width=reduce_vector_width, reduce_unroll=reduce_unroll)
                self._reduce_kernel_coupled_multi_cache[key] = kernel_reduce_dHdq

        kernel_reduce_dHdp = None
        if dHdp_reduce_nodes:
            key = (tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype, reduce_vector_width, reduce_unroll)
            kernel_reduce_dHdp = self._reduce_kernel_coupled_multi_cache.get(key)
            if kernel_reduce_dHdp is None:
                kernel_reduce_dHdp = self._build_coupled_reduce_kernel_multi(
                    tuple(dHdp_reduce_nodes), q.device, q.shape, q.dtype,
                    vector_width=reduce_vector_width, reduce_unroll=reduce_unroll)
                self._reduce_kernel_coupled_multi_cache[key] = kernel_reduce_dHdp

        kernel_qp_reduce = None
        if use_fused_qp:
            dHdq_elem_use = dHdq_elem_uop
            dHdp_elem_use = dHdp_elem_uop
            key_qp = (self.H, q.device, q.shape, q.dtype, "qp_new_reduce", dt, unroll_steps,
                      vector_width, reduce_unroll, tuple(dHdq_reduce_nodes), tuple(dHdp_reduce_nodes))
            kernel_qp_reduce = self._elem_kernel_coupled_cache.get(key_qp)
            if kernel_qp_reduce is None:
                kernel_qp_reduce = self._build_coupled_kernel_qp_new_with_reduce(
                    dt, q.device, q.shape, q.dtype, dHdq_elem_use, dHdp_elem_use,
                    dHdq_reduce_nodes, dHdp_reduce_nodes, dHdq_placeholders, dHdp_placeholders,
                    unroll_steps=unroll_steps, vector_width=vector_width, reduce_unroll=reduce_unroll)
                self._elem_kernel_coupled_cache[key_qp] = kernel_qp_reduce
        else:
            kernel_qp_new = None
            kernel_p_half = None
            kernel_qp_update = None
            kernel_qp_update_qp = None
            if not dHdq_reduce_nodes and not dHdp_reduce_nodes:
                key_half = (self.H, q.device, q.shape, q.dtype, "p_half", dt)
                kernel_p_half = self._elem_kernel_coupled_cache.get(key_half)
                if kernel_p_half is None:
                    kernel_p_half = self._build_coupled_elem_kernel(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdq_placeholders, "p_half")
                    self._elem_kernel_coupled_cache[key_half] = kernel_p_half
                key_qp_update = (self.H, q.device, q.shape, q.dtype, "qp_update", dt)
                kernel_qp_update = self._elem_kernel_coupled_cache.get(key_qp_update)
                if kernel_qp_update is None:
                    kernel_qp_update = self._build_coupled_elem_kernel_qp_update(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                        dHdq_placeholders, dHdp_placeholders)
                    self._elem_kernel_coupled_cache[key_qp_update] = kernel_qp_update
                key_qp_update_qp = (self.H, q.device, q.shape, q.dtype, "qp_update_qp", dt)
                kernel_qp_update_qp = self._elem_kernel_coupled_cache.get(key_qp_update_qp)
                if kernel_qp_update_qp is None:
                    kernel_qp_update_qp = self._build_coupled_elem_kernel_qp_update_from_qp(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                        dHdq_placeholders, dHdp_placeholders)
                    self._elem_kernel_coupled_cache[key_qp_update_qp] = kernel_qp_update_qp
            else:
                key_half = (self.H, q.device, q.shape, q.dtype, "p_half", dt)
                kernel_p_half = self._elem_kernel_coupled_cache.get(key_half)
                if kernel_p_half is None:
                    kernel_p_half = self._build_coupled_elem_kernel(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdq_placeholders, "p_half")
                    self._elem_kernel_coupled_cache[key_half] = kernel_p_half

                key_qp_update = (self.H, q.device, q.shape, q.dtype, "qp_update", dt)
                kernel_qp_update = self._elem_kernel_coupled_cache.get(key_qp_update)
                if kernel_qp_update is None:
                    kernel_qp_update = self._build_coupled_elem_kernel_qp_update(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                        dHdq_placeholders, dHdp_placeholders)
                    self._elem_kernel_coupled_cache[key_qp_update] = kernel_qp_update
                key_qp_update_qp = (self.H, q.device, q.shape, q.dtype, "qp_update_qp", dt)
                kernel_qp_update_qp = self._elem_kernel_coupled_cache.get(key_qp_update_qp)
                if kernel_qp_update_qp is None:
                    kernel_qp_update_qp = self._build_coupled_elem_kernel_qp_update_from_qp(
                        dt, q.device, q.shape, q.dtype, dHdq_elem_uop, dHdp_elem_uop,
                        dHdq_placeholders, dHdp_placeholders)
                    self._elem_kernel_coupled_cache[key_qp_update_qp] = kernel_qp_update_qp

            kernel_reduce_qnew = None
            if dHdq_reduce_nodes:
                key_qnew = (tuple(dHdq_reduce_nodes), self.H, q.device, q.shape, q.dtype,
                            dt, reduce_vector_width, reduce_unroll)
                kernel_reduce_qnew = self._reduce_kernel_coupled_qnew_cache.get(key_qnew)
                if kernel_reduce_qnew is None:
                    kernel_reduce_qnew = self._build_coupled_reduce_kernel_qnew_multi(
                        dt, q.device, q.shape, q.dtype, dHdp_elem_uop, tuple(dHdq_reduce_nodes),
                        dHdp_placeholders, vector_width=reduce_vector_width, reduce_unroll=reduce_unroll)
                    self._reduce_kernel_coupled_qnew_cache[key_qnew] = kernel_reduce_qnew

        q_cur = q
        p_cur = p
        buf_key = (q.device, q.dtype, q.shape)
        q_buf, p_buf = self._scan_tmp_buf_cache.get(buf_key, (None, None))
        if q_buf is None or p_buf is None or q_buf.shape != q.shape or p_buf.shape != p.shape:
            q_buf = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_buf = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            self._scan_tmp_buf_cache[buf_key] = (q_buf, p_buf)

        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and q.shape and q.shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        if use_fused_qp:
            for _ in range(steps // unroll_steps):
                out = Tensor.custom_kernel(q_cur, p_cur, q_buf, p_buf, fxn=kernel_qp_reduce)
                q_new, p_new = out[2], out[3]
                Tensor.realize(q_new, p_new)
                q_cur, p_cur = q_new, p_new
        else:
            acc_bufs_dHdq = []
            if dHdq_reduce_nodes:
                key = (q.device, q.dtype, len(dHdq_reduce_nodes))
                acc_bufs_dHdq = self._reduce_acc_buf_cache_dHdq.get(key, [])
                if len(acc_bufs_dHdq) != len(dHdq_reduce_nodes):
                    acc_bufs_dHdq = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdq_reduce_nodes]
                    self._reduce_acc_buf_cache_dHdq[key] = acc_bufs_dHdq
            acc_bufs_dHdp = []
            if dHdp_reduce_nodes:
                key = (q.device, q.dtype, len(dHdp_reduce_nodes))
                acc_bufs_dHdp = self._reduce_acc_buf_cache_dHdp.get(key, [])
                if len(acc_bufs_dHdp) != len(dHdp_reduce_nodes):
                    acc_bufs_dHdp = [Tensor.empty(1, device=q.device, dtype=q.dtype) for _ in dHdp_reduce_nodes]
                    self._reduce_acc_buf_cache_dHdp[key] = acc_bufs_dHdp
            def run_split_step(q_cur: Tensor, p_cur: Tensor) -> tuple[Tensor, Tensor]:
                accs_dHdq = []
                if kernel_reduce_dHdq is not None:
                    out = Tensor.custom_kernel(q_cur, p_cur, *acc_bufs_dHdq, fxn=kernel_reduce_dHdq)
                    accs_dHdq = list(out[2:2+len(acc_bufs_dHdq)])
                p_half = Tensor.custom_kernel(q_cur, p_cur, *accs_dHdq, p_buf, fxn=kernel_p_half)[-1]
                accs_dHdp = []
                if kernel_reduce_dHdp is not None:
                    out = Tensor.custom_kernel(q_cur, p_half, *acc_bufs_dHdp, fxn=kernel_reduce_dHdp)
                    accs_dHdp = list(out[2:2+len(acc_bufs_dHdp)])
                accs_dHdq_2 = []
                if kernel_reduce_qnew is not None:
                    red_out = Tensor.custom_kernel(q_cur, p_half, *accs_dHdp, *acc_bufs_dHdq, fxn=kernel_reduce_qnew)
                    accs_dHdq_2 = list(red_out[2:2+len(acc_bufs_dHdq)])
                if getenv("TINYGRAD_COUPLED_QP_UPDATE_QP", 0):
                    q_new, p_new = Tensor.custom_kernel(
                        q_cur, p_cur, *accs_dHdq, *accs_dHdp, *accs_dHdq_2, q_buf, p_buf,
                        fxn=kernel_qp_update_qp)[-2:]
                else:
                    q_new, p_new = Tensor.custom_kernel(
                        q_cur, p_half, *accs_dHdp, *accs_dHdq_2, q_buf, p_buf, fxn=kernel_qp_update)[-2:]
                return q_new, p_new
            for _ in range(steps // unroll_steps):
                for _ in range(unroll_steps):
                    q_new, p_new = run_split_step(q_cur, p_cur)
                    Tensor.realize(q_new, p_new)
                    q_cur, p_cur = q_new, p_new

        return q_cur, p_cur

    def _build_coupled_update_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                     vector_width: int) -> Callable:
        def kernel(q: UOp, p: UOp, dHdq: UOp, dHdp: UOp, q_out: UOp, p_half_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
            ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            base = ranges[-1] * UOp.const(dtypes.index, vector_width)
            q_ptr = q.index(*ranges[:-1], base, ptr=True)
            p_ptr = p.index(*ranges[:-1], base, ptr=True)
            dHdq_ptr = dHdq.index(*ranges[:-1], base, ptr=True)
            dHdp_ptr = dHdp.index(*ranges[:-1], base, ptr=True)
            vec_dtype = q.dtype.base.vec(vector_width)
            q_vec = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
            p_vec = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            dHdq_vec = dHdq_ptr.cast(vec_dtype.ptr(size=dHdq_ptr.dtype.size, addrspace=dHdq_ptr.dtype.addrspace)).load()
            dHdp_vec = dHdp_ptr.cast(vec_dtype.ptr(size=dHdp_ptr.dtype.size, addrspace=dHdp_ptr.dtype.addrspace)).load()
            dt_uop = UOp.const(vec_dtype, dt)
            half_dt = UOp.const(vec_dtype, 0.5*dt)
            p_half = p_vec - half_dt * dHdq_vec
            q_new = q_vec + dt_uop * dHdp_vec
            q_out_ptr = q_out.index(*ranges[:-1], base, ptr=True)
            p_half_ptr = p_half_out.index(*ranges[:-1], base, ptr=True)
            store_q = q_out_ptr.cast(vec_dtype.ptr(size=q_out_ptr.dtype.size, addrspace=q_out_ptr.dtype.addrspace)).store(q_new)
            store_p_half = p_half_ptr.cast(vec_dtype.ptr(size=p_half_ptr.dtype.size, addrspace=p_half_ptr.dtype.addrspace)).store(p_half)
            return UOp.group(store_q, store_p_half).end(*ranges).sink(
                arg=KernelInfo(name="coupled_update_qp", opts_to_apply=()))

        return kernel

    def _build_coupled_update_p_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                       vector_width: int) -> Callable:
        def kernel(p_half: UOp, dHdq: UOp, p_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
            ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            base = ranges[-1] * UOp.const(dtypes.index, vector_width)
            p_half_ptr = p_half.index(*ranges[:-1], base, ptr=True)
            dHdq_ptr = dHdq.index(*ranges[:-1], base, ptr=True)
            vec_dtype = p_half.dtype.base.vec(vector_width)
            p_half_vec = p_half_ptr.cast(vec_dtype.ptr(size=p_half_ptr.dtype.size, addrspace=p_half_ptr.dtype.addrspace)).load()
            dHdq_vec = dHdq_ptr.cast(vec_dtype.ptr(size=dHdq_ptr.dtype.size, addrspace=dHdq_ptr.dtype.addrspace)).load()
            half_dt = UOp.const(vec_dtype, 0.5*dt)
            p_new = p_half_vec - half_dt * dHdq_vec
            p_out_ptr = p_out.index(*ranges[:-1], base, ptr=True)
            store_p = p_out_ptr.cast(vec_dtype.ptr(size=p_out_ptr.dtype.size, addrspace=p_out_ptr.dtype.addrspace)).store(p_new)
            return store_p.end(*ranges).sink(arg=KernelInfo(name="coupled_update_p", opts_to_apply=()))

        return kernel

    def _build_coupled_update_kernel_fused(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                           vector_width: int) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        const_vec = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="x"), _const_to_vec),
        ])
        drop_vector_expand = PatternMatcher([
            (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_vector_expand),
        ])
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_out: UOp, p_half_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
            ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            base = ranges[-1] * UOp.const(dtypes.index, vector_width)
            q_ptr = q.index(*ranges[:-1], base, ptr=True)
            p_ptr = p.index(*ranges[:-1], base, ptr=True)
            vec_dtype = q.dtype.base.vec(vector_width)
            q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
            p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            dHdq = dHdq_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdp = dHdp_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdq = graph_rewrite(dHdq, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
            dHdp = graph_rewrite(dHdp, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
            dHdq = graph_rewrite(dHdq, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
            dHdp = graph_rewrite(dHdp, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
            dHdq = graph_rewrite(dHdq, strip_device_consts, name="strip_device_consts")
            dHdp = graph_rewrite(dHdp, strip_device_consts, name="strip_device_consts")
            if dHdq.dtype.count == 1:
                dHdq = dHdq.broadcast(vector_width)
            if dHdp.dtype.count == 1:
                dHdp = dHdp.broadcast(vector_width)
            dt_uop = UOp.const(vec_dtype, dt)
            half_dt = UOp.const(vec_dtype, 0.5*dt)
            p_half = p_val - half_dt * dHdq
            q_new = q_val + dt_uop * dHdp
            q_out_ptr = q_out.index(*ranges[:-1], base, ptr=True)
            p_half_ptr = p_half_out.index(*ranges[:-1], base, ptr=True)
            store_q = q_out_ptr.cast(vec_dtype.ptr(size=q_out_ptr.dtype.size, addrspace=q_out_ptr.dtype.addrspace)).store(q_new)
            store_p_half = p_half_ptr.cast(vec_dtype.ptr(size=p_half_ptr.dtype.size, addrspace=p_half_ptr.dtype.addrspace)).store(p_half)
            return UOp.group(store_q, store_p_half).end(*ranges).sink(
                arg=KernelInfo(name="coupled_update_qp_fused", opts_to_apply=()))

        return kernel

    def _build_coupled_update_p_kernel_fused(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                             vector_width: int) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, _ = self._get_coupled_grad_uops(device, shape, dtype)
        const_vec = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="x"), _const_to_vec),
        ])
        drop_vector_expand = PatternMatcher([
            (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_vector_expand),
        ])
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p_half: UOp, p_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
            ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
            base = ranges[-1] * UOp.const(dtypes.index, vector_width)
            q_ptr = q.index(*ranges[:-1], base, ptr=True)
            p_ptr = p_half.index(*ranges[:-1], base, ptr=True)
            vec_dtype = p_half.dtype.base.vec(vector_width)
            q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
            p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            dHdq = dHdq_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdq = graph_rewrite(dHdq, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
            dHdq = graph_rewrite(dHdq, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
            dHdq = graph_rewrite(dHdq, strip_device_consts, name="strip_device_consts")
            if dHdq.dtype.count == 1:
                dHdq = dHdq.broadcast(vector_width)
            half_dt = UOp.const(vec_dtype, 0.5*dt)
            p_new = p_val - half_dt * dHdq
            p_out_ptr = p_out.index(*ranges[:-1], base, ptr=True)
            store_p = p_out_ptr.cast(vec_dtype.ptr(size=p_out_ptr.dtype.size, addrspace=p_out_ptr.dtype.addrspace)).store(p_new)
            return store_p.end(*ranges).sink(arg=KernelInfo(name="coupled_update_p_fused", opts_to_apply=()))

        return kernel

    def _build_implicit_mid_init_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_mid_out: UOp, p_mid_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            dHdq = dHdq_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdp = dHdp_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
            dHdq = graph_rewrite(dHdq, strip_device_consts, name="strip_device_consts")
            dHdp = graph_rewrite(dHdp, strip_device_consts, name="strip_device_consts")
            half_dt = UOp.const(dHdq.dtype, 0.5 * dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_mid = q_val + half_dt * dHdp
            p_mid = p_val - half_dt * dHdq
            store_q = q_mid_out.index(*ranges, ptr=True).store(q_mid)
            store_p = p_mid_out.index(*ranges, ptr=True).store(p_mid)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="implicit_mid_init", opts_to_apply=()))

        return kernel

    def _build_implicit_mid_iter_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_mid: UOp, p_mid: UOp, q_mid_out: UOp, p_mid_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            q_mid_val = q_mid.vindex(*ranges)
            p_mid_val = p_mid.vindex(*ranges)
            dHdq_mid = dHdq_uop.substitute({q_sym_uop: q_mid_val, p_sym_uop: p_mid_val})
            dHdp_mid = dHdp_uop.substitute({q_sym_uop: q_mid_val, p_sym_uop: p_mid_val})
            dHdq_mid = graph_rewrite(dHdq_mid, strip_device_consts, name="strip_device_consts")
            dHdp_mid = graph_rewrite(dHdp_mid, strip_device_consts, name="strip_device_consts")
            half_dt = UOp.const(dHdq_mid.dtype, 0.5 * dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_next = q_val + half_dt * dHdp_mid
            p_next = p_val - half_dt * dHdq_mid
            store_q = q_mid_out.index(*ranges, ptr=True).store(q_next)
            store_p = p_mid_out.index(*ranges, ptr=True).store(p_next)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="implicit_mid_iter", opts_to_apply=()))

        return kernel

    def _build_implicit_mid_final_kernel(self, dt: float, device: str, shape: tuple[int, ...], dtype) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_mid: UOp, p_mid: UOp, q_out: UOp, p_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            q_mid_val = q_mid.vindex(*ranges)
            p_mid_val = p_mid.vindex(*ranges)
            dHdq_mid = dHdq_uop.substitute({q_sym_uop: q_mid_val, p_sym_uop: p_mid_val})
            dHdp_mid = dHdp_uop.substitute({q_sym_uop: q_mid_val, p_sym_uop: p_mid_val})
            dHdq_mid = graph_rewrite(dHdq_mid, strip_device_consts, name="strip_device_consts")
            dHdp_mid = graph_rewrite(dHdp_mid, strip_device_consts, name="strip_device_consts")
            dt_uop = UOp.const(dHdq_mid.dtype, dt, shape=tuple(r.vmax + 1 for r in ranges))
            q_next = q_val + dt_uop * dHdp_mid
            p_next = p_val - dt_uop * dHdq_mid
            store_q = q_out.index(*ranges, ptr=True).store(q_next)
            store_p = p_out.index(*ranges, ptr=True).store(p_next)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="implicit_mid_final", opts_to_apply=()))

        return kernel

    def _build_implicit_mid_full_kernel(self, dt: float, iters: int, device: str, shape: tuple[int, ...], dtype,
                                        vector_width: int) -> Callable:
        if iters < 1:
            raise ValueError("iters must be >= 1")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        const_vec = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="x"), _const_to_vec),
        ])
        drop_vector_expand = PatternMatcher([
            (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_vector_expand),
        ])
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_out: UOp, p_out: UOp) -> UOp:
            use_vec = vector_width > 1
            if use_vec:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape[:-1])]
                ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                vec_dtype = q.dtype.base.vec(vector_width)
                q_ptr = q.index(*ranges[:-1], base, ptr=True)
                p_ptr = p.index(*ranges[:-1], base, ptr=True)
                q_val = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                p_val = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
            else:
                ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
                q_val = q.vindex(*ranges)
                p_val = p.vindex(*ranges)

            def grad_uop(q_uop: UOp, p_uop: UOp) -> tuple[UOp, UOp]:
                dHdq = dHdq_uop.substitute({q_sym_uop: q_uop, p_sym_uop: p_uop})
                dHdp = dHdp_uop.substitute({q_sym_uop: q_uop, p_sym_uop: p_uop})
                if use_vec:
                    dHdq = graph_rewrite(dHdq, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
                    dHdp = graph_rewrite(dHdp, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
                    dHdq = graph_rewrite(dHdq, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
                    dHdp = graph_rewrite(dHdp, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
                dHdq = graph_rewrite(dHdq, strip_device_consts, name="strip_device_consts")
                dHdp = graph_rewrite(dHdp, strip_device_consts, name="strip_device_consts")
                if use_vec:
                    if dHdq.dtype.count == 1: dHdq = dHdq.broadcast(vector_width)
                    if dHdp.dtype.count == 1: dHdp = dHdp.broadcast(vector_width)
                return dHdq, dHdp

            shape_vals = tuple(r.vmax + 1 for r in ranges)
            dHdq_mid, dHdp_mid = grad_uop(q_val, p_val)
            half_dt = UOp.const(dHdq_mid.dtype, 0.5 * dt, shape=None if use_vec else shape_vals)
            q_mid = q_val + half_dt * dHdp_mid
            p_mid = p_val - half_dt * dHdq_mid
            for _ in range(iters - 1):
                dHdq_mid, dHdp_mid = grad_uop(q_mid, p_mid)
                q_mid = q_val + half_dt * dHdp_mid
                p_mid = p_val - half_dt * dHdq_mid
            dHdq_mid, dHdp_mid = grad_uop(q_mid, p_mid)
            dt_uop = UOp.const(dHdq_mid.dtype, dt, shape=None if use_vec else shape_vals)
            q_next = q_val + dt_uop * dHdp_mid
            p_next = p_val - dt_uop * dHdq_mid
            if use_vec:
                q_out_ptr = q_out.index(*ranges[:-1], base, ptr=True)
                p_out_ptr = p_out.index(*ranges[:-1], base, ptr=True)
                store_q = q_out_ptr.cast(vec_dtype.ptr(size=q_out_ptr.dtype.size, addrspace=q_out_ptr.dtype.addrspace)).store(q_next)
                store_p = p_out_ptr.cast(vec_dtype.ptr(size=p_out_ptr.dtype.size, addrspace=p_out_ptr.dtype.addrspace)).store(p_next)
            else:
                store_q = q_out.index(*ranges, ptr=True).store(q_next)
                store_p = p_out.index(*ranges, ptr=True).store(p_next)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="implicit_mid_full", opts_to_apply=()))

        return kernel

    def _evolve_coupled_two_phase_vec(self, q: Tensor, p: Tensor, dt: float, steps: int,
                                      vector_width: int) -> tuple[Tensor, Tensor]:
        if vector_width < 2:
            raise ValueError("vector_width must be >= 2 for two-phase path")
        if q.shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
        vec_grad_key = (q.device, q.shape, q.dtype, vector_width)
        vec_grad_ok = self._grad_kernel_coupled_vec_ok.get(vec_grad_key, None)
        use_vec_grad = vec_grad_ok if vec_grad_ok is not None else not self._coupled_has_reduce(q.device, q.shape, q.dtype)
        if use_vec_grad:
            key = vec_grad_key
            grad_kernel = self._grad_kernel_coupled_vec_cache.get(key)
            if grad_kernel is None:
                grad_kernel = self._build_coupled_grad_kernel_vec(q.device, q.shape, q.dtype, vector_width)
                self._grad_kernel_coupled_vec_cache[key] = grad_kernel
            if vec_grad_ok is None:
                try:
                    q_probe = q.detach().clone().realize()
                    p_probe = p.detach().clone().realize()
                    dHdq_probe = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
                    dHdp_probe = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                    out = Tensor.custom_kernel(q_probe, p_probe, dHdq_probe, dHdp_probe, fxn=grad_kernel)
                    Tensor.realize(out[2], out[3])
                    self._grad_kernel_coupled_vec_ok[vec_grad_key] = True
                except Exception:
                    self._grad_kernel_coupled_vec_ok[vec_grad_key] = False
                    use_vec_grad = False
        if not use_vec_grad:
            key = (q.device, q.shape, q.dtype)
            grad_kernel = self._grad_kernel_coupled_cache.get(key)
            if grad_kernel is None:
                grad_kernel = self._build_coupled_grad_kernel(q.device, q.shape, q.dtype)
                self._grad_kernel_coupled_cache[key] = grad_kernel
        key_u = (dt, q.device, q.shape, q.dtype, vector_width)
        update_kernel = self._update_kernel_coupled_cache.get(key_u)
        if update_kernel is None:
            update_kernel = self._build_coupled_update_kernel(dt, q.device, q.shape, q.dtype, vector_width)
            self._update_kernel_coupled_cache[key_u] = update_kernel
        update_p_kernel = self._update_kernel_coupled_p_cache.get(key_u)
        if update_p_kernel is None:
            update_p_kernel = self._build_coupled_update_p_kernel(dt, q.device, q.shape, q.dtype, vector_width)
            self._update_kernel_coupled_p_cache[key_u] = update_p_kernel

        q_buf_a = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        q_buf_b = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_half_a = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        p_half_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        p_buf_a = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        p_buf_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        key_u = (dt, q.device, q.shape, q.dtype, vector_width)
        use_fused = not self._coupled_has_reduce(q.device, q.shape, q.dtype)
        fused_ok = self._update_kernel_coupled_fused_ok.get(key_u, None)
        if fused_ok is False:
            use_fused = False
        dHdq_a = dHdp_a = dHdq_b = dHdp_b = None
        if use_fused:
            fused_update = self._update_kernel_coupled_fused_cache.get(key_u)
            if fused_update is None:
                fused_update = self._build_coupled_update_kernel_fused(dt, q.device, q.shape, q.dtype, vector_width)
                self._update_kernel_coupled_fused_cache[key_u] = fused_update
            fused_update_p = self._update_kernel_coupled_p_fused_cache.get(key_u)
            if fused_update_p is None:
                fused_update_p = self._build_coupled_update_p_kernel_fused(dt, q.device, q.shape, q.dtype, vector_width)
                self._update_kernel_coupled_p_fused_cache[key_u] = fused_update_p
            if fused_ok is None:
                try:
                    q_probe = q.detach().clone().realize()
                    p_probe = p.detach().clone().realize()
                    q_tmp = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
                    p_half_tmp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                    p_tmp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
                    out = Tensor.custom_kernel(q_probe, p_probe, q_tmp, p_half_tmp, fxn=fused_update)
                    Tensor.realize(out[2], out[3])
                    out = Tensor.custom_kernel(out[2], out[3], p_tmp, fxn=fused_update_p)
                    Tensor.realize(out[2])
                    self._update_kernel_coupled_fused_ok[key_u] = True
                except Exception:
                    self._update_kernel_coupled_fused_ok[key_u] = False
                    use_fused = False
        if not use_fused:
            dHdq_a = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            dHdp_a = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            dHdq_b = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            dHdp_b = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        q_cur, p_cur = q, p
        flip = False
        for _ in range(steps):
            if flip:
                q_out, p_half_out, p_out = q_buf_b, p_half_b, p_buf_b
            else:
                q_out, p_half_out, p_out = q_buf_a, p_half_a, p_buf_a
            if use_fused:
                out = Tensor.custom_kernel(q_cur, p_cur, q_out, p_half_out, fxn=fused_update)
                q_new, p_half = out[2], out[3]
                Tensor.realize(q_new, p_half)
                out = Tensor.custom_kernel(q_new, p_half, p_out, fxn=fused_update_p)
                p_new = out[2]
                Tensor.realize(p_new)
            else:
                if flip:
                    dHdq_out, dHdp_out = dHdq_b, dHdp_b
                    dHdq2_out, dHdp2_out = dHdq_a, dHdp_a
                else:
                    dHdq_out, dHdp_out = dHdq_a, dHdp_a
                    dHdq2_out, dHdp2_out = dHdq_b, dHdp_b
                out = Tensor.custom_kernel(q_cur, p_cur, dHdq_out, dHdp_out, fxn=grad_kernel)
                dHdq, dHdp = out[2], out[3]
                out = Tensor.custom_kernel(q_cur, p_cur, dHdq, dHdp, q_out, p_half_out, fxn=update_kernel)
                q_new, p_half = out[4], out[5]
                Tensor.realize(q_new, p_half)
                out = Tensor.custom_kernel(q_new, p_half, dHdq2_out, dHdp2_out, fxn=grad_kernel)
                dHdq2 = out[2]
                out = Tensor.custom_kernel(p_half, dHdq2, p_out, fxn=update_p_kernel)
                p_new = out[2]
                Tensor.realize(p_new)
            q_cur, p_cur = q_new, p_new
            flip = not flip
        return q_cur, p_cur


    def _build_leapfrog_scan_kernel(self, dt: float, steps: int, unroll_steps: int, vector_width: int,
                                    device: str, shape: tuple[int, ...], dtype, ho_closed_form: bool = False) -> Callable:
        q_sym = Tensor.empty((), device=device, dtype=dtype, requires_grad=True)
        p_sym = Tensor.empty((), device=device, dtype=dtype, requires_grad=True)
        H_sym = self.H(q_sym, p_sym)
        dHdq_sym, dHdp_sym = H_sym.gradient(q_sym, p_sym)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])
        dHdq_uop = graph_rewrite(dHdq_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdq_uop = graph_rewrite(dHdq_uop, symbolic, name="symbolic_dHdq")
        dHdp_uop = graph_rewrite(dHdp_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdp_uop = graph_rewrite(dHdp_uop, symbolic, name="symbolic_dHdp")

        ho_coeffs = self._get_ho_coeffs(device, dtype)
        ho_dHdq, ho_dHdp, ho_c, ho_d = ho_coeffs if ho_coeffs is not None else (None, None, None, None)
        ho_fast_path = ho_coeffs is not None

        def grad_uop(q_uop: UOp, p_uop: UOp) -> tuple[UOp, UOp]:
            sub = {q_sym.uop: q_uop, p_sym.uop: p_uop}
            return dHdq_uop.substitute(sub), dHdp_uop.substitute(sub)

        def kernel(q: UOp, p: UOp) -> UOp:
            tile_steps = steps // unroll_steps
            step = UOp.range(tile_steps, 0)
            q_step = q.after(step)
            p_step = p.after(step)

            q_flat = q_step.flatten()
            p_flat = p_step.flatten()
            idx = UOp.range(q_flat.size, 1) if vector_width == 1 else None
            if vector_width == 1:
                q_elem = q_flat.after(step)[idx]
                p_elem = p_flat.after(step)[idx]
            else:
                outer = q_flat.size // vector_width
                oidx = UOp.range(outer, 1)
                base = oidx * UOp.const(dtypes.index, vector_width)
                q_ptr = q_flat.after(step).index(base, ptr=True)
                p_ptr = p_flat.after(step).index(base, ptr=True)
                q_vec_ptr = q_ptr.cast(q_flat.dtype.base.vec(vector_width).ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                p_vec_ptr = p_ptr.cast(p_flat.dtype.base.vec(vector_width).ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                q_elem = q_vec_ptr.load()
                p_elem = p_vec_ptr.load()

            if ho_fast_path:
                dt_uop = q_elem.const_like(dt)
                half_dt = q_elem.const_like(0.5*dt)
                a = q_elem.const_like(ho_dHdq)
                b = q_elem.const_like(ho_dHdp)
                c = q_elem.const_like(ho_c)
                d = q_elem.const_like(ho_d)
                ab_one = (ho_dHdq == 1.0 and ho_dHdp == 1.0)
                trig_key = (dt, unroll_steps, device, dtype, ho_dHdq, ho_dHdp)
                trig = self._ho_trig_cache.get(trig_key)
                if trig is None:
                    if ho_closed_form and ho_dHdq != 0.0 and ho_dHdp != 0.0:
                        w2 = ho_dHdq * ho_dHdp
                        if w2 > 0.0 and math.isfinite(w2):
                            w = math.sqrt(w2)
                            theta = dt * unroll_steps * w
                            if math.isfinite(theta):
                                sin_t = math.sin(theta)
                                cos_t = math.cos(theta)
                                a_over_w = ho_dHdq / w
                                b_over_w = ho_dHdp / w
                                if all(math.isfinite(x) for x in (sin_t, cos_t, a_over_w, b_over_w)):
                                    trig = (sin_t, cos_t, a_over_w, b_over_w)
                                    self._ho_trig_cache[trig_key] = trig
                use_rotate = trig is not None and ho_closed_form and ho_dHdq != 0.0 and ho_dHdp != 0.0
                if use_rotate:
                    sin_t = q_elem.const_like(trig[0])
                    cos_t = q_elem.const_like(trig[1])
                    a_over_w = q_elem.const_like(trig[2])
                    b_over_w = q_elem.const_like(trig[3])
                    shift_key = (device, dtype, ho_dHdq, ho_dHdp, ho_c, ho_d)
                    shift = self._ho_shift_cache.get(shift_key)
                    if shift is None:
                        if ho_dHdq != 0.0 and ho_dHdp != 0.0:
                            q_off = ho_c / ho_dHdq
                            p_off = ho_d / ho_dHdp
                            if math.isfinite(q_off) and math.isfinite(p_off):
                                shift = (q_off, p_off)
                                self._ho_shift_cache[shift_key] = shift
                    if shift is None:
                        q_off = c / a
                        p_off = d / b
                    else:
                        q_off = q_elem.const_like(shift[0])
                        p_off = q_elem.const_like(shift[1])
                    q_shift = q_elem + q_off
                    p_shift = p_elem + p_off
                    q_next = q_shift * cos_t + (b_over_w * p_shift) * sin_t
                    p_next = p_shift * cos_t - (a_over_w * q_shift) * sin_t
                    q_elem = q_next - q_off
                    p_elem = p_next - p_off
                else:
                    for _ in range(unroll_steps):
                        if ab_one:
                            p_elem = p_elem - half_dt * (q_elem + c)
                            q_elem = q_elem + dt_uop * (p_elem + d)
                            p_elem = p_elem - half_dt * (q_elem + c)
                        else:
                            p_elem = p_elem - half_dt * (a * q_elem + c)
                            q_elem = q_elem + dt_uop * (b * p_elem + d)
                            p_elem = p_elem - half_dt * (a * q_elem + c)
            else:
                dt_uop = q_elem.const_like(dt)
                half_dt = q_elem.const_like(0.5*dt)
                for _ in range(unroll_steps):
                    dHdq_1, _ = grad_uop(q_elem, p_elem)
                    p_half = p_elem - half_dt * dHdq_1
                    _, dHdp = grad_uop(q_elem, p_half)
                    q_elem = q_elem + dt_uop * dHdp
                    dHdq_2, _ = grad_uop(q_elem, p_half)
                    p_elem = p_half - half_dt * dHdq_2

            if vector_width == 1:
                store_q = q_flat.after(step)[idx].store(q_elem)
                store_p = p_flat.after(step)[idx].store(p_elem)
                return UOp.group(store_q, store_p).end(idx, step).sink(arg=KernelInfo(name=f"leapfrog_scan_{steps}", opts_to_apply=()))

            store_q = q_vec_ptr.store(q_elem)
            store_p = p_vec_ptr.store(p_elem)
            return UOp.group(store_q, store_p).end(oidx, step).sink(arg=KernelInfo(name=f"leapfrog_scan_{steps}", opts_to_apply=()))

        return kernel

    def _get_coupled_grad_uops(self, device: str, shape: tuple[int, ...], dtype) -> tuple[UOp, UOp, UOp, UOp]:
        key = (self.H, device, shape, dtype)
        cached = self._grad_uop_cache.get(key)
        if cached is not None:
            return cached
        q_sym = Tensor.empty(*shape, device=device, dtype=dtype, requires_grad=True)
        p_sym = Tensor.empty(*shape, device=device, dtype=dtype, requires_grad=True)
        H_sym = self.H(q_sym, p_sym)
        dHdq_sym, dHdp_sym = H_sym.gradient(q_sym, p_sym)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])
        dHdq_uop = graph_rewrite(dHdq_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdq_uop = graph_rewrite(dHdq_uop, symbolic, name="symbolic_dHdq")
        dHdp_uop = graph_rewrite(dHdp_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdp_uop = graph_rewrite(dHdp_uop, symbolic, name="symbolic_dHdp")
        self._grad_uop_cache[key] = (q_sym.uop, p_sym.uop, dHdq_uop, dHdp_uop)
        return self._grad_uop_cache[key]

    def _get_ho_coeffs(self, device: str, dtype) -> tuple[float, float, float, float]|None:
        key = (self.H, device, dtype)
        cached = self._ho_coeff_cache.get(key, None)
        if cached is not None:
            return cached
        q_sym = Tensor.empty((), device=device, dtype=dtype, requires_grad=True)
        p_sym = Tensor.empty((), device=device, dtype=dtype, requires_grad=True)
        H_sym = self.H(q_sym, p_sym)
        dHdq_sym, dHdp_sym = H_sym.gradient(q_sym, p_sym)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])
        dHdq_uop = graph_rewrite(dHdq_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdq_uop = graph_rewrite(dHdq_uop, symbolic, name="symbolic_dHdq")
        dHdp_uop = graph_rewrite(dHdp_sym.uop, strip_device_consts, name="strip_device_consts")
        dHdp_uop = graph_rewrite(dHdp_uop, symbolic, name="symbolic_dHdp")

        def _const_val(uop: UOp) -> float|None:
            if uop.op is Ops.CONST: return uop.arg
            if uop.op is Ops.VCONST:
                try:
                    vals = list(uop.arg)
                    if vals and all(v == vals[0] for v in vals):
                        return vals[0]
                except Exception:
                    return None
            if uop.op is Ops.CAST: return _const_val(uop.src[0])
            return None

        def _sym_or_cast(uop: UOp, sym: UOp) -> bool:
            if uop is sym: return True
            return uop.op is Ops.CAST and uop.src[0] is sym

        def _extract_affine(uop: UOp, sym: UOp) -> tuple[float, float]|None:
            if _sym_or_cast(uop, sym): return (1.0, 0.0)
            if uop.op is Ops.CAST: return _extract_affine(uop.src[0], sym)
            if uop.op is Ops.MUL:
                if _sym_or_cast(uop.src[0], sym):
                    c = _const_val(uop.src[1])
                    return (c, 0.0) if c is not None else None
                if _sym_or_cast(uop.src[1], sym):
                    c = _const_val(uop.src[0])
                    return (c, 0.0) if c is not None else None
            if uop.op is Ops.ADD:
                lhs = _extract_affine(uop.src[0], sym)
                rhs = _extract_affine(uop.src[1], sym)
                if lhs is None or rhs is None: return None
                return (lhs[0] + rhs[0], lhs[1] + rhs[1])
            if uop.op is Ops.SUB:
                lhs = _extract_affine(uop.src[0], sym)
                rhs = _extract_affine(uop.src[1], sym)
                if lhs is None or rhs is None: return None
                return (lhs[0] - rhs[0], lhs[1] - rhs[1])
            c = _const_val(uop)
            if c is not None: return (0.0, c)
            return None

        dHdq_aff = _extract_affine(dHdq_uop, q_sym.uop)
        dHdp_aff = _extract_affine(dHdp_uop, p_sym.uop)
        ret = (dHdq_aff[0], dHdp_aff[0], dHdq_aff[1], dHdp_aff[1]) if dHdq_aff and dHdp_aff else None
        self._ho_coeff_cache[key] = ret
        return ret

    def _build_leapfrog_scan_kernel_coupled(self, dt: float, steps: int, device: str, shape: tuple[int, ...], dtype,
                                            unroll_steps: int = 1, vector_width: int = 1) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        dHdq_reduce_nodes = [u for u in dHdq_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        dHdp_reduce_nodes = [u for u in dHdp_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        has_reduce = len(dHdq_reduce_nodes) > 0 or len(dHdp_reduce_nodes) > 0
        use_vec = vector_width > 1
        if has_reduce and not getenv("TINYGRAD_COUPLED_FUSED_EXPERIMENTAL", 0):
            raise ValueError("coupled reductions require experimental fused path")
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])
        reduce_placeholders_dHdq = {red: UOp.const(red.dtype, 0, device) for red in dHdq_reduce_nodes} if has_reduce else {}
        reduce_placeholders_dHdp = {red: UOp.const(red.dtype, 0, device) for red in dHdp_reduce_nodes} if has_reduce else {}
        if reduce_placeholders_dHdq or reduce_placeholders_dHdp:
            def replace_reduce(ctx: dict, uop: UOp) -> UOp|None:
                repl = ctx["reduce_placeholders"].get(uop)
                if repl is None: return None
                return repl
            reduce_pm = PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="uop"), replace_reduce)], compiled=False)
            dHdq_elem_uop = graph_rewrite(
                dHdq_uop, reduce_pm, ctx={"reduce_placeholders": reduce_placeholders_dHdq}, name="replace_reduce")
            dHdp_elem_uop = graph_rewrite(
                dHdp_uop, reduce_pm, ctx={"reduce_placeholders": reduce_placeholders_dHdp}, name="replace_reduce")
            if use_vec:
                const_vec = PatternMatcher([
                    (UPat((Ops.CONST, Ops.VCONST), name="x"), _const_to_vec),
                ], compiled=False)
                dHdq_elem_uop = graph_rewrite(dHdq_elem_uop, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
                dHdp_elem_uop = graph_rewrite(dHdp_elem_uop, const_vec, ctx={"vector_width": vector_width}, name="const_vec")
                drop_scalar_expand = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"),
                     lambda x: x.src[0] if x.src[0]._shape is None else None),
                ], compiled=False)
                dHdq_elem_uop = graph_rewrite(dHdq_elem_uop, drop_scalar_expand, ctx={}, name="drop_scalar_expand")
                dHdp_elem_uop = graph_rewrite(dHdp_elem_uop, drop_scalar_expand, ctx={}, name="drop_scalar_expand")
                drop_vector_expand = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_vector_expand),
                ], compiled=False)
                dHdq_elem_uop = graph_rewrite(
                    dHdq_elem_uop, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
                dHdp_elem_uop = graph_rewrite(
                    dHdp_elem_uop, drop_vector_expand, ctx={"vector_width": vector_width}, name="drop_vector_expand")
        else:
            dHdq_elem_uop = dHdq_uop
            dHdp_elem_uop = dHdp_uop

        def kernel(q: UOp, p: UOp) -> UOp:
            tile_steps = steps // unroll_steps
            step = UOp.range(tile_steps, 0)
            q_base = q.after(step)
            p_base = p.after(step)
            def rewrite_with_ranges(uop: UOp, ranges: list[UOp], reduce_counter: list[int]) -> UOp:
                lower_reduce_axis = PatternMatcher([
                    (UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis),
                ], compiled=False)
                const_to_vindex = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex),
                ], compiled=False)
                drop_scalar_value_index = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index),
                ], compiled=False)
                drop_const_reshape = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape),
                ], compiled=False)
                broadcast_value_index = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index),
                ], compiled=False)
                broadcast_scalar_index = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index),
                ], compiled=False)
                broadcast_scalar_base = PatternMatcher([
                    (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base),
                ], compiled=False)
                uop = graph_rewrite(
                    uop,
                    lower_reduce_axis,
                    ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                    name="lower_reduce_axis",
                )
                uop = graph_rewrite(
                    uop,
                    const_to_vindex,
                    ctx={"ranges": ranges},
                    name="const_to_vindex",
                )
                uop = graph_rewrite(
                    uop,
                    drop_scalar_value_index,
                    ctx={},
                    name="drop_scalar_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    drop_const_reshape,
                    ctx={},
                    name="drop_const_reshape",
                )
                uop = graph_rewrite(
                    uop,
                    broadcast_value_index,
                    ctx={"ranges": ranges},
                    name="broadcast_value_index",
                )
                uop = graph_rewrite(
                    uop,
                    broadcast_scalar_index,
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_index",
                )
                uop = graph_rewrite(
                    uop,
                    broadcast_scalar_base,
                    ctx={"ranges": ranges},
                    name="broadcast_scalar_base",
                )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                    ctx={"ranges": ranges},
                    name="reindex_reduce_input",
                )
                uop = graph_rewrite(uop, strip_device_consts, name="strip_device_consts")
                return uop

            if has_reduce:
                ranges_reduce = [UOp.range(s, i+1+len(shape)) for i,s in enumerate(shape)]
                reduce_counter = [max(r.arg[0] for r in ranges_reduce) + 1]
                reduce_full = {red: (len(red.arg[1]) == len(shape) and all(i in red.arg[1] for i in range(len(shape))))
                               for red in dHdq_reduce_nodes + dHdp_reduce_nodes}
                if use_vec:
                    ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
                    ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                    zero_idxs = tuple(UOp.const(dtypes.index, 0) for _ in range(len(shape)))
                    ranges_reduce_vec = [UOp.range(s, i+1+len(shape)) for i, s in enumerate(shape[:-1])]
                    ranges_reduce_vec.append(UOp.range(shape[-1] // vector_width, 2 * len(shape)))
                    reduce_counter_vec = [max(r.arg[0] for r in ranges_reduce_vec) + 1]
                    def compute_accs(reduce_nodes: list[UOp], reduce_placeholders: dict[UOp, UOp],
                                     q_uop: UOp, p_uop: UOp, reg_base: int) -> dict[UOp, UOp]:
                        if not reduce_nodes: return {}
                        acc_loads = {}
                        index_map = {red: i for i, red in enumerate(reduce_nodes)}
                        sub_reduce = {q_sym_uop: q_uop.vindex(*ranges_reduce), p_sym_uop: p_uop.vindex(*ranges_reduce)}
                        base = ranges_reduce_vec[-1] * UOp.const(dtypes.index, vector_width)
                        q_ptr = q_uop.index(*ranges_reduce_vec[:-1], base, ptr=True)
                        p_ptr = p_uop.index(*ranges_reduce_vec[:-1], base, ptr=True)
                        vec_dtype = q_uop.dtype.base.vec(vector_width)
                        q_val_red = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace)).load()
                        p_val_red = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace)).load()
                        sub_reduce_vec = {q_sym_uop: q_val_red, p_sym_uop: p_val_red}
                        full_nodes = [red for red in reduce_nodes if reduce_full.get(red, False)]
                        can_group = len(full_nodes) > 1
                        if can_group:
                            for red in full_nodes:
                                if any(u.op is Ops.REDUCE_AXIS for u in red.src[0].toposort()):
                                    can_group = False
                                    break
                        grouped = set()
                        if can_group:
                            expr_cache: dict[bytes, UOp] = {}
                            op_groups: dict[Ops, list[UOp]] = {}
                            for red in full_nodes:
                                op_groups.setdefault(red.arg[0], []).append(red)
                            for op_kind, group in op_groups.items():
                                exprs = []
                                for red in group:
                                    expr = red.src[0].substitute(sub_reduce_vec)
                                    expr = rewrite_with_ranges(expr, ranges_reduce_vec, reduce_counter_vec)
                                    if expr.dtype.count > 1:
                                        expr = _hreduce_vector(expr, op_kind)
                                    cached = expr_cache.get(expr.key)
                                    if cached is None:
                                        expr_cache[expr.key] = expr
                                    else:
                                        expr = cached
                                    exprs.append(expr)
                                vec_expr = exprs[0].vectorize(*exprs[1:]) if len(exprs) > 1 else exprs[0]
                                red_uop = UOp(
                                    Ops.REDUCE, vec_expr.dtype, src=(vec_expr,)+tuple(ranges_reduce_vec), arg=op_kind)
                                red_val = red_uop.vindex(*zero_idxs)
                                reduce_ranges = [r for r in red_uop.ranges if r.arg[-1] == AxisType.REDUCE]
                                for j, red in enumerate(group):
                                    idx = index_map[red]
                                    acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=reg_base + idx)
                                    acc_store = acc.index(UOp.const(dtypes.int, 0)).store(red_val.gep(j)).end(*reduce_ranges, step)
                                    acc_idx = acc.after(acc_store).index(UOp.const(dtypes.int, 0))
                                    acc_loads[reduce_placeholders[red]] = acc_idx.broadcast(vector_width)
                                    grouped.add(red)
                        for i, red in enumerate(reduce_nodes):
                            if red in grouped: continue
                            if reduce_full.get(red, False):
                                red_val = red.substitute(sub_reduce_vec)
                                red_val = rewrite_with_ranges(red_val, ranges_reduce_vec, reduce_counter_vec)
                                if red_val.dtype.count > 1:
                                    red_val = _hreduce_vector(red_val, red.arg[0])
                                acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=reg_base + i)
                                reduce_ranges = [r for r in red_val.ranges if r.arg[-1] == AxisType.REDUCE]
                                acc_store = acc.index(UOp.const(dtypes.int, 0)).store(red_val).end(*reduce_ranges, step)
                                acc_idx = acc.after(acc_store).index(UOp.const(dtypes.int, 0))
                                acc_loads[reduce_placeholders[red]] = acc_idx.broadcast(vector_width)
                            else:
                                red_val = red.substitute(sub_reduce)
                                red_val = rewrite_with_ranges(red_val, ranges_reduce, reduce_counter)
                                red_val = red_val.vindex(*ranges)
                                acc_loads[reduce_placeholders[red]] = red_val.broadcast(vector_width)
                        return acc_loads
                else:
                    ranges = [UOp.range(s, i+1) for i,s in enumerate(shape)]
                    zero_idxs = tuple(UOp.const(dtypes.index, 0) for _ in range(len(shape)))
                    def compute_accs(reduce_nodes: list[UOp], reduce_placeholders: dict[UOp, UOp],
                                     q_uop: UOp, p_uop: UOp, reg_base: int) -> dict[UOp, UOp]:
                        if not reduce_nodes: return {}
                        acc_loads = {}
                        index_map = {red: i for i, red in enumerate(reduce_nodes)}
                        sub_reduce = {q_sym_uop: q_uop.vindex(*ranges_reduce), p_sym_uop: p_uop.vindex(*ranges_reduce)}
                        full_nodes = [red for red in reduce_nodes if reduce_full.get(red, False)]
                        can_group = len(full_nodes) > 1
                        if can_group:
                            for red in full_nodes:
                                if any(u.op is Ops.REDUCE_AXIS for u in red.src[0].toposort()):
                                    can_group = False
                                    break
                        grouped = set()
                        if can_group:
                            expr_cache: dict[bytes, UOp] = {}
                            op_groups: dict[Ops, list[UOp]] = {}
                            for red in full_nodes:
                                op_groups.setdefault(red.arg[0], []).append(red)
                            for op_kind, group in op_groups.items():
                                exprs = []
                                for red in group:
                                    expr = red.src[0].substitute(sub_reduce)
                                    expr = rewrite_with_ranges(expr, ranges_reduce, reduce_counter)
                                    if expr.dtype.count > 1:
                                        expr = _hreduce_vector(expr, op_kind)
                                    cached = expr_cache.get(expr.key)
                                    if cached is None:
                                        expr_cache[expr.key] = expr
                                    else:
                                        expr = cached
                                    exprs.append(expr)
                                vec_expr = exprs[0].vectorize(*exprs[1:]) if len(exprs) > 1 else exprs[0]
                                red_uop = UOp(
                                    Ops.REDUCE, vec_expr.dtype, src=(vec_expr,)+tuple(ranges_reduce), arg=op_kind)
                                red_val = red_uop.vindex(*zero_idxs)
                                reduce_ranges = [r for r in red_uop.ranges if r.arg[-1] == AxisType.REDUCE]
                                for j, red in enumerate(group):
                                    idx = index_map[red]
                                    acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=reg_base + idx)
                                    acc_store = acc.index(UOp.const(dtypes.int, 0)).store(red_val.gep(j)).end(*reduce_ranges, step)
                                    acc_idx = acc.after(acc_store).index(UOp.const(dtypes.int, 0))
                                    acc_loads[reduce_placeholders[red]] = acc_idx
                                    grouped.add(red)
                        for i, red in enumerate(reduce_nodes):
                            if red in grouped: continue
                            red_val = red.substitute(sub_reduce)
                            red_val = rewrite_with_ranges(red_val, ranges_reduce, reduce_counter)
                            if reduce_full.get(red, False):
                                acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=reg_base + i)
                                reduce_ranges = [r for r in red_val.ranges if r.arg[-1] == AxisType.REDUCE]
                                acc_store = acc.index(UOp.const(dtypes.int, 0)).store(red_val).end(*reduce_ranges, step)
                                acc_idx = acc.after(acc_store).index(UOp.const(dtypes.int, 0))
                                acc_loads[reduce_placeholders[red]] = acc_idx
                            else:
                                acc_loads[reduce_placeholders[red]] = red_val.vindex(*ranges)
                        return acc_loads

                phase_groups = []
                dep = None
                shape_vals = tuple(r.vmax + 1 for r in ranges)
                for _ in range(unroll_steps):
                    q_phase = q_base if dep is None else q_base.after(dep)
                    p_phase = p_base if dep is None else p_base.after(dep)
                    acc_loads1 = compute_accs(dHdq_reduce_nodes, reduce_placeholders_dHdq, q_phase, p_phase, 0)
                    if use_vec:
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                        q_ptr = q_phase.index(*ranges[:-1], base, ptr=True)
                        p_ptr = p_phase.index(*ranges[:-1], base, ptr=True)
                        vec_dtype = q_phase.dtype.base.vec(vector_width)
                        q_vec_ptr = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                        p_vec_ptr = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                        q_val = q_vec_ptr.load()
                        p_val = p_vec_ptr.load()
                    else:
                        q_val = q_phase.vindex(*ranges)
                        p_val = p_phase.vindex(*ranges)
                    dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    if acc_loads1:
                        dHdq_1 = dHdq_1.substitute(acc_loads1, name="reduce_placeholders")
                        dHdq_1 = graph_rewrite(
                            dHdq_1,
                            PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                            ctx={}, name="drop_reg_expand")
                    half_dt = UOp.const(dHdq_1.dtype, 0.5*dt, shape=shape_vals if not use_vec else None)
                    neg_one = UOp.const(dHdq_1.dtype, -1.0, shape=shape_vals if not use_vec else None)
                    p_half = p_val + (half_dt * dHdq_1) * neg_one
                    if use_vec:
                        store_p_half = p_vec_ptr.store(p_half)
                    else:
                        store_p_half = p_phase.index(*ranges, ptr=True).store(p_half)
                    dep = store_p_half

                    q_phase = q_base.after(dep)
                    p_phase = p_base.after(dep)
                    reg_base = len(dHdq_reduce_nodes)
                    acc_loads2 = compute_accs(dHdp_reduce_nodes, reduce_placeholders_dHdp, q_phase, p_phase, reg_base)
                    if use_vec:
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                        q_ptr = q_phase.index(*ranges[:-1], base, ptr=True)
                        p_ptr = p_phase.index(*ranges[:-1], base, ptr=True)
                        vec_dtype = q_phase.dtype.base.vec(vector_width)
                        q_vec_ptr = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                        p_vec_ptr = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                        q_val = q_vec_ptr.load()
                        p_val = p_vec_ptr.load()
                    else:
                        q_val = q_phase.vindex(*ranges)
                        p_val = p_phase.vindex(*ranges)
                    dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    if acc_loads2:
                        dHdp = dHdp.substitute(acc_loads2, name="reduce_placeholders")
                        dHdp = graph_rewrite(
                            dHdp,
                            PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                            ctx={}, name="drop_reg_expand")
                    dt_uop = UOp.const(dHdp.dtype, dt, shape=shape_vals if not use_vec else None)
                    q_new = q_val + dt_uop * dHdp
                    if use_vec:
                        store_q_new = q_vec_ptr.store(q_new)
                    else:
                        store_q_new = q_phase.index(*ranges, ptr=True).store(q_new)
                    dep = store_q_new

                    q_phase = q_base.after(dep)
                    p_phase = p_base.after(dep)
                    reg_base = len(dHdq_reduce_nodes) + len(dHdp_reduce_nodes)
                    acc_loads3 = compute_accs(dHdq_reduce_nodes, reduce_placeholders_dHdq, q_phase, p_phase, reg_base)
                    if use_vec:
                        base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                        q_ptr = q_phase.index(*ranges[:-1], base, ptr=True)
                        p_ptr = p_phase.index(*ranges[:-1], base, ptr=True)
                        vec_dtype = q_phase.dtype.base.vec(vector_width)
                        q_vec_ptr = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                        p_vec_ptr = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                        q_val = q_vec_ptr.load()
                        p_val = p_vec_ptr.load()
                    else:
                        q_val = q_phase.vindex(*ranges)
                        p_val = p_phase.vindex(*ranges)
                    dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    if acc_loads3:
                        dHdq_2 = dHdq_2.substitute(acc_loads3, name="reduce_placeholders")
                        dHdq_2 = graph_rewrite(
                            dHdq_2,
                            PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_reg_expand)], compiled=False),
                            ctx={}, name="drop_reg_expand")
                    half_dt = UOp.const(dHdq_2.dtype, 0.5*dt, shape=shape_vals if not use_vec else None)
                    p_new = p_val + (half_dt * dHdq_2) * neg_one
                    if use_vec:
                        store_p_new = p_vec_ptr.store(p_new)
                    else:
                        store_p_new = p_phase.index(*ranges, ptr=True).store(p_new)
                    dep = store_p_new
                    phase = UOp.group(store_p_half, store_q_new, store_p_new)
                    phase_groups.append(phase.end(*ranges, step))

                store_group = UOp.group(*phase_groups)
                opts_to_apply = ()
            else:
                if use_vec:
                    ranges = [UOp.range(s, i+1) for i,s in enumerate(shape[:-1])]
                    ranges.append(UOp.range(shape[-1] // vector_width, len(shape)))
                else:
                    ranges = [UOp.range(s, i+1) for i,s in enumerate(shape)]
                def grad_uop(q_uop: UOp, p_uop: UOp) -> tuple[UOp, UOp]:
                    sub = {q_sym_uop: q_uop, p_sym_uop: p_uop}
                    dHdq = dHdq_elem_uop.substitute(sub)
                    dHdp = dHdp_elem_uop.substitute(sub)
                    return dHdq, dHdp

                if use_vec:
                    base = ranges[-1] * UOp.const(dtypes.index, vector_width)
                    q_ptr = q_base.index(*ranges[:-1], base, ptr=True)
                    p_ptr = p_base.index(*ranges[:-1], base, ptr=True)
                    vec_dtype = q_base.dtype.base.vec(vector_width)
                    q_vec_ptr = q_ptr.cast(vec_dtype.ptr(size=q_ptr.dtype.size, addrspace=q_ptr.dtype.addrspace))
                    p_vec_ptr = p_ptr.cast(vec_dtype.ptr(size=p_ptr.dtype.size, addrspace=p_ptr.dtype.addrspace))
                    q_val = q_vec_ptr.load()
                    p_val = p_vec_ptr.load()
                else:
                    q_val = q_base.vindex(*ranges)
                    p_val = p_base.vindex(*ranges)
                for _ in range(unroll_steps):
                    dHdq_1, _ = grad_uop(q_val, p_val)
                    if use_vec:
                        dt_uop = UOp.const(q_val.dtype, dt)
                        half_dt = UOp.const(q_val.dtype, 0.5*dt)
                    else:
                        dt_uop = dHdq_1.const_like(dt)
                        half_dt = dHdq_1.const_like(0.5*dt)
                        try:
                            if dt_uop.shape is not None and len(dt_uop.shape) == len(ranges):
                                dt_uop = dt_uop.vindex(*ranges)
                            if half_dt.shape is not None and len(half_dt.shape) == len(ranges):
                                half_dt = half_dt.vindex(*ranges)
                        except Exception:
                            pass
                    p_half = p_val - half_dt * dHdq_1
                    _, dHdp = grad_uop(q_val, p_half)
                    q_new = q_val + dt_uop * dHdp
                    dHdq_2, _ = grad_uop(q_new, p_half)
                    p_new = p_half - half_dt * dHdq_2
                    q_val, p_val = q_new, p_new

                if use_vec:
                    store_q = q_vec_ptr.store(q_val)
                    store_p = p_vec_ptr.store(p_val)
                else:
                    store_q = q_base.index(*ranges, ptr=True).store(q_val)
                    store_p = p_base.index(*ranges, ptr=True).store(p_val)
                store_group = UOp.group(store_q, store_p).end(*ranges, step)
                opts_to_apply = ()
            if use_vec:
                axis = len(ranges) - 1
                opts_to_apply = (Opt(OptOps.UNROLL, axis, vector_width),)
            kernel_sink = store_group.sink(
                arg=KernelInfo(name=f"leapfrog_scan_coupled_{steps}", opts_to_apply=opts_to_apply))
            reduce_counter = [len(ranges) + 1]
            lower_reduce_axis = PatternMatcher([
                (UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis),
            ], compiled=False)
            const_to_vindex = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex),
            ], compiled=False)
            drop_scalar_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index),
            ], compiled=False)
            drop_const_reshape = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape),
            ], compiled=False)
            broadcast_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index),
            ], compiled=False)
            broadcast_scalar_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index),
            ], compiled=False)
            broadcast_scalar_base = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base),
            ], compiled=False)
            kernel_sink = graph_rewrite(
                kernel_sink,
                lower_reduce_axis,
                ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                name="lower_reduce_axis",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                const_to_vindex,
                ctx={"ranges": ranges},
                name="const_to_vindex",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_scalar_value_index,
                ctx={},
                name="drop_scalar_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_const_reshape,
                ctx={},
                name="drop_const_reshape",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_value_index,
                ctx={"ranges": ranges},
                name="broadcast_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_index,
                ctx={"ranges": ranges},
                name="broadcast_scalar_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_base,
                ctx={"ranges": ranges},
                name="broadcast_scalar_base",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                ctx={"ranges": ranges},
                name="reindex_reduce_input",
            )
            kernel_sink = graph_rewrite(kernel_sink, strip_device_consts, name="strip_device_consts")
            return _cse_uops(kernel_sink)

        return kernel

    def _build_leapfrog_step_kernel_coupled(self, dt: float, device: str, shape: tuple[int, ...], dtype) -> Callable:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(device, shape, dtype)
        strip_device_consts = PatternMatcher([
            (UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const),
        ])

        def kernel(q: UOp, p: UOp, q_out: UOp, p_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i,s in enumerate(shape)]
            reduce_counter = [len(ranges) + 1]
            def grad_uop(q_uop: UOp, p_uop: UOp) -> tuple[UOp, UOp]:
                sub = {q_sym_uop: q_uop, p_sym_uop: p_uop}
                dHdq = dHdq_uop.substitute(sub)
                dHdp = dHdp_uop.substitute(sub)
                return dHdq, dHdp

            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)

            dHdq_1, _ = grad_uop(q_val, p_val)
            dt_uop = dHdq_1.const_like(dt)
            half_dt = dHdq_1.const_like(0.5*dt)
            try:
                if dt_uop.shape is not None and len(dt_uop.shape) == len(ranges):
                    dt_uop = dt_uop.vindex(*ranges)
                if half_dt.shape is not None and len(half_dt.shape) == len(ranges):
                    half_dt = half_dt.vindex(*ranges)
            except Exception:
                pass
            p_half = p_val - half_dt * dHdq_1
            _, dHdp = grad_uop(q_val, p_half)
            q_new = q_val + dt_uop * dHdp
            dHdq_2, _ = grad_uop(q_new, p_half)
            p_new = p_half - half_dt * dHdq_2

            store_q = q_out.index(*ranges, ptr=True).store(q_new.vindex(*ranges))
            store_p = p_out.index(*ranges, ptr=True).store(p_new.vindex(*ranges))
            kernel_sink = UOp.group(store_q, store_p).end(*ranges).sink(
                arg=KernelInfo(name="leapfrog_step_coupled", opts_to_apply=()))
            lower_reduce_axis = PatternMatcher([
                (UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis),
            ], compiled=False)
            const_to_vindex = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex),
            ], compiled=False)
            drop_scalar_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_scalar_value_index),
            ], compiled=False)
            drop_const_reshape = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_reshape),
            ], compiled=False)
            broadcast_value_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_value_index),
            ], compiled=False)
            broadcast_scalar_index = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_index),
            ], compiled=False)
            broadcast_scalar_base = PatternMatcher([
                (UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _broadcast_scalar_base),
            ], compiled=False)
            kernel_sink = graph_rewrite(
                kernel_sink,
                lower_reduce_axis,
                ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                name="lower_reduce_axis",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                const_to_vindex,
                ctx={"ranges": ranges},
                name="const_to_vindex",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_scalar_value_index,
                ctx={},
                name="drop_scalar_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                drop_const_reshape,
                ctx={},
                name="drop_const_reshape",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_value_index,
                ctx={"ranges": ranges},
                name="broadcast_value_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_index,
                ctx={"ranges": ranges},
                name="broadcast_scalar_index",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                broadcast_scalar_base,
                ctx={"ranges": ranges},
                name="broadcast_scalar_base",
            )
            kernel_sink = graph_rewrite(
                kernel_sink,
                PatternMatcher([(UPat(Ops.REDUCE, name="x"), _reindex_reduce_input)], compiled=False),
                ctx={"ranges": ranges},
                name="reindex_reduce_input",
            )
            kernel_sink = graph_rewrite(kernel_sink, strip_device_consts, name="strip_device_consts")
            return _cse_uops(kernel_sink)

        return kernel


    def compile_unrolled_step(self, dt: float, unroll: int):
        """Prototype unroll/scan by compiling N steps into one captured graph."""
        if unroll < 1:
            raise ValueError("unroll must be >= 1")

        def unrolled_step(q: Tensor, p: Tensor):
            for _ in range(unroll):
                q, p = self.step(q, p, dt)
            return q, p

        return TinyJit(unrolled_step)

    def compile_unrolled_step_inplace(self, dt: float, unroll: int):
        """In-place unroll by compiling N step_inplace calls into one captured graph."""
        if unroll < 1:
            raise ValueError("unroll must be >= 1")

        def unrolled_step(q: Tensor, p: Tensor, dt_inner: float):
            for _ in range(unroll):
                q, p = self.step_inplace(q, p, dt_inner)
            return q, p

        return TinyJit(unrolled_step)

    def compile_unrolled_step_buffered(self, dt: float, unroll: int):
        """Unroll using two buffers to avoid in-place aliasing for implicit steps."""
        if unroll < 1:
            raise ValueError("unroll must be >= 1")

        def unrolled_step(q: Tensor, p: Tensor, q_buf: Tensor, p_buf: Tensor):
            q_a, p_a = q, p
            q_b, p_b = q_buf, p_buf
            for _ in range(unroll):
                if self._select_integrator_name(q_a, p_a) == "implicit":
                    q_b, p_b = self._implicit_midpoint_into_fast(q_a, p_a, q_b, p_b, dt)
                else:
                    q_b, p_b = self.step(q_a, p_a, dt)
                q_a, p_a, q_b, p_b = q_b, p_b, q_a, p_a
            return q_a, p_a

        return TinyJit(unrolled_step)

    def compile_unrolled_step_implicit(self, dt: float, unroll: int, iters: int, realize_steps: bool = False):
        """Unroll implicit midpoint with fixed iterations, optionally realizing between steps."""
        if unroll < 1:
            raise ValueError("unroll must be >= 1")
        if iters < 1:
            raise ValueError("iters must be >= 1 for implicit unroll")

        def unrolled_step(q: Tensor, p: Tensor):
            for _ in range(unroll):
                q, p = implicit_midpoint_fixed(q, p, self.H, dt, iters)
                if realize_steps:
                    q, p = q.realize(), p.realize()
            return q, p

        return TinyJit(unrolled_step)

    def compile_implicit_step_fixed(self, dt: float, iters: int):
        """Single implicit midpoint step with fixed iterations, compiled."""
        if iters < 1:
            raise ValueError("iters must be >= 1 for implicit step")

        def step_fn(q: Tensor, p: Tensor):
            return implicit_midpoint_fixed(q, p, self.H, dt, iters)

        return TinyJit(step_fn)

    def evolve_unrolled(self, q: Tensor, p: Tensor, dt: float, steps: int, unroll: int = 8,
                        record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        """Prototype unrolled evolution; records at unroll boundaries only."""
        if steps % unroll != 0:
            raise ValueError("steps must be divisible by unroll")
        if record_every % unroll != 0:
            raise ValueError("record_every must be divisible by unroll")

        q_history: list[Tensor] = []
        p_history: list[Tensor] = []
        step = self.compile_unrolled_step(dt, unroll)

        for i in range(0, steps, unroll):
            if i % record_every == 0:
                q_history.append(q.detach())
                p_history.append(p.detach())
            q, p = step(q, p)

        q_history.append(q.detach())
        p_history.append(p.detach())

        history = []
        for q_t, p_t in zip(q_history, p_history):
            q_np = q_t.numpy().copy()
            p_np = p_t.numpy().copy()
            e = float(self.H(q_t, p_t).numpy())
            history.append((q_np, p_np, e))

        return q, p, history

# ============================================================================
# LIE-POISSON MECHANICS (Phase 2: Rigid Body Dynamics)
# ============================================================================
#
# For systems on Lie algebras (like rigid body rotation), the Poisson structure
# is state-dependent: J = J(z). The equation of motion is:
#
#     dz/dt = J(z) · ∇H(z)
#
# For so(3) (angular momentum): J(L) acts via cross product
#     dL/dt = L × ∇H(L)
#
# This is Euler's equation when H = 0.5 * L · (I⁻¹ L)

def _reshape_for_vec(z: Tensor, vec_width: int, vec_dim: int) -> tuple[Tensor, tuple[int, ...] | None]:
  if vec_width <= 1:
    return z, None
  if not z.shape or z.shape[-1] != vec_dim:
    return z, None
  if z.numel() % (vec_width * vec_dim) != 0:
    return z, None
  orig = z.shape
  return z.reshape(-1, vec_width, vec_dim), orig


def _reshape_from_vec(z: Tensor, orig: tuple[int, ...] | None) -> Tensor:
  return z.reshape(orig) if orig is not None else z


def cross(a: Tensor, b: Tensor) -> Tensor:
  """Cross product a × b for 3-vectors (supports leading batch dims)."""
  ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
  bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
  return Tensor.stack(
    ay * bz - az * by,
    az * bx - ax * bz,
    ax * by - ay * bx,
    dim=-1,
  )


def lie_poisson_euler_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
  """1st-order Lie-Poisson Euler for so(3): dL/dt = L × ∇H(L)"""
  dHdL = _grad_LP(L, H_func)
  return L + dt * cross(L, dHdL)


def lie_poisson_midpoint_so3(L: Tensor, H_func, dt: float = 0.01,
                             tol: float = 1e-10, max_iter: int = 10) -> Tensor:
  """Implicit Midpoint for so(3). Preserves Casimirs (|L|²) to machine precision."""
  dHdL = _grad_LP(L, H_func)
  L_next = L + dt * cross(L, dHdL)

  for _ in range(max_iter):
    L_mid = 0.5 * (L + L_next)
    dHdL_mid = _grad_LP(L_mid, H_func)
    L_new = L + dt * cross(L_mid, dHdL_mid)
    if (L_new - L_next).abs().max().numpy() < tol:
      break
    L_next = L_new

  return L_next


def lie_poisson_rk4_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
  """4th-order RK4 for so(3). Good accuracy, does NOT preserve Casimirs exactly."""
  def f(L_val):
    return cross(L_val, _grad_LP(L_val, H_func))

  k1 = f(L)
  k2 = f(L + 0.5 * dt * k1)
  k3 = f(L + 0.5 * dt * k2)
  k4 = f(L + dt * k3)
  return L + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def lie_poisson_splitting_so3(L: Tensor, I_inv: Tensor, dt: float = 0.01) -> Tensor:
  """
  Explicit splitting method for free rigid body.
  Preserves |L|² exactly (Casimir). 2nd order via Strang splitting.
  """
  def rotate_axis(L_val: Tensor, axis: int, angle: Tensor) -> Tensor:
    j, k = (axis + 1) % 3, (axis + 2) % 3
    c, s = angle.cos(), angle.sin()
    L_j_new = c * L_val[..., j] - s * L_val[..., k]
    L_k_new = s * L_val[..., j] + c * L_val[..., k]
    if axis == 0: return Tensor.stack([L_val[..., 0], L_j_new, L_k_new], dim=-1)
    if axis == 1: return Tensor.stack([L_k_new, L_val[..., 1], L_j_new], dim=-1)
    return Tensor.stack([L_j_new, L_k_new, L_val[..., 2]], dim=-1)

  # Strang splitting: half-step forward, half-step reverse
  for axis in range(3):
    L = rotate_axis(L, axis, 0.5 * dt * L[..., axis] * I_inv[axis])
  for axis in [2, 1, 0]:
    L = rotate_axis(L, axis, 0.5 * dt * L[..., axis] * I_inv[axis])
  return L


class LiePoissonSystem:
  """
  Physics on a Lie algebra, defined by its Hamiltonian.

  For the free rigid body: H(L) = 0.5 * sum(L² / I)
  """
  INTEGRATORS = {
    "euler": lie_poisson_euler_so3,
    "midpoint": lie_poisson_midpoint_so3,
    "rk4": lie_poisson_rk4_so3,
  }

  def __init__(self, H_func, algebra: str = "so3", integrator: str = "midpoint", policy: SymplecticPolicy | None = None):
    if algebra != "so3":
      raise ValueError(f"Only so3 supported, got: {algebra}")
    self.H = H_func
    self.algebra = algebra
    if policy is None:
      from tinygrad.physics_profile import get_default_profile
      policy = get_default_profile().policy
    self.policy = policy
    if integrator == "auto":
      strict_drift = self.policy.drift_target is not None and self.policy.drift_target <= 1e-6
      integrator = "midpoint" if self.policy.accuracy != "fast" or strict_drift else "euler"
    if integrator not in self.INTEGRATORS:
      raise ValueError(f"Unknown integrator: {integrator}")
    self._step = self.INTEGRATORS[integrator]
    self.integrator_name = integrator

  def step(self, z: Tensor, dt: float = 0.01) -> Tensor:
    return self._step(z, self.H, dt)

  def energy(self, z: Tensor) -> float:
    return float(self.H(z).numpy())

  def casimir(self, z: Tensor) -> float:
    return float((z * z).sum().numpy())

  def evolve(self, z: Tensor, dt: float, steps: int,
             record_every: int = 1, policy: SymplecticPolicy | None = None,
             scan: bool | None = None, unroll: int | None = None,
             vector_width: int | None = None) -> tuple[Tensor, list]:
    policy = policy or self.policy
    use_scan = policy.should_scan(steps, z.shape, z.device) if scan is None else scan
    op_cost = None
    if getenv("TINYGRAD_PHYSICS_UNROLL_COST", 1):
      try:
        op_cost = _count_uops_limit(self.H(z).uop, 512)
      except Exception:
        op_cost = None
    def make_step(scale: float):
      return lambda z_in: self.step(z_in, dt * scale)
    def energy_fn(z_in: Tensor):
      return float(self.H(z_in).numpy())
    dt_scale, _ = policy.adjust_dt(steps, z.shape, z.device, (z,), make_step, energy_fn, "TINYGRAD_PHYSICS_DRIFT_LIE")
    if dt_scale != 1.0:
      dt *= dt_scale
    if vector_width is None:
      vector_width = policy.choose_vector_width(z.shape, z.device, 3)
    z_use, z_shape = _reshape_for_vec(z, vector_width, 3)
    if unroll is None and use_scan:
      unroll = policy.choose_unroll(steps, z_use.shape, z_use.device, op_cost=op_cost)
    if unroll is not None:
      if record_every % unroll != 0:
        record_every = unroll
      out = self.evolve_unrolled(z_use, dt, steps, unroll, record_every=record_every)
      policy.maybe_print_report(steps, z.shape, z.device)
      z_out, hist = out
      if z_shape is not None:
        hist = [(z_t.reshape(z_shape), e, c) for (z_t, e, c) in hist]
      return _reshape_from_vec(z_out, z_shape), hist
    z_history: list[Tensor] = []
    for i in range(steps):
      if i % record_every == 0:
        z_history.append(z_use.detach())
      z_use = self.step(z_use, dt)
    z_history.append(z_use.detach())
    history = []
    for z_t in z_history:
      z_emit = _reshape_from_vec(z_t, z_shape)
      history.append((z_emit.numpy().copy(), self.energy(z_emit), self.casimir(z_emit)))
    policy.maybe_print_report(steps, z.shape, z.device)
    return _reshape_from_vec(z_use, z_shape), history

  def compile_unrolled_step(self, dt: float, unroll: int):
    """Compile an unrolled step for Lie-Poisson systems."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")

    def unrolled_step(z: Tensor):
      for _ in range(unroll):
        z = self.step(z, dt)
      return z

    return TinyJit(unrolled_step)

  def evolve_unrolled(self, z: Tensor, dt: float, steps: int, unroll: int,
                      record_every: int = 1) -> tuple[Tensor, list]:
    """Evolve using an unrolled compiled step to reduce Python overhead."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if steps % unroll != 0:
      raise ValueError("steps must be divisible by unroll")
    if record_every % unroll != 0:
      raise ValueError("record_every must be divisible by unroll")

    step = self.compile_unrolled_step(dt, unroll)
    z_history: list[Tensor] = []
    for i in range(0, steps, unroll):
      if i % record_every == 0:
        z_history.append(z.detach())
      z = step(z)
    z_history.append(z.detach())
    history = [(z_t.numpy().copy(), self.energy(z_t), self.casimir(z_t)) for z_t in z_history]
    return z, history


def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
  """Quaternion multiplication q1 * q2. Format: [w, x, y, z]."""
  w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
  w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
  return Tensor.stack([
    w1*w2 - x1*x2 - y1*y2 - z1*z2,
    w1*x2 + x1*w2 + y1*z2 - z1*y2,
    w1*y2 - x1*z2 + y1*w2 + z1*x2,
    w1*z2 + x1*y2 - y1*x2 + z1*w2,
  ], dim=-1)


def quaternion_normalize(q: Tensor) -> Tensor:
  """Normalize quaternion to unit length."""
  norm = (q * q).sum(axis=-1).sqrt()
  return q / norm.unsqueeze(-1)


def integrate_orientation(quat: Tensor, omega: Tensor, dt: float, project: bool = True) -> Tensor:
  """Integrate orientation quaternion: dq/dt = 0.5 * q ⊗ [0, ω]"""
  zero = omega[..., 0] * 0
  omega_quat = Tensor.stack([zero, omega[..., 0], omega[..., 1], omega[..., 2]], dim=-1)
  dq = quaternion_multiply(quat, omega_quat) * 0.5
  out = quat + dt * dq
  if project:
    out = quaternion_normalize(out)
  return out


class RigidBodySystem:
  """
  Complete rigid body simulation: Lie-Poisson dynamics (L) + quaternion kinematics (q).
  """
  def __init__(self, I: Tensor, integrator: str = "midpoint", policy: SymplecticPolicy | None = None):
    self.I = I
    self.I_inv = 1.0 / I
    self.H = lambda L: 0.5 * (L * L * self.I_inv).sum()
    if policy is None:
      from tinygrad.physics_profile import get_default_profile
      policy = get_default_profile().policy
    self.policy = policy
    if integrator == "auto":
      strict_drift = self.policy.drift_target is not None and self.policy.drift_target <= 1e-6
      integrator = "midpoint" if self.policy.accuracy in ("precise", "ultra_precise") or strict_drift else "splitting"
    self.integrator_name = integrator
    if integrator == "splitting":
      self._step_L = lambda L, dt: lie_poisson_splitting_so3(L, self.I_inv, dt)
    elif integrator in LiePoissonSystem.INTEGRATORS or integrator == "auto":
      self._lp = LiePoissonSystem(self.H, algebra="so3", integrator=integrator, policy=self.policy)
      self._step_L = lambda L, dt: self._lp.step(L, dt)
    else:
      raise ValueError(f"Unknown integrator: {integrator}")

  def step(self, L: Tensor, q: Tensor, dt: float = 0.01, project: bool = True) -> tuple[Tensor, Tensor]:
    L_new = self._step_L(L, dt)
    return L_new, integrate_orientation(q, L_new * self.I_inv, dt, project=project)

  def step_fused(self, L: Tensor, q: Tensor, dt: float = 0.01, project: bool = True) -> tuple[Tensor, Tensor]:
    L_new = self._step_L(L, dt)
    q_new = integrate_orientation(q, L_new * self.I_inv, dt, project=project)
    return L_new, q_new

  def energy(self, L: Tensor) -> float:
    return float(self.H(L).numpy())

  def casimir(self, L: Tensor) -> float:
    return float((L * L).sum().numpy())

  def evolve(self, L: Tensor, q: Tensor, dt: float, steps: int,
             record_every: int = 1, policy: SymplecticPolicy | None = None,
             scan: bool | None = None, unroll: int | None = None) -> tuple[Tensor, Tensor, list]:
    policy = policy or self.policy
    use_scan = policy.should_scan(steps, L.shape, L.device) if scan is None else scan
    op_cost = None
    if getenv("TINYGRAD_PHYSICS_UNROLL_COST", 1):
      try:
        op_cost = _count_uops_limit(self.H(L).uop, 512)
      except Exception:
        op_cost = None
    def make_step(scale: float):
      return lambda L_in, q_in: self.step_fused(L_in, q_in, dt * scale)
    def energy_fn(L_in: Tensor, q_in: Tensor):
      return float(self.H(L_in).numpy())
    dt_scale, _ = policy.adjust_dt(steps, L.shape, L.device, (L, q), make_step, energy_fn, "TINYGRAD_PHYSICS_DRIFT_RIGID")
    if dt_scale != 1.0:
      dt *= dt_scale
    vec = policy.choose_vector_width(L.shape, L.device, 3)
    L_use, L_shape = _reshape_for_vec(L, vec, 3)
    q_use, q_shape = _reshape_for_vec(q, vec, 4)
    if q_shape is None:
      L_use, L_shape = L, None
      q_use, q_shape = q, None
    if unroll is None and use_scan:
      unroll = policy.choose_unroll(steps, L_use.shape, L_use.device, op_cost=op_cost)
    def constraint_fn(state: tuple[Tensor, Tensor]):
      return (state[1] * state[1]).sum(axis=-1).mean()
    def step_fn(state: tuple[Tensor, Tensor], i: int, stride: int):
      L_i, q_i = state
      project = (i + 1) % stride == 0
      return self.step_fused(L_i, q_i, dt, project=project)
    project_every = policy.choose_projection_stride(
      steps, L_use.shape, L_use.device, "unit_norm_q", (L_use, q_use), step_fn, constraint_fn)
    if unroll is not None:
      if record_every % unroll != 0:
        record_every = unroll
      out = self.evolve_unrolled(L_use, q_use, dt, steps, unroll, record_every=record_every, project_every=project_every)
      policy.maybe_print_report(steps, L.shape, L.device)
      L_out, q_out, history = out
      if L_shape is not None:
        history = [(L_t.reshape(L_shape), q_t.reshape(q_shape), e, c) for (L_t, q_t, e, c) in history]
      return _reshape_from_vec(L_out, L_shape), _reshape_from_vec(q_out, q_shape), history
    history = []
    for i in range(steps):
      project = (i + 1) % project_every == 0
      if i % record_every == 0:
        L_emit = _reshape_from_vec(L_use, L_shape)
        q_emit = _reshape_from_vec(q_use, q_shape)
        history.append((L_emit.numpy().copy(), q_emit.numpy().copy(), self.energy(L_emit), self.casimir(L_emit)))
      L_use, q_use = self.step_fused(L_use, q_use, dt, project=project)
    L_emit = _reshape_from_vec(L_use, L_shape)
    q_emit = _reshape_from_vec(q_use, q_shape)
    history.append((L_emit.numpy().copy(), q_emit.numpy().copy(), self.energy(L_emit), self.casimir(L_emit)))
    policy.maybe_print_report(steps, L.shape, L.device)
    return _reshape_from_vec(L_use, L_shape), _reshape_from_vec(q_use, q_shape), history

  def compile_unrolled_step(self, dt: float, unroll: int, project_every: int = 1):
    """Compile an unrolled rigid-body step."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if project_every < 1:
      raise ValueError("project_every must be >= 1")

    def unrolled_step(L: Tensor, q: Tensor):
      for _ in range(unroll):
        project = (_ + 1) % project_every == 0
        L, q = self.step_fused(L, q, dt, project=project)
      return L, q

    return TinyJit(unrolled_step)

  def evolve_unrolled(self, L: Tensor, q: Tensor, dt: float, steps: int, unroll: int,
                      record_every: int = 1, project_every: int = 1) -> tuple[Tensor, Tensor, list]:
    """Evolve using an unrolled compiled step to reduce Python overhead."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if steps % unroll != 0:
      raise ValueError("steps must be divisible by unroll")
    if record_every % unroll != 0:
      raise ValueError("record_every must be divisible by unroll")

    step = self.compile_unrolled_step(dt, unroll, project_every=project_every)
    history = []
    for i in range(0, steps, unroll):
      if i % record_every == 0:
        history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
      L, q = step(L, q)
    history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
    return L, q, history


class ProductManifold:
  """
  Product Manifold SO(3)* × S².
  State = (L, γ) where L is angular momentum and γ is the body-frame gravity direction.
  """
  def __init__(self, L: Tensor, gamma: Tensor):
    self.L = L
    self.gamma = gamma

  def normalize_gamma(self) -> 'ProductManifold':
    norm = (self.gamma * self.gamma).sum(axis=-1).sqrt()
    return ProductManifold(self.L, self.gamma / norm.unsqueeze(-1))

  def realize(self) -> 'ProductManifold':
    self.L.realize()
    self.gamma.realize()
    return self

  @staticmethod
  def from_euler_angles(L: Tensor, theta: float, phi: float, dtype=dtypes.float64) -> 'ProductManifold':
    base = Tensor([
      math.sin(theta) * math.cos(phi),
      math.sin(theta) * math.sin(phi),
      math.cos(theta),
    ], dtype=dtype)
    if L.ndim > 1:
      base = base.reshape((1,) * (L.ndim - 1) + (3,)).expand(*L.shape[:-1], 3).contiguous()
    gamma = base
    return ProductManifold(L, gamma)

  def casimirs(self) -> tuple[Tensor, Tensor]:
    C1 = (self.gamma * self.gamma).sum(axis=-1)
    C2 = (self.L * self.gamma).sum(axis=-1)
    return C1, C2


class ControlInput:
  """
  External control input for non-Hamiltonian forcing terms.
  """
  def __init__(self, fn: Callable):
    self.fn = fn

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)


class SatelliteControlIntegrator:
  """
  Symplectic split control integrator for rigid-body attitude control.
  """
  def __init__(self, I_inv: Tensor, control: ControlInput, dt: float, policy: SymplecticPolicy | None = None):
    self.I_inv = I_inv
    self.control = control
    self.dt = dt
    if policy is None:
      from tinygrad.physics_profile import get_default_profile
      policy = get_default_profile().policy
    self.policy = policy

  def step(self, L: Tensor, quat: Tensor, project: bool = True) -> tuple[Tensor, Tensor]:
    omega = L * self.I_inv
    q_err = (quat[..., 0:1] < 0).where(-quat, quat)
    u = self.control(q_err, omega)
    L_half = L + 0.5 * self.dt * u
    L_free = lie_poisson_splitting_so3(L_half, self.I_inv, self.dt)
    omega_free = L_free * self.I_inv
    quat_free = integrate_orientation(quat, omega_free, self.dt, project=project)
    q_err2 = (quat_free[..., 0:1] < 0).where(-quat_free, quat_free)
    u2 = self.control(q_err2, omega_free)
    L_next = L_free + 0.5 * self.dt * u2
    return L_next, quat_free

  def compile_unrolled_step(self, unroll: int, project_every: int = 1):
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if project_every < 1:
      raise ValueError("project_every must be >= 1")

    def unrolled_step(L: Tensor, quat: Tensor):
      for _ in range(unroll):
        project = (_ + 1) % project_every == 0
        L, quat = self.step(L, quat, project=project)
      return L, quat

    return TinyJit(unrolled_step)

  def evolve_unrolled(self, L: Tensor, quat: Tensor, steps: int, unroll: int,
                      record_every: int = 1, project_every: int = 1) -> tuple[Tensor, Tensor, list]:
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if steps % unroll != 0:
      raise ValueError("steps must be divisible by unroll")
    if record_every % unroll != 0:
      raise ValueError("record_every must be divisible by unroll")

    step = self.compile_unrolled_step(unroll, project_every=project_every)
    history = []
    for i in range(0, steps, unroll):
      if i % record_every == 0:
        history.append((L.detach(), quat.detach()))
      L, quat = step(L, quat)
    history.append((L.detach(), quat.detach()))
    return L, quat, history

  def evolve(self, L: Tensor, quat: Tensor, steps: int, record_every: int = 1,
             policy: SymplecticPolicy | None = None, scan: bool | None = None,
             unroll: int | None = None) -> tuple[Tensor, Tensor, list]:
    policy = policy or self.policy
    use_scan = policy.should_scan(steps, L.shape, L.device) if scan is None else scan
    vec = policy.choose_vector_width(L.shape, L.device, 3)
    L_use, L_shape = _reshape_for_vec(L, vec, 3)
    q_use, q_shape = _reshape_for_vec(quat, vec, 4)
    if q_shape is None:
      L_use, L_shape = L, None
      q_use, q_shape = quat, None
    op_cost = None
    if getenv("TINYGRAD_PHYSICS_UNROLL_COST", 1):
      try:
        op_cost = _count_uops_limit((L_use * L_use).sum().uop, 512)
      except Exception:
        op_cost = None
    if unroll is None and use_scan:
      unroll = policy.choose_unroll(steps, L_use.shape, L_use.device, candidates=[5, 10, 20], op_cost=op_cost)
    def constraint_fn(state: tuple[Tensor, Tensor]):
      return (state[1] * state[1]).sum(axis=-1).mean()
    def step_fn(state: tuple[Tensor, Tensor], i: int, stride: int):
      L_i, q_i = state
      project = (i + 1) % stride == 0
      return self.step(L_i, q_i, project=project)
    project_every = policy.choose_projection_stride(
      steps, L_use.shape, L_use.device, "unit_norm_q", (L_use, q_use), step_fn, constraint_fn)
    if unroll is not None:
      if record_every % unroll != 0:
        record_every = unroll
      out = self.evolve_unrolled(L_use, q_use, steps, unroll, record_every=record_every, project_every=project_every)
      policy.maybe_print_report(steps, L.shape, L.device)
      L_out, q_out, history = out
      if L_shape is not None:
        history = [(_reshape_from_vec(L_t, L_shape), _reshape_from_vec(q_t, q_shape)) for (L_t, q_t) in history]
      return _reshape_from_vec(L_out, L_shape), _reshape_from_vec(q_out, q_shape), history
    history = []
    for i in range(steps):
      project = (i + 1) % project_every == 0
      if i % record_every == 0:
        L_emit = _reshape_from_vec(L_use, L_shape)
        q_emit = _reshape_from_vec(q_use, q_shape)
        history.append((L_emit.detach(), q_emit.detach()))
      L_use, q_use = self.step(L_use, q_use, project=project)
    L_emit = _reshape_from_vec(L_use, L_shape)
    q_emit = _reshape_from_vec(q_use, q_shape)
    history.append((L_emit.detach(), q_emit.detach()))
    policy.maybe_print_report(steps, L.shape, L.device)
    return _reshape_from_vec(L_use, L_shape), _reshape_from_vec(q_use, q_shape), history


class LiePoissonBracketE3:
  """
  Lie-Poisson structure for e(3)* (heavy top).
  """
  def flow(self, state: ProductManifold, grad_L: Tensor, grad_gamma: Tensor) -> tuple[Tensor, Tensor]:
    dL_dt = cross(state.L, grad_L) + cross(state.gamma, grad_gamma)
    dgamma_dt = cross(state.gamma, grad_L)
    return dL_dt, dgamma_dt


class HeavyTopHamiltonian:
  """
  Hamiltonian for the heavy top.
  H(L, γ) = ½ L · (I⁻¹ · L) + mgl γ₃
  """
  def __init__(self, I1: float, I2: float, I3: float, mgl: float, dtype=dtypes.float64):
    self.I_inv = Tensor([1.0/I1, 1.0/I2, 1.0/I3], dtype=dtype)
    self.mgl = mgl
    self.dtype = dtype

  def __call__(self, state: ProductManifold) -> Tensor:
    omega = self.I_inv * state.L
    T = 0.5 * (state.L * omega).sum()
    V = self.mgl * state.gamma[2]
    return T + V

  def angular_velocity(self, L: Tensor) -> Tensor:
    return self.I_inv * L


class HeavyTopIntegrator:
  """
  Symplectic integrator for the heavy top.
  """
  def __init__(self, hamiltonian: HeavyTopHamiltonian, dt: float = 0.001, policy: SymplecticPolicy | None = None):
    self.H = hamiltonian
    self.dt = dt
    self.bracket = LiePoissonBracketE3()
    self._eps = Tensor([1e-10], dtype=self.H.dtype)
    self._e3 = Tensor([0.0, 0.0, 1.0], dtype=self.H.dtype)
    self._mgl_e3 = self._e3 * self.H.mgl
    if policy is None:
      from tinygrad.physics_profile import get_default_profile
      policy = get_default_profile().policy
    self.policy = policy

  def _rotate_vector(self, v: Tensor, axis: Tensor, angle: Tensor) -> Tensor:
    axis_norm = (axis * axis).sum().sqrt()
    k = axis / (axis_norm + self._eps)
    c = angle.cos()
    s = angle.sin()
    if c.ndim < v.ndim:
      c = c.unsqueeze(-1)
      s = s.unsqueeze(-1)
    k_cross_v = cross(k, v)
    k_dot_v = (k * v).sum(axis=-1)
    if k_dot_v.ndim < v.ndim:
      k_dot_v = k_dot_v.unsqueeze(-1)
    return v * c + k_cross_v * s + k * k_dot_v * (1 - c)

  def _derivatives(self, L: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    omega = self.H.I_inv * L
    dL = cross(L, omega) + cross(gamma, self._e3) * self.H.mgl
    dgamma = cross(gamma, omega)
    return dL, dgamma

  def step_rk4(self, state: ProductManifold) -> ProductManifold:
    L, gamma = state.L, state.gamma
    dt = self.dt
    dL1, dg1 = self._derivatives(L, gamma)
    L2 = L + dL1 * (dt/2)
    g2 = gamma + dg1 * (dt/2)
    dL2, dg2 = self._derivatives(L2, g2)
    L3 = L + dL2 * (dt/2)
    g3 = gamma + dg2 * (dt/2)
    dL3, dg3 = self._derivatives(L3, g3)
    L4 = L + dL3 * dt
    g4 = gamma + dg3 * dt
    dL4, dg4 = self._derivatives(L4, g4)
    L_new = L + (dL1 + dL2 * 2 + dL3 * 2 + dL4) * (dt/6)
    gamma_new = gamma + (dg1 + dg2 * 2 + dg3 * 2 + dg4) * (dt/6)
    return ProductManifold(L_new, gamma_new).normalize_gamma().realize()

  def step_rk4_tensors(self, L: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    dt = self.dt
    dL1, dg1 = self._derivatives(L, gamma)
    L2 = L + dL1 * (dt/2)
    g2 = gamma + dg1 * (dt/2)
    dL2, dg2 = self._derivatives(L2, g2)
    L3 = L + dL2 * (dt/2)
    g3 = gamma + dg2 * (dt/2)
    dL3, dg3 = self._derivatives(L3, g3)
    L4 = L + dL3 * dt
    g4 = gamma + dg3 * dt
    dL4, dg4 = self._derivatives(L4, g4)
    L_new = L + (dL1 + dL2 * 2 + dL3 * 2 + dL4) * (dt/6)
    gamma_new = gamma + (dg1 + dg2 * 2 + dg3 * 2 + dg4) * (dt/6)
    state = ProductManifold(L_new, gamma_new).normalize_gamma()
    return state.L, state.gamma

  def step_symmetric(self, state: ProductManifold, project: bool = True) -> ProductManifold:
    L, gamma = state.L, state.gamma
    dt = self.dt
    omega3 = self.H.I_inv[2] * L[..., 2]
    angle = omega3 * (dt/2)
    e3 = Tensor([0.0, 0.0, 1.0], dtype=self.H.dtype)
    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)
    dL, dgamma = self._derivatives(L, gamma)
    L_mid = L + dL * (dt/2)
    gamma_mid = gamma + dgamma * (dt/2)
    dL_mid, dgamma_mid = self._derivatives(L_mid, gamma_mid)
    L = L + dL_mid * dt
    gamma = gamma + dgamma_mid * dt
    omega3 = self.H.I_inv[2] * L[..., 2]
    angle = omega3 * (dt/2)
    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)
    state = ProductManifold(L, gamma)
    if project:
      state = state.normalize_gamma()
    return state.realize()

  def step_symmetric_tensors(self, L: Tensor, gamma: Tensor, project: bool = True) -> tuple[Tensor, Tensor]:
    dt = self.dt
    omega3 = self.H.I_inv[2] * L[..., 2]
    angle = omega3 * (dt/2)
    e3 = Tensor([0.0, 0.0, 1.0], dtype=self.H.dtype)
    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)
    dL, dgamma = self._derivatives(L, gamma)
    L_mid = L + dL * (dt/2)
    gamma_mid = gamma + dgamma * (dt/2)
    dL_mid, dgamma_mid = self._derivatives(L_mid, gamma_mid)
    L = L + dL_mid * dt
    gamma = gamma + dgamma_mid * dt
    omega3 = self.H.I_inv[2] * L[..., 2]
    angle = omega3 * (dt/2)
    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)
    state = ProductManifold(L, gamma)
    if project:
      state = state.normalize_gamma()
    return state.L, state.gamma

  def step_fused_tensors(self, L: Tensor, gamma: Tensor, project: bool = True) -> tuple[Tensor, Tensor]:
    return self.step_symmetric_tensors(L, gamma, project=project)

  def step(self, state: ProductManifold, project: bool = True) -> ProductManifold:
    return self.step_symmetric(state, project=project)

  def step_tensors(self, L: Tensor, gamma: Tensor, project: bool = True) -> tuple[Tensor, Tensor]:
    return self.step_symmetric_tensors(L, gamma, project=project)

  def step_explicit(self, state: ProductManifold, project: bool = True) -> ProductManifold:
    omega = self.H.angular_velocity(state.L)
    grad_gamma = self._mgl_e3
    dL_dt, dgamma_dt = self.bracket.flow(state, omega, grad_gamma)
    L_new = state.L + dL_dt * self.dt
    gamma_new = state.gamma + dgamma_dt * self.dt
    state = ProductManifold(L_new, gamma_new)
    if project:
      state = state.normalize_gamma()
    return state.realize()

  def step_explicit_tensors(self, L: Tensor, gamma: Tensor, project: bool = True) -> tuple[Tensor, Tensor]:
    omega = self.H.angular_velocity(L)
    grad_gamma = self._mgl_e3
    dL_dt, dgamma_dt = self.bracket.flow(ProductManifold(L, gamma), omega, grad_gamma)
    L_new = L + dL_dt * self.dt
    gamma_new = gamma + dgamma_dt * self.dt
    state = ProductManifold(L_new, gamma_new)
    if project:
      state = state.normalize_gamma()
    return state.L, state.gamma

  def compile_unrolled_step(self, unroll: int, method: str = "splitting", project_every: int = 1):
    """Compile an unrolled heavy-top step."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if project_every < 1:
      raise ValueError("project_every must be >= 1")
    if method == "rk4":
      raise ValueError("rk4 is not symplectic; use method='splitting'")
    if method == "splitting":
      step_fn = self.step_symmetric_tensors
    else:
      step_fn = self.step_explicit_tensors

    def unrolled_step(L: Tensor, gamma: Tensor):
      for _ in range(unroll):
        project = (_ + 1) % project_every == 0
        L, gamma = step_fn(L, gamma, project=project)
      return L, gamma

    return TinyJit(unrolled_step)

  def evolve_unrolled(self, L: Tensor, gamma: Tensor, steps: int, unroll: int,
                      method: str = "splitting", record_every: int = 1,
                      project_every: int = 1) -> tuple[Tensor, Tensor, list]:
    """Evolve heavy-top dynamics using unrolled compiled steps."""
    if unroll < 1:
      raise ValueError("unroll must be >= 1")
    if steps % unroll != 0:
      raise ValueError("steps must be divisible by unroll")
    if record_every % unroll != 0:
      raise ValueError("record_every must be divisible by unroll")

    step = self.compile_unrolled_step(unroll, method=method, project_every=project_every)
    history = []
    for i in range(0, steps, unroll):
      if i % record_every == 0:
        history.append((L.detach(), gamma.detach()))
      L, gamma = step(L, gamma)
    history.append((L.detach(), gamma.detach()))
    return L, gamma, history

  def evolve(self, L: Tensor, gamma: Tensor, steps: int, method: str = "splitting",
             record_every: int = 1, policy: SymplecticPolicy | None = None,
             scan: bool | None = None, unroll: int | None = None) -> tuple[Tensor, Tensor, list]:
    policy = policy or self.policy
    use_scan = policy.should_scan(steps, L.shape, L.device) if scan is None else scan
    base_dt = self.dt
    op_cost = None
    if getenv("TINYGRAD_PHYSICS_UNROLL_COST", 1):
      try:
        op_cost = _count_uops_limit(self.H(ProductManifold(L, gamma)).uop, 512)
      except Exception:
        op_cost = None
    vec = policy.choose_vector_width(L.shape, L.device, 3)
    L_use, L_shape = _reshape_for_vec(L, vec, 3)
    g_use, g_shape = _reshape_for_vec(gamma, vec, 3)
    if g_shape is None:
      L_use, L_shape = L, None
      g_use, g_shape = gamma, None
    def make_step(scale: float):
      def step_fn(L_in: Tensor, g_in: Tensor):
        self.dt = base_dt * scale
        if method == "splitting":
          return self.step_symmetric_tensors(L_in, g_in)
        return self.step_explicit_tensors(L_in, g_in)
      return step_fn
    def energy_fn(L_in: Tensor, g_in: Tensor):
      return float(self.H(ProductManifold(L_in, g_in)).numpy())
    dt_scale, _ = policy.adjust_dt(steps, L.shape, L.device, (L, gamma), make_step, energy_fn, "TINYGRAD_PHYSICS_DRIFT_HEAVY")
    self.dt = base_dt * dt_scale
    if unroll is None and use_scan:
      unroll = policy.choose_unroll(steps, L_use.shape, L_use.device, op_cost=op_cost)
    def constraint_fn(state: tuple[Tensor, Tensor]):
      return (state[1] * state[1]).sum(axis=-1).mean()
    def step_fn(state: tuple[Tensor, Tensor], i: int, stride: int):
      L_i, g_i = state
      project = (i + 1) % stride == 0
      if method == "splitting":
        return self.step_tensors(L_i, g_i, project=project)
      return self.step_explicit_tensors(L_i, g_i, project=project)
    project_every = policy.choose_projection_stride(
      steps, L_use.shape, L_use.device, "unit_norm_gamma", (L_use, g_use), step_fn, constraint_fn)
    if unroll is not None:
      if record_every % unroll != 0:
        record_every = unroll
      out = self.evolve_unrolled(
        L_use, g_use, steps, unroll, method=method, record_every=record_every, project_every=project_every)
      policy.maybe_print_report(steps, L.shape, L.device)
      L_out, g_out, history = out
      if L_shape is not None:
        history = [(_reshape_from_vec(L_t, L_shape), _reshape_from_vec(g_t, g_shape)) for (L_t, g_t) in history]
      return _reshape_from_vec(L_out, L_shape), _reshape_from_vec(g_out, g_shape), history
    history = []
    for i in range(steps):
      project = (i + 1) % project_every == 0
      if i % record_every == 0:
        L_emit = _reshape_from_vec(L_use, L_shape)
        g_emit = _reshape_from_vec(g_use, g_shape)
        history.append((L_emit.detach(), g_emit.detach()))
      if method == "splitting":
        L_use, g_use = self.step_tensors(L_use, g_use, project=project)
      else:
        L_use, g_use = self.step_explicit_tensors(L_use, g_use, project=project)
    L_emit = _reshape_from_vec(L_use, L_shape)
    g_emit = _reshape_from_vec(g_use, g_shape)
    history.append((L_emit.detach(), g_emit.detach()))
    policy.maybe_print_report(steps, L.shape, L.device)
    return _reshape_from_vec(L_use, L_shape), _reshape_from_vec(g_use, g_shape), history
