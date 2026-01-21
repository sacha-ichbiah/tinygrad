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

from tinygrad.engine.jit import TinyJit, JitError
from tinygrad.engine.realize import capturing
from tinygrad.helpers import getenv
from tinygrad.codegen.opt import Opt, OptOps
from tinygrad.dtype import dtypes, AddrSpace
from tinygrad.tensor import Tensor
from tinygrad.uop.ops import AxisType, GroupOp, KernelInfo, Ops, UOp, PatternMatcher, UPat, graph_rewrite, resolve
from tinygrad.uop.symbolic import symbolic
from typing import Callable
import time

# ============================================================================
# CORE: HAMILTONIAN MECHANICS VIA AUTOGRAD
# ============================================================================

_grad_H_jit_cache: dict[tuple, TinyJit] = {}


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
    reduced = scalar.broadcast(reduced.dtype.count)
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
    shape = x.shape
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
        return None
    ranges = ctx["ranges"]
    if any(r.op is not Ops.RANGE for r in ranges): return None
    if shape is None or len(shape) != len(ranges): return None
    idx = tuple(UOp.const(dtypes.index, 0) if resolve(s == 1) else ranges[i] for i, s in enumerate(shape))
    return base.src[0].vindex(*idx, dtype=x.dtype)


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

def _drop_empty_reshape(ctx: dict, x: UOp) -> UOp|None:
  if x.op not in (Ops.RESHAPE, Ops.EXPAND): return None
  try:
    shape = x.shape
  except Exception:
    return None
  if shape is None or len(shape) != 0: return None
  return x.src[0]

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
    q_next = q + dt * dHdp
    p_next = p - dt * dHdq

    for _ in range(max_iter):
        q_mid = 0.5 * (q + q_next)
        p_mid = 0.5 * (p + p_next)
        dHdq_mid, dHdp_mid = _grad_H(q_mid, p_mid, H_func)
        q_new = q + dt * dHdp_mid
        p_new = p - dt * dHdq_mid
        diff_q = (q_new - q_next).abs().max().numpy()
        diff_p = (p_new - p_next).abs().max().numpy()
        if diff_q < tol and diff_p < tol:
            break
        q_next, p_next = q_new, p_new

    return q_next.realize(), p_next.realize()


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

    def __init__(self, H_func, integrator: str = "leapfrog"):
        self.H = H_func
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step = self.INTEGRATORS[integrator]
        self.integrator_name = integrator
        self._jit_step = TinyJit(self.step)
        self._jit_step_inplace = TinyJit(self.step_inplace)
        self._scan_kernel_cache: dict[tuple[float, int, int, int, str, tuple[int, ...], object], Callable] = {}
        self._scan_kernel_coupled_cache: dict[tuple[float, int, int, int, str, tuple[int, ...], object], Callable] = {}
        self._step_kernel_coupled_cache: dict[tuple[float, str, tuple[int, ...], object], Callable] = {}
        self._grad_kernel_coupled_cache: dict[tuple[str, tuple[int, ...], object], Callable] = {}
        self._update_kernel_coupled_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._update_kernel_coupled_p_cache: dict[tuple[float, str, tuple[int, ...], object, int], Callable] = {}
        self._scan_kernel_coupled_tune_cache: dict[tuple[float, int, str, tuple[int, ...], object], int] = {}
        self._reduce_kernel_coupled_cache: dict[tuple[UOp, str, tuple[int, ...], object], Callable] = {}
        self._reduce_kernel_coupled_multi_cache: dict[tuple[tuple[UOp, ...], str, tuple[int, ...], object, int], Callable] = {}
        self._reduce_kernel_coupled_qnew_cache: dict[tuple[tuple[UOp, ...], object, str, tuple[int, ...], object, float, int], Callable] = {}
        self._reduce_kernel_coupled_tune_cache: dict[tuple[str, tuple[int, ...], object], int] = {}
        self._reduce_kernel_coupled_unroll_tune_cache: dict[tuple[str, tuple[int, ...], object, int], int] = {}
        self._reduce_acc_buf_cache_dHdq: dict[tuple[str, object, int], list[Tensor]] = {}
        self._reduce_acc_buf_cache_dHdp: dict[tuple[str, object, int], list[Tensor]] = {}
        self._scan_tmp_buf_cache: dict[tuple[str, object, tuple[int, ...]], tuple[Tensor, Tensor]] = {}
        self._elem_kernel_coupled_cache: dict[tuple[object, str, tuple[int, ...], object, str, float], Callable] = {}
        self._grad_uop_cache: dict[tuple[object, str, tuple[int, ...], object], tuple[UOp, UOp, UOp, UOp]] = {}

    def step(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        return self._step(q, p, self.H, dt)

    def step_inplace(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        """In-place step for scan loops; updates q and p buffers via assign."""
        q_new, p_new = self.step(q, p, dt)
        q.assign(q_new)
        p.assign(p_new)
        return q, p

    def energy(self, q: Tensor, p: Tensor) -> float:
        return float(self.H(q, p).numpy())

    def evolve(self, q: Tensor, p: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
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

        return q, p, history

    def evolve_scan(self, q: Tensor, p: Tensor, dt: float, steps: int, scan: int = 8,
                    record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        """Higher-level scan API: unrolls multiple steps while preserving record cadence."""
        if scan <= 1:
            return self.evolve(q, p, dt, steps, record_every=record_every)
        if q.requires_grad or p.requires_grad:
            return self.evolve(q, p, dt, steps, record_every=record_every)
        if self.integrator_name == "leapfrog" and scan >= steps and record_every >= steps and q.shape == p.shape:
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
        if self.integrator_name != "leapfrog":
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
        q = q.contiguous().realize()
        p = p.contiguous().realize()

        if q.device != p.device:
            raise ValueError("scan kernel requires q and p on the same device")

        key = (dt, steps, unroll_steps, vector_width, q.device, q.shape, q.dtype)
        kernel = self._scan_kernel_cache.get(key)
        if kernel is None:
            kernel = self._build_leapfrog_scan_kernel(dt, steps, unroll_steps, vector_width, q.device, q.shape, q.dtype)
            self._scan_kernel_cache[key] = kernel

        q_start = q.detach()
        p_start = p.detach()
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
        if self.integrator_name != "leapfrog":
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
        if self.integrator_name != "leapfrog":
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
        if not double_buffer:
            if vector_width > 1 and not getenv("TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL", 0):
                q_out, p_out = self._evolve_coupled_two_phase_vec(q, p, dt, steps, vector_width)
            else:
                q_out, p_out = self._evolve_scan_kernel_coupled_split(
                    q, p, dt, steps, unroll_steps=unroll_steps, vector_width=vector_width)
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
                            q_out, p_out = self._evolve_coupled_two_phase_vec(q, p, dt, steps, vector_width)
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
            store_dHdq = dHdq_out.index(*ranges, ptr=True).store(dHdq.vindex(*ranges))
            store_dHdp = dHdp_out.index(*ranges, ptr=True).store(dHdp.vindex(*ranges))
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

            op_kind = reduce_uops[0].arg[0]
            all_same_op = all(red.arg[0] == op_kind for red in reduce_uops)
            any_nested_reduce = any(u.op is Ops.REDUCE_AXIS for red in reduce_uops for u in red.src[0].toposort())
            if all_same_op and not any_nested_reduce and len(reduce_uops) > 1:
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
                exprs = []
                for red in reduce_uops:
                    expr = red.src[0].substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    expr = rewrite_elem(expr, reduce_ranges)
                    if expr.dtype.count > 1:
                        scalar = expr.gep(0)
                        for i in range(1, expr.dtype.count):
                            scalar = scalar + expr.gep(i)
                        expr = scalar
                    exprs.append(expr)
                vec_expr = exprs[0].vectorize(*exprs[1:]) if len(exprs) > 1 else exprs[0]
                red = UOp(Ops.REDUCE, vec_expr.dtype, src=(vec_expr,)+tuple(reduce_ranges), arg=op_kind)
                idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(reduce_ranges)))
                red_val = red.vindex(*idx)
                reduce_ranges_used = [r for r in red.ranges if r.arg[-1] == AxisType.REDUCE]
                opts = ()
                if reduce_unroll > 1 and reduce_ranges_used:
                    opts = (Opt(OptOps.UNROLL, reduce_ranges_used[-1].arg[0], reduce_unroll),)
                stores = []
                for i, out in enumerate(outs):
                    stores.append(out.index(UOp.const(dtypes.index, 0), ptr=True).store(red_val.gep(i)).end(*reduce_ranges_used))
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
                    scalar = reduced.gep(0)
                    for i in range(1, reduced.dtype.count):
                        scalar = scalar + reduced.gep(i)
                    reduced = scalar.broadcast(reduced.dtype.count)
                idx = tuple(UOp.const(dtypes.index, 0) for _ in range(len(ranges)))
                return reduced.vindex(*idx)

            stores = []
            opts = ()
            for reduce_uop, out in zip(reduce_uops, outs, strict=True):
                red = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
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
                half_dt = UOp.const(expr.dtype, 0.5 * dt, shape=shape_vals)
                neg_one = UOp.const(expr.dtype, -1.0, shape=shape_vals)
                out_val = p_val + (half_dt * expr) * neg_one
            else:
                dt_uop = UOp.const(expr.dtype, dt, shape=shape_vals)
                out_val = q_val + dt_uop * expr
            store = out.index(*ranges, ptr=True).store(out_val)
            return store.end(*ranges).sink(arg=KernelInfo(name=f"coupled_{op_name}", opts_to_apply=()))

        return kernel

    def _build_coupled_elem_kernel_qp_new(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                          dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                          placeholders_dHdq: list[UOp]) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)

        def kernel(q: UOp, p: UOp, acc0: UOp, acc1: UOp, q_out: UOp, p_out: UOp) -> UOp:
            ranges = [UOp.range(s, i+1) for i, s in enumerate(shape)]
            def rewrite_elem(uop: UOp) -> UOp:
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": q.device},
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
                    ctx={"ranges": ranges, "device": q.device},
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
                return uop
            q_val = q.vindex(*ranges)
            p_val = p.vindex(*ranges)
            acc0_val = acc0.vindex(UOp.const(dtypes.index, 0))
            acc1_val = acc1.vindex(UOp.const(dtypes.index, 0))
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
            dHdq_2 = dHdq_elem_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
            if placeholders_dHdq:
                acc_loads = {placeholders_dHdq[i]: acc1_val for i in range(len(placeholders_dHdq))}
                dHdq_2 = dHdq_2.substitute(acc_loads, name="reduce_placeholders")
            dHdq_2 = rewrite_elem(dHdq_2)
            p_new = p_half + (half_dt * dHdq_2) * neg_one
            store_q = q_out.index(*ranges, ptr=True).store(q_new)
            store_p = p_out.index(*ranges, ptr=True).store(p_new)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="coupled_qp_new", opts_to_apply=()))

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
            def rewrite_elem(uop: UOp) -> UOp:
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges, "device": q.device},
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
                    ctx={"ranges": ranges, "device": q.device},
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
            half_dt = UOp.const(dHdq_2.dtype, 0.5 * dt, shape=shape_vals)
            neg_one = UOp.const(dHdq_2.dtype, -1.0, shape=shape_vals)
            p_new = p_half_val + (half_dt * dHdq_2) * neg_one
            store_q = q_out.index(*ranges, ptr=True).store(q_new)
            store_p = p_out.index(*ranges, ptr=True).store(p_new)
            return UOp.group(store_q, store_p).end(*ranges).sink(arg=KernelInfo(name="coupled_qp_update", opts_to_apply=()))

        return kernel

    def _build_coupled_kernel_qp_new_with_reduce(self, dt: float, device: str, shape: tuple[int, ...], dtype,
                                                 dHdq_elem_uop: UOp, dHdp_elem_uop: UOp,
                                                 reduce_uops: list[UOp], placeholders_dHdq: list[UOp],
                                                 unroll_steps: int = 1, vector_width: int = 1) -> Callable:
        q_sym_uop, p_sym_uop, _, _ = self._get_coupled_grad_uops(device, shape, dtype)
        if len(reduce_uops) == 0:
            raise ValueError("reduce_uops must be non-empty")
        for reduce_uop in reduce_uops:
            axis = reduce_uop.arg[1]
            if len(axis) != len(shape) or not all(i in axis for i in range(len(shape))):
                raise ValueError("split reduce kernel only supports full reductions")
        if unroll_steps < 1:
            raise ValueError("unroll_steps must be >= 1")
        if vector_width < 1:
            raise ValueError("vector_width must be >= 1")
        if vector_width > 1 and shape and shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")

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
                for i in range(len(reduce_uops))
            ]
            acc1_regs = [
                UOp(Ops.DEFINE_REG, acc_dtype.ptr(size=unroll_steps, addrspace=AddrSpace.REG), arg=i+len(reduce_uops))
                for i in range(len(reduce_uops))
            ]
            def rewrite_elem(uop: UOp, ranges_override: list[UOp]|None = None) -> UOp:
                ranges_use = ranges_override if ranges_override is not None else ranges
                if any(r.op is not Ops.RANGE for r in ranges_use):
                    uop = graph_rewrite(
                        uop,
                        PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _drop_const_expand_any)], compiled=False),
                        ctx={},
                        name="drop_const_expand_any",
                    )
                uop = graph_rewrite(
                    uop,
                    PatternMatcher([(UPat((Ops.RESHAPE, Ops.EXPAND), name="x"), _const_to_vindex)], compiled=False),
                    ctx={"ranges": ranges_use},
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
                    PatternMatcher([(UPat((Ops.CONST, Ops.VCONST), name="c"), _strip_device_const)], compiled=False),
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
                    if target.shape is None: return uop
                except Exception:
                    return uop
                try:
                    if uop.shape != target.shape: return uop._broadcast_to(target.shape)
                except Exception:
                    pass
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
                q_lanes = [q_val.gep(i) for i in range(vector_width)]
                p_lanes = [p_val.gep(i) for i in range(vector_width)]
            else:
                q_val = q.vindex(*ranges)
                p_val = p.vindex(*ranges)
            shape_vals = None
            const_dtype = q_lanes[0].dtype if use_vec else q.dtype
            half_dt = UOp.const(const_dtype, 0.5 * dt, shape=shape_vals)
            neg_one = UOp.const(const_dtype, -1.0, shape=shape_vals)
            dt_uop = UOp.const(const_dtype, dt, shape=shape_vals)
            for step in range(unroll_steps):
                q_red = None
                p_red = None
                if use_vec:
                    q_red = q_lanes[0]
                    p_red = p_lanes[0]
                    for j in range(1, vector_width):
                        q_red = q_red + q_lanes[j]
                        p_red = p_red + p_lanes[j]
                acc0_vals = []
                acc0_idx = UOp.const(dtypes.index, step)
                for i, reduce_uop in enumerate(reduce_uops):
                    if use_vec:
                        red0 = reduce_uop.substitute({q_sym_uop: q_red, p_sym_uop: p_red})
                    else:
                        red0 = reduce_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    red0 = graph_rewrite(
                        red0,
                        PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                        ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                        name="lower_reduce_axis",
                    )
                    reduce_ranges0 = [r for r in red0.ranges if r.arg[-1] == AxisType.REDUCE]
                    acc0_store = acc0_regs[i].index(acc0_idx).store(red0).end(*reduce_ranges0)
                    acc0_val = acc0_regs[i].after(acc0_store).index(acc0_idx)
                    if not use_vec and shape_vals is not None:
                        try:
                            if acc0_val.shape != shape_vals:
                                acc0_val = acc0_val._broadcast_to(shape_vals)
                        except Exception:
                            pass
                    acc0_vals.append(acc0_val)
                if use_vec:
                    p_half_lanes = []
                    q_new_lanes = []
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
                        dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_lanes[i], p_sym_uop: p_half})
                        dHdp = rewrite_elem(dHdp, ranges_lane)
                        dHdp = ensure_broadcast(dHdp, q_lanes[i])
                        q_new = q_lanes[i] + dt_uop * dHdp
                        p_half_lanes.append(p_half)
                        q_new_lanes.append(q_new)
                    q_new_red = q_new_lanes[0]
                    p_half_red = p_half_lanes[0]
                    for i in range(1, vector_width):
                        q_new_red = q_new_red + q_new_lanes[i]
                        p_half_red = p_half_red + p_half_lanes[i]
                    red1 = reduce_uop.substitute({q_sym_uop: q_new_red, p_sym_uop: p_half_red})
                else:
                    dHdq_1 = dHdq_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_val})
                    if placeholders_dHdq:
                        acc_loads = {placeholders_dHdq[j]: acc0_vals[j] for j in range(len(placeholders_dHdq))}
                        dHdq_1 = dHdq_1.substitute(acc_loads, name="reduce_placeholders")
                    dHdq_1 = rewrite_elem(dHdq_1)
                    p_half = p_val + (half_dt * dHdq_1) * neg_one
                    dHdp = dHdp_elem_uop.substitute({q_sym_uop: q_val, p_sym_uop: p_half})
                    dHdp = rewrite_elem(dHdp)
                    q_new = q_val + dt_uop * dHdp
                    red1 = reduce_uop.substitute({q_sym_uop: q_new, p_sym_uop: p_half})
                acc1_vals = []
                acc1_idx = UOp.const(dtypes.index, step)
                for i, reduce_uop in enumerate(reduce_uops):
                    red1 = reduce_uop.substitute({q_sym_uop: q_new_red if use_vec else q_new, p_sym_uop: p_half_red if use_vec else p_half})
                    red1 = graph_rewrite(
                        red1,
                        PatternMatcher([(UPat(Ops.REDUCE_AXIS, name="x"), _lower_reduce_axis)], compiled=False),
                        ctx={"reduce_counter": reduce_counter, "ranges": ranges},
                        name="lower_reduce_axis",
                    )
                    reduce_ranges1 = [r for r in red1.ranges if r.arg[-1] == AxisType.REDUCE]
                    acc1_store = acc1_regs[i].index(acc1_idx).store(red1).end(*reduce_ranges1)
                    acc1_val = acc1_regs[i].after(acc1_store).index(acc1_idx)
                    if not use_vec and shape_vals is not None:
                        try:
                            if acc1_val.shape != shape_vals:
                                acc1_val = acc1_val._broadcast_to(shape_vals)
                        except Exception:
                            pass
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
            sink = UOp.group(store_q, store_p).end(*ranges).sink(
                arg=KernelInfo(name=f"coupled_qp_new_with_reduce_{unroll_steps}", opts_to_apply=()))
            return _cse_uops(sink)

        return kernel

    def _evolve_scan_kernel_coupled_split(self, q: Tensor, p: Tensor, dt: float, steps: int,
                                          unroll_steps: int = 1, vector_width: int = 1) -> tuple[Tensor, Tensor]:
        q_sym_uop, p_sym_uop, dHdq_uop, dHdp_uop = self._get_coupled_grad_uops(q.device, q.shape, q.dtype)
        dHdq_reduce_nodes = [u for u in dHdq_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        dHdp_reduce_nodes = [u for u in dHdp_uop.toposort() if u.op is Ops.REDUCE_AXIS]
        reduce_placeholders_dHdq = {red: UOp.unique_const(red.dtype, 0, q.device) for red in dHdq_reduce_nodes}
        reduce_placeholders_dHdp = {red: UOp.unique_const(red.dtype, 0, q.device) for red in dHdp_reduce_nodes}
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

        use_fused_qp = len(dHdq_reduce_nodes) >= 1 and len(dHdp_reduce_nodes) == 0
        reduce_vector_width = 1
        if vector_width > 1 and q.shape and q.shape[-1] % vector_width == 0:
            reduce_vector_width = vector_width
        if getenv("TINYGRAD_COUPLED_REDUCE_TUNE", 0):
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
        reduce_unroll = 1
        if getenv("TINYGRAD_COUPLED_REDUCE_UNROLL_TUNE", 0):
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
            key_qp = (self.H, q.device, q.shape, q.dtype, "qp_new_reduce", dt, unroll_steps, vector_width, tuple(dHdq_reduce_nodes))
            kernel_qp_reduce = self._elem_kernel_coupled_cache.get(key_qp)
            if kernel_qp_reduce is None:
                kernel_qp_reduce = self._build_coupled_kernel_qp_new_with_reduce(
                    dt, q.device, q.shape, q.dtype, dHdq_elem_use, dHdp_elem_use, dHdq_reduce_nodes, dHdq_placeholders,
                    unroll_steps=unroll_steps, vector_width=vector_width)
                self._elem_kernel_coupled_cache[key_qp] = kernel_qp_reduce
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
            for _ in range(steps // unroll_steps):
                for _ in range(unroll_steps):
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
                    q_new, p_new = Tensor.custom_kernel(
                        q_cur, p_half, *accs_dHdp, *accs_dHdq_2, q_buf, p_buf, fxn=kernel_qp_update)[-2:]
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

    def _evolve_coupled_two_phase_vec(self, q: Tensor, p: Tensor, dt: float, steps: int,
                                      vector_width: int) -> tuple[Tensor, Tensor]:
        if vector_width < 2:
            raise ValueError("vector_width must be >= 2 for two-phase path")
        if q.shape[-1] % vector_width != 0:
            raise ValueError("vector_width must divide the last dimension")
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

        q_cur, p_cur = q, p
        for _ in range(steps):
            dHdq = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            dHdp = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            out = Tensor.custom_kernel(q_cur, p_cur, dHdq, dHdp, fxn=grad_kernel)
            dHdq, dHdp = out[2], out[3]
            q_new = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            p_half = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            out = Tensor.custom_kernel(q_cur, p_cur, dHdq, dHdp, q_new, p_half, fxn=update_kernel)
            q_new, p_half = out[4], out[5]
            Tensor.realize(q_new, p_half)
            dHdq2 = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
            dHdp2 = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            out = Tensor.custom_kernel(q_new, p_half, dHdq2, dHdp2, fxn=grad_kernel)
            dHdq2 = out[2]
            p_new = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
            out = Tensor.custom_kernel(p_half, dHdq2, p_new, fxn=update_p_kernel)
            p_new = out[2]
            Tensor.realize(p_new)
            q_cur, p_cur = q_new, p_new
        return q_cur, p_cur


    def _build_leapfrog_scan_kernel(self, dt: float, steps: int, unroll_steps: int, vector_width: int,
                                    device: str, shape: tuple[int, ...], dtype) -> Callable:
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

            for _ in range(unroll_steps):
                dHdq_1, _ = grad_uop(q_elem, p_elem)
                dt_uop = dHdq_1.const_like(dt)
                half_dt = dHdq_1.const_like(0.5*dt)
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
        reduce_placeholders_dHdq = {red: UOp.unique_const(red.dtype, 0, device) for red in dHdq_reduce_nodes} if has_reduce else {}
        reduce_placeholders_dHdp = {red: UOp.unique_const(red.dtype, 0, device) for red in dHdp_reduce_nodes} if has_reduce else {}
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
                    def compute_accs(reduce_nodes: list[UOp], reduce_placeholders: dict[UOp, UOp],
                                     q_uop: UOp, p_uop: UOp, reg_base: int) -> dict[UOp, UOp]:
                        if not reduce_nodes: return {}
                        acc_loads = {}
                        sub_reduce = {q_sym_uop: q_uop.vindex(*ranges_reduce), p_sym_uop: p_uop.vindex(*ranges_reduce)}
                        for i, red in enumerate(reduce_nodes):
                            red_val = red.substitute(sub_reduce)
                            red_val = rewrite_with_ranges(red_val, ranges_reduce, reduce_counter)
                            if reduce_full.get(red, False):
                                acc = UOp(Ops.DEFINE_REG, red.dtype.ptr(size=1, addrspace=AddrSpace.REG), arg=reg_base + i)
                                reduce_ranges = [r for r in red_val.ranges if r.arg[-1] == AxisType.REDUCE]
                                acc_store = acc.index(UOp.const(dtypes.int, 0)).store(red_val).end(*reduce_ranges, step)
                                acc_idx = acc.after(acc_store).index(UOp.const(dtypes.int, 0))
                                acc_loads[reduce_placeholders[red]] = acc_idx.broadcast(vector_width)
                            else:
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
                        sub_reduce = {q_sym_uop: q_uop.vindex(*ranges_reduce), p_sym_uop: p_uop.vindex(*ranges_reduce)}
                        for i, red in enumerate(reduce_nodes):
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
# FAST INTEGRATORS - Using analytical gradients (no autograd overhead)
# ============================================================================

def fast_leapfrog(q: Tensor, p: Tensor, dHdq_func: Callable, dHdp_func: Callable,
                  dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """Fast leapfrog using pre-computed gradient functions."""
    p_half = p - (0.5 * dt) * dHdq_func(q, p)
    q_new = q + dt * dHdp_func(q, p_half)
    p_new = p_half - (0.5 * dt) * dHdq_func(q_new, p_half)
    return q_new.realize(), p_new.realize()


def fast_yoshida4(q: Tensor, p: Tensor, dHdq_func: Callable, dHdp_func: Callable,
                  dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """Fast Yoshida4 using pre-computed gradient functions."""
    def substep(q, p, h):
        p_half = p - (0.5 * h) * dHdq_func(q, p)
        q_new = q + h * dHdp_func(q, p_half)
        p_new = p_half - (0.5 * h) * dHdq_func(q_new, p_half)
        return q_new, p_new
    q, p = substep(q, p, _W1 * dt)
    q, p = substep(q, p, _W0 * dt)
    q, p = substep(q, p, _W1 * dt)
    return q.realize(), p.realize()


# ============================================================================
# UTILITIES
# ============================================================================

def cross(a: Tensor, b: Tensor) -> Tensor:
    """Cross product of 3D vectors. Shape: (..., 3)."""
    return Tensor.stack([
        a[..., 1]*b[..., 2] - a[..., 2]*b[..., 1],
        a[..., 2]*b[..., 0] - a[..., 0]*b[..., 2],
        a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    ], dim=-1)


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

def _grad_LP(z: Tensor, H_func) -> Tensor:
    """Compute gradient of Hamiltonian for Lie-Poisson systems."""
    z_grad = z.detach().requires_grad_(True)
    H = H_func(z_grad)
    H.backward()
    return z_grad.grad.detach() if z_grad.grad is not None else z * 0


def lie_poisson_euler_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """1st-order Lie-Poisson Euler for so(3): dL/dt = L × ∇H(L)"""
    dHdL = _grad_LP(L, H_func)
    return (L + dt * cross(L, dHdL)).realize()


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

    return L_next.realize()


def lie_poisson_rk4_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """4th-order RK4 for so(3). Good accuracy, does NOT preserve Casimirs exactly."""
    def f(L_val):
        return cross(L_val, _grad_LP(L_val, H_func))

    k1 = f(L)
    k2 = f(L + 0.5 * dt * k1)
    k3 = f(L + 0.5 * dt * k2)
    k4 = f(L + dt * k3)
    return (L + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).realize()


def lie_poisson_splitting_so3(L: Tensor, I_inv: Tensor, dt: float = 0.01) -> Tensor:
    """
    Explicit splitting method for free rigid body.
    Preserves |L|² exactly (Casimir). 2nd order via Strang splitting.
    """
    def rotate_axis(L_val: Tensor, axis: int, angle: Tensor) -> Tensor:
        j, k = (axis + 1) % 3, (axis + 2) % 3
        c, s = angle.cos(), angle.sin()
        L_j_new = c * L_val[j] - s * L_val[k]
        L_k_new = s * L_val[j] + c * L_val[k]
        if axis == 0: return Tensor.stack([L_val[0], L_j_new, L_k_new])
        if axis == 1: return Tensor.stack([L_k_new, L_val[1], L_j_new])
        return Tensor.stack([L_j_new, L_k_new, L_val[2]])

    # Strang splitting: half-step forward, half-step reverse
    for axis in range(3):
        L = rotate_axis(L, axis, 0.5 * dt * L[axis] * I_inv[axis])
    for axis in [2, 1, 0]:
        L = rotate_axis(L, axis, 0.5 * dt * L[axis] * I_inv[axis])
    return L.realize()


class LiePoissonSystem:
    """
    Physics on a Lie algebra, defined by its Hamiltonian.

    For the free rigid body: H(L) = 0.5 * sum(L² / I)

    Example:
        I_inv = Tensor([1.0, 0.5, 0.333])
        def H(L): return 0.5 * (L * L * I_inv).sum()
        system = LiePoissonSystem(H, algebra="so3")
        L = system.step(L, dt=0.01)
    """
    INTEGRATORS = {
        "euler": lie_poisson_euler_so3,
        "midpoint": lie_poisson_midpoint_so3,
        "rk4": lie_poisson_rk4_so3,
    }

    def __init__(self, H_func, algebra: str = "so3", integrator: str = "midpoint"):
        if algebra != "so3":
            raise ValueError(f"Only so3 supported, got: {algebra}")
        self.H = H_func
        self.algebra = algebra
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
               record_every: int = 1) -> tuple[Tensor, list]:
        z_history: list[Tensor] = []
        for i in range(steps):
            if i % record_every == 0:
                z_history.append(z.detach())
            z = self.step(z, dt)
        z_history.append(z.detach())
        history = [(z_t.numpy().copy(), self.energy(z_t), self.casimir(z_t)) for z_t in z_history]
        return z, history


# ============================================================================
# RIGID BODY UTILITIES
# ============================================================================

def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Quaternion multiplication q1 * q2. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return Tensor.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit length."""
    return q / (q * q).sum().sqrt()


def integrate_orientation(quat: Tensor, omega: Tensor, dt: float) -> Tensor:
    """Integrate orientation quaternion: dq/dt = 0.5 * q ⊗ [0, ω]"""
    omega_quat = Tensor.stack([Tensor(0.0), omega[0], omega[1], omega[2]])
    dq = quaternion_multiply(quat, omega_quat) * 0.5
    return quaternion_normalize(quat + dt * dq).realize()


class RigidBodySystem:
    """
    Complete rigid body simulation: Lie-Poisson dynamics (L) + quaternion kinematics (q).

    Example (Tennis Racket Theorem):
        system = RigidBodySystem(Tensor([1.0, 2.0, 3.0]), integrator="midpoint")
        L = Tensor([0.01, 2.0, 0.01])  # Near intermediate axis
        q = Tensor([1.0, 0.0, 0.0, 0.0])  # Identity
        for _ in range(1000): L, q = system.step(L, q, dt=0.01)
    """
    def __init__(self, I: Tensor, integrator: str = "midpoint"):
        self.I = I
        self.I_inv = 1.0 / I
        self.H = lambda L: 0.5 * (L * L * self.I_inv).sum()
        self.integrator_name = integrator
        if integrator == "splitting":
            self._step_L = lambda L, dt: lie_poisson_splitting_so3(L, self.I_inv, dt)
        elif integrator in LiePoissonSystem.INTEGRATORS:
            self._lp = LiePoissonSystem(self.H, algebra="so3", integrator=integrator)
            self._step_L = lambda L, dt: self._lp.step(L, dt)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    def step(self, L: Tensor, q: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        L_new = self._step_L(L, dt)
        return L_new, integrate_orientation(q, L_new * self.I_inv, dt)

    def energy(self, L: Tensor) -> float:
        return float(self.H(L).numpy())

    def casimir(self, L: Tensor) -> float:
        return float((L * L).sum().numpy())

    def evolve(self, L: Tensor, q: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        history = []
        for i in range(steps):
            if i % record_every == 0:
                history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
            L, q = self.step(L, q, dt)
        history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
        return L, q, history


# ============================================================================
# POINT VORTEX DYNAMICS (Phase 3.1: Fluid Mechanics)
# ============================================================================
#
# Point vortices in 2D form a Hamiltonian system with non-standard symplectic
# structure. Each vortex has position (x_i, y_i) and circulation Γ_i.
#
# Hamiltonian: H = -1/(4π) Σ_{i<j} Γ_i Γ_j log|r_ij|
#
# The equations of motion (Kirchhoff):
#     Γ_i dx_i/dt = +∂H/∂y_i
#     Γ_i dy_i/dt = -∂H/∂x_i
#
# This is STILL autograd-friendly! We compute ∂H/∂z and apply the
# circulation-weighted Poisson structure.

def _grad_vortex(z: Tensor, H_func) -> Tensor:
    """Compute gradient of vortex Hamiltonian."""
    z_grad = z.detach().requires_grad_(True)
    H = H_func(z_grad)
    H.backward()
    return z_grad.grad.detach() if z_grad.grad is not None else z * 0


def point_vortex_hamiltonian(gamma: Tensor, softening: float = 1e-6):
    """
    Returns the Hamiltonian for N point vortices.

    H = -1/(4π) Σ_{i<j} Γ_i Γ_j log|r_ij|

    Args:
        gamma: Tensor of shape (N,) containing circulations
        softening: Small value to avoid log(0) singularity

    The gradient ∂H/∂z is computed via autograd.
    """
    import math
    n = gamma.shape[0]

    def H(z):
        # z has shape (2*N,) = [x0, y0, x1, y1, ...]
        x = z.reshape(n, 2)[:, 0]  # shape (N,)
        y = z.reshape(n, 2)[:, 1]  # shape (N,)

        # Compute pairwise distances: r_ij = sqrt((xi-xj)² + (yi-yj)²)
        dx = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N)
        dy = y.unsqueeze(1) - y.unsqueeze(0)  # (N, N)
        r_sq = dx * dx + dy * dy + softening * softening  # avoid log(0)

        # Γ_i Γ_j matrix
        gamma_ij = gamma.unsqueeze(1) * gamma.unsqueeze(0)  # (N, N)

        # H = -1/(4π) Σ_{i<j} Γ_i Γ_j log(r_ij)
        # Use upper triangle (i < j) by masking diagonal and lower
        log_r = (r_sq.sqrt() + 1e-10).log()
        H_matrix = -gamma_ij * log_r / (4 * math.pi)

        # Sum upper triangle only (avoid double counting)
        # Mask: 1 for i < j, 0 otherwise
        mask_data = []
        for i in range(n):
            row = [1.0 if j > i else 0.0 for j in range(n)]
            mask_data.extend(row)
        mask = Tensor(mask_data).reshape(n, n)

        return (H_matrix * mask).sum()

    return H


def vortex_euler(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """
    1st-order Euler for point vortices.

    Equations: Γ_i dz_i/dt = J · ∇H where J = [[0, 1], [-1, 0]]
    """
    dHdz = _grad_vortex(z, H_func)
    n = gamma.shape[0]

    # Build the circulation-weighted Poisson update
    # dx_i/dt = (1/Γ_i) ∂H/∂y_i
    # dy_i/dt = -(1/Γ_i) ∂H/∂x_i
    dHdz_reshaped = dHdz.reshape(n, 2)  # (N, 2) = [[dH/dx0, dH/dy0], ...]
    dHdx = dHdz_reshaped[:, 0]
    dHdy = dHdz_reshaped[:, 1]

    # Velocity: v_i = (1/Γ_i) * J · ∇_i H
    vx = dHdy / gamma   # dx/dt = (1/Γ) ∂H/∂y
    vy = -dHdx / gamma  # dy/dt = -(1/Γ) ∂H/∂x

    v = Tensor.stack([vx, vy], dim=1).reshape(-1)  # flatten to (2N,)
    return (z + dt * v).realize()


def vortex_rk4(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """4th-order Runge-Kutta for point vortices."""
    def velocity(z_val):
        dHdz = _grad_vortex(z_val, H_func)
        n = gamma.shape[0]
        dHdz_reshaped = dHdz.reshape(n, 2)
        dHdx = dHdz_reshaped[:, 0]
        dHdy = dHdz_reshaped[:, 1]
        vx = dHdy / gamma
        vy = -dHdx / gamma
        return Tensor.stack([vx, vy], dim=1).reshape(-1)

    k1 = velocity(z)
    k2 = velocity(z + 0.5 * dt * k1)
    k3 = velocity(z + 0.5 * dt * k2)
    k4 = velocity(z + dt * k3)
    return (z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).realize()


def vortex_midpoint(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01,
                    tol: float = 1e-10, max_iter: int = 10) -> Tensor:
    """Implicit midpoint for point vortices. Preserves Hamiltonian better."""
    def velocity(z_val):
        dHdz = _grad_vortex(z_val, H_func)
        n = gamma.shape[0]
        dHdz_reshaped = dHdz.reshape(n, 2)
        dHdx = dHdz_reshaped[:, 0]
        dHdy = dHdz_reshaped[:, 1]
        vx = dHdy / gamma
        vy = -dHdx / gamma
        return Tensor.stack([vx, vy], dim=1).reshape(-1)

    # Initial guess
    z_next = z + dt * velocity(z)

    for _ in range(max_iter):
        z_mid = 0.5 * (z + z_next)
        z_new = z + dt * velocity(z_mid)
        if (z_new - z_next).abs().max().numpy() < tol:
            break
        z_next = z_new

    return z_next.realize()


class PointVortexSystem:
    """
    Point vortex dynamics in 2D - Kirchhoff's equations.

    This is the "Hello World" of fluid mechanics: discrete vortices
    interacting via the Biot-Savart law.

    The Hamiltonian is defined automatically from circulations.
    Autograd computes all the interaction forces!

    Example:
        gamma = Tensor([1.0, 1.0, -1.0])  # 3 vortices
        z = Tensor([0.0, 1.0,   # vortex 0 at (0, 1)
                    1.0, 0.0,   # vortex 1 at (1, 0)
                    -1.0, 0.0]) # vortex 2 at (-1, 0)
        system = PointVortexSystem(gamma)
        z = system.step(z, dt=0.01)
    """
    INTEGRATORS = {
        "euler": vortex_euler,
        "rk4": vortex_rk4,
        "midpoint": vortex_midpoint,
    }

    def __init__(self, gamma: Tensor, integrator: str = "rk4", softening: float = 1e-6):
        self.gamma = gamma
        self.n_vortices = gamma.shape[0]
        self.H = point_vortex_hamiltonian(gamma, softening)
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step_func = self.INTEGRATORS[integrator]
        self.integrator_name = integrator

    def step(self, z: Tensor, dt: float = 0.01) -> Tensor:
        return self._step_func(z, self.gamma, self.H, dt)

    def energy(self, z: Tensor) -> float:
        return float(self.H(z).numpy())

    def momentum(self, z: Tensor) -> tuple[float, float]:
        """Linear impulse: P = Σ Γ_i r_i (conserved)."""
        n = self.gamma.shape[0]
        pos = z.reshape(n, 2)
        px = (self.gamma * pos[:, 0]).sum().numpy()
        py = (self.gamma * pos[:, 1]).sum().numpy()
        return float(px), float(py)

    def angular_momentum(self, z: Tensor) -> float:
        """Angular impulse: L = Σ Γ_i |r_i|² (conserved)."""
        n = self.gamma.shape[0]
        pos = z.reshape(n, 2)
        r_sq = (pos * pos).sum(axis=1)
        return float((self.gamma * r_sq).sum().numpy())

    def evolve(self, z: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, list]:
        z_history: list[Tensor] = []
        for i in range(steps):
            if i % record_every == 0:
                z_history.append(z.detach())
            z = self.step(z, dt)
        z_history.append(z.detach())

        history = []
        for z_t in z_history:
            z_np = z_t.numpy().copy()
            e = self.energy(z_t)
            px, py = self.momentum(z_t)
            L = self.angular_momentum(z_t)
            history.append((z_np, e, px, py, L))

        return z, history


# ============================================================================
# QUANTUM MECHANICS (Phase 4: Wavefunction Dynamics)
# ============================================================================
#
# Quantum mechanics uses complex wavefunctions ψ(x) evolving via Schrödinger:
#
#     iℏ ∂ψ/∂t = Ĥψ
#
# For a free particle: Ĥ = p̂²/2m = -ℏ²∇²/2m
#
# Solution: ψ(t) = e^(-iĤt/ℏ)ψ(0)
#
# In Fourier space, this becomes diagonal:
#     ψ̃(k,t) = e^(-iℏk²t/2m) ψ̃(k,0)
#
# This is the "split-operator" method - efficient and unitary!

import numpy as np


def fft(x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
    """
    1D FFT of complex tensor (real, imag) -> (real, imag).
    Uses numpy FFT internally.
    """
    c = x_real.numpy() + 1j * x_imag.numpy()
    ft = np.fft.fft(c)
    return Tensor(ft.real.copy()), Tensor(ft.imag.copy())


def ifft(x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
    """
    1D inverse FFT of complex tensor.
    """
    c = x_real.numpy() + 1j * x_imag.numpy()
    out = np.fft.ifft(c)
    return Tensor(out.real.copy()), Tensor(out.imag.copy())


def complex_mul(a_real: Tensor, a_imag: Tensor,
                b_real: Tensor, b_imag: Tensor) -> tuple[Tensor, Tensor]:
    """Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)"""
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    return real, imag


def complex_exp(phase: Tensor) -> tuple[Tensor, Tensor]:
    """e^(i*phase) = cos(phase) + i*sin(phase)"""
    return phase.cos(), phase.sin()


def wavefunction_norm(psi_real: Tensor, psi_imag: Tensor, dx: float) -> float:
    """Compute ∫|ψ|² dx (should equal 1 for normalized wavefunction)."""
    prob = psi_real * psi_real + psi_imag * psi_imag
    return float((prob.sum() * dx).numpy())


def normalize_wavefunction(psi_real: Tensor, psi_imag: Tensor,
                           dx: float) -> tuple[Tensor, Tensor]:
    """Normalize wavefunction so ∫|ψ|² dx = 1."""
    norm = np.sqrt(wavefunction_norm(psi_real, psi_imag, dx))
    return psi_real / norm, psi_imag / norm


def free_particle_propagator(k: Tensor, dt: float, hbar: float = 1.0,
                             m: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Free particle propagator in momentum space: e^(-i*ℏk²*dt/2m)

    This is the core of quantum evolution for a free particle.
    """
    phase = -hbar * k * k * dt / (2 * m)
    return complex_exp(phase)


def potential_propagator(V: Tensor, dt: float,
                         hbar: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Potential propagator in position space: e^(-i*V*dt/ℏ)

    For split-operator method with potentials.
    """
    phase = -V * dt / hbar
    return complex_exp(phase)


class QuantumSystem:
    """
    Quantum system for 1D wavefunction evolution.

    Uses split-operator method with FFT for efficient unitary evolution.

    The Schrödinger equation iℏ∂ψ/∂t = Ĥψ is solved via:
        ψ(t+dt) = e^(-iV̂dt/2ℏ) · FFT⁻¹[e^(-iT̂dt/ℏ) · FFT[e^(-iV̂dt/2ℏ)ψ(t)]]

    For free particle (V=0), this simplifies to momentum-space evolution.

    Example:
        system = QuantumSystem(N=512, L=20.0, m=1.0, hbar=1.0)
        psi_r, psi_i = system.gaussian_wavepacket(x0=0, sigma=1, k0=2)
        psi_r, psi_i = system.step(psi_r, psi_i, dt=0.01)
    """

    def __init__(self, N: int, L: float, m: float = 1.0, hbar: float = 1.0,
                 V: Tensor | None = None):
        """
        Initialize quantum system.

        Args:
            N: Number of grid points
            L: Domain size [-L/2, L/2]
            m: Particle mass
            hbar: Reduced Planck constant
            V: Potential energy V(x) as Tensor, or None for free particle
        """
        self.N = N
        self.L = L
        self.m = m
        self.hbar = hbar
        self.dx = L / N

        # Position grid
        self.x = Tensor(np.linspace(-L/2, L/2, N, endpoint=False))

        # Momentum grid (FFT frequencies)
        k_arr = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.k = Tensor(k_arr)

        # Potential
        self.V = V

    def gaussian_wavepacket(self, x0: float = 0.0, sigma: float = 1.0,
                            k0: float = 0.0) -> tuple[Tensor, Tensor]:
        """
        Create a Gaussian wavepacket: ψ(x) = (2πσ²)^(-1/4) exp(-(x-x0)²/4σ² + ik0*x)

        Args:
            x0: Initial position
            sigma: Width of wavepacket
            k0: Initial momentum (wavenumber)

        Returns:
            (psi_real, psi_imag): Complex wavefunction as two real tensors
        """
        x_np = self.x.numpy()

        # Gaussian envelope
        norm = (2 * np.pi * sigma**2) ** (-0.25)
        envelope = norm * np.exp(-(x_np - x0)**2 / (4 * sigma**2))

        # Phase factor e^(ik0*x)
        phase = k0 * x_np

        psi_real = Tensor(envelope * np.cos(phase))
        psi_imag = Tensor(envelope * np.sin(phase))

        return psi_real, psi_imag

    def step(self, psi_real: Tensor, psi_imag: Tensor,
             dt: float) -> tuple[Tensor, Tensor]:
        """
        Evolve wavefunction by one time step using split-operator method.

        For free particle: ψ(t+dt) = FFT⁻¹[e^(-iℏk²dt/2m) · FFT[ψ(t)]]
        With potential: uses Strang splitting for 2nd order accuracy.
        """
        if self.V is None:
            # Free particle - pure momentum space evolution
            # FFT to momentum space
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)

            # Apply kinetic propagator
            prop_r, prop_i = free_particle_propagator(self.k, dt, self.hbar, self.m)
            psi_k_r, psi_k_i = complex_mul(psi_k_r, psi_k_i, prop_r, prop_i)

            # IFFT back to position space
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)
        else:
            # Strang splitting: V/2 -> T -> V/2
            # Half step in potential
            prop_r, prop_i = potential_propagator(self.V, dt/2, self.hbar)
            psi_real, psi_imag = complex_mul(psi_real, psi_imag, prop_r, prop_i)

            # Full step in kinetic (momentum space)
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)
            prop_r, prop_i = free_particle_propagator(self.k, dt, self.hbar, self.m)
            psi_k_r, psi_k_i = complex_mul(psi_k_r, psi_k_i, prop_r, prop_i)
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)

            # Half step in potential
            prop_r, prop_i = potential_propagator(self.V, dt/2, self.hbar)
            psi_real, psi_imag = complex_mul(psi_real, psi_imag, prop_r, prop_i)

        return psi_real.realize(), psi_imag.realize()

    def probability_density(self, psi_real: Tensor, psi_imag: Tensor) -> Tensor:
        """Compute |ψ|² - probability density."""
        return psi_real * psi_real + psi_imag * psi_imag

    def expectation_x(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute <x> = ∫ψ*xψ dx"""
        prob = self.probability_density(psi_real, psi_imag)
        return float((prob * self.x).sum().numpy() * self.dx)

    def expectation_x2(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute <x²> = ∫ψ*x²ψ dx"""
        prob = self.probability_density(psi_real, psi_imag)
        return float((prob * self.x * self.x).sum().numpy() * self.dx)

    def width(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute wavepacket width: σ = √(<x²> - <x>²)"""
        x_mean = self.expectation_x(psi_real, psi_imag)
        x2_mean = self.expectation_x2(psi_real, psi_imag)
        var = x2_mean - x_mean**2
        return np.sqrt(max(var, 0))

    def energy(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """
        Compute total energy <H> = <T> + <V>.

        <T> is computed in momentum space for accuracy.
        """
        # Kinetic energy in momentum space
        # FFT normalization: |ψ̃|² needs to be divided by N² and multiplied by L
        psi_k_r, psi_k_i = fft(psi_real, psi_imag)
        prob_k = psi_k_r * psi_k_r + psi_k_i * psi_k_i
        # Normalize: dk = 2π/L, and FFT gives ψ̃ = Δx * Σ ψ * exp(...) so |ψ̃|² ~ Δx² * N
        # Correct normalization factor for <k²>
        norm_factor = self.dx * self.dx / self.L
        T = float((prob_k * self.hbar**2 * self.k * self.k / (2 * self.m)).sum().numpy() * norm_factor)

        # Potential energy
        if self.V is not None:
            prob_x = self.probability_density(psi_real, psi_imag)
            V_exp = float((prob_x * self.V).sum().numpy() * self.dx)
        else:
            V_exp = 0.0

        return T + V_exp

    def norm(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute ∫|ψ|² dx (should be 1 for normalized wavefunction)."""
        return wavefunction_norm(psi_real, psi_imag, self.dx)

    def evolve(self, psi_real: Tensor, psi_imag: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        """
        Evolve wavefunction for multiple steps.

        Returns:
            (psi_real, psi_imag, history) where history contains
            (x_array, prob_array, norm, width, x_mean, energy) for each recorded step.
        """
        history = []
        x_np = self.x.numpy()

        for i in range(steps):
            if i % record_every == 0:
                prob = self.probability_density(psi_real, psi_imag).numpy()
                n = self.norm(psi_real, psi_imag)
                w = self.width(psi_real, psi_imag)
                x_mean = self.expectation_x(psi_real, psi_imag)
                e = self.energy(psi_real, psi_imag)
                history.append((x_np.copy(), prob.copy(), n, w, x_mean, e))
            psi_real, psi_imag = self.step(psi_real, psi_imag, dt)

        # Record final state
        prob = self.probability_density(psi_real, psi_imag).numpy()
        n = self.norm(psi_real, psi_imag)
        w = self.width(psi_real, psi_imag)
        x_mean = self.expectation_x(psi_real, psi_imag)
        e = self.energy(psi_real, psi_imag)
        history.append((x_np.copy(), prob.copy(), n, w, x_mean, e))

        return psi_real, psi_imag, history

    def imaginary_time_step(self, psi_real: Tensor, psi_imag: Tensor,
                            dtau: float) -> tuple[Tensor, Tensor]:
        """
        Imaginary time evolution: ψ(τ+dτ) = e^(-Ĥdτ/ℏ)ψ(τ) / norm

        Wick rotation t → -iτ turns Schrödinger into diffusion equation.
        Higher energy states decay faster → relaxes to ground state.

        Must renormalize after each step (evolution is not unitary).
        """
        if self.V is None:
            # Free particle
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)
            # Kinetic decay: e^(-ℏk²dτ/2m) is real
            decay = (-self.hbar * self.k * self.k * dtau / (2 * self.m)).exp()
            psi_k_r = psi_k_r * decay
            psi_k_i = psi_k_i * decay
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)
        else:
            # Strang splitting with real exponential decay
            # Half step potential: e^(-Vdτ/2ℏ)
            decay_V_half = (-self.V * dtau / (2 * self.hbar)).exp()
            psi_real = psi_real * decay_V_half
            psi_imag = psi_imag * decay_V_half

            # Full step kinetic in momentum space
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)
            decay_T = (-self.hbar * self.k * self.k * dtau / (2 * self.m)).exp()
            psi_k_r = psi_k_r * decay_T
            psi_k_i = psi_k_i * decay_T
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)

            # Half step potential
            psi_real = psi_real * decay_V_half
            psi_imag = psi_imag * decay_V_half

        # Renormalize (critical for imaginary time!)
        psi_real, psi_imag = normalize_wavefunction(psi_real, psi_imag, self.dx)
        return psi_real.realize(), psi_imag.realize()

    def find_ground_state(self, psi_real: Tensor, psi_imag: Tensor,
                          dtau: float = 0.01, steps: int = 1000,
                          tol: float = 1e-10, record_every: int = 10) -> tuple[Tensor, Tensor, list]:
        """
        Find ground state via imaginary time evolution.

        Args:
            psi_real, psi_imag: Initial guess (any state with ground state overlap)
            dtau: Imaginary time step
            steps: Maximum steps
            tol: Energy convergence tolerance
            record_every: Record history every N steps

        Returns:
            (psi_real, psi_imag, history) where history contains
            (prob_array, energy, width) for each recorded step.
        """
        history = []
        x_np = self.x.numpy()
        E_prev = self.energy(psi_real, psi_imag)

        for i in range(steps):
            if i % record_every == 0:
                prob = self.probability_density(psi_real, psi_imag).numpy()
                E = self.energy(psi_real, psi_imag)
                w = self.width(psi_real, psi_imag)
                history.append((x_np.copy(), prob.copy(), E, w))

            psi_real, psi_imag = self.imaginary_time_step(psi_real, psi_imag, dtau)

            # Check convergence
            if i % record_every == 0 and i > 0:
                E_new = self.energy(psi_real, psi_imag)
                if abs(E_new - E_prev) < tol:
                    break
                E_prev = E_new

        # Record final state
        prob = self.probability_density(psi_real, psi_imag).numpy()
        E = self.energy(psi_real, psi_imag)
        w = self.width(psi_real, psi_imag)
        history.append((x_np.copy(), prob.copy(), E, w))

        return psi_real, psi_imag, history

    def harmonic_potential(self, omega: float = 1.0) -> Tensor:
        """Create harmonic oscillator potential: V(x) = ½mω²x²"""
        return 0.5 * self.m * omega**2 * self.x * self.x

    def ground_state_exact(self, omega: float = 1.0) -> tuple[Tensor, Tensor, float]:
        """
        Exact ground state of harmonic oscillator.

        ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ)
        E₀ = ℏω/2

        Returns:
            (psi_real, psi_imag, E0)
        """
        x_np = self.x.numpy()
        alpha = self.m * omega / self.hbar
        norm = (alpha / np.pi) ** 0.25
        psi = norm * np.exp(-alpha * x_np**2 / 2)
        E0 = self.hbar * omega / 2
        return Tensor(psi), Tensor(np.zeros_like(psi)), E0


# ============================================================================
# ISING MODEL (Phase 5: Statistical Mechanics)
# ============================================================================
#
# The Ising model describes interacting spins on a lattice:
#
#     H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i
#
# where s_i ∈ {+1, -1} are discrete spins, J is the coupling constant,
# h is the external magnetic field, and <i,j> denotes nearest neighbors.
#
# For J > 0 (ferromagnetic), aligned spins are favored.
# The 2D Ising model exhibits a phase transition at T_c ≈ 2.269 J/k_B.
#
# Unlike continuous Hamiltonian systems, the Ising model evolves via
# stochastic Monte Carlo dynamics (Metropolis algorithm) to sample
# from the Boltzmann distribution P(s) ∝ exp(-H/k_B T).


def ising_energy(spins: Tensor, J: float = 1.0, h: float = 0.0) -> float:
    """
    Compute the Ising model Hamiltonian for a 2D lattice.

    H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    Uses periodic boundary conditions.

    Args:
        spins: Tensor of shape (L, L) with values +1 or -1
        J: Coupling constant (J > 0 for ferromagnetic)
        h: External magnetic field

    Returns:
        Total energy of the configuration
    """
    # Nearest neighbor interactions (periodic boundary conditions)
    # Shift right and down to get neighbors
    spins_np = spins.numpy()
    L = spins_np.shape[0]

    # Sum over all nearest-neighbor pairs
    interaction = (
        spins_np * np.roll(spins_np, 1, axis=0) +  # up neighbor
        spins_np * np.roll(spins_np, 1, axis=1)    # left neighbor
    )

    H = -J * interaction.sum() - h * spins_np.sum()
    return float(H)


def ising_magnetization(spins: Tensor) -> float:
    """
    Compute the magnetization per spin: m = (1/N) Σ s_i

    Returns value in [-1, 1]. |m| → 1 means ordered (ferromagnetic),
    m → 0 means disordered (paramagnetic).
    """
    return float(spins.numpy().mean())


def ising_magnetization_abs(spins: Tensor) -> float:
    """Compute the absolute magnetization per spin: |m|"""
    return abs(ising_magnetization(spins))


def metropolis_step(spins: Tensor, beta: float, J: float = 1.0,
                    h: float = 0.0) -> Tensor:
    """
    One Metropolis Monte Carlo sweep over all spins.

    For each spin, propose a flip and accept with probability:
        P_accept = min(1, exp(-β ΔE))

    where ΔE is the energy change from flipping that spin.

    Args:
        spins: Current spin configuration (L, L)
        beta: Inverse temperature β = 1/(k_B T)
        J: Coupling constant
        h: External magnetic field

    Returns:
        Updated spin configuration
    """
    spins_np = spins.numpy().copy()
    L = spins_np.shape[0]

    # For 2D Ising, ΔE for flipping spin s_i = 2 s_i (J Σ_neighbors s_j + h)
    for _ in range(L * L):
        # Pick random site
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        s = spins_np[i, j]

        # Sum of neighbors (periodic boundary conditions)
        neighbors_sum = (
            spins_np[(i + 1) % L, j] +
            spins_np[(i - 1) % L, j] +
            spins_np[i, (j + 1) % L] +
            spins_np[i, (j - 1) % L]
        )

        # Energy change from flipping spin
        delta_E = 2 * s * (J * neighbors_sum + h)

        # Metropolis acceptance
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            spins_np[i, j] = -s

    return Tensor(spins_np)


def wolff_step(spins: Tensor, beta: float, J: float = 1.0) -> Tensor:
    """
    One Wolff cluster flip step (faster near critical temperature).

    The Wolff algorithm builds a cluster of aligned spins with probability
    P_add = 1 - exp(-2βJ) and flips the entire cluster. This dramatically
    reduces critical slowing down near T_c.

    Args:
        spins: Current spin configuration (L, L)
        beta: Inverse temperature β = 1/(k_B T)
        J: Coupling constant

    Returns:
        Updated spin configuration

    Note: External field h is not supported in Wolff algorithm.
    """
    spins_np = spins.numpy().copy()
    L = spins_np.shape[0]

    # Probability to add neighbor to cluster
    p_add = 1.0 - np.exp(-2 * beta * J)

    # Pick random seed spin
    i0, j0 = np.random.randint(0, L), np.random.randint(0, L)
    seed_spin = spins_np[i0, j0]

    # Build cluster using BFS
    cluster = {(i0, j0)}
    frontier = [(i0, j0)]

    while frontier:
        i, j = frontier.pop()
        for di, dj in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            ni, nj = (i + di) % L, (j + dj) % L
            if (ni, nj) not in cluster and spins_np[ni, nj] == seed_spin:
                if np.random.random() < p_add:
                    cluster.add((ni, nj))
                    frontier.append((ni, nj))

    # Flip all spins in cluster
    for i, j in cluster:
        spins_np[i, j] = -spins_np[i, j]

    return Tensor(spins_np)


class IsingSystem:
    """
    2D Ising model on a square lattice with periodic boundary conditions.

    The Ising model is the "Hello World" of statistical mechanics:
        H = -J Σ_{<i,j>} s_i s_j - h Σ_i s_i

    Unlike continuous Hamiltonian systems, the Ising model evolves via
    stochastic Monte Carlo dynamics to sample the Boltzmann distribution.

    The system exhibits a phase transition:
        - T < T_c: Ordered (ferromagnetic), |m| ≈ 1
        - T > T_c: Disordered (paramagnetic), m ≈ 0
        - T_c = 2 / ln(1 + √2) ≈ 2.269 (for J = k_B = 1)

    Example:
        system = IsingSystem(L=32, J=1.0, h=0.0, T=2.0)
        spins = system.random_spins()  # Start hot (random)
        for _ in range(1000):
            spins = system.step(spins)  # Equilibrate
        print(f"Magnetization: {system.magnetization(spins):.3f}")
    """

    # Critical temperature for 2D Ising model (exact, Onsager 1944)
    T_CRITICAL = 2.0 / np.log(1 + np.sqrt(2))  # ≈ 2.269

    ALGORITHMS = {"metropolis": metropolis_step, "wolff": wolff_step}

    def __init__(self, L: int, J: float = 1.0, h: float = 0.0, T: float = 2.0,
                 algorithm: str = "metropolis"):
        """
        Initialize Ising system.

        Args:
            L: Lattice size (L × L spins)
            J: Coupling constant (J > 0 for ferromagnetic)
            h: External magnetic field
            T: Temperature (in units where k_B = 1)
            algorithm: "metropolis" or "wolff"
        """
        self.L = L
        self.J = J
        self.h = h
        self.T = T
        self.beta = 1.0 / T

        if algorithm not in self.ALGORITHMS:
            raise ValueError(f"Unknown algorithm: {algorithm}. Use: {list(self.ALGORITHMS.keys())}")
        if algorithm == "wolff" and h != 0:
            raise ValueError("Wolff algorithm does not support external field h ≠ 0")
        self._step_func = self.ALGORITHMS[algorithm]
        self.algorithm_name = algorithm

    def random_spins(self) -> Tensor:
        """Create random (hot) initial configuration."""
        return Tensor(np.random.choice([-1, 1], size=(self.L, self.L)).astype(np.float32))

    def uniform_spins(self, value: int = 1) -> Tensor:
        """Create uniform (cold) initial configuration."""
        return Tensor(np.full((self.L, self.L), value, dtype=np.float32))

    def step(self, spins: Tensor) -> Tensor:
        """Perform one Monte Carlo step."""
        if self.algorithm_name == "wolff":
            return self._step_func(spins, self.beta, self.J)
        return self._step_func(spins, self.beta, self.J, self.h)

    def set_temperature(self, T: float):
        """Change the temperature."""
        self.T = T
        self.beta = 1.0 / T

    def energy(self, spins: Tensor) -> float:
        """Compute total energy H."""
        return ising_energy(spins, self.J, self.h)

    def energy_per_spin(self, spins: Tensor) -> float:
        """Compute energy per spin: E/N."""
        return self.energy(spins) / (self.L * self.L)

    def magnetization(self, spins: Tensor) -> float:
        """Compute magnetization per spin: m = (1/N) Σ s_i."""
        return ising_magnetization(spins)

    def magnetization_abs(self, spins: Tensor) -> float:
        """Compute absolute magnetization per spin: |m|."""
        return ising_magnetization_abs(spins)

    def evolve(self, spins: Tensor, steps: int, record_every: int = 1,
               warmup: int = 0) -> tuple[Tensor, list]:
        """
        Evolve the system and record history.

        Args:
            spins: Initial spin configuration
            steps: Number of Monte Carlo steps
            record_every: Record every N steps
            warmup: Number of initial steps to skip (thermalization)

        Returns:
            (final_spins, history) where history contains
            (spins_array, energy, magnetization) for each recorded step.
        """
        # Warmup (thermalization)
        for _ in range(warmup):
            spins = self.step(spins)

        history = []
        for i in range(steps):
            if i % record_every == 0:
                spins_np = spins.numpy().copy()
                e = self.energy_per_spin(spins)
                m = self.magnetization_abs(spins)
                history.append((spins_np, e, m))
            spins = self.step(spins)

        # Record final state
        spins_np = spins.numpy().copy()
        e = self.energy_per_spin(spins)
        m = self.magnetization_abs(spins)
        history.append((spins_np, e, m))

        return spins, history

    def measure(self, spins: Tensor, steps: int, warmup: int = 100) -> dict:
        """
        Measure thermal averages after equilibration.

        Returns dict with <E>, <E²>, <|m|>, <m²>, specific heat C, susceptibility χ.
        """
        # Thermalize
        for _ in range(warmup):
            spins = self.step(spins)

        E_vals, M_vals = [], []
        for _ in range(steps):
            spins = self.step(spins)
            E_vals.append(self.energy_per_spin(spins))
            M_vals.append(self.magnetization_abs(spins))

        E_mean = np.mean(E_vals)
        E2_mean = np.mean(np.array(E_vals) ** 2)
        M_mean = np.mean(M_vals)
        M2_mean = np.mean(np.array(M_vals) ** 2)

        N = self.L * self.L
        # Specific heat: C = (β²/N) * (<E²> - <E>²)
        C = (self.beta ** 2) * N * (E2_mean - E_mean ** 2)
        # Susceptibility: χ = β * N * (<m²> - <m>²)
        chi = self.beta * N * (M2_mean - M_mean ** 2)

        return {
            "E_mean": E_mean,
            "E2_mean": E2_mean,
            "M_mean": M_mean,
            "M2_mean": M2_mean,
            "specific_heat": C,
            "susceptibility": chi,
            "temperature": self.T,
        }


# ============================================================================
# CONTINUOUS SPIN ISING (Soft Spins with Hamiltonian Dynamics)
# ============================================================================
#
# For autograd-friendly Ising dynamics, we can use continuous "soft spins"
# σ ∈ ℝ with a double-well potential that pushes them toward ±1:
#
#     H = Σ_i p_i²/2 + λ Σ_i (σ_i² - 1)² - J Σ_{<i,j>} σ_i σ_j - h Σ_i σ_i
#
# This gives Hamiltonian dynamics that relaxes toward Ising-like states.


def soft_ising_hamiltonian(L: int, J: float = 1.0, h: float = 0.0, lam: float = 1.0):
    """
    Create a Hamiltonian for continuous (soft) spins on a 2D lattice.

    H = Σ p²/2m + λ Σ(σ² - 1)² - J Σ_{<i,j>} σ_i σ_j - h Σ σ

    The λ(σ² - 1)² term is a double-well potential pushing spins toward ±1.

    Args:
        L: Lattice size (L × L)
        J: Coupling constant
        h: External field
        lam: Double-well strength (larger = more discrete-like)

    Returns:
        H_func(q, p) for use with HamiltonianSystem
    """
    def H(q: Tensor, p: Tensor) -> Tensor:
        # q is flattened (L*L,), reshape to (L, L)
        sigma = q.reshape(L, L)

        # Kinetic energy
        T = 0.5 * (p * p).sum()

        # Double-well potential: pushes σ toward ±1
        V_well = lam * ((sigma * sigma - 1) ** 2).sum()

        # Nearest-neighbor interaction (periodic BC)
        # Use tinygrad operations for autograd
        sigma_right = sigma.cat(sigma[:, :1], dim=1)[:, 1:]  # shift left
        sigma_down = sigma.cat(sigma[:1, :], dim=0)[1:, :]   # shift up
        V_coupling = -J * (sigma * sigma_right + sigma * sigma_down).sum()

        # External field
        V_field = -h * sigma.sum()

        return T + V_well + V_coupling + V_field

    return H


class SoftIsingSystem:
    """
    Continuous-spin Ising model with Hamiltonian dynamics.

    Uses "soft spins" σ ∈ ℝ with a double-well potential to approximate
    discrete Ising dynamics. This allows gradient-based Hamiltonian mechanics.

    The Hamiltonian:
        H = Σ p²/2 + λ Σ(σ² - 1)² - J Σ_{<i,j>} σ_i σ_j - h Σ σ

    Example:
        system = SoftIsingSystem(L=16, J=1.0, lam=5.0)
        q, p = system.random_state()
        for _ in range(100):
            q, p = system.step(q, p, dt=0.01)
    """

    def __init__(self, L: int, J: float = 1.0, h: float = 0.0, lam: float = 5.0,
                 integrator: str = "leapfrog"):
        """
        Initialize soft Ising system.

        Args:
            L: Lattice size (L × L)
            J: Coupling constant
            h: External field
            lam: Double-well strength
            integrator: "euler", "leapfrog", "yoshida4", or "implicit"
        """
        self.L = L
        self.J = J
        self.h = h
        self.lam = lam
        self.H_func = soft_ising_hamiltonian(L, J, h, lam)
        self._system = HamiltonianSystem(self.H_func, integrator=integrator)

    def random_state(self, noise: float = 0.1) -> tuple[Tensor, Tensor]:
        """Create random initial state near ±1."""
        sigma = np.random.choice([-1.0, 1.0], size=(self.L * self.L,))
        sigma = sigma + noise * np.random.randn(self.L * self.L)
        q = Tensor(sigma.astype(np.float32))
        p = Tensor(np.zeros(self.L * self.L, dtype=np.float32))
        return q, p

    def uniform_state(self, value: float = 1.0) -> tuple[Tensor, Tensor]:
        """Create uniform initial state."""
        q = Tensor(np.full(self.L * self.L, value, dtype=np.float32))
        p = Tensor(np.zeros(self.L * self.L, dtype=np.float32))
        return q, p

    def step(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        """Perform one symplectic integration step."""
        return self._system.step(q, p, dt)

    def energy(self, q: Tensor, p: Tensor) -> float:
        """Compute total Hamiltonian."""
        return self._system.energy(q, p)

    def magnetization(self, q: Tensor) -> float:
        """Compute magnetization per spin."""
        return float(q.numpy().mean())

    def discretized_spins(self, q: Tensor) -> Tensor:
        """Get discrete spins by taking sign of continuous values."""
        return Tensor(np.sign(q.numpy()).astype(np.float32))


# ============================================================================
# KOSTERLITZ-THOULESS / XY MODEL (Phase 6: Topological Physics)
# ============================================================================
#
# The 2D XY model is defined by continuous spins θ_i ∈ [0, 2π) on a lattice:
#
#     H = -J Σ_{<ij>} cos(θ_i - θ_j)
#
# Key physics:
#     - Below T_KT: vortices bound in pairs, quasi-long-range order
#     - Above T_KT: vortices unbind, exponential decay of correlations
#     - T_KT ≈ 0.89 J (Kosterlitz-Thouless transition)
#
# The vortex-antivortex interaction follows a Coulomb gas mapping:
#     H_vortex ~ -πJ Σ_{i<j} n_i n_j log|r_ij|
# where n_i = ±1 is the vorticity (winding number).


def xy_hamiltonian_lattice(theta: Tensor, J: float = 1.0) -> Tensor:
    """
    XY model Hamiltonian on a 2D lattice with periodic boundary conditions.

    H = -J Σ_{<ij>} cos(θ_i - θ_j)

    Args:
        theta: Tensor of shape (L, L) containing spin angles in [0, 2π)
        J: Coupling constant (positive = ferromagnetic)

    Returns:
        Total energy as a scalar Tensor
    """
    # Compute angle differences to neighbors using roll for periodic BC
    # Roll by -1 gets the next neighbor (i+1)
    theta_np = theta.numpy()
    L = theta_np.shape[0]

    # Differences to right and down neighbors
    diff_right = theta_np - np.roll(theta_np, -1, axis=1)  # θ_{i,j} - θ_{i,j+1}
    diff_down = theta_np - np.roll(theta_np, -1, axis=0)   # θ_{i,j} - θ_{i+1,j}

    # H = -J Σ cos(Δθ)
    energy = -J * (np.cos(diff_right).sum() + np.cos(diff_down).sum())
    return Tensor([energy])


def _xy_grad(theta: Tensor, J: float = 1.0) -> Tensor:
    """
    Compute gradient of XY Hamiltonian analytically.

    ∂H/∂θ_i = J Σ_{j∈neighbors} sin(θ_i - θ_j)
    """
    theta_np = theta.numpy()
    L = theta_np.shape[0]

    # Contributions from all 4 neighbors
    grad = np.zeros_like(theta_np)

    # Right neighbor: sin(θ_{i,j} - θ_{i,j+1})
    grad += J * np.sin(theta_np - np.roll(theta_np, -1, axis=1))
    # Left neighbor: sin(θ_{i,j} - θ_{i,j-1})
    grad += J * np.sin(theta_np - np.roll(theta_np, 1, axis=1))
    # Down neighbor: sin(θ_{i,j} - θ_{i+1,j})
    grad += J * np.sin(theta_np - np.roll(theta_np, -1, axis=0))
    # Up neighbor: sin(θ_{i,j} - θ_{i-1,j})
    grad += J * np.sin(theta_np - np.roll(theta_np, 1, axis=0))

    return Tensor(grad.astype(np.float32))


def detect_vortices(theta: Tensor) -> tuple[list, list]:
    """
    Detect vortices and antivortices on the XY lattice.

    Uses plaquette winding number: n = (1/2π) Σ_plaquette Δθ

    A vortex has winding +1 (angles increase by 2π around plaquette).
    An antivortex has winding -1 (angles decrease by 2π).

    Returns:
        (vortices, antivortices): Lists of (x, y) positions (plaquette centers)
    """
    theta_np = theta.numpy()
    L = theta_np.shape[0]

    def wrap_angle(a):
        """Wrap angle difference to [-π, π]."""
        return np.arctan2(np.sin(a), np.cos(a))

    vortices = []
    antivortices = []

    for i in range(L):
        for j in range(L):
            # Plaquette corners: (i,j) -> (i,j+1) -> (i+1,j+1) -> (i+1,j) -> (i,j)
            i1 = (i + 1) % L
            j1 = (j + 1) % L

            # Angle differences around plaquette (counterclockwise)
            d1 = wrap_angle(theta_np[i, j1] - theta_np[i, j])      # right
            d2 = wrap_angle(theta_np[i1, j1] - theta_np[i, j1])    # down
            d3 = wrap_angle(theta_np[i1, j] - theta_np[i1, j1])    # left
            d4 = wrap_angle(theta_np[i, j] - theta_np[i1, j])      # up

            winding = (d1 + d2 + d3 + d4) / (2 * np.pi)

            if winding > 0.5:
                vortices.append((i + 0.5, j + 0.5))
            elif winding < -0.5:
                antivortices.append((i + 0.5, j + 0.5))

    return vortices, antivortices


def xy_langevin_step(theta: Tensor, J: float = 1.0, dt: float = 0.01,
                     temperature: float = 0.5, gamma: float = 1.0) -> Tensor:
    """
    Overdamped Langevin dynamics for XY model.

    dθ/dt = -(1/γ) ∂H/∂θ + √(2T/γ) η(t)

    where η is Gaussian white noise.

    Args:
        theta: Current angles (L, L)
        J: Coupling constant
        dt: Time step
        temperature: Temperature (T_KT ≈ 0.89 J)
        gamma: Damping coefficient

    Returns:
        Updated angles (wrapped to [0, 2π))
    """
    # Compute gradient analytically
    dHdtheta = _xy_grad(theta, J)

    # Deterministic drift toward lower energy
    drift = -dHdtheta.numpy() / gamma

    # Stochastic thermal noise
    noise_strength = np.sqrt(2 * temperature * dt / gamma)
    noise = noise_strength * np.random.randn(*theta.shape).astype(np.float32)

    # Update angles
    theta_new = theta.numpy() + dt * drift + noise

    # Wrap to [0, 2π)
    theta_new = theta_new % (2 * np.pi)

    return Tensor(theta_new.astype(np.float32))


def xy_metropolis_step(theta: Tensor, J: float = 1.0, beta: float = 1.0,
                       delta: float = 0.5) -> Tensor:
    """
    Metropolis Monte Carlo step for XY model.

    For each spin, propose θ' = θ + uniform(-δ, δ) and accept with
    probability min(1, exp(-β ΔE)).

    Args:
        theta: Current angles (L, L)
        J: Coupling constant
        beta: Inverse temperature β = 1/T
        delta: Maximum angle change per proposal

    Returns:
        Updated angles
    """
    theta_np = theta.numpy().copy()
    L = theta_np.shape[0]

    for _ in range(L * L):
        i, j = np.random.randint(0, L), np.random.randint(0, L)
        theta_old = theta_np[i, j]

        # Propose new angle
        theta_new = theta_old + np.random.uniform(-delta, delta)

        # Compute energy change (only affected bonds)
        neighbors = [
            theta_np[(i + 1) % L, j],
            theta_np[(i - 1) % L, j],
            theta_np[i, (j + 1) % L],
            theta_np[i, (j - 1) % L]
        ]

        E_old = -J * sum(np.cos(theta_old - n) for n in neighbors)
        E_new = -J * sum(np.cos(theta_new - n) for n in neighbors)
        delta_E = E_new - E_old

        # Metropolis acceptance
        if delta_E <= 0 or np.random.random() < np.exp(-beta * delta_E):
            theta_np[i, j] = theta_new % (2 * np.pi)

    return Tensor(theta_np.astype(np.float32))


class XYLatticeSystem:
    """
    2D XY model on a lattice - the canonical system for Kosterlitz-Thouless physics.

    The XY model consists of planar spins (angles θ ∈ [0, 2π)) on a 2D lattice:
        H = -J Σ_{<ij>} cos(θ_i - θ_j)

    The system exhibits the Kosterlitz-Thouless transition:
    - T < T_KT ≈ 0.89 J: Vortices bound in pairs, power-law correlations
    - T > T_KT: Free vortices proliferate, exponential correlation decay

    This is a topological phase transition - no symmetry breaking, but
    the proliferation of topological defects (vortices) destroys order.

    Example:
        system = XYLatticeSystem(L=32, J=1.0, temperature=0.5)
        theta = system.random_state()
        for _ in range(1000):
            theta = system.step(theta, dt=0.01)
        vortices, antivortices = system.detect_vortices(theta)
        print(f"Found {len(vortices)} vortices, {len(antivortices)} antivortices")
    """

    # Kosterlitz-Thouless transition temperature (approximate)
    T_KT = 0.89  # In units where J = 1

    def __init__(self, L: int, J: float = 1.0, temperature: float = 0.5,
                 gamma: float = 1.0, dynamics: str = "langevin"):
        """
        Initialize XY lattice system.

        Args:
            L: Lattice size (L × L)
            J: Coupling constant
            temperature: Temperature (T_KT ≈ 0.89 J for the transition)
            gamma: Damping coefficient (for Langevin dynamics)
            dynamics: "langevin" or "metropolis"
        """
        self.L = L
        self.J = J
        self.temperature = temperature
        self.gamma = gamma
        self.dynamics = dynamics

    def random_state(self) -> Tensor:
        """Initialize with random angles (high-T like)."""
        return Tensor(np.random.uniform(0, 2 * np.pi, (self.L, self.L)).astype(np.float32))

    def ordered_state(self, angle: float = 0.0) -> Tensor:
        """Initialize with all spins aligned (low-T ground state)."""
        return Tensor(np.full((self.L, self.L), angle, dtype=np.float32))

    def single_vortex(self, x0: int = None, y0: int = None, charge: int = 1) -> Tensor:
        """
        Create a configuration with a single vortex at (x0, y0).

        θ(x, y) = charge × arctan2(y - y0, x - x0)

        Note: On a periodic lattice, a single vortex is topologically
        forbidden (total charge must be 0). This creates a defect that
        will be screened by boundary effects.
        """
        if x0 is None: x0 = self.L // 2
        if y0 is None: y0 = self.L // 2

        theta = np.zeros((self.L, self.L), dtype=np.float32)
        for i in range(self.L):
            for j in range(self.L):
                dx = i - x0
                dy = j - y0
                if dx == 0 and dy == 0:
                    theta[i, j] = 0
                else:
                    theta[i, j] = charge * np.arctan2(dy, dx)
        return Tensor(theta % (2 * np.pi))

    def vortex_pair(self, separation: int = 4) -> Tensor:
        """
        Create a vortex-antivortex pair configuration.

        This is the fundamental excitation in the KT model.
        At T < T_KT, pairs remain bound. At T > T_KT, they unbind.
        """
        x0 = self.L // 2 - separation // 2
        x1 = self.L // 2 + separation // 2
        y0 = self.L // 2

        theta = np.zeros((self.L, self.L), dtype=np.float32)
        for i in range(self.L):
            for j in range(self.L):
                # Vortex at (x0, y0)
                angle1 = np.arctan2(j - y0, i - x0)
                # Antivortex at (x1, y0)
                angle2 = -np.arctan2(j - y0, i - x1)
                theta[i, j] = angle1 + angle2
        return Tensor(theta % (2 * np.pi))

    def step(self, theta: Tensor, dt: float = 0.01) -> Tensor:
        """Evolve the system by one time step."""
        if self.dynamics == "langevin":
            return xy_langevin_step(theta, self.J, dt, self.temperature, self.gamma)
        else:  # metropolis
            return xy_metropolis_step(theta, self.J, 1.0 / self.temperature, delta=0.5)

    def energy(self, theta: Tensor) -> float:
        """Compute total energy."""
        return float(xy_hamiltonian_lattice(theta, self.J).numpy()[0])

    def energy_per_spin(self, theta: Tensor) -> float:
        """Energy per spin (ground state is -2J per spin)."""
        return self.energy(theta) / (self.L * self.L)

    def detect_vortices(self, theta: Tensor) -> tuple[list, list]:
        """Detect vortices and antivortices."""
        return detect_vortices(theta)

    def vortex_count(self, theta: Tensor) -> tuple[int, int]:
        """Count vortices and antivortices."""
        v, av = self.detect_vortices(theta)
        return len(v), len(av)

    def vortex_density(self, theta: Tensor) -> float:
        """Total vortex + antivortex density."""
        n_v, n_av = self.vortex_count(theta)
        return (n_v + n_av) / (self.L * self.L)

    def magnetization(self, theta: Tensor) -> tuple[float, float]:
        """
        Compute magnetization M = (1/N) Σ (cos θ, sin θ).

        Returns (|M|, angle). For XY model, |M| → 0 in thermodynamic limit
        but correlations <S_i · S_j> can have power-law decay below T_KT.
        """
        theta_np = theta.numpy()
        mx = np.mean(np.cos(theta_np))
        my = np.mean(np.sin(theta_np))
        return np.sqrt(mx**2 + my**2), np.arctan2(my, mx)

    def helicity_modulus(self, theta: Tensor, delta: float = 0.01) -> float:
        """
        Compute helicity modulus (superfluid stiffness) Υ.

        Υ = (1/L²) d²F/dφ² where φ is a twist in boundary conditions.
        This is the order parameter for the KT transition:
        - T < T_KT: Υ > 0 (universal jump to 2T/π at T_KT)
        - T > T_KT: Υ = 0

        Uses finite difference approximation.
        """
        theta_np = theta.numpy()
        L = self.L

        # Energy at twist 0
        E0 = self.energy(theta)

        # Energy at twist +δ (add phase gradient in x direction)
        theta_plus = theta_np.copy()
        for i in range(L):
            theta_plus[:, i] += delta * i / L
        E_plus = float(xy_hamiltonian_lattice(Tensor(theta_plus), self.J).numpy()[0])

        # Energy at twist -δ
        theta_minus = theta_np.copy()
        for i in range(L):
            theta_minus[:, i] -= delta * i / L
        E_minus = float(xy_hamiltonian_lattice(Tensor(theta_minus), self.J).numpy()[0])

        # Second derivative: Υ = d²E/dφ² / L²
        d2E = (E_plus - 2 * E0 + E_minus) / (delta ** 2)
        return d2E / (L * L)

    def evolve(self, theta: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, list]:
        """Evolve system and record history."""
        history = []

        for i in range(steps):
            if i % record_every == 0:
                theta_np = theta.numpy().copy()
                e = self.energy_per_spin(theta)
                v, av = self.detect_vortices(theta)
                m_mag, m_angle = self.magnetization(theta)
                history.append({
                    'theta': theta_np,
                    'energy': e,
                    'vortices': v,
                    'antivortices': av,
                    'n_vortices': len(v),
                    'n_antivortices': len(av),
                    'magnetization': m_mag,
                })
            theta = self.step(theta, dt)

        # Record final state
        theta_np = theta.numpy().copy()
        e = self.energy_per_spin(theta)
        v, av = self.detect_vortices(theta)
        m_mag, _ = self.magnetization(theta)
        history.append({
            'theta': theta_np,
            'energy': e,
            'vortices': v,
            'antivortices': av,
            'n_vortices': len(v),
            'n_antivortices': len(av),
            'magnetization': m_mag,
        })

        return theta, history


# ============================================================================
# XY VORTEX GAS MODEL (Coulomb Gas Representation)
# ============================================================================
#
# The XY model vortices can be mapped exactly to a 2D Coulomb gas:
#
#     H = -πJ Σ_{i<j} n_i n_j log|r_ij/a| + E_core × N_vortices
#
# where n_i = ±1 is the vortex charge (winding number) and a is the
# lattice spacing (UV cutoff).
#
# This "dual" description makes the KT physics transparent:
# - Vortex pairs attract (opposite charges)
# - Free energy balance: E_pair ~ 2πJ log(R) vs S_pair ~ 2 log(R)
# - Unbinding at T_KT where entropy wins: k_B T_KT = πJ/2


def xy_vortex_hamiltonian(charges: Tensor, J: float = 1.0,
                          E_core: float = 0.0, a: float = 1.0):
    """
    Coulomb gas Hamiltonian for XY model vortices.

    H = -πJ Σ_{i<j} n_i n_j log|r_ij/a| + E_core × N

    Args:
        charges: Tensor of shape (N,) with values +1 or -1
        J: XY coupling constant
        E_core: Vortex core energy
        a: Lattice spacing (UV cutoff)

    Returns:
        Hamiltonian function H(z) where z contains positions
    """
    n = charges.shape[0]

    def H(z):
        x = z.reshape(n, 2)[:, 0]
        y = z.reshape(n, 2)[:, 1]

        # Pairwise distances
        dx = x.unsqueeze(1) - x.unsqueeze(0)
        dy = y.unsqueeze(1) - y.unsqueeze(0)
        r = (dx * dx + dy * dy + a * a).sqrt()  # Regularized at short distance

        # Charge product
        charge_ij = charges.unsqueeze(1) * charges.unsqueeze(0)

        # H = -πJ Σ_{i<j} n_i n_j log(r/a)
        log_r = (r / a).log()
        H_matrix = -np.pi * J * charge_ij * log_r

        # Upper triangle only (i < j)
        mask = Tensor([[1.0 if j > i else 0.0 for j in range(n)] for i in range(n)])
        H_int = (H_matrix * mask).sum()

        return H_int + E_core * n

    return H


def xy_vortex_dynamics(z: Tensor, charges: Tensor, J: float = 1.0,
                       dt: float = 0.01, temperature: float = 0.0,
                       gamma: float = 1.0, a: float = 1.0) -> Tensor:
    """
    Overdamped Langevin dynamics for XY vortices.

    dz_i/dt = -(1/γ) ∂H/∂z_i + √(2T/γ) η(t)

    Unlike point vortices in fluids (which have circulation-weighted
    symplectic structure), XY vortices follow standard overdamped dynamics.
    """
    n = charges.shape[0]
    z_np = z.numpy().reshape(n, 2)
    charges_np = charges.numpy()

    # Compute forces from Coulomb interaction
    # F_i = πJ Σ_j n_i n_j (r_ij / |r_ij|²)
    force = np.zeros_like(z_np)

    for i in range(n):
        for j in range(n):
            if i != j:
                dx = z_np[i, 0] - z_np[j, 0]
                dy = z_np[i, 1] - z_np[j, 1]
                r_sq = dx * dx + dy * dy + a * a
                # Force from j on i
                coeff = np.pi * J * charges_np[i] * charges_np[j] / r_sq
                force[i, 0] += coeff * dx
                force[i, 1] += coeff * dy

    # Overdamped: v = F / γ
    v = force / gamma

    # Thermal noise
    if temperature > 0:
        noise = np.sqrt(2 * temperature * dt / gamma) * np.random.randn(n, 2)
        z_new = z_np + dt * v + noise
    else:
        z_new = z_np + dt * v

    return Tensor(z_new.flatten().astype(np.float32))


class XYVortexGas:
    """
    Vortex gas model for the Kosterlitz-Thouless transition.

    Treats vortices as point particles with Coulomb (log) interactions:
        H = -πJ Σ_{i<j} n_i n_j log|r_ij| + E_core × N

    This is the "dual" picture of the XY model, making the KT physics
    transparent:
    - Vortex-antivortex pairs attract (like +/- charges in 2D Coulomb)
    - Binding energy ~ 2πJ log(R), entropy ~ 2 log(R)
    - Unbinding when entropy wins: T_KT = πJ/2

    Example:
        # Create a vortex-antivortex pair
        charges = Tensor([1.0, -1.0])
        z = Tensor([0.0, 1.0, 0.0, -1.0])  # (x0,y0,x1,y1)
        system = XYVortexGas(charges, J=1.0, temperature=0.3)

        # Below T_KT, pair stays bound
        for _ in range(1000):
            z = system.step(z, dt=0.01)
        print(f"Pair separation: {system.pair_separation(z):.2f}")
    """

    def __init__(self, charges: Tensor, J: float = 1.0, E_core: float = 0.0,
                 temperature: float = 0.0, gamma: float = 1.0, a: float = 1.0):
        """
        Initialize vortex gas.

        Args:
            charges: Vortex charges (+1 or -1)
            J: XY coupling constant
            E_core: Vortex core energy
            temperature: Temperature for Langevin dynamics
            gamma: Damping coefficient
            a: Short-distance cutoff (lattice spacing)
        """
        self.charges = charges
        self.n_vortices = charges.shape[0]
        self.J = J
        self.E_core = E_core
        self.temperature = temperature
        self.gamma = gamma
        self.a = a
        self.H = xy_vortex_hamiltonian(charges, J, E_core, a)

    def step(self, z: Tensor, dt: float = 0.01) -> Tensor:
        """Evolve vortex positions."""
        return xy_vortex_dynamics(z, self.charges, self.J, dt,
                                  self.temperature, self.gamma, self.a)

    def energy(self, z: Tensor) -> float:
        """Compute interaction energy."""
        return float(self.H(z).numpy())

    def pair_separation(self, z: Tensor, i: int = 0, j: int = 1) -> float:
        """Distance between two vortices."""
        pos = z.numpy().reshape(self.n_vortices, 2)
        return float(np.sqrt((pos[i, 0] - pos[j, 0])**2 + (pos[i, 1] - pos[j, 1])**2))

    def total_charge(self) -> int:
        """Total vorticity (should be 0 for neutrality)."""
        return int(self.charges.numpy().sum())

    def center_of_mass(self, z: Tensor) -> tuple[float, float]:
        """Center of mass of all vortices."""
        pos = z.numpy().reshape(self.n_vortices, 2)
        return float(pos[:, 0].mean()), float(pos[:, 1].mean())

    def evolve(self, z: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, list]:
        """Evolve and record history."""
        history = []

        for i in range(steps):
            if i % record_every == 0:
                pos = z.numpy().reshape(self.n_vortices, 2)
                history.append({
                    'positions': pos.copy(),
                    'energy': self.energy(z),
                    'separation': self.pair_separation(z) if self.n_vortices >= 2 else 0,
                })
            z = self.step(z, dt)

        pos = z.numpy().reshape(self.n_vortices, 2)
        history.append({
            'positions': pos.copy(),
            'energy': self.energy(z),
            'separation': self.pair_separation(z) if self.n_vortices >= 2 else 0,
        })

        return z, history


def create_vortex_pair(separation: float = 2.0) -> tuple[Tensor, Tensor]:
    """Create a bound vortex-antivortex pair."""
    charges = Tensor([1.0, -1.0])
    z = Tensor([0.0, separation/2, 0.0, -separation/2])
    return charges, z


def create_vortex_gas(n_pairs: int, box_size: float = 10.0) -> tuple[Tensor, Tensor]:
    """Create charge-neutral vortex gas with random positions."""
    charges = [1.0 if i % 2 == 0 else -1.0 for i in range(2 * n_pairs)]
    positions = np.random.uniform(-box_size/2, box_size/2, 4 * n_pairs)
    return Tensor(charges), Tensor(positions.astype(np.float32))


# Backward compatibility
def symplectic_step(q, p, force, dt=0.01, mass=1.0):
    """DEPRECATED: Use HamiltonianSystem."""
    return q + (p + force * dt) / mass * dt, p + force * dt

hamiltonian_step = leapfrog
hamiltonian_yoshida4 = yoshida4
