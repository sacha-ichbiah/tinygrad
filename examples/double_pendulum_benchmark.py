"""
Double Pendulum Benchmark (Level 1.2)

Micro-benchmark for the non-separable double pendulum Hamiltonian.
"""

import time
import numpy as np
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem
try:
  from examples.double_pendulum import double_pendulum_hamiltonian
except ImportError:
  from double_pendulum import double_pendulum_hamiltonian


def _parse_int_flag(args, name: str, default: int) -> int:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return int(arg[len(prefix):])
  return default


def _parse_float_flag(args, name: str, default: float) -> float:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return float(arg[len(prefix):])
  return default
def _parse_bool_flag(args, name: str) -> bool:
  flag = f"--{name}"
  if flag in args:
    return True
  for arg in args:
    if arg.startswith(flag + "="):
      return arg.split("=", 1)[1].lower() in ("1", "true", "yes", "y")
  return False


def _parse_integrator(args, default="implicit"):
  for arg in args:
    if not arg.startswith("--"):
      return arg
  return default


def _make_state():
  q = Tensor([np.pi / 2, 0.0], requires_grad=True)
  p = Tensor([0.0, 0.0], requires_grad=True)
  return q, p

def _energy(q, p, H):
  return float(H(q, p).numpy())

def _default_implicit_iters(dt: float) -> int:
  if dt <= 0.0025:
    return 2
  if dt <= 0.005:
    return 3
  if dt <= 0.01:
    return 3
  if dt <= 0.02:
    return 6
  return 8


def _parse_float_pair(args, name: str) -> tuple[float, float]|None:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      parts = arg[len(prefix):].split(",")
      if len(parts) == 2:
        return float(parts[0]), float(parts[1])
  return None


def _choose_implicit_iters(system, dt: float, steps: int, unroll: int, buffered: bool, functional: bool,
                           iters_min: int, iters_max: int,
                           target_drift: float, target_steps: float):
  if steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")
  rows = []
  for iters in range(iters_min, iters_max + 1):
    import os
    os.environ["TINYGRAD_IMPLICIT_ITERS"] = str(iters)
    q, p = _make_state()
    e0 = _energy(q, p, system.H)
    q_buf = p_buf = None
    if buffered:
      step = system.compile_unrolled_step_buffered(dt, unroll)
      q_buf = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
      p_buf = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
    elif functional:
      step = system.compile_unrolled_step_implicit(dt, unroll, iters)
    else:
      step = system.compile_unrolled_step(dt, unroll)
    start = time.perf_counter()
    for _ in range(steps // unroll):
      if buffered:
        q, p = step(q, p, q_buf, p_buf)
      else:
        q, p = step(q, p)
    q.numpy()
    p.numpy()
    elapsed = time.perf_counter() - start
    e1 = _energy(q, p, system.H)
    drift = abs(e1 - e0) / (abs(e0) + 1e-12)
    steps_s = steps / elapsed
    rows.append((iters, steps_s, drift))
  # pick smallest iters meeting drift/steps targets
  for iters, steps_s, drift in rows:
    if drift <= target_drift and (target_steps <= 0 or steps_s >= target_steps):
      return iters, rows
  # fallback: best drift, then best steps/s
  rows_sorted = sorted(rows, key=lambda r: (r[2], -r[1]))
  return rows_sorted[0][0], rows


def benchmark(integrator: str, steps: int, repeats: int, dt: float, jit: bool, unroll: int,
              implicit_iters: int, report_energy: bool, buffered: bool, functional: bool, mixed: bool):
  if unroll > 1 and steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")

  H = double_pendulum_hamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
  system = HamiltonianSystem(H, integrator=integrator)

  def run_once() -> tuple[float, float|None]:
    q, p = _make_state()
    e0 = _energy(q, p, H) if report_energy else None
    if unroll > 1:
      q_buf = p_buf = None
      if integrator == "implicit" and buffered:
        step = system.compile_unrolled_step_buffered(dt, unroll)
        q_buf = Tensor.empty(*q.shape, device=q.device, dtype=q.dtype)
        p_buf = Tensor.empty(*p.shape, device=p.device, dtype=p.dtype)
        q, p = step(q, p, q_buf, p_buf)
      elif integrator == "implicit" and functional:
        step = system.compile_unrolled_step_implicit(dt, unroll, implicit_iters)
        q, p = step(q, p)
      elif integrator == "implicit" and mixed:
        step = system.compile_implicit_step_fixed(dt, implicit_iters)
        q, p = step(q, p)
      else:
        step = system.compile_unrolled_step(dt, unroll)
        q, p = step(q, p)
      steps_per_call = unroll
    else:
      step = TinyJit(system.step) if jit else system.step
      if jit:
        q, p = step(q, p, dt)
      steps_per_call = 1

    start = time.perf_counter()
    for _ in range(steps // steps_per_call):
      if unroll > 1:
        if integrator == "implicit" and buffered:
          q, p = step(q, p, q_buf, p_buf)
        elif integrator == "implicit" and mixed:
          for _ in range(unroll):
            q, p = step(q, p)
        else:
          q, p = step(q, p)
      else:
        q, p = step(q, p, dt)
    q.numpy()
    p.numpy()
    energy_drift = None
    if report_energy:
      e1 = _energy(q, p, H)
      energy_drift = abs(e1 - e0) / (abs(e0) + 1e-12)
    return time.perf_counter() - start, energy_drift

  results = [run_once() for _ in range(repeats)]
  best = min(t for t, _ in results)
  best_drift = None
  if report_energy:
    best_drift = min((d for _, d in results if d is not None), default=None)
  label = "autograd"
  if unroll > 1:
    label += f" + unroll={unroll} + TinyJit"
  elif jit:
    label += " + TinyJit"

  print("=" * 60)
  print("DOUBLE PENDULUM BENCHMARK (AUTOGRAD)")
  print("=" * 60)
  print(f"Integrator: {integrator}, steps: {steps}, repeats: {repeats}, dt: {dt}, unroll: {unroll}, implicit-iters: {implicit_iters}")
  line = f"{label:28s}: {best*1e3:.2f} ms  ({steps/best:,.1f} steps/s)"
  if report_energy and best_drift is not None:
    line += f"  drift {best_drift:.2e}"
  print(line)


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  fast = _parse_bool_flag(args, "fast")
  buffered = _parse_bool_flag(args, "buffered")
  functional = _parse_bool_flag(args, "functional")
  mixed = _parse_bool_flag(args, "mixed")
  sweep_iters = _parse_bool_flag(args, "sweep-iters")
  drift_target = _parse_float_flag(args, "drift", 1e-4)
  steps_target = _parse_float_flag(args, "min-steps", 0.0)
  iters_range = _parse_float_pair(args, "iters-range")
  integrator = _parse_integrator(args, default="implicit")
  steps = _parse_int_flag(args, "steps", 2000)
  repeats = _parse_int_flag(args, "repeats", 3)
  dt = _parse_float_flag(args, "dt", 0.01)
  unroll = _parse_int_flag(args, "unroll", 1)
  implicit_iters = _parse_int_flag(args, "implicit-iters", 0)
  jit = "--jit" in args
  report_energy = _parse_bool_flag(args, "energy-drift")
  if fast:
    jit = True
    if unroll == 1:
      unroll = 4
    report_energy = True
  if integrator == "implicit":
    if sweep_iters:
      it_min, it_max = (2, 10)
      if iters_range is not None:
        it_min, it_max = int(iters_range[0]), int(iters_range[1])
      system = HamiltonianSystem(double_pendulum_hamiltonian(), integrator=integrator)
      implicit_iters, rows = _choose_implicit_iters(
        system, dt, steps, unroll, buffered, functional, it_min, it_max, drift_target, steps_target)
      print("-" * 60)
      print("IMPLICIT ITERS SWEEP")
      print("-" * 60)
      for iters, steps_s, drift in rows:
        print(f"iters {iters:2d} | {steps_s:9.1f} steps/s | drift {drift:.2e}")
      print(f"chosen implicit-iters: {implicit_iters}")
    if implicit_iters <= 0:
      implicit_iters = _default_implicit_iters(dt)
    if implicit_iters > 0:
      import os
      os.environ["TINYGRAD_IMPLICIT_ITERS"] = str(implicit_iters)
  benchmark(integrator=integrator, steps=steps, repeats=repeats, dt=dt, jit=jit,
            unroll=unroll, implicit_iters=implicit_iters, report_energy=report_energy,
            buffered=buffered, functional=functional, mixed=mixed)
