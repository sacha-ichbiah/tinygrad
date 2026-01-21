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


def _parse_integrator(args, default="implicit"):
  for arg in args:
    if not arg.startswith("--"):
      return arg
  return default


def _make_state():
  q = Tensor([np.pi / 2, 0.0], requires_grad=True)
  p = Tensor([0.0, 0.0], requires_grad=True)
  return q, p


def benchmark(integrator: str, steps: int, repeats: int, dt: float, jit: bool, unroll: int):
  if unroll > 1 and steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")

  H = double_pendulum_hamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)
  system = HamiltonianSystem(H, integrator=integrator)

  def run_once() -> float:
    q, p = _make_state()
    if unroll > 1:
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
        q, p = step(q, p)
      else:
        q, p = step(q, p, dt)
    q.numpy()
    p.numpy()
    return time.perf_counter() - start

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  label = "autograd"
  if unroll > 1:
    label += f" + unroll={unroll} + TinyJit"
  elif jit:
    label += " + TinyJit"

  print("=" * 60)
  print("DOUBLE PENDULUM BENCHMARK (AUTOGRAD)")
  print("=" * 60)
  print(f"Integrator: {integrator}, steps: {steps}, repeats: {repeats}, dt: {dt}, unroll: {unroll}")
  print(f"{label:28s}: {best*1e3:.2f} ms  ({steps/best:,.1f} steps/s)")


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  integrator = _parse_integrator(args, default="implicit")
  steps = _parse_int_flag(args, "steps", 2000)
  repeats = _parse_int_flag(args, "repeats", 3)
  dt = _parse_float_flag(args, "dt", 0.01)
  unroll = _parse_int_flag(args, "unroll", 1)
  jit = "--jit" in args
  benchmark(integrator=integrator, steps=steps, repeats=repeats, dt=dt, jit=jit, unroll=unroll)
