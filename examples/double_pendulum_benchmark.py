"""
Double Pendulum Benchmark (Level 1.2)

Micro-benchmark for the non-separable double pendulum Hamiltonian.
"""

import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import simulate_hamiltonian
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


def benchmark(steps: int, repeats: int, dt: float, report_energy: bool):
  if unroll > 1 and steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")

  H = double_pendulum_hamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81)

  def run_once() -> tuple[float, float|None]:
    q, p = _make_state()
    e0 = _energy(q, p, H) if report_energy else None
    start = time.perf_counter()
    q, p, _ = simulate_hamiltonian(H, q, p, dt=dt, steps=steps, record_every=steps)
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
  print("=" * 60)
  print("DOUBLE PENDULUM BENCHMARK (AUTOGRAD)")
  print("=" * 60)
  print(f"Integrator: auto, steps: {steps}, repeats: {repeats}, dt: {dt}")
  line = f"autograd: {best*1e3:.2f} ms  ({steps/best:,.1f} steps/s)"
  if report_energy and best_drift is not None:
    line += f"  drift {best_drift:.2e}"
  print(line)


if __name__ == "__main__":
  import sys
  args = sys.argv[1:]
  steps = _parse_int_flag(args, "steps", 2000)
  repeats = _parse_int_flag(args, "repeats", 3)
  dt = _parse_float_flag(args, "dt", 0.01)
  report_energy = _parse_bool_flag(args, "energy-drift")
  benchmark(steps=steps, repeats=repeats, dt=dt, report_energy=report_energy)
