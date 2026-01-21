"""
Coupled scan kernel benchmark for TinyPhysics.

Focus: fused vs split scan kernel performance without history overhead.
"""

import os
import sys
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.append(ROOT)

from examples.kepler import kepler_hamiltonian
from examples.double_pendulum import double_pendulum_hamiltonian


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


def _parse_str_flag(args, name: str, default: str) -> str:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return arg[len(prefix):]
  return default


def _parse_bool_flag(args, name: str) -> bool:
  flag = f"--{name}"
  if flag in args:
    return True
  for arg in args:
    if arg.startswith(flag + "="):
      return arg.split("=", 1)[1].lower() in ("1", "true", "yes", "y")
  return False


def _make_state(hamiltonian: str):
  if hamiltonian == "kepler":
    GM, m, a = 1.0, 1.0, 1.0
    e = 0.6
    r_aphelion = a * (1 + e)
    v_aphelion = np.sqrt(GM * (1 - e) / (a * (1 + e)))
    q = Tensor([r_aphelion, 0.0], requires_grad=False)
    p = Tensor([0.0, m * v_aphelion], requires_grad=False)
    H = kepler_hamiltonian(GM=GM, m=m)
    return H, q, p
  if hamiltonian == "double_pendulum":
    q = Tensor([np.pi / 2, 0.0], requires_grad=False)
    p = Tensor([0.0, 0.0], requires_grad=False)
    H = double_pendulum_hamiltonian()
    return H, q, p
  if hamiltonian == "harmonic":
    def H(q, p):
      return (p * p).sum() * 0.5 + (q * q).sum() * 0.5
    q = Tensor([1.0], requires_grad=False)
    p = Tensor([0.0], requires_grad=False)
    return H, q, p
  raise ValueError(f"unknown hamiltonian: {hamiltonian}")


def _set_env(flag: str, enabled: bool):
  if enabled:
    os.environ[flag] = "1"
  else:
    os.environ.pop(flag, None)


def bench_scan(hamiltonian: str, dt: float, steps: int, unroll: int, vector_width: int,
               warmup: int, repeats: int, fallback: bool):
  H, q, p = _make_state(hamiltonian)
  system = HamiltonianSystem(H, integrator="leapfrog")
  if steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")
  used_fallback = False

  def run_once():
    q_tmp = q.detach().clone().realize()
    p_tmp = p.detach().clone().realize()
    start = time.perf_counter()
    nonlocal used_fallback
    try:
      q_out, p_out = system._evolve_scan_kernel_coupled_split(
        q_tmp, p_tmp, dt, steps, unroll_steps=unroll, vector_width=vector_width)
      Tensor.realize(q_out, p_out)
    except Exception:
      if not fallback:
        raise
      used_fallback = True
      q_out, p_out, _ = system.evolve_scan_kernel_coupled_fused(
        q_tmp, p_tmp, dt, steps, unroll_steps=unroll, vector_width=vector_width)
      Tensor.realize(q_out, p_out)
    return time.perf_counter() - start

  for _ in range(warmup):
    run_once()

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  steps_s = steps / best if best > 0 else float("inf")
  return best, steps_s, used_fallback


def main():
  import sys
  args = sys.argv[1:]
  hamiltonian = _parse_str_flag(args, "hamiltonian", "kepler")
  steps = _parse_int_flag(args, "steps", 8000)
  dt = _parse_float_flag(args, "dt", 0.01)
  unroll = _parse_int_flag(args, "unroll", 4)
  vector_width = _parse_int_flag(args, "vector-width", 1)
  warmup = _parse_int_flag(args, "warmup", 1)
  repeats = _parse_int_flag(args, "repeats", 3)
  fallback = _parse_bool_flag(args, "fallback")

  _set_env("TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL", _parse_bool_flag(args, "fused-vec"))
  _set_env("TINYGRAD_COUPLED_FUSED_TUNE", _parse_bool_flag(args, "fused-tune"))
  _set_env("TINYGRAD_COUPLED_REDUCE_TUNE", _parse_bool_flag(args, "reduce-tune"))
  _set_env("TINYGRAD_COUPLED_REDUCE_UNROLL_TUNE", _parse_bool_flag(args, "reduce-unroll-tune"))
  _set_env("TINYGRAD_COUPLED_REDUCE_TUNE_BOTH", _parse_bool_flag(args, "reduce-tune-both"))

  elapsed, steps_s, used_fallback = bench_scan(
    hamiltonian, dt, steps, unroll, vector_width, warmup, repeats, fallback)

  print("=" * 60)
  print("COUPLED SCAN BENCHMARK")
  print("=" * 60)
  print(f"Hamiltonian: {hamiltonian}")
  print(f"steps: {steps}, dt: {dt}, unroll: {unroll}, vector_width: {vector_width}")
  print(f"best: {elapsed*1e3:.2f} ms  ({steps_s:,.1f} steps/s)")
  if used_fallback:
    print("note: fallback path used (fused kernel failed)")


if __name__ == "__main__":
  main()
