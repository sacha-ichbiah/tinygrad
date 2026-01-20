"""
Harmonic Oscillator Benchmark Matrix

Comprehensive overview of integrator x JIT x unroll combinations.
"""

import time
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


def harmonic_hamiltonian(k: float = 1.0, m: float = 1.0):
  def H(q, p):
    T = (p * p).sum() / (2 * m)
    V = k * (q * q).sum() / 2
    return T + V
  return H


def _parse_list_flag(args, name: str, default: list[str]) -> list[str]:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      raw = arg[len(prefix):]
      return [v for v in raw.split(",") if v]
  return default


def _parse_int_flag(args, name: str, default: int) -> int:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return int(arg[len(prefix):])
  return default


def _bench(system: HamiltonianSystem, steps: int, repeats: int, jit: bool, unroll: int) -> tuple[float, float]:
  def run_once() -> float:
    q = Tensor([1.0])
    p = Tensor([0.0])
    if unroll > 1:
      step = system.compile_unrolled_step(0.01, unroll)
      q, p = step(q, p)
      steps_per_call = unroll
    else:
      step = TinyJit(system.step) if jit else system.step
      if jit:
        q, p = step(q, p, 0.01)
      steps_per_call = 1

    start = time.perf_counter()
    for _ in range(steps // steps_per_call):
      if unroll > 1:
        q, p = step(q, p)
      else:
        q, p = step(q, p, 0.01)
    q.numpy()
    p.numpy()
    return time.perf_counter() - start

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  return best, steps / best


def main():
  import sys
  args = sys.argv[1:]

  integrators = _parse_list_flag(args, "integrators", ["euler", "leapfrog", "yoshida4"])
  unrolls = [int(v) for v in _parse_list_flag(args, "unrolls", ["1", "2", "4", "8", "16"])]
  steps = _parse_int_flag(args, "steps", 512)
  repeats = _parse_int_flag(args, "repeats", 3)
  include_jit = "--jit" in args

  print("=" * 72)
  print("HARMONIC OSCILLATOR BENCHMARK MATRIX (AUTOGRAD)")
  print("=" * 72)
  print(f"steps={steps} repeats={repeats} integrators={','.join(integrators)} unrolls={','.join(map(str, unrolls))} jit={include_jit}")
  print("-" * 72)
  print(f"{'integrator':12s} {'jit':5s} {'unroll':6s} {'ms':>10s} {'steps/s':>12s}")

  for integrator in integrators:
    H = harmonic_hamiltonian()
    system = HamiltonianSystem(H, integrator=integrator)

    for unroll in unrolls:
      if steps % unroll != 0:
        continue
      if unroll > 1:
        jit = True
      else:
        jit = include_jit

      elapsed, steps_per_s = _bench(system, steps, repeats, jit=jit, unroll=unroll)
      label_jit = "yes" if jit else "no"
      print(f"{integrator:12s} {label_jit:5s} {unroll:6d} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")


if __name__ == "__main__":
  main()
