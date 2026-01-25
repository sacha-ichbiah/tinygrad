import time
import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.compiler import UniversalSymplecticCompiler
from tinygrad.physics import FieldOperator


def _bench(label: str, fn, iters: int = 5, warmup: int = 3):
  for _ in range(warmup):
    fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_canonical(n: int = 1024):
  def H(q, p):
    return 0.5 * (q*q).sum() + 0.5 * (p*p).sum()
  q = Tensor(np.random.randn(n).astype(np.float32))
  p = Tensor(np.random.randn(n).astype(np.float32))
  sim = UniversalSymplecticCompiler(kind="canonical", H=H, integrator="leapfrog")
  def run():
    sim.step((q, p), 0.01)
  _bench(f"canonical n={n}", run)


def bench_lie_poisson():
  def H(L):
    return 0.5 * (L*L).sum()
  L = Tensor(np.random.randn(3).astype(np.float32))
  sim = UniversalSymplecticCompiler(kind="lie_poisson", H=H, integrator="midpoint")
  def run():
    sim.step(L, 0.01)
  _bench("lie_poisson so3", run)


def bench_poisson(n: int = 128):
  w = Tensor(np.random.randn(n, n).astype(np.float32))
  def run():
    FieldOperator.poisson_solve2(w).realize()
  _bench(f"poisson_solve2 n={n}", run)


if __name__ == "__main__":
  bench_canonical()
  bench_lie_poisson()
  bench_poisson()
