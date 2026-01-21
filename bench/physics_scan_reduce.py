import os
import time
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


def bench(fn, iters=50, warmup=5):
  for _ in range(warmup):
    fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  return (time.perf_counter() - start) / iters


def main():
  def H(q, p):
    qsum = q.sum()
    psum = p.sum()
    return (qsum * qsum) + (qsum * psum) + (psum * psum)

  system = HamiltonianSystem(H, integrator="leapfrog")
  q = Tensor.arange(1024) * 0.01
  p = Tensor.arange(1024) * 0.02

  def run():
    system.evolve_scan_kernel(q, p, dt=0.01, steps=8, coupled=True)

  t = bench(run)
  print(f"avg {t*1e3:.3f} ms")
  if os.getenv("TINYGRAD_COUPLED_REDUCE_TUNE", "0") == "1":
    print("reduce tune enabled")


if __name__ == "__main__":
  main()
