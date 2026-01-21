import time
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


def bench(fn, iters=30, warmup=5):
  for _ in range(warmup):
    fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  return (time.perf_counter() - start) / iters


def bench_steps(system, q, p, steps, iters=30):
  def run():
    system.evolve_scan_kernel(q, p, dt=0.01, steps=steps)
  def run_tuned():
    system.evolve_scan_kernel(q, p, dt=0.01, steps=steps, scan_tune=True)

  base = bench(run, iters=iters)
  _ = bench(run_tuned, iters=3, warmup=1)  # populate cache
  tuned = bench(run_tuned, iters=iters)
  return base, tuned


def main():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  system = HamiltonianSystem(H, integrator="leapfrog")
  q = Tensor.arange(1024) * 0.01
  p = Tensor.arange(1024) * 0.02

  for steps in (8, 32, 128, 512, 2048):
    base, tuned = bench_steps(system, q, p, steps, iters=20)
    print(f"steps {steps:4d} | baseline {base*1e3:7.3f} ms | tuned {tuned*1e3:7.3f} ms")


if __name__ == "__main__":
  main()
