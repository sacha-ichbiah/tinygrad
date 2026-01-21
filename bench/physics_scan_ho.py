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


def bench_steps(system, q, p, steps, iters=30, warmup=5):
  def run():
    system.evolve_scan_kernel(q, p, dt=0.01, steps=steps)

  def run_tuned():
    system.evolve_scan_kernel(q, p, dt=0.01, steps=steps, scan_tune=True)

  base = bench(run, iters=iters, warmup=warmup)
  _ = bench(run_tuned, iters=3, warmup=1)  # populate cache
  tuned = bench(run_tuned, iters=iters, warmup=warmup)
  return base, tuned


def bench_hamiltonian(name, H):
  system = HamiltonianSystem(H, integrator="leapfrog")
  q = Tensor.arange(1024) * 0.01
  p = Tensor.arange(1024) * 0.02
  steps_list = (8, 32, 128, 512, 2048)
  print(name)
  for steps in steps_list:
    base, tuned = bench_steps(system, q, p, steps, iters=20, warmup=5)
    base_sps = steps / base
    tuned_sps = steps / tuned
    print(
      f"steps {steps:4d} | baseline {base*1e3:7.3f} ms {base_sps:9.0f} steps/s | tuned {tuned*1e3:7.3f} ms {tuned_sps:9.0f} steps/s"
    )


def main():
  def H_ho(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  def H_quartic(q, p):
    return (p * p).sum() / 2 + (q * q * q * q).sum() / 4

  bench_hamiltonian("harmonic oscillator", H_ho)
  bench_hamiltonian("quartic potential", H_quartic)


if __name__ == "__main__":
  main()
