import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import fft_plan


def _bench(label: str, fn, iters: int = 5):
  fn()
  fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_fft_plan_1d(n: int, iters: int = 5):
  x = Tensor(np.random.randn(n, 2).astype(np.float32))
  plan = fft_plan(x, kind="1d")
  def run():
    out = plan(x)
    out.realize()
  _bench(f"fft_plan1d n={n}", run, iters=iters)


def bench_fft_plan_2d(n: int, iters: int = 3):
  x = Tensor(np.random.randn(n, n, 2).astype(np.float32))
  plan = fft_plan(x, kind="2d")
  def run():
    out = plan(x)
    out.realize()
  _bench(f"fft_plan2d n={n}x{n}", run, iters=iters)


def main():
  for n in (128, 256, 512):
    bench_fft_plan_1d(n, iters=3)
  for n in (64, 128):
    bench_fft_plan_2d(n, iters=2)


if __name__ == "__main__":
  main()
