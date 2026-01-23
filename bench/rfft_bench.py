import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import rfft1d, rfft2d


def _bench(label: str, fn, iters: int = 5):
  fn()
  fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_rfft1d(n: int, iters: int = 5):
  x = Tensor(np.random.randn(n).astype(np.float32))
  def run():
    out = rfft1d(x)
    out.realize()
  _bench(f"rfft1d n={n}", run, iters=iters)


def bench_rfft2d(n: int, iters: int = 3):
  x = Tensor(np.random.randn(n, n).astype(np.float32))
  def run():
    out = rfft2d(x)
    out.realize()
  _bench(f"rfft2d n={n}x{n}", run, iters=iters)


def main():
  for n in (128, 256, 512):
    bench_rfft1d(n, iters=3)
  for n in (64, 128):
    bench_rfft2d(n, iters=2)


if __name__ == "__main__":
  main()
