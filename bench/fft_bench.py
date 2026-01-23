import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import fft1d, fft2d


def _bench(label: str, fn, iters: int = 5):
  fn()
  fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_fft1d(n: int, device: str = "CPU", iters: int = 5):
  x = np.random.randn(n, 2).astype(np.float32)
  t = Tensor(x, device=device)
  def run():
    out = fft1d(t)
    if out.ndim:
      out.realize()
  _bench(f"fft1d n={n}", run, iters=iters)


def bench_fft2d(n: int, device: str = "CPU", iters: int = 3):
  x = np.random.randn(n, n, 2).astype(np.float32)
  t = Tensor(x, device=device)
  def run():
    out = fft2d(t)
    out.realize()
  _bench(f"fft2d n={n}x{n}", run, iters=iters)


def main():
  for n in (128, 256, 512):
    bench_fft1d(n, iters=3)
  for n in (64, 128):
    bench_fft2d(n)


if __name__ == "__main__":
  main()
