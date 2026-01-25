import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import fft3d


def _bench(label: str, fn, iters: int = 3, warmup: int = 3):
  for _ in range(warmup):
    fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_fft3d(n: int, iters: int = 3, warmup: int = 3):
  x = Tensor(np.random.randn(n, n, n).astype(np.float32))
  def run():
    out = fft3d(x)
    out.realize()
  _bench(f"fft3d n={n}^3", run, iters=iters, warmup=warmup)


def main():
  for n in (16, 32, 64):
    bench_fft3d(n, iters=5, warmup=4)


if __name__ == "__main__":
  main()
