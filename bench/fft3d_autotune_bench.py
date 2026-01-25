import os
import time
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.fft import fft3d, _fft3d_plan_cache, _fft3d_contig_threshold_cache


def _bench(n: int, iters: int, warmup: int, thresholds: list[int]):
  os.environ.setdefault("TINYGRAD_FFT_JIT", "0")
  x = Tensor(np.random.randn(n, n, n).astype(np.float32))
  best_thr, best_time = None, float("inf")
  for thr in thresholds:
    _fft3d_plan_cache.clear()
    _fft3d_contig_threshold_cache.clear()
    os.environ["TINYGRAD_FFT_CONTIGUOUS_3D_THRESHOLD"] = str(thr)
    for _ in range(warmup):
      fft3d(x).realize()
    start = time.perf_counter()
    for _ in range(iters):
      fft3d(x).realize()
    dt = (time.perf_counter() - start) / iters
    if dt < best_time:
      best_time, best_thr = dt, thr
  print(f"n={n}^3: best_thr={best_thr}, {best_time*1e3:.3f} ms")


def main():
  sizes = os.getenv("TINYGRAD_FFT_3D_AUTOTUNE_SIZES", "8")
  ns = [int(v) for v in sizes.split(",") if v]
  iters = int(os.getenv("TINYGRAD_FFT_3D_AUTOTUNE_ITERS", "1"))
  warmup = int(os.getenv("TINYGRAD_FFT_3D_AUTOTUNE_WARMUP", "0"))
  thr_env = os.getenv("TINYGRAD_FFT_3D_AUTOTUNE_THRESHOLDS", "0,4096")
  thresholds = [int(v) for v in thr_env.split(",") if v]
  for n in ns:
    _bench(n, iters, warmup, thresholds)


if __name__ == "__main__":
  main()
