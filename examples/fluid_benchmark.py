"""Benchmark for fluid simulation and FFT performance.

This benchmark measures FFT and fluid simulation performance on different backends.
"""
import os
import time
import argparse
import numpy as np

# Set FFT fix before importing tinygrad
os.environ.setdefault("TINYGRAD_FFT_SPLIT_RADIX_THRESHOLD", "1024")

from tinygrad.tensor import Tensor
from tinygrad.fft import rfft2d, irfft2d

from tinyphysics.structures.vorticity import VorticityStructure
from tinyphysics.systems.vorticity import kelvin_helmholtz_ic, compute_enstrophy


def benchmark_fft(N: int = 128, warmup: int = 10, iters: int = 100) -> float:
  """Benchmark FFT roundtrip performance."""
  x = Tensor(np.random.randn(N, N).astype(np.float32))

  # Warmup
  for _ in range(warmup):
    y = irfft2d(rfft2d(x), n=(N, N))
    y.realize()

  # Benchmark
  start = time.perf_counter()
  for _ in range(iters):
    y = irfft2d(rfft2d(x), n=(N, N))
    y.realize()
  elapsed = time.perf_counter() - start

  ops_per_sec = iters / elapsed
  ms_per_op = elapsed * 1000 / iters
  print(f"FFT roundtrip N={N}: {ops_per_sec:.1f} ops/sec ({ms_per_op:.2f} ms/op)")
  return ops_per_sec


def benchmark_fluid(N: int = 128, warmup: int = 10, iters: int = 100, midpoint_iters: int = 3) -> float:
  """Benchmark fluid simulation step performance."""
  L = 2 * np.pi
  solver = VorticityStructure(N, L=L, dealias=2.0/3.0, dtype=np.float32)
  W = Tensor(kelvin_helmholtz_ic(N, L))

  # Warmup
  for _ in range(warmup):
    W = solver.step(W, dt=0.01, method='midpoint', iters=midpoint_iters)
  W.realize()

  # Benchmark
  start = time.perf_counter()
  for _ in range(iters):
    W = solver.step(W, dt=0.01, method='midpoint', iters=midpoint_iters)
  W.realize()
  elapsed = time.perf_counter() - start

  steps_per_sec = iters / elapsed
  ms_per_step = elapsed * 1000 / iters
  print(f"Fluid sim N={N} iters={midpoint_iters}: {steps_per_sec:.1f} steps/sec ({ms_per_step:.2f} ms/step)")
  return steps_per_sec


def benchmark_fluid_accuracy(N: int = 128, steps: int = 1000, dt: float = 0.01, midpoint_iters: int = 3):
  """Benchmark fluid simulation accuracy (enstrophy conservation)."""
  L = 2 * np.pi
  solver = VorticityStructure(N, L=L, dealias=2.0/3.0, dtype=np.float32)
  W = Tensor(kelvin_helmholtz_ic(N, L))

  Z_start = compute_enstrophy(W.numpy(), L, N)

  start = time.perf_counter()
  for _ in range(steps):
    W = solver.step(W, dt=dt, method='midpoint', iters=midpoint_iters)
  W.realize()
  elapsed = time.perf_counter() - start

  Z_end = compute_enstrophy(W.numpy(), L, N)
  drift = abs(Z_end - Z_start) / abs(Z_start)

  print(f"Fluid accuracy N={N} steps={steps} iters={midpoint_iters}:")
  print(f"  Enstrophy: {Z_start:.4f} -> {Z_end:.4f} (drift {drift:.2e})")
  print(f"  Time: {elapsed:.2f}s ({steps/elapsed:.1f} steps/sec)")
  return drift


def main():
  parser = argparse.ArgumentParser(description="Benchmark fluid simulation and FFT")
  parser.add_argument("--fft-only", action="store_true", help="Only run FFT benchmarks")
  parser.add_argument("--fluid-only", action="store_true", help="Only run fluid benchmarks")
  parser.add_argument("--accuracy", action="store_true", help="Run accuracy test")
  parser.add_argument("-N", type=int, default=128, help="Grid size")
  parser.add_argument("--iters", type=int, default=100, help="Benchmark iterations")
  parser.add_argument("--midpoint-iters", type=int, default=3, help="Midpoint iterations")
  args = parser.parse_args()

  device = os.environ.get("DEVICE", "CPU")
  print(f"Device: {device}")
  print("=" * 50)

  if not args.fluid_only:
    print("\nFFT Benchmarks:")
    for N in [64, 128, 256]:
      benchmark_fft(N, iters=args.iters)

  if not args.fft_only:
    print("\nFluid Simulation Benchmarks:")
    for N in [64, 128]:
      benchmark_fluid(N, iters=args.iters, midpoint_iters=args.midpoint_iters)

  if args.accuracy:
    print("\nAccuracy Test:")
    benchmark_fluid_accuracy(args.N, steps=1000, midpoint_iters=args.midpoint_iters)

  print("\n" + "=" * 50)
  print("Benchmark complete.")


if __name__ == "__main__":
  main()
