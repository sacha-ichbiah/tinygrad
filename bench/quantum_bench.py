import time
import os
import sys

ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
  sys.path.append(ROOT)

from tinygrad.tensor import Tensor
from tinyphysics.quantum import QuantumHamiltonianCompiler, gaussian_wavepacket


def _bench(label: str, fn, iters: int = 5, warmup: int = 3):
  for _ in range(warmup):
    fn()
  start = time.perf_counter()
  for _ in range(iters):
    fn()
  dt = (time.perf_counter() - start) / iters
  print(f"{label}: {dt*1e3:.3f} ms")


def bench_1d(n: int, steps: int):
  dx = 0.1
  x = (Tensor.arange(n) - n / 2) * dx
  compiler = QuantumHamiltonianCompiler((x,), dt=0.01)
  psi = gaussian_wavepacket(x, x0=0.0, k0=1.5, sigma=1.0)
  step = compiler.compile_unrolled(steps)
  def run():
    out = step(psi)
    out.realize()
  _bench(f"quantum1d n={n} steps={steps}", run, iters=3, warmup=2)


def bench_2d(n: int, steps: int):
  dx = 0.2
  x = (Tensor.arange(n) - n / 2) * dx
  y = (Tensor.arange(n) - n / 2) * dx
  compiler = QuantumHamiltonianCompiler((x, y), dt=0.01)
  psi_x = gaussian_wavepacket(x, x0=0.0, k0=1.0, sigma=1.0)
  psi_y = gaussian_wavepacket(y, x0=0.0, k0=0.5, sigma=1.0)
  ax = psi_x[..., 0].reshape(n, 1)
  bx = psi_x[..., 1].reshape(n, 1)
  ay = psi_y[..., 0].reshape(1, n)
  by = psi_y[..., 1].reshape(1, n)
  real = ax * ay - bx * by
  imag = ax * by + bx * ay
  psi = Tensor.stack([real, imag], dim=-1)
  step = compiler.compile_unrolled(steps)
  def run():
    out = step(psi)
    out.realize()
  _bench(f"quantum2d n={n} steps={steps}", run, iters=2, warmup=2)


def bench_3d(n: int, steps: int):
  dx = 0.3
  x = (Tensor.arange(n) - n / 2) * dx
  y = (Tensor.arange(n) - n / 2) * dx
  z = (Tensor.arange(n) - n / 2) * dx
  compiler = QuantumHamiltonianCompiler((x, y, z), dt=0.01)
  psi_x = gaussian_wavepacket(x, x0=0.0, k0=0.7, sigma=1.0)
  psi_y = gaussian_wavepacket(y, x0=0.0, k0=0.0, sigma=1.0)
  psi_z = gaussian_wavepacket(z, x0=0.0, k0=0.0, sigma=1.0)
  ax = psi_x[..., 0].reshape(n, 1, 1)
  bx = psi_x[..., 1].reshape(n, 1, 1)
  ay = psi_y[..., 0].reshape(1, n, 1)
  by = psi_y[..., 1].reshape(1, n, 1)
  az = psi_z[..., 0].reshape(1, 1, n)
  bz = psi_z[..., 1].reshape(1, 1, n)
  real_xy = ax * ay - bx * by
  imag_xy = ax * by + bx * ay
  real = real_xy * az - imag_xy * bz
  imag = real_xy * bz + imag_xy * az
  psi = Tensor.stack([real, imag], dim=-1)
  step = compiler.compile_unrolled(steps)
  def run():
    out = step(psi)
    out.realize()
  _bench(f"quantum3d n={n} steps={steps}", run, iters=1, warmup=1)


def main():
  bench_1d(512, steps=4)
  bench_2d(64, steps=3)
  bench_3d(16, steps=1)


if __name__ == "__main__":
  main()
