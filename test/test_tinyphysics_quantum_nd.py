import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.quantum import QuantumHamiltonianCompiler, QuantumSplitOperatorND, gaussian_wavepacket


def _norm(psi: Tensor, dx_prod: float) -> float:
  prob = psi[..., 0] * psi[..., 0] + psi[..., 1] * psi[..., 1]
  return float((prob.sum() * dx_prod).numpy())


def test_wavepacket_norm_conserved_2d():
  n = 32
  dx = 0.2
  x = (Tensor.arange(n) - n / 2) * dx
  y = (Tensor.arange(n) - n / 2) * dx
  solver = QuantumSplitOperatorND((x, y), dt=0.01)
  psi_x = gaussian_wavepacket(x, x0=0.0, k0=1.0, sigma=1.0)
  psi_y = gaussian_wavepacket(y, x0=0.0, k0=0.5, sigma=1.0)
  ax = psi_x[..., 0].reshape(n, 1)
  bx = psi_x[..., 1].reshape(n, 1)
  ay = psi_y[..., 0].reshape(1, n)
  by = psi_y[..., 1].reshape(1, n)
  real = ax * ay - bx * by
  imag = ax * by + bx * ay
  psi = Tensor.stack([real, imag], dim=-1)
  psi = solver.step(psi)
  assert abs(_norm(psi, dx * dx) - 1.0) < 1e-3


def test_compiler_matches_split_operator():
  n = 32
  dx = 0.2
  x = (Tensor.arange(n) - n / 2) * dx
  y = (Tensor.arange(n) - n / 2) * dx
  solver = QuantumSplitOperatorND((x, y), dt=0.01)
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
  out_a = solver.step(psi)
  out_b = compiler.step(psi)
  diff = (out_a - out_b).abs().max().numpy()
  assert diff < 1e-5


def test_compiler_jit_step():
  n = 16
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
  step = compiler.compile()
  out_a = compiler.step(psi)
  out_b = step(psi)
  diff = (out_a - out_b).abs().max().numpy()
  assert diff < 1e-5


def test_compiler_unrolled_matches_iterated():
  n = 16
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
  step = compiler.compile_unrolled(3)
  out_b = step(psi)
  out_a = psi
  for _ in range(3):
    out_a = compiler.step(out_a)
  diff = (out_a - out_b).abs().max().numpy()
  assert diff < 1e-5


def test_wavepacket_norm_conserved_3d():
  n = 16
  dx = 0.3
  x = (Tensor.arange(n) - n / 2) * dx
  y = (Tensor.arange(n) - n / 2) * dx
  z = (Tensor.arange(n) - n / 2) * dx
  solver = QuantumSplitOperatorND((x, y, z), dt=0.01)
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
  psi = solver.step(psi)
  assert abs(_norm(psi, dx * dx * dx) - 1.0) < 1e-3
