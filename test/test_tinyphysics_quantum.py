import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.quantum import QuantumSplitOperator1D, gaussian_wavepacket


def _norm(psi: Tensor, dx: float) -> float:
  prob = psi[..., 0] * psi[..., 0] + psi[..., 1] * psi[..., 1]
  return float((prob.sum() * dx).numpy())


def test_wavepacket_norm_conserved():
  n = 128
  dx = 0.1
  x = (Tensor.arange(n) - n / 2) * dx
  solver = QuantumSplitOperator1D(x, dt=0.01)
  psi = gaussian_wavepacket(x, x0=0.0, k0=1.5, sigma=1.0)
  for _ in range(5):
    psi = solver.step(psi)
  assert abs(_norm(psi, dx) - 1.0) < 1e-3


def test_imaginary_time_normalizes():
  n = 128
  dx = 0.1
  x = (Tensor.arange(n) - n / 2) * dx
  V = lambda x: 0.5 * x * x
  solver = QuantumSplitOperator1D(x, dt=0.02, V=V)
  psi = gaussian_wavepacket(x, x0=0.0, k0=0.0, sigma=1.0)
  psi = solver.step_imaginary(psi)
  assert abs(_norm(psi, dx) - 1.0) < 1e-3
