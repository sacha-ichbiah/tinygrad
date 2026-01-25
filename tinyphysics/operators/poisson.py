import math
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.fft import rfft2d, irfft2d


def _complex_mul_real(z: Tensor, r: Tensor) -> Tensor:
  return Tensor.stack([z[..., 0] * r, z[..., 1] * r], dim=-1)


def _complex_mul_i(z: Tensor, r: Tensor, sign: float = 1.0) -> Tensor:
  if sign >= 0:
    return Tensor.stack([-z[..., 1] * r, z[..., 0] * r], dim=-1)
  return Tensor.stack([z[..., 1] * r, -z[..., 0] * r], dim=-1)


def poisson_solve_fft2(vorticity: Tensor, L: float = 2 * math.pi) -> Tensor:
  """Solve Laplacian(psi) = vorticity using FFT on device."""
  if vorticity.ndim != 2 or vorticity.shape[0] != vorticity.shape[1]:
    raise ValueError("vorticity must be square 2D grid")
  n = int(vorticity.shape[0])
  kx = np.fft.fftfreq(n, d=L / n) * 2 * math.pi
  ky = np.fft.rfftfreq(n, d=L / n) * 2 * math.pi
  KX, KY = np.meshgrid(kx, ky, indexing="ij")
  K2 = KX * KX + KY * KY
  K2[0, 0] = 1.0
  invK2 = 1.0 / K2
  invK2[0, 0] = 0.0
  K2_t = Tensor(invK2.astype(np.float32), device=vorticity.device, dtype=vorticity.dtype)
  w_hat = rfft2d(vorticity)
  psi_hat = _complex_mul_real(w_hat, -K2_t)
  psi = irfft2d(psi_hat, n=(n, n))
  return psi


def velocity_from_streamfunction_fft2(psi: Tensor, L: float = 2 * math.pi) -> tuple[Tensor, Tensor]:
  """Compute velocity field from streamfunction using FFT on device."""
  if psi.ndim != 2 or psi.shape[0] != psi.shape[1]:
    raise ValueError("psi must be square 2D grid")
  n = int(psi.shape[0])
  kx = np.fft.fftfreq(n, d=L / n) * 2 * math.pi
  ky = np.fft.rfftfreq(n, d=L / n) * 2 * math.pi
  KX, KY = np.meshgrid(kx, ky, indexing="ij")
  KX_t = Tensor(KX.astype(np.float32), device=psi.device, dtype=psi.dtype)
  KY_t = Tensor(KY.astype(np.float32), device=psi.device, dtype=psi.dtype)
  psi_hat = rfft2d(psi)
  dpsidx_hat = _complex_mul_i(psi_hat, KX_t, sign=1.0)
  dpsidy_hat = _complex_mul_i(psi_hat, KY_t, sign=1.0)
  dpsidx = irfft2d(dpsidx_hat, n=(n, n))
  dpsidy = irfft2d(dpsidy_hat, n=(n, n))
  u = dpsidy
  v = -dpsidx
  return u, v


__all__ = [
  "_complex_mul_real",
  "_complex_mul_i",
  "poisson_solve_fft2",
  "velocity_from_streamfunction_fft2",
]
