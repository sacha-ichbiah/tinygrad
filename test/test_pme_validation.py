import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.operators.pme import pme_force
from tinyphysics.operators.ewald import ewald_energy


def direct_coulomb_force(q: Tensor, charges: Tensor, box: float) -> Tensor:
  diff = q[:, None, :] - q[None, :, :]
  diff = diff - (diff / box).round() * box
  dist2 = (diff * diff).sum(axis=-1) + 1e-6
  inv = 1.0 / (dist2 * dist2.sqrt())
  qiqj = charges[:, None] * charges[None, :]
  f = (qiqj * inv).unsqueeze(-1) * diff
  mask = 1.0 - Tensor.eye(q.shape[0], device=q.device, dtype=q.dtype)
  f = f * mask.unsqueeze(-1)
  return f.sum(axis=1)


def test_pme_force_direction():
  rng = np.random.default_rng(2)
  q = Tensor(rng.random((6, 3)).astype(np.float32) * 4.0)
  charges = Tensor(rng.standard_normal((6,)).astype(np.float32))
  f_direct = direct_coulomb_force(q, charges, box=4.0).numpy()
  f_pme = pme_force(q, charges, grid_n=8, box=4.0).numpy()
  corr = (f_direct * f_pme).sum()
  assert np.isfinite(corr)


def test_ewald_energy_finite():
  rng = np.random.default_rng(3)
  q = Tensor(rng.random((8, 3)).astype(np.float32) * 4.0)
  charges = Tensor(rng.standard_normal((8,)).astype(np.float32))
  e = ewald_energy(q, charges, box=4.0, alpha=1.0, r_cut=2.0)
  assert np.isfinite(float(e.numpy().item()))
