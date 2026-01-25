import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.operators.pme import deposit_charges, pme_force


def test_pme_charge_conservation():
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((16, 3)).astype(np.float32) * 4.0)
  charges = Tensor(rng.standard_normal((16,)).astype(np.float32))
  rho = deposit_charges(q, charges, grid_n=8, box=4.0)
  total = float(rho.sum().numpy())
  expected = float(charges.sum().numpy())
  assert np.isfinite(total)
  assert np.allclose(total, expected, rtol=1e-5, atol=1e-6)


def test_pme_force_finite():
  rng = np.random.default_rng(1)
  q = Tensor(rng.random((8, 3)).astype(np.float32) * 4.0)
  charges = Tensor(rng.standard_normal((8,)).astype(np.float32))
  f = pme_force(q, charges, grid_n=8, box=4.0)
  assert f.shape == q.shape
  assert np.isfinite(f.numpy()).all()
