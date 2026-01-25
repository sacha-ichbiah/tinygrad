import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.systems.electrostatics import direct_coulomb_force
from tinyphysics.operators.pme import pme_force


def test_electrostatics_force_correlation():
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((12, 3)).astype(np.float32) * 4.0)
  charges = Tensor(rng.standard_normal((12,)).astype(np.float32))
  f_direct = direct_coulomb_force(q, charges, box=4.0).numpy()
  f_pme = pme_force(q, charges, grid_n=8, box=4.0).numpy()
  corr = (f_direct * f_pme).sum()
  assert np.isfinite(corr)
