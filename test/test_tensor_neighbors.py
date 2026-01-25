import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.operators.tensor_neighbors import neighbor_force_tensor, neighbor_force_tensor_bins
from tinyphysics.operators.neighbor import neighbor_forces


def test_tensor_neighbor_force_shape():
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((8, 3)).astype(np.float32) * 10.0)
  m = Tensor(np.ones((8,), dtype=np.float32))
  f = neighbor_force_tensor(q, m, 1.0, 1e-2, 10.0, 1.0)
  assert f.shape == q.shape


def test_tensor_bins_matches_neighbor():
  rng = np.random.default_rng(1)
  q = rng.random((32, 3)).astype(np.float32) * 10.0
  m = np.ones((32,), dtype=np.float32)
  f_ref = neighbor_forces(q, m, 1.0, 1e-2, 10.0, 1.0)
  f_tb = neighbor_force_tensor_bins(Tensor(q), Tensor(m), 1.0, 1e-2, 10.0, 1.0, max_per=16).numpy()
  err = np.max(np.abs(f_tb - f_ref))
  assert err < 1e-2
