import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.operators.spatial import grad2_op, div2_op, curl2_op, laplacian2_op, poisson_solve2_op


def test_operator_wrappers_shapes():
  f = Tensor(np.random.randn(8, 8).astype(np.float32))
  u = Tensor(np.random.randn(8, 8).astype(np.float32))
  v = Tensor(np.random.randn(8, 8).astype(np.float32))
  gx, gy = grad2_op()(f)
  div = div2_op()(u, v)
  curl = curl2_op()(u, v)
  lap = laplacian2_op()(f)
  psi = poisson_solve2_op()(f)
  assert gx.shape == f.shape
  assert gy.shape == f.shape
  assert div.shape == f.shape
  assert curl.shape == f.shape
  assert lap.shape == f.shape
  assert psi.shape == f.shape
