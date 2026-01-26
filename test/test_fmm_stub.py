import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.operators.fmm import FMMStub


def test_fmm_stub_raises():
  q = Tensor(np.random.randn(4, 3).astype(np.float32))
  charges = Tensor(np.random.randn(4).astype(np.float32))
  fmm = FMMStub()
  try:
    _ = fmm.force(q, charges)
    assert False, "FMMStub should raise"
  except NotImplementedError:
    assert True
