import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.structures.vorticity import VorticityStructure


def test_vorticity_operator_path_runs():
  w = Tensor(np.random.randn(16, 16).astype(np.float32))
  solver = VorticityStructure(N=16)
  w1, history = solver.evolve(w, dt=0.01, steps=3, record_every=1, use_operator=True)
  assert w1.shape == w.shape
  assert len(history) == 4
