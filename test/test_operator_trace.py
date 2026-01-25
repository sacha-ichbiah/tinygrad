import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.vorticity import VorticityStructure


def test_operator_trace_collects():
  w = Tensor(np.random.randn(8, 8).astype(np.float32))
  structure = VorticityStructure(N=8)
  trace: list[str] = []
  _ = structure._rhs_operator(w, trace)
  assert len(trace) > 0
