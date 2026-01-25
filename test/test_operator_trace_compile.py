import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.vorticity import VorticityStructure


def test_operator_trace_compile_hook():
  w = Tensor(np.random.randn(8, 8).astype(np.float32))
  structure = VorticityStructure(N=8)
  prog = compile_structure(state=w, H=lambda x: (x * x).sum(), structure=structure, integrator="euler")
  assert hasattr(prog, "program")
