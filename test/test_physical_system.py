import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.physical import PhysicalSystem
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.constraints import ConstrainedStructure


def test_physical_system_canonical():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(8).astype(np.float32))
  p = Tensor(np.random.randn(8).astype(np.float32))
  sys = PhysicalSystem(state=(q, p), H_func=H, structure=CanonicalStructure())
  prog = sys.compile()
  (q1, p1), _ = prog.evolve((q, p), 0.01, 10)
  assert q1.shape == q.shape
  assert p1.shape == p.shape


def test_physical_system_constrained():
  def H(q, p):
    return 0.5 * (p * p).sum()

  def constraint(q):
    return (q * q).sum() - 1.0

  q = Tensor([1.5, 0.0, 0.0])
  p = Tensor([0.1, 0.0, 0.0])
  sys = PhysicalSystem(state=(q, p), H_func=H, structure=ConstrainedStructure(CanonicalStructure(), constraint))
  prog = sys.compile()
  (q1, p1), _ = prog.evolve((q, p), 0.1, 1)
  assert q1.shape == q.shape
  assert p1.shape == p.shape
  assert abs(float(constraint(q1).numpy())) < 1e-5
