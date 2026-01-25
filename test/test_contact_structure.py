import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.conformal import ConformalStructure
from tinyphysics.structures.contact import LangevinStructure


def test_contact_structure_reduces_energy():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  s = Tensor([0.0])
  structure = ConformalStructure(alpha=0.5, use_contact=True)
  prog = compile_structure(state=(q, p, s), H=H, structure=structure)
  e0 = float(H(q, p).numpy())
  (q1, p1, _), _ = prog.evolve((q, p, s), 0.05, 50)
  e1 = float(H(q1, p1).numpy())
  assert e1 < e0


def test_contact_structure_diagnostics():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  s = Tensor([0.0])
  structure = ConformalStructure(alpha=0.2, use_contact=True)
  prog = compile_structure(state=(q, p, s), H=H, structure=structure, contact_diagnostics=True)
  (_, _, _), history = prog.evolve((q, p, s), 0.05, 3)
  q0, p0, s0, e0 = history[0]
  assert q0.shape == q.shape
  assert p0.shape == p.shape
  assert s0.shape == s.shape
  assert e0.numel() == 1


def test_langevin_structure_damps_energy():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  structure = LangevinStructure(gamma=0.5, kT=0.0, noise=False)
  prog = compile_structure(state=(q, p), H=H, structure=structure)
  e0 = float(H(q, p).numpy())
  (q1, p1), _ = prog.evolve((q, p), 0.05, 20)
  e1 = float(H(q1, p1).numpy())
  assert e1 < e0
