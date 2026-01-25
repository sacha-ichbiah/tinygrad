import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure


def _harmonic_split_ops(k: float = 1.0, m: float = 1.0):
  def kick(state, dt):
    q, p = state
    return q, p - dt * k * q

  def drift(state, dt):
    q, p = state
    return q + dt * (p / m), p

  return [kick, drift]


def _energy(q: Tensor, p: Tensor):
  return float((0.5 * (q * q).sum() + 0.5 * (p * p).sum()).numpy())


def test_split_schedule_strang_vs_yoshida():
  q = Tensor(np.random.randn(16).astype(np.float32))
  p = Tensor(np.random.randn(16).astype(np.float32))
  structure = CanonicalStructure(split_ops=_harmonic_split_ops())
  def H(qv, pv):
    return 0.5 * (qv * qv).sum() + 0.5 * (pv * pv).sum()
  prog_strang = compile_structure(state=(q, p), H=H, structure=structure, integrator="split", split_schedule="strang")
  prog_y4 = compile_structure(state=(q, p), H=H, structure=structure, integrator="split", split_schedule="yoshida4")

  q_s, p_s = q, p
  q_y, p_y = q, p
  e0 = _energy(q, p)
  for _ in range(50):
    (q_s, p_s), _ = prog_strang.evolve((q_s, p_s), 0.01, 1, record_every=1)
    (q_y, p_y), _ = prog_y4.evolve((q_y, p_y), 0.01, 1, record_every=1)

  e_s = abs(_energy(q_s, p_s) - e0)
  e_y = abs(_energy(q_y, p_y) - e0)
  assert e_y <= e_s + 1e-4
