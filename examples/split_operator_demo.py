import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure


def harmonic_split_ops(k: float = 1.0, m: float = 1.0):
  def kick(state, dt):
    q, p = state
    return q, p - dt * k * q

  def drift(state, dt):
    q, p = state
    return q + dt * (p / m), p

  return [kick, drift]


def run_demo():
  q = Tensor(np.random.randn(128).astype(np.float32))
  p = Tensor(np.random.randn(128).astype(np.float32))
  structure = CanonicalStructure(split_ops=harmonic_split_ops())
  prog = compile_structure(state=(q, p), H=None, structure=structure, integrator="split", split_schedule="strang")
  (q, p), _ = prog.evolve((q, p), 0.01, 100)
  print(q.shape, p.shape)


if __name__ == "__main__":
  run_demo()
