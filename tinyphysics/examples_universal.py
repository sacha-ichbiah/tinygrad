import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.lie_poisson import SO3Structure
from tinygrad.physics import FieldOperator


def demo_canonical(dt: float = 0.01, steps: int = 10):
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(128).astype(np.float32))
  p = Tensor(np.random.randn(128).astype(np.float32))
  prog = compile_structure(state=(q, p), H=H, structure=CanonicalStructure(), integrator="leapfrog")
  (q, p), _ = prog.evolve((q, p), dt, steps)
  return q, p


def demo_lie_poisson_so3(dt: float = 0.01, steps: int = 10):
  def H(L: Tensor):
    return 0.5 * (L * L).sum()
  L = Tensor(np.random.randn(3).astype(np.float32))
  prog = compile_structure(state=L, H=H, structure=SO3Structure(), integrator="midpoint")
  L, _ = prog.evolve(L, dt, steps)
  return L


def demo_poisson_solve():
  w = Tensor(np.random.randn(128, 128).astype(np.float32))
  psi = FieldOperator.poisson_solve2(w, L=2 * math.pi)
  return psi


if __name__ == "__main__":
  q, p = demo_canonical()
  L = demo_lie_poisson_so3()
  psi = demo_poisson_solve()
  print(q.shape, p.shape, L.shape, psi.shape)
