import math
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.lie_poisson import SO3Structure
from tinyphysics.structures.contact import BerendsenBarostatStructure
from tinyphysics.systems.molecular import lj_pressure
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


def demo_barostat(dt: float = 0.01, steps: int = 5):
  q = Tensor(np.random.randn(32, 3).astype(np.float32))
  p = Tensor(np.random.randn(32, 3).astype(np.float32)) * 0.1
  box = Tensor([6.0], dtype=q.dtype)

  def H(qv, pv):
    return 0.5 * (pv * pv).sum()

  def pressure_fn(qv, pv, boxv):
    return lj_pressure(qv, pv, sigma=1.0, epsilon=1.0, softening=1e-6, box=boxv[0], r_cut=2.5, periodic=False)

  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, pressure_fn=pressure_fn)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (q, p, box), _ = prog.evolve((q, p, box), dt, steps)
  return q, p, box


if __name__ == "__main__":
  q, p = demo_canonical()
  L = demo_lie_poisson_so3()
  psi = demo_poisson_solve()
  qn, pn, box = demo_barostat()
  print(q.shape, p.shape, L.shape, psi.shape, qn.shape, pn.shape, box.shape)
