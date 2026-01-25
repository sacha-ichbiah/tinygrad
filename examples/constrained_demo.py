import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.physical import PhysicalSystem
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.constraints import ConstrainedStructure


def main():
  def H(q, p):
    return 0.5 * (p * p).sum()

  def constraint(q):
    return (q * q).sum() - 1.0

  q = Tensor(np.array([1.5, 0.0, 0.0], dtype=np.float32))
  p = Tensor(np.array([0.1, 0.0, 0.0], dtype=np.float32))
  sys = PhysicalSystem(
    state=(q, p),
    H_func=H,
    structure=ConstrainedStructure(CanonicalStructure(), constraint),
    project_every=1,
  )
  prog = sys.compile()
  (q1, p1), _ = prog.evolve((q, p), 0.1, 3)
  g = float(constraint(q1).numpy())
  print("constraint:", g)
  print("q:", q1.numpy())
  print("p:", p1.numpy())


if __name__ == "__main__":
  main()
