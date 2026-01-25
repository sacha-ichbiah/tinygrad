import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.systems.electrostatics import ElectrostaticsSystem


def main():
  n = 64
  box = 8.0
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((n, 3)).astype(np.float32) * box)
  p = Tensor(rng.standard_normal((n, 3)).astype(np.float32) * 0.1)
  charges = Tensor(rng.standard_normal((n,)).astype(np.float32))

  system = ElectrostaticsSystem(charges=charges, box=box, method="pme", grid_n=16)
  prog = system.compile(q, p)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 5)
  print(q1.shape, p1.shape)


if __name__ == "__main__":
  main()
