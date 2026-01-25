import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.systems.molecular import LennardJonesSystem


def main():
  n = 64
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((n, 3)).astype(np.float32) * 10.0)
  p = Tensor(rng.standard_normal((n, 3)).astype(np.float32) * 0.1)
  m = Tensor(np.ones((n,), dtype=np.float32))
  system = LennardJonesSystem(mass=m, box=10.0, r_cut=2.5, method="auto")
  prog = system.compile(q, p)
  (q, p), _ = prog.evolve((q, p), 0.005, 10)
  print("q mean:", q.mean().numpy(), "p mean:", p.mean().numpy())


if __name__ == "__main__":
  main()
