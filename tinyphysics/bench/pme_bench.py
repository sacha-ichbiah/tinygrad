import time
import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.systems.electrostatics import direct_coulomb_force
from tinyphysics.operators.pme import pme_force


def bench_pme(n: int = 256, grid_n: int = 32, box: float = 10.0):
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((n, 3)).astype(np.float32) * box)
  charges = Tensor(rng.standard_normal((n,)).astype(np.float32))
  t0 = time.time()
  _ = pme_force(q, charges, grid_n=grid_n, box=box).realize()
  t_pme = time.time() - t0

  t0 = time.time()
  _ = direct_coulomb_force(q, charges, box=box).realize()
  t_direct = time.time() - t0
  return t_pme, t_direct


if __name__ == "__main__":
  t_pme, t_direct = bench_pme()
  print(f"pme: {t_pme:.4f}s")
  print(f"direct: {t_direct:.4f}s")
