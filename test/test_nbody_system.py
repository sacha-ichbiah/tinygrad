import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.systems.nbody import NBodySystem


def test_nbody_tensor_batch():
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((2, 8, 3)).astype(np.float32))
  p = Tensor(rng.random((2, 8, 3)).astype(np.float32))
  m = Tensor(np.ones((2, 8), dtype=np.float32))
  system = NBodySystem(mass=m, method="tensor")
  prog = system.compile(q, p)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 2)
  assert q1.shape == q.shape
  assert p1.shape == p.shape


def test_nbody_neighbor_batch():
  rng = np.random.default_rng(1)
  q = Tensor(rng.random((2, 6, 3)).astype(np.float32) * 10.0)
  p = Tensor(rng.random((2, 6, 3)).astype(np.float32))
  m = Tensor(np.ones((6,), dtype=np.float32))
  system = NBodySystem(mass=m, method="neighbor", r_cut=5.0, box=10.0)
  prog = system.compile(q, p)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 1)
  assert q1.shape == q.shape
  assert p1.shape == p.shape
