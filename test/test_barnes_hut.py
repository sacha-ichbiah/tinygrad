import numpy as np
from tinyphysics.operators.barnes_hut import barnes_hut_forces


def _naive_forces(pos: np.ndarray, mass: np.ndarray, G: float, softening: float):
  n = pos.shape[0]
  forces = np.zeros_like(pos)
  eps2 = softening * softening
  for i in range(n):
    for j in range(i + 1, n):
      d = pos[j] - pos[i]
      r2 = (d * d).sum() + eps2
      inv = 1.0 / (r2 * np.sqrt(r2))
      f = G * mass[i] * mass[j] * inv * d
      forces[i] += f
      forces[j] -= f
  return forces


def test_barnes_hut_matches_naive_small():
  rng = np.random.default_rng(0)
  n = 32
  pos = rng.random((n, 3)).astype(np.float64)
  mass = rng.random((n,)).astype(np.float64) + 0.1
  G = 1.0
  softening = 1e-2
  exact = _naive_forces(pos, mass, G, softening)
  approx = barnes_hut_forces(pos, mass, G, softening, theta=0.2, leaf_size=1)
  denom = np.linalg.norm(exact) + 1e-12
  rel = np.linalg.norm(approx - exact) / denom
  assert rel < 0.5
