import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)


from tinygrad.tensor import Tensor
from tinyphysics.operators.tensor_neighbors import neighbor_force_tensor_bins


def build_cell_list(pos: np.ndarray, box: float, cell_size: float):
  n = pos.shape[0]
  ncell = int(box / cell_size)
  ncell = max(ncell, 1)
  head = -np.ones((ncell, ncell, ncell), dtype=np.int32)
  nxt = -np.ones((n,), dtype=np.int32)
  inv = 1.0 / cell_size
  for i in range(n):
    cx = int(pos[i, 0] * inv) % ncell
    cy = int(pos[i, 1] * inv) % ncell
    cz = int(pos[i, 2] * inv) % ncell
    nxt[i] = head[cx, cy, cz]
    head[cx, cy, cz] = i
  return head, nxt, ncell


def count_pairs(pos: np.ndarray, box: float, r_cut: float):
  head, nxt, ncell = build_cell_list(pos, box, r_cut)
  r2 = r_cut * r_cut
  count = 0
  for cx in range(ncell):
    for cy in range(ncell):
      for cz in range(ncell):
        i = head[cx, cy, cz]
        while i != -1:
          for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
              for dz in (-1, 0, 1):
                nx = (cx + dx) % ncell
                ny = (cy + dy) % ncell
                nz = (cz + dz) % ncell
                j = head[nx, ny, nz]
                while j != -1:
                  if j > i:
                    d = pos[i] - pos[j]
                    d -= np.round(d / box) * box
                    if (d * d).sum() < r2:
                      count += 1
                  j = nxt[j]
          i = nxt[i]
  return count


def bench_neighbors(n: int = 4096, box: float = 10.0, r_cut: float = 1.0):
  rng = np.random.default_rng(0)
  pos = rng.random((n, 3)).astype(np.float32) * box
  t0 = time.time()
  count = count_pairs(pos, box, r_cut)
  dt = time.time() - t0
  return dt, count


def bench_tensor_bins(n: int = 2048, box: float = 10.0, r_cut: float = 1.0, max_per: int = 16):
  rng = np.random.default_rng(0)
  pos = Tensor(rng.random((n, 3)).astype(np.float32) * box)
  mass = Tensor(np.ones((n,), dtype=np.float32))
  t0 = time.time()
  out = neighbor_force_tensor_bins(pos, mass, 1.0, 1e-2, box, r_cut, max_per=max_per)
  _ = out.realize()
  dt = time.time() - t0
  return dt


if __name__ == "__main__":
  t, count = bench_neighbors()
  print(f"neighbors: {t:.4f}s, pairs={count}")
  if os.getenv("TINYGRAD_BENCH_TENSOR_BINS", "0"):
    t_bins = bench_tensor_bins()
    print(f"tensor_bins: {t_bins:.4f}s")
