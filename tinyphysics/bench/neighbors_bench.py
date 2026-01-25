import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)


from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinyphysics.operators.tensor_neighbors import neighbor_force_tensor_bins, _build_cell_table, _linear_cell_index, _round_nearest


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


def bench_tensor_bins_table(n: int = 512, box: float = 10.0, r_cut: float = 1.0, max_per: int = 16):
  rng = np.random.default_rng(0)
  pos = Tensor(rng.random((n, 3)).astype(np.float32) * box)
  t0 = time.time()
  _ = _build_cell_table(pos, box, r_cut, max_per=max_per)
  dt = time.time() - t0
  return dt


def bench_tensor_bins_force(n: int = 512, box: float = 10.0, r_cut: float = 1.0, max_per: int = 16):
  rng = np.random.default_rng(0)
  pos = Tensor(rng.random((n, 3)).astype(np.float32) * box)
  mass = Tensor(np.ones((n,), dtype=np.float32))
  t0 = time.time()
  table, cell_to_seg, max_per, ncell, _ = _build_cell_table(pos, box, r_cut, max_per=max_per)
  inv = 1.0 / r_cut
  coords = (pos * inv).cast(dtypes.int32) % ncell
  offsets = Tensor([
    [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
    [-1, 0, -1],  [-1, 0, 0],  [-1, 0, 1],
    [-1, 1, -1],  [-1, 1, 0],  [-1, 1, 1],
    [0, -1, -1],  [0, -1, 0],  [0, -1, 1],
    [0, 0, -1],   [0, 0, 0],   [0, 0, 1],
    [0, 1, -1],   [0, 1, 0],   [0, 1, 1],
    [1, -1, -1],  [1, -1, 0],  [1, -1, 1],
    [1, 0, -1],   [1, 0, 0],   [1, 0, 1],
    [1, 1, -1],   [1, 1, 0],   [1, 1, 1],
  ], device=pos.device, dtype=dtypes.int32)
  neigh_coords = (coords.unsqueeze(1) + offsets.reshape(1, 27, 3)) % ncell
  neigh_cell = _linear_cell_index(neigh_coords, ncell)
  seg = cell_to_seg.gather(0, neigh_cell.reshape(-1)).reshape(n, 27)
  seg_clamped = seg.maximum(0)
  flat = seg_clamped.reshape(-1)
  idx = flat.reshape(-1, 1).expand(flat.shape[0], max_per)
  cand = table.gather(0, idx).reshape(n, 27, max_per)
  valid = (seg >= 0).unsqueeze(-1) & (cand >= 0)
  cand = valid.where(cand, 0)
  k = 27 * max_per
  cand_flat = cand.reshape(n, k)
  valid_flat = valid.reshape(n, k).cast(pos.dtype)
  flat_idx = cand_flat.reshape(-1)
  pos_idx = flat_idx.reshape(-1, 1).expand(flat_idx.shape[0], 3)
  pos_j = pos.gather(0, pos_idx).reshape(n, k, 3)
  mass_j = mass.gather(0, flat_idx).reshape(n, k)
  pos_i = pos.unsqueeze(1)
  diff = pos_j - pos_i
  diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1)
  r2 = dist2 + 1e-2 * 1e-2
  self_mask = (cand_flat == Tensor.arange(n, device=pos.device, dtype=dtypes.int32).unsqueeze(1)).cast(pos.dtype)
  cut_mask = (dist2 < (r_cut * r_cut)).cast(pos.dtype)
  valid_w = valid_flat * (1.0 - self_mask) * cut_mask
  r2 = r2 + (1.0 - valid_w) * 1e9
  inv = 1.0 / (r2 * r2.sqrt())
  f = (mass_j * mass.unsqueeze(1) * inv) * valid_w
  out = (f.unsqueeze(-1) * diff).sum(axis=1)
  _ = out.realize()
  dt = time.time() - t0
  return dt


if __name__ == "__main__":
  t, count = bench_neighbors()
  print(f"neighbors: {t:.4f}s, pairs={count}")
  if os.getenv("TINYGRAD_BENCH_TENSOR_BINS", "0"):
    t_bins = bench_tensor_bins()
    print(f"tensor_bins: {t_bins:.4f}s")
  if os.getenv("TINYGRAD_BENCH_TENSOR_BINS_TABLE", "0"):
    t_table = bench_tensor_bins_table()
    print(f"tensor_bins_table: {t_table:.4f}s")
  if os.getenv("TINYGRAD_BENCH_TENSOR_BINS_FORCE", "0"):
    t_force = bench_tensor_bins_force()
    print(f"tensor_bins_force: {t_force:.4f}s")
