from __future__ import annotations
import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinyphysics.operators.neighbor import neighbor_pairs

_OFFSETS_CACHE: dict[str, Tensor] = {}


def _linear_cell_index(coords: Tensor, ncell: int) -> Tensor:
  # coords: (B, N, 3) int
  return coords[..., 0] * (ncell * ncell) + coords[..., 1] * ncell + coords[..., 2]


def _round_nearest(x: Tensor) -> Tensor:
  return (x >= 0).where((x + 0.5).floor(), (x - 0.5).ceil())


def build_cell_bins(pos: Tensor, box: float, r_cut: float):
  if pos.ndim == 2:
    pos = pos.unsqueeze(0)
  ncell = int(box / r_cut)
  ncell = max(ncell, 1)
  inv = 1.0 / r_cut
  coords = (pos * inv).cast(dtypes.int32) % ncell
  cell_id = _linear_cell_index(coords, ncell)
  order = cell_id.argsort(dim=1)
  sorted_cells = cell_id.gather(1, order)
  return order, sorted_cells, ncell


def _build_cell_table(pos: Tensor, box: float, r_cut: float, max_per: int | None = None):
  n = int(pos.shape[0])
  order, cell_sorted, ncell = build_cell_bins(pos, box, r_cut)
  if order.ndim == 2:
    order = order[0]
    cell_sorted = cell_sorted[0]
  if n == 0:
    return None, None, 0, ncell, order
  diff = (cell_sorted[1:] != cell_sorted[:-1])
  seg = Tensor.zeros(1, device=pos.device, dtype=dtypes.int32).cat(diff.cast(dtypes.int32), dim=0).cumsum()
  seg_f = seg.cast(dtypes.float32)
  num_seg = int(seg_f.max().item()) + 1
  idx = Tensor.arange(n, device=pos.device, dtype=dtypes.int32)
  init = Tensor.full((num_seg,), float(n), device=pos.device, dtype=dtypes.float32)
  start = init.scatter_reduce(0, seg, idx.cast(dtypes.float32), reduce="amin", include_self=True)
  start_i = start.cast(dtypes.int32)
  slot = idx - start_i.gather(0, seg)
  slot_f = slot.cast(dtypes.float32)
  max_found = int(slot_f.max().item()) + 1
  if max_found < 1:
    max_found = 1
  if max_per is None:
    max_per = max_found
  else:
    max_per = int(max_per)
    if max_per < 1:
      max_per = 1
  if max_per is not None:
    slot = slot.minimum(max_per - 1)
  table = Tensor.full((num_seg * max_per,), -1, device=pos.device, dtype=dtypes.int32)
  table = table.scatter(0, seg * max_per + slot, order)
  table = table.reshape(num_seg, max_per)
  cell_unique = cell_sorted.gather(0, start_i)
  cell_to_seg = Tensor.full((ncell * ncell * ncell,), -1, device=pos.device, dtype=dtypes.int32)
  cell_to_seg = cell_to_seg.scatter(0, cell_unique, Tensor.arange(num_seg, device=pos.device, dtype=dtypes.int32))
  return table.realize(), cell_to_seg.realize(), max_per, ncell, order.realize()


def neighbor_force_tensor_bins(pos: Tensor, mass: Tensor, G: float, softening: float, box: float,
                               r_cut: float, max_per: int | None = None) -> Tensor:
  if pos.ndim == 3:
    outs = []
    for b in range(pos.shape[0]):
      mb = mass[b] if mass.ndim == 2 else mass
      outs.append(neighbor_force_tensor_bins(pos[b], mb, G, softening, box, r_cut))
    return Tensor.stack(outs, dim=0)
  if "CPU" in pos.device.upper():
    n = int(pos.shape[0])
    cpu_thresh = int(getenv("TINYGRAD_TENSOR_BINS_CPU_THRESHOLD", 512))
    if n <= cpu_thresh:
      return neighbor_force_tensor(pos, mass, G, softening, box, r_cut)
  if mass.ndim == 1:
    mass = mass
  n = int(pos.shape[0])
  if n == 0:
    return pos.zeros_like()
  table, cell_to_seg, max_per, ncell, _ = _build_cell_table(pos, box, r_cut, max_per=max_per)
  if table is None or max_per == 0:
    return pos.zeros_like()

  inv = 1.0 / r_cut
  coords = (pos * inv).cast(dtypes.int32) % ncell
  cell_id = _linear_cell_index(coords, ncell)

  cache_key = str(pos.device)
  offsets = _OFFSETS_CACHE.get(cache_key)
  if offsets is None:
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
    _OFFSETS_CACHE[cache_key] = offsets

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
  r2 = dist2 + softening * softening
  self_mask = (cand_flat == Tensor.arange(n, device=pos.device, dtype=dtypes.int32).unsqueeze(1)).cast(pos.dtype)
  cut_mask = (dist2 < (r_cut * r_cut)).cast(pos.dtype)
  valid_w = valid_flat * (1.0 - self_mask) * cut_mask
  r2 = r2 + (1.0 - valid_w) * 1e9
  inv = 1.0 / (r2 * r2.sqrt())
  f = G * (mass_j * mass.unsqueeze(1) * inv) * valid_w
  out = (f.unsqueeze(-1) * diff).sum(axis=1)
  return out


def autotune_max_per(pos: Tensor, box: float, r_cut: float, candidates: tuple[int, ...] = (8, 16, 32, 64)) -> int:
  if pos.ndim == 3:
    pos = pos[0]
  ncell = int(box / r_cut)
  ncell = max(ncell, 1)
  inv = 1.0 / r_cut
  coords = (pos * inv).cast(dtypes.int32) % ncell
  cell_id = _linear_cell_index(coords, ncell).reshape(-1)
  counts = Tensor.zeros(ncell * ncell * ncell, device=pos.device, dtype=dtypes.float32)
  ones = Tensor.ones(cell_id.shape[0], device=pos.device, dtype=dtypes.float32)
  counts = counts.scatter_reduce(0, cell_id, ones, reduce="sum", include_self=True)
  max_found = int(counts.max().item())
  for c in candidates:
    if max_found <= c:
      return c
  return max_found


def neighbor_force_tensor(pos: Tensor, mass: Tensor, G: float, softening: float, box: float, r_cut: float):
  if pos.ndim == 2:
    pos = pos.unsqueeze(0)
  if mass.ndim == 1:
    mass = mass.unsqueeze(0)
  order, cell_id, ncell = build_cell_bins(pos, box, r_cut)
  pos_s = pos.gather(1, order.unsqueeze(-1).expand(pos.shape[0], pos.shape[1], pos.shape[2]))
  mass_s = mass.gather(1, order)
  diff = pos_s[:, None, :, :] - pos_s[:, :, None, :]
  diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1)
  r2 = dist2 + softening * softening
  n = r2.shape[-1]
  eye = Tensor.eye(n, device=pos.device, dtype=pos.dtype).reshape(1, n, n)
  cut = (dist2 < (r_cut * r_cut)).cast(pos.dtype)
  maskf = cut * (1.0 - eye)
  r2 = r2 + (1.0 - maskf) * 1e9
  inv = 1.0 / (r2 * r2.sqrt())
  mij = mass_s[:, :, None] * mass_s[:, None, :]
  f = G * (mij * inv).unsqueeze(-1) * diff
  out_s = f.sum(axis=2)

  # scatter back to original order
  inv_order = order.argsort(dim=1)
  out = out_s.gather(1, inv_order.unsqueeze(-1).expand(out_s.shape[0], out_s.shape[1], out_s.shape[2]))
  return out[0] if out.shape[0] == 1 else out


def neighbor_force_from_pairs(pos: Tensor, mass: Tensor, pairs: list[tuple[int, int]],
                              G: float, softening: float, box: float | None = None) -> Tensor:
  if pos.ndim == 2:
    pos = pos.unsqueeze(0)
  if mass.ndim == 1:
    mass = mass.unsqueeze(0)
  b, n, _ = pos.shape
  if len(pairs) == 0:
    out = pos.zeros_like()
    return out[0] if out.shape[0] == 1 else out
  idx_i = Tensor([p[0] for p in pairs], device=pos.device, dtype=dtypes.int32)
  idx_j = Tensor([p[1] for p in pairs], device=pos.device, dtype=dtypes.int32)
  pos_flat = pos.reshape(b * n, 3)
  mass_flat = mass.reshape(b * n)
  gi = pos_flat.gather(0, idx_i.unsqueeze(-1).expand(idx_i.shape[0], 3))
  gj = pos_flat.gather(0, idx_j.unsqueeze(-1).expand(idx_j.shape[0], 3))
  mi = mass_flat.gather(0, idx_i)
  mj = mass_flat.gather(0, idx_j)
  d = gj - gi
  if box is not None:
    d = d - _round_nearest(d / box) * box
  r2 = (d * d).sum(axis=-1) + softening * softening
  inv = 1.0 / (r2 * r2.sqrt())
  f = G * (mi * mj * inv).unsqueeze(-1) * d
  zeros = pos_flat.zeros_like()
  idx_i_exp = idx_i.unsqueeze(-1).expand(idx_i.shape[0], 3)
  idx_j_exp = idx_j.unsqueeze(-1).expand(idx_j.shape[0], 3)
  out = zeros.scatter_reduce(0, idx_i_exp, f, reduce="sum", include_self=True)
  out = out.scatter_reduce(0, idx_j_exp, -f, reduce="sum", include_self=True)
  out = out.reshape(b, n, 3)
  return out[0] if out.shape[0] == 1 else out


__all__ = [
  "build_cell_bins",
  "neighbor_force_tensor",
  "neighbor_force_from_pairs",
  "neighbor_force_tensor_bins",
  "autotune_max_per",
]
