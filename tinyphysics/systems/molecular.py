from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinygrad.helpers import getenv
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.operators.neighbor import neighbor_pairs
from tinyphysics.operators.tensor_neighbors import build_cell_table, _linear_cell_index, _round_nearest, autotune_max_per, CellTable

_OFFSETS_CACHE: dict[str, Tensor] = {}


def _lj_force_tensor(q: Tensor, sigma: float, epsilon: float, softening: float, box: float, r_cut: float,
                     force_shift: bool = False) -> Tensor:
  if q.ndim == 2:
    q = q.unsqueeze(0)
  diff = q[:, None, :, :] - q[:, :, None, :]
  diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1) + softening * softening
  n = dist2.shape[-1]
  eye = Tensor.eye(n, device=q.device, dtype=q.dtype).reshape(1, n, n)
  cut = (dist2 < (r_cut * r_cut)).cast(q.dtype)
  mask = cut * (1.0 - eye)
  inv_r2 = 1.0 / (dist2 + (1.0 - mask) * 1e9)
  sr2 = (sigma * sigma) * inv_r2
  sr6 = sr2 * sr2 * sr2
  force_scalar = 24.0 * epsilon * inv_r2 * (2.0 * sr6 * sr6 - sr6)
  if force_shift:
    rc2 = r_cut * r_cut
    inv_rc2 = 1.0 / rc2
    sr2c = (sigma * sigma) * inv_rc2
    sr6c = sr2c * sr2c * sr2c
    f_rc = 24.0 * epsilon * inv_rc2 * (2.0 * sr6c * sr6c - sr6c)
    inv_r = inv_r2.sqrt()
    force_scalar = force_scalar - f_rc * (r_cut * inv_r)
  f = force_scalar.unsqueeze(-1) * diff * mask.unsqueeze(-1)
  out = f.sum(axis=2)
  return out[0] if out.shape[0] == 1 else out


def lj_energy(q: Tensor, sigma: float, epsilon: float, softening: float, box: float, r_cut: float,
              periodic: bool = True, shift: bool = False) -> Tensor:
  if q.ndim == 2:
    q = q.unsqueeze(0)
  diff = q[:, None, :, :] - q[:, :, None, :]
  if periodic:
    diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1) + softening * softening
  n = dist2.shape[-1]
  eye = Tensor.eye(n, device=q.device, dtype=q.dtype).reshape(1, n, n)
  cut = (dist2 < (r_cut * r_cut)).cast(q.dtype)
  mask = cut * (1.0 - eye)
  inv_r2 = 1.0 / (dist2 + (1.0 - mask) * 1e9)
  sr2 = (sigma * sigma) * inv_r2
  sr6 = sr2 * sr2 * sr2
  lj = 4.0 * epsilon * (sr6 * sr6 - sr6)
  if shift:
    rc2 = r_cut * r_cut
    inv_rc2 = 1.0 / rc2
    sr2c = (sigma * sigma) * inv_rc2
    sr6c = sr2c * sr2c * sr2c
    shift_val = 4.0 * epsilon * (sr6c * sr6c - sr6c)
    lj = lj - shift_val
  return 0.5 * (lj * mask).sum()

def lj_pressure(q: Tensor, p: Tensor, sigma: float, epsilon: float, softening: float, box: float | Tensor,
                r_cut: float, periodic: bool = True) -> Tensor:
  if q.ndim == 2:
    q = q.unsqueeze(0)
  if p.ndim == 2:
    p = p.unsqueeze(0)
  diff = q[:, None, :, :] - q[:, :, None, :]
  if periodic:
    diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1) + softening * softening
  n = dist2.shape[-1]
  eye = Tensor.eye(n, device=q.device, dtype=q.dtype).reshape(1, n, n)
  cut = (dist2 < (r_cut * r_cut)).cast(q.dtype)
  mask = cut * (1.0 - eye)
  inv_r2 = 1.0 / (dist2 + (1.0 - mask) * 1e9)
  sr2 = (sigma * sigma) * inv_r2
  sr6 = sr2 * sr2 * sr2
  force_scalar = 24.0 * epsilon * inv_r2 * (2.0 * sr6 * sr6 - sr6)
  virial = 0.5 * (force_scalar * (diff * diff).sum(axis=-1) * mask).sum()
  K = 0.5 * (p * p).sum()
  if isinstance(box, Tensor):
    box_t = box
  else:
    box_t = Tensor([float(box)], device=q.device, dtype=q.dtype)
  V = box_t * box_t * box_t
  return (2.0 * K + virial) / (3.0 * V)


def _lj_force_from_pairs(q: Tensor, pairs: list[tuple[int, int]], sigma: float, epsilon: float,
                         softening: float, box: float, r_cut: float, force_shift: bool = False) -> Tensor:
  if q.ndim == 2:
    q = q.unsqueeze(0)
  b, n, _ = q.shape
  if len(pairs) == 0:
    out = q.zeros_like()
    return out[0] if out.shape[0] == 1 else out
  idx_i = Tensor([p[0] for p in pairs], device=q.device, dtype=dtypes.int32)
  idx_j = Tensor([p[1] for p in pairs], device=q.device, dtype=dtypes.int32)
  pos_flat = q.reshape(b * n, 3)
  gi = pos_flat.gather(0, idx_i.unsqueeze(-1).expand(idx_i.shape[0], 3))
  gj = pos_flat.gather(0, idx_j.unsqueeze(-1).expand(idx_j.shape[0], 3))
  d = gj - gi
  d = d - _round_nearest(d / box) * box
  dist2 = (d * d).sum(axis=-1) + softening * softening
  inv_r2 = 1.0 / dist2
  sr2 = (sigma * sigma) * inv_r2
  sr6 = sr2 * sr2 * sr2
  force_scalar = 24.0 * epsilon * inv_r2 * (2.0 * sr6 * sr6 - sr6)
  if force_shift:
    rc2 = r_cut * r_cut
    inv_rc2 = 1.0 / rc2
    sr2c = (sigma * sigma) * inv_rc2
    sr6c = sr2c * sr2c * sr2c
    f_rc = 24.0 * epsilon * inv_rc2 * (2.0 * sr6c * sr6c - sr6c)
    inv_r = inv_r2.sqrt()
    force_scalar = force_scalar - f_rc * (r_cut * inv_r)
  f = force_scalar.unsqueeze(-1) * d
  zeros = pos_flat.zeros_like()
  idx_i_exp = idx_i.unsqueeze(-1).expand(idx_i.shape[0], 3)
  idx_j_exp = idx_j.unsqueeze(-1).expand(idx_j.shape[0], 3)
  out = zeros.scatter_reduce(0, idx_i_exp, f, reduce="sum", include_self=True)
  out = out.scatter_reduce(0, idx_j_exp, -f, reduce="sum", include_self=True)
  out = out.reshape(b, n, 3)
  return out[0] if out.shape[0] == 1 else out


def _lj_force_tensor_bins(q: Tensor, sigma: float, epsilon: float, softening: float, box: float, r_cut: float,
                          max_per: int | None = None, cell_table=None, force_shift: bool = False) -> Tensor:
  if q.ndim == 3:
    outs = []
    for b in range(q.shape[0]):
      outs.append(_lj_force_tensor_bins(q[b], sigma, epsilon, softening, box, r_cut, max_per=max_per,
                                        cell_table=None, force_shift=force_shift))
    return Tensor.stack(outs, dim=0)
  n = int(q.shape[0])
  if n == 0:
    return q.zeros_like()
  if cell_table is not None:
    if cell_table.n != n or cell_table.box != box or cell_table.r_cut != r_cut:
      cell_table = None
  if cell_table is None:
    cell_table = build_cell_table(q, box, r_cut, max_per=max_per)
  if cell_table is None or cell_table.max_per == 0:
    return q.zeros_like()
  table = cell_table.table
  cell_to_seg = cell_table.cell_to_seg
  max_per = cell_table.max_per
  ncell = cell_table.ncell

  inv = 1.0 / r_cut
  coords = (q * inv).cast(dtypes.int32) % ncell

  cache_key = str(q.device)
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
    ], device=q.device, dtype=dtypes.int32)
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
  valid_flat = valid.reshape(n, k).cast(q.dtype)
  flat_idx = cand_flat.reshape(-1)
  pos_idx = flat_idx.reshape(-1, 1).expand(flat_idx.shape[0], 3)
  pos_j = q.gather(0, pos_idx).reshape(n, k, 3)
  pos_i = q.unsqueeze(1)
  diff = pos_j - pos_i
  diff = diff - _round_nearest(diff / box) * box
  dist2 = (diff * diff).sum(axis=-1)
  self_mask = (cand_flat == Tensor.arange(n, device=q.device, dtype=dtypes.int32).unsqueeze(1)).cast(q.dtype)
  cut_mask = (dist2 < (r_cut * r_cut)).cast(q.dtype)
  valid_w = valid_flat * (1.0 - self_mask) * cut_mask
  dist2 = dist2 + softening * softening
  inv_r2 = 1.0 / (dist2 + (1.0 - valid_w) * 1e9)
  sr2 = (sigma * sigma) * inv_r2
  sr6 = sr2 * sr2 * sr2
  force_scalar = 24.0 * epsilon * inv_r2 * (2.0 * sr6 * sr6 - sr6)
  if force_shift:
    rc2 = r_cut * r_cut
    inv_rc2 = 1.0 / rc2
    sr2c = (sigma * sigma) * inv_rc2
    sr6c = sr2c * sr2c * sr2c
    f_rc = 24.0 * epsilon * inv_rc2 * (2.0 * sr6c * sr6c - sr6c)
    inv_r = inv_r2.sqrt()
    force_scalar = force_scalar - f_rc * (r_cut * inv_r)
  f = (force_scalar * valid_w).unsqueeze(-1) * diff
  out = f.sum(axis=1)
  return out


@dataclass
class LennardJonesSystem:
  mass: Tensor
  sigma: float = 1.0
  epsilon: float = 1.0
  r_cut: float = 2.5
  softening: float = 1e-6
  box: float = 10.0
  method: str = "auto"  # auto|tensor|neighbor|tensor_bins
  max_per: int | None | str = None
  table_every: int | None = None
  _table_cache: CellTable | None = None
  _table_steps: int = 0
  force_shift: bool = False

  def _force(self, q: Tensor) -> Tensor:
    method = self.method
    if method == "auto":
      n = int(q.shape[0])
      if "CPU" in q.device.upper():
        method = "neighbor" if n >= 512 else "tensor"
      else:
        # On GPU, all-pairs is faster for N < 2048 due to cell-list overhead
        method = "tensor" if n < 2048 else "tensor_bins"
    if method == "neighbor" and "CPU" in q.device.upper():
      pairs = neighbor_pairs(q.detach().numpy(), self.box, self.r_cut)
      return _lj_force_from_pairs(q, pairs, self.sigma, self.epsilon, self.softening, self.box, self.r_cut,
                                  force_shift=self.force_shift)
    if method == "tensor_bins":
      max_per = self.max_per
      if isinstance(max_per, str) and max_per == "auto":
        max_per = autotune_max_per(q, self.box, self.r_cut)
      cell_table = None
      if self.table_every is not None and self.table_every > 0:
        if self._table_cache is None or (self._table_steps % self.table_every) == 0:
          self._table_cache = build_cell_table(q, self.box, self.r_cut, max_per=max_per)
        self._table_steps += 1
        cell_table = self._table_cache
      return _lj_force_tensor_bins(q, self.sigma, self.epsilon, self.softening, self.box, self.r_cut,
                                   max_per=max_per, cell_table=cell_table, force_shift=self.force_shift)
    return _lj_force_tensor(q, self.sigma, self.epsilon, self.softening, self.box, self.r_cut,
                            force_shift=self.force_shift)

  def split_ops(self):
    def kick(state, dt):
      q, p = state
      f = self._force(q)
      return q, p + dt * f

    def drift(state, dt):
      q, p = state
      if self.mass.ndim == 1:
        invm = (1.0 / self.mass).reshape(-1, 1)
      else:
        invm = (1.0 / self.mass).reshape(self.mass.shape[0], self.mass.shape[1], 1)
      return q + dt * (p * invm), p

    return [kick, drift]

  def compile(self, q: Tensor, p: Tensor, integrator: str = "split", split_schedule: str = "strang"):
    structure = CanonicalStructure(split_ops=self.split_ops())
    def H(qv, pv):
      return 0.5 * (pv * pv).sum()
    base = compile_structure(state=(q, p), H=H, structure=structure, integrator=integrator, split_schedule=split_schedule)

    class _NoVecProgram:
      def __init__(self, prog, split_ops):
        self._prog = prog
        self._split_ops = split_ops  # [kick, drift]

      def step(self, state, dt: float):
        return self._prog.step(state, dt)

      def evolve(self, state, dt: float, steps: int, **kwargs):
        record_every = kwargs.get("record_every", 1)
        realize_every = kwargs.get("realize_every", max(1, steps // 10))
        history = []
        cur = state

        # Use direct split ops for better GPU efficiency
        kick, drift = self._split_ops
        for i in range(steps):
          if i % record_every == 0:
            history.append(cur)

          # Strang splitting: kick(dt/2) -> drift(dt) -> kick(dt/2)
          cur = kick(cur, 0.5 * dt)
          cur = drift(cur, dt)
          cur = kick(cur, 0.5 * dt)

          # Periodic realize to avoid graph explosion
          if (i + 1) % realize_every == 0:
            cur[0].realize()
            cur[1].realize()

        history.append(cur)
        return cur, history

      def compile_unrolled_step(self, dt: float, unroll: int):
        def run(state):
          cur = state
          for _ in range(unroll):
            cur = self._prog.step(cur, dt)
          return cur
        return run

    return _NoVecProgram(base, self.split_ops())


__all__ = ["LennardJonesSystem", "lj_energy", "lj_pressure"]
