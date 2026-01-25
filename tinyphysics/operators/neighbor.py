from __future__ import annotations
import numpy as np


def cell_list(pos: np.ndarray, box: float, r_cut: float):
  n = pos.shape[0]
  ncell = int(box / r_cut)
  ncell = max(ncell, 1)
  head = -np.ones((ncell, ncell, ncell), dtype=np.int32)
  nxt = -np.ones((n,), dtype=np.int32)
  inv = 1.0 / r_cut
  for i in range(n):
    cx = int(pos[i, 0] * inv) % ncell
    cy = int(pos[i, 1] * inv) % ncell
    cz = int(pos[i, 2] * inv) % ncell
    nxt[i] = head[cx, cy, cz]
    head[cx, cy, cz] = i
  return head, nxt, ncell


def neighbor_pairs(pos: np.ndarray, box: float, r_cut: float):
  head, nxt, ncell = cell_list(pos, box, r_cut)
  r2 = r_cut * r_cut
  pairs = []
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
                      pairs.append((i, j))
                  j = nxt[j]
          i = nxt[i]
  return pairs


def neighbor_forces(pos: np.ndarray, mass: np.ndarray, G: float, softening: float, box: float, r_cut: float):
  n = pos.shape[0]
  pairs = neighbor_pairs(pos, box, r_cut)
  forces = np.zeros_like(pos)
  eps2 = softening * softening
  for i, j in pairs:
    d = pos[j] - pos[i]
    d -= np.round(d / box) * box
    r2 = (d * d).sum() + eps2
    inv = 1.0 / (r2 * np.sqrt(r2))
    f = G * mass[i] * mass[j] * inv * d
    forces[i] += f
    forces[j] -= f
  return forces


__all__ = ["cell_list", "neighbor_pairs", "neighbor_forces"]
