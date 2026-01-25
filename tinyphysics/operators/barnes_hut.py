from __future__ import annotations
import numpy as np


class _Node:
  def __init__(self, center: np.ndarray, half: float, idxs: np.ndarray):
    self.center = center
    self.half = half
    self.idxs = idxs
    self.mass = 0.0
    self.com = np.zeros(3, dtype=np.float64)
    self.children: list[_Node] = []

  def is_leaf(self) -> bool:
    return len(self.children) == 0


def _build_tree(pos: np.ndarray, mass: np.ndarray, center: np.ndarray, half: float, idxs: np.ndarray, leaf_size: int):
  node = _Node(center, half, idxs)
  if idxs.size == 0:
    return node
  node.mass = float(mass[idxs].sum())
  if node.mass > 0:
    node.com = (pos[idxs] * mass[idxs, None]).sum(axis=0) / node.mass
  if idxs.size <= leaf_size:
    return node
  offsets = np.array([[dx, dy, dz] for dx in (-0.5, 0.5) for dy in (-0.5, 0.5) for dz in (-0.5, 0.5)])
  for off in offsets:
    c = center + off * half
    mask = np.all(np.abs(pos[idxs] - c) <= half * 0.5 + 1e-12, axis=1)
    child_idxs = idxs[mask]
    if child_idxs.size == 0:
      continue
    child = _build_tree(pos, mass, c, half * 0.5, child_idxs, leaf_size)
    node.children.append(child)
  return node


def barnes_hut_forces(pos: np.ndarray, mass: np.ndarray, G: float, softening: float,
                       theta: float = 0.5, leaf_size: int = 16):
  n = pos.shape[0]
  mins = pos.min(axis=0)
  maxs = pos.max(axis=0)
  center = 0.5 * (mins + maxs)
  half = float(np.max(maxs - mins) * 0.5 + 1e-6)
  root = _build_tree(pos, mass, center, half, np.arange(n), leaf_size)
  forces = np.zeros_like(pos)
  eps2 = softening * softening

  def walk(i: int, node: _Node):
    if node.mass == 0:
      return
    if node.is_leaf() and node.idxs.size <= 1 and node.idxs[0] == i:
      return
    d = node.com - pos[i]
    r2 = (d * d).sum() + eps2
    if node.is_leaf() or (node.half / np.sqrt(r2) < theta):
      inv = 1.0 / (r2 * np.sqrt(r2))
      forces[i] += G * mass[i] * node.mass * inv * d
      return
    for child in node.children:
      walk(i, child)

  for i in range(n):
    walk(i, root)
  return forces


__all__ = ["barnes_hut_forces"]
