import numpy as np

from tinygrad.tensor import Tensor


def ewald_energy(q: Tensor, charges: Tensor, box: float, alpha: float, r_cut: float) -> Tensor:
  if q.ndim != 2 or q.shape[1] != 3:
    raise ValueError("q must be (N,3)")
  if charges.ndim != 1 or charges.shape[0] != q.shape[0]:
    raise ValueError("charges must be (N,)")
  diff = q[:, None, :] - q[None, :, :]
  diff = diff - (diff / box).round() * box
  dist2 = (diff * diff).sum(axis=-1)
  mask = 1.0 - Tensor.eye(q.shape[0], device=q.device, dtype=q.dtype)
  r = dist2.sqrt() + (1.0 - mask) * 1e9
  qiqj = charges[:, None] * charges[None, :]
  erfc = (-(alpha * r) * (alpha * r)).exp()
  energy = 0.5 * (qiqj * erfc / r * mask).sum()
  return energy


__all__ = ["ewald_energy"]
