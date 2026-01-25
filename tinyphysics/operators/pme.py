import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
from tinyphysics.operators.poisson import poisson_solve_fft3


def deposit_charges(q: Tensor, charges: Tensor, grid_n: int, box: float) -> Tensor:
  if q.ndim != 2 or q.shape[1] != 3:
    raise ValueError("q must be (N,3)")
  if charges.ndim != 1 or charges.shape[0] != q.shape[0]:
    raise ValueError("charges must be (N,)")
  inv = grid_n / box
  coords = (q * inv).floor().cast(dtypes.int32) % grid_n
  linear = coords[:, 0] * (grid_n * grid_n) + coords[:, 1] * grid_n + coords[:, 2]
  grid = Tensor.zeros((grid_n * grid_n * grid_n,), device=q.device, dtype=q.dtype)
  grid = grid.scatter_reduce(0, linear, charges, reduce="sum", include_self=True)
  return grid.reshape(grid_n, grid_n, grid_n)


def field_from_potential(phi: Tensor, box: float) -> tuple[Tensor, Tensor, Tensor]:
  if phi.ndim != 3:
    raise ValueError("phi must be 3D")
  n = int(phi.shape[0])
  dx = box / n
  ex = -(phi.roll(-1, 0) - phi.roll(1, 0)) / (2.0 * dx)
  ey = -(phi.roll(-1, 1) - phi.roll(1, 1)) / (2.0 * dx)
  ez = -(phi.roll(-1, 2) - phi.roll(1, 2)) / (2.0 * dx)
  return ex, ey, ez


def pme_force(q: Tensor, charges: Tensor, grid_n: int, box: float) -> Tensor:
  rho = deposit_charges(q, charges, grid_n, box)
  phi = poisson_solve_fft3(rho, L=box)
  ex, ey, ez = field_from_potential(phi, box)
  coords = (q * (grid_n / box)).floor().cast(dtypes.int32) % grid_n
  idx = coords[:, 0] * (grid_n * grid_n) + coords[:, 1] * grid_n + coords[:, 2]
  ex_f = ex.reshape(-1).gather(0, idx)
  ey_f = ey.reshape(-1).gather(0, idx)
  ez_f = ez.reshape(-1).gather(0, idx)
  e = Tensor.stack([ex_f, ey_f, ez_f], dim=-1)
  return e * charges.reshape(-1, 1)


__all__ = ["deposit_charges", "field_from_potential", "pme_force"]
