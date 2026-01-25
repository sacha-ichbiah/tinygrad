import math

from tinygrad.tensor import Tensor

from tinyphysics.operators.poisson import poisson_solve_fft2


class FieldOperator:
  """Lightweight operator wrapper for periodic grid fields."""
  @staticmethod
  def grad2(f: Tensor, L: float | None = None) -> tuple[Tensor, Tensor]:
    n = int(f.shape[0])
    dx = (L / n) if L is not None else 1.0
    fx = (f.roll(-1, 0) - f.roll(1, 0)) * (0.5 / dx)
    fy = (f.roll(-1, 1) - f.roll(1, 1)) * (0.5 / dx)
    return fx, fy

  @staticmethod
  def div2(u: Tensor, v: Tensor, L: float | None = None) -> Tensor:
    n = int(u.shape[0])
    dx = (L / n) if L is not None else 1.0
    du = (u.roll(-1, 0) - u.roll(1, 0)) * (0.5 / dx)
    dv = (v.roll(-1, 1) - v.roll(1, 1)) * (0.5 / dx)
    return du + dv

  @staticmethod
  def curl2(u: Tensor, v: Tensor, L: float | None = None) -> Tensor:
    n = int(u.shape[0])
    dx = (L / n) if L is not None else 1.0
    dv_dx = (v.roll(-1, 0) - v.roll(1, 0)) * (0.5 / dx)
    du_dy = (u.roll(-1, 1) - u.roll(1, 1)) * (0.5 / dx)
    return dv_dx - du_dy

  @staticmethod
  def laplacian2(f: Tensor, L: float | None = None) -> Tensor:
    n = int(f.shape[0])
    dx = (L / n) if L is not None else 1.0
    return (f.roll(1, 0) + f.roll(-1, 0) + f.roll(1, 1) + f.roll(-1, 1) - 4 * f) * (1.0 / (dx * dx))

  @staticmethod
  def poisson_solve2(f: Tensor, L: float = 2 * math.pi) -> Tensor:
    return poisson_solve_fft2(f, L=L)


__all__ = ["FieldOperator"]
