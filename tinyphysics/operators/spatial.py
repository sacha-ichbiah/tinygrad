import math

from tinygrad.tensor import Tensor

from tinyphysics.operators.poisson import poisson_solve_fft2
from tinyphysics.operators.operator import Operator


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


def grad2_op(L: float | None = None) -> Operator:
  return Operator("grad2", lambda f: FieldOperator.grad2(f, L=L))


def div2_op(L: float | None = None) -> Operator:
  return Operator("div2", lambda u, v: FieldOperator.div2(u, v, L=L))


def curl2_op(L: float | None = None) -> Operator:
  return Operator("curl2", lambda u, v: FieldOperator.curl2(u, v, L=L))


def laplacian2_op(L: float | None = None) -> Operator:
  return Operator("laplacian2", lambda f: FieldOperator.laplacian2(f, L=L))


def poisson_solve2_op(L: float = 2 * math.pi) -> Operator:
  return Operator("poisson_solve2", lambda f: FieldOperator.poisson_solve2(f, L=L))


def operator_signature(ops: list[str]) -> str:
  return "->".join(ops)


__all__ = [
  "FieldOperator",
  "grad2_op",
  "div2_op",
  "curl2_op",
  "laplacian2_op",
  "poisson_solve2_op",
  "operator_signature",
]
