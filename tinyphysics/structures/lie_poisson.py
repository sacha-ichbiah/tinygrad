from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.structure import StructureKind


def _cross(a: Tensor, b: Tensor) -> Tensor:
  ax, ay, az = a[0], a[1], a[2]
  bx, by, bz = b[0], b[1], b[2]
  return Tensor.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])


@dataclass
class LiePoissonStructure:
  J: Callable[[Tensor], Tensor] | None = None
  kind: StructureKind = StructureKind.LIE_POISSON

  def bracket(self, state: Tensor, grad: Tensor) -> Tensor:
    if self.J is None:
      raise ValueError("LiePoissonStructure requires J(state) or a subclass override")
    J_state = self.J(state)
    if callable(J_state):
      return J_state(grad)
    if getattr(J_state, "ndim", 0) == 2:
      return J_state @ grad
    return (J_state * grad).sum(axis=-1)


  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state: Tensor) -> Callable | None:
    return None


@dataclass
class SO3Structure(LiePoissonStructure):
  def bracket(self, state: Tensor, grad: Tensor) -> Tensor:
    return _cross(state, grad)



__all__ = ["LiePoissonStructure", "SO3Structure"]
