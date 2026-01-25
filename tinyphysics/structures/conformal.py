from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.structure import StructureKind


@dataclass
class ConformalStructure:
  alpha: float = 0.0
  use_contact: bool = False
  kind: StructureKind = StructureKind.DISSIPATIVE

  def bracket(self, state: tuple[Tensor, Tensor], grad: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    dq, dp = grad
    return dp, -dq


  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state: tuple[Tensor, Tensor]) -> Callable | None:
    return None


__all__ = ["ConformalStructure"]
