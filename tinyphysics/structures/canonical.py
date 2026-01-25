from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.structure import StructureKind


@dataclass
class CanonicalStructure:
  split_ops: list[Callable] | None = None
  kind: StructureKind = StructureKind.CANONICAL

  def bracket(self, state: tuple[Tensor, Tensor], grad: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    dq, dp = grad
    return dp, -dq


  def split(self, H_func: Callable | None) -> list[Callable] | None:
    if self.split_ops is not None:
      return list(self.split_ops)
    if H_func is not None and hasattr(H_func, "split_ops"):
      ops = getattr(H_func, "split_ops")
      if isinstance(ops, (list, tuple)) and len(ops) > 0:
        return list(ops)
    return None

  def constraints(self, state: tuple[Tensor, Tensor]) -> Callable | None:
    return None


__all__ = ["CanonicalStructure"]
