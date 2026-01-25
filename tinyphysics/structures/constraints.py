from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.structure import Structure, StructureKind


@dataclass
class ConstrainedStructure:
  base: Structure
  constraint_fn: Callable[[Tensor | tuple[Tensor, ...]], Tensor]

  @property
  def kind(self) -> StructureKind:
    return self.base.kind

  def bracket(self, state, grad):
    return self.base.bracket(state, grad)

  def split(self, H_func: Callable | None):
    return self.base.split(H_func)

  def constraints(self, state):
    return self.constraint_fn


__all__ = ["ConstrainedStructure"]
