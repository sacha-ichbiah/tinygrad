from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure


@dataclass
class CanonicalSystem:
  H: Callable
  integrator: str = "auto"
  policy: object | None = None
  split_schedule: str = "strang"
  structure: CanonicalStructure = field(default_factory=CanonicalStructure)

  def compile(self, state: tuple[Tensor, Tensor]):
    return compile_structure(
      state=state,
      H=self.H,
      structure=self.structure,
      integrator=self.integrator,
      policy=self.policy,
      split_schedule=self.split_schedule,
    )


def compile_canonical(state: tuple[Tensor, Tensor], H: Callable, **kwargs):
  return compile_structure(state=state, H=H, structure=CanonicalStructure(), **kwargs)


__all__ = ["CanonicalSystem", "compile_canonical"]
