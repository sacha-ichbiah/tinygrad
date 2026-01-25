from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.lie_poisson import LiePoissonStructure, SO3Structure


@dataclass
class LiePoissonSystem:
  H: Callable
  structure: LiePoissonStructure = field(default_factory=LiePoissonStructure)
  integrator: str = "midpoint"
  policy: object | None = None

  def compile(self, state: Tensor):
    return compile_structure(state=state, H=self.H, structure=self.structure, integrator=self.integrator, policy=self.policy)


def compile_lie_poisson(state: Tensor, H: Callable, J: Callable, **kwargs):
  return compile_structure(state=state, H=H, structure=LiePoissonStructure(J=J), **kwargs)


def compile_so3(state: Tensor, H: Callable, **kwargs):
  return compile_structure(state=state, H=H, structure=SO3Structure(), **kwargs)


__all__ = ["LiePoissonSystem", "compile_lie_poisson", "compile_so3", "SO3Structure"]
