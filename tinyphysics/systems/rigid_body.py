from __future__ import annotations
from dataclasses import dataclass, field
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.lie_poisson import SO3Structure


@dataclass
class RigidBodySystem:
  H: Callable
  integrator: str = "midpoint"
  policy: object | None = None
  structure: SO3Structure = field(default_factory=SO3Structure)

  def compile(self, state: Tensor):
    return compile_structure(state=state, H=self.H, structure=self.structure, integrator=self.integrator, policy=self.policy)


__all__ = ["RigidBodySystem"]
