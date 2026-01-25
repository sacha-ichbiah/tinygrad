from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.core.structure import Structure


@dataclass
class PhysicalSystem:
  state: Tensor | tuple[Tensor, Tensor]
  H_func: Callable | None
  structure: Structure
  integrator: str = "auto"
  policy: object | None = None
  split_schedule: str = "strang"
  project_every: int | None = None
  contact_diagnostics: bool = False

  def compile(self):
    return compile_structure(
      state=self.state,
      H=self.H_func,
      structure=self.structure,
      integrator=self.integrator,
      policy=self.policy,
      split_schedule=self.split_schedule,
      project_every=self.project_every,
      contact_diagnostics=self.contact_diagnostics,
    )


__all__ = ["PhysicalSystem"]
