from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Protocol

from tinygrad.tensor import Tensor


class StructureKind(str, Enum):
  CANONICAL = "canonical"
  QUANTUM = "quantum"
  LIE_POISSON = "lie_poisson"
  DISSIPATIVE = "dissipative"


class Structure(Protocol):
  kind: StructureKind

  def bracket(self, state: Tensor | tuple[Tensor, ...], grad: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]: ...

  def split(self, H_func: Callable | None) -> list[Callable] | None: ...

  def constraints(self, state: Tensor | tuple[Tensor, ...]) -> Callable | None: ...


@dataclass
class StructureProgram:
  """Thin wrapper around a compiled program with a common interface."""
  program: Any

  def step(self, state, dt: float):
    return self.program.step(state, dt) if hasattr(self.program, "step") else self.program(state, dt)

  def evolve(self, state, dt: float, steps: int, **kwargs):
    """Evolve the system forward in time.

    Args:
        state: Current state (Tensor or tuple of Tensors)
        dt: Time step
        steps: Number of integration steps
        **kwargs: Additional arguments passed to underlying program

    Returns:
        (final_state, history) tuple
    """
    if hasattr(self.program, "evolve"):
      return self.program.evolve(state, dt, steps, **kwargs)
    raise ValueError("program does not support evolve")

  def compile_unrolled_step(self, dt: float, unroll: int):
    if hasattr(self.program, "compile_unrolled_step"):
      return self.program.compile_unrolled_step(dt, unroll)
    raise ValueError("program does not support compile_unrolled_step")


__all__ = ["StructureKind", "Structure", "StructureProgram"]
