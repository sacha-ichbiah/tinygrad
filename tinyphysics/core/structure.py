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
  _fused: Any = None  # Lazy-initialized FusedEvolution

  def step(self, state, dt: float):
    return self.program.step(state, dt) if hasattr(self.program, "step") else self.program(state, dt)

  def evolve(self, state, dt: float, steps: int, fused: bool = False, **kwargs):
    """Evolve the system forward in time.

    Args:
        state: Current state (Tensor or tuple of Tensors)
        dt: Time step
        steps: Number of integration steps
        fused: If True, use fused evolution compiler (faster, builds single graph)
        **kwargs: Additional arguments passed to underlying program

    Returns:
        (final_state, history) tuple
    """
    # Check for fused mode
    if fused and hasattr(self.program, "step"):
      return self._evolve_fused(state, dt, steps, **kwargs)

    # Fallback to program's native evolve
    if hasattr(self.program, "evolve"):
      return self.program.evolve(state, dt, steps, **kwargs)
    raise ValueError("program does not support evolve")

  def _evolve_fused(self, state, dt: float, steps: int, record_every: int = 1, **kwargs):
    """Fused evolution - builds entire evolution as single UOp DAG."""
    if self._fused is None:
      from tinyphysics.core.fused import FusedEvolution
      max_fused = kwargs.pop("max_fused", 4096)
      object.__setattr__(self, "_fused", FusedEvolution(self.step, max_fused=max_fused))

    project_fn = getattr(self.program, "project", None)
    project_every = kwargs.get("project_every", 0)
    return self._fused.evolve(state, dt, steps, record_every, project_fn, project_every)

  def compile_unrolled_step(self, dt: float, unroll: int):
    if hasattr(self.program, "compile_unrolled_step"):
      return self.program.compile_unrolled_step(dt, unroll)
    raise ValueError("program does not support compile_unrolled_step")


__all__ = ["StructureKind", "Structure", "StructureProgram"]
