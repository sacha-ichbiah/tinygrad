from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.physics import SymplecticProgram, compile_symplectic_program


@dataclass
class UniversalSymplecticCompiler:
  """Hamiltonian in, symplectic simulation out."""
  kind: str = "canonical"
  H: Callable | None = None
  integrator: str = "auto"
  policy: object | None = None
  constraint: Callable | None = None
  constraint_tol: float = 1e-9
  constraint_iters: int = 10
  program: SymplecticProgram | None = None

  def compile(self, sample_state: tuple[Tensor, Tensor] | Tensor | None = None, **kwargs) -> SymplecticProgram:
    self.program = compile_symplectic_program(
      self.kind,
      H=self.H,
      integrator=self.integrator,
      policy=self.policy,
      constraint=self.constraint,
      constraint_tol=self.constraint_tol,
      constraint_iters=self.constraint_iters,
      sample_state=sample_state if isinstance(sample_state, tuple) else None,
      **kwargs,
    )
    return self.program

  def step(self, state, dt: float):
    if self.program is None:
      self.compile(sample_state=state if isinstance(state, tuple) else None)
    return self.program.step(state, dt)

  def evolve(self, state, dt: float, steps: int, record_every: int = 1, unroll: int | None = None,
             vector_width: int | None = None):
    if self.program is None:
      self.compile(sample_state=state if isinstance(state, tuple) else None)
    return self.program.evolve(state, dt, steps, record_every=record_every, unroll=unroll, vector_width=vector_width)


def compile_universal(kind: str, H: Callable | None = None, **kwargs) -> SymplecticProgram:
  """Functional interface for the universal symplectic compiler."""
  return compile_symplectic_program(kind, H=H, **kwargs)
