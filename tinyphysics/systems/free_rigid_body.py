"""
Free Rigid Body System - Blueprint-compliant wrapper for SO(3) dynamics.

The free rigid body evolves via Lie-Poisson mechanics:
    dL/dt = L × ω  where ω = I⁻¹L

This is the SO(3) bracket structure applied to the rigid body Hamiltonian:
    H(L) = 0.5 * L · (I⁻¹ L)

Conservation laws:
- Energy H is conserved (Hamiltonian)
- Casimir |L|² is conserved (for any Hamiltonian on so(3)*)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.physics import RigidBodySystem as _RigidBodySystem
from tinyphysics.core.structure import StructureKind, StructureProgram
from tinyphysics.structures.lie_poisson import SO3Structure


@dataclass
class FreeRigidBodyStructure:
  """
  Lie-Poisson structure for free rigid body on SO(3).

  The bracket is the standard SO(3) bracket: {f, g} = L · (∇f × ∇g)
  Which gives Euler's equations: dL/dt = L × ∇H
  """
  I: Tensor
  kind: StructureKind = StructureKind.LIE_POISSON

  def bracket(self, state: Tensor, grad: Tensor) -> Tensor:
    """SO(3) bracket: L × grad(H)"""
    return state.cross(grad)

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state: Tensor) -> Callable | None:
    return None


@dataclass
class FreeRigidBodySystem:
  """
  Blueprint-compliant wrapper for free rigid body simulation.

  Combines Lie-Poisson dynamics for angular momentum L with
  quaternion kinematics for orientation tracking.

  Example:
      system = FreeRigidBodySystem(I=Tensor([1.0, 2.0, 3.0]))
      prog = system.compile()
      (L, q), history = prog.evolve((L, q), dt=0.01, steps=1000)

  The intermediate axis (I2) exhibits the Tennis Racket / Dzhanibekov effect.
  """
  I: Tensor
  integrator: str = "auto"
  policy: object | None = None

  def hamiltonian(self, L: Tensor) -> Tensor:
    """Rigid body Hamiltonian: H = 0.5 * L · (I⁻¹ L)"""
    I_inv = 1.0 / self.I
    return 0.5 * (L * L * I_inv).sum()

  def energy(self, L: Tensor) -> float:
    """Compute energy for given angular momentum."""
    return float(self.hamiltonian(L).numpy())

  def casimir(self, L: Tensor) -> float:
    """Compute Casimir |L|² (conserved for any Hamiltonian on so(3)*)."""
    return float((L * L).sum().numpy())

  def compile(self) -> StructureProgram:
    """Compile the rigid body system into a StructureProgram."""
    system = _RigidBodySystem(
      I=self.I,
      integrator=self.integrator,
      policy=self.policy
    )
    return StructureProgram(_FreeRigidBodyProgram(system, self))


class _FreeRigidBodyProgram:
  """Internal wrapper providing uniform StructureProgram interface."""

  def __init__(self, system: _RigidBodySystem, parent: FreeRigidBodySystem):
    self.system = system
    self.parent = parent
    self.integrator_name = system.integrator_name

  def step(self, state: tuple[Tensor, Tensor], dt: float) -> tuple[Tensor, Tensor]:
    L, q = state
    L_new, q_new = self.system.step(L, q, dt)
    return L_new, q_new

  def evolve(self, state: tuple[Tensor, Tensor], dt: float, steps: int,
             record_every: int = 1, **kwargs) -> tuple[tuple[Tensor, Tensor], list]:
    L, q = state
    L, q, history = self.system.evolve(L, q, dt, steps, record_every=record_every, **kwargs)
    return (L, q), history

  def energy(self, L: Tensor) -> float:
    return self.parent.energy(L)

  def casimir(self, L: Tensor) -> float:
    return self.parent.casimir(L)


__all__ = ["FreeRigidBodyStructure", "FreeRigidBodySystem"]
