"""
Controlled Rigid Body System for Satellite Attitude Control.

This extends the blueprint's structure-preserving approach to controlled systems.
The free dynamics follow SO(3) Lie-Poisson structure, with control applied via
symplectic splitting (kick-drift-kick).

While controlled systems don't conserve energy, they can still preserve the
geometric structure of the underlying phase space.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.physics import SatelliteControlIntegrator, ControlInput
from tinyphysics.core.structure import StructureKind, StructureProgram


@dataclass
class ControlledSO3Structure:
  """
  Structure for controlled rigid body dynamics on SO(3).

  The free dynamics follow Euler's equations (Lie-Poisson on so(3)*):
      dL/dt = L × ω  where ω = I^{-1} L

  Control is applied via symplectic splitting:
      dL/dt = L × ω + τ_control

  Quaternion kinematics track attitude:
      dq/dt = 0.5 * q ⊗ ω
  """
  I_inv: Tensor
  control: ControlInput
  dt: float
  kind: StructureKind = StructureKind.LIE_POISSON

  def bracket(self, state: tuple[Tensor, Tensor], grad: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """
    The bracket for controlled SO(3) systems.

    For the free part, this is the standard SO(3) bracket: L × grad(H).
    Control is handled separately in the split integrator.
    """
    L, quat = state
    dL, dq = grad
    # SO(3) bracket for angular momentum
    L_flow = L.cross(dL)
    # Quaternion flow is handled by the integrator
    return L_flow, dq

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    """Control is applied via operator splitting in the integrator."""
    return None

  def constraints(self, state: tuple[Tensor, Tensor]) -> Callable | None:
    """Quaternion normalization constraint."""
    return None


@dataclass
class ControlledRigidBodySystem:
  """
  Blueprint-compliant wrapper for satellite attitude control.

  Uses the structure-preserving split integrator:
  1. Half control kick
  2. Free Lie-Poisson evolution
  3. Quaternion integration
  4. Half control kick

  Example:
      control = ControlInput(lambda q_err, omega: -Kp * q_err[..., 1:] - Kd * omega)
      system = ControlledRigidBodySystem(
          I_inv=I_inv,
          control=control,
          dt=0.01
      )
      prog = system.compile()
      (L, quat), history = prog.evolve((L, quat), steps=1000)
  """
  I_inv: Tensor
  control: ControlInput
  dt: float
  policy: object | None = None

  def compile(self) -> StructureProgram:
    """Compile the controlled rigid body system into a StructureProgram."""
    integrator = SatelliteControlIntegrator(
      I_inv=self.I_inv,
      control=self.control,
      dt=self.dt,
      policy=self.policy
    )
    return StructureProgram(_ControlledRigidBodyProgram(integrator, self.dt))


class _ControlledRigidBodyProgram:
  """Internal wrapper to provide uniform interface."""
  def __init__(self, integrator: SatelliteControlIntegrator, dt: float):
    self.integrator = integrator
    self.dt = dt

  def step(self, state: tuple[Tensor, Tensor], dt: float) -> tuple[Tensor, Tensor]:
    L, quat = state
    L_new, quat_new = self.integrator.step(L, quat)
    return L_new, quat_new

  def evolve(self, state: tuple[Tensor, Tensor], dt: float, steps: int,
             record_every: int = 1, **kwargs) -> tuple[tuple[Tensor, Tensor], list]:
    L, quat = state
    L, quat, history = self.integrator.evolve(L, quat, steps, record_every=record_every, **kwargs)
    return (L, quat), history


__all__ = ["ControlledSO3Structure", "ControlledRigidBodySystem"]
