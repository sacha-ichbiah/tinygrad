"""
Heavy Top System - Blueprint-compliant wrapper for SO(3) × S² dynamics.

The heavy top lives on a Product Manifold: SO(3)* × S²
- L: Angular momentum in body frame (on so(3)*)
- γ: Gravity direction in body frame (on S²)

The Lie-Poisson bracket couples both components:
  {Lᵢ, Lⱼ} = εᵢⱼₖ Lₖ   (angular momentum algebra)
  {γᵢ, Lⱼ} = εᵢⱼₖ γₖ   (γ transforms as a vector under rotations)
  {γᵢ, γⱼ} = 0          (γ components commute)

Hamiltonian:
  H(L, γ) = ½ L · (I⁻¹ L) + mgl γ₃

Conservation laws:
- Energy H is conserved
- Casimir C1 = |γ|² = 1 (γ lives on sphere)
- Casimir C2 = L · γ (projection of L onto gravity direction)
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad import dtypes
from tinygrad.tensor import Tensor
from tinygrad.physics import (
  HeavyTopHamiltonian as _HeavyTopHamiltonian,
  HeavyTopIntegrator as _HeavyTopIntegrator,
  ProductManifold,
)
from tinyphysics.core.structure import StructureKind, StructureProgram


def _cross(a: Tensor, b: Tensor) -> Tensor:
  """Cross product for 3-vectors."""
  ax, ay, az = a[..., 0], a[..., 1], a[..., 2]
  bx, by, bz = b[..., 0], b[..., 1], b[..., 2]
  return Tensor.stack([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx], dim=-1)


@dataclass
class HeavyTopStructure:
  """
  Lie-Poisson structure for the heavy top on SO(3)* × S².

  The bracket for the coupled system (L, γ):
    {Lᵢ, Lⱼ} = εᵢⱼₖ Lₖ
    {γᵢ, Lⱼ} = εᵢⱼₖ γₖ
    {γᵢ, γⱼ} = 0
  """
  kind: StructureKind = StructureKind.LIE_POISSON

  def bracket(self, state: tuple[Tensor, Tensor], grad: tuple[Tensor, Tensor]) -> tuple[Tensor, Tensor]:
    """
    Heavy top bracket for state (L, γ).

    Given gradients (∂H/∂L, ∂H/∂γ), compute:
      dL/dt = L × ∂H/∂L + γ × ∂H/∂γ
      dγ/dt = γ × ∂H/∂L
    """
    L, gamma = state
    dH_dL, dH_dgamma = grad
    # dL/dt = L × ω + γ × (mgl e₃) where ω = I⁻¹L
    # But in general bracket form: dL/dt = L × ∂H/∂L + γ × ∂H/∂γ
    dL = _cross(L, dH_dL) + _cross(gamma, dH_dgamma)
    # dγ/dt = γ × ω = γ × ∂H/∂L
    dgamma = _cross(gamma, dH_dL)
    return dL, dgamma

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state: tuple[Tensor, Tensor]) -> Callable | None:
    """Constraint: |γ|² = 1 (γ lives on unit sphere)."""
    return None


@dataclass
class HeavyTopSystem:
  """
  Blueprint-compliant wrapper for the heavy top simulation.

  The heavy top is a symmetric or asymmetric top under gravity,
  exhibiting rich dynamics including precession and nutation.

  Example:
      system = HeavyTopSystem(
          I1=1.0, I2=1.0, I3=0.5,
          mgl=1.0,
          dt=0.001
      )
      prog = system.compile()

      # Initialize state from Euler angles
      L = Tensor([0.1, 0.0, 5.0])  # Spinning around z-axis
      gamma = ProductManifold.from_euler_angles(L, theta=pi/6, phi=0).gamma
      (L, gamma), history = prog.evolve((L, gamma), steps=10000)
  """
  I1: float
  I2: float
  I3: float
  mgl: float
  dt: float
  dtype: object = dtypes.float64
  policy: object | None = None

  def hamiltonian(self, L: Tensor, gamma: Tensor) -> Tensor:
    """
    Heavy top Hamiltonian: H = ½ L · (I⁻¹ L) + mgl γ₃

    Kinetic energy from rotation + potential energy from gravity.
    """
    I_inv = Tensor([1.0/self.I1, 1.0/self.I2, 1.0/self.I3], dtype=self.dtype)
    omega = I_inv * L
    T = 0.5 * (L * omega).sum()
    V = self.mgl * gamma[..., 2]
    return T + V

  def compile(self) -> StructureProgram:
    """Compile the heavy top system into a StructureProgram."""
    hamiltonian = _HeavyTopHamiltonian(
      I1=self.I1, I2=self.I2, I3=self.I3,
      mgl=self.mgl,
      dtype=self.dtype
    )
    integrator = _HeavyTopIntegrator(
      hamiltonian=hamiltonian,
      dt=self.dt,
      policy=self.policy
    )
    return StructureProgram(_HeavyTopProgram(integrator, hamiltonian, self))


class _HeavyTopProgram:
  """Internal wrapper providing uniform StructureProgram interface."""

  def __init__(self, integrator: _HeavyTopIntegrator, hamiltonian: _HeavyTopHamiltonian, parent: HeavyTopSystem):
    self.integrator = integrator
    self.H = hamiltonian
    self.parent = parent

  def step(self, state: tuple[Tensor, Tensor], dt: float) -> tuple[Tensor, Tensor]:
    L, gamma = state
    pm = ProductManifold(L, gamma)
    pm_new = self.integrator.step_symmetric(pm)
    return pm_new.L, pm_new.gamma

  def evolve(self, state: tuple[Tensor, Tensor], dt: float, steps: int,
             record_every: int = 1, method: str = "splitting", **kwargs) -> tuple[tuple[Tensor, Tensor], list]:
    L, gamma = state
    L, gamma, history = self.integrator.evolve(L, gamma, steps, method=method, record_every=record_every, **kwargs)
    return (L, gamma), history

  def energy(self, L: Tensor, gamma: Tensor) -> float:
    """Compute energy for given state."""
    return float(self.H(ProductManifold(L, gamma)).numpy())

  def casimirs(self, L: Tensor, gamma: Tensor) -> tuple[float, float]:
    """Compute Casimirs C1=|γ|² and C2=L·γ."""
    C1 = float((gamma * gamma).sum().numpy())
    C2 = float((L * gamma).sum().numpy())
    return C1, C2


__all__ = ["HeavyTopStructure", "HeavyTopSystem", "ProductManifold"]
