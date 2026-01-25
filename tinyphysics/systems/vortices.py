from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.physics import point_vortex_hamiltonian
from tinyphysics.core.compiler import compile_structure
from tinyphysics.core.structure import StructureKind


@dataclass
class PointVortexStructure:
  gamma: Tensor
  kind: StructureKind = StructureKind.LIE_POISSON

  def bracket(self, state: Tensor, grad: Tensor) -> Tensor:
    """
    Kirchhoff's bracket for point vortices.

    State z is flat: [x0, y0, x1, y1, ...] shape (2*n,)
    Grad dH/dz is also flat: [dH/dx0, dH/dy0, dH/dx1, dH/dy1, ...] shape (2*n,)

    The equations of motion are:
        Γ_i dx_i/dt = +∂H/∂y_i
        Γ_i dy_i/dt = -∂H/∂x_i

    So: dz/dt = J(z) * grad(H) where J swaps and scales by 1/Γ
    """
    n = int(self.gamma.shape[0])
    # Reshape grad to (n, 2) for convenience
    grad_2d = grad.reshape(n, 2)
    gx = grad_2d[:, 0]  # dH/dx_i
    gy = grad_2d[:, 1]  # dH/dy_i
    invg = 1.0 / self.gamma
    # Kirchhoff equations: dx/dt = (1/Γ) * dH/dy, dy/dt = -(1/Γ) * dH/dx
    dx = gy * invg
    dy = -gx * invg
    # Stack and flatten back to match state shape
    return Tensor.stack([dx, dy], dim=-1).flatten()

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state: Tensor) -> Callable | None:
    return None


@dataclass
class PointVortexSystem:
  gamma: Tensor
  integrator: str = "midpoint"
  policy: object | None = None

  def compile(self, state: Tensor):
    H = lambda z: point_vortex_hamiltonian(z, self.gamma)
    structure = PointVortexStructure(self.gamma)
    return compile_structure(state=state, H=H, structure=structure, integrator=self.integrator, policy=self.policy)


__all__ = ["PointVortexStructure", "PointVortexSystem"]
