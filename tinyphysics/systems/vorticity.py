"""Vorticity system adapter for 2D ideal fluid simulation.

This module provides a high-level interface for creating and running
2D Euler vorticity simulations using the structure-preserving compiler.
"""
import math
import numpy as np
from typing import Callable

from tinygrad.tensor import Tensor

from tinyphysics.structures.vorticity import VorticityStructure


def create_vorticity_system(
    N: int,
    L: float = 2*math.pi,
    dealias: float = 2.0/3.0,
    dtype=np.float32
) -> VorticityStructure:
  """Create a 2D Euler vorticity solver.

  Args:
    N: Grid size (NxN)
    L: Domain size (default 2π for periodic domain)
    dealias: De-aliasing parameter (default 2/3 rule)
    dtype: Data type (default float32)

  Returns:
    VorticityStructure that can be used for simulation

  Example:
    >>> solver = create_vorticity_system(N=64)
    >>> w_final, history = solver.evolve(w_init, dt=0.02, steps=1000)
  """
  return VorticityStructure(N=N, L=L, dealias=dealias, dtype=dtype)


def kelvin_helmholtz_ic(N: int, L: float = 2*math.pi, delta: float = 0.5, pert_amp: float = 0.1) -> np.ndarray:
  """Create Kelvin-Helmholtz instability initial condition.

  Two counter-rotating vortex strips with sinusoidal perturbation.

  Args:
    N: Grid size
    L: Domain size
    delta: Width of vortex strips
    pert_amp: Perturbation amplitude

  Returns:
    Initial vorticity field (NxN numpy array)
  """
  x = np.linspace(0, L, N, endpoint=False)
  y = np.linspace(0, L, N, endpoint=False)
  X, Y = np.meshgrid(x, y, indexing='ij')

  omega = np.zeros((N, N), dtype=np.float32)
  pert = pert_amp * np.sin(X)

  # Strip 1 at y = L/4
  y1 = L / 4
  omega += (1/delta) * (1.0 / np.cosh((Y - y1)/delta)**2) * (1 + pert)

  # Strip 2 at y = 3L/4 (reverse sign)
  y2 = 3 * L / 4
  omega -= (1/delta) * (1.0 / np.cosh((Y - y2)/delta)**2) * (1 + pert)

  return omega


def taylor_green_ic(N: int, L: float = 2*math.pi) -> np.ndarray:
  """Create Taylor-Green vortex initial condition.

  A classic test case with known analytical solution.

  Args:
    N: Grid size
    L: Domain size

  Returns:
    Initial vorticity field (NxN numpy array)
  """
  x = np.linspace(0, L, N, endpoint=False)
  y = np.linspace(0, L, N, endpoint=False)
  X, Y = np.meshgrid(x, y, indexing='ij')

  # ω = 2 cos(x) cos(y)
  omega = 2.0 * np.cos(X) * np.cos(Y)
  return omega.astype(np.float32)


def compute_enstrophy(w: np.ndarray | Tensor, L: float, N: int) -> float:
  """Compute enstrophy Z = ½∫ω² dx.

  Enstrophy is conserved by 2D Euler equations (in the inviscid limit).
  """
  if isinstance(w, Tensor):
    w = w.numpy()
  dx = L / N
  return 0.5 * float((w ** 2).sum()) * dx * dx


def compute_energy(w: np.ndarray | Tensor, L: float, N: int, structure: VorticityStructure | None = None) -> float:
  """Compute kinetic energy E = ½∫ω·ψ dx.

  Energy is conserved by 2D Euler equations.
  """
  from tinyphysics.operators.poisson import poisson_solve_fft2

  if isinstance(w, Tensor):
    w_np = w.numpy()
  else:
    w_np = w

  w_t = Tensor(w_np)
  psi = poisson_solve_fft2(w_t, L=L).numpy()
  dx = L / N
  return 0.5 * float((w_np * psi).sum()) * dx * dx


def operator_trace(structure: VorticityStructure) -> tuple[str, ...]:
  trace: list[str] = []
  structure.operator_trace(trace)
  return tuple(trace)


__all__ = [
  "create_vorticity_system",
  "kelvin_helmholtz_ic",
  "taylor_green_ic",
  "compute_enstrophy",
  "compute_energy",
  "operator_trace",
]
