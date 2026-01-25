"""2D Euler vorticity as a Lie-Poisson structure.

The 2D incompressible Euler equations in vorticity form:
  dω/dt = -u·∇ω

can be written as a Lie-Poisson system with bracket:
  {F, G} = ∫ ω [δF/δω, δG/δω] dx

where [f,g] = ∂f/∂x ∂g/∂y - ∂f/∂y ∂g/∂x is the Jacobian.

For Hamiltonian H = ½∫ω·ψ dx (kinetic energy), where ψ is the streamfunction,
the evolution dω/dt = {ω, H} gives the advection equation.
"""
import math
import numpy as np
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.fft import rfft2d, irfft2d
from tinygrad.engine.jit import TinyJit

from tinyphysics.core.structure import Structure, StructureKind
from tinyphysics.operators.poisson import _complex_mul_real, _complex_mul_i


class VorticityStructure(Structure):
  """2D Euler vorticity as Lie-Poisson structure with spectral methods."""
  kind = StructureKind.LIE_POISSON

  def __init__(self, N: int, L: float = 2*math.pi, dealias: float = 2.0/3.0, dtype=np.float32):
    self.N = N
    self.L = L
    self.dealias = dealias
    self.dtype = dtype
    self._setup_spectral()
    self._step_cache: dict[tuple, Callable] = {}
    self._unroll_cache: dict[tuple, Callable] = {}

  def _setup_spectral(self):
    """Precompute wavenumbers and de-aliasing mask."""
    N, L = self.N, self.L
    kx = np.fft.fftfreq(N, d=L/N) * 2 * math.pi
    ky = np.fft.rfftfreq(N, d=L/N) * 2 * math.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    K2 = KX**2 + KY**2
    K2[0, 0] = 1.0
    invK2 = 1.0 / K2
    invK2[0, 0] = 0.0

    self.KX = Tensor(KX.astype(self.dtype))
    self.KY = Tensor(KY.astype(self.dtype))
    self.invK2 = Tensor(invK2.astype(self.dtype))

    cutoff = self.dealias * np.max(np.abs(kx))
    self.mask = Tensor(((np.abs(KX) < cutoff) & (np.abs(KY) < cutoff)).astype(self.dtype))

  def bracket(self, w: Tensor, grad_H: Tensor) -> Tensor:
    """Lie-Poisson bracket: returns -u·∇ω (advection).

    Note: For 2D Euler, the bracket is independent of grad_H in the usual sense.
    The evolution is purely determined by the vorticity through velocity.
    """
    return self._rhs(w)

  def _rhs(self, w: Tensor) -> Tensor:
    """Compute right-hand side: -u·∇ω with de-aliasing."""
    w_hat = rfft2d(w)
    psi_hat = _complex_mul_real(w_hat, self.invK2)

    # Compute velocity and vorticity gradients in spectral space
    u_hat = _complex_mul_i(psi_hat, self.KY, sign=1.0)    # u = ∂ψ/∂y
    v_hat = _complex_mul_i(psi_hat, self.KX, sign=-1.0)   # v = -∂ψ/∂x
    dwdx_hat = _complex_mul_i(w_hat, self.KX, sign=1.0)
    dwdy_hat = _complex_mul_i(w_hat, self.KY, sign=1.0)

    # Batch iFFT for efficiency
    batch = Tensor.stack([u_hat, v_hat, dwdx_hat, dwdy_hat], dim=0)
    real_batch = irfft2d(batch, n=(self.N, self.N))
    u, v, dwdx, dwdy = real_batch[0], real_batch[1], real_batch[2], real_batch[3]

    # Advection term in physical space
    adv = u * dwdx + v * dwdy

    # De-alias in spectral space
    adv_hat = _complex_mul_real(rfft2d(adv), self.mask)
    return -irfft2d(adv_hat, n=(self.N, self.N))

  def diagnostics(self, w: Tensor) -> tuple[Tensor, Tensor]:
    """Return (energy, enstrophy) for vorticity field."""
    w_hat = rfft2d(w)
    psi_hat = _complex_mul_real(w_hat, self.invK2)
    psi = irfft2d(psi_hat, n=(self.N, self.N))
    dx = self.L / self.N
    area = dx * dx
    energy = 0.5 * (w * psi).sum() * area
    enstrophy = 0.5 * (w * w).sum() * area
    return energy, enstrophy

  def step(self, w: Tensor, dt: float, method: str = "midpoint", iters: int = 2) -> Tensor:
    """Single timestep using implicit midpoint or explicit Euler."""
    if method == "euler":
      return (w + dt * self._rhs(w)).realize()

    if method != "midpoint":
      raise ValueError(f"Unknown method: {method}")

    # Implicit midpoint with fixed-point iteration
    w_next = w + dt * self._rhs(w)
    for _ in range(max(1, iters)):
      w_mid = 0.5 * (w + w_next)
      w_next = w + dt * self._rhs(w_mid)
    return w_next.realize()

  def evolve(self, w0: Tensor | np.ndarray, dt: float, steps: int,
             record_every: int = 1, method: str = "midpoint", iters: int = 2,
             unroll: int | None = None, diagnostics: bool = False) -> tuple[Tensor, list]:
    """Evolve vorticity field for multiple steps.

    Args:
      w0: Initial vorticity (Tensor or ndarray)
      dt: Timestep
      steps: Number of steps
      record_every: Record history every N steps
      method: Integration method ("midpoint" or "euler")
      iters: Iterations for implicit midpoint
      unroll: Number of steps to unroll in JIT (auto-selected if None)

    Returns:
      (final_state, history) where history is list of numpy arrays
    """
    if isinstance(w0, np.ndarray):
      w = Tensor(w0.astype(self.dtype))
    else:
      w = w0

    # Auto-select unroll factor that divides record_every evenly
    if unroll is None and steps >= 50:
      # Find largest unroll <= 8 that divides record_every
      for u in [8, 5, 4, 2, 1]:
        if record_every % u == 0:
          unroll = u
          break
      else:
        unroll = 1

    if diagnostics:
      e0, z0 = self.diagnostics(w)
      history = [(w.numpy().copy(), float(e0.numpy()), float(z0.numpy()))]
    else:
      history = [w.numpy().copy()]

    if unroll and unroll > 1 and steps >= unroll:
      # Use JIT-compiled unrolled step
      step_fn = self._get_unrolled_step(dt, unroll, method, iters)

      i = 0
      while i + unroll <= steps:
        w = step_fn(w)
        i += unroll
        if i % record_every == 0:
          if diagnostics:
            e, z = self.diagnostics(w)
            history.append((w.numpy().copy(), float(e.numpy()), float(z.numpy())))
          else:
            history.append(w.numpy().copy())

      # Handle remaining steps
      for j in range(i, steps):
        w = self.step(w, dt, method, iters)
        if (j + 1) % record_every == 0:
          if diagnostics:
            e, z = self.diagnostics(w)
            history.append((w.numpy().copy(), float(e.numpy()), float(z.numpy())))
          else:
            history.append(w.numpy().copy())
    else:
      # Non-unrolled path
      for i in range(steps):
        w = self.step(w, dt, method, iters)
        if (i + 1) % record_every == 0:
          if diagnostics:
            e, z = self.diagnostics(w)
            history.append((w.numpy().copy(), float(e.numpy()), float(z.numpy())))
          else:
            history.append(w.numpy().copy())

    return w, history

  def _get_unrolled_step(self, dt: float, unroll: int, method: str, iters: int) -> Callable:
    """Get or create JIT-compiled unrolled step function."""
    key = (dt, unroll, method, iters)
    if key in self._unroll_cache:
      return self._unroll_cache[key]

    def unrolled_fn(w: Tensor) -> Tensor:
      for _ in range(unroll):
        w = self.step(w, dt, method, iters)
      return w

    jit_fn = TinyJit(unrolled_fn)
    self._unroll_cache[key] = jit_fn
    return jit_fn

  def compile_unrolled_step(self, dt: float, unroll: int, method: str = "midpoint", iters: int = 3) -> Callable:
    """Public interface for getting unrolled step function."""
    return self._get_unrolled_step(dt, unroll, method, iters)

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    """Operator splitting not implemented for vorticity dynamics."""
    return None

  def constraints(self, state: Tensor) -> Callable | None:
    """No constraints for 2D Euler."""
    return None


__all__ = ["VorticityStructure"]
