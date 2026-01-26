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
from tinyphysics.operators.poisson import _complex_mul_real
from tinyphysics.operators.spatial import grad2_op, poisson_solve2_op


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

    # Precompute combined operators for velocity computation
    # u = ∂ψ/∂y where ψ = invK2 * w, so u_hat = i*KY*invK2*w_hat
    # v = -∂ψ/∂x, so v_hat = -i*KX*invK2*w_hat
    self.invK2_KY = Tensor((invK2 * KY).astype(self.dtype))  # For u velocity
    self.invK2_KX = Tensor((invK2 * KX).astype(self.dtype))  # For v velocity

    # Pre-stack wavenumber arrays for vectorized computation
    # Stack order: [invK2_KY, invK2_KX, KX, KY] for [u, v, dwdx, dwdy]
    self._k_stack_real = Tensor(np.stack([
      -invK2 * KY,  # u_r coefficient (multiply by -w_i)
      invK2 * KX,   # v_r coefficient (multiply by w_i)
      -KX,          # dwdx_r coefficient (multiply by w_i, then negate)
      -KY,          # dwdy_r coefficient (multiply by w_i, then negate)
    ], axis=0).astype(self.dtype))

    self._k_stack_imag = Tensor(np.stack([
      invK2 * KY,   # u_i coefficient (multiply by w_r)
      -invK2 * KX,  # v_i coefficient (multiply by -w_r)
      KX,           # dwdx_i coefficient (multiply by w_r)
      KY,           # dwdy_i coefficient (multiply by w_r)
    ], axis=0).astype(self.dtype))

    cutoff = self.dealias * np.max(np.abs(kx))
    self.mask = Tensor(((np.abs(KX) < cutoff) & (np.abs(KY) < cutoff)).astype(self.dtype))

  def bracket(self, w: Tensor, grad_H: Tensor) -> Tensor:
    """Lie-Poisson bracket: returns -u·∇ω (advection).

    Note: For 2D Euler, the bracket is independent of grad_H in the usual sense.
    The evolution is purely determined by the vorticity through velocity.
    """
    return self._rhs(w)

  def _rhs(self, w: Tensor) -> Tensor:
    """Compute right-hand side: -u·∇ω with de-aliasing.

    Optimized version with vectorized spectral operations using pre-stacked wavenumbers.
    """
    w_hat = rfft2d(w)
    w_r, w_i = w_hat[..., 0], w_hat[..., 1]

    # Vectorized spectral computation using pre-stacked wavenumber arrays
    # real_parts[i] = _k_stack_real[i] * w_i for u, v, dwdx, dwdy
    # imag_parts[i] = _k_stack_imag[i] * w_r for u, v, dwdx, dwdy
    real_parts = (self._k_stack_real * w_i).unsqueeze(-1)  # [4, N, N//2+1, 1]
    imag_parts = (self._k_stack_imag * w_r).unsqueeze(-1)  # [4, N, N//2+1, 1]
    batch = Tensor.cat(real_parts, imag_parts, dim=-1)  # [4, N, N//2+1, 2]

    real_batch = irfft2d(batch, n=(self.N, self.N))
    u, v, dwdx, dwdy = real_batch[0], real_batch[1], real_batch[2], real_batch[3]

    # Advection term in physical space
    adv = u * dwdx + v * dwdy

    # De-alias in spectral space
    adv_hat = rfft2d(adv)
    adv_hat_masked = adv_hat * self.mask.unsqueeze(-1)
    return -irfft2d(adv_hat_masked, n=(self.N, self.N))

  def _rhs_operator(self, w: Tensor, trace: list[str] | None = None) -> Tensor:
    if trace is not None:
      trace.extend(["poisson_solve2", "grad2", "grad2"])
    psi = poisson_solve2_op(L=self.L)(w)
    dpsidx, dpsidy = grad2_op(L=self.L)(psi)
    dwdx, dwdy = grad2_op(L=self.L)(w)
    u = dpsidy
    v = -dpsidx
    adv = u * dwdx + v * dwdy
    return -adv

  def operator_trace(self, out: list[str]):
    out.extend(["poisson_solve2", "grad2", "grad2"])

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

  def step(self, w: Tensor, dt: float, method: str = "midpoint", iters: int = 2, use_operator: bool = False) -> Tensor:
    """Single timestep using implicit midpoint or explicit Euler."""
    rhs = self._rhs_operator if use_operator else self._rhs
    if method == "euler":
      return (w + dt * rhs(w)).realize()

    if method != "midpoint":
      raise ValueError(f"Unknown method: {method}")

    # Implicit midpoint with fixed-point iteration
    w_next = w + dt * rhs(w)
    for _ in range(max(1, iters)):
      w_mid = 0.5 * (w + w_next)
      w_next = w + dt * rhs(w_mid)
    return w_next.realize()

  def evolve(self, w0: Tensor | np.ndarray, dt: float, steps: int,
             record_every: int = 1, method: str = "midpoint", iters: int = 2,
             unroll: int | None = None, diagnostics: bool = False, use_operator: bool = False) -> tuple[Tensor, list]:
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
      step_fn = self._get_unrolled_step(dt, unroll, method, iters, use_operator)

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
        w = self.step(w, dt, method, iters, use_operator=use_operator)
        if (j + 1) % record_every == 0:
          if diagnostics:
            e, z = self.diagnostics(w)
            history.append((w.numpy().copy(), float(e.numpy()), float(z.numpy())))
          else:
            history.append(w.numpy().copy())
    else:
      # Non-unrolled path
      for i in range(steps):
        w = self.step(w, dt, method, iters, use_operator=use_operator)
        if (i + 1) % record_every == 0:
          if diagnostics:
            e, z = self.diagnostics(w)
            history.append((w.numpy().copy(), float(e.numpy()), float(z.numpy())))
          else:
            history.append(w.numpy().copy())

    return w, history

  def _get_unrolled_step(self, dt: float, unroll: int, method: str, iters: int, use_operator: bool) -> Callable:
    """Get or create JIT-compiled unrolled step function."""
    key = (dt, unroll, method, iters, use_operator)
    if key in self._unroll_cache:
      return self._unroll_cache[key]

    def unrolled_fn(w: Tensor) -> Tensor:
      for _ in range(unroll):
        w = self.step(w, dt, method, iters, use_operator=use_operator)
      return w

    jit_fn = TinyJit(unrolled_fn)
    self._unroll_cache[key] = jit_fn
    return jit_fn

  def compile_unrolled_step(self, dt: float, unroll: int, method: str = "midpoint", iters: int = 3,
                            use_operator: bool = False) -> Callable:
    """Public interface for getting unrolled step function."""
    return self._get_unrolled_step(dt, unroll, method, iters, use_operator)

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    """Operator splitting not implemented for vorticity dynamics."""
    return None

  def constraints(self, state: Tensor) -> Callable | None:
    """No constraints for 2D Euler."""
    return None


__all__ = ["VorticityStructure"]
