import math
import numpy as np

from tinygrad.helpers import getenv
from tinygrad.tensor import Tensor
from tinygrad.engine.jit import TinyJit, JitError
from tinygrad.engine.realize import capturing
from tinygrad.fft import fft1d, ifft1d, fft2d, ifft2d, fft3d, ifft3d, fft_plan
from tinyphysics.core.structure import StructureKind


def _as_complex(x: Tensor) -> Tensor:
  if x.ndim >= 1 and x.shape[-1] == 2:
    return x
  return Tensor.stack([x, x.zeros_like()], dim=-1)


def _complex_mul(a: Tensor, b: Tensor) -> Tensor:
  ar, ai = a[..., 0], a[..., 1]
  br, bi = b[..., 0], b[..., 1]
  real = ar * br - ai * bi
  imag = ar * bi + ai * br
  return Tensor.stack([real, imag], dim=-1)


def _complex_abs2(a: Tensor) -> Tensor:
  return a[..., 0] * a[..., 0] + a[..., 1] * a[..., 1]


def _exp_i(phase: Tensor) -> Tensor:
  return Tensor.stack([phase.cos(), phase.sin()], dim=-1)


def _numpy_fft_nd(psi: Tensor, inverse: bool) -> Tensor:
  import numpy as np
  data = psi.numpy()
  c = data[..., 0] + 1j * data[..., 1]
  out = np.fft.ifftn(c) if inverse else np.fft.fftn(c)
  return Tensor.stack([Tensor(out.real, device=psi.device), Tensor(out.imag, device=psi.device)], dim=-1)


def fftfreq(n: int, d: float, device: str, dtype) -> Tensor:
  idx = Tensor.arange(n, device=device, dtype=dtype, requires_grad=False)
  half = n // 2
  neg = idx - n
  k = (idx < half).where(idx, neg)
  return k / (n * d)


def gaussian_wavepacket(x: Tensor, x0: float, k0: float, sigma: float) -> Tensor:
  """Create a normalized Gaussian wavepacket.

  The wavepacket has the form:
    psi(x) = (2*pi*sigma^2)^(-1/4) * exp(-(x-x0)^2/(4*sigma^2)) * exp(i*k0*x)

  This gives |psi|^2 ~ exp(-(x-x0)^2/(2*sigma^2)) with standard deviation sigma.

  Args:
    x: 1D grid tensor
    x0: Center position
    k0: Initial momentum (wavenumber)
    sigma: Width parameter (standard deviation of |psi|^2)

  Returns:
    Complex wavepacket as Tensor with shape (..., 2) for [real, imag]
  """
  dx = x - x0
  # Use exp(-x^2/(4*sigma^2)) so that |psi|^2 ~ exp(-x^2/(2*sigma^2)) has std=sigma
  env = (-0.25 * (dx / sigma) * (dx / sigma)).exp()
  phase = x * k0
  psi = _complex_mul(_as_complex(env), _exp_i(phase))
  return normalize(psi, float(x[1].item() - x[0].item()) if x.shape[0] > 1 else 1.0)


def normalize(psi: Tensor, dx: float = 1.0, eps: float = 1e-12) -> Tensor:
  norm = (_complex_abs2(psi).sum() * dx).sqrt()
  norm = norm + eps
  return Tensor.stack([psi[..., 0] / norm, psi[..., 1] / norm], dim=-1)


class QuantumSplitOperator1D:
  def __init__(self, x: Tensor, dt: float, mass: float = 1.0, hbar: float = 1.0, V=None, g: float = 0.0):
    if x.ndim != 1:
      raise ValueError("x must be a 1D grid")
    if mass == 0:
      raise ValueError("mass must be non-zero")
    self.x = x
    self.dt = dt
    self.mass = mass
    self.hbar = hbar
    self.g = g
    self.V = V
    self.dx = float(x[1].item() - x[0].item()) if x.shape[0] > 1 else 1.0
    k = fftfreq(x.shape[0], self.dx, x.device, x.dtype) * (2.0 * math.pi)
    self.k2 = k * k
    self._phase_k = _exp_i((-0.5 * self.dt * self.hbar / self.mass) * self.k2).realize()
    self._decay_k = ((-0.5 * self.dt * self.hbar / self.mass) * self.k2).exp().realize()
    self._phase_v: Tensor | None = None
    self._decay_v: Tensor | None = None
    self._use_fft_plan = True
    self._fft_plan = None
    self._ifft_plan = None
    try:
      dummy = Tensor.zeros(x.shape[0], 2, device=x.device, dtype=x.dtype)
      self._fft_plan = fft_plan(dummy, inverse=False, kind="1d")
      self._ifft_plan = fft_plan(dummy, inverse=True, kind="1d")
    except Exception:
      pass
    self._unroll_cache: dict[int, TinyJit] = {}
    if self.g == 0.0:
      if self.V is None:
        Vx = self.x.zeros_like()
      elif callable(self.V):
        Vx = self.V(self.x)
      else:
        Vx = self.V
      self._phase_v = _exp_i((-0.5 * self.dt / self.hbar) * Vx).realize()
      self._decay_v = ((-0.5 * self.dt / self.hbar) * Vx).exp().realize()

  def _potential(self, psi: Tensor) -> Tensor:
    if self.V is None:
      Vx = self.x.zeros_like()
    elif callable(self.V):
      Vx = self.V(self.x)
    else:
      Vx = self.V
    if self.g != 0.0:
      Vx = Vx + self.g * _complex_abs2(psi)
    return Vx

  def _step_raw(self, psi: Tensor) -> Tensor:
    psi = _as_complex(psi)
    if self._phase_v is None:
      Vx = self._potential(psi)
      phase_v = _exp_i((-0.5 * self.dt / self.hbar) * Vx)
    else:
      phase_v = self._phase_v
    psi = _complex_mul(psi, phase_v)

    # Use numpy FFT for large arrays (tinygrad FFT has roundtrip issues for N >= 128)
    N = psi.shape[0]
    if N >= 128:
      psi_np = psi.numpy()
      psi_complex = psi_np[..., 0] + 1j * psi_np[..., 1]
      k = self.k2.numpy()
      phase_k = self._phase_k.numpy()
      phase_k_complex = phase_k[..., 0] + 1j * phase_k[..., 1]
      psi_k = np.fft.fft(psi_complex)
      psi_k = psi_k * phase_k_complex
      psi_new = np.fft.ifft(psi_k)
      psi = Tensor.stack([Tensor(psi_new.real.astype(np.float32), device=psi.device),
                          Tensor(psi_new.imag.astype(np.float32), device=psi.device)], dim=-1)
    else:
      if self._use_fft_plan and self._fft_plan is not None and not capturing:
        psi_k = self._fft_plan(psi)
      else:
        psi_k = fft1d(psi)
      psi_k = _complex_mul(psi_k, self._phase_k)
      if self._use_fft_plan and self._ifft_plan is not None and not capturing:
        psi = self._ifft_plan(psi_k)
      else:
        psi = ifft1d(psi_k)

    psi = _complex_mul(psi, phase_v)
    return psi

  def renormalize(self, psi: Tensor) -> Tensor:
    return normalize(psi, self.dx)

  def step(self, psi: Tensor, renormalize: bool = True) -> Tensor:
    psi = self._step_raw(psi)
    return self.renormalize(psi) if renormalize else psi

  def step_imaginary(self, psi: Tensor) -> Tensor:
    psi = _as_complex(psi)
    if self._decay_v is None:
      Vx = self._potential(psi)
      decay = ((-0.5 * self.dt / self.hbar) * Vx).exp()
    else:
      decay = self._decay_v
    psi = Tensor.stack([psi[..., 0] * decay, psi[..., 1] * decay], dim=-1)
    if self._use_fft_plan and self._fft_plan is not None and not capturing:
      psi_k = self._fft_plan(psi)
    else:
      psi_k = fft1d(psi)
    psi_k = Tensor.stack([psi_k[..., 0] * self._decay_k, psi_k[..., 1] * self._decay_k], dim=-1)
    if self._use_fft_plan and self._ifft_plan is not None and not capturing:
      psi = self._ifft_plan(psi_k)
    else:
      psi = ifft1d(psi_k)
    psi = Tensor.stack([psi[..., 0] * decay, psi[..., 1] * decay], dim=-1)
    return self.renormalize(psi)

  def _prob_numpy(self, psi: Tensor):
    """Get |psi|^2 as numpy array for history recording."""
    prob = psi[..., 0] * psi[..., 0] + psi[..., 1] * psi[..., 1]
    return prob.numpy().copy()

  def _get_unrolled_step(self, unroll: int):
    """Get or create unrolled step function.

    Note: JIT compilation is disabled when numpy FFT is used (N >= 128)
    because numpy operations aren't JIT-compatible.
    """
    if unroll in self._unroll_cache:
      return self._unroll_cache[unroll]

    # Check if numpy FFT fallback is used (N >= 128)
    use_numpy_fft = self.x.shape[0] >= 128

    def unrolled_fn(psi: Tensor) -> Tensor:
      for _ in range(unroll):
        psi = self._step_raw(psi)
      return self.renormalize(psi).realize()

    if use_numpy_fft:
      # Don't use JIT when numpy FFT is used (breaks JIT tracing)
      self._unroll_cache[unroll] = unrolled_fn
      return unrolled_fn
    else:
      # Use JIT for pure tinygrad FFT (N < 128)
      jit_fn = TinyJit(unrolled_fn)
      self._unroll_cache[unroll] = jit_fn
      return jit_fn

  def evolve(self, psi: Tensor, steps: int, record_every: int = 1, unroll: int | None = None):
    """Evolve wavefunction with JIT-compiled unrolled steps.

    Args:
      psi: Initial wavefunction
      steps: Number of time steps
      record_every: Record history every N steps
      unroll: Steps per kernel (auto-selected if None)

    Returns:
      (final_psi, history) where history is list of |psi|^2 numpy arrays
    """
    # Auto-select unroll factor that divides record_every
    if unroll is None and steps >= 50:
      for u in [8, 5, 4, 2, 1]:
        if record_every % u == 0:
          unroll = u
          break
      else:
        unroll = 1

    history = [self._prob_numpy(psi)]  # Record initial state

    if unroll and unroll > 1 and steps >= unroll:
      # Use JIT-compiled unrolled step
      step_fn = self._get_unrolled_step(unroll)

      i = 0
      while i + unroll <= steps:
        psi = step_fn(psi)
        i += unroll
        if i % record_every == 0:
          history.append(self._prob_numpy(psi))

      # Handle remaining steps
      for j in range(i, steps):
        psi = self.step(psi)
        if (j + 1) % record_every == 0:
          history.append(self._prob_numpy(psi))
    else:
      # Non-unrolled fallback
      for i in range(steps):
        psi = self.step(psi)
        if (i + 1) % record_every == 0:
          history.append(self._prob_numpy(psi))

    return psi, history


class QuantumSplitOperatorND:
  def __init__(self, grids: tuple[Tensor, ...], dt: float, mass: float = 1.0, hbar: float = 1.0, V=None, g: float = 0.0):
    if len(grids) not in (2, 3):
      raise ValueError("grids must be a tuple of 2 or 3 1D tensors")
    if any(g.ndim != 1 for g in grids):
      raise ValueError("grids must be 1D tensors")
    if mass == 0:
      raise ValueError("mass must be non-zero")
    self.grids = grids
    self.dt = dt
    self.mass = mass
    self.hbar = hbar
    self.V = V
    self.g = g
    self.deltas = tuple(float(g[1].item() - g[0].item()) if g.shape[0] > 1 else 1.0 for g in grids)
    self.dx_prod = 1.0
    for d in self.deltas:
      self.dx_prod *= d
    self.k2 = self._build_k2()
    self._phase_k = _exp_i((-0.5 * self.dt * self.hbar / self.mass) * self.k2).realize()
    self._decay_k = ((-0.5 * self.dt * self.hbar / self.mass) * self.k2).exp().realize()
    self._phase_v: Tensor | None = None
    self._decay_v: Tensor | None = None
    self._use_fft_plan = True
    self._fft_plan = None
    self._ifft_plan = None
    try:
      shape = tuple(g.shape[0] for g in grids)
      dummy = Tensor.zeros(*shape, 2, device=grids[0].device, dtype=grids[0].dtype)
      kind = "2d" if len(grids) == 2 else "3d"
      self._fft_plan = fft_plan(dummy, inverse=False, kind=kind)
      self._ifft_plan = fft_plan(dummy, inverse=True, kind=kind)
    except Exception:
      pass
    if self.g == 0.0:
      if self.V is None:
        Vx = grids[0].zeros_like()
      elif callable(self.V):
        Vx = self.V(*self.grids)
      else:
        Vx = self.V
      self._phase_v = _exp_i((-0.5 * self.dt / self.hbar) * Vx).realize()
      self._decay_v = ((-0.5 * self.dt / self.hbar) * Vx).exp().realize()

  def _build_k2(self) -> Tensor:
    k_axes = []
    for g, d in zip(self.grids, self.deltas):
      k = fftfreq(g.shape[0], d, g.device, g.dtype) * (2.0 * math.pi)
      k_axes.append(k)
    if len(k_axes) == 2:
      kx = k_axes[0].reshape(k_axes[0].shape[0], 1)
      ky = k_axes[1].reshape(1, k_axes[1].shape[0])
      return kx * kx + ky * ky
    kx = k_axes[0].reshape(k_axes[0].shape[0], 1, 1)
    ky = k_axes[1].reshape(1, k_axes[1].shape[0], 1)
    kz = k_axes[2].reshape(1, 1, k_axes[2].shape[0])
    return kx * kx + ky * ky + kz * kz

  def _potential(self, psi: Tensor) -> Tensor:
    if self.V is None:
      Vx = psi[..., 0].zeros_like()
    elif callable(self.V):
      Vx = self.V(*self.grids)
    else:
      Vx = self.V
    if self.g != 0.0:
      Vx = Vx + self.g * _complex_abs2(psi)
    return Vx

  def _fft(self, psi: Tensor) -> Tensor:
    if psi.device == "CPU" and getenv("TINYGRAD_QUANTUM_NUMPY_FFT", 0) and len(self.grids) == 3:
      return _numpy_fft_nd(psi, inverse=False)
    if psi.device == "CPU" and getenv("TINYGRAD_QUANTUM_NUMPY_FFT", 0):
      return _numpy_fft_nd(psi, inverse=False)
    if self._use_fft_plan and self._fft_plan is not None and not capturing:
      return self._fft_plan(psi)
    if len(self.grids) == 2:
      return fft2d(psi)
    return fft3d(psi)

  def _ifft(self, psi: Tensor) -> Tensor:
    if psi.device == "CPU" and getenv("TINYGRAD_QUANTUM_NUMPY_FFT", 0) and len(self.grids) == 3:
      return _numpy_fft_nd(psi, inverse=True)
    if psi.device == "CPU" and getenv("TINYGRAD_QUANTUM_NUMPY_FFT", 0):
      return _numpy_fft_nd(psi, inverse=True)
    if self._use_fft_plan and self._ifft_plan is not None and not capturing:
      return self._ifft_plan(psi)
    if len(self.grids) == 2:
      return ifft2d(psi)
    return ifft3d(psi)

  def _step_raw(self, psi: Tensor) -> Tensor:
    psi = _as_complex(psi)
    if self._phase_v is None:
      Vx = self._potential(psi)
      phase_v = _exp_i((-0.5 * self.dt / self.hbar) * Vx)
    else:
      phase_v = self._phase_v
    psi = _complex_mul(psi, phase_v)
    psi_k = self._fft(psi)
    psi_k = _complex_mul(psi_k, self._phase_k)
    psi = self._ifft(psi_k)
    psi = _complex_mul(psi, phase_v)
    return psi

  def renormalize(self, psi: Tensor) -> Tensor:
    return normalize(psi, self.dx_prod)

  def step(self, psi: Tensor, renormalize: bool = True) -> Tensor:
    psi = self._step_raw(psi)
    return self.renormalize(psi) if renormalize else psi

  def step_imaginary(self, psi: Tensor) -> Tensor:
    psi = _as_complex(psi)
    if self._decay_v is None:
      Vx = self._potential(psi)
      decay = ((-0.5 * self.dt / self.hbar) * Vx).exp()
    else:
      decay = self._decay_v
    psi = Tensor.stack([psi[..., 0] * decay, psi[..., 1] * decay], dim=-1)
    psi_k = self._fft(psi)
    psi_k = Tensor.stack([psi_k[..., 0] * self._decay_k, psi_k[..., 1] * self._decay_k], dim=-1)
    psi = self._ifft(psi_k)
    psi = Tensor.stack([psi[..., 0] * decay, psi[..., 1] * decay], dim=-1)
    return self.renormalize(psi)


class QuantumHamiltonianCompiler:
  def __init__(self, grids: tuple[Tensor, ...], dt: float, mass: float = 1.0, hbar: float = 1.0, V=None, g: float = 0.0):
    self.grids = grids
    self.dt = dt
    self.mass = mass
    self.hbar = hbar
    self.V = V
    self.g = g
    if len(grids) == 1:
      self._solver = QuantumSplitOperator1D(grids[0], dt, mass=mass, hbar=hbar, V=V, g=g)
    else:
      self._solver = QuantumSplitOperatorND(grids, dt, mass=mass, hbar=hbar, V=V, g=g)
    self._jit_cache: dict[str, TinyJit] = {}
    self._jit_failed: set[str] = set()
    self._jit_checked: set[str] = set()
    self._step_cache: dict[tuple, TinyJit] = {}

  def step(self, psi: Tensor) -> Tensor:
    return self._solver.step(psi)

  @property
  def dx(self) -> float:
    if len(self.grids) == 1:
      return self._solver.dx
    return self._solver.dx_prod

  def norm(self, psi: Tensor) -> float:
    """Compute norm of wavefunction."""
    return float((_complex_abs2(psi).sum() * self.dx).sqrt().numpy())

  def prob_density(self, psi: Tensor) -> Tensor:
    """Compute probability density |psi|^2."""
    return _complex_abs2(psi)

  def expectation_x(self, psi: Tensor) -> float:
    """Compute <x> for 1D wavefunction."""
    if len(self.grids) != 1:
      raise ValueError("expectation_x only for 1D")
    x = self.grids[0]
    prob = _complex_abs2(psi)
    return float((x * prob).sum().numpy() * self.dx)

  def width(self, psi: Tensor) -> float:
    """Compute width sqrt(<x^2> - <x>^2) for 1D."""
    if len(self.grids) != 1:
      raise ValueError("width only for 1D")
    x = self.grids[0]
    prob = _complex_abs2(psi)
    dx = self.dx
    x_mean = (x * prob).sum() * dx
    x2_mean = (x * x * prob).sum() * dx
    var = x2_mean - x_mean * x_mean
    return float(var.sqrt().numpy())

  def kinetic_energy(self, psi: Tensor) -> float:
    """Compute kinetic energy <T> = (hbar^2/2m) * <k^2>.

    Note: Uses numpy FFT for numerical stability with large grids.
    """
    if len(self.grids) == 1:
      import numpy as np
      psi_np = psi.numpy()
      psi_complex = psi_np[..., 0] + 1j * psi_np[..., 1]
      psi_k_np = np.fft.fft(psi_complex)
      prob_k = np.abs(psi_k_np)**2
      k = np.fft.fftfreq(self.grids[0].shape[0], self.dx) * (2.0 * math.pi)
      k2 = k * k
      # <k^2> = sum(k^2 * |psi_k|^2) / sum(|psi_k|^2)
      k2_mean = (prob_k * k2).sum() / prob_k.sum()
      T = 0.5 * self.hbar * self.hbar / self.mass * k2_mean
      return float(T)
    raise NotImplementedError("kinetic_energy for ND")

  def evolve(self, psi: Tensor, dt: float, steps: int, record_every: int = 1, unroll: int = 8,
             renormalize: bool = False, **kwargs) -> tuple[Tensor, list]:
    """Evolve wavefunction and record history.

    Args:
      psi: Initial wavefunction
      dt: Time step (must match compiler's dt)
      steps: Number of steps
      record_every: Record state every N steps (0 to disable)
      unroll: Number of steps to unroll
      renormalize: Whether to renormalize after each step (default False for unitary evolution)

    Returns:
      (final_psi, history) where history is list of (psi, step_idx) tuples
    """
    history = []

    # Record initial state
    if record_every > 0:
      history.append((psi.numpy().copy(), 0))

    # Evolve step by step (simpler but correct approach)
    for i in range(steps):
      psi = self._solver._step_raw(psi)
      if renormalize:
        psi = self._solver.renormalize(psi)
      step_idx = i + 1
      if record_every > 0 and step_idx % record_every == 0:
        history.append((psi.numpy().copy(), step_idx))

    # Record final state if not already recorded
    if record_every > 0 and steps % record_every != 0:
      history.append((psi.numpy().copy(), steps))

    return psi, history

  def step_imaginary(self, psi: Tensor) -> Tensor:
    return self._solver.step_imaginary(psi)

  def compile(self, imaginary: bool = False):
    key = "imag" if imaginary else "real"
    if key in self._jit_failed:
      return self._solver.step_imaginary if imaginary else self._solver.step
    plan = self._jit_cache.get(key)
    if plan is None:
      if imaginary:
        def plan_fn(x: Tensor) -> Tensor:
          return self._solver.step_imaginary(x).realize()
      else:
        def plan_fn(x: Tensor) -> Tensor:
          return self._solver.step(x).realize()
      plan = self._jit_cache.setdefault(key, TinyJit(plan_fn))
    if not capturing:
      def run(x: Tensor) -> Tensor:
        try:
          return plan(x)
        except JitError:
          self._jit_cache.pop(key, None)
          self._jit_failed.add(key)
          return self._solver.step_imaginary(x) if imaginary else self._solver.step(x)
      return run
    return self._solver.step_imaginary if imaginary else self._solver.step

  def compile_unrolled(self, steps: int, imaginary: bool = False, renorm_every: int = 1):
    if steps < 1:
      raise ValueError("steps must be >= 1")
    if renorm_every < 1:
      raise ValueError("renorm_every must be >= 1")
    if len(self.grids) == 3 and not capturing:
      def run(x: Tensor) -> Tensor:
        out = x
        if renorm_every == 1:
          for _ in range(steps):
            out = self._solver.step_imaginary(out) if imaginary else self._solver.step(out)
          return out
        count = 0
        for _ in range(steps):
          if imaginary:
            out = self._solver.step_imaginary(out)
          else:
            out = self._solver._step_raw(out)
            count += 1
            if count == renorm_every:
              out = self._solver.renormalize(out)
              count = 0
        if not imaginary and count:
          out = self._solver.renormalize(out)
        return out
      return run
    if getattr(self._solver, "_fft_plan", None) is not None and not capturing:
      step = self.compile(imaginary=imaginary)
      def run(x: Tensor) -> Tensor:
        out = x
        if renorm_every == 1 or imaginary:
          for _ in range(steps):
            out = step(out)
          return out
        count = 0
        for _ in range(steps):
          out = self._solver._step_raw(out)
          count += 1
          if count == renorm_every:
            out = self._solver.renormalize(out)
            count = 0
        if count:
          out = self._solver.renormalize(out)
        return out
      return run
    if len(self.grids) == 1:
      return self._compile_unrolled_1d(steps, imaginary)
    if len(self.grids) == 2:
      return self._compile_unrolled_2d(steps, imaginary)


class QuantumSplitStructure:
  kind: StructureKind = StructureKind.QUANTUM

  def __init__(self, split_ops):
    self._split_ops = split_ops

  def bracket(self, state, grad):
    raise NotImplementedError("Quantum structures use split operators")


  def split(self, H_func):
    return list(self._split_ops)

  def constraints(self, state):
    return None


class QuantumCompilerStructure:
  kind: StructureKind = StructureKind.QUANTUM

  def __init__(self, compiler: QuantumHamiltonianCompiler):
    self.compiler = compiler

  def bracket(self, state, grad):
    raise NotImplementedError("Quantum structures use unitary flow")


  def split(self, H_func):
    return None

  def constraints(self, state):
    return None

  def step(self, psi: Tensor, dt: float):
    return self.compiler.step(psi)

  def compile_unrolled_step(self, dt: float, unroll: int):
    return self.compiler.compile_unrolled(unroll)


    return self._compile_unrolled_3d(steps, imaginary)

  def _compile_unrolled_1d(self, steps: int, imaginary: bool):
    key = f"{'imag' if imaginary else 'real'}_1d_n{steps}"
    if key in self._jit_failed:
      return self._solver.step_imaginary if imaginary else self._solver.step
    plan = self._jit_cache.get(key)
    if plan is None:
      if imaginary:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out)
          return out.realize()
      else:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step(out)
          return out.realize()
      plan = self._jit_cache.setdefault(key, TinyJit(plan_fn))
    if not capturing:
      def run(x: Tensor) -> Tensor:
        try:
          out = plan(x)
          if key not in self._jit_checked and getenv("TINYGRAD_QUANTUM_JIT_CHECK", 0):
            ref = x
            for _ in range(steps):
              ref = self._solver.step_imaginary(ref) if imaginary else self._solver.step(ref)
            diff = (out - ref).abs().max().numpy()
            if diff > 1e-5:
              self._jit_cache.pop(key, None)
              self._jit_failed.add(key)
              return ref
            self._jit_checked.add(key)
          return out
        except JitError:
          self._jit_cache.pop(key, None)
          self._jit_failed.add(key)
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out) if imaginary else self._solver.step(out)
          return out
      return run
    if imaginary:
      def run(x: Tensor) -> Tensor:
        out = x
        for _ in range(steps):
          out = self._solver.step_imaginary(out)
        return out
      return run
    def run(x: Tensor) -> Tensor:
      out = x
      for _ in range(steps):
        out = self._solver.step(out)
      return out
    return run

  def _compile_unrolled_2d(self, steps: int, imaginary: bool):
    key = f"{'imag' if imaginary else 'real'}_2d_n{steps}"
    if key in self._jit_failed:
      return self._solver.step_imaginary if imaginary else self._solver.step
    plan = self._jit_cache.get(key)
    if plan is None:
      if imaginary:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out)
          return out.realize()
      else:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step(out)
          return out.realize()
      plan = self._jit_cache.setdefault(key, TinyJit(plan_fn))
    if not capturing:
      def run(x: Tensor) -> Tensor:
        try:
          out = plan(x)
          if key not in self._jit_checked and getenv("TINYGRAD_QUANTUM_JIT_CHECK", 0):
            ref = x
            for _ in range(steps):
              ref = self._solver.step_imaginary(ref) if imaginary else self._solver.step(ref)
            diff = (out - ref).abs().max().numpy()
            if diff > 1e-5:
              self._jit_cache.pop(key, None)
              self._jit_failed.add(key)
              return ref
            self._jit_checked.add(key)
          return out
        except JitError:
          self._jit_cache.pop(key, None)
          self._jit_failed.add(key)
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out) if imaginary else self._solver.step(out)
          return out
      return run
    if imaginary:
      def run(x: Tensor) -> Tensor:
        out = x
        for _ in range(steps):
          out = self._solver.step_imaginary(out)
        return out
      return run
    def run(x: Tensor) -> Tensor:
      out = x
      for _ in range(steps):
        out = self._solver.step(out)
      return out
    return run

  def _compile_unrolled_3d(self, steps: int, imaginary: bool):
    key = f"{'imag' if imaginary else 'real'}_3d_n{steps}"
    if key in self._jit_failed:
      return self._solver.step_imaginary if imaginary else self._solver.step
    plan = self._jit_cache.get(key)
    if plan is None:
      if imaginary:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out)
          return out.realize()
      else:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step(out)
          return out.realize()
      plan = self._jit_cache.setdefault(key, TinyJit(plan_fn))
    if not capturing:
      def run(x: Tensor) -> Tensor:
        try:
          out = plan(x)
          if key not in self._jit_checked and getenv("TINYGRAD_QUANTUM_JIT_CHECK", 0):
            ref = x
            for _ in range(steps):
              ref = self._solver.step_imaginary(ref) if imaginary else self._solver.step(ref)
            diff = (out - ref).abs().max().numpy()
            if diff > 1e-5:
              self._jit_cache.pop(key, None)
              self._jit_failed.add(key)
              return ref
            self._jit_checked.add(key)
          return out
        except JitError:
          self._jit_cache.pop(key, None)
          self._jit_failed.add(key)
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out) if imaginary else self._solver.step(out)
          return out
      return run
    if imaginary:
      def run(x: Tensor) -> Tensor:
        out = x
        for _ in range(steps):
          out = self._solver.step_imaginary(out)
        return out
      return run
    def run(x: Tensor) -> Tensor:
      out = x
      for _ in range(steps):
        out = self._solver.step(out)
      return out
    return run
    key = f"{'imag' if imaginary else 'real'}_n{steps}"
    plan = self._jit_cache.get(key)
    if plan is None:
      if imaginary:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out)
          return out.realize()
      else:
        def plan_fn(x: Tensor) -> Tensor:
          out = x
          for _ in range(steps):
            out = self._solver.step(out)
          return out.realize()
      plan = self._jit_cache.setdefault(key, TinyJit(plan_fn))
    if not capturing:
      def run(x: Tensor) -> Tensor:
        try:
          return plan(x)
        except JitError:
          self._jit_cache.pop(key, None)
          out = x
          for _ in range(steps):
            out = self._solver.step_imaginary(out) if imaginary else self._solver.step(out)
          return out
      return run
    if imaginary:
      def run(x: Tensor) -> Tensor:
        out = x
        for _ in range(steps):
          out = self._solver.step_imaginary(out)
        return out
      return run
    def run(x: Tensor) -> Tensor:
      out = x
      for _ in range(steps):
        out = self._solver.step(out)
      return out
    return run


__all__ = [
  "QuantumHamiltonianCompiler",
  "QuantumSplitOperator1D",
  "QuantumSplitOperatorND",
  "QuantumSplitStructure",
  "QuantumCompilerStructure",
  "gaussian_wavepacket",
]
