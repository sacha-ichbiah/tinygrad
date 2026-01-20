import numpy as np

from tinygrad import dtypes
from tinygrad.tensor import Tensor


class LinearSymplecticSolver:
  """Fast-path for linear Hamiltonians using matrix-powered leapfrog integration."""

  def __init__(self, k: float, m: float, dt: float = 0.01):
    if m == 0:
      raise ValueError("m must be non-zero")
    self.k = k
    self.m = m
    self.dt = dt
    self.M_step = self._build_step_matrix(k, m, dt)
    self.propagator_cache: dict[tuple[int, str, object], Tensor] = {}

  @staticmethod
  def _build_step_matrix(k: float, m: float, dt: float) -> np.ndarray:
    # Leapfrog (velocity Verlet) in matrix form: Kick_half @ Drift @ Kick_half.
    D = np.array([[1.0, dt / m], [0.0, 1.0]], dtype=np.float64)
    K_half = np.array([[1.0, 0.0], [-0.5 * k * dt, 1.0]], dtype=np.float64)
    return K_half @ D @ K_half

  def compile_propagator(self, steps: int, device: str, dtype) -> Tensor:
    if steps < 0:
      raise ValueError("steps must be >= 0")
    key = (steps, device, dtype)
    cached = self.propagator_cache.get(key)
    if cached is not None:
      return cached

    if steps == 0:
      M_total = np.eye(2, dtype=np.float64)
    else:
      M_total = np.linalg.matrix_power(self.M_step, steps)

    target_dtype = np.float64 if dtype == dtypes.double else np.float32
    M_tensor = Tensor(M_total.astype(target_dtype), device=device)
    self.propagator_cache[key] = M_tensor
    return M_tensor

  def forward(self, q0: Tensor, p0: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    if q0.shape != p0.shape:
      raise ValueError("q0 and p0 must have the same shape")
    if q0.dtype != p0.dtype:
      raise ValueError("q0 and p0 must have the same dtype")
    if q0.device != p0.device:
      raise ValueError("q0 and p0 must be on the same device")

    M = self.compile_propagator(steps, q0.device, q0.dtype)
    m00, m01 = M[0, 0], M[0, 1]
    m10, m11 = M[1, 0], M[1, 1]
    q1 = m00 * q0 + m01 * p0
    p1 = m10 * q0 + m11 * p0
    return q1, p1
