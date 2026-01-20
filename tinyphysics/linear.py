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


def _build_leapfrog_step_matrix(K: np.ndarray, M_inv: np.ndarray, dt: float) -> np.ndarray:
  if K.shape != M_inv.shape:
    raise ValueError("K and M_inv must have the same shape")
  if K.shape[0] != K.shape[1]:
    raise ValueError("K must be square")
  n = K.shape[0]
  I = np.eye(n, dtype=np.float64)
  zeros = np.zeros((n, n), dtype=np.float64)
  K_half = np.block([[I, zeros], [-0.5 * dt * K, I]])
  D = np.block([[I, dt * M_inv], [zeros, I]])
  return K_half @ D @ K_half


class LinearSymplecticSystem:
  """Decompose quadratic Hamiltonians into linear maps and apply matrix-powered leapfrog."""

  def __init__(self, H, dt: float = 0.01, linear_tol: float = 1e-6, min_steps: int = 1000, fd_eps: float = 1e-3):
    self.H = H
    self.dt = dt
    self.linear_tol = linear_tol
    self.min_steps = min_steps
    self.fd_eps = fd_eps
    self._decomposed = False
    self._size: int | None = None
    self._shape: tuple[int, ...] | None = None
    self._K: np.ndarray | None = None
    self._M_inv: np.ndarray | None = None
    self._M_step: np.ndarray | None = None
    self._propagator_cache: dict[tuple[int, str, object], Tensor] = {}

  def should_use_fast_path(self, steps: int, min_steps: int | None = None) -> bool:
    threshold = self.min_steps if min_steps is None else min_steps
    return steps >= threshold

  def _eval_H(self, q_np: np.ndarray, p_np: np.ndarray) -> float:
    q_t = Tensor(q_np.astype(np.float32))
    p_t = Tensor(p_np.astype(np.float32))
    return float(self.H(q_t, p_t).numpy())

  def _grad_np(self, q_np: np.ndarray, p_np: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = q_np.size
    dHdq = np.zeros(n, dtype=np.float64)
    dHdp = np.zeros(n, dtype=np.float64)
    eps = self.fd_eps

    for i in range(n):
      q_plus = q_np.copy()
      q_minus = q_np.copy()
      q_plus[i] += eps
      q_minus[i] -= eps
      dHdq[i] = (self._eval_H(q_plus, p_np) - self._eval_H(q_minus, p_np)) / (2.0 * eps)

      p_plus = p_np.copy()
      p_minus = p_np.copy()
      p_plus[i] += eps
      p_minus[i] -= eps
      dHdp[i] = (self._eval_H(q_np, p_plus) - self._eval_H(q_np, p_minus)) / (2.0 * eps)

    return dHdq, dHdp

  def _check_linearity(self, K: np.ndarray, M_inv: np.ndarray) -> None:
    if self._size is None:
      raise ValueError("size must be set before linearity checks")
    rng = np.random.default_rng(0)
    q = rng.standard_normal(self._size).astype(np.float64)
    p = rng.standard_normal(self._size).astype(np.float64)
    dHdq, dHdp = self._grad_np(q, p)
    if not np.allclose(dHdq, K @ q, rtol=self.linear_tol, atol=self.linear_tol):
      raise ValueError("dH/dq is not linear in q")
    if not np.allclose(dHdp, M_inv @ p, rtol=self.linear_tol, atol=self.linear_tol):
      raise ValueError("dH/dp is not linear in p")

  def decompose(self, q0: Tensor, p0: Tensor) -> None:
    if q0.shape != p0.shape:
      raise ValueError("q0 and p0 must have the same shape")
    if q0.dtype != p0.dtype:
      raise ValueError("q0 and p0 must have the same dtype")
    if q0.device != p0.device:
      raise ValueError("q0 and p0 must be on the same device")

    q_flat = q0.detach().numpy().reshape(-1).astype(np.float32)
    n = q_flat.size
    zeros = np.zeros(n, dtype=np.float64)

    K = np.zeros((n, n), dtype=np.float64)
    M_inv = np.zeros((n, n), dtype=np.float64)

    for i in range(n):
      e = zeros.copy()
      e[i] = 1.0
      dHdq, dHdp = self._grad_np(e, zeros)
      K[:, i] = dHdq
      if np.max(np.abs(dHdp)) > self.linear_tol:
        raise ValueError("Hamiltonian mixes q and p (dH/dp depends on q)")

    for i in range(n):
      e = zeros.copy()
      e[i] = 1.0
      dHdq, dHdp = self._grad_np(zeros, e)
      M_inv[:, i] = dHdp
      if np.max(np.abs(dHdq)) > self.linear_tol:
        raise ValueError("Hamiltonian mixes q and p (dH/dq depends on p)")

    self._size = n
    self._shape = q0.shape
    self._check_linearity(K, M_inv)
    self._K = K
    self._M_inv = M_inv
    self._M_step = _build_leapfrog_step_matrix(K, M_inv, self.dt)
    self._propagator_cache.clear()
    self._decomposed = True

  def compile_propagator(self, steps: int, device: str, dtype) -> Tensor:
    if steps < 0:
      raise ValueError("steps must be >= 0")
    if not self._decomposed or self._M_step is None:
      raise ValueError("system must be decomposed before compiling")
    key = (steps, device, dtype)
    cached = self._propagator_cache.get(key)
    if cached is not None:
      return cached

    if steps == 0:
      n = self._size if self._size is not None else 0
      M_total = np.eye(2 * n, dtype=np.float64)
    else:
      M_total = np.linalg.matrix_power(self._M_step, steps)

    target_dtype = np.float64 if dtype == dtypes.double else np.float32
    M_tensor = Tensor(M_total.astype(target_dtype), device=device)
    self._propagator_cache[key] = M_tensor
    return M_tensor

  def forward(self, q0: Tensor, p0: Tensor, steps: int) -> tuple[Tensor, Tensor]:
    if not self._decomposed:
      self.decompose(q0, p0)
    if self._shape != q0.shape:
      self.decompose(q0, p0)

    if q0.shape != p0.shape:
      raise ValueError("q0 and p0 must have the same shape")
    if q0.dtype != p0.dtype:
      raise ValueError("q0 and p0 must have the same dtype")
    if q0.device != p0.device:
      raise ValueError("q0 and p0 must be on the same device")

    if self._size is None:
      raise ValueError("system size not initialized")
    M = self.compile_propagator(steps, q0.device, q0.dtype)

    q_flat = q0.reshape(-1)
    p_flat = p0.reshape(-1)
    n = self._size
    A = M[:n, :n]
    B = M[:n, n:]
    C = M[n:, :n]
    D = M[n:, n:]
    q1 = (A * q_flat).sum(axis=1) + (B * p_flat).sum(axis=1)
    p1 = (C * q_flat).sum(axis=1) + (D * p_flat).sum(axis=1)
    return q1.reshape(q0.shape), p1.reshape(p0.shape)
