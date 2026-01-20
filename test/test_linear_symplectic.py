import numpy as np

from tinygrad.helpers import Context
from tinygrad.tensor import Tensor
from tinyphysics.linear import LinearSymplecticSystem, _build_leapfrog_step_matrix


def test_linear_symplectic_system_oscillator():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  dt = 0.05
  steps = 20
  system = LinearSymplecticSystem(H, dt=dt, linear_tol=1e-4, fd_eps=1e-3)

  q0 = Tensor(np.array([0.1, -0.2, 0.3], dtype=np.float32))
  p0 = Tensor(np.array([0.0, 0.05, -0.1], dtype=np.float32))

  with Context(NOOPT=1, DEVECTORIZE=0):
    q1, p1 = system.forward(q0, p0, steps)

  n = q0.numel()
  K = np.eye(n, dtype=np.float64)
  M_inv = np.eye(n, dtype=np.float64)
  M_step = _build_leapfrog_step_matrix(K, M_inv, dt)
  M_total = np.linalg.matrix_power(M_step, steps)

  state = np.concatenate([q0.numpy(), p0.numpy()], axis=0)
  ref = M_total @ state

  np.testing.assert_allclose(q1.numpy(), ref[:n], rtol=2e-3, atol=2e-4)
  np.testing.assert_allclose(p1.numpy(), ref[n:], rtol=2e-3, atol=2e-4)


def test_linear_symplectic_system_coupled():
  K = np.array([[2.0, -0.5, 0.0], [-0.5, 1.5, 0.2], [0.0, 0.2, 1.0]], dtype=np.float64)
  M_inv = np.diag([1.2, 0.8, 2.0]).astype(np.float64)

  def H(q, p):
    q_np = q.numpy().astype(np.float64)
    p_np = p.numpy().astype(np.float64)
    val = 0.5 * (q_np @ K @ q_np) + 0.5 * (p_np @ M_inv @ p_np)
    return Tensor(np.array(val, dtype=np.float32))

  dt = 0.05
  steps = 12
  system = LinearSymplecticSystem(H, dt=dt, linear_tol=1e-3, fd_eps=1e-3)

  q0 = Tensor(np.array([0.1, -0.2, 0.3], dtype=np.float32))
  p0 = Tensor(np.array([0.0, 0.05, -0.1], dtype=np.float32))

  with Context(NOOPT=1, DEVECTORIZE=0):
    q1, p1 = system.forward(q0, p0, steps)

  M_step = _build_leapfrog_step_matrix(K, M_inv, dt)
  M_total = np.linalg.matrix_power(M_step, steps)
  state = np.concatenate([q0.numpy(), p0.numpy()], axis=0)
  ref = M_total @ state

  n = q0.numel()
  np.testing.assert_allclose(q1.numpy(), ref[:n], rtol=2e-3, atol=2e-4)
  np.testing.assert_allclose(p1.numpy(), ref[n:], rtol=2e-3, atol=2e-4)
