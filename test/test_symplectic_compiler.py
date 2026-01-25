import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import (
  HamiltonianSystem, compile_symplectic_program, symplectic_adjoint, rattle_project,
  conformal_symplectic_euler, poisson_solve_fft2
)


def test_symplectic_program_matches_system():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, 0.5])
  p = Tensor([0.2, -0.1])
  dt = 0.05

  sys = HamiltonianSystem(H, integrator="leapfrog")
  prog = compile_symplectic_program("canonical", H=H, integrator="leapfrog")

  q1, p1 = sys.step(q, p, dt)
  q2, p2 = prog.step((q, p), dt)

  assert (q1 - q2).abs().max().numpy() < 1e-6
  assert (p1 - p2).abs().max().numpy() < 1e-6


def test_compiler_auto_split_pass():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, 0.5])
  p = Tensor([0.2, -0.1])
  prog = compile_symplectic_program("canonical", H=H, integrator="auto", sample_state=(q, p))
  assert [op.name for op in prog.ops] == ["SYMP_KICK_HALF", "SYMP_DRIFT", "SYMP_KICK_HALF"]


def test_compiler_auto_implicit_pass():
  def H(q, p):
    return (q * p).sum()

  q = Tensor([1.0, 0.5])
  p = Tensor([0.2, -0.1])
  prog = compile_symplectic_program("canonical", H=H, integrator="auto", sample_state=(q, p))
  assert prog.ops[0].name == "SYMP_STEP"


def test_compiler_linear_fast_path():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, 0.5])
  p = Tensor([0.2, -0.1])
  prog = compile_symplectic_program("canonical", H=H, integrator="leapfrog", sample_state=(q, p))
  assert prog.ops[0].name in ("SYMP_STEP", "SYMP_STEP_LINEAR")

  q6 = Tensor([1.0, 0.5, -0.2, 0.1, 0.3, -0.4])
  p6 = Tensor([0.2, -0.1, 0.4, -0.3, 0.05, -0.25])
  prog6 = compile_symplectic_program("canonical", H=H, integrator="leapfrog", sample_state=(q6, p6))
  assert prog6.ops[0].name in ("SYMP_STEP", "SYMP_STEP_LINEAR")


def test_symplectic_adjoint_matches_autograd():
  dt = 0.05
  def step_fn(state, dt):
    q, p = state
    return q + dt * p, p - dt * q

  q0 = Tensor([1.0, 0.5], requires_grad=True)
  p0 = Tensor([0.2, -0.1], requires_grad=True)
  q1, p1 = step_fn((q0, p0), dt)
  loss = (q1 * q1).sum() + (p1 * p1).sum()
  loss.backward()
  gq, gp = q0.grad.detach(), p0.grad.detach()

  q0d = q0.detach()
  p0d = p0.detach()
  history = [(q0d, p0d), step_fn((q0d, p0d), dt)]
  grad_final = (2 * history[-1][0], 2 * history[-1][1])
  adj_q, adj_p = symplectic_adjoint(step_fn, history, grad_final, dt)

  assert (gq - adj_q).abs().max().numpy() < 1e-6
  assert (gp - adj_p).abs().max().numpy() < 1e-6


def test_rattle_projection():
  def constraint(q):
    return (q * q).sum() - 1.0

  q = Tensor([1.5, 0.0, 0.0])
  p = Tensor([0.2, 0.1, -0.3])
  q2, p2 = rattle_project(q, p, constraint, tol=1e-8, max_iter=20)

  g = float(constraint(q2).numpy())
  assert abs(g) < 1e-6

  q_req = q2.detach().requires_grad_(True)
  constraint(q_req).backward()
  grad = q_req.grad.detach()
  ortho = float((grad * p2).sum().numpy())
  assert abs(ortho) < 1e-6


def test_project_every_stride():
  def H(q, p):
    return (p * p).sum() / 2

  def constraint(q):
    return (q * q).sum() - 1.0

  q = Tensor([1.0, 0.0, 0.0])
  p = Tensor([0.4, 0.0, 0.0])
  prog = compile_symplectic_program("canonical", H=H, constraint=constraint, project_every=2)
  assert prog.project_every == 2
  (_, _), history = prog.evolve((q, p), 0.1, 2, unroll=1)
  q1, _ = history[1]
  q2, _ = history[2]
  g1 = float(constraint(q1).numpy())
  g2 = float(constraint(q2).numpy())
  assert abs(g1) < 1e-6
  assert abs(g2) < 1e-6


def test_conformal_dissipation_reduces_energy():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, -0.5])
  p = Tensor([0.2, 0.3])
  e0 = float(H(q, p).numpy())
  for _ in range(10):
    q, p = conformal_symplectic_euler(q, p, H, dt=0.05, alpha=1.0)
  e1 = float(H(q, p).numpy())
  assert e1 < e0


def test_poisson_fft2_solver():
  n = 32
  L = 2 * np.pi
  x = np.linspace(0, L, n, endpoint=False)
  X, Y = np.meshgrid(x, x, indexing="ij")
  kx = 1
  ky = 2
  w = np.sin(kx * X) * np.sin(ky * Y)
  w_t = Tensor(w.astype(np.float32))
  psi = poisson_solve_fft2(w_t, L=L).numpy()

  k2 = kx * kx + ky * ky
  target = -w / k2
  err = np.max(np.abs(psi - target))
  assert err < 1e-3
