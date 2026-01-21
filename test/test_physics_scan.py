import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


def test_leapfrog_scan_kernel_scalar():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0]), Tensor([0.0])
  q2, p2 = Tensor([1.0]), Tensor([0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=20, record_every=20)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=20)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-6, atol=1e-6)


def test_leapfrog_scan_kernel_vector():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=20, record_every=20)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=20)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-6, atol=1e-6)


def test_leapfrog_scan_kernel_coupled():
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=20, record_every=20)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=20, coupled=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-6, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_unroll():
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=20, record_every=20)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=20, coupled=True, unroll_steps=4)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-6, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-6, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_fused():
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True, coupled_fused=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_fused_vec_experimental():
  import os
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  os.environ["TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL"] = "1"
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True, coupled_fused=True, vector_width=2)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_multi_reduce():
  def H(q, p):
    qsum = q.sum()
    psum = p.sum()
    return (qsum * qsum) + (qsum * psum) + (psum * psum)

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.25, -0.5, 1.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.25, -0.5, 1.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_fused_full_reduce():
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1, p1 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.0, 0.0, 0.0])
  q2, p2 = Tensor([1.0, 0.5, -0.25, 0.75]), Tensor([0.0, 0.0, 0.0, 0.0])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True, coupled_fused=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_fused_full_reduce_vec_experimental():
  import os
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1 = Tensor([0.1 * i for i in range(8)])
  p1 = Tensor([0.0 for _ in range(8)])
  q2 = Tensor([0.1 * i for i in range(8)])
  p2 = Tensor([0.0 for _ in range(8)])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  prev = os.environ.get("TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL")
  os.environ["TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL"] = "1"
  try:
    q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True, coupled_fused=True, vector_width=4)
  finally:
    if prev is None:
      os.environ.pop("TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL", None)
    else:
      os.environ["TINYGRAD_COUPLED_FUSED_VEC_EXPERIMENTAL"] = prev

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_fused_dHdp_reduce_only():
  def H(q, p):
    psum = p.sum()
    return (q * q).sum() / 2 + (psum * psum) / 2

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1 = Tensor([0.1 * i for i in range(8)])
  p1 = Tensor([0.0 for _ in range(8)])
  q2 = Tensor([0.1 * i for i in range(8)])
  p2 = Tensor([0.0 for _ in range(8)])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True, coupled_fused=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)


def test_leapfrog_scan_kernel_coupled_multi_reduce_large():
  def H(q, p):
    qsum = q.sum()
    psum = p.sum()
    return (qsum * qsum) + (qsum * psum) + (psum * psum)

  system = HamiltonianSystem(H, integrator="leapfrog")

  q1 = Tensor([0.1 * i for i in range(16)])
  p1 = Tensor([0.2 * (i - 8) for i in range(16)])
  q2 = Tensor([0.1 * i for i in range(16)])
  p2 = Tensor([0.2 * (i - 8) for i in range(16)])

  q1, p1, _ = system.evolve(q1, p1, dt=0.01, steps=12, record_every=12)
  q2, p2, _ = system.evolve_scan_kernel(q2, p2, dt=0.01, steps=12, coupled=True)

  np.testing.assert_allclose(q1.numpy(), q2.numpy(), rtol=1e-3, atol=1e-6)
  np.testing.assert_allclose(p1.numpy(), p2.numpy(), rtol=1e-3, atol=1e-6)
