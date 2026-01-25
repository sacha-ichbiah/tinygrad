import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.contact import LangevinStructure, BerendsenBarostatStructure
from tinyphysics.systems.molecular import LennardJonesSystem, lj_energy, lj_pressure


def test_lennard_jones_shapes():
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((16, 3)).astype(np.float32) * 10.0)
  p = Tensor(rng.standard_normal((16, 3)).astype(np.float32) * 0.1)
  m = Tensor(np.ones((16,), dtype=np.float32))
  system = LennardJonesSystem(mass=m, box=10.0, r_cut=2.5, method="tensor")
  prog = system.compile(q, p)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 2)
  assert q1.shape == q.shape
  assert p1.shape == p.shape
  assert np.isfinite(q1.numpy()).all()
  assert np.isfinite(p1.numpy()).all()


def test_lennard_jones_langevin_runs():
  Tensor.manual_seed(0)
  rng = np.random.default_rng(1)
  q = Tensor(rng.random((32, 3)).astype(np.float32) * 6.0)
  p = Tensor(np.zeros((32, 3), dtype=np.float32))
  box = 6.0

  def H(qv, pv):
    return 0.5 * (pv * pv).sum() + lj_energy(qv, sigma=1.0, epsilon=1.0, softening=1e-6, box=box, r_cut=2.5, periodic=False)

  structure = LangevinStructure(gamma=0.1, kT=1.0, noise=True)
  prog = compile_structure(state=(q, p), H=H, structure=structure)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 5)
  assert np.isfinite(q1.numpy()).all()
  assert np.isfinite(p1.numpy()).all()


def test_lennard_jones_tensor_bins_runs():
  rng = np.random.default_rng(2)
  q = Tensor(rng.random((64, 3)).astype(np.float32) * 8.0)
  p = Tensor(rng.standard_normal((64, 3)).astype(np.float32) * 0.1)
  m = Tensor(np.ones((64,), dtype=np.float32))
  system = LennardJonesSystem(mass=m, box=8.0, r_cut=2.5, method="tensor_bins", max_per=32, table_every=1)
  prog = system.compile(q, p)
  (q1, p1), _ = prog.evolve((q, p), 0.01, 2)
  assert np.isfinite(q1.numpy()).all()
  assert np.isfinite(p1.numpy()).all()


def test_lennard_jones_pressure_finite():
  rng = np.random.default_rng(3)
  q = Tensor(rng.random((16, 3)).astype(np.float32) * 6.0)
  p = Tensor(rng.standard_normal((16, 3)).astype(np.float32) * 0.1)
  P = lj_pressure(q, p, sigma=1.0, epsilon=1.0, softening=1e-6, box=6.0, r_cut=2.5, periodic=False)
  assert np.isfinite(float(P.numpy().item()))


def test_lennard_jones_barostat_callback_runs():
  rng = np.random.default_rng(4)
  q = Tensor(rng.random((32, 3)).astype(np.float32) * 6.0)
  p = Tensor(rng.standard_normal((32, 3)).astype(np.float32) * 0.1)
  box = Tensor([6.0], dtype=q.dtype)

  def H(qv, pv):
    kinetic = 0.5 * (pv * pv).sum()
    potential = lj_energy(qv, sigma=1.0, epsilon=1.0, softening=1e-6, box=6.0, r_cut=2.5, periodic=False, shift=True)
    return kinetic + potential

  def pressure_fn(qv, pv, boxv):
    return lj_pressure(qv, pv, sigma=1.0, epsilon=1.0, softening=1e-6, box=boxv[0], r_cut=2.5, periodic=False)

  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, pressure_fn=pressure_fn)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (q1, p1, box1), _ = prog.evolve((q, p, box), 0.01, 3)
  assert np.isfinite(q1.numpy()).all()
  assert np.isfinite(p1.numpy()).all()
  assert np.isfinite(box1.numpy()).all()


def test_lennard_jones_energy_shift():
  q = Tensor(np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32))
  e0 = lj_energy(q, sigma=1.0, epsilon=1.0, softening=0.0, box=10.0, r_cut=2.5, periodic=False, shift=False)
  e1 = lj_energy(q, sigma=1.0, epsilon=1.0, softening=0.0, box=10.0, r_cut=2.5, periodic=False, shift=True)
  rc = 2.5
  sr2c = (1.0 / rc) ** 2
  sr6c = sr2c * sr2c * sr2c
  shift_val = 4.0 * (sr6c * sr6c - sr6c)
  diff = float((e0 - e1).numpy())
  assert np.isfinite(diff)
  assert np.allclose(diff, shift_val, rtol=1e-5, atol=1e-6)


def test_lennard_jones_force_shift():
  r = 1.0
  q = Tensor(np.array([[0.0, 0.0, 0.0], [r, 0.0, 0.0]], dtype=np.float32))
  p = Tensor(np.zeros((2, 3), dtype=np.float32))
  m = Tensor(np.ones((2,), dtype=np.float32))
  system0 = LennardJonesSystem(mass=m, box=10.0, r_cut=2.5, method="tensor", force_shift=False)
  system1 = LennardJonesSystem(mass=m, box=10.0, r_cut=2.5, method="tensor", force_shift=True)
  f0 = system0._force(q).numpy()
  f1 = system1._force(q).numpy()
  rc = 2.5
  sr2c = (1.0 / rc) ** 2
  sr6c = sr2c * sr2c * sr2c
  f_rc = 24.0 * (1.0 / (rc * rc)) * (2.0 * sr6c * sr6c - sr6c)
  expected = f_rc * (rc / r) * (-r)
  diff = f0[0, 0] - f1[0, 0]
  assert np.isfinite(diff)
  assert np.allclose(abs(diff), abs(expected), rtol=1e-5, atol=1e-6)
