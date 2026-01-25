import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.conformal import ConformalStructure
from tinyphysics.structures.contact import LangevinStructure, NoseHooverChainStructure, BerendsenBarostatStructure


def test_contact_structure_reduces_energy():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  s = Tensor([0.0])
  structure = ConformalStructure(alpha=0.5, use_contact=True)
  prog = compile_structure(state=(q, p, s), H=H, structure=structure)
  e0 = float(H(q, p).numpy())
  (q1, p1, _), _ = prog.evolve((q, p, s), 0.05, 50)
  e1 = float(H(q1, p1).numpy())
  assert e1 < e0


def test_contact_structure_diagnostics():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  s = Tensor([0.0])
  structure = ConformalStructure(alpha=0.2, use_contact=True)
  prog = compile_structure(state=(q, p, s), H=H, structure=structure, contact_diagnostics=True)
  (_, _, _), history = prog.evolve((q, p, s), 0.05, 3)
  q0, p0, s0, e0 = history[0]
  assert q0.shape == q.shape
  assert p0.shape == p.shape
  assert s0.shape == s.shape
  assert e0.numel() == 1


def test_langevin_structure_damps_energy():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  structure = LangevinStructure(gamma=0.5, kT=0.0, noise=False)
  prog = compile_structure(state=(q, p), H=H, structure=structure)
  e0 = float(H(q, p).numpy())
  (q1, p1), _ = prog.evolve((q, p), 0.05, 20)
  e1 = float(H(q1, p1).numpy())
  assert e1 < e0


def test_langevin_structure_noise_generates_energy():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(32).astype(np.float32))
  p = Tensor(np.zeros((32,), dtype=np.float32))
  structure = LangevinStructure(gamma=0.2, kT=1.0, noise=True, diagnostics=True)
  prog = compile_structure(state=(q, p), H=H, structure=structure)
  (_, _), history = prog.evolve((q, p), 0.01, 10, record_every=1)
  kin = np.array([float(h[2].numpy()) for h in history[1:]])
  assert kin.mean() > 0.0


def test_nose_hoover_chain_structure_runs():
  Tensor.manual_seed(0)
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(4).astype(np.float32))
  p = Tensor(np.random.randn(4).astype(np.float32))
  xi = Tensor([0.0])
  structure = NoseHooverChainStructure(chain_len=1, kT=1.0, Q=1.0, diagnostics=True)
  prog = compile_structure(state=(q, p, xi), H=H, structure=structure)
  (q1, p1, xi1), history = prog.evolve((q, p, xi), 0.02, 10, record_every=2)
  assert q1.shape == q.shape
  assert p1.shape == p.shape
  assert xi1.shape == xi.shape
  assert len(history) > 0


def test_berendsen_barostat_runs():
  Tensor.manual_seed(0)
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(8, 3).astype(np.float32))
  p = Tensor(np.random.randn(8, 3).astype(np.float32))
  box = Tensor([6.0], dtype=q.dtype)
  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, diagnostics=True)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (q1, p1, box1), history = prog.evolve((q, p, box), 0.02, 5, record_every=1)
  assert q1.shape == q.shape
  assert p1.shape == p.shape
  assert box1.shape == box.shape
  assert len(history) > 0


def test_berendsen_barostat_diagnostics_pressure():
  Tensor.manual_seed(0)
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(8, 3).astype(np.float32))
  p = Tensor(np.random.randn(8, 3).astype(np.float32))
  box = Tensor([5.0], dtype=q.dtype)
  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, diagnostics=True)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (_, _, _), history = prog.evolve((q, p, box), 0.02, 3, record_every=1)
  sample = history[0]
  assert len(sample) == 5
  assert np.isfinite(float(sample[4].numpy().item()))


def test_berendsen_barostat_pressure_finite_over_steps():
  Tensor.manual_seed(0)
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(8, 3).astype(np.float32))
  p = Tensor(np.random.randn(8, 3).astype(np.float32))
  box = Tensor([5.0], dtype=q.dtype)
  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, diagnostics=True)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (_, _, _), history = prog.evolve((q, p, box), 0.02, 5, record_every=1)
  pressures = [float(h[4].numpy().item()) for h in history]
  assert np.isfinite(np.array(pressures)).all()


def test_berendsen_barostat_box_finite():
  Tensor.manual_seed(0)
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()

  q = Tensor(np.random.randn(16, 3).astype(np.float32))
  p = Tensor(np.random.randn(16, 3).astype(np.float32))
  box = Tensor([4.0], dtype=q.dtype)
  structure = BerendsenBarostatStructure(target_P=1.0, tau=0.5, kappa=1.0)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (q1, p1, box1), _ = prog.evolve((q, p, box), 0.02, 10)
  assert np.isfinite(q1.numpy()).all()
  assert np.isfinite(p1.numpy()).all()
  assert np.isfinite(box1.numpy()).all()
