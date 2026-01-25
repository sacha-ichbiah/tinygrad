import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.lie_poisson import SO3Structure
from tinyphysics.structures.conformal import ConformalStructure
from tinyphysics.structures.contact import LangevinStructure, BerendsenBarostatStructure
from tinyphysics.systems.molecular import LennardJonesSystem, lj_pressure, lj_energy
from tinyphysics.structures.commutator import QuantumHamiltonianCompiler, QuantumCompilerStructure, gaussian_wavepacket
from tinygrad.physics import IdealFluidVorticity2D


def bench_canonical(steps: int = 100, n: int = 1024):
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(n).astype(np.float32))
  p = Tensor(np.random.randn(n).astype(np.float32))
  prog = compile_structure(state=(q, p), H=H, structure=CanonicalStructure(), integrator="leapfrog")
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_so3(steps: int = 100):
  def H(L: Tensor):
    return 0.5 * (L * L).sum()
  L = Tensor(np.random.randn(3).astype(np.float32))
  prog = compile_structure(state=L, H=H, structure=SO3Structure(), integrator="midpoint")
  t0 = time.time()
  L, _ = prog.evolve(L, 0.01, steps)
  _ = L.realize()
  return time.time() - t0


def bench_quantum(steps: int = 50, n: int = 128):
  x = Tensor.linspace(0, 1, n)
  psi = gaussian_wavepacket(x, x0=0.5, k0=5.0, sigma=0.1)
  compiler = QuantumHamiltonianCompiler((x,), dt=0.01)
  structure = QuantumCompilerStructure(compiler)
  prog = compile_structure(structure=structure, H=None)
  t0 = time.time()
  out = psi
  for _ in range(steps):
    out = prog.step(out, 0.01)
  _ = out.realize()
  return time.time() - t0


def bench_constraint(steps: int = 100):
  def H(q, p):
    return 0.5 * (p * p).sum()
  def constraint_fn(q):
    return (q * q).sum() - 1.0
  q = Tensor([1.0, 0.0])
  p = Tensor([0.0, 1.0])
  prog = compile_structure(state=(q, p), H=H, structure=CanonicalStructure(), integrator="leapfrog", constraint=constraint_fn)
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_dissipative(steps: int = 200):
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(64).astype(np.float32))
  p = Tensor(np.random.randn(64).astype(np.float32))
  prog = compile_structure(state=(q, p), H=H, structure=ConformalStructure(alpha=0.1), integrator="auto")
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_thermostat(steps: int = 200):
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(64).astype(np.float32))
  p = Tensor(np.random.randn(64).astype(np.float32))
  prog = compile_structure(state=(q, p), H=H, structure=LangevinStructure(gamma=0.2, kT=0.0, noise=False))
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_barostat(steps: int = 100):
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(64, 3).astype(np.float32))
  p = Tensor(np.random.randn(64, 3).astype(np.float32))
  box = Tensor([6.0], dtype=q.dtype)
  prog = compile_structure(state=(q, p, box), H=H, structure=BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0))
  t0 = time.time()
  (q, p, box), _ = prog.evolve((q, p, box), 0.01, steps)
  _ = q.realize(); _ = p.realize(); _ = box.realize()
  return time.time() - t0


def bench_fluid(steps: int = 20, n: int = 64):
  solver = IdealFluidVorticity2D(n)
  w = Tensor(np.random.randn(n, n).astype(np.float32))
  t0 = time.time()
  out = w
  for _ in range(steps):
    out = solver._step_tensor(out, 0.01)
  _ = out.realize()
  return time.time() - t0


def bench_lj_tensor_bins(steps: int = 10, n: int = 512):
  q = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.5 + 5.0
  p = Tensor(np.zeros((n, 3), dtype=np.float32))
  m = Tensor(np.ones((n,), dtype=np.float32))
  system = LennardJonesSystem(mass=m, box=10.0, r_cut=2.5, method="tensor_bins", max_per="auto")
  prog = system.compile(q, p)
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.005, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_lj_barostat(steps: int = 10, n: int = 256):
  q = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.5 + 5.0
  p = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.1
  box = Tensor([10.0], dtype=q.dtype)
  def H(qv, pv):
    kinetic = 0.5 * (pv * pv).sum()
    potential = lj_energy(qv, sigma=1.0, epsilon=1.0, softening=1e-6, box=10.0, r_cut=2.5, periodic=False, shift=True)
    return kinetic + potential
  def pressure_fn(qv, pv, boxv):
    return lj_pressure(qv, pv, sigma=1.0, epsilon=1.0, softening=1e-6, box=boxv[0], r_cut=2.5, periodic=False)
  prog = compile_structure(state=(q, p, box), H=H, structure=BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0,
                                                                                         pressure_fn=pressure_fn))
  t0 = time.time()
  (q, p, box), _ = prog.evolve((q, p, box), 0.005, steps)
  _ = q.realize(); _ = p.realize(); _ = box.realize()
  return time.time() - t0


def _parse_threshold(env_key: str) -> float | None:
  raw = os.getenv(env_key, "")
  if raw == "":
    return None
  try:
    val = float(raw)
  except ValueError:
    return None
  return val if val > 0 else None


def _check_threshold(name: str, elapsed: float, max_s: float | None):
  if max_s is None:
    return
  if elapsed > max_s:
    raise RuntimeError(f"benchmark {name} exceeded threshold: {elapsed:.4f}s > {max_s:.4f}s")


if __name__ == "__main__":
  t_can = bench_canonical()
  t_lp = bench_so3()
  t_q = bench_quantum()
  t_c = bench_constraint()
  t_d = bench_dissipative()
  t_f = bench_fluid()
  t_th = bench_thermostat() if os.getenv("TINYGRAD_BENCH_THERMOSTAT", "0") else None
  t_lj = bench_lj_tensor_bins() if os.getenv("TINYGRAD_BENCH_LJ", "0") else None
  t_baro = bench_barostat() if os.getenv("TINYGRAD_BENCH_BAROSTAT", "0") else None
  t_lj_baro = bench_lj_barostat() if os.getenv("TINYGRAD_BENCH_LJ_BAROSTAT", "0") else None
  _check_threshold("canonical", t_can, _parse_threshold("TINYGRAD_BENCH_CANONICAL_MAX"))
  _check_threshold("so3", t_lp, _parse_threshold("TINYGRAD_BENCH_SO3_MAX"))
  _check_threshold("quantum", t_q, _parse_threshold("TINYGRAD_BENCH_QUANTUM_MAX"))
  _check_threshold("constraint", t_c, _parse_threshold("TINYGRAD_BENCH_CONSTRAINT_MAX"))
  _check_threshold("dissipative", t_d, _parse_threshold("TINYGRAD_BENCH_DISSIPATIVE_MAX"))
  _check_threshold("fluid", t_f, _parse_threshold("TINYGRAD_BENCH_FLUID_MAX"))
  if t_th is not None:
    _check_threshold("thermostat", t_th, _parse_threshold("TINYGRAD_BENCH_THERMOSTAT_MAX"))
  if t_lj is not None:
    _check_threshold("lj", t_lj, _parse_threshold("TINYGRAD_BENCH_LJ_MAX"))
  if t_baro is not None:
    _check_threshold("barostat", t_baro, _parse_threshold("TINYGRAD_BENCH_BAROSTAT_MAX"))
  if t_lj_baro is not None:
    _check_threshold("lj_barostat", t_lj_baro, _parse_threshold("TINYGRAD_BENCH_LJ_BAROSTAT_MAX"))
  print(f"canonical: {t_can:.4f}s")
  print(f"so3: {t_lp:.4f}s")
  print(f"quantum: {t_q:.4f}s")
  print(f"constraint: {t_c:.4f}s")
  print(f"dissipative: {t_d:.4f}s")
  print(f"fluid: {t_f:.4f}s")
  if t_th is not None:
    print(f"thermostat: {t_th:.4f}s")
  if t_lj is not None:
    print(f"lj: {t_lj:.4f}s")
  if t_baro is not None:
    print(f"barostat: {t_baro:.4f}s")
  if t_lj_baro is not None:
    print(f"lj_barostat: {t_lj_baro:.4f}s")
