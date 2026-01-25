import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.physical import PhysicalSystem
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.structures.lie_poisson import SO3Structure
from tinyphysics.structures.conformal import ConformalStructure
from tinyphysics.structures.commutator import QuantumHamiltonianCompiler, QuantumCompilerStructure, gaussian_wavepacket
from tinyphysics.systems.nbody import NBodySystem


def demo_canonical():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(32).astype(np.float32))
  p = Tensor(np.random.randn(32).astype(np.float32))
  sys = PhysicalSystem(state=(q, p), H_func=H, structure=CanonicalStructure(), project_every=2)
  prog = sys.compile()
  (q, p), _ = prog.evolve((q, p), 0.01, 10)
  return float(H(q, p).numpy())


def demo_lie_poisson():
  def H(L):
    return 0.5 * (L * L).sum()
  L = Tensor(np.random.randn(3).astype(np.float32))
  sys = PhysicalSystem(state=L, H_func=H, structure=SO3Structure())
  prog = sys.compile()
  L, _ = prog.evolve(L, 0.01, 10)
  return float((L * L).sum().numpy())


def demo_quantum():
  x = Tensor.linspace(0, 1, 128)
  psi = gaussian_wavepacket(x, x0=0.5, k0=5.0, sigma=0.1)
  compiler = QuantumHamiltonianCompiler((x,), dt=0.01)
  sys = PhysicalSystem(state=psi, H_func=None, structure=QuantumCompilerStructure(compiler))
  prog = sys.compile()
  out = psi
  for _ in range(5):
    out = prog.step(out, 0.01)
  return float((out[..., 0] * out[..., 0] + out[..., 1] * out[..., 1]).sum().numpy())


def demo_contact():
  def H(q, p):
    return 0.5 * (q * q).sum() + 0.5 * (p * p).sum()
  q = Tensor(np.random.randn(8).astype(np.float32))
  p = Tensor(np.random.randn(8).astype(np.float32))
  s = Tensor([0.0])
  sys = PhysicalSystem(state=(q, p, s), H_func=H, structure=ConformalStructure(alpha=0.2, use_contact=True))
  prog = sys.compile()
  (q, p, _), _ = prog.evolve((q, p, s), 0.01, 10)
  return float(H(q, p).numpy())


def demo_nbody():
  n = 32
  q = Tensor(np.random.randn(n, 3).astype(np.float32))
  p = Tensor(np.random.randn(n, 3).astype(np.float32))
  m = Tensor(np.ones((n,), dtype=np.float32))
  system = NBodySystem(mass=m, method="tensor_bins", max_per=16)
  prog = system.compile(q, p)
  (q, p), _ = prog.evolve((q, p), 0.01, 2)
  return float((p * p).sum().numpy())


if __name__ == "__main__":
  print("canonical energy:", demo_canonical())
  print("lie-poisson |L|^2:", demo_lie_poisson())
  print("quantum norm:", demo_quantum())
  print("contact energy:", demo_contact())
  print("nbody kinetic:", demo_nbody())
