import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.contact import LangevinStructure
from tinyphysics.systems.molecular import lj_energy


def main():
  Tensor.manual_seed(0)
  n = 32
  box = 6.0
  q = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.5 + box * 0.5
  p = Tensor(np.zeros((n, 3), dtype=np.float32))

  def H(qv, pv):
    kinetic = 0.5 * (pv * pv).sum()
    # Set shift=True for energy-shifted cutoff (force_shift is on LennardJonesSystem).
    potential = lj_energy(qv, sigma=1.0, epsilon=1.0, softening=1e-6, box=box, r_cut=2.5, periodic=False, shift=True)
    return kinetic + potential

  structure = LangevinStructure(gamma=0.1, kT=1.0, noise=True, diagnostics=True)
  prog = compile_structure(state=(q, p), H=H, structure=structure)
  (_, _), history = prog.evolve((q, p), 0.01, 20, record_every=5)
  kin = [float(h[2].numpy()) for h in history]
  print(f"LJ Langevin: steps={len(history)-1}, kin_mean={np.mean(kin):.4f}")


if __name__ == "__main__":
  main()
