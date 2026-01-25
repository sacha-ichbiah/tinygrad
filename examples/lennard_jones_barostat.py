import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.contact import BerendsenBarostatStructure
from tinyphysics.systems.molecular import lj_energy, lj_pressure


def main():
  Tensor.manual_seed(0)
  n = 64
  box0 = 8.0
  q = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.5 + box0 * 0.5
  p = Tensor(np.random.randn(n, 3).astype(np.float32)) * 0.1
  box = Tensor([box0], dtype=q.dtype)

  def H(qv, pv):
    kinetic = 0.5 * (pv * pv).sum()
    potential = lj_energy(qv, sigma=1.0, epsilon=1.0, softening=1e-6, box=box0, r_cut=2.5, periodic=False, shift=True)
    return kinetic + potential

  def pressure_fn(qv, pv, boxv):
    return lj_pressure(qv, pv, sigma=1.0, epsilon=1.0, softening=1e-6, box=boxv[0], r_cut=2.5, periodic=False)

  structure = BerendsenBarostatStructure(target_P=1.0, tau=1.0, kappa=1.0, diagnostics=True, pressure_fn=pressure_fn)
  prog = compile_structure(state=(q, p, box), H=H, structure=structure)
  (q, p, box), history = prog.evolve((q, p, box), 0.01, 20, record_every=5)
  boxes = [float(h[2].numpy()) for h in history]
  print(f"Berendsen LJ: steps={len(history)-1}, box0={boxes[0]:.3f}, boxT={boxes[-1]:.3f}")


if __name__ == "__main__":
  main()
