from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.operators.pme import pme_force
from tinyphysics.operators.ewald import ewald_energy


def direct_coulomb_force(q: Tensor, charges: Tensor, box: float, softening: float = 1e-6) -> Tensor:
  diff = q[:, None, :] - q[None, :, :]
  diff = diff - (diff / box).round() * box
  dist2 = (diff * diff).sum(axis=-1) + softening
  inv = 1.0 / (dist2 * dist2.sqrt())
  qiqj = charges[:, None] * charges[None, :]
  f = (qiqj * inv).unsqueeze(-1) * diff
  mask = 1.0 - Tensor.eye(q.shape[0], device=q.device, dtype=q.dtype)
  f = f * mask.unsqueeze(-1)
  return f.sum(axis=1)


@dataclass
class ElectrostaticsSystem:
  charges: Tensor
  box: float = 10.0
  method: str = "direct"  # direct|pme|ewald
  grid_n: int = 32
  alpha: float = 1.0
  r_cut: float = 2.5

  def _force(self, q: Tensor) -> Tensor:
    if self.method == "pme":
      return pme_force(q, self.charges, self.grid_n, self.box)
    return direct_coulomb_force(q, self.charges, self.box)

  def split_ops(self):
    def kick(state, dt):
      q, p = state
      f = self._force(q)
      return q, p + dt * f

    def drift(state, dt):
      q, p = state
      return q + dt * p, p

    return [kick, drift]

  def compile(self, q: Tensor, p: Tensor, integrator: str = "split", split_schedule: str = "strang"):
    structure = CanonicalStructure(split_ops=self.split_ops())
    def H(qv, pv):
      kinetic = 0.5 * (pv * pv).sum()
      if self.method == "ewald":
        potential = ewald_energy(qv, self.charges, self.box, self.alpha, self.r_cut)
      else:
        potential = Tensor([0.0], device=qv.device, dtype=qv.dtype)
      return kinetic + potential
    base = compile_structure(state=(q, p), H=H, structure=structure, integrator=integrator, split_schedule=split_schedule)

    class _NoVecProgram:
      def __init__(self, prog):
        self._prog = prog

      def step(self, state, dt: float):
        return self._prog.step(state, dt)

      def evolve(self, state, dt: float, steps: int, **kwargs):
        history = []
        cur = state
        record_every = kwargs.get("record_every", 1)
        for i in range(steps):
          if i % record_every == 0:
            history.append(cur)
          cur = self._prog.step(cur, dt)
        history.append(cur)
        return cur, history

      def compile_unrolled_step(self, dt: float, unroll: int):
        def run(state):
          cur = state
          for _ in range(unroll):
            cur = self._prog.step(cur, dt)
          return cur
        return run

    return _NoVecProgram(base)


__all__ = ["ElectrostaticsSystem", "direct_coulomb_force"]
