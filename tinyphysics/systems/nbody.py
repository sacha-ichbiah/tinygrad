from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.helpers import getenv
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.canonical import CanonicalStructure
from tinyphysics.operators.neighbor import neighbor_forces
from tinyphysics.operators.tensor_neighbors import (
  neighbor_force_tensor,
  neighbor_force_from_pairs,
  neighbor_force_tensor_bins,
  autotune_max_per,
)
from tinyphysics.operators.barnes_hut import barnes_hut_forces


@dataclass
class NBodySystem:
  mass: Tensor
  G: float = 1.0
  softening: float = 1e-2
  r_cut: float = 1.0
  box: float = 10.0
  method: str = "neighbor"  # neighbor|barnes_hut|naive|tensor|tensor_bins|auto
  max_per: int | None | str = None

  def _force_np(self, q_np: np.ndarray, m: np.ndarray | None = None) -> np.ndarray:
    if m is None:
      m = self.mass.numpy().astype(np.float64)
    if self.method == "barnes_hut":
      return barnes_hut_forces(q_np, m, self.G, self.softening)
    if self.method == "naive":
      n = q_np.shape[0]
      forces = np.zeros_like(q_np)
      eps2 = self.softening * self.softening
      for i in range(n):
        for j in range(i + 1, n):
          d = q_np[j] - q_np[i]
          r2 = (d * d).sum() + eps2
          inv = 1.0 / (r2 * np.sqrt(r2))
          f = self.G * m[i] * m[j] * inv * d
          forces[i] += f
          forces[j] -= f
      return forces
    return neighbor_forces(q_np, m, self.G, self.softening, self.box, self.r_cut)

  def _force_tensor(self, q: Tensor) -> Tensor:
    eps2 = self.softening * self.softening
    if q.ndim == 2:
      q_use = q.unsqueeze(0)
    else:
      q_use = q
    if self.mass.ndim == 1:
      m = self.mass.unsqueeze(0)
    else:
      m = self.mass
    diff = q_use[:, :, None, :] - q_use[:, None, :, :]
    r2 = (diff * diff).sum(axis=-1) + eps2
    n = r2.shape[-1]
    mask = Tensor.eye(n, device=q.device, dtype=q.dtype).reshape(1, n, n)
    r2 = r2 + mask * 1e9
    inv = 1.0 / (r2 * r2.sqrt())
    mij = m[:, :, None] * m[:, None, :]
    f = self.G * (mij * inv).unsqueeze(-1) * diff
    out = f.sum(axis=2)
    return out[0] if q.ndim == 2 else out

  def split_ops(self):
    def kick(state, dt):
      q, p = state
      method = self.method
      if method == "auto":
        if "CPU" in q.device.upper():
          n = int(q.shape[0])
          if n <= 512:
            method = "tensor"
          else:
            method = "neighbor"
        else:
          method = "tensor_bins"
      if method == "tensor_bins":
        if "CPU" in q.device.upper():
          cpu_thresh = int(getenv("TINYGRAD_TENSOR_BINS_CPU_THRESHOLD", 512))
          if int(q.shape[0]) <= cpu_thresh:
            return q, p + dt * self._force_tensor(q)
        max_per = self.max_per
        if isinstance(max_per, str) and max_per == "auto":
          max_per = autotune_max_per(q, self.box, self.r_cut)
        f = neighbor_force_tensor_bins(q, self.mass, self.G, self.softening, self.box, self.r_cut, max_per=max_per)
      elif method == "tensor":
        if "CPU" in q.device.upper() and int(getenv("TINYGRAD_NBODY_TENSOR_CPU", 0)) == 0:
          q_np = q.detach().numpy()
          m_np = self.mass.numpy().astype(np.float64)
          if q_np.ndim == 3:
            if m_np.ndim == 2:
              forces = np.stack([self._force_np(q_np[i], m_np[i]) for i in range(q_np.shape[0])], axis=0)
            else:
              forces = np.stack([self._force_np(q_np[i], m_np) for i in range(q_np.shape[0])], axis=0)
          else:
            forces = self._force_np(q_np, m_np)
          f = Tensor(forces.astype(np.float32), device=q.device, dtype=q.dtype)
        else:
          f = self._force_tensor(q)
      else:
        q_np = q.detach().numpy()
        m_np = self.mass.numpy().astype(np.float64)
        if q_np.ndim == 3:
          if m_np.ndim == 2:
            forces = np.stack([self._force_np(q_np[i], m_np[i]) for i in range(q_np.shape[0])], axis=0)
          else:
            forces = np.stack([self._force_np(q_np[i], m_np) for i in range(q_np.shape[0])], axis=0)
        else:
          forces = self._force_np(q_np, m_np)
        f = Tensor(forces.astype(np.float32), device=q.device, dtype=q.dtype)
      return q, p + dt * f

    def drift(state, dt):
      q, p = state
      if self.mass.ndim == 1:
        invm = (1.0 / self.mass).reshape(-1, 1)
        if q.ndim == 3:
          invm = invm.reshape(1, invm.shape[0], 1)
      else:
        invm = (1.0 / self.mass).reshape(self.mass.shape[0], self.mass.shape[1], 1)
      return q + dt * (p * invm), p

    return [kick, drift]

  def compile(self, q: Tensor, p: Tensor, integrator: str = "split", split_schedule: str = "strang"):
    structure = CanonicalStructure(split_ops=self.split_ops())
    def H(qv, pv):
      return 0.5 * (pv * pv).sum()
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


__all__ = ["NBodySystem"]
