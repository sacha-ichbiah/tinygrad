from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.structure import StructureKind


def contact_extend(q: Tensor, p: Tensor, s: Tensor, H: Callable, alpha: float, dt: float) -> tuple[Tensor, Tensor, Tensor]:
  q_req = q.detach().requires_grad_(True)
  p_req = p.detach().requires_grad_(True)
  Hval = H(q_req, p_req)
  Hval.backward()
  dHdq = q_req.grad.detach()
  dHdp = p_req.grad.detach()
  qn = q + dt * dHdp
  pn = p - dt * dHdq - dt * alpha * p
  sn = s + dt * (p * dHdp).sum() * alpha
  return qn, pn, sn


def langevin_extend(q: Tensor, p: Tensor, H: Callable, gamma: float, kT: float, dt: float, noise: bool) -> tuple[Tensor, Tensor]:
  q_req = q.detach().requires_grad_(True)
  p_req = p.detach().requires_grad_(True)
  Hval = H(q_req, p_req)
  Hval.backward()
  dHdq = q_req.grad.detach()
  dHdp = p_req.grad.detach()
  qn = q + dt * dHdp
  pn = p - dt * dHdq - dt * gamma * p
  if noise and kT > 0.0 and gamma > 0.0:
    sigma = (2.0 * gamma * kT * dt) ** 0.5
    pn = pn + sigma * Tensor.randn(*p.shape, device=p.device, dtype=p.dtype)
  return qn, pn


@dataclass
class ContactStructure:
  alpha: float = 0.0
  kind: StructureKind = StructureKind.DISSIPATIVE

  def bracket(self, state, grad):
    dq, dp, ds = grad
    return dp, -dq, -ds


  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state) -> Callable | None:
    return None


@dataclass
class LangevinStructure:
  gamma: float = 0.1
  kT: float = 0.0
  noise: bool = False
  kind: StructureKind = StructureKind.DISSIPATIVE

  def bracket(self, state, grad):
    dq, dp = grad
    return dp, -dq

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state) -> Callable | None:
    return None

  def build_program(self, H: Callable):
    gamma = self.gamma
    kT = self.kT
    noise = self.noise
    class _LangevinProgram:
      def __init__(self, H_func):
        self.H = H_func

      def step(self, state, dt: float):
        q, p = state
        return langevin_extend(q, p, self.H, gamma, kT, dt, noise)

      def evolve(self, state, dt: float, steps: int, record_every: int = 1, **_):
        history = []
        q, p = state
        for i in range(steps):
          if i % record_every == 0:
            history.append((q.detach(), p.detach()))
          q, p = langevin_extend(q, p, self.H, gamma, kT, dt, noise)
        history.append((q.detach(), p.detach()))
        return (q, p), history
    return _LangevinProgram(H)


__all__ = ["ContactStructure", "contact_extend", "LangevinStructure", "langevin_extend"]
