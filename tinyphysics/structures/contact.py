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
  sn = s + dt * ((p * dHdp).sum() - Hval) * alpha
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


def nose_hoover_chain_extend(q: Tensor, p: Tensor, xi: Tensor, H: Callable, Q: float, kT: float, dt: float) -> tuple[Tensor, Tensor, Tensor]:
  q_req = q.detach().requires_grad_(True)
  p_req = p.detach().requires_grad_(True)
  Hval = H(q_req, p_req)
  Hval.backward()
  dHdq = q_req.grad.detach()
  dHdp = p_req.grad.detach()
  dof = float(p.numel())
  K = 0.5 * (p * dHdp).sum()
  xi = xi + 0.5 * dt * (2.0 * K - dof * kT) / Q
  scale = (-xi * (0.5 * dt)).exp()
  p = p * scale
  p = p - 0.5 * dt * dHdq
  q = q + dt * dHdp
  q_req = q.detach().requires_grad_(True)
  p_req = p.detach().requires_grad_(True)
  Hval = H(q_req, p_req)
  Hval.backward()
  dHdq = q_req.grad.detach()
  dHdp = p_req.grad.detach()
  p = p - 0.5 * dt * dHdq
  p = p * scale
  K = 0.5 * (p * dHdp).sum()
  xi = xi + 0.5 * dt * (2.0 * K - dof * kT) / Q
  return q, p, xi


def berendsen_extend(q: Tensor, p: Tensor, box: Tensor, H: Callable, target_P: float, tau: float,
                     kappa: float, dt: float, pressure_fn: Callable | None) -> tuple[Tensor, Tensor, Tensor]:
  q_req = q.detach().requires_grad_(True)
  p_req = p.detach().requires_grad_(True)
  Hval = H(q_req, p_req)
  Hval.backward()
  if q_req.grad is None:
    dHdq = q.zeros_like()
  else:
    dHdq = q_req.grad.detach()
  if p_req.grad is None:
    dHdp = p.zeros_like()
  else:
    dHdp = p_req.grad.detach()
  qn = q + dt * dHdp
  pn = p - dt * dHdq
  if pressure_fn is None:
    K = 0.5 * (p * dHdp).sum()
    V = box * box * box
    P = (2.0 * K) / (3.0 * V)
  else:
    P = pressure_fn(q, p, box)
  finite = (P == P) & (P.abs() < 1e30)
  if isinstance(P, Tensor):
    P = finite.where(P, Tensor([target_P], device=q.device, dtype=q.dtype))
  scale = 1.0 + (dt * kappa * (P - target_P) / tau)
  scale = scale.maximum(1e-6)
  gamma = scale ** (1.0 / 3.0)
  qn = qn * gamma
  pn = pn * gamma
  boxn = box * gamma
  return qn, pn, boxn


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
  diagnostics: bool = False
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
    diagnostics = self.diagnostics
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
            if diagnostics:
              history.append((q.detach(), p.detach(), 0.5 * (p * p).sum().detach()))
            else:
              history.append((q.detach(), p.detach()))
          q, p = langevin_extend(q, p, self.H, gamma, kT, dt, noise)
        if diagnostics:
          history.append((q.detach(), p.detach(), 0.5 * (p * p).sum().detach()))
        else:
          history.append((q.detach(), p.detach()))
        return (q, p), history
    return _LangevinProgram(H)


@dataclass
class NoseHooverChainStructure:
  """Minimal Nose-Hoover chain thermostat (chain_len=1)."""
  chain_len: int = 1
  kT: float = 1.0
  Q: float = 1.0
  diagnostics: bool = False
  kind: StructureKind = StructureKind.DISSIPATIVE

  def bracket(self, state, grad):
    dq, dp = grad
    return dp, -dq

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state) -> Callable | None:
    return None

  def build_program(self, H: Callable):
    if self.chain_len != 1:
      raise NotImplementedError("Only chain_len=1 is supported right now.")
    kT = self.kT
    Q = self.Q
    diagnostics = self.diagnostics
    class _NHCProgram:
      def __init__(self, H_func):
        self.H = H_func

      def step(self, state, dt: float):
        q, p, xi = state
        return nose_hoover_chain_extend(q, p, xi, self.H, Q, kT, dt)

      def evolve(self, state, dt: float, steps: int, record_every: int = 1, **_):
        history = []
        q, p, xi = state
        for i in range(steps):
          if i % record_every == 0:
            if diagnostics:
              history.append((q.detach(), p.detach(), xi.detach(), 0.5 * (p * p).sum().detach()))
            else:
              history.append((q.detach(), p.detach(), xi.detach()))
          q, p, xi = nose_hoover_chain_extend(q, p, xi, self.H, Q, kT, dt)
        if diagnostics:
          history.append((q.detach(), p.detach(), xi.detach(), 0.5 * (p * p).sum().detach()))
        else:
          history.append((q.detach(), p.detach(), xi.detach()))
        return (q, p, xi), history
    return _NHCProgram(H)


@dataclass
class BerendsenBarostatStructure:
  target_P: float = 1.0
  tau: float = 1.0
  kappa: float = 1.0
  pressure_fn: Callable | None = None
  diagnostics: bool = False
  kind: StructureKind = StructureKind.DISSIPATIVE

  def bracket(self, state, grad):
    dq, dp, _ = grad
    return dp, -dq, state[2].zeros_like()

  def split(self, H_func: Callable | None) -> list[Callable] | None:
    return None

  def constraints(self, state) -> Callable | None:
    return None

  def build_program(self, H: Callable):
    target_P = self.target_P
    tau = self.tau
    kappa = self.kappa
    pressure_fn = self.pressure_fn
    diagnostics = self.diagnostics
    class _BerendsenProgram:
      def __init__(self, H_func):
        self.H = H_func

      def step(self, state, dt: float):
        q, p, box = state
        return berendsen_extend(q, p, box, self.H, target_P, tau, kappa, dt, pressure_fn)

      def evolve(self, state, dt: float, steps: int, record_every: int = 1, **_):
        history = []
        q, p, box = state
        for i in range(steps):
          if i % record_every == 0:
            if diagnostics:
              if pressure_fn is None:
                K = 0.5 * (p * p).sum()
                V = box * box * box
                P = (2.0 * K) / (3.0 * V)
              else:
                P = pressure_fn(q, p, box)
              history.append((q.detach(), p.detach(), box.detach(), 0.5 * (p * p).sum().detach(), P.detach()))
            else:
              history.append((q.detach(), p.detach(), box.detach()))
          q, p, box = berendsen_extend(q, p, box, self.H, target_P, tau, kappa, dt, pressure_fn)
        if diagnostics:
          if pressure_fn is None:
            K = 0.5 * (p * p).sum()
            V = box * box * box
            P = (2.0 * K) / (3.0 * V)
          else:
            P = pressure_fn(q, p, box)
          history.append((q.detach(), p.detach(), box.detach(), 0.5 * (p * p).sum().detach(), P.detach()))
        else:
          history.append((q.detach(), p.detach(), box.detach()))
        return (q, p, box), history
    return _BerendsenProgram(H)


__all__ = [
  "ContactStructure",
  "contact_extend",
  "LangevinStructure",
  "langevin_extend",
  "NoseHooverChainStructure",
  "nose_hoover_chain_extend",
  "BerendsenBarostatStructure",
  "berendsen_extend",
]
