from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinygrad.physics import compile_symplectic_program
from tinyphysics.core.structure import Structure, StructureKind, StructureProgram
from tinyphysics.schedules import SCHEDULES
from tinyphysics.structures.commutator import QuantumHamiltonianCompiler


@dataclass
class StructureCompiler:
  """Unified structure-preserving compiler (symplectic + unitary)."""
  structure: Structure | None = None
  H: Callable | None = None
  integrator: str = "auto"
  policy: object | None = None
  constraint: Callable | None = None
  constraint_tol: float = 1e-9
  constraint_iters: int = 10
  program: StructureProgram | None = None

  # quantum
  grids: tuple[Tensor, ...] | None = None
  dt: float | None = None
  mass: float = 1.0
  hbar: float = 1.0
  V: Callable | None = None
  g: float = 0.0

  def compile(self, sample_state: tuple[Tensor, Tensor] | Tensor | None = None, **kwargs) -> StructureProgram:
    if self.structure is None:
      if self.grids is None or self.dt is None:
        raise ValueError("structure compiler requires a Structure or quantum grids+dt")
      qc = QuantumHamiltonianCompiler(self.grids, self.dt, mass=self.mass, hbar=self.hbar, V=self.V, g=self.g)
      self.program = StructureProgram(qc)
      return self.program
    self.program = compile_structure(
      state=sample_state,
      H=self.H,
      structure=self.structure,
      integrator=self.integrator,
      policy=self.policy,
      constraint=self.constraint,
      constraint_tol=self.constraint_tol,
      constraint_iters=self.constraint_iters,
      **kwargs,
    )
    return self.program

  def step(self, state, dt: float):
    if self.program is None:
      self.compile(sample_state=state if isinstance(state, tuple) else None)
    return self.program.step(state, dt)

  def evolve(self, state, dt: float, steps: int, **kwargs):
    if self.program is None:
      self.compile(sample_state=state if isinstance(state, tuple) else None)
    return self.program.evolve(state, dt, steps, **kwargs)


def _compile_quantum_from_structure(structure: Structure, H: Callable | None):
  if hasattr(structure, "step"):
    return StructureProgram(structure)
  split_ops = structure.split(H)
  if split_ops is None:
    raise ValueError("quantum structure must provide step() or split() ops")

  class _SplitProgram:
    def __init__(self, ops):
      self.ops = ops

    def step(self, psi: Tensor, dt: float):
      out = psi
      for op in self.ops:
        out = op(out, dt)
      return out

    def compile_unrolled_step(self, dt: float, unroll: int):
      if unroll < 1:
        raise ValueError("unroll must be >= 1")
      def run(x: Tensor):
        out = x
        for _ in range(unroll):
          for op in self.ops:
            out = op(out, dt)
        return out
      return run

  return StructureProgram(_SplitProgram(split_ops))


def compile_structure(
  state: tuple[Tensor, Tensor] | Tensor | None = None,
  H: Callable | None = None,
  structure: Structure | None = None,
  *,
  policy: object | None = None,
  integrator: str = "auto",
  constraint: Callable | None = None,
  constraint_tol: float = 1e-9,
  constraint_iters: int = 10,
  contact_diagnostics: bool = False,
  **kwargs,
) -> StructureProgram:
  if structure is None:
    kind = kwargs.get("kind")
    if kind is None:
      raise ValueError("compile_structure requires structure or kind")
    compiler = StructureCompiler(structure=None, H=H, integrator=integrator, policy=policy, constraint=constraint,
                                 constraint_tol=constraint_tol, constraint_iters=constraint_iters, **kwargs)
    return compiler.compile(sample_state=state)

  if structure.kind == StructureKind.QUANTUM:
    return _compile_quantum_from_structure(structure, H)

  if structure.kind == StructureKind.CANONICAL:
    operator_trace: list[str] = []
    if getattr(structure, "operator_trace", None) is not None:
      try:
        structure.operator_trace(operator_trace)
      except Exception:
        pass
    if operator_trace:
      kwargs.setdefault("operator_trace", tuple(operator_trace))
    constraint_fn = constraint
    if constraint_fn is None:
      constraint_fn = structure.constraints(state) if hasattr(structure, "constraints") else None
    split_ops = structure.split(H) if hasattr(structure, "split") else None
    if split_ops is not None and integrator == "auto":
      integrator = "split"
    if split_ops is not None:
      schedule = kwargs.get("split_schedule", "strang")
      scheduler = SCHEDULES.get(schedule, None)
      if scheduler is not None:
        split_ops = scheduler(split_ops)
    prog = compile_symplectic_program(
      "canonical",
      H=H,
      integrator=integrator,
      policy=policy,
      constraint=constraint_fn,
      constraint_tol=constraint_tol,
      constraint_iters=constraint_iters,
      project_every=kwargs.pop("project_every", None),
      split_ops=split_ops,
      sample_state=state if isinstance(state, tuple) else None,
      **kwargs,
    )
    return StructureProgram(prog)

  if structure.kind == StructureKind.LIE_POISSON:
    operator_trace: list[str] = []
    if getattr(structure, "operator_trace", None) is not None:
      try:
        structure.operator_trace(operator_trace)
      except Exception:
        pass
    if operator_trace:
      kwargs.setdefault("operator_trace", tuple(operator_trace))
    def J_func(z: Tensor):
      return lambda grad: structure.bracket(z, grad)
    prog = compile_symplectic_program(
      "lie_poisson",
      H=H,
      integrator=integrator,
      policy=policy,
      J=J_func,
      sample_state=state if isinstance(state, tuple) else None,
      **kwargs,
    )
    return StructureProgram(prog)

  if structure.kind == StructureKind.DISSIPATIVE:
    if hasattr(structure, "build_program"):
      return StructureProgram(structure.build_program(H))
    use_contact = getattr(structure, "use_contact", False)
    if use_contact:
      alpha = getattr(structure, "alpha", 0.0)
      from tinyphysics.structures.contact import contact_extend
      class _ContactProgram:
        def __init__(self, H_func, alpha_val, diagnostics: bool):
          self.H = H_func
          self.alpha = alpha_val
          self.diagnostics = diagnostics

        def step(self, state, dt: float):
          q, p, s = state
          return contact_extend(q, p, s, self.H, self.alpha, dt)

        def evolve(self, state, dt: float, steps: int, record_every: int = 1, **_):
          history = []
          q, p, s = state
          for i in range(steps):
            if i % record_every == 0:
              if self.diagnostics:
                Hval = self.H(q, p).detach()
                history.append((q.detach(), p.detach(), s.detach(), Hval, (Hval + s).detach()))
              else:
                history.append((q.detach(), p.detach(), s.detach()))
            q, p, s = contact_extend(q, p, s, self.H, self.alpha, dt)
          if self.diagnostics:
            Hval = self.H(q, p).detach()
            history.append((q.detach(), p.detach(), s.detach(), Hval, (Hval + s).detach()))
          else:
            history.append((q.detach(), p.detach(), s.detach()))
          return (q, p, s), history
      return StructureProgram(_ContactProgram(H, alpha, contact_diagnostics))
    alpha = getattr(structure, "alpha", 0.0)
    prog = compile_symplectic_program(
      "conformal",
      H=H,
      integrator=integrator,
      policy=policy,
      alpha=alpha,
      sample_state=state if isinstance(state, tuple) else None,
      **kwargs,
    )
    return StructureProgram(prog)

  raise ValueError(f"Unknown structure kind: {structure.kind}")


__all__ = ["StructureCompiler", "compile_structure"]
