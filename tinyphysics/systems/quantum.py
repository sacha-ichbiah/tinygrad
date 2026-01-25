from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinygrad.tensor import Tensor
from tinyphysics.core.compiler import compile_structure
from tinyphysics.structures.commutator import QuantumHamiltonianCompiler, QuantumCompilerStructure


@dataclass
class QuantumSystem:
  grids: tuple[Tensor, ...]
  dt: float
  mass: float = 1.0
  hbar: float = 1.0
  V: Callable | None = None
  g: float = 0.0

  def compile(self):
    compiler = QuantumHamiltonianCompiler(self.grids, self.dt, mass=self.mass, hbar=self.hbar, V=self.V, g=self.g)
    structure = QuantumCompilerStructure(compiler)
    return compile_structure(structure=structure, H=None)


__all__ = ["QuantumSystem", "QuantumHamiltonianCompiler"]
