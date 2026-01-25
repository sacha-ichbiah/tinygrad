from typing import TYPE_CHECKING

__all__ = [
  "UniversalSymplecticCompiler",
  "compile_universal",
  "StructureCompiler",
  "compile_structure",
  "StructureKind",
  "Structure",
  "StructureProgram",
  "QuantumHamiltonianCompiler",
  "QuantumSplitOperator1D",
  "QuantumSplitOperatorND",
  "gaussian_wavepacket",
  "PhysicalSystem",
]

if TYPE_CHECKING:
  from .compiler import UniversalSymplecticCompiler, compile_universal
  from .core.compiler import StructureCompiler, compile_structure
  from .core.structure import Structure, StructureKind, StructureProgram
  from .quantum import QuantumHamiltonianCompiler, QuantumSplitOperator1D, QuantumSplitOperatorND, gaussian_wavepacket
  from .physical import PhysicalSystem


def __getattr__(name: str):
  if name in ("UniversalSymplecticCompiler", "compile_universal"):
    from .compiler import UniversalSymplecticCompiler, compile_universal
    return UniversalSymplecticCompiler if name == "UniversalSymplecticCompiler" else compile_universal
  if name in ("StructureCompiler", "compile_structure"):
    from .core.compiler import StructureCompiler, compile_structure
    return StructureCompiler if name == "StructureCompiler" else compile_structure
  if name in ("Structure", "StructureKind", "StructureProgram"):
    from .core.structure import Structure, StructureKind, StructureProgram
    return {"Structure": Structure, "StructureKind": StructureKind, "StructureProgram": StructureProgram}[name]
  if name in ("QuantumHamiltonianCompiler", "QuantumSplitOperator1D", "QuantumSplitOperatorND", "gaussian_wavepacket"):
    from .quantum import QuantumHamiltonianCompiler, QuantumSplitOperator1D, QuantumSplitOperatorND, gaussian_wavepacket
    return {
      "QuantumHamiltonianCompiler": QuantumHamiltonianCompiler,
      "QuantumSplitOperator1D": QuantumSplitOperator1D,
      "QuantumSplitOperatorND": QuantumSplitOperatorND,
      "gaussian_wavepacket": gaussian_wavepacket,
    }[name]
  if name == "PhysicalSystem":
    from .physical import PhysicalSystem
    return PhysicalSystem
  raise AttributeError(name)
