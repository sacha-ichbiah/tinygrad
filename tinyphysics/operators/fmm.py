from __future__ import annotations
from dataclasses import dataclass

from tinygrad.tensor import Tensor


@dataclass
class FMMPlan:
  order: int = 2
  max_points: int = 1024


class FMMStub:
  """Placeholder Fast Multipole Method API (not implemented)."""
  def __init__(self, plan: FMMPlan | None = None):
    self.plan = plan or FMMPlan()

  def force(self, q: Tensor, charges: Tensor) -> Tensor:
    raise NotImplementedError("FMM is not implemented yet")


__all__ = ["FMMPlan", "FMMStub"]
