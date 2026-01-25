from __future__ import annotations
from dataclasses import dataclass
from typing import Callable

from tinyphysics.operators.operator import Operator


@dataclass
class TracedOperator(Operator):
  """Operator that reports usage to the compiler via a simple callback."""
  _trace: Callable[[str], None] | None = None

  def __call__(self, *args, **kwargs):
    if self._trace is not None:
      self._trace(self.name)
    return super().__call__(*args, **kwargs)


__all__ = ["TracedOperator"]
