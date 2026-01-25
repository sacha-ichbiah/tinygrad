from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class Operator:
  """Minimal operator wrapper to keep linear ops explicit."""
  name: str
  fn: Callable[..., Any]

  def __call__(self, *args, **kwargs):
    return self.fn(*args, **kwargs)


__all__ = ["Operator"]
