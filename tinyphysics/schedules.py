from __future__ import annotations
from typing import Callable


def strang(ops: list[Callable]) -> list[Callable]:
  if not ops:
    return []
  if len(ops) == 1:
    return ops
  first = ops[0]
  rest = ops[1:]
  return [first, *rest, first]


def yoshida4(ops: list[Callable]) -> list[Callable]:
  if len(ops) == 0:
    return []
  if len(ops) == 1:
    return ops
  # Standard 4th-order Yoshida coefficients for symmetric composition
  w1 = 0.6756035959798289
  w0 = -0.1756035959798288
  seq = []
  for w in (w1, w0, w1):
    for op in ops:
      def _wrap(fn, c):
        return lambda state, dt: fn(state, c * dt)
      seq.append(_wrap(op, w))
  return seq


SCHEDULES = {
  "strang": strang,
  "yoshida4": yoshida4,
}


__all__ = ["SCHEDULES", "strang", "yoshida4"]
