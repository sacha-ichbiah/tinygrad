from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from tinygrad.tensor import Tensor
from tinygrad.physics import IdealFluidVorticity2D


@dataclass
class IdealFluidSystem2D:
  N: int
  L: float = 2 * np.pi
  dealias: float = 2.0 / 3.0
  dtype = np.float32

  def __post_init__(self):
    self._solver = IdealFluidVorticity2D(self.N, L=self.L, dealias=self.dealias, dtype=self.dtype)

  def step(self, w: Tensor, dt: float, method: str = "midpoint", iters: int = 5) -> Tensor:
    return self._solver._step_tensor(w, dt, method=method, iters=iters)


__all__ = ["IdealFluidSystem2D"]
