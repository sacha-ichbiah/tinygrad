import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.linear import LinearSymplecticSystem


def H(q, p):
  return (p * p).sum() / 2 + (q * q).sum() / 2


def main():
  dt = 0.01
  steps = 10000
  system = LinearSymplecticSystem(H, dt=dt)

  q0 = Tensor(np.array([1.0, -0.5, 0.25], dtype=np.float32))
  p0 = Tensor(np.array([0.0, 0.1, -0.2], dtype=np.float32))

  q1, p1 = system.forward(q0, p0, steps)
  print("q_end", q1.numpy())
  print("p_end", p1.numpy())


if __name__ == "__main__":
  main()
