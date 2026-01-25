import numpy as np

from tinygrad.tensor import Tensor
from tinyphysics.operators.spatial import grad2_op, div2_op, laplacian2_op, poisson_solve2_op


def main():
  n = 32
  L = 2 * np.pi
  x = np.linspace(0, L, n, endpoint=False)
  y = np.linspace(0, L, n, endpoint=False)
  X, Y = np.meshgrid(x, y, indexing="ij")
  f = Tensor((np.sin(X) * np.cos(Y)).astype(np.float32))

  gx, gy = grad2_op(L=L)(f)
  lap = laplacian2_op(L=L)(f)
  psi = poisson_solve2_op(L=L)(f)
  div = div2_op(L=L)(gx, gy)

  print(gx.shape, gy.shape, lap.shape, psi.shape, div.shape)


if __name__ == "__main__":
  main()
