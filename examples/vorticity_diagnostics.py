import numpy as np

from tinyphysics.systems.vorticity import create_vorticity_system, taylor_green_ic


def main():
  n = 64
  L = 2 * np.pi
  w0 = taylor_green_ic(n, L=L)
  solver = create_vorticity_system(N=n, L=L)
  w1, history = solver.evolve(w0, dt=0.02, steps=20, record_every=5, method="midpoint", iters=3, diagnostics=True)
  e0, z0 = history[0][1], history[0][2]
  e1, z1 = history[-1][1], history[-1][2]
  print(f"Vorticity diagnostics: E0={e0:.6f} Z0={z0:.6f} -> E1={e1:.6f} Z1={z1:.6f}")
  print(f"final shape={w1.shape}")


if __name__ == "__main__":
  main()
