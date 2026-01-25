import numpy as np
from tinygrad.physics import IdealFluidVorticity2D


def test_ideal_fluid_vorticity_fast():
  n = 16
  L = 2 * np.pi
  x = np.linspace(0, L, n, endpoint=False)
  y = np.linspace(0, L, n, endpoint=False)
  X, Y = np.meshgrid(x, y, indexing="ij")
  w0 = (np.sin(X) * np.cos(Y)).astype(np.float32)
  solver = IdealFluidVorticity2D(n, L=L, dealias=2.0/3.0, dtype=np.float32)
  w1, history = solver.evolve(w0, dt=0.05, steps=4, record_every=2, method="midpoint", iters=3)
  assert w1.shape == (n, n)
  assert len(history) == 3
  assert np.isfinite(w1).all()


def test_ideal_fluid_vorticity_diagnostics():
  n = 8
  L = 2 * np.pi
  x = np.linspace(0, L, n, endpoint=False)
  y = np.linspace(0, L, n, endpoint=False)
  X, Y = np.meshgrid(x, y, indexing="ij")
  w0 = (np.sin(X) * np.cos(Y)).astype(np.float32)
  solver = IdealFluidVorticity2D(n, L=L, dealias=2.0/3.0, dtype=np.float32)
  w1, history = solver.evolve(w0, dt=0.05, steps=2, record_every=1, method="midpoint", iters=2, diagnostics=True)
  assert w1.shape == (n, n)
  assert len(history) == 3
  w_hist, e_hist, z_hist = history[0]
  assert w_hist.shape == (n, n)
  assert np.isfinite(e_hist)
  assert np.isfinite(z_hist)
