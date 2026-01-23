from tinygrad.tensor import Tensor
from tinygrad.physics import PointVortexSystem


def _max_rel_drift(history):
  e0 = float(history[0][1])
  denom = max(abs(e0), 1e-12)
  return max(abs(float(h[1]) - e0) / denom for h in history)


def test_point_vortex_midpoint_stable():
  gamma = Tensor([1.0, -1.0, 1.0])
  z0 = Tensor([[0.5, 0.0],
               [-0.5, 0.0],
               [0.0, 0.8]])
  steps = 200
  dt = 0.01

  euler = PointVortexSystem(gamma, integrator="euler")
  midpoint = PointVortexSystem(gamma, integrator="midpoint")

  _, euler_hist = euler.evolve(z0, dt=dt, steps=steps, record_every=1)
  _, midpoint_hist = midpoint.evolve(z0, dt=dt, steps=steps, record_every=1)

  assert _max_rel_drift(midpoint_hist) < _max_rel_drift(euler_hist)
