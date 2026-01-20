import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


def test_leapfrog_more_stable_than_euler():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q0, p0 = Tensor([1.0, 0.5, -0.25]), Tensor([0.0, 0.0, 0.0])
  steps = 200
  dt = 0.05

  def max_rel_drift(history):
    e0 = float(history[0][2])
    denom = max(abs(e0), 1e-12)
    return max(abs(float(e) - e0) / denom for (_, _, e) in history)

  euler = HamiltonianSystem(H, integrator="euler")
  leapfrog = HamiltonianSystem(H, integrator="leapfrog")

  _, _, euler_hist = euler.evolve(q0, p0, dt=dt, steps=steps, record_every=1)
  _, _, leapfrog_hist = leapfrog.evolve(q0, p0, dt=dt, steps=steps, record_every=1)

  euler_drift = max_rel_drift(euler_hist)
  leapfrog_drift = max_rel_drift(leapfrog_hist)

  assert leapfrog_drift < euler_drift
