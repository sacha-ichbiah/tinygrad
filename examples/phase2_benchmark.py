import argparse
import time
import numpy as np
import os

from tinygrad.tensor import Tensor
from tinygrad.physics import RigidBodySystem, HeavyTopHamiltonian, ProductManifold, ControlInput, SatelliteControlIntegrator
from tinyphysics.core.compiler import compile_structure
from tinygrad.physics_profile import get_profile


def bench_rigid_body(batch: int, steps: int, dt: float, unroll: int, scan: bool, auto_unroll: bool, profile: str) -> tuple[float, float, dict | None]:
  I = Tensor([1.0, 2.0, 3.0])
  policy = get_profile(profile).policy
  system = RigidBodySystem(I, integrator="midpoint", policy=policy)
  L0_base = Tensor([0.01, 2.0, 0.01])
  L = L0_base.reshape(1, 3).expand(batch, 3).contiguous()
  q = Tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4).expand(batch, 4).contiguous()

  def make_state():
    L0 = L0_base.reshape(1, 3).expand(batch, 3).contiguous()
    q0 = Tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4).expand(batch, 4).contiguous()
    return L0, q0

  if auto_unroll:
    unroll = policy.choose_unroll(steps, L.shape, L.device, candidates=[2, 4, 8, 16], step_factory=lambda u: (lambda: system.compile_unrolled_step(dt, u)(L, q)))

  step = system.compile_unrolled_step(dt, unroll)
  for _ in range(5):
    L, q = step(L, q)

  start = time.perf_counter()
  if scan:
    L, q, _ = system.evolve_unrolled(L, q, dt, steps, unroll, record_every=steps)
  else:
    for _ in range(steps // unroll):
      L, q = step(L, q)
    L.numpy(); q.numpy()
  elapsed = time.perf_counter() - start
  steps_s = steps / elapsed

  L_sample = L[0]
  E0 = system.energy(L0_base)
  E1 = system.energy(L_sample)
  drift = abs(E1 - E0) / abs(E0)
  report = policy.report(steps, L.shape, L.device)
  return steps_s, drift, report


def bench_heavy_top(batch: int, steps: int, dt: float, unroll: int, scan: bool, auto_unroll: bool, profile: str) -> tuple[float, float, dict | None]:
  I1 = 1.0
  I2 = 1.0
  I3 = 0.5
  mgl = 1.0
  L0 = Tensor([0.1, 0.0, 5.0])
  L = L0.reshape(1, 3).expand(batch, 3).contiguous()
  gamma = ProductManifold.from_euler_angles(L, np.pi/6, 0.0).gamma
  policy = get_profile(profile).policy
  H = HeavyTopHamiltonian(I1, I2, I3, mgl, dtype=L.dtype)
  integrator = compile_structure(state=(L, gamma), H=H, kind="e3", policy=policy)

  def make_state():
    L0_state = L0.reshape(1, 3).expand(batch, 3).contiguous()
    g0 = ProductManifold.from_euler_angles(L0_state, np.pi/6, 0.0).gamma
    return L0_state, g0

  if auto_unroll:
    unroll = policy.choose_unroll(
      steps, L.shape, L.device, candidates=[2, 4, 8, 16],
      step_factory=lambda u: (lambda: integrator.compile_unrolled_step(dt, u)((L, gamma))),
    )

  step = integrator.compile_unrolled_step(dt, unroll)
  for _ in range(5):
    L, gamma = step((L, gamma))

  start = time.perf_counter()
  if scan:
    (L, gamma), _ = integrator.evolve((L, gamma), dt, steps, record_every=steps, unroll=unroll)
  else:
    for _ in range(steps // unroll):
      L, gamma = step((L, gamma))
    L.numpy(); gamma.numpy()
  elapsed = time.perf_counter() - start
  steps_s = steps / elapsed

  L_sample = L[0]
  gamma_sample = gamma[0]
  gamma0 = ProductManifold.from_euler_angles(L0, np.pi/6, 0.0).gamma
  E0 = H(ProductManifold(L0, gamma0)).numpy()
  E1 = H(ProductManifold(L_sample, gamma_sample)).numpy()
  drift = abs(E1 - E0) / abs(E0)
  report = policy.report(steps, L.shape, L.device)
  return steps_s, drift, report


def bench_satellite(batch: int, steps: int, dt: float, unroll: int, scan: bool, auto_unroll: bool, profile: str) -> tuple[float, float, dict | None]:
  I = Tensor([1.0, 2.0, 3.0])
  I_inv = 1.0 / I
  omega_init = np.array([1.0, -2.0, 0.5], dtype=np.float32)
  L = Tensor(omega_init * I.numpy()).reshape(1, 3).expand(batch, 3).contiguous()
  theta = np.deg2rad(90)
  quat = Tensor([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0]).reshape(1, 4).expand(batch, 4).contiguous()
  Kp = 20.0
  Kd = 10.0
  policy = get_profile(profile).policy
  control = ControlInput(lambda q_err, omega: -Kp * q_err[..., 1:] - Kd * omega)
  integrator = SatelliteControlIntegrator(I_inv, control, dt, policy=policy)

  def make_state():
    L0 = Tensor(omega_init * I.numpy()).reshape(1, 3).expand(batch, 3).contiguous()
    q0 = Tensor([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0]).reshape(1, 4).expand(batch, 4).contiguous()
    return L0, q0

  if auto_unroll:
    unroll = policy.choose_unroll(steps, L.shape, L.device, candidates=[5, 10, 20], step_factory=lambda u: (lambda: integrator.compile_unrolled_step(u)(L, quat)))

  step = integrator.compile_unrolled_step(unroll)
  for _ in range(5):
    L, quat = step(L, quat)

  start = time.perf_counter()
  if scan:
    L, quat, _ = integrator.evolve_unrolled(L, quat, steps, unroll, record_every=steps)
  else:
    for _ in range(steps // unroll):
      L, quat = step(L, quat)
    L.numpy(); quat.numpy()
  elapsed = time.perf_counter() - start
  steps_s = steps / elapsed

  omega = (L * I_inv).numpy()
  omega_mag = float(np.linalg.norm(omega[0]))
  report = policy.report(steps, L.shape, L.device)
  return steps_s, omega_mag, report


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=256)
  parser.add_argument("--steps", type=int, default=20000)
  parser.add_argument("--dt", type=float, default=0.01)
  parser.add_argument("--unroll", type=int, default=8)
  parser.add_argument("--scan", action="store_true", default=True)
  parser.add_argument("--no-scan", action="store_false", dest="scan")
  parser.add_argument("--auto-unroll", action="store_true", default=True)
  parser.add_argument("--no-auto-unroll", action="store_false", dest="auto_unroll")
  parser.add_argument("--profile", type=str, default=os.getenv("TINYGRAD_PHYSICS_PROFILE", "balanced"))
  args = parser.parse_args()
  if args.steps % args.unroll != 0:
    raise ValueError("steps must be divisible by unroll")

  steps_s, drift, report = bench_rigid_body(args.batch, args.steps, args.dt, args.unroll, args.scan, args.auto_unroll, args.profile)
  print(f"Rigid body: {steps_s:,.1f} steps/s (energy drift ~{drift:.2e})")
  if report is not None:
    print(f"  Policy: {report}")

  ht_dt = min(args.dt, 0.002)
  steps_s, drift, report = bench_heavy_top(args.batch, args.steps, ht_dt, args.unroll, args.scan, args.auto_unroll, args.profile)
  print(f"Heavy top: {steps_s:,.1f} steps/s (energy drift ~{drift:.2e})")
  if report is not None:
    print(f"  Policy: {report}")

  sat_unroll = max(args.unroll, 10)
  steps_s, omega_mag, report = bench_satellite(args.batch, args.steps, args.dt, sat_unroll, args.scan, args.auto_unroll, args.profile)
  print(f"Satellite: {steps_s:,.1f} steps/s (final |ω|={omega_mag:.3f})")
  if report is not None:
    print(f"  Policy: {report}")


if __name__ == "__main__":
  main()
