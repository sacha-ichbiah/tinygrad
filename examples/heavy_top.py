"""
TinyPhysics 2.2: The Heavy Top

THE TINYPHYSICS WAY (Blueprint-Compliant):
    1. Define Hamiltonian H(L,γ) = ½ L·(I⁻¹L) + mgl γ₃
    2. Use HeavyTopStructure (Product Manifold SO(3)* × S²)
    3. Compile to StructureProgram
    4. Euler-Poisson equations derived automatically

The Heavy Top lives on a **Product Manifold**: SO(3)* × S²
- L: Angular momentum in body frame (on so(3)*)
- γ: Gravity direction in body frame (on S²)

The Lie-Poisson structure couples both:
  {Lᵢ, Lⱼ} = εᵢⱼₖ Lₖ   (angular momentum algebra)
  {γᵢ, Lⱼ} = εᵢⱼₖ γₖ   (γ transforms as a vector)
  {γᵢ, γⱼ} = 0          (γ components commute)

Conservation laws:
- Energy H is conserved (Hamiltonian)
- Casimir C1 = |γ|² = 1 (γ lives on sphere)
- Casimir C2 = L · γ (projection onto gravity)
"""

from tinygrad import Tensor, dtypes
from tinygrad.physics import ProductManifold
from tinyphysics.systems.heavy_top import HeavyTopSystem
import argparse
import time
from tinygrad.physics_profile import get_profile
import numpy as np
import json
import os

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def simulate_heavy_top(
    L0: list[float],
    theta0: float,
    phi0: float = 0.0,
    I1: float = 1.0,
    I2: float = 1.0,
    I3: float = 0.5,
    mgl: float = 1.0,
    dt: float = 0.001,
    steps: int = 10000,
    method: str = "splitting",
    batch_size: int = 1,
    unroll_steps: int = 8,
    scan: bool = True,
    auto_unroll: bool = True,
    viewer_batch: int = 4,
    benchmark: bool = False,
    render: bool = False,
    profile: str = "balanced",
    report_policy: bool = False,
) -> dict:
  """
  Simulate the Heavy Top using the TinyPhysics blueprint API.

  The system uses:
  - Product Manifold structure SO(3)* × S² with coupled Lie-Poisson bracket
  - Symplectic splitting integrator
  - Automatic conservation of energy and Casimirs

  Args:
    L0: Initial angular momentum [L1, L2, L3]
    theta0: Initial tilt angle from vertical (0 = upright)
    phi0: Initial azimuthal angle
    I1, I2, I3: Principal moments of inertia
    mgl: Gravitational torque parameter
    dt: Time step
    steps: Number of steps
    method: "splitting" (symplectic)

  Returns:
    Dictionary with trajectories and conservation diagnostics
  """
  # Initialize state
  L = Tensor(L0, dtype=dtypes.float64)
  if batch_size > 1:
    L = L.reshape(1, 3).expand(batch_size, 3).contiguous()
  state = ProductManifold.from_euler_angles(L, theta0, phi0)
  gamma = state.gamma

  # Setup policy
  if benchmark and profile == "balanced":
    profile = "fast"
  policy = get_profile(profile).policy

  # BLUEPRINT API: Create HeavyTopSystem
  system = HeavyTopSystem(
    I1=I1, I2=I2, I3=I3,
    mgl=mgl,
    dt=dt,
    dtype=dtypes.float64,
    policy=policy
  )

  # COMPILE: Returns a StructureProgram
  prog = system.compile()

  print("=" * 60)
  print("HEAVY TOP - TinyPhysics Blueprint API")
  print("=" * 60)
  print(f"\nStructure: Product Manifold SO(3)* × S²")
  print(f"  {{Lᵢ, Lⱼ}} = εᵢⱼₖ Lₖ")
  print(f"  {{γᵢ, Lⱼ}} = εᵢⱼₖ γₖ")
  print(f"\nHamiltonian: H = ½ L·(I⁻¹L) + mgl γ₃")
  print(f"Inertia: I1={I1}, I2={I2}, I3={I3}")
  print(f"Gravity: mgl={mgl}")

  # Storage
  history = {
    'L': [], 'gamma': [], 'energy': [],
    'C1': [], 'C2': [], 'time': []
  }
  collect_history = not benchmark

  # Initial values
  if not benchmark:
    L_sample = L[0] if L.ndim > 1 else L
    gamma_sample = gamma[0] if gamma.ndim > 1 else gamma
    E0 = prog.program.energy(L_sample, gamma_sample)
    C1_0, C2_0 = prog.program.casimirs(L_sample, gamma_sample)

  # Select integration method
  if method == "rk4":
    raise ValueError("rk4 is not symplectic; use method='splitting'")
  unroll = None if auto_unroll else unroll_steps
  if unroll is not None and steps % unroll != 0:
    raise ValueError("steps must be divisible by unroll")

  # EVOLVE using compiled program
  sample_interval = steps if benchmark else max(1, steps // 100)
  start_time = time.perf_counter() if benchmark else None
  if benchmark:
    scan = True

  (L, gamma), hist_t = prog.evolve((L, gamma), dt=dt, steps=steps,
                                    method=method, record_every=sample_interval,
                                    scan=scan, unroll=unroll, policy=policy)

  # Collect history for visualization
  if collect_history:
    for idx, (L_t, g_t) in enumerate(hist_t):
      L_sample = L_t[:viewer_batch] if L_t.ndim > 1 else L_t
      g_sample = g_t[:viewer_batch] if g_t.ndim > 1 else g_t
      if isinstance(L_sample, Tensor) and L_sample.ndim > 1:
        E = prog.program.energy(L_sample[0], g_sample[0])
        C1 = float((g_sample[0] * g_sample[0]).sum().numpy())
        C2 = float((L_sample[0] * g_sample[0]).sum().numpy())
      else:
        E = prog.program.energy(L_sample, g_sample)
        C1 = float((g_sample * g_sample).sum().numpy())
        C2 = float((L_sample * g_sample).sum().numpy())
      history['time'].append(idx * sample_interval * dt)
      history['L'].append(L_sample.numpy().copy())
      history['gamma'].append(g_sample.numpy().copy())
      history['energy'].append(E)
      history['C1'].append(C1)
      history['C2'].append(C2)

  if benchmark and start_time is not None:
    elapsed = time.perf_counter() - start_time
    steps_s = steps / elapsed if elapsed > 0 else float("inf")
    print(f"Performance: {steps_s:,.1f} steps/s")

  if report_policy:
    report = policy.report(steps, L.shape, L.device)
    if report is not None:
      print(f"Policy: {report}")

  # Final conservation check
  if not benchmark:
    L_sample = L[0] if L.ndim > 1 else L
    gamma_sample = gamma[0] if gamma.ndim > 1 else gamma
    E_final = prog.program.energy(L_sample, gamma_sample)
    C1_final, C2_final = prog.program.casimirs(L_sample, gamma_sample)

    history['diagnostics'] = {
      'E0': E0,
      'E_final': E_final,
      'dE_relative': abs(E_final - E0) / abs(E0) if E0 != 0 else abs(E_final - E0),
      'C1_0': C1_0,
      'C1_final': C1_final,
      'C2_0': C2_0,
      'C2_final': C2_final,
    }
    if render:
      generate_viewer(history['gamma'])
  return history


def generate_viewer(history_gamma):
  history_list = [g.tolist() if hasattr(g, "tolist") else g for g in history_gamma]
  html_content = f"""
<!DOCTYPE html>
<html>
<head>
  <title>Heavy Top Precession</title>
  <style>
    body {{ font-family: sans-serif; background: #000; color: #fff; display: flex; flex-direction: column; align-items: center; }}
    canvas {{ border: 1px solid #444; }}
  </style>
</head>
<body>
  <h1>Heavy Top Precession</h1>
  <p>Trace of γ projected on ground (x,y).</p>
  <canvas id="simCanvas" width="600" height="600"></canvas>
  <script>
    const history = {json.dumps(history_list)};
    const canvas = document.getElementById('simCanvas');
    const ctx = canvas.getContext('2d');
    const cx = canvas.width / 2;
    const cy = canvas.height / 2;
    const scale = 200;
    let frame = 0;

    function draw() {{
      ctx.fillStyle = 'rgba(0,0,0,0.2)';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      const g = history[frame];
      const show = Array.isArray(g[0]) ? g : [g];
      const grid = Math.ceil(Math.sqrt(show.length));
      show.forEach((v, idx) => {{
        const gx = idx % grid;
        const gy = Math.floor(idx / grid);
        const ox = cx + (gx - (grid - 1)/2) * 220;
        const oy = cy + (gy - (grid - 1)/2) * 220;
        const x = ox + v[0] * scale;
        const y = oy - v[1] * scale;
        ctx.fillStyle = '#0f0';
        ctx.beginPath();
        ctx.arc(x, y, 3, 0, 2 * Math.PI);
        ctx.fill();
      }});
      frame = (frame + 1) % history.length;
      requestAnimationFrame(draw);
    }}
    draw();
  </script>
</body>
</html>
  """
  viewer_path = 'examples/heavy_top_viewer.html'
  with open(viewer_path, 'w') as f:
    f.write(html_content)
  print(f"Viewer generated: {os.path.abspath(viewer_path)}")

# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=int(os.getenv("TINYGRAD_BATCH", "1")))
  parser.add_argument("--steps", type=int, default=800)
  parser.add_argument("--dt", type=float, default=0.002)
  parser.add_argument("--method", type=str, default="splitting")
  parser.add_argument("--unroll", type=int, default=8)
  parser.add_argument("--scan", action="store_true", default=True)
  parser.add_argument("--no-scan", action="store_false", dest="scan")
  parser.add_argument("--auto-unroll", action="store_true", default=True)
  parser.add_argument("--no-auto-unroll", action="store_false", dest="auto_unroll")
  parser.add_argument("--viewer-batch", type=int, default=4)
  parser.add_argument("--profile", type=str, default=os.getenv("TINYGRAD_PHYSICS_PROFILE", "balanced"))
  parser.add_argument("--benchmark", action="store_true")
  parser.add_argument("--render", action="store_true")
  parser.add_argument("--policy-report", action="store_true")
  args = parser.parse_args()

  print("=" * 60)
  print("TinyPhysics 2.2: The Heavy Top")
  print("=" * 60)
  print()

  # Physical setup: A symmetric top (I1 = I2)
  # Tilted 30 degrees from vertical, spinning around its axis
  print("Configuration:")
  print("  Moments of inertia: I1=I2=1.0, I3=0.5 (symmetric top)")
  print("  Gravitational parameter: mgl = 1.0")
  print("  Initial tilt: 30 degrees from vertical")
  print("  Initial spin: L3 = 5.0 (fast rotation around axis)")
  print()

  # Run simulation (100 steps with dt=0.01 = 1 time unit)
  results = simulate_heavy_top(
    L0=[0.1, 0.0, 5.0],  # Mostly spinning around z-axis
    theta0=np.pi/6,       # 30 degrees tilt
    I1=1.0, I2=1.0, I3=0.5,
    mgl=1.0,
    dt=args.dt,
    steps=args.steps,
    method=args.method,
    batch_size=args.batch,
    unroll_steps=args.unroll,
    scan=args.scan,
    auto_unroll=args.auto_unroll,
    viewer_batch=args.viewer_batch,
    benchmark=args.benchmark,
    render=args.render,
    profile=args.profile,
    report_policy=args.policy_report,
  )

  if not args.benchmark:
    diag = results['diagnostics']

    print("Conservation Analysis:")
    print("-" * 40)
    print(f"  Energy:")
    print(f"    Initial:  {diag['E0']:.10f}")
    print(f"    Final:    {diag['E_final']:.10f}")
    print(f"    Relative error: {diag['dE_relative']:.2e}")
    print()
    print(f"  Casimir C1 = |γ|² (should be 1.0):")
    print(f"    Initial:  {diag['C1_0']:.10f}")
    print(f"    Final:    {diag['C1_final']:.10f}")
    print()
    print(f"  Casimir C2 = L·γ (conserved):")
    print(f"    Initial:  {diag['C2_0']:.10f}")
    print(f"    Final:    {diag['C2_final']:.10f}")
    print()

    # Check success metric from roadmap
    if diag['dE_relative'] < 1e-6:
      print("SUCCESS: Energy conserved to < 10^-6 precision!")
    else:
      print(f"WARNING: Energy drift detected: {diag['dE_relative']:.2e}")

    print()
    print("=" * 60)
    print("The Product Manifold SO(3) × S² is working!")
  print("=" * 60)
