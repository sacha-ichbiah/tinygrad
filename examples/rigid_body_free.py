"""
Free Rigid Body Simulation - Tennis Racket Theorem / Dzhanibekov Effect

THE TINYPHYSICS WAY (Blueprint-Compliant):
    1. Define Hamiltonian H(L) = 0.5 * L · (I⁻¹ L)
    2. Use SO3Structure (Lie-Poisson bracket: L × grad)
    3. Compile to StructureProgram
    4. Euler's equations derived automatically: dL/dt = L × ∇H(L)

The key insight: For rigid bodies, the Poisson bracket is not canonical.
The SO(3) Lie-Poisson structure encodes the geometry of rotational motion.

Conservation laws:
- Energy H = 0.5 * L · (I⁻¹ L) is conserved (Hamiltonian)
- Casimir |L|² is conserved (for any Hamiltonian on so(3)*)
"""
import argparse
import time
from tinygrad.physics_profile import get_profile
from tinygrad.tensor import Tensor
from tinyphysics.systems.free_rigid_body import FreeRigidBodySystem
import json
import os


def run_simulation(batch_size: int = 1, steps: int = 2000, dt: float = 0.01,
                   unroll_steps: int = 8, scan: bool = True, auto_unroll: bool = True,
                   viewer_batch: int = 4, benchmark: bool = False, profile: str = "balanced",
                   report_policy: bool = False):
    """
    Simulate free rigid body dynamics using the TinyPhysics blueprint API.

    The system uses:
    - SO(3) Lie-Poisson structure for angular momentum dynamics
    - Quaternion kinematics for orientation tracking
    - Structure-preserving integration (midpoint or splitting)
    """
    # Principal moments of inertia: I1 < I2 < I3
    # Intermediate axis (I2) is unstable - the Tennis Racket Theorem!
    I = Tensor([1.0, 2.0, 3.0])

    # Setup policy
    if benchmark and profile == "balanced":
        profile = "fast"
    policy = get_profile(profile).policy

    # BLUEPRINT API: Create FreeRigidBodySystem
    system = FreeRigidBodySystem(
        I=I,
        integrator="auto",
        policy=policy
    )

    # COMPILE: Returns a StructureProgram
    prog = system.compile()

    # Initial state: rotation near the intermediate axis (I2)
    # Small perturbations in L1 and L3 will cause tumbling
    L = Tensor([0.01, 2.0, 0.01])
    q = Tensor([1.0, 0.0, 0.0, 0.0])
    if batch_size > 1:
        L = L.reshape(1, 3).expand(batch_size, 3).contiguous()
        q = q.reshape(1, 4).expand(batch_size, 4).contiguous()

    print("=" * 60)
    print("FREE RIGID BODY - TinyPhysics Blueprint API")
    print("=" * 60)
    print(f"\nStructure: SO(3) Lie-Poisson")
    print(f"  bracket(L, grad) = L × grad")
    print(f"\nHamiltonian: H(L) = 0.5 * L · (I⁻¹ L)")
    print(f"Inertia: I = {I.numpy()}")
    print(f"Integrator: {prog.program.integrator_name}")
    print(f"Initial L (body frame): {L[0].numpy() if L.ndim > 1 else L.numpy()}")

    # Record initial conservation quantities
    L_sample = L[0] if L.ndim > 1 else L
    if not benchmark:
        H_start = prog.program.energy(L_sample)
        C_start = prog.program.casimir(L_sample)

    # EVOLVE using compiled program
    record_every = steps if benchmark else 10
    unroll = None if auto_unroll else unroll_steps
    if unroll is not None and steps % unroll != 0:
        raise ValueError("steps must be divisible by unroll")

    start_time = time.perf_counter() if benchmark else None
    if benchmark:
        scan = True

    (L, q), history = prog.evolve((L, q), dt=dt, steps=steps,
                                   record_every=record_every, scan=scan, unroll=unroll, policy=policy)

    if benchmark and start_time is not None:
        elapsed = time.perf_counter() - start_time
        steps_s = steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance: {steps_s:,.1f} steps/s")

    if report_policy:
        report = policy.report(steps, L.shape, L.device)
        if report is not None:
            print(f"Policy: {report}")

    if not benchmark:
        # Check conservation
        L_sample = L[0] if L.ndim > 1 else L
        H_end = prog.program.energy(L_sample)
        C_end = prog.program.casimir(L_sample)

        print(f"\nEnergy Conservation:")
        print(f"  H_start: {H_start:.10f}")
        print(f"  H_end:   {H_end:.10f}")
        print(f"  Drift:   {abs(H_end - H_start)/abs(H_start):.2e}")
        print()
        print(f"Casimir Conservation (|L|²):")
        print(f"  C_start: {C_start:.10f}")
        print(f"  C_end:   {C_end:.10f}")
        print(f"  Drift:   {abs(C_end - C_start)/abs(C_start):.2e}")

    # Extract quaternions for visualization
    if not benchmark:
        show_batch = min(batch_size, viewer_batch)
        history_q = []
        for _, q_hist, _, _ in history:
            if hasattr(q_hist, "ndim") and q_hist.ndim == 2:
                history_q.append(q_hist[:show_batch].tolist())
            else:
                history_q.append(q_hist.tolist())
        generate_viewer(history_q, I.numpy().tolist())


def generate_viewer(history_q, inertias):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Free Rigid Body (Dzhanibekov Effect)</title>
    <style>
        body {{ font-family: sans-serif; background: #000; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #444; }}
        .info {{ margin: 10px; text-align: center; }}
    </style>
</head>
<body>
    <h1>Rigid Body Tumbling</h1>
    <p>Yellow axis marks the unstable intermediate axis (I₂).</p>
    <p class="info">Using Lie-Poisson mechanics with implicit midpoint integrator.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history_q)};
        const I = {json.dumps(inertias)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 100;

        // Box geometry scaled by inertia (larger I = smaller dimension)
        const vertices = [
            [-1, -0.5, -0.2], [1, -0.5, -0.2], [1, 0.5, -0.2], [-1, 0.5, -0.2],
            [-1, -0.5, 0.2], [1, -0.5, 0.2], [1, 0.5, 0.2], [-1, 0.5, 0.2]
        ];

        let frame = 0;

        function rotateByQuat(v, q) {{
            const [w, x, y, z] = q;
            const [vx, vy, vz] = v;
            const x2 = x + x, y2 = y + y, z2 = z + z;
            const xx = x * x2, xy = x * y2, xz = x * z2;
            const yy = y * y2, yz = y * z2, zz = z * z2;
            const wx = w * x2, wy = w * y2, wz = w * z2;
            return [
                (1 - (yy + zz)) * vx + (xy - wz) * vy + (xz + wy) * vz,
                (xy + wz) * vx + (1 - (xx + zz)) * vy + (yz - wx) * vz,
                (xz - wy) * vx + (yz + wx) * vy + (1 - (xx + yy)) * vz
            ];
        }}

        function draw() {{
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            const qs = history[frame];
            const show = Array.isArray(qs[0]) ? qs : [qs];
            const grid = Math.ceil(Math.sqrt(show.length));
            show.forEach((q, idx) => {{
                const gx = idx % grid;
                const gy = Math.floor(idx / grid);
                const ox = cx + (gx - (grid - 1)/2) * 220;
                const oy = cy + (gy - (grid - 1)/2) * 220;
                const projected = vertices.map(v => {{
                    const rot = rotateByQuat(v, q);
                    return [ox + rot[0] * scale, oy + rot[1] * scale];
                }});

                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                const edges = [
                    [0,1], [1,2], [2,3], [3,0],
                    [4,5], [5,6], [6,7], [7,4],
                    [0,4], [1,5], [2,6], [3,7]
                ];
                ctx.beginPath();
                edges.forEach(e => {{
                    ctx.moveTo(projected[e[0]][0], projected[e[0]][1]);
                    ctx.lineTo(projected[e[1]][0], projected[e[1]][1]);
                }});
                ctx.stroke();

                const axes = [[1.5, 0, 0], [0, 1.2, 0], [0, 0, 0.8]];
                const colors = ['#f00', '#ff0', '#00f'];
                axes.forEach((axis, i) => {{
                    const rot = rotateByQuat(axis, q);
                    ctx.strokeStyle = colors[i];
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(ox, oy);
                    ctx.lineTo(ox + rot[0] * scale, oy + rot[1] * scale);
                    ctx.stroke();
                }});
            }});

            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}

        draw();
    </script>
</body>
</html>
    """

    viewer_path = 'examples/rigid_body_free_viewer.html'
    with open(viewer_path, 'w') as f:
        f.write(html_content)
    print(f"\nViewer generated: {os.path.abspath(viewer_path)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=int(os.getenv("TINYGRAD_BATCH", "1")))
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--unroll", type=int, default=8)
    parser.add_argument("--scan", action="store_true", default=True)
    parser.add_argument("--no-scan", action="store_false", dest="scan")
    parser.add_argument("--auto-unroll", action="store_true", default=True)
    parser.add_argument("--no-auto-unroll", action="store_false", dest="auto_unroll")
    parser.add_argument("--viewer-batch", type=int, default=4)
    parser.add_argument("--profile", type=str, default=os.getenv("TINYGRAD_PHYSICS_PROFILE", "balanced"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--policy-report", action="store_true")
    args = parser.parse_args()
    run_simulation(
        batch_size=args.batch,
        steps=args.steps,
        dt=args.dt,
        unroll_steps=args.unroll,
        scan=args.scan,
        auto_unroll=args.auto_unroll,
        viewer_batch=args.viewer_batch,
        profile=args.profile,
        benchmark=args.benchmark,
        report_policy=args.policy_report,
    )
