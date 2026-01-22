"""
Free Rigid Body Simulation - Tennis Racket Theorem / Dzhanibekov Effect

This demonstrates the tiny physics approach to rigid body dynamics using
Lie-Poisson mechanics. Instead of manually implementing Euler's equations,
we define only the Hamiltonian H(L) and let the system derive the dynamics.

The key insight: For rigid bodies, the Poisson bracket is not canonical.
Instead of dq/dt = +∂H/∂p, dp/dt = -∂H/∂q, we have:

    dL/dt = L × ∇H(L)    (Euler's equations)

This is the so(3) Lie-Poisson structure. The cross product encodes the
non-canonical geometry of rotational motion.

Conservation laws:
- Energy H = 0.5 * L · (I⁻¹ L) is conserved
- Casimir |L|² is conserved (for any Hamiltonian on so(3))
"""
import argparse
import time
import numpy as np
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import RigidBodySystem
import json
import os


def _auto_unroll(candidates: list[int], steps: int, make_state, step_factory) -> int:
    best = None
    best_t = None
    trial_steps = min(steps, 2000)
    for unroll in candidates:
        if steps % unroll != 0: continue
        L, q = make_state()
        step = step_factory(unroll)
        for _ in range(3):
            L, q = step(L, q)
        t0 = time.perf_counter()
        for _ in range(trial_steps // unroll):
            L, q = step(L, q)
        L.numpy(); q.numpy()
        t1 = time.perf_counter()
        if best_t is None or t1 - t0 < best_t:
            best_t = t1 - t0
            best = unroll
    return best if best is not None else candidates[0]


def run_simulation(batch_size: int = 1, steps: int = 2000, dt: float = 0.01,
                   unroll_steps: int = 8, scan: bool = False, auto_unroll: bool = False,
                   viewer_batch: int = 4, benchmark: bool = False):
    # Principal moments of inertia: I1 < I2 < I3
    # Intermediate axis (I2) is unstable - the Tennis Racket Theorem!
    I = Tensor([1.0, 2.0, 3.0])

    # Create the rigid body system using the tiny physics approach
    # The Hamiltonian H(L) = 0.5 * L · (I⁻¹ L) is defined internally
    system = RigidBodySystem(I, integrator="midpoint")

    dt = dt
    steps = steps
    unroll_steps = unroll_steps

    # Initial state: rotation near the intermediate axis (I2)
    # Small perturbations in L1 and L3 will cause tumbling
    L = Tensor([0.01, 2.0, 0.01])
    q = Tensor([1.0, 0.0, 0.0, 0.0])
    if batch_size > 1:
        L = L.reshape(1, 3).expand(batch_size, 3).contiguous()
        q = q.reshape(1, 4).expand(batch_size, 4).contiguous()

    print(f"Free Rigid Body Simulation (Tennis Racket Theorem)")
    print(f"Inertia: I = {I.numpy()}")
    print(f"Integrator: {system.integrator_name}")
    print(f"Initial L (body frame): {L.numpy()}")
    print()

    # Record initial conservation quantities
    L_sample = L[0] if L.ndim > 1 else L
    H_start = system.energy(L_sample)
    C_start = system.casimir(L_sample)

    # Run simulation
    record_every = 10
    if auto_unroll:
        candidates = [2, 4, 8, 16]
        def make_state():
            L0 = Tensor([0.01, 2.0, 0.01]).reshape(1, 3).expand(batch_size, 3).contiguous() if batch_size > 1 else Tensor([0.01, 2.0, 0.01])
            q0 = Tensor([1.0, 0.0, 0.0, 0.0]).reshape(1, 4).expand(batch_size, 4).contiguous() if batch_size > 1 else Tensor([1.0, 0.0, 0.0, 0.0])
            return L0, q0
        def step_factory(unroll):
            return system.compile_unrolled_step(dt, unroll)
        unroll_steps = _auto_unroll(candidates, steps, make_state, step_factory)
    if steps % unroll_steps != 0:
        raise ValueError("steps must be divisible by unroll_steps")

    start_time = time.perf_counter() if benchmark else None
    if scan:
        if record_every % unroll_steps != 0:
            record_every = unroll_steps
        L, q, history = system.evolve_unrolled(L, q, dt, steps, unroll_steps, record_every=record_every)
    else:
        def step_unrolled(L_in: Tensor, q_in: Tensor):
            for _ in range(unroll_steps):
                L_in, q_in = system.step(L_in, q_in, dt)
            return L_in, q_in
        step = TinyJit(step_unrolled)
        history = []
        for i in range(0, steps, unroll_steps):
            if i % record_every == 0:
                L_sample = L[0] if L.ndim > 1 else L
                q_sample = q[0] if q.ndim > 1 else q
                history.append((L_sample.numpy().copy(), q_sample.numpy().copy(), system.energy(L_sample), system.casimir(L_sample)))
            L, q = step(L, q)
        L_sample = L[0] if L.ndim > 1 else L
        q_sample = q[0] if q.ndim > 1 else q
        history.append((L_sample.numpy().copy(), q_sample.numpy().copy(), system.energy(L_sample), system.casimir(L_sample)))
    if benchmark and start_time is not None:
        elapsed = time.perf_counter() - start_time
        steps_s = steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance: {steps_s:,.1f} steps/s")

    # Check conservation
    H_end = system.energy(L_sample)
    C_end = system.casimir(L_sample)

    print(f"Energy Conservation:")
    print(f"  H_start: {H_start:.10f}")
    print(f"  H_end:   {H_end:.10f}")
    print(f"  Drift:   {abs(H_end - H_start)/abs(H_start):.2e}")
    print()
    print(f"Casimir Conservation (|L|²):")
    print(f"  C_start: {C_start:.10f}")
    print(f"  C_end:   {C_end:.10f}")
    print(f"  Drift:   {abs(C_end - C_start)/abs(C_start):.2e}")

    # Extract quaternions for visualization
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
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--auto-unroll", action="store_true")
    parser.add_argument("--viewer-batch", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
    args = parser.parse_args()
    run_simulation(
        batch_size=args.batch,
        steps=args.steps,
        dt=args.dt,
        unroll_steps=args.unroll,
        scan=args.scan,
        auto_unroll=args.auto_unroll,
        viewer_batch=args.viewer_batch,
        benchmark=args.benchmark,
    )
