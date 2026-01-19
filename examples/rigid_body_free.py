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
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import RigidBodySystem
import json
import os


def run_simulation():
    # Principal moments of inertia: I1 < I2 < I3
    # Intermediate axis (I2) is unstable - the Tennis Racket Theorem!
    I = Tensor([1.0, 2.0, 3.0])

    # Create the rigid body system using the tiny physics approach
    # The Hamiltonian H(L) = 0.5 * L · (I⁻¹ L) is defined internally
    system = RigidBodySystem(I, integrator="midpoint")

    dt = 0.01
    steps = 2000

    # Initial state: rotation near the intermediate axis (I2)
    # Small perturbations in L1 and L3 will cause tumbling
    L = Tensor([0.01, 2.0, 0.01])

    # Initial orientation: identity quaternion [w, x, y, z]
    q = Tensor([1.0, 0.0, 0.0, 0.0])

    print(f"Free Rigid Body Simulation (Tennis Racket Theorem)")
    print(f"Inertia: I = {I.numpy()}")
    print(f"Integrator: {system.integrator_name}")
    print(f"Initial L (body frame): {L.numpy()}")
    print()

    # Record initial conservation quantities
    H_start = system.energy(L)
    C_start = system.casimir(L)

    # Run simulation
    L, q, history = system.evolve(L, q, dt, steps, record_every=10)

    # Check conservation
    H_end = system.energy(L)
    C_end = system.casimir(L)

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
    history_q = [h[1].tolist() for h in history]
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

            const q = history[frame];
            const projected = vertices.map(v => {{
                const rot = rotateByQuat(v, q);
                return [cx + rot[0] * scale, cy + rot[1] * scale];
            }});

            // Draw box edges
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

            // Draw body axes: Red (stable), Yellow (unstable), Blue (stable)
            const axes = [[1.5, 0, 0], [0, 1.2, 0], [0, 0, 0.8]];
            const colors = ['#f00', '#ff0', '#00f'];
            axes.forEach((axis, i) => {{
                const rot = rotateByQuat(axis, q);
                ctx.strokeStyle = colors[i];
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + rot[0] * scale, cy + rot[1] * scale);
                ctx.stroke();
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
    run_simulation()
