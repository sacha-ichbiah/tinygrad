"""
Satellite Attitude Control - Controlled Rigid Body on SO(3)

THE TINYPHYSICS WAY (Blueprint-Compliant):
    1. Free dynamics follow Lie-Poisson structure: dL/dt = L × ω
    2. Control applied via symplectic splitting (kick-drift-kick)
    3. ControlledRigidBodySystem wraps both L and quaternion evolution
    4. Structure-preserving integration maintains geometric properties

This demonstrates controlled systems within the blueprint framework.
"""

import argparse
import time
from tinygrad.physics_profile import get_profile
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import ControlInput
from tinyphysics.systems.satellite import ControlledRigidBodySystem
from tinygrad.helpers import getenv
import json
import os

def run_simulation(steps: int = 1500, dt: float = 0.01, unroll_steps: int = 10,
                   batch_size: int = 1, auto_unroll: bool = True, scan: bool = True,
                   viewer_batch: int = 4, benchmark: bool = False, profile: str = "balanced",
                   report_policy: bool = False):
    """
    Simulate satellite attitude control using the TinyPhysics blueprint API.

    The system uses:
    - Lie-Poisson structure for free rigid body dynamics (SO(3))
    - PD control law for attitude stabilization
    - Symplectic splitting to preserve geometric structure
    """
    # Satellite Parameters - Asymmetric Body
    I = Tensor([1.0, 2.0, 3.0])
    I_inv = 1.0 / I

    # Initial State: Tumbling with high angular velocity
    omega_init = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    L_init = omega_init * I.numpy()
    L = Tensor(L_init, requires_grad=False)

    # Initial orientation: rotated 90 deg around X
    theta = np.deg2rad(90)
    quat_init = np.array([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0], dtype=np.float32)
    quat = Tensor(quat_init, requires_grad=False)

    if batch_size > 1:
        L = L.reshape(1, 3).expand(batch_size, 3).contiguous()
        quat = quat.reshape(1, 4).expand(batch_size, 4).contiguous()

    # Target Orientation: Identity [1, 0, 0, 0]
    target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Controller Gains
    Kp = 20.0  # Proportional Gain (Orientation Error)
    Kd = 10.0  # Derivative Gain (Damping/Angular Velocity)

    print("=" * 60)
    print("SATELLITE ATTITUDE CONTROL - TinyPhysics Blueprint API")
    print("=" * 60)
    print(f"\nFree dynamics: Lie-Poisson on SO(3)")
    print(f"  dL/dt = L × ω  where ω = I⁻¹L")
    print(f"\nControl: PD law via symplectic splitting")
    print(f"  τ = -Kp * q_err - Kd * ω")
    print(f"  Kp={Kp}, Kd={Kd}")
    print(f"\nTarget: Untumble and lock to identity quaternion")
    print(f"Simulation: {steps} steps, dt={dt}")

    # Setup policy
    if benchmark and profile == "balanced":
        profile = "fast"
    policy = get_profile(profile).policy

    # BLUEPRINT API: Create control input
    control = ControlInput(lambda q_err, omega: -Kp * q_err[..., 1:] - Kd * omega)

    # BLUEPRINT API: Create ControlledRigidBodySystem
    system = ControlledRigidBodySystem(
        I_inv=I_inv,
        control=control,
        dt=dt,
        policy=policy
    )

    # COMPILE: Returns a StructureProgram
    prog = system.compile()
    print(f"\nCompiled: {type(prog.program).__name__}")

    # EVOLVE with explicit history recording for viewer
    history_q = [] if not benchmark else None
    unroll = None if auto_unroll else unroll_steps
    if unroll is not None and steps % unroll != 0:
        raise ValueError("steps must be divisible by unroll")

    start_time = time.perf_counter() if benchmark else None
    if benchmark:
        scan = True

    record_every = steps if benchmark else 10

    # Use the compiled program's evolve
    (L, quat), hist = prog.evolve((L, quat), dt=dt, steps=steps,
                                   record_every=record_every, scan=scan, unroll=unroll)

    # Extract history for viewer
    if history_q is not None:
        for L_t, q_t in hist:
            if q_t.ndim > 1:
                show = min(batch_size, viewer_batch)
                history_q.append(q_t[:show].numpy().tolist())
            else:
                history_q.append(q_t.numpy().tolist())

    # Final state analysis
    L_sample = L[0] if L.ndim > 1 else L
    q_sample = quat[0] if quat.ndim > 1 else quat
    omega_curr = (L_sample * I_inv).numpy()
    q_curr = q_sample.numpy()
    ohm_mag = np.linalg.norm(omega_curr)
    err_align = 1.0 - abs(np.dot(q_curr, target_q))

    print(f"\nFinal Angular Velocity Magnitude: {ohm_mag:.6f}")
    print(f"Alignment Error: {err_align:.6f}")

    if benchmark and start_time is not None:
        elapsed = time.perf_counter() - start_time
        steps_s = steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance: {steps_s:,.1f} steps/s")

    if report_policy:
        report = policy.report(steps, L.shape, L.device)
        if report is not None:
            print(f"Policy: {report}")

    if getenv("TINYGRAD_PHYSICS_REPORT", 0):
        report = {
            "final_omega": float(ohm_mag),
            "alignment_error": float(err_align),
            "steps": steps,
            "dt": dt,
        }
        print(f"Control report: {report}")

    if not benchmark and history_q is not None:
        generate_viewer(history_q)

def generate_viewer(history):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Satellite Attitude Control</title>
    <style>
        body {{ font-family: sans-serif; background: #000; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #444; }}
    </style>
</head>
<body>
    <h1>Satellite Control (PD)</h1>
    <p>Target: Stationary aligned with axes.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 150; 
        
        // Satellite Geometry (Box + Solar Panels)
        const body_verts = [
            [-0.5,-0.5,0.5], [0.5,-0.5,0.5], [0.5,0.5,0.5], [-0.5,0.5,0.5], // Front
            [-0.5,-0.5,-0.5], [0.5,-0.5,-0.5], [0.5,0.5,-0.5], [-0.5,0.5,-0.5] // Back
        ];
        
        const panel_left = [ [-1.5, -0.2, 0], [-0.5, -0.2, 0], [-0.5, 0.2, 0], [-1.5, 0.2, 0] ];
        const panel_right = [ [0.5, -0.2, 0], [1.5, -0.2, 0], [1.5, 0.2, 0], [0.5, 0.2, 0] ];
        
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
        
        function drawPoly(pts, q, fillStyle, strokeStyle, ox, oy) {{
            ctx.beginPath();
            let first = true;
            for(let v of pts) {{
                const r = rotateByQuat(v, q);
                const x = ox + r[0]*scale;
                const y = oy - r[1]*scale;
                if(first) {{ ctx.moveTo(x,y); first=false; }}
                else ctx.lineTo(x,y);
            }}
            ctx.closePath();
            ctx.fillStyle = fillStyle;
            ctx.fill();
            ctx.strokeStyle = strokeStyle;
            ctx.stroke();
        }}
        
        function draw() {{
            ctx.fillStyle = 'rgba(0,0,0,0.3)';
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            const qs = history[frame];
            const show = Array.isArray(qs[0]) ? qs : [qs];
            const grid = Math.ceil(Math.sqrt(show.length));
            
            // Draw Target Frame (Ghost) - Faint
            // Identity quaternion is [1,0,0,0] -> No rotation.
            // Just raw projection.
            // drawAxes([1,0,0,0], 0.2); 
            
            // Draw Satellite
            // We need proper depth sorting or just simple wireframe.
            // Simple painter's alg is hard without Z-sort.
            // Let's just draw wireframe panels and body.
            
            show.forEach((q, idx) => {{
                const gx = idx % grid;
                const gy = Math.floor(idx / grid);
                const ox = cx + (gx - (grid - 1)/2) * 240;
                const oy = cy + (gy - (grid - 1)/2) * 240;

                drawPoly(panel_left, q, 'rgba(0,0,255,0.5)', '#0af', ox, oy);
                drawPoly(panel_right, q, 'rgba(0,0,255,0.5)', '#0af', ox, oy);
                
                const r_verts = body_verts.map(v => rotateByQuat(v, q));
                const proj = r_verts.map(v => [ox + v[0]*scale, oy - v[1]*scale]);
                
                ctx.strokeStyle = '#ddd';
                const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
                ctx.beginPath();
                for(let e of edges) {{
                    ctx.moveTo(proj[e[0]][0], proj[e[0]][1]);
                    ctx.lineTo(proj[e[1]][0], proj[e[1]][1]);
                }}
                ctx.stroke();
                
                const axes = [[1,0,0], [0,1,0], [0,0,1]];
                const colors = ['#f00', '#0f0', '#00f'];
                for(let i=0; i<3; i++) {{
                    const r = rotateByQuat(axes[i], q);
                    ctx.strokeStyle = colors[i];
                    ctx.lineWidth = 3;
                    ctx.beginPath();
                    ctx.moveTo(ox, oy);
                    ctx.lineTo(ox + r[0]*scale, oy - r[1]*scale);
                    ctx.stroke();
                }}
            }});

            frame = (frame + 1);
            if(frame >= history.length) frame = history.length - 1; // Hold at end
            requestAnimationFrame(draw);
        }}
        draw();
    </script>
</body>
</html>
    """
    with open('examples/satellite_control_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/satellite_control_viewer.html')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=int(os.getenv("TINYGRAD_BATCH", "1")))
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--unroll", type=int, default=10)
    parser.add_argument("--auto-unroll", action="store_true", default=True)
    parser.add_argument("--no-auto-unroll", action="store_false", dest="auto_unroll")
    parser.add_argument("--scan", action="store_true", default=True)
    parser.add_argument("--no-scan", action="store_false", dest="scan")
    parser.add_argument("--viewer-batch", type=int, default=4)
    parser.add_argument("--profile", type=str, default=os.getenv("TINYGRAD_PHYSICS_PROFILE", "balanced"))
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--policy-report", action="store_true")
    args = parser.parse_args()
    run_simulation(
        steps=args.steps,
        dt=args.dt,
        unroll_steps=args.unroll,
        batch_size=args.batch,
        auto_unroll=args.auto_unroll,
        scan=args.scan,
        viewer_batch=args.viewer_batch,
        benchmark=args.benchmark,
        profile=args.profile,
        report_policy=args.policy_report,
    )
