import argparse
import numpy as np
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import ControlInput, SatelliteControlIntegrator
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


def run_simulation(steps: int = 1500, dt: float = 0.01, unroll_steps: int = 10,
                   batch_size: int = 1, auto_unroll: bool = False, scan: bool = False,
                   viewer_batch: int = 4, benchmark: bool = False):
    # Satellite Parameters
    # Asymmetric Body
    I = Tensor([1.0, 2.0, 3.0])
    I_inv = 1.0 / I
    
    dt = dt
    steps = steps
    unroll_steps = unroll_steps
    
    # Initial State: Tumbling
    # High Angular Velocity
    omega_init = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    L_init = omega_init * I.numpy()
    L = Tensor(L_init, requires_grad=False)
    
    # Random initial orientation
    # Let's start rotated 90 deg around X
    theta = np.deg2rad(90)
    quat_init = np.array([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0], dtype=np.float32)
    quat = Tensor(quat_init, requires_grad=False)
    if batch_size > 1:
        L = L.reshape(1, 3).expand(batch_size, 3).contiguous()
        quat = quat.reshape(1, 4).expand(batch_size, 4).contiguous()
    
    # Target Orientation: Identity [1, 0, 0, 0]
    target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Controller Gains
    Kp = 20.0 # Proportional Gain (Orientation Error)
    Kd = 10.0 # Derivative Gain (Damping/Angular Velocity)
    
    print(f"Start Satellite Control Simulation")
    print(f"Target: Untumble and Lock to Identity Quaternion")
    
    history_q = []
    control = ControlInput(lambda q_err, omega: -Kp * q_err[..., 1:] - Kd * omega)
    integrator = SatelliteControlIntegrator(I_inv, control, dt)

    def step(L_val: Tensor, quat_val: Tensor):
        return integrator.step(L_val, quat_val)

    if auto_unroll:
        candidates = [5, 10, 20]
        def make_state():
            L0 = Tensor(L_init, requires_grad=False)
            q0 = Tensor(quat_init, requires_grad=False)
            if batch_size > 1:
                L0 = L0.reshape(1, 3).expand(batch_size, 3).contiguous()
                q0 = q0.reshape(1, 4).expand(batch_size, 4).contiguous()
            return L0, q0
        def step_factory(unroll):
            return integrator.compile_unrolled_step(unroll)
        unroll_steps = _auto_unroll(candidates, steps, make_state, step_factory)
    if steps % unroll_steps != 0:
        raise ValueError("steps must be divisible by unroll_steps")
    start_time = time.perf_counter() if benchmark else None
    if scan:
        record_every = 10
        if record_every % unroll_steps != 0:
            record_every = unroll_steps
        L, quat, hist = integrator.evolve_unrolled(L, quat, steps, unroll_steps, record_every=record_every)
        for L_t, q_t in hist:
            if q_t.ndim > 1:
                show = min(batch_size, viewer_batch)
                history_q.append(q_t[:show].numpy().tolist())
            else:
                history_q.append(q_t.numpy().tolist())
        L_sample = L[0] if L.ndim > 1 else L
        q_sample = quat[0] if quat.ndim > 1 else quat
        omega_curr = (L_sample * I_inv).numpy()
        q_curr = q_sample.numpy()
        ohm_mag = np.linalg.norm(omega_curr)
        err_align = 1.0 - abs(np.dot(q_curr, target_q))
        print(f"End Sim.")
        print(f"Final Omega: {ohm_mag:.6f}")
        print(f"Alignment Error: {err_align:.6f}")
    else:
        step_jit = integrator.compile_unrolled_step(unroll_steps)
        for i in range(0, steps, unroll_steps):
            L, quat = step_jit(L, quat)

            if i % 10 == 0:
                if quat.ndim > 1:
                    show = min(batch_size, viewer_batch)
                    history_q.append(quat[:show].numpy().tolist())
                else:
                    history_q.append(quat.numpy().tolist())

            # Check Convergence
            if i + unroll_steps >= steps:
                L_sample = L[0] if L.ndim > 1 else L
                q_sample = quat[0] if quat.ndim > 1 else quat
                omega_curr = (L_sample * I_inv).numpy()
                q_curr = q_sample.numpy()
                ohm_mag = np.linalg.norm(omega_curr)
                err_align = 1.0 - abs(np.dot(q_curr, target_q))

                print(f"End Sim.")
                print(f"Final Omega: {ohm_mag:.6f}")
                print(f"Alignment Error: {err_align:.6f}")

    if benchmark and start_time is not None:
        elapsed = time.perf_counter() - start_time
        steps_s = steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance: {steps_s:,.1f} steps/s")

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
    parser.add_argument("--auto-unroll", action="store_true")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--viewer-batch", type=int, default=4)
    parser.add_argument("--benchmark", action="store_true")
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
    )
