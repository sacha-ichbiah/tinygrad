import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import cross
import json
import os

def run_simulation():
    # Satellite Parameters
    # Asymmetric Body
    I = Tensor([1.0, 2.0, 3.0])
    I_inv = 1.0 / I
    
    dt = 0.01
    steps = 1500
    
    # Initial State: Tumbling
    # High Angular Velocity
    omega_init = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    L_init = omega_init * I.numpy()
    L = Tensor(L_init, requires_grad=True)
    
    # Random initial orientation
    # Let's start rotated 90 deg around X
    theta = np.deg2rad(90)
    quat_init = np.array([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0], dtype=np.float32)
    quat = Tensor(quat_init, requires_grad=True)
    
    # Target Orientation: Identity [1, 0, 0, 0]
    target_q = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    
    # Controller Gains
    Kp = 20.0 # Proportional Gain (Orientation Error)
    Kd = 10.0 # Derivative Gain (Damping/Angular Velocity)
    
    print(f"Start Satellite Control Simulation")
    print(f"Target: Untumble and Lock to Identity Quaternion")
    
    history_q = []
    
    for i in range(steps):
        
        # 1. Measurement / Estimation
        q_curr = quat.numpy()
        omega_curr = (L * I_inv).numpy()
        
        # 2. Control Law (PD)
        # Error Quaternion q_e = q_target^* * q
        # Since q_target is Identity, q_error is just q_curr.
        # However, to avoid "double cover" issue (q and -q are same), 
        # we check the sign of w component.
        
        # Proper error calc: q_err = q_curr "minus" q_target
        # Local error vector ~ imaginary part of quaternion if target is identity.
        # But if q_w < 0, we flip sign to take shortest path.
        if q_curr[0] < 0:
            q_err = -q_curr
        else:
            q_err = q_curr
            
        # Error Vector (xyz part)
        e_vec = q_err[1:] 
        
        # Torque u = -Kp * error - Kd * omega
        # Note: Torque is applied to the BODY.
        # Euler Eq: dL/dt + w x L = u_ext
        # We need u_ext to counteract the motion.
        
        u = -Kp * e_vec - Kd * omega_curr
        u_tensor = Tensor(u)
        
        # 3. Dynamics Integration (RK4 with Control)
        L_np = L.numpy()
        
        def dynamics_control(L_val):
            omega = L_val * I_inv
            # dL/dt = L x omega + u
            
            dL = cross(L_val, omega) + u_tensor
            
            return dL

        # RK4
        def f(l_vec):
            return dynamics_control(Tensor(l_vec)).numpy()
            
        k1 = f(L_np)
        k2 = f(L_np + 0.5*dt*k1)
        k3 = f(L_np + 0.5*dt*k2)
        k4 = f(L_np + dt*k3)
        
        L_next = L_np + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        L = Tensor(L_next, requires_grad=True)
        
        # 4. Kinematics Update
        # dq/dt = 0.5 * q * omega
        s1, v1 = q_curr[0], q_curr[1:]
        s2, v2 = 0.0, omega_curr
        
        s_new = s1*s2 - np.dot(v1, v2)
        v_new = s1*v2 + s2*v1 + np.cross(v1, v2)
        dq = np.concatenate(([s_new], v_new)) * 0.5
        q_next = q_curr + dq * dt
        q_next /= np.linalg.norm(q_next)
        
        quat = Tensor(q_next, requires_grad=True)
        
        if i % 10 == 0:
            history_q.append(q_next.tolist())
            
        # Check Convergence
        if i == steps - 1:
            ohm_mag = np.linalg.norm(omega_curr)
            # Error from identity [1,0,0,0]
            # Since q ~ -q, error is 1 - |q.dot(target)|
            err_align = 1.0 - abs(np.dot(q_curr, target_q))
            
            print(f"End Sim.")
            print(f"Final Omega: {ohm_mag:.6f}")
            print(f"Alignment Error: {err_align:.6f}")
            
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
        
        function drawPoly(pts, q, fillStyle, strokeStyle) {{
            ctx.beginPath();
            let first = true;
            for(let v of pts) {{
                const r = rotateByQuat(v, q);
                const x = cx + r[0]*scale;
                const y = cy - r[1]*scale;
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
            
            const q = history[frame];
            
            // Draw Target Frame (Ghost) - Faint
            // Identity quaternion is [1,0,0,0] -> No rotation.
            // Just raw projection.
            // drawAxes([1,0,0,0], 0.2); 
            
            // Draw Satellite
            // We need proper depth sorting or just simple wireframe.
            // Simple painter's alg is hard without Z-sort.
            // Let's just draw wireframe panels and body.
            
            drawPoly(panel_left, q, 'rgba(0,0,255,0.5)', '#0af');
            drawPoly(panel_right, q, 'rgba(0,0,255,0.5)', '#0af');
            
            // Body edges manually
            const r_verts = body_verts.map(v => rotateByQuat(v, q));
            const proj = r_verts.map(v => [cx + v[0]*scale, cy - v[1]*scale]);
            
            ctx.strokeStyle = '#ddd';
            const edges = [[0,1],[1,2],[2,3],[3,0],[4,5],[5,6],[6,7],[7,4],[0,4],[1,5],[2,6],[3,7]];
            ctx.beginPath();
            for(let e of edges) {{
                ctx.moveTo(proj[e[0]][0], proj[e[0]][1]);
                ctx.lineTo(proj[e[1]][0], proj[e[1]][1]);
            }}
            ctx.stroke();
            
            // Draw Axis Arrows
            const axes = [[1,0,0], [0,1,0], [0,0,1]];
            const colors = ['#f00', '#0f0', '#00f'];
            for(let i=0; i<3; i++) {{
                const r = rotateByQuat(axes[i], q);
                ctx.strokeStyle = colors[i];
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + r[0]*scale, cy - r[1]*scale);
                ctx.stroke();
            }}

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
    run_simulation()
