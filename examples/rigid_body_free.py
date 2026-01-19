import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import cross
import json
import os

def run_simulation():
    # Parameters for Tennis Racket Theorem
    # I1 < I2 < I3
    I1, I2, I3 = 1.0, 2.0, 3.0
    I = Tensor([I1, I2, I3])
    I_inv = 1.0 / I
    
    dt = 0.01
    steps = 2000
    
    # Initial State: Rotation near intermediate axis (I2)
    # L = (eps, 1.0, eps)
    L_init = np.array([0.01, 2.0, 0.01], dtype=np.float32)
    L = Tensor(L_init, requires_grad=True)
    
    # Orientation (Quaternion) q = [w, x, y, z]
    # Start identity
    quat_init = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    quat = Tensor(quat_init, requires_grad=True)
    
    print(f"Start Rigid Body Simulation: I={I.numpy()}")
    print(f"Initial L (Body Frame): {L.numpy()}")
    
    history_rot = [] # Store rotation matrices or quaternions
    
    for i in range(steps):
        # We use explicit RK4 for L_body to handle the structure
        
        # Dynamics: dL/dt = L x (I^-1 L)  (Euler's Equations in Body Frame)
        
        def dynamics_L(L_val):
            # L_val is (3,) Tensor
            omega = L_val * I_inv
            return cross(L_val, omega)
        
        # RK4 Step for L (Local implementation)
        def rk4_step(state, f, dt_val):
            k1 = f(state)
            k2 = f(state + k1 * (0.5 * dt_val))
            k3 = f(state + k2 * (0.5 * dt_val))
            k4 = f(state + k3 * dt_val)
            return state + (k1 + k2*2.0 + k3*2.0 + k4) * (dt_val / 6.0)
            
        L_next = rk4_step(L, dynamics_L, dt)
        
        # Break graph for next iteration
        L = Tensor(L_next.numpy(), requires_grad=True)
        
        # Orientation Update
        # dq/dt = 0.5 * q * omega (quaternion multiplication)
        # We perform simple update: q_new = q + dq/dt * dt, then normalize
        omega = (L * I_inv) # Body frame omega
        
        # Quaternion kinematics
        # q = [qw, qx, qy, qz]
        # w = [0, wx, wy, wz]
        # dq/dt = 0.5 * q * w
        # We'll do this numerically with numpy for simplicity as it is just visualization
        q_np = quat.numpy()
        w_np = omega.numpy()
        
        # Omega quaternion (0, wx, wy, wz)
        # Q_w = [0, w]
        # Q_new = Q_old + 0.5 * Q_old * Q_w * dt
        # Multiplication:
        # [s1, v1] * [s2, v2] = [s1s2 - v1.v2, s1v2 + s2v1 + v1xv2]
        
        s1, v1 = q_np[0], q_np[1:]
        s2, v2 = 0.0, w_np
        
        s_new = s1*s2 - np.dot(v1, v2)
        v_new = s1*v2 + s2*v1 + np.cross(v1, v2)
        
        dq = np.concatenate(([s_new], v_new)) * 0.5
        
        q_next_np = q_np + dq * dt
        # Normalize
        q_next_np /= np.linalg.norm(q_next_np)
        
        quat = Tensor(q_next_np, requires_grad=True)
        
        if i % 10 == 0:
            history_rot.append(q_next_np.tolist())
            
        # Check Conservation
        if i == 0:
            # Hamiltonian H = 0.5 * L . (I^-1 L)
            H_start = (0.5 * (L*L*I_inv).sum()).numpy()
            L2_start = (L*L).sum().numpy()
            
    # Final check
    H_end = (0.5 * (L*L*I_inv).sum()).numpy()
    L2_end = (L*L).sum().numpy()
    
    print(f"Energy Conservation:")
    print(f"H_start: {H_start:.6f}")
    print(f"H_end:   {H_end:.6f}")
    print(f"Drift:   {abs(H_end - H_start)/abs(H_start):.2e}")
    
    print(f"\nCasimir Conservation (|L|^2):")
    print(f"L2_start: {L2_start:.6f}")
    print(f"L2_end:   {L2_end:.6f}")
    print(f"Drift:    {abs(L2_end - L2_start)/abs(L2_start):.2e}")

    generate_viewer(history_rot, [I1, I2, I3])

def generate_viewer(history_rot, Inertias):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Free Rigid Body (Dzhanibekov Effect)</title>
    <style>
        body {{ font-family: sans-serif; background: #000; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #444; }}
    </style>
</head>
<body>
    <h1>Rigid Body Tumbling</h1>
    <p>Yellow ends marks the unstable axis.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history_rot)};
        const I = {json.dumps(Inertias)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 100; 
        
        // Define a Box Geometry
        const vertices = [
            [-1, -0.5, -0.2], [1, -0.5, -0.2], [1, 0.5, -0.2], [-1, 0.5, -0.2], // Back face
            [-1, -0.5, 0.2], [1, -0.5, 0.2], [1, 0.5, 0.2], [-1, 0.5, 0.2]   // Front face
        ];
        
        // Axis coloring: X=Red, Y=Green, Z=Blue
        // I1 < I2 < I3
        // We set L near I2 (Y-axis). Y-axis is the intermediate axis (Unstable).
        // The box dimensions roughly correspond to I. (Small dim -> Large I, Large dim -> Small I)
        // I ~ m r^2.
        
        let frame = 0;
        
        function rotateByQuat(v, q) {{
            // Rotate vector v by quaternion q
            // v' = q v q*
            const [w, x, y, z] = q;
            const [vx, vy, vz] = v;
            
            // This is efficient formula ... or just convert Q to Matrix
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
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            const q = history[frame];
            
            // Project and Draw Lines
            ctx.beginPath();
            const projected = vertices.map(v => {{
                 const rot = rotateByQuat(v, q);
                 // Weak perspective
                 const z = rot[2] + 4; 
                 const fov = 300;
                 return [cx + rot[0] * scale, cy + rot[1] * scale];
            }});
            
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            
            // Connections
            const edges = [
                [0,1], [1,2], [2,3], [3,0], // Back
                [4,5], [5,6], [6,7], [7,4], // Front
                [0,4], [1,5], [2,6], [3,7]  // Sides
            ];
            
            edges.forEach(e => {{
                ctx.moveTo(projected[e[0]][0], projected[e[0]][1]);
                ctx.lineTo(projected[e[1]][0], projected[e[1]][1]);
            }});
            ctx.stroke();
            
            // Draw Axes attached to body
            const axes = [[1.5,0,0], [0,1.2,0], [0,0,0.8]];
            const colors = ['#f00', '#ff0', '#00f']; // Red, Yellow (Unstable), Blue
            
            axes.forEach((axis, i) => {{
                const rot = rotateByQuat(axis, q);
                ctx.strokeStyle = colors[i];
                ctx.lineWidth = 3;
                ctx.beginPath();
                ctx.moveTo(cx, cy);
                ctx.lineTo(cx + rot[0]*scale, cy + rot[1]*scale);
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
    
    with open('examples/rigid_body_free_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/rigid_body_free_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
