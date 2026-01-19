import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import cross
import json
import os

def run_simulation():
    # Parameters for Heavy Symmetric Top
    # Symmetric: I1 = I2 != I3
    I1 = 1.0
    I2 = 1.0
    I3 = 3.0
    I = Tensor([I1, I2, I3])
    I_inv = 1.0 / I
    
    m = 1.0
    g = 9.81
    l = 1.0 # Distance to Center of Mass along z-axis
    
    dt = 0.005 # Smaller timestep for this fast dynamics
    steps = 4000
    
    # Initial Conditions (Body Frame)
    # Fast spin around axes 3
    omega_3 = 10.0
    # Small tilt (nutation kick)
    # L = I * omega
    L_init = np.array([0.1, 0.1, I3 * omega_3], dtype=np.float32)
    L = Tensor(L_init, requires_grad=True)
    
    # Gamma: Direction of Gravity in Body Frame.
    # Initially, let's say Top is tilted by angle theta from vertical.
    # Vertical Z axis in space. Top axis z tilted.
    # Gamma = R^T * k_space. 
    # If top is tilted by 30 deg around x: Gamma = [0, sin(30), cos(30)]
    theta = np.deg2rad(30)
    Gamma_init = np.array([0, np.sin(theta), np.cos(theta)], dtype=np.float32)
    Gamma = Tensor(Gamma_init, requires_grad=True)
    
    print(f"Start Heavy Top Simulation")
    print(f"Initial L: {L.numpy()}")
    print(f"Initial Gamma: {Gamma.numpy()}")
    
    history_gamma = [] # Trace the path of "Down" on the unit sphere
    history_axis = [] # Trace the Top's axis in Space Frame (we need R for that)
    # Actually, Gamma IS the Down vector seen in Body Frame. 
    # To visualize Precession, it is often easier to plot the Top Axis in the Space Frame.
    # But Gamma is sufficient to show the "motion on the sphere". 
    # Let's track Gamma vector in Body Frame (which moves). 
    # AND let's track R (Orientation) to show real space motion.
    
    # Quaternion for visualization
    # We need to integrate orientation too.
    # q represents R from Body to Space.
    # R_space_body.
    # If Gamma_body is [0, sin, cos], and Gravity_space is [0,0,1] ??
    # Actually usually Gamma = R^{-1} e_z.
    # So e_z (Up in Space) seen in Body.
    # If Gamma is [0, sin, cos], it means Up is tilted.
    
    # Init Quaternion from tilt
    # Tilt by theta around X axis.
    # q = [cos(t/2), sin(t/2), 0, 0]
    quat_init = np.array([np.cos(theta/2), np.sin(theta/2), 0.0, 0.0], dtype=np.float32)
    quat = Tensor(quat_init, requires_grad=True)
    
    for i in range(steps):
        # State Vector Y = [L, Gamma] (Size 6)
        
        def dynamics(L_val, Gamma_val):
            # L, Gamma are tensors
            
            # Gradients of Hamiltonian
            # H = 0.5 L . I^-1 L + mgl * Gamma_3
            
            Omega = L_val * I_inv # dH/dL
            
            # dH/dGamma = mgl * e3 = [0, 0, mgl]
            dH_dGamma = Tensor([0.0, 0.0, m * g * l])
            
            # Equations of Motion (SE(3) Lie Poisson):
            # 1. dL/dt = L x Omega + Gamma x dH/dGamma
            # 2. dGamma/dt = Gamma x Omega
            
            dL = cross(L_val, Omega) + cross(Gamma_val, dH_dGamma)
            dGamma = cross(Gamma_val, Omega)
            
            return dL, dGamma

        # RK4 Integration
        L_np = L.numpy()
        G_np = Gamma.numpy()
        
        def f(l_n, g_n):
            dl, dg = dynamics(Tensor(l_n), Tensor(g_n))
            return dl.numpy(), dg.numpy()
            
        k1_l, k1_g = f(L_np, G_np)
        
        k2_l, k2_g = f(L_np + 0.5*dt*k1_l, G_np + 0.5*dt*k1_g)
        k3_l, k3_g = f(L_np + 0.5*dt*k2_l, G_np + 0.5*dt*k2_g)
        
        k4_l, k4_g = f(L_np + dt*k3_l, G_np + dt*k3_g)
        
        L_next = L_np + (dt/6.0)*(k1_l + 2*k2_l + 2*k3_l + k4_l)
        G_next = G_np + (dt/6.0)*(k1_g + 2*k2_g + 2*k3_g + k4_g)
        
        # Normalize Gamma (Constraint enforcement)
        G_next /= np.linalg.norm(G_next)
        
        L = Tensor(L_next, requires_grad=True)
        Gamma = Tensor(G_next, requires_grad=True)
        
        # Orientaiton (Quaternion) Update
        # dq/dt = 0.5 * q * Omega
        omega_curr = (L * I_inv).numpy()
        q_np = quat.numpy()
        
        s1, v1 = q_np[0], q_np[1:]
        s2, v2 = 0.0, omega_curr
        
        s_new = s1*s2 - np.dot(v1, v2)
        v_new = s1*v2 + s2*v1 + np.cross(v1, v2)
        dq = np.concatenate(([s_new], v_new)) * 0.5
        q_next = q_np + dq * dt
        q_next /= np.linalg.norm(q_next)
        
        quat = Tensor(q_next, requires_grad=True)
        
        if i % 10 == 0:
            history_gamma.append(G_next.tolist())
            history_axis.append(q_next.tolist()) # We'll decode q in viewer
            
        # Conservation Checks
        if i == 0:
            H_start = (0.5*(L*L*I_inv).sum() + m*g*l*Gamma[2]).numpy()
            LG_start = (L*Gamma).sum().numpy()
            
    H_end = (0.5*(L*L*I_inv).sum() + m*g*l*Gamma[2]).numpy()
    LG_end = (L*Gamma).sum().numpy()
    
    print(f"Energy H: Start {H_start:.4f}, End {H_end:.4f}, Drift {abs(H_end-H_start)/abs(H_start):.2e}")
    # Component of L along Gravity is conserved for Symmetric top
    print(f"Casimir L.G: Start {LG_start:.4f}, End {LG_end:.4f}, Drift {abs(LG_end-LG_start)/abs(LG_start):.2e}")
    
    generate_viewer(history_axis) # Pass quaternions

def generate_viewer(history):
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
    <p>Trace of the Top's Tip (Green) and CM (Red) projected on ground.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history)}; // Quaternions
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 150; 
        
        let frame = 0;
        
        // Trail buffers
        const trail = [];
        const maxTrail = 200;
        
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
            ctx.fillStyle = 'rgba(0,0,0,0.2)';
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            const q = history[frame];
            
            // Top Axis in Body is Z [0,0,1] (if symmetric around Z)
            // But usually heavy top is symmetric about Z.
            // Let's project the Tip of the top.
            const tip_body = [0, 0, 1.5]; 
            const tip_space = rotateByQuat(tip_body, q);
            
            // Draw Trail (Projected on XY plane - overhead view)
            trail.push([cx + tip_space[0]*scale, cy - tip_space[1]*scale]);
            if (trail.length > maxTrail) trail.shift();
            
            ctx.strokeStyle = '#0f0';
            ctx.beginPath();
            if(trail.length > 0) ctx.moveTo(trail[0][0], trail[0][1]);
            for(let p of trail) ctx.lineTo(p[0], p[1]);
            ctx.stroke();
            
            // Draw Top "Cone" roughly
            const base_pts = [[0.5,0,0], [-0.5,0,0], [0,0.5,0], [0,-0.5,0]];
            
            ctx.fillStyle = '#ff4444';
            ctx.beginPath();
            const top_tip = [cx + tip_space[0]*scale, cy - tip_space[1]*scale];
            
            // Origin (Pivot)
            const pivot = [cx, cy]; // Origin is fixed in space (lagrange top)
            
            ctx.moveTo(pivot[0], pivot[1]);
            ctx.lineTo(top_tip[0], top_tip[1]);
            ctx.strokeStyle = '#fff';
            ctx.stroke();
            
            // Draw Tip
            ctx.beginPath();
            ctx.arc(top_tip[0], top_tip[1], 5, 0, 2*Math.PI);
            ctx.fill();
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}
        draw();
    </script>
</body>
</html>
    """
    with open('examples/heavy_top_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/heavy_top_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
