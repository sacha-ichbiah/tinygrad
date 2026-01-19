import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import symplectic_step
import json
import os

def run_simulation():
    # Parameters
    G = 1.0
    M = 1.0
    mu = 1.0  # Reduced mass approx (for test particle m << M) -> m
    
    # Initial State: Highly Elliptical Orbit
    # Periapsis distance r_p = 1.0
    # Velocity at periapsis v_p > v_circ
    # v_circ = sqrt(GM/r_p) = 1.0
    # v_esc = sqrt(2GM/r_p) = 1.414
    # let's pick v_p = 1.2 (Elliptical)
    
    r_p = 1.0
    v_p_mag = 1.2
    
    q_init = np.array([r_p, 0.0], dtype=np.float32)
    p_init = np.array([0.0, mu * v_p_mag], dtype=np.float32)
    
    q = Tensor(q_init, requires_grad=True)
    p = Tensor(p_init, requires_grad=True)
    
    dt = 0.01
    steps = 2000
    
    print(f"Start Kepler Simulation: e ~ 0.5")
    
    history_q = []
    
    for i in range(steps):
        # 1. Compute Potential Force
        # H = p^2/2m - GMm/r
        # V = - GMm * (q^2)^-0.5
        
        # Using tinygrad primitives for singularity handling (Rsqrt)
        r2 = (q * q).sum()
        # Add epsilon for numerical safety if needed, but for mathematical purity try without first
        # r = r2.sqrt()
        # V = -G * M * mu / r
        
        # Better with rsqrt: 1/sqrt(x)
        inv_r = r2.rsqrt() 
        V = -G * M * mu * inv_r
        
        # Force = -grad(V)
        # We can implement this analytically or via autograd.
        # Autograd is the "TinyPhysics" way.
        
        # We need force for symplectic_step.
        # Force = -dV/dq
        
        grads = V.backward()
        # q.grad contains dV/dq = -Force
        force = -q.grad
        
        # Reset grad is implicit in next backward pass? 
        # Tinygrad accumulates gradients. We MUST zero them if we loop.
        # But we create NEW computational graph each step (q is new tensor), 
        # so previous q.grad doesn't affect new q.
        # HOWEVER, q.grad is attached to the tensor `q`.
        # When we do `q, p = step(...)`, `q` becomes a new Tensor.
        # The OLD `q` stays in memory until GC, with its grad.
        # The NEW `q` has no grad yet.
        # So we don't need to manually zero grad if we are stepping forward in time 
        # creating new graph nodes.
        
        # 2. Symplectic Step
        # This works because H is separable H = T(p) + V(q)
        q, p = symplectic_step(q, p, force, dt=dt, mass=mu)
        
        if i % 5 == 0:
            history_q.append(q.numpy().tolist())
            
        # Check Energy Conservation
        if i == 0:
            K = (p*p).sum() / (2*mu)
            E_start = (K + V).numpy()
            
    # Final Energy
    r2 = (q * q).sum()
    inv_r = r2.rsqrt()
    V = -G * M * mu * inv_r
    K = (p*p).sum() / (2*mu)
    E_end = (K + V).numpy()
    
    print(f"E_start: {E_start:.6f}")
    print(f"E_end:   {E_end:.6f}")
    print(f"Drift:   {abs(E_end - E_start) / abs(E_start):.2e}")
    
    generate_viewer(history_q)

def generate_viewer(history_q):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kepler Problem</title>
    <style>
        body {{ font-family: sans-serif; background: #000; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #444; }}
    </style>
</head>
<body>
    <h1>Kepler Orbit (1/r Potential)</h1>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history_q)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 50; 
        
        // Draw Star
        function drawStar() {{
            ctx.fillStyle = '#ffaa00';
            ctx.beginPath();
            ctx.arc(cx, cy, 10, 0, 2*Math.PI);
            ctx.fill();
        }}
        
        ctx.fillStyle = 'rgba(0,0,0,1)';
        ctx.fillRect(0,0, canvas.width, canvas.height);
        
        // Draw Orbit Path
        ctx.strokeStyle = '#444';
        ctx.beginPath();
        if (history.length > 0) {{
             ctx.moveTo(cx + history[0][0]*scale, cy - history[0][1]*scale);
             for(let i=1; i<history.length; i++) {{
                 ctx.lineTo(cx + history[i][0]*scale, cy - history[i][1]*scale);
             }}
        }}
        ctx.stroke();
        
        let frame = 0;
        
        function animate() {{
            // Trail effect
            ctx.fillStyle = 'rgba(0,0,0,0.1)';
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            drawStar();
            
            const pos = history[frame];
            const x = cx + pos[0] * scale;
            const y = cy - pos[1] * scale; // Flip Y for screen coords
            
            ctx.fillStyle = '#00ccff';
            ctx.beginPath();
            ctx.arc(x, y, 5, 0, 2*Math.PI);
            ctx.fill();
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(animate);
        }}
        
        animate();
    </script>
</body>
</html>
    """
    
    with open('examples/kepler_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/kepler_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
