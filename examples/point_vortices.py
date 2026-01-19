import numpy as np
from tinygrad.tensor import Tensor
import json
import os

def run_simulation():
    # Parameters
    # N Point Vortices
    # Case 1: Vortex Dipole (Two vortices, opposite circulation) -> Translates
    # Case 2: Two same sign -> Rotate
    # Let's do 4 vortices: A dipole pair and another dipole pair colliding?
    # Or just random.
    # Let's do a "Leapfrogging" setup if possible, or just random N=10.
    
    np.random.seed(42)
    N = 4
    
    # Random Positions in [-1, 1]
    q_init = np.random.randn(N, 2).astype(np.float32)
    
    # Circulations Gamma
    # Random +/- 1
    gamma_np = np.random.choice([-1.0, 1.0], size=(N,)).astype(np.float32)
    # Enforce sum(Gamma) != 0 to make it interesting, or 0.
    
    # Tinygrad Tensors
    # State q = [x, y] of all vortices. 
    # This acts as Phase Space! x ~ q, y ~ p (scaled by Gamma)
    # Strictly: Gamma_i x_i is conjugate to y_i (or similar)
    
    q = Tensor(q_init, requires_grad=True)
    Gamma = Tensor(gamma_np)
    
    dt = 0.01
    steps = 2000
    
    print(f"Start Point Vortex Simulation (N={N})")
    print(f"Circulations: {Gamma.numpy()}")
    
    history_q = []
    
    for i in range(steps):
        # Hamiltonian Dynamics for Point Vortices
        # H = -1/(2pi) * Sum_{i<j} Gamma_i Gamma_j ln(r_ij)
        
        # We need dH/dx and dH/dy to get velocity.
        # But wait, the symplectic form is non-standard:
        # Gamma_i dx/dt = dH/dy
        # Gamma_i dy/dt = -dH/dx
        
        # So Velocity V = [dx/dt, dy/dt]
        # dx/dt = (1/Gamma_i) dH/dy
        # dy/dt = -(1/Gamma_i) dH/dx
        
        # Let's compute H and use Autograd!
        
        def hamiltonian(pos):
            # pos is (N, 2)
            # Pairwise distances
            # (N, 1, 2) - (1, N, 2)
            diff = pos.unsqueeze(1) - pos.unsqueeze(0)
            
            # r^2 = dx^2 + dy^2
            r2 = (diff*diff).sum(axis=2)
            
            # Avoid diagonal (r=0) -> log(0)
            # Add identity matrix to diagonal or mask?
            # log(r) = 0.5 * log(r^2)
            # We can just add epsilon, or better:
            # Since i<j, we can mask the diagonal/lower triangle.
            
            # Softening for numerical stability
            eps = 1e-2
            log_r = (r2 + eps).log() * 0.5
            
            # Interaction Strength matrix: Gamma_i * Gamma_j
            G_mat = Gamma.unsqueeze(1) * Gamma.unsqueeze(0)
            
            # Sum i < j
            # We can sum all non-diagonal and divide by 2
            # Mask diagonal
            # tinygrad doesn't have diagonal mask easy?
            # But log(r_ii) ~ log(eps).
            # If we sum all and subtract diagonal, effectively.
            # But the formula is Sum i != j.
            
            # H = -1/(2pi) * 0.5 * Sum_{i!=j} G_i G_j ln r_ij
            # Factor 0.5 compensates for double counting ij and ji.
            
            # Diagonals: G_i^2 * ln(eps). This contributes constant shift to H. 
            # Force gradient will be 0 if eps is constant?
            # Wait, d/dx (x-x) = 0. So self-interaction force is 0. 
            # So simple sum is fine!
            
            H_val = - (1.0 / (4.0 * np.pi)) * (G_mat * log_r).sum()
            return H_val

        # Get Gradients
        # We need to act carefully. Compute H, backward.
        
        # We need current q to calculate grads.
        # Detach to separate graph from previous steps?
        # q is updated each step.
        
        # If we use RK4, we need dynamics function f(q).
        
        def dynamics(q_curr):
            # q_curr is Tensor
            H = hamiltonian(q_curr)
            
            # Calculate gradients
            # We need to clear grads? 
            # Creating new graph branch from q_curr.
            grads = H.backward() # dH/dq is stored in q_curr.grad
            
            # Wait, if we pass q_curr which is intermediate node?
            # We should pass a LEAF.
            # So in RK4, we create `q_in = Tensor(val, requires_grad=True)`
            
            pass 
            # But wait, inside dynamics() we need H(q).
            # If we call H(q).backward(), tinygrad computes dH/dq.
            # Return value.
            
            return None # Implementation below
            
        def f(q_vec):
            # q_vec is numpy array
            t = Tensor(q_vec, requires_grad=True)
            H = hamiltonian(t)
            H.backward()
            grad = t.grad.numpy() # (N, 2) -> [dH/dx, dH/dy]
            
            dH_dx = grad[:, 0]
            dH_dy = grad[:, 1]
            
            # Equations:
            # dx/dt = (1/Gamma) * dH/dy
            # dy/dt = -(1/Gamma) * dH/dx
            
            vx = dH_dy / Gamma.numpy()
            vy = -dH_dx / Gamma.numpy()
            
            return np.stack([vx, vy], axis=1)
            
        # RK4 Step
        q_np = q.numpy()
        
        k1 = f(q_np)
        k2 = f(q_np + 0.5*dt*k1)
        k3 = f(q_np + 0.5*dt*k2)
        k4 = f(q_np + dt*k3)
        
        q_new = q_np + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        q = Tensor(q_new, requires_grad=True)
        
        if i % 10 == 0:
            history_q.append(q_new.tolist())
            
        # Conservation Checks
        if i == 0:
            H_start = hamiltonian(q).numpy()
            # Center of Vorticity: Sum Gamma * r
            C_start = (q.numpy() * Gamma.numpy()[:, None]).sum(axis=0)
            # Angular Momentum: Sum Gamma * r^2
            L_start = (Gamma.numpy() * (q.numpy()**2).sum(axis=1)).sum()
            
    # Final Checks
    H_end = hamiltonian(q).numpy()
    C_end = (q.numpy() * Gamma.numpy()[:, None]).sum(axis=0)
    L_end = (Gamma.numpy() * (q.numpy()**2).sum(axis=1)).sum()
    
    print(f"Energy H: Start {H_start:.4f}, End {H_end:.4f}, Drift {abs(H_end-H_start)/abs(H_start):.2e}")
    print(f"Center C: Drift {np.linalg.norm(C_end - C_start):.2e}")
    print(f"AngMom L: Start {L_start:.4f}, End {L_end:.4f}, Drift {abs(L_end-L_start)/abs(L_start):.2e}")
    
    generate_viewer(history_q, Gamma.numpy())

def generate_viewer(history, gammas):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Point Vortices</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #555; background: #000; }}
    </style>
</head>
<body>
    <h1>Point Vortex Dynamics</h1>
    <p>Red = Positive Circulation, Blue = Negative.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history)};
        const gammas = {json.dumps(gammas.tolist())};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 100; 
        
        let frame = 0;
        
        // Trail
        const trails = gammas.map(() => []);
        const maxTrail = 100;
        
        function draw() {{
            // Fade out
            ctx.fillStyle = 'rgba(0,0,0,0.2)';
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            const positions = history[frame];
            
            for(let i=0; i<gammas.length; i++) {{
                const x = cx + positions[i][0] * scale;
                const y = cy - positions[i][1] * scale;
                
                // Trail
                trails[i].push([x,y]);
                if(trails[i].length > maxTrail) trails[i].shift();
                
                ctx.beginPath();
                ctx.strokeStyle = gammas[i] > 0 ? 'rgba(255,100,100,0.5)' : 'rgba(100,100,255,0.5)';
                if(trails[i].length > 0) ctx.moveTo(trails[i][0][0], trails[i][0][1]);
                for(let p of trails[i]) ctx.lineTo(p[0], p[1]);
                ctx.stroke();
                
                // Vortex
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2*Math.PI);
                ctx.fillStyle = gammas[i] > 0 ? '#f55' : '#55f';
                ctx.fill();
            }}
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}
        
        draw();
    </script>
</body>
</html>
    """
    
    with open('examples/point_vortices_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/point_vortices_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
