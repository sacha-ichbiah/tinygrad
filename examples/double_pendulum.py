import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import symplectic_step
import json
import os

def run_simulation():
    # Parameters
    g = 9.81
    m1 = 1.0
    m2 = 1.0
    l1 = 1.0
    l2 = 1.0
    
    dt = 0.01
    steps = 1000
    
    # Initial State (Angles in radians, Momenta)
    # High energy state to show chaos
    theta1_init = np.pi / 2 # Horizontal
    theta2_init = 0.0       # Hanging down relative to first arm
    p1_init = 0.0
    p2_init = 0.0
    
    # We pack q = [theta1, theta2], p = [p1, p2]
    q = Tensor([theta1_init, theta2_init], requires_grad=True)
    p = Tensor([p1_init, p2_init], requires_grad=True)
    
    # Constants for Mass Matrix
    # M11 = (m1 + m2) * l1^2
    # M22 = m2 * l2^2
    # M12 = m2 * l1 * l2 * cos(theta1 - theta2)
    c1 = (m1 + m2) * l1**2
    c2 = m2 * l2**2
    c3 = m2 * l1 * l2
    
    print(f"Start Double Pendulum Simulation")
    print(f"Initial: theta1={q.numpy()[0]:.2f}, theta2={q.numpy()[1]:.2f}")
    
    history_q = []
    
    for i in range(steps):
        # 1. Compute Hamiltonian H(q, p)
        # We need to construct H explicitly so we can differentiate it to get forces.
        # Actually, symplectic_step needs 'force' which is -dH/dq.
        # And usually 'velocity' dq/dt = dH/dp.
        # But `symplectic_step` assumes separable Hamiltonian H = p^2/2m + V(q) where dq/dt = p/m.
        # THE DOUBLE PENDULUM IS NON-SEPARABLE in generalized coordinates if using p!
        # T = 0.5 * p^T * M(q)^-1 * p
        # dH/dp = M(q)^-1 * p  (Velocity depends on position!)
        # dH/dq = d/dq (0.5 * p^T * M(q)^-1 * p) + dV/dq
        
        # CRITICAL: `symplectic_step` in `tinygrad/physics.py` implements Semi-Implicit Euler:
        # p_new = p - dH/dq * dt
        # q_new = q + p_new/m * dt  <-- ASSUMES dq/dt = p/m
        
        # Our `symplectic_step` implementation:
        # q_new = q + (p_new / mass) * dt
        # It assumes mass is constant diagonal.
        
        # FOR DOUBLE PENDULUM, dq/dt = M(q)^{-1} p.
        # The simple `symplectic_step` is NOT SUFFICIENT for non-separable Hamiltonians / position-dependent mass.
        # We need to Upgrade the Integrator or Manually implement the step for this problem.
        # Since this is a "Physics Engine" building task, we should probably add `GeneralSymplecticStep` 
        # or implement it manually in the loop.
        
        # Let's implement the Hamiltonian equations manually in the loop using Autograd for derivatives.
        # H = T + V
        # dq/dt = dH/dp
        # dp/dt = -dH/dq
        
        # Use Symplectic Euler or Velocity Verlet? 
        # For non-separable H, symplectic integrators are harder (often implicit).
        # But for H(q,p) = T(q,p) + V(q), split is T and V.
        # Explicit Step:
        # 1. p_{t+1/2} = p_t - 0.5 * dt * dH/dq(q_t, p_t)
        # 2. q_{t+1}   = q_t + dt * dH/dp(q_t, p_{t+1/2})   <-- But H depends on q!
        # 
        # If H is non-separable, standard Störmer-Verlet requires implicit solve.
        # HOWEVER, we can use a simpler explicit method like RK4 (not symplectic) or a generalized symplectic one.
        # Given the "Roadmap" implies "Symplectic Euler Op" was Level 1.1, and this is 1.2 "New Primitive Needed" -> ChainRule (Dense).
        # Maybe we stick to RK4 for this complex system if Symplectic is too hard, OR we try to approximate.
        # But wait, looking at the Roadmap:
        # Level 1.2 "Double Pendulum". Primitive: "ChainRule (Dense)". 
        # It doesn't explicitly ask for a new integrator.
        # But preserving energy is the goal.
        
        # Let's try to implement a simple "Symplectic-like" split if possible, 
        # or just use RK4 which is robust albeit non-symplectic (energy drift over long time).
        # Actually, let's use the Autograd to get the exact equations of motion and just forward integrate.
        # If we use tinygrad, we can defining H, then:
        # dots = grad(H) -> (dH/dq, dH/dp)
        # state += dots * dt ... (Forward Euler) -> Drifts fast.
        # state += RK4_step(state) -> Better.
        
        # Let's implement RK4 for the Double Pendulum to show off Autograd's power handling the complex derivatives.
        # Defining H is enough to get ALL motion equations!
        
        def hamiltonian(q_val, p_val):
            # q: (2,)
            # p: (2,)
            theta1, theta2 = q_val[0], q_val[1]
            p1, p2 = p_val[0], p_val[1]
            
            # Mass Matrix Inversion
            # det = M11*M22 - M12^2
            # M11 = c1, M22 = c2, M12 = c3 * cos(t1-t2)
            
            # Use tinygrad ops
            cos_delta = (theta1 - theta2).cos()
            M11 = c1
            M22 = c2
            M12 = c3 * cos_delta
            
            det = M11 * M22 - M12 * M12
            
            # Inverse M elements
            # M_inv = 1/det * [[M22, -M12], [-M12, M11]]
            inv_det = 1.0 / det
            
            # T = 0.5 * p^T * M_inv * p
            #   = 0.5 * inv_det * (M22 p1^2 - 2 M12 p1 p2 + M11 p2^2)
            T = 0.5 * inv_det * (M22 * p1**2 - 2 * M12 * p1 * p2 + M11 * p2**2)
            
            # V = -m1 g l1 cos t1 - m2 g (l1 cos t1 + l2 cos t2)
            V = -m1 * g * l1 * theta1.cos() - m2 * g * (l1 * theta1.cos() + l2 * theta2.cos())
            
            return T + V

        # RK4 Step
        # k1 = f(y)
        # k2 = f(y + 0.5*dt*k1)
        # k3 = f(y + 0.5*dt*k2)
        # k4 = f(y + dt*k3)
        # y += dt/6 * (k1 + 2k2 + 2k3 + k4)
        
        # State vector y = [q, p] (size 4)
        # dy/dt = [dH/dp, -dH/dq]
        
        def get_derivs(q_curr, p_curr):
            # We need gradients of H w.r.t q and p
            # H(q, p)
            # In tinygrad, we compute H, then backward.
            
            # We need to make sure we don't accumulate gradients on the persistent state tensors
            # So we detach or create new leaf nodes.
            # q_curr and p_curr are already numpy arrays here
            q_in = Tensor(q_curr, requires_grad=True)
            p_in = Tensor(p_curr, requires_grad=True)
            
            H = hamiltonian(q_in, p_in)
            H.backward()
            
            dq_dt = p_in.grad # dH/dp
            dp_dt = -q_in.grad # -dH/dq
            
            return dq_dt.numpy(), dp_dt.numpy() # Return as numpy for RK4 math
            
        # RK4 Implementation with Numpy (using Autograd for F)
        q_np = q.numpy()
        p_np = p.numpy()
        
        # k1
        dq1, dp1 = get_derivs(q_np, p_np)
        
        # k2
        dq2, dp2 = get_derivs(q_np + 0.5*dt*dq1, p_np + 0.5*dt*dp1)
        
        # k3
        dq3, dp3 = get_derivs(q_np + 0.5*dt*dq2, p_np + 0.5*dt*dp2)
        
        # k4
        dq4, dp4 = get_derivs(q_np + dt*dq3, p_np + dt*dp3)
        
        # Update
        q_np = q_np + (dt/6.0) * (dq1 + 2*dq2 + 2*dq3 + dq4)
        p_np = p_np + (dt/6.0) * (dp1 + 2*dp2 + 2*dp3 + dp4)
        
        # Update Tensors
        q = Tensor(q_np, requires_grad=True)
        p = Tensor(p_np, requires_grad=True)
        
        # Check energy for conservation
        if i == 0:
            H_start = hamiltonian(q, p).numpy()
            print(f"H_start: {H_start}")
        
        if i % 10 == 0:
            history_q.append(q.numpy().tolist())
            
    H_end = hamiltonian(q, p).numpy()
    print(f"H_end: {H_end}")
    print(f"Drift: {abs(H_end - H_start)/abs(H_start):.2e}")
            
    # Viewer
    generate_viewer(history_q, l1, l2)

def generate_viewer(history_q, l1, l2):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Double Pendulum</title>
    <style>
        body {{ font-family: sans-serif; background: #111; color: #eee; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #333; background: #000; }}
    </style>
</head>
<body>
    <h1>Double Pendulum</h1>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history_q)};
        const l1 = {l1} * 100;
        const l2 = {l2} * 100;
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 3;
        
        let frame = 0;
        
        function draw() {{
            ctx.fillStyle = 'rgba(0, 0, 0, 0.3)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            const angles = history[frame];
            const t1 = angles[0];
            const t2 = angles[1];
            
            const x1 = cx + l1 * Math.sin(t1);
            const y1 = cy + l1 * Math.cos(t1);
            
            const x2 = x1 + l2 * Math.sin(t2);
            const y2 = y1 + l2 * Math.cos(t2);
            
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();
            
            // Masses
            ctx.fillStyle = '#f00';
            ctx.beginPath(); ctx.arc(x1, y1, 8, 0, 2*Math.PI); ctx.fill();
            
            ctx.fillStyle = '#0f0';
            ctx.beginPath(); ctx.arc(x2, y2, 8, 0, 2*Math.PI); ctx.fill();
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}
        
        draw();
    </script>
</body>
</html>
    """
    with open('examples/double_pendulum_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/double_pendulum_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
