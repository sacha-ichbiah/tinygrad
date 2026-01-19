import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import symplectic_step
import json
import os

def run_simulation():
    # Simulation Parameters
    N = 5  # More bodies for chaos
    G = 1.0 # Gravitational constant
    dt = 0.001 
    steps = 2000 
    
    # 1. Define Initial State
    # Random reproducible seed
    np.random.seed(42)
    
    # Positions: Random distribution
    q_init = np.random.randn(N, 2).astype(np.float32) * 2.0
    # Momenta: Random distribution (small velocities)
    # Ensure zero total momentum to keep it centered-ish
    p_init = np.random.randn(N, 2).astype(np.float32) * 0.5
    p_init -= p_init.mean(axis=0) # Center of momentum frame
    
    # Masses (N,)
    masses_np = np.random.uniform(0.5, 2.0, size=(N,)).astype(np.float32)
    
    q = Tensor(q_init, requires_grad=True)
    p = Tensor(p_init, requires_grad=True)
    masses = Tensor(masses_np) # (N,)
    
    print(f"Start N-Body Simulation: N={N}, dt={dt}, steps={steps}")
    print(f"Initial Setup: Random Cloud")
    
    history_q = []
    
    for i in range(steps):
        # A. Compute Potential Energy U
        # U = -Sum(G * mi * mj / |qi - qj|)
        
        # Broadcast to get pairwise differences: (N, 1, 2) - (1, N, 2) -> (N, N, 2)
        diff = q.unsqueeze(1) - q.unsqueeze(0)
        
        # Distances: (N, N)
        # add epsilon to avoid diag division by zero (or mask it)
        dist = (diff * diff).sum(axis=2).sqrt()
        
        # Mask diagonal (set to infinity or large number so 1/dist is 0)
        # Ideally we'd use a mask, but for simple potential calc:
        # We can add a "softening" parameter to gravity: 1 / sqrt(r^2 + eps^2)
        softening = 1e-2
        inv_dist = (diff * diff).sum(axis=2).add(softening**2).rsqrt()
        
        # Remove self-interaction (diagonal is where inv_dist would be massive if not for softening)
        # Actually with softening, self-interaction gives a constant potential -G*m^2/eps
        # which contributes 0 force. So it is fine!
        
        # Pairwise Energy matrix
        # (N, 1) * (1, N) * (N, N) -> (N, N)
        # G * mi * mj * (1/r)
        m_pairs = masses.unsqueeze(1) * masses.unsqueeze(0)
        
        # U = -0.5 * G * sum(m_i * m_j * inv_dist)
        # 0.5 because we sum i,j and j,i
        U = -0.5 * G * (m_pairs * inv_dist).sum()
        
        # Force = -grad(U)
        # tinygrad automatically handles the sum and broadcasting in backward!
        grads = U.backward()
        # The gradient of U w.r.t q is -Force.
        # So Force = -grads
        # Wait, usually we do loss.backward(). 
        # Here we want force = -dU/dq.
        # calling .backward() on U, populates q.grad with dU/dq.
        # So force = -q.grad
        
        # However, calling backward() accumulates gradients. We need to clear them or handle it.
        # But q is updated every step. The q from previous step is a new graph node.
        # So q.grad will be None initially for the new q.
        
        # BUT: tinygrad tensor.grad is populated on the tensor.
        # We need to extract it.
        
        # NOTE: tinyphysics/physics.py symplectic_step expects a 'force' tensor 
        # that doesn't necessarily need to be part of the backward graph for *this* step's update,
        # but for the *overall* backprop through time.
        # Actually, if we want to differentiate through the simulation, 'force' must be connected to 'q'.
        # If we just take `-q.grad`, that is a detached value (usually).
        # Wait, in tinygrad `backward()` computes gradients of leaf nodes? Or all nodes with requires_grad?
        # Typically leaf nodes. `q` from the start of the loop is not a leaf node after step 0 (it's result of Add).
        # So `backward` might not populate `q.grad` if `q` is intermediate.
        # We need `tinygrad.grad` equivalent or `U.grad(q)` if it existed.
        
        # WORKAROUND for "Force from Energy" in Autograd:
        # We want vector computation graph for Force.
        # F = - grad(U).
        # In modern AD (like JAX/PyTorch functional), we do grad(U)(q).
        # In tinygrad Tensor-based AD, we usually do scalar.backward().
        
        # Let's try explicitly using `grad` if available, or simpler:
        # If we use `U.backward()`, it backprops to the inputs. 
        # If `q` is intermediate, we need to `retain_grad` or equivalent.
        # But actually, we want the *symbolic* force for the forward pass, 
        # OR we just compute the numerical force for the forward pass.
        
        # For a "Physics as Compiler" approach, we ideally want the Force *Graph*.
        # But `symplectic_step` takes `force` as input.
        # If we pass a numerical constant force (detached), we break the graph for higher order derivatives 
        # (like optimizing initial conditions).
        # If we want end-to-end diff, `force` must be a function of `q`.
        # `q.grad` provides numerical value.
        
        # Let's check `tinygrad` capabilities. 
        # If we assume we just want to run the sim first:
        # We can do `force = -grad(U)`.
        # To get that in tinygrad without "functional" grad:
        # We might have to implement the force explicitly for now, 
        # OR use the `U.backward()` -> read `q.grad` -> `force = q.grad.detach() * -1` 
        # -> BUT re-attach? No, that breaks the graph.
        
        # Architecture decision:
        # 1. Explicit Force Calculation (Vectorized Newton) -> Easy, preserves graph.
        # 2. Autograd Force -> Ideally we want this to define system by Hamiltonian only.
        
        # Let's Implement Analytical Force for N-Body using Vector Math (Broadcasting),
        # because it guarantees graph connectivity.
        # F_i = Sum_j ( G * mi * mj * (rj - ri) / |rj - ri|^3 )
        
        diff = q.unsqueeze(1) - q.unsqueeze(0) # (N, N, 2) : r_j - r_i (if we carefully check signs)
        # Actually `q.unsqueeze(1)` is `q` along dim 0 broadcasted to dim 1? 
        # q shape (N, 2).
        # q.unsqueeze(1) -> (N, 1, 2) i-index
        # q.unsqueeze(0) -> (1, N, 2) j-index
        # diff[i, j] = q[i] - q[j]. 
        # We want force on i. F_i = Sum_j G mi mj (qj - qi) / dist^3
        # qj - qi = - (qi - qj) = -diff[i, j]
        
        # dist2 = |diff|^2 + epsilon
        dist2 = (diff*diff).sum(axis=2) + softening**2
        dist_inv3 = dist2.pow(-1.5)
        
        # F_ij magnitude = G * mi * mj * dist_inv3
        # F_ij vector = F_magnitude * (qj - qi) 
        #             = G * mi * mj * dist_inv3 * (q[j] - q[i])
        #             = G * mi * mj * dist_inv3 * (-diff)
        
        # masses pairwise
        m_matrix = masses.unsqueeze(1) * masses.unsqueeze(0) # (N, N)
        
        # Force matrix (N, N, 2)
        # We need to sum over j (dim 1)
        force_matrix = - (diff * dist_inv3.unsqueeze(2)) * (G * m_matrix.unsqueeze(2))
        
        # Net force on each body (N, 2)
        force = force_matrix.sum(axis=1)
        
        # Symplectic Step
        # Need masses broadcasted for division (N, 1) or (N,)
        q, p = symplectic_step(q, p, force, dt=dt, mass=masses.unsqueeze(1))
        
        if i % 2 == 0:
            history_q.append(q.numpy().tolist())
        
        # Calculate Total Energy for verification
        if i == 0 or i == steps - 1:
            # Kinetic Energy: Sum(p^2 / 2m)
            K = (p.pow(2).sum(axis=1) / (2 * masses)).sum()
            # Potential Energy (already computed in U variable, but let's be careful about U calculation)
            # U in loop was: -0.5 * G * sum(mi * mj / (r + eps))
            # That is the correct Potential Energy
            E = K + U
            if i == 0: E_initial = E.numpy()
            if i == steps - 1: E_final = E.numpy()

    print(f"End Sim. Final q[0]: {q.numpy()[0]}")
    
    print(f"\nEnergy Conservation Check:")
    print(f"Initial Energy: {E_initial:.6f}")
    print(f"Final Energy:   {E_final:.6f}")
    drift = abs((E_final - E_initial) / E_initial)
    print(f"Relative Drift: {drift:.2e}")
    if drift < 1e-4:
        print("SUCCESS: Energy conserved (drift < 1e-4)")
    else:
        print("WARNING: Energy drift high. Check symplectic step or timestep.")
    
    # Generate Viewer
    generate_viewer_nbody(history_q, N)

def generate_viewer_nbody(history_q, N):
    # history_q is list of (N, 2) arrays (as lists)
    
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>TinyPhysics N-Body</title>
    <style>
        body {{ font-family: sans-serif; background: #000; color: #eee; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #333; }}
    </style>
</head>
<body>
    <h1>N-Body Gravity (N={N})</h1>
    <canvas id="simCanvas" width="800" height="600"></canvas>
    <script>
        const history = {json.dumps(history_q)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const scale = 100; // Pixels per unit
        
        let frame = 0;
        
        function draw() {{
            // Fade effect
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, width, height);
            
            const positions = history[frame];
            
            for (let i = 0; i < positions.length; i++) {{
                const x = positions[i][0] * scale + width/2;
                const y = positions[i][1] * scale + height/2;
                
                ctx.beginPath();
                ctx.arc(x, y, i===0 ? 8 : 4, 0, 2*Math.PI); // First body bigger
                ctx.fillStyle = i===0 ? '#ffcc00' : '#00ccff';
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
    
    with open('examples/n_body_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/n_body_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
