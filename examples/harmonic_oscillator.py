import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import symplectic_step
import os
try:
    from examples.physics_viewer import PhysicsViewer1D
except ImportError:
    from physics_viewer import PhysicsViewer1D

def run_simulation():
    # 1. Define Initial State (Learnable!)
    # We set requires_grad=True to show we can differentiate through the physics
    q = Tensor([1.0], requires_grad=True)  # Start at x=1
    p = Tensor([0.0], requires_grad=True)  # Start at rest
    
    # Physics Constants
    k = 1.0   # Spring constant
    m = 1.0   # Mass
    dt = 0.1
    steps = 100
    
    print(f"Start: q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}")

    # 2. The Physics Loop (The Computational Graph)
    history_q = []
    history_p = []
    
    for i in range(steps):
        # A. Compute Potentials/Forces (The Hamiltonian Part)
        # Force = -grad(U), for spring U=0.5*k*q^2 -> F = -k*q
        force = -k * q 
        
        # B. Apply the Custom Op
        # The graph does not grow by 10 nodes here, just 1 node: SymplecticEuler
        q, p = symplectic_step(q, p, force, dt=dt, mass=m)
        
        # Collect full history for viewer
        history_q.append(float(q.numpy()[0]))
        history_p.append(float(p.numpy()[0]))

    print(f"End:   q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}")
    
    # 3. The "Magic": Backprop through time
    # Let's ask: "How much does the final position depend on the initial velocity?"
    target = 0.0 # We want to end at 0
    loss = ((q - target)**2).sum()
    
    loss.backward()
    
    print(f"\nGradient Analysis:")
    # Check if p.grad is not None before accessing it
    if p.grad is not None:
        print(f"dL/dp_initial: {p.grad.numpy()[0]:.4f}")
        print("If this is non-zero, the solver successfully differentiated through 100 physics steps!")
    else:
        print("dL/dp_initial is None. Something went wrong with gradient propagation.")

    # 4. Generate Viewer
    viewer = PhysicsViewer1D(title="Harmonic Oscillator (Symplectic)")
    viewer.render(history_q, history_p, dt, "examples/harmonic_oscillator_viewer.html")



if __name__ == "__main__":
    run_simulation()
