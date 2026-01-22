"""
Harmonic Oscillator - The simplest physics system (Level 1.1)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. The system automatically derives equations of motion via autograd
    3. Symplectic integrator preserves energy

Hamiltonian: H = p²/2m + kq²/2

This is the "Hello World" of physics simulation.
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import simulate_hamiltonian

try:
    from examples.physics_viewer import PhysicsViewer1D
except ImportError:
    try:
        from physics_viewer import PhysicsViewer1D
    except ImportError:
        PhysicsViewer1D = None


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def harmonic_hamiltonian(k: float = 1.0, m: float = 1.0):
    """
    Returns the Hamiltonian for a harmonic oscillator.

    H = T + V = p²/2m + kq²/2

    The symplectic integrator will automatically derive:
        dq/dt = dH/dp = p/m     (velocity)
        dp/dt = -dH/dq = -kq    (spring force)

    No need to manually compute F = -kx - autograd does it!
    """
    def H(q, p):
        T = (p * p).sum() / (2 * m)  # Kinetic energy
        V = k * (q * q).sum() / 2     # Potential energy
        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(scan: int = 1):
    """
    Simulate a harmonic oscillator using the TinyPhysics compiler approach.
    """
    # Physics constants
    k = 1.0   # Spring constant
    m = 1.0   # Mass
    dt = 0.1
    steps = 100

    # Initial state
    q = Tensor([1.0], requires_grad=True)  # Start displaced
    p = Tensor([0.0], requires_grad=True)  # Start at rest

    # Expected period: T = 2*pi*sqrt(m/k)
    T_period = 2 * np.pi * np.sqrt(m / k)

    print("=" * 60)
    print("HARMONIC OSCILLATOR - TinyPhysics Compiler Approach")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H(q, p) = p²/2m + kq²/2")
    print(f"\nEquations of motion derived automatically via autograd:")
    print(f"  dq/dt = +dH/dp = p/m")
    print(f"  dp/dt = -dH/dq = -kq")
    print(f"\nIntegrator: auto")
    print(f"Period: {T_period:.4f}, Simulation time: {dt*steps:.1f} ({dt*steps/T_period:.1f} periods)")

    # CREATE THE HAMILTONIAN SYSTEM
    H = harmonic_hamiltonian(k=k, m=m)
    # Initial energy
    E_start = float(H(q, p).numpy())
    print(f"\nInitial: q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}, E={E_start:.6f}")

    # EVOLVE
    q, p, history = simulate_hamiltonian(H, q, p, dt=dt, steps=steps, record_every=1)

    # Final state
    E_end = float(H(q, p).numpy())
    E_drift = abs(E_end - E_start) / abs(E_start)

    print(f"Final:   q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}, E={E_end:.6f}")
    print(f"Energy Drift: {E_drift:.2e}")

    # Backprop through physics (differentiable simulation!)
    print(f"\nDifferentiable Physics Test:")
    target = 0.0
    loss = ((q - target)**2).sum()
    loss.backward()

    if p.grad is not None:
        print(f"dL/dp_initial = {p.grad.numpy()[0]:.4f}")
        print("Gradients flow through {steps} physics steps!")
    else:
        print("(Gradient not available in current mode)")

    # Generate viewer
    if PhysicsViewer1D is not None:
        history_q = [float(h[0][0]) for h in history]
        history_p = [float(h[1][0]) for h in history]
        viewer = PhysicsViewer1D(title=f"Harmonic Oscillator (auto)")
        viewer.render(history_q, history_p, dt, "examples/harmonic_oscillator_viewer.html")

    return E_start, E_end, E_drift


if __name__ == "__main__":
    run_simulation()
