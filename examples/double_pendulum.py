"""
Double Pendulum - Chaotic dynamics (Level 1.2)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. The system automatically derives equations of motion via autograd
    3. Symplectic integrator preserves energy

This is a NON-SEPARABLE Hamiltonian (T depends on q via the mass matrix).
The implicit midpoint integrator handles this correctly.

Hamiltonian: H = T(q, p) + V(q)
where T = 0.5 * p^T * M(q)^{-1} * p (position-dependent mass matrix)
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import simulate_hamiltonian
import json
import os


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def double_pendulum_hamiltonian(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """
    Returns the Hamiltonian for a double pendulum.

    State: q = [theta1, theta2], p = [p1, p2]

    The mass matrix M(q) depends on the angle difference (theta1 - theta2),
    making this a NON-SEPARABLE Hamiltonian. Autograd handles the complex
    derivatives automatically!

    H = T + V where:
        T = 0.5 * p^T * M^{-1}(q) * p
        V = -m1*g*l1*cos(theta1) - m2*g*(l1*cos(theta1) + l2*cos(theta2))
    """
    # Mass matrix constants
    c1 = (m1 + m2) * l1**2  # M11
    c2 = m2 * l2**2          # M22
    c3 = m2 * l1 * l2        # M12 coefficient

    def H(q, p):
        theta1, theta2 = q[0], q[1]
        p1, p2 = p[0], p[1]

        # Mass matrix M(q) - depends on position!
        cos_delta = (theta1 - theta2).cos()
        M11 = c1
        M22 = c2
        M12 = c3 * cos_delta

        # Determinant and inverse
        det = M11 * M22 - M12 * M12
        inv_det = 1.0 / det

        # Kinetic energy: T = 0.5 * p^T * M^{-1} * p
        # M^{-1} = (1/det) * [[M22, -M12], [-M12, M11]]
        T = 0.5 * inv_det * (M22 * p1**2 - 2 * M12 * p1 * p2 + M11 * p2**2)

        # Potential energy: V = gravitational potential
        V = -m1 * g * l1 * theta1.cos() - m2 * g * (l1 * theta1.cos() + l2 * theta2.cos())

        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(dt=0.01, steps=1000):
    """
    Simulate a double pendulum using the TinyPhysics compiler approach.

    For non-separable Hamiltonians, the compiler selects an implicit integrator
    when needed for stability and energy conservation.
    """
    # Physics constants
    m1, m2 = 1.0, 1.0
    l1, l2 = 1.0, 1.0
    g = 9.81

    # Initial state - high energy to show chaos
    theta1_init = np.pi / 2  # Horizontal
    theta2_init = 0.0        # Hanging
    p1_init = 0.0
    p2_init = 0.0

    q = Tensor([theta1_init, theta2_init], requires_grad=True)
    p = Tensor([p1_init, p2_init], requires_grad=True)

    print("=" * 60)
    print("DOUBLE PENDULUM - TinyPhysics Compiler Approach")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H(q, p) = 0.5 * p^T * M(q)^{{-1}} * p + V(q)")
    print(f"\nThis is a NON-SEPARABLE Hamiltonian (mass matrix depends on q).")
    print(f"Autograd handles the complex derivatives automatically!")
    print(f"\nIntegrator: auto")
    print(f"Initial: theta1={theta1_init:.2f}, theta2={theta2_init:.2f}")

    # CREATE THE HAMILTONIAN SYSTEM
    H = double_pendulum_hamiltonian(m1=m1, m2=m2, l1=l1, l2=l2, g=g)
    # Initial energy
    E_start = float(H(q, p).numpy())
    print(f"\nInitial Energy: {E_start:.6f}")

    # EVOLVE
    q, p, history = simulate_hamiltonian(H, q, p, dt=dt, steps=steps, record_every=10)

    # Final state
    E_end = float(H(q, p).numpy())
    E_drift = abs(E_end - E_start) / abs(E_start)

    print(f"Final Energy:   {E_end:.6f}")
    print(f"Energy Drift:   {E_drift:.2e}")

    # Generate viewer
    history_q = [h[0].tolist() for h in history]
    generate_viewer(history_q, l1, l2)

    return E_start, E_end, E_drift


def generate_viewer(history_q, l1, l2):
    """Generate HTML visualization."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Double Pendulum - TinyPhysics</title>
    <style>
        body {{
            font-family: monospace;
            background: #111;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        canvas {{ border: 1px solid #333; background: #000; }}
        .info {{
            background: #1a1a1a;
            padding: 15px;
            border-radius: 5px;
            margin: 10px;
            max-width: 500px;
        }}
        code {{ color: #4f4; }}
    </style>
</head>
<body>
    <h1>Double Pendulum - Physics Compiled from Energy</h1>
    <div class="info">
        <p>Define ONLY the Hamiltonian:</p>
        <code>H = 0.5 * p^T * M(q)^-1 * p + V(q)</code>
        <p>Non-separable system - autograd handles complex derivatives!</p>
    </div>
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

            // Rods
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(x1, y1);
            ctx.lineTo(x2, y2);
            ctx.stroke();

            // Masses
            ctx.fillStyle = '#f00';
            ctx.beginPath();
            ctx.arc(x1, y1, 8, 0, 2*Math.PI);
            ctx.fill();

            ctx.fillStyle = '#0f0';
            ctx.beginPath();
            ctx.arc(x2, y2, 8, 0, 2*Math.PI);
            ctx.fill();

            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}

        draw();
    </script>
</body>
</html>
    """

    filepath = 'examples/double_pendulum_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


if __name__ == "__main__":
    run_simulation()
