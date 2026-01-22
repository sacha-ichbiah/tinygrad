"""
N-Body Gravity - Multiple gravitationally interacting bodies (Level 1.4)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. Use BROADCASTING for efficient O(N²) pairwise interactions
    3. Autograd derives all N² forces automatically!

Hamiltonian: H = Sum_i(|p_i|²/2m_i) - Sum_{i<j}(G*m_i*m_j/|r_i - r_j|)

This demonstrates the "compiler" + "broadcast" primitives from the roadmap.
"""

import numpy as np
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem
import json
import os
import time


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def nbody_hamiltonian(masses: np.ndarray, G: float = 1.0, softening: float = 0.01):
    """
    Returns the Hamiltonian for N gravitationally interacting bodies.

    State:
        q: positions, shape (N, D) where D is dimensionality
        p: momenta, shape (N, D)

    H = T + V where:
        T = Sum_i |p_i|²/(2*m_i)                    (kinetic energy)
        V = -Sum_{i<j} G*m_i*m_j/|r_i - r_j|       (gravitational potential)

    The BROADCAST primitive computes all N² pairwise interactions efficiently.
    Autograd then derives all N² forces automatically!
    """
    N = len(masses)
    m = Tensor(masses)  # (N,)
    m_i = m.reshape(N, 1)
    m_j = m.reshape(1, N)
    mass_prod = m_i * m_j
    eps_sq = softening ** 2

    def H(q, p):
        # Kinetic energy: T = Sum_i |p_i|²/(2*m_i)
        # p shape: (N, D), m shape: (N, 1)
        T = ((p * p).sum(axis=1) / (2 * m)).sum()

        # Potential energy using BROADCASTING
        # q shape: (N, D)
        # q.unsqueeze(0): (1, N, D) - body j
        # q.unsqueeze(1): (N, 1, D) - body i
        # diff[i,j] = q[i] - q[j], shape (N, N, D)
        diff = q.unsqueeze(1) - q.unsqueeze(0)

        # Distance matrix |r_ij| with softening
        # dist_sq[i,j] = |q[i] - q[j]|², shape (N, N)
        dist_sq = (diff * diff).sum(axis=2) + eps_sq

        # inv_dist[i,j] = 1/|r_ij|, shape (N, N)
        inv_dist = dist_sq.rsqrt()

        # Mass product matrix: m_i * m_j, shape (N, N)
        # Potential: V = -0.5 * G * Sum_{i,j} m_i * m_j / |r_ij|
        # Factor 0.5 because we sum over all pairs twice (i,j) and (j,i)
        # Note: diagonal (i=i) gives -G*m_i²/eps which is constant, doesn't affect dynamics
        V = -0.5 * G * (mass_prod * inv_dist).sum()

        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(N=5, integrator="leapfrog", dt=0.001, steps=10000, config="random",
                   use_scan=True, unroll_steps=4, scan_tune=False, record_every=10,
                   render=False, diagnostics=False, fast_force=True):
    """
    Simulate N-body gravitational dynamics.

    The user defines ONLY the Hamiltonian.
    Broadcasting handles O(N²) interactions efficiently.
    Autograd derives all forces automatically.
    """
    G = 1.0
    np.random.seed(42)

    if config == "random":
        # Random cloud of bodies
        q_init = np.random.randn(N, 2).astype(np.float32) * 2.0
        p_init = np.random.randn(N, 2).astype(np.float32) * 0.5
        p_init -= p_init.mean(axis=0)  # Zero total momentum
        masses = np.random.uniform(0.5, 2.0, N).astype(np.float32)
    elif config == "solar":
        # Simple solar system: Sun + planets
        N = 4
        masses = np.array([10.0, 0.1, 0.1, 0.1], dtype=np.float32)  # Sun much heavier
        q_init = np.array([
            [0.0, 0.0],      # Sun at center
            [1.0, 0.0],      # Planet 1
            [0.0, 1.5],      # Planet 2
            [-2.0, 0.0],     # Planet 3
        ], dtype=np.float32)
        # Circular orbit velocities: v = sqrt(GM/r)
        p_init = np.array([
            [0.0, 0.0],
            [0.0, np.sqrt(G * masses[0] / 1.0) * masses[1]],
            [-np.sqrt(G * masses[0] / 1.5) * masses[2], 0.0],
            [0.0, -np.sqrt(G * masses[0] / 2.0) * masses[3]],
        ], dtype=np.float32)

    q = Tensor(q_init, requires_grad=False)
    p = Tensor(p_init, requires_grad=False)

    print("=" * 60)
    print(f"N-BODY GRAVITY - TinyPhysics Compiler Approach (N={N})")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H = Sum_i |p_i|²/2m_i - Sum_{{i<j}} G*m_i*m_j/|r_ij|")
    print(f"\nBROADCAST computes all {N}² pairwise interactions.")
    if fast_force and integrator == "leapfrog":
        print(f"FAST FORCE path: direct gravity forces (no autograd).")
    else:
        print(f"AUTOGRAD derives all {N}² forces automatically!")
    print(f"\nIntegrator: {integrator}")
    print(f"Config: {config}, dt={dt}, steps={steps}")

    # CREATE THE HAMILTONIAN SYSTEM
    H = nbody_hamiltonian(masses, G=G)
    system = HamiltonianSystem(H, integrator=integrator)

    # Initial energy and momentum
    if diagnostics:
        E_start = system.energy(q, p)
        P_start = p.numpy().sum(axis=0)
        print(f"\nInitial Energy: {E_start:.6f}")
        print(f"Initial Momentum: [{P_start[0]:.6f}, {P_start[1]:.6f}]")

    # EVOLVE
    start_time = time.perf_counter()
    history = None
    if fast_force and integrator == "leapfrog":
        m = Tensor(masses)
        m_i = m.reshape(N, 1, 1)
        m_j = m.reshape(1, N, 1)
        eps_sq = 0.01 * 0.01
        def forces(q_in: Tensor) -> Tensor:
            diff = q_in.unsqueeze(1) - q_in.unsqueeze(0)
            dist_sq = (diff * diff).sum(axis=2) + eps_sq
            inv_dist3 = dist_sq.rsqrt() / dist_sq
            return (-G * m_i * (m_j * diff * inv_dist3.unsqueeze(2)).sum(axis=1))
        def step(q_in: Tensor, p_in: Tensor):
            f = forces(q_in)
            p_half = p_in + (0.5 * dt) * f
            q_new = q_in + dt * (p_half / m.reshape(N, 1))
            f_new = forces(q_new)
            p_new = p_half + (0.5 * dt) * f_new
            return q_new, p_new
        step_jit = TinyJit(step)
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps")
        if render:
            history = []
        step_count = 0
        for _ in range(steps // unroll_steps):
            for _ in range(unroll_steps):
                q, p = step_jit(q, p)
                if render and (step_count % record_every == 0):
                    history.append((q.numpy().copy(), p.numpy().copy(), step_count * dt))
                step_count += 1
        if render and not history:
            history = [(q.numpy().copy(), p.numpy().copy(), steps * dt)]
        else:
            q.numpy()
            p.numpy()
    elif use_scan and integrator == "leapfrog":
        if steps % unroll_steps != 0:
            raise ValueError("steps must be divisible by unroll_steps for scan")
        try:
            q, p, history = system.evolve_scan_kernel(
                q, p, dt=dt, steps=steps, coupled=True, coupled_fused=True,
                unroll_steps=unroll_steps, scan_tune=scan_tune,
            )
        except Exception as e:
            print(f"Scan kernel failed ({e}); falling back to evolve.")
            history = None
    if history is None:
        q, p, history = system.evolve(q, p, dt=dt, steps=steps, record_every=record_every)
    elapsed = time.perf_counter() - start_time
    steps_s = steps / elapsed if elapsed > 0 else float("inf")
    print(f"Performance: {steps_s:,.1f} steps/s")

    # Final state
    if diagnostics:
        E_end = system.energy(q, p)
        P_end = p.numpy().sum(axis=0)
        E_drift = abs(E_end - E_start) / abs(E_start)
        print(f"\nFinal Energy:   {E_end:.6f}")
        print(f"Final Momentum: [{P_end[0]:.6f}, {P_end[1]:.6f}]")
        print(f"Energy Drift:   {E_drift:.2e}")

    # Generate viewer
    if render:
        history_q = [h[0].tolist() for h in history]
        generate_viewer(history_q, N, masses)

    if diagnostics:
        return E_start, E_end, E_drift
    return None, None, None


def generate_viewer(history_q, N, masses):
    """Generate HTML visualization."""
    # Normalize masses for display
    max_mass = max(masses)
    sizes = [max(4, int(12 * m / max_mass)) for m in masses]

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>N-Body Gravity - TinyPhysics</title>
    <style>
        body {{
            font-family: monospace;
            background: #000;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        canvas {{ border: 1px solid #333; }}
        .info {{
            background: #111;
            padding: 15px;
            margin: 10px;
            border-radius: 5px;
            max-width: 500px;
        }}
        code {{ color: #4f4; }}
        .controls {{
            display: flex;
            gap: 16px;
            align-items: center;
            margin: 10px;
        }}
        input[type="range"] {{ width: 220px; }}
    </style>
</head>
<body>
    <h1>N-Body Gravity - Physics Compiled from Energy</h1>
    <div class="info">
        <p>Define ONLY the Hamiltonian:</p>
        <code>H = T - G*Sum m_i*m_j/|r_ij|</code>
        <p>Broadcasting handles O(N²) interactions.</p>
        <p>Autograd derives all forces!</p>
    </div>
    <div class="controls">
        <label>Speed: <span id="speedLabel">1.0×</span></label>
        <input id="speedSlider" type="range" min="0.25" max="8" step="0.25" value="1">
    </div>
    <canvas id="simCanvas" width="800" height="600"></canvas>
    <script>
        const history = {json.dumps(history_q)};
        const N = {N};
        const sizes = {json.dumps(sizes)};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const scale = 100;

        const colors = ['#ffcc00', '#00ccff', '#ff6666', '#66ff66', '#ff66ff', '#66ffff'];

        let frame = 0;
        let speed = 1.0;
        const speedSlider = document.getElementById('speedSlider');
        const speedLabel = document.getElementById('speedLabel');
        speedSlider.addEventListener('input', (e) => {{
            speed = parseFloat(e.target.value);
            speedLabel.textContent = `${{speed.toFixed(2)}}×`;
        }});

        function draw() {{
            ctx.fillStyle = 'rgba(0, 0, 0, 0.1)';
            ctx.fillRect(0, 0, width, height);

            const positions = history[Math.floor(frame)];

            for (let i = 0; i < N; i++) {{
                const x = positions[i][0] * scale + width/2;
                const y = positions[i][1] * scale + height/2;

                ctx.beginPath();
                ctx.arc(x, y, sizes[i], 0, 2*Math.PI);
                ctx.fillStyle = colors[i % colors.length];
                ctx.fill();
            }}

            frame += speed;
            if (frame >= history.length) frame = frame % history.length;
            requestAnimationFrame(draw);
        }}

        draw();
    </script>
</body>
</html>
    """

    filepath = 'examples/n_body_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


def compare_configs():
    """Compare different configurations."""
    print("=" * 60)
    print("N-BODY CONFIGURATIONS")
    print("=" * 60)

    for config in ["random", "solar"]:
        print(f"\n{'='*60}")
        run_simulation(config=config, dt=0.001, steps=2000)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--solar":
        run_simulation(config="solar", dt=0.001, steps=5000)
    elif len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_configs()
    else:
        run_simulation(N=5, config="random", dt=0.001, steps=10000)
