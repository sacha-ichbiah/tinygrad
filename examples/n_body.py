"""
N-Body Gravity - Multiple gravitationally interacting bodies (Level 1.4)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. Use BROADCASTING for efficient O(N²) pairwise interactions
    3. Autograd derives all N² forces automatically!

Hamiltonian: H = Sum_i(|p_i|²/2m_i) - Sum_{i<j}(G*m_i*m_j/|r_i - r_j|)

This demonstrates the "compiler" + "broadcast" primitives from the roadmap.
"""

import argparse
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import simulate_hamiltonian
import json
import os
import time


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def nbody_hamiltonian(masses: np.ndarray, G: float = 1.0, softening: float = 0.01,
                      use_soa: bool = False, block_size: int | None = None):
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
    def H(q, p):
        m = Tensor(masses, device=q.device)  # (N,)
        m_i = m.reshape(N, 1)
        m_j = m.reshape(1, N)
        mass_prod = m_i * m_j
        eps_sq = softening ** 2
        # Kinetic energy: T = Sum_i |p_i|²/(2*m_i)
        # p shape: (N, D), m shape: (N, 1)
        if p.shape[1] == 2:
            p2 = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1]
            T = (p2 / (2 * m)).sum()
        elif p.shape[1] == 3:
            p2 = p[:, 0] * p[:, 0] + p[:, 1] * p[:, 1] + p[:, 2] * p[:, 2]
            T = (p2 / (2 * m)).sum()
        else:
            T = ((p * p).sum(axis=1) / (2 * m)).sum()

        # Potential energy using BROADCASTING (with optional layout/tiling)
        if block_size is not None and block_size > 0 and block_size < N:
            V_acc = Tensor.zeros((), device=q.device, dtype=q.dtype)
            if use_soa:
                qT = q.transpose(0, 1)
            else:
                qx = q[:, 0]
                qy = q[:, 1]
                qz = q[:, 2] if q.shape[1] > 2 else None
            for i0 in range(0, N, block_size):
                i1 = min(i0 + block_size, N)
                qi = q[i0:i1]
                mi = m[i0:i1]
                if use_soa:
                    qiT = qi.transpose(0, 1)
                    if qiT.shape[0] <= 3:
                        diff = qiT.unsqueeze(2) - qT.unsqueeze(1)
                        dist_sq = (diff * diff).sum(axis=0) + eps_sq
                    else:
                        dx = qiT[0].unsqueeze(1) - qT[0].unsqueeze(0)
                        dy = qiT[1].unsqueeze(1) - qT[1].unsqueeze(0)
                        if qiT.shape[0] == 2:
                            dist_sq = dx * dx + dy * dy + eps_sq
                        else:
                            dz = qiT[2].unsqueeze(1) - qT[2].unsqueeze(0)
                            dist_sq = dx * dx + dy * dy + dz * dz + eps_sq
                else:
                    dx = qi[:, 0].unsqueeze(1) - qx.unsqueeze(0)
                    dy = qi[:, 1].unsqueeze(1) - qy.unsqueeze(0)
                    if qi.shape[1] == 2:
                        dist_sq = dx * dx + dy * dy + eps_sq
                    else:
                        dz = qi[:, 2].unsqueeze(1) - qz.unsqueeze(0)
                        dist_sq = dx * dx + dy * dy + dz * dz + eps_sq
                inv_dist = dist_sq.rsqrt()
                V_acc = V_acc + (mi.reshape(i1 - i0, 1) * m_j * inv_dist).sum()
            V = -0.5 * G * V_acc
        else:
            if use_soa:
                qT = q.transpose(0, 1)
                if qT.shape[0] <= 3:
                    diff = qT.unsqueeze(2) - qT.unsqueeze(1)
                    dist_sq = (diff * diff).sum(axis=0) + eps_sq
                else:
                    qx = qT[0]
                    qy = qT[1]
                    dx = qx.unsqueeze(1) - qx.unsqueeze(0)
                    dy = qy.unsqueeze(1) - qy.unsqueeze(0)
                    if qT.shape[0] == 2:
                        dist_sq = dx * dx + dy * dy + eps_sq
                    else:
                        dz = qT[2]
                        dist_sq = dx * dx + dy * dy + (dz.unsqueeze(1) - dz.unsqueeze(0)) ** 2 + eps_sq
            else:
                qx = q[:, 0]
                qy = q[:, 1]
                dx = qx.unsqueeze(1) - qx.unsqueeze(0)
                dy = qy.unsqueeze(1) - qy.unsqueeze(0)
                if q.shape[1] == 2:
                    dist_sq = dx * dx + dy * dy + eps_sq
                else:
                    dz = q[:, 2]
                    dist_sq = dx * dx + dy * dy + (dz.unsqueeze(1) - dz.unsqueeze(0)) ** 2 + eps_sq
            inv_dist = dist_sq.rsqrt()
            # Potential: V = -0.5 * G * Sum_{i,j} m_i * m_j / |r_ij|
            V = -0.5 * G * (mass_prod * inv_dist).sum()

        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(N=5, integrator="leapfrog", dt=0.001, steps=10000, config="random",
                   use_scan=True, unroll_steps=4, scan_tune=False, record_every=10,
                   render=False, diagnostics=False, benchmark=False, warmup_steps=0,
                   bench_steps=None, bench_repeats=3, bench_profile=False, bench_kernel_timing=False,
                   use_soa: bool | None = False, block_size=None, block_auto=True):
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
    if use_soa is None:
        use_soa = N >= 32 and q_init.shape[1] <= 4
    if block_auto and block_size is None and N >= 64:
        block_size = 32 if N < 128 else 64

    print("=" * 60)
    print(f"N-BODY GRAVITY - TinyPhysics Compiler Approach (N={N})")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H = Sum_i |p_i|²/2m_i - Sum_{{i<j}} G*m_i*m_j/|r_ij|")
    print(f"\nBROADCAST computes all {N}² pairwise interactions.")
    print(f"AUTOGRAD derives all {N}² forces automatically!")
    print(f"\nIntegrator: auto")
    print(f"Config: {config}, dt={dt}, steps={steps}")

    # CREATE THE HAMILTONIAN SYSTEM
    H = nbody_hamiltonian(masses, G=G, use_soa=use_soa, block_size=block_size)
    system = None
    if unroll_steps <= 0 and use_scan and integrator == "leapfrog":
        for cand in (16, 8, 4, 2, 1):
            if steps % cand == 0:
                unroll_steps = cand
                break

    # Initial energy and momentum
    if diagnostics:
        E_start = float(H(q, p).numpy())
        P_start = p.numpy().sum(axis=0)
        print(f"\nInitial Energy: {E_start:.6f}")
        print(f"Initial Momentum: [{P_start[0]:.6f}, {P_start[1]:.6f}]")

    # EVOLVE / BENCHMARK
    def pick_unroll(steps_run: int) -> int:
        if unroll_steps > 0: return unroll_steps
        for cand in (16, 8, 4, 2, 1):
            if steps_run % cand == 0:
                return cand
        return 1

    def evolve_steps(q, p, steps_run, record_every_run):
        history = None
        if use_scan and integrator == "leapfrog":
            local_unroll = pick_unroll(steps_run)
            if steps_run % local_unroll != 0:
                local_unroll = 1
            try:
                if system is None:
                    from tinygrad.physics import HamiltonianSystem
                    system = HamiltonianSystem(H, integrator="leapfrog")
                q, p, history = system.evolve_scan_kernel(
                    q, p, dt=dt, steps=steps_run, coupled=True, coupled_fused=True,
                    unroll_steps=local_unroll, scan_tune=scan_tune,
                )
            except Exception as e:
                print(f"Scan kernel failed ({e}); falling back to evolve.")
                history = None
        if history is None:
            q, p, history = simulate_hamiltonian(H, q, p, dt=dt, steps=steps_run, record_every=record_every_run)
        return q, p, history

    from tinygrad.helpers import GlobalCounters
    if benchmark:
        if bench_steps is None: bench_steps = steps
        if warmup_steps > 0:
            q, p, _ = evolve_steps(q, p, warmup_steps, warmup_steps)
        times = []
        best = None
        kernel_times = []
        for _ in range(max(1, bench_repeats)):
            if bench_profile: GlobalCounters.reset()
            start_time = time.perf_counter()
            q, p, _ = evolve_steps(q, p, bench_steps, bench_steps)
            elapsed = time.perf_counter() - start_time
            times.append(elapsed)
            if best is None or elapsed < best: best = elapsed
            if bench_profile:
                kernel_times.append(GlobalCounters.time_sum_s)
                print(f"Kernel count: {GlobalCounters.kernel_count}, kernel time: {GlobalCounters.time_sum_s:.6f}s")
            if bench_kernel_timing:
                print(f"Elapsed: {elapsed:.6f}s")
        elapsed = best if best is not None else times[-1]
        times_sorted = sorted(times)
        avg_time = sum(times) / len(times)
        med_time = times_sorted[len(times_sorted) // 2]
        steps_s = bench_steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance (best): {steps_s:,.1f} steps/s")
        print(f"Performance (avg): {(bench_steps / avg_time):,.1f} steps/s")
        print(f"Performance (median): {(bench_steps / med_time):,.1f} steps/s")
        if kernel_times:
            avg_kernel_time = sum(kernel_times) / len(kernel_times)
            print(f"Kernel time (avg): {avg_kernel_time:.6f}s")
            if GlobalCounters.kernel_count > 0 and avg_kernel_time > 0:
                print(f"Avg time/kernel: {(avg_kernel_time / GlobalCounters.kernel_count):.9f}s")
        history = []
    else:
        start_time = time.perf_counter()
        history = None
        try:
            q, p, history = evolve_steps(q, p, steps, record_every)
        except Exception as e:
            print(f"Compiler failed ({e}); retrying on PYTHON backend.")
            q = Tensor(q_init, requires_grad=False, device="PYTHON")
            p = Tensor(p_init, requires_grad=False, device="PYTHON")
            system = HamiltonianSystem(H, integrator=integrator)
            q, p, history = simulate_hamiltonian(H, q, p, dt=dt, steps=steps, record_every=record_every)
        elapsed = time.perf_counter() - start_time
        steps_s = steps / elapsed if elapsed > 0 else float("inf")
        print(f"Performance: {steps_s:,.1f} steps/s")

    # Final state
    if diagnostics:
        E_end = float(H(q, p).numpy())
        P_end = p.numpy().sum(axis=0)
        E_drift = abs(E_end - E_start) / abs(E_start)
        print(f"\nFinal Energy:   {E_end:.6f}")
        print(f"Final Momentum: [{P_end[0]:.6f}, {P_end[1]:.6f}]")
        print(f"Energy Drift:   {E_drift:.2e}")

    # Generate viewer
    if render and history:
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["random", "solar"], default="random")
    parser.add_argument("--solar", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--dt", type=float, default=0.001)
    parser.add_argument("--N", type=int, default=5)
    parser.add_argument("--unroll-steps", type=int, default=4)
    parser.add_argument("--soa", choices=["auto", "on", "off"], default="off")
    parser.add_argument("--block-size", type=int, default=0)
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--bench-steps", type=int, default=None)
    parser.add_argument("--bench-repeats", type=int, default=3)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--kernel-timing", action="store_true")
    args = parser.parse_args()

    if args.compare:
        compare_configs()
    else:
        config = "solar" if args.solar else args.config
        use_soa = None if args.soa == "auto" else args.soa == "on"
        block_auto = args.block_size == 0
        run_simulation(
            N=args.N,
            config=config,
            dt=args.dt,
            steps=args.steps,
            unroll_steps=args.unroll_steps,
            benchmark=args.benchmark,
            warmup_steps=args.warmup_steps,
            bench_steps=args.bench_steps,
            bench_repeats=args.bench_repeats,
            bench_profile=args.profile,
            bench_kernel_timing=args.kernel_timing,
            use_soa=use_soa,
            block_size=args.block_size if args.block_size > 0 else None,
            block_auto=block_auto,
        )
