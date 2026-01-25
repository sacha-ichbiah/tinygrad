"""
Quantum Tunneling - Particle Through a Barrier (Level 4.2)

THE TINYPHYSICS WAY:
    1. Define Hamiltonian: H = p²/2m + V(x)
    2. Solve Schrödinger equation with split-operator method
    3. Watch the quantum magic - particles tunnel through walls!

A wavepacket with energy E < V₀ approaches a potential barrier.
Classically: 100% reflection
Quantum mechanically: partial transmission (tunneling)!

Transmission coefficient for rectangular barrier:
    T ≈ exp(-2κa) where κ = √(2m(V₀-E))/ℏ, a = barrier width

Uses the TinyPhysics structure-preserving compiler (blueprint_v3).
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics import QuantumSplitOperator1D, gaussian_wavepacket
import json
import os


def rectangular_barrier(x: np.ndarray, x0: float, width: float, V0: float) -> np.ndarray:
    """Rectangular potential barrier centered at x0 with given width and height V0."""
    return np.where(np.abs(x - x0) < width / 2, V0, 0.0)


def compute_norm(psi: Tensor, dx: float) -> float:
    """Compute norm: integral of |psi|^2 dx."""
    prob = psi[..., 0] * psi[..., 0] + psi[..., 1] * psi[..., 1]
    return float((prob.sum() * dx).numpy())


def compute_probability_density(psi: Tensor) -> Tensor:
    """Compute |psi|^2."""
    return psi[..., 0] * psi[..., 0] + psi[..., 1] * psi[..., 1]


def compute_energy(psi: Tensor, x: Tensor, V: Tensor, dx: float, hbar: float, m: float) -> float:
    """Compute expectation value of energy <H> = <T> + <V>.

    Uses momentum-space representation for kinetic energy.
    Note: This is approximate due to FFT normalization conventions.
    """
    from tinyphysics.structures.commutator import fftfreq
    from tinygrad.fft import fft1d
    import math

    N = x.shape[0]
    k = fftfreq(N, dx, x.device, x.dtype) * (2.0 * math.pi)
    k2 = k * k

    # FFT of psi - note: fft1d uses standard normalization
    psi_k = fft1d(psi)
    # |psi_k|^2 needs to be normalized by N^2 for Parseval's theorem
    psi_k_abs2 = psi_k[..., 0] * psi_k[..., 0] + psi_k[..., 1] * psi_k[..., 1]

    # <T> = (1/N) * sum_k |psi_k|^2 * (hbar*k)^2 / (2m) * dx
    # The factor of dx comes from the discrete->continuous conversion
    T_expect = float(((psi_k_abs2 * k2).sum() / (N * N) * (hbar * hbar) / (2 * m) * dx).numpy())

    # <V> = integral |psi|^2 * V(x) dx
    prob = compute_probability_density(psi)
    V_expect = float((prob * V).sum().numpy() * dx)

    return T_expect + V_expect


def run_simulation(N: int = 1024, L: float = 100.0, sigma: float = 3.0,
                   k0: float = 1.5, V0: float = 2.0, barrier_width: float = 1.0,
                   dt: float = 0.01, steps: int = 4000):
    """
    Simulate quantum tunneling through a rectangular barrier.

    Uses TinyPhysics QuantumSplitOperator1D following blueprint_v3.

    Args:
        N: Number of grid points
        L: Domain size [-L/2, L/2]
        sigma: Initial wavepacket width
        k0: Initial momentum (wavenumber)
        V0: Barrier height
        barrier_width: Width of the barrier
        dt: Time step
        steps: Number of time steps
    """
    hbar, m = 1.0, 1.0
    dx = L / N

    # Initial kinetic energy
    E_kinetic = hbar**2 * k0**2 / (2 * m)

    print("=" * 60)
    print("QUANTUM TUNNELING - TinyPhysics (Blueprint v3)")
    print("=" * 60)
    print(f"\nHamiltonian: H = p²/2m + V(x)")
    print(f"Potential: Rectangular barrier at x=0")
    print(f"Structure: QuantumSplitOperator1D (unitary evolution)")
    print(f"\nParameters:")
    print(f"  Grid: {N} points, Domain: [-{L/2:.0f}, {L/2:.0f}]")
    print(f"  Wavepacket: σ={sigma}, k₀={k0}")
    print(f"  Initial kinetic energy E = ℏ²k₀²/2m = {E_kinetic:.3f}")
    print(f"  Barrier: V₀={V0}, width={barrier_width}")
    print(f"  E/V₀ = {E_kinetic/V0:.3f}")

    if E_kinetic < V0:
        print(f"\n  E < V₀: CLASSICALLY FORBIDDEN (tunneling regime)")
        kappa = np.sqrt(2 * m * (V0 - E_kinetic)) / hbar
        T_theory = np.exp(-2 * kappa * barrier_width)
        print(f"  Theoretical T ≈ exp(-2κa) ≈ {T_theory:.4f}")
    else:
        print(f"\n  E > V₀: Classical transmission allowed")
        T_theory = None

    # Create grid and potential (blueprint pattern)
    x_np = np.linspace(-L/2, L/2, N, endpoint=False).astype(np.float32)
    x = Tensor(x_np)
    V_np = rectangular_barrier(x_np, x0=0.0, width=barrier_width, V0=V0).astype(np.float32)
    V = Tensor(V_np)

    # Create quantum solver using TinyPhysics API (blueprint_v3 Section 6.3)
    solver = QuantumSplitOperator1D(x, dt=dt, mass=m, hbar=hbar, V=V)

    # Initial wavepacket using tinyphysics gaussian_wavepacket
    x_start = -25.0
    psi = gaussian_wavepacket(x, x0=x_start, k0=k0, sigma=sigma)

    # Initial measurements
    norm_start = compute_norm(psi, dx)
    energy_start = compute_energy(psi, x, V, dx, hbar, m)

    print(f"\nInitial state:")
    print(f"  Position: x₀ = {x_start}")
    print(f"  Norm: {norm_start:.6f}")
    print(f"  Total energy: {energy_start:.4f}")

    # Evolve using JIT-compiled unrolled steps (blueprint pattern)
    print(f"\nEvolving for {steps} steps (dt={dt})...")
    record_every = max(1, steps // 200)

    # Use compiler's evolve() method for JIT-compiled batched evolution
    psi, prob_history = solver.evolve(psi, steps=steps, record_every=record_every)

    # Compute diagnostics only at end (for conservation checking)
    prob_final = prob_history[-1]
    norm_final = compute_norm(psi, dx)
    energy_final = compute_energy(psi, x, V, dx, hbar, m)

    # Calculate transmission and reflection coefficients
    transmitted_region = x_np > barrier_width / 2 + 5
    reflected_region = x_np < -barrier_width / 2 - 5
    T_measured = prob_final[transmitted_region].sum() * dx
    R_measured = prob_final[reflected_region].sum() * dx

    print(f"\nFinal state:")
    print(f"  Norm: {norm_final:.6f}")
    print(f"  Energy: {energy_final:.4f}")
    print(f"  Norm drift: {abs(norm_final - norm_start):.2e}")
    print(f"  Energy drift: {abs(energy_final - energy_start):.2e}")

    print(f"\nTunneling results:")
    print(f"  Transmission T: {T_measured:.4f}")
    print(f"  Reflection R: {R_measured:.4f}")
    print(f"  T + R = {T_measured + R_measured:.4f} (should be ≈1)")
    if T_theory is not None:
        print(f"  Theoretical T: {T_theory:.4f}")

    # Generate viewer (pass prob history and conservation values)
    generate_viewer(prob_history, V_np, x_np, V0, barrier_width, E_kinetic, T_theory,
                    hbar, m, dt, L, N, norm_start, norm_final, energy_start, energy_final)

    return T_measured, R_measured, norm_final - norm_start


def generate_viewer(prob_history, V_np, x_np, V0, barrier_width, E_kinetic, T_theory, hbar, m, dt, L, N,
                    norm_start, norm_final, energy_start, energy_final):
    """Generate HTML visualization."""
    x_data = x_np.tolist()
    V_data = V_np.tolist()
    prob_data = [p.tolist() for p in prob_history]
    # For conservation display, use start and final values
    norm_start_val = norm_start
    norm_final_val = norm_final
    energy_start_val = energy_start
    energy_final_val = energy_final
    dx = L / N

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Tunneling - TinyPhysics</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #0a0a1a 0%, #1a1a2e 100%);
            color: #eee;
            margin: 0;
            min-height: 100vh;
            padding: 20px;
        }}
        h1 {{ text-align: center; color: #fff; margin-bottom: 10px; font-weight: 300; }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; font-size: 14px; }}
        .main-container {{
            display: flex; gap: 30px; justify-content: center; flex-wrap: wrap;
            max-width: 1400px; margin: 0 auto;
        }}
        .wave-section {{ display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border-radius: 10px; box-shadow: 0 10px 40px rgba(0,0,0,0.5); }}
        .controls {{
            margin-top: 15px; display: flex; gap: 10px; align-items: center;
        }}
        button {{
            background: #f4a; border: none; color: #fff; padding: 10px 20px;
            border-radius: 5px; cursor: pointer; font-weight: bold; transition: all 0.2s;
        }}
        button:hover {{ background: #f5b; transform: scale(1.05); }}
        .speed-control {{ display: flex; align-items: center; gap: 8px; color: #888; }}
        input[type="range"] {{ width: 100px; accent-color: #f4a; }}
        .stats-panel {{
            background: rgba(20, 20, 40, 0.9); border-radius: 10px;
            padding: 20px; min-width: 320px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .stat-group {{ margin-bottom: 20px; }}
        .stat-group h3 {{
            color: #f4a; margin: 0 0 12px 0; font-size: 14px;
            text-transform: uppercase; letter-spacing: 1px;
        }}
        .stat-row {{
            display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #fff; font-family: monospace; }}
        .stat-value.good {{ color: #4f4; }}
        .stat-value.highlight {{ color: #ff0; font-weight: bold; }}
        .quantum-info {{
            background: rgba(255, 68, 170, 0.1); border-left: 3px solid #f4a;
            padding: 10px 15px; margin: 10px 0; border-radius: 0 5px 5px 0;
        }}
        .quantum-info .title {{ color: #f4a; font-weight: bold; font-size: 12px; }}
        .quantum-info .desc {{ color: #aaa; font-size: 12px; margin-top: 4px; }}
        .formula {{
            font-family: monospace; background: rgba(0,0,0,0.3);
            padding: 8px 12px; border-radius: 4px; color: #4f4;
            display: inline-block; margin: 5px 0;
        }}
        .bar-container {{ height: 20px; background: #222; border-radius: 3px; margin: 5px 0; overflow: hidden; display: flex; }}
        .bar-t {{ background: linear-gradient(90deg, #4f4, #2a2); height: 100%; transition: width 0.3s; }}
        .bar-r {{ background: linear-gradient(90deg, #f44, #a22); height: 100%; transition: width 0.3s; }}
    </style>
</head>
<body>
    <h1>Quantum Tunneling</h1>
    <div class="subtitle">Wavepacket passing through a classically forbidden barrier (TinyPhysics Blueprint v3)</div>

    <div class="main-container">
        <div class="wave-section">
            <canvas id="waveCanvas" width="700" height="400"></canvas>
            <div class="controls">
                <button id="playBtn">Pause</button>
                <button id="resetBtn">Reset</button>
                <div class="speed-control">
                    <span>Speed:</span>
                    <input type="range" id="speedSlider" min="1" max="10" value="3">
                    <span id="speedLabel">3x</span>
                </div>
            </div>
        </div>

        <div class="stats-panel">
            <div class="stat-group">
                <h3>Transmission / Reflection</h3>
                <div class="stat-row">
                    <span class="stat-label">Transmitted T</span>
                    <span class="stat-value highlight" id="tValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Reflected R</span>
                    <span class="stat-value" id="rValue">-</span>
                </div>
                <div class="bar-container">
                    <div class="bar-t" id="tBar" style="width: 0%"></div>
                    <div class="bar-r" id="rBar" style="width: 0%"></div>
                </div>
                <div class="stat-row">
                    <span class="stat-label">T + R</span>
                    <span class="stat-value good" id="sumValue">-</span>
                </div>
                {"<div class='stat-row'><span class='stat-label'>Theory T</span><span class='stat-value'>" + f"{T_theory:.4f}" + "</span></div>" if T_theory else ""}
            </div>

            <div class="stat-group">
                <h3>Conservation</h3>
                <div class="stat-row">
                    <span class="stat-label">Norm (initial)</span>
                    <span class="stat-value good">{norm_start_val:.6f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Norm (final)</span>
                    <span class="stat-value good">{norm_final_val:.6f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Norm drift</span>
                    <span class="stat-value good">{abs(norm_final_val - norm_start_val):.2e}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Energy drift</span>
                    <span class="stat-value good">{abs(energy_final_val - energy_start_val):.2e}</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Parameters</h3>
                <div class="stat-row">
                    <span class="stat-label">Kinetic energy E</span>
                    <span class="stat-value">{E_kinetic:.3f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Barrier V0</span>
                    <span class="stat-value">{V0:.3f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">E / V0</span>
                    <span class="stat-value">{E_kinetic/V0:.3f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Barrier width</span>
                    <span class="stat-value">{barrier_width}</span>
                </div>
            </div>

            <div class="quantum-info">
                <div class="title">Quantum Tunneling</div>
                <div class="desc">Classically, E &lt; V0 means 100% reflection. Quantum mechanics allows the wavefunction to penetrate the barrier!</div>
                <div class="formula">T = e^(-2*kappa*a), kappa = sqrt(2m(V0-E))/hbar</div>
            </div>
        </div>
    </div>

    <script>
        const x = {json.dumps(x_data)};
        const V = {json.dumps(V_data)};
        const probHistory = {json.dumps(prob_data)};
        const V0 = {V0};
        const barrierWidth = {barrier_width};
        const totalFrames = probHistory.length;

        let frame = 0, playing = true, speed = 3;
        const canvas = document.getElementById('waveCanvas');
        const ctx = canvas.getContext('2d');
        const dx = {dx};

        document.getElementById('playBtn').onclick = () => {{
            playing = !playing;
            document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
        }};
        document.getElementById('resetBtn').onclick = () => {{ frame = 0; }};
        document.getElementById('speedSlider').oninput = (e) => {{
            speed = parseInt(e.target.value);
            document.getElementById('speedLabel').textContent = speed + 'x';
        }};

        function drawWavefunction() {{
            const prob = probHistory[frame];
            const maxProb = Math.max(...probHistory.flat()) * 1.1;
            const maxV = V0 * 1.2;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Potential barrier (filled)
            ctx.fillStyle = 'rgba(255, 100, 100, 0.3)';
            ctx.beginPath();
            ctx.moveTo(50, 350);
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 40) / 80) * 600;
                const py = 350 - (V[i] / maxV) * 250;
                ctx.lineTo(px, py);
            }}
            ctx.lineTo(650, 350);
            ctx.closePath();
            ctx.fill();

            // Barrier outline
            ctx.strokeStyle = '#f44';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 40) / 80) * 600;
                const py = 350 - (V[i] / maxV) * 250;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Energy level line
            const E_kinetic = {E_kinetic};
            const eY = 350 - (E_kinetic / maxV) * 250;
            ctx.strokeStyle = 'rgba(255, 255, 0, 0.5)';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(50, eY);
            ctx.lineTo(650, eY);
            ctx.stroke();
            ctx.setLineDash([]);
            ctx.fillStyle = '#ff0';
            ctx.font = '12px sans-serif';
            ctx.fillText('E', 655, eY + 4);

            // Wavefunction |psi|^2
            const gradient = ctx.createLinearGradient(0, 100, 0, 350);
            gradient.addColorStop(0, 'rgba(170, 68, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(170, 68, 255, 0.1)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.moveTo(50, 350);
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 40) / 80) * 600;
                const py = 350 - (prob[i] / maxProb) * 300;
                ctx.lineTo(px, py);
            }}
            ctx.lineTo(650, 350);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = '#a4f';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 40) / 80) * 600;
                const py = 350 - (prob[i] / maxProb) * 300;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Labels
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.fillText('x', 660, 365);
            ctx.fillText('|psi|^2, V', 10, 45);
            ctx.fillStyle = '#f44';
            ctx.fillText('V(x)', 340, 120);
        }}

        function updateStats() {{
            const prob = probHistory[frame];
            // Calculate T and R
            let T = 0, R = 0;
            for (let i = 0; i < x.length; i++) {{
                if (x[i] > barrierWidth/2 + 5) T += prob[i] * dx;
                if (x[i] < -barrierWidth/2 - 5) R += prob[i] * dx;
            }}

            document.getElementById('tValue').textContent = T.toFixed(4);
            document.getElementById('rValue').textContent = R.toFixed(4);
            document.getElementById('sumValue').textContent = (T + R).toFixed(4);
            document.getElementById('tBar').style.width = (T * 100) + '%';
            document.getElementById('rBar').style.width = (R * 100) + '%';
        }}

        function animate() {{
            drawWavefunction();
            updateStats();
            if (playing) frame = (frame + speed) % totalFrames;
            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
    """

    filepath = 'examples/tunneling_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test with smaller grid
        run_simulation(N=512, L=80.0, sigma=3.0, k0=1.5, V0=2.0,
                       barrier_width=1.0, dt=0.02, steps=2000)
    else:
        run_simulation()
