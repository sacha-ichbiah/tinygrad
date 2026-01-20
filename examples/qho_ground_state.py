"""
Quantum Harmonic Oscillator - Finding Ground State (Level 4.3)

THE TINYPHYSICS WAY:
    1. Define Hamiltonian: H = p²/2m + ½mω²x²
    2. Use imaginary time evolution (Wick rotation: t → -iτ)
    3. Watch an arbitrary state relax to the ground state!

The trick: Replace t → -iτ in Schrödinger equation:
    Real time:      iℏ ∂ψ/∂t = Ĥψ  →  ψ(t) = e^(-iĤt/ℏ)ψ(0)  (oscillates)
    Imaginary time: ℏ ∂ψ/∂τ = -Ĥψ  →  ψ(τ) = e^(-Ĥτ/ℏ)ψ(0)   (decays)

Higher energy eigenstates decay faster → system relaxes to ground state!

Ground state of harmonic oscillator:
    ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ)
    E₀ = ℏω/2
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import QuantumSystem
import json
import os


def run_simulation(N: int = 512, L: float = 20.0, omega: float = 1.0,
                   dtau: float = 0.05, steps: int = 500):
    """
    Find ground state of quantum harmonic oscillator via imaginary time evolution.

    Args:
        N: Number of grid points
        L: Domain size [-L/2, L/2]
        omega: Oscillator frequency
        dtau: Imaginary time step
        steps: Number of imaginary time steps
    """
    hbar, m = 1.0, 1.0

    # Exact ground state energy
    E0_exact = hbar * omega / 2

    print("=" * 60)
    print("QUANTUM HARMONIC OSCILLATOR - Ground State via Imaginary Time")
    print("=" * 60)
    print(f"\nHamiltonian: H = p²/2m + ½mω²x²")
    print(f"\nImaginary time evolution (Wick rotation t → -iτ):")
    print(f"  ψ(τ) = e^(-Ĥτ/ℏ)ψ(0) / ||...||")
    print(f"  Higher energies decay faster → relaxes to ground state")
    print(f"\nParameters:")
    print(f"  Grid: {N} points, Domain: [-{L/2:.0f}, {L/2:.0f}]")
    print(f"  ω = {omega}, ℏ = {hbar}, m = {m}")
    print(f"  Imaginary time step dτ = {dtau}")
    print(f"  Steps: {steps}")
    print(f"\nExact ground state:")
    print(f"  E₀ = ℏω/2 = {E0_exact:.6f}")
    print(f"  ψ₀(x) = (mω/πℏ)^(1/4) exp(-mωx²/2ℏ)")

    # Create quantum system with harmonic potential
    system = QuantumSystem(N=N, L=L, m=m, hbar=hbar)
    V = system.harmonic_potential(omega)
    system.V = V

    # Start with a random-looking initial state (superposition of excited states)
    # Use a displaced, asymmetric Gaussian
    x_np = system.x.numpy()
    psi_init = np.exp(-(x_np - 2)**2 / 2) + 0.5 * np.exp(-(x_np + 1)**2 / 0.5)
    psi_init = psi_init / np.sqrt(np.sum(psi_init**2) * system.dx)
    psi_r = Tensor(psi_init)
    psi_i = Tensor(np.zeros_like(psi_init))

    # Initial energy (should be higher than ground state)
    E_init = system.energy(psi_r, psi_i)
    print(f"\nInitial state (random superposition):")
    print(f"  Energy: {E_init:.6f}")
    print(f"  Excess energy: {E_init - E0_exact:.6f}")

    # Find ground state via imaginary time evolution
    print(f"\nRunning imaginary time evolution...")
    psi_r, psi_i, history = system.find_ground_state(
        psi_r, psi_i, dtau=dtau, steps=steps, tol=1e-12, record_every=5
    )

    # Final state
    E_final = system.energy(psi_r, psi_i)
    width_final = system.width(psi_r, psi_i)

    # Exact ground state for comparison
    psi_exact_r, psi_exact_i, _ = system.ground_state_exact(omega)
    width_exact = system.width(psi_exact_r, psi_exact_i)

    # Overlap with exact ground state: |<ψ|ψ₀>|²
    psi_final_np = psi_r.numpy()
    psi_exact_np = psi_exact_r.numpy()
    overlap = abs(np.sum(psi_final_np * psi_exact_np) * system.dx) ** 2

    print(f"\nFinal state after {len(history)} recorded steps:")
    print(f"  Energy: {E_final:.6f}")
    print(f"  Exact E₀: {E0_exact:.6f}")
    print(f"  Energy error: {abs(E_final - E0_exact):.2e}")
    print(f"  Width σ: {width_final:.6f}")
    print(f"  Exact σ: {width_exact:.6f}")
    print(f"  Overlap |<ψ|ψ₀>|²: {overlap:.6f}")

    # Generate viewer
    generate_viewer(history, x_np, V.numpy(), E0_exact, omega, hbar, m,
                    psi_exact_np, width_exact)

    return E_final, E0_exact, overlap


def generate_viewer(history, x_np, V_np, E0_exact, omega, hbar, m,
                    psi_exact_np, width_exact):
    """Generate HTML visualization."""
    x_data = x_np.tolist()
    V_data = V_np.tolist()
    prob_data = [h[1].tolist() for h in history]
    energy_data = [h[2] for h in history]
    width_data = [h[3] for h in history]
    exact_prob = (psi_exact_np ** 2).tolist()

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Quantum Harmonic Oscillator - Ground State</title>
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
        .controls {{ margin-top: 15px; display: flex; gap: 10px; align-items: center; }}
        button {{
            background: #4af; border: none; color: #000; padding: 10px 20px;
            border-radius: 5px; cursor: pointer; font-weight: bold; transition: all 0.2s;
        }}
        button:hover {{ background: #5bf; transform: scale(1.05); }}
        .speed-control {{ display: flex; align-items: center; gap: 8px; color: #888; }}
        input[type="range"] {{ width: 100px; accent-color: #4af; }}
        .stats-panel {{
            background: rgba(20, 20, 40, 0.9); border-radius: 10px;
            padding: 20px; min-width: 320px; box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .stat-group {{ margin-bottom: 20px; }}
        .stat-group h3 {{
            color: #4af; margin: 0 0 12px 0; font-size: 14px;
            text-transform: uppercase; letter-spacing: 1px;
        }}
        .stat-row {{
            display: flex; justify-content: space-between; padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #fff; font-family: monospace; }}
        .stat-value.good {{ color: #4f4; }}
        .stat-value.highlight {{ color: #ff0; }}
        .quantum-info {{
            background: rgba(74, 170, 255, 0.1); border-left: 3px solid #4af;
            padding: 10px 15px; margin: 10px 0; border-radius: 0 5px 5px 0;
        }}
        .quantum-info .title {{ color: #4af; font-weight: bold; font-size: 12px; }}
        .quantum-info .desc {{ color: #aaa; font-size: 12px; margin-top: 4px; }}
        .formula {{
            font-family: monospace; background: rgba(0,0,0,0.3);
            padding: 8px 12px; border-radius: 4px; color: #4f4;
            display: inline-block; margin: 5px 0;
        }}
    </style>
</head>
<body>
    <h1>Quantum Harmonic Oscillator</h1>
    <div class="subtitle">Finding ground state via imaginary time evolution</div>

    <div class="main-container">
        <div class="wave-section">
            <canvas id="waveCanvas" width="650" height="400"></canvas>
            <div class="controls">
                <button id="playBtn">Pause</button>
                <button id="resetBtn">Reset</button>
                <div class="speed-control">
                    <span>Speed:</span>
                    <input type="range" id="speedSlider" min="1" max="10" value="2">
                    <span id="speedLabel">2x</span>
                </div>
            </div>
            <canvas id="energyCanvas" width="650" height="150" style="margin-top: 20px;"></canvas>
        </div>

        <div class="stats-panel">
            <div class="stat-group">
                <h3>Current State</h3>
                <div class="stat-row">
                    <span class="stat-label">Imaginary time τ</span>
                    <span class="stat-value" id="tauValue">0</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Energy</span>
                    <span class="stat-value highlight" id="energyValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Exact E₀ = ℏω/2</span>
                    <span class="stat-value good">{E0_exact:.6f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Error</span>
                    <span class="stat-value good" id="errorValue">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Wavefunction Shape</h3>
                <div class="stat-row">
                    <span class="stat-label">Width σ</span>
                    <span class="stat-value" id="widthValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Exact σ</span>
                    <span class="stat-value good">{width_exact:.6f}</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Parameters</h3>
                <div class="stat-row">
                    <span class="stat-label">Frequency ω</span>
                    <span class="stat-value">{omega}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">ℏ, m</span>
                    <span class="stat-value">{hbar}, {m}</span>
                </div>
            </div>

            <div class="quantum-info">
                <div class="title">Imaginary Time Evolution</div>
                <div class="desc">Wick rotation t → -iτ converts oscillation to decay. Higher energy states decay faster!</div>
                <div class="formula">ψ(τ) = e^(-Ĥτ/ℏ)ψ(0) / norm</div>
            </div>

            <div class="quantum-info">
                <div class="title">Ground State</div>
                <div class="desc">The system relaxes to the lowest energy eigenstate - a Gaussian centered at origin.</div>
                <div class="formula">ψ₀ = (mω/πℏ)^(1/4) e^(-mωx²/2ℏ)</div>
            </div>
        </div>
    </div>

    <script>
        const x = {json.dumps(x_data)};
        const V = {json.dumps(V_data)};
        const probHistory = {json.dumps(prob_data)};
        const energyHistory = {json.dumps(energy_data)};
        const widthHistory = {json.dumps(width_data)};
        const exactProb = {json.dumps(exact_prob)};
        const E0 = {E0_exact};
        const totalFrames = probHistory.length;

        let frame = 0, playing = true, speed = 2;
        const canvas = document.getElementById('waveCanvas');
        const ctx = canvas.getContext('2d');
        const eCanvas = document.getElementById('energyCanvas');
        const ectx = eCanvas.getContext('2d');

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
            const maxProb = Math.max(...probHistory.flat(), ...exactProb) * 1.2;
            const maxV = Math.max(...V) * 0.3;

            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Potential (parabola)
            ctx.strokeStyle = 'rgba(255, 200, 100, 0.4)';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 10) / 20) * 550;
                const py = 350 - (V[i] / maxV) * 100;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Exact ground state (target)
            ctx.strokeStyle = 'rgba(100, 255, 100, 0.5)';
            ctx.lineWidth = 2;
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 10) / 20) * 550;
                const py = 350 - (exactProb[i] / maxProb) * 300;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();
            ctx.setLineDash([]);

            // Current wavefunction
            const gradient = ctx.createLinearGradient(0, 50, 0, 350);
            gradient.addColorStop(0, 'rgba(74, 170, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(74, 170, 255, 0.1)');

            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.moveTo(50, 350);
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 10) / 20) * 550;
                const py = 350 - (prob[i] / maxProb) * 300;
                ctx.lineTo(px, py);
            }}
            ctx.lineTo(600, 350);
            ctx.closePath();
            ctx.fill();

            ctx.strokeStyle = '#4af';
            ctx.lineWidth = 2;
            ctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 10) / 20) * 550;
                const py = 350 - (prob[i] / maxProb) * 300;
                if (i === 0) ctx.moveTo(px, py);
                else ctx.lineTo(px, py);
            }}
            ctx.stroke();

            // Labels
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.fillText('x', 610, 365);
            ctx.fillText('|ψ|²', 10, 45);
            ctx.fillStyle = 'rgba(255, 200, 100, 0.8)';
            ctx.fillText('V(x) = ½mω²x²', 480, 280);
            ctx.fillStyle = 'rgba(100, 255, 100, 0.8)';
            ctx.fillText('Exact ψ₀', 520, 100);
            ctx.fillStyle = '#4af';
            ctx.fillText('Current ψ', 520, 120);
        }}

        function drawEnergyPlot() {{
            ectx.fillStyle = '#0a0a1a';
            ectx.fillRect(0, 0, eCanvas.width, eCanvas.height);

            const E_max = Math.max(...energyHistory) * 1.1;
            const E_min = E0 * 0.9;

            // Ground state energy line
            const y0 = 130 - ((E0 - E_min) / (E_max - E_min)) * 100;
            ectx.strokeStyle = 'rgba(100, 255, 100, 0.5)';
            ectx.setLineDash([5, 5]);
            ectx.beginPath();
            ectx.moveTo(50, y0);
            ectx.lineTo(600, y0);
            ectx.stroke();
            ectx.setLineDash([]);
            ectx.fillStyle = 'rgba(100, 255, 100, 0.8)';
            ectx.font = '12px sans-serif';
            ectx.fillText('E₀', 605, y0 + 4);

            // Energy curve
            ectx.strokeStyle = '#ff0';
            ectx.lineWidth = 2;
            ectx.beginPath();
            for (let i = 0; i <= frame; i++) {{
                const px = 50 + (i / (totalFrames - 1)) * 550;
                const py = 130 - ((energyHistory[i] - E_min) / (E_max - E_min)) * 100;
                if (i === 0) ectx.moveTo(px, py);
                else ectx.lineTo(px, py);
            }}
            ectx.stroke();

            // Current point
            const currentX = 50 + (frame / (totalFrames - 1)) * 550;
            const currentY = 130 - ((energyHistory[frame] - E_min) / (E_max - E_min)) * 100;
            ectx.fillStyle = '#ff0';
            ectx.beginPath();
            ectx.arc(currentX, currentY, 5, 0, 2 * Math.PI);
            ectx.fill();

            // Labels
            ectx.fillStyle = '#888';
            ectx.fillText('Energy vs Imaginary Time τ', 250, 20);
            ectx.fillText('E', 15, 70);
            ectx.fillText('τ', 610, 140);
        }}

        function updateStats() {{
            document.getElementById('tauValue').textContent = frame;
            document.getElementById('energyValue').textContent = energyHistory[frame].toFixed(6);
            document.getElementById('errorValue').textContent = Math.abs(energyHistory[frame] - E0).toExponential(2);
            document.getElementById('widthValue').textContent = widthHistory[frame].toFixed(6);
        }}

        function animate() {{
            drawWavefunction();
            drawEnergyPlot();
            updateStats();
            if (playing && frame < totalFrames - 1) frame += speed;
            if (frame >= totalFrames) frame = totalFrames - 1;
            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
    """

    filepath = 'examples/qho_ground_state_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        run_simulation(N=256, L=15.0, omega=1.0, dtau=0.1, steps=200)
    else:
        run_simulation()
