"""
Wavepacket Spreading - Free Particle Quantum Mechanics (Level 4.1)

THE TINYPHYSICS WAY:
    1. Define the system via the Hamiltonian: H = p²/2m
    2. Solve Schrödinger equation: iℏ ∂ψ/∂t = Hψ
    3. Use split-operator method with FFT for unitary evolution

A Gaussian wavepacket in free space spreads over time due to the
uncertainty principle - different momentum components travel at
different speeds.

Theoretical spreading: σ(t) = σ₀ √(1 + (ℏt/2mσ₀²)²)

This demonstrates quantum mechanics simulation using the same
"compiler" philosophy - physics defined by energy alone.
"""

import numpy as np
from tinygrad.physics import QuantumSystem
import json
import os


def run_simulation(N: int = 512, L: float = 60.0, sigma: float = 1.0,
                   k0: float = 0.0, dt: float = 0.02, steps: int = 1000):
    """
    Simulate a Gaussian wavepacket spreading in free space.

    Args:
        N: Number of grid points
        L: Domain size [-L/2, L/2]
        sigma: Initial wavepacket width
        k0: Initial momentum (wavenumber) - gives wavepacket velocity
        dt: Time step
        steps: Number of time steps
    """
    # Physical constants (natural units: ℏ = m = 1)
    hbar = 1.0
    m = 1.0

    print("=" * 60)
    print("WAVEPACKET SPREADING - Quantum Mechanics with TinyPhysics")
    print("=" * 60)
    print(f"\nSchrödinger equation: iℏ ∂ψ/∂t = Ĥψ")
    print(f"Free particle Hamiltonian: Ĥ = p̂²/2m = -ℏ²∇²/2m")
    print(f"\nSplit-operator method:")
    print(f"  ψ(t+dt) = FFT⁻¹[e^(-iℏk²dt/2m) · FFT[ψ(t)]]")
    print(f"\nParameters:")
    print(f"  Grid points: {N}")
    print(f"  Domain: [-{L/2:.1f}, {L/2:.1f}]")
    print(f"  Initial width σ₀: {sigma}")
    print(f"  Initial momentum k₀: {k0}")
    print(f"  ℏ = {hbar}, m = {m}")
    print(f"  Time step dt: {dt}, Steps: {steps}")
    print(f"  Total time: {dt * steps:.2f}")

    # Create quantum system
    system = QuantumSystem(N=N, L=L, m=m, hbar=hbar, V=None)

    # Initial Gaussian wavepacket
    psi_r, psi_i = system.gaussian_wavepacket(x0=0.0, sigma=sigma, k0=k0)

    # Initial measurements
    norm_start = system.norm(psi_r, psi_i)
    width_start = system.width(psi_r, psi_i)
    energy_start = system.energy(psi_r, psi_i)

    print(f"\nInitial state:")
    print(f"  Norm: {norm_start:.6f} (should be 1)")
    print(f"  Width σ: {width_start:.4f}")
    print(f"  Energy <H>: {energy_start:.4f}")

    # Theoretical prediction for width spreading
    # σ(t) = σ₀ √(1 + (ℏt/2mσ₀²)²)
    t_final = dt * steps
    alpha = hbar / (2 * m * sigma**2)
    width_theory = sigma * np.sqrt(1 + (alpha * t_final)**2)
    print(f"\nTheoretical final width: {width_theory:.4f}")

    # Evolve
    print(f"\nEvolving wavefunction...")
    psi_r, psi_i, history = system.evolve(psi_r, psi_i, dt=dt, steps=steps,
                                          record_every=max(1, steps // 200))

    # Final measurements
    norm_end = system.norm(psi_r, psi_i)
    width_end = system.width(psi_r, psi_i)
    energy_end = system.energy(psi_r, psi_i)

    print(f"\nFinal state:")
    print(f"  Norm: {norm_end:.6f}")
    print(f"  Width σ: {width_end:.4f}")
    print(f"  Energy <H>: {energy_end:.4f}")

    norm_drift = abs(norm_end - norm_start) / norm_start
    energy_drift = abs(energy_end - energy_start) / abs(energy_start) if energy_start != 0 else 0
    width_error = abs(width_end - width_theory) / width_theory

    print(f"\nConservation:")
    print(f"  Norm drift: {norm_drift:.2e}")
    print(f"  Energy drift: {energy_drift:.2e}")
    print(f"  Width vs theory error: {width_error:.2e}")

    # Generate viewer
    generate_viewer(history, sigma, k0, hbar, m, dt)

    return norm_drift, energy_drift, width_error


def generate_viewer(history, sigma0, k0, hbar, m, dt):
    """Generate HTML visualization for wavepacket."""
    # Extract data for JavaScript
    x_data = history[0][0].tolist()
    prob_data = [h[1].tolist() for h in history]
    norm_data = [h[2] for h in history]
    width_data = [h[3] for h in history]
    x_mean_data = [h[4] for h in history]
    energy_data = [h[5] for h in history]

    # Calculate theoretical width curve
    alpha = hbar / (2 * m * sigma0**2)
    times = [i * dt * (len(history) - 1) / (len(history) - 1) for i in range(len(history))]
    width_theory = [sigma0 * np.sqrt(1 + (alpha * t)**2) for t in
                    np.linspace(0, dt * (len(history) - 1) * (len(prob_data) - 1) // max(len(prob_data) - 1, 1),
                                len(history))]
    # Recalculate times properly
    total_time = dt * (len(prob_data) - 1) * (2000 // 200) if len(prob_data) > 1 else 0
    width_theory = [sigma0 * np.sqrt(1 + (alpha * t)**2)
                    for t in np.linspace(0, total_time, len(history))]

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Wavepacket Spreading - TinyPhysics</title>
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
        h1 {{
            text-align: center;
            color: #fff;
            margin-bottom: 10px;
            font-weight: 300;
        }}
        .subtitle {{
            text-align: center;
            color: #888;
            margin-bottom: 30px;
            font-size: 14px;
        }}
        .main-container {{
            display: flex;
            gap: 30px;
            justify-content: center;
            flex-wrap: wrap;
            max-width: 1400px;
            margin: 0 auto;
        }}
        .wave-section {{
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        canvas {{
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            background: #a4f;
            border: none;
            color: #fff;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }}
        button:hover {{ background: #b5f; transform: scale(1.05); }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
        }}
        input[type="range"] {{
            width: 100px;
            accent-color: #a4f;
        }}
        .stats-panel {{
            background: rgba(20, 20, 40, 0.9);
            border-radius: 10px;
            padding: 20px;
            min-width: 320px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .stat-group {{
            margin-bottom: 20px;
        }}
        .stat-group h3 {{
            color: #a4f;
            margin: 0 0 12px 0;
            font-size: 14px;
            text-transform: uppercase;
            letter-spacing: 1px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #fff; font-family: monospace; }}
        .stat-value.good {{ color: #4f4; }}
        .quantum-info {{
            background: rgba(170, 68, 255, 0.1);
            border-left: 3px solid #a4f;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .quantum-info .title {{
            color: #a4f;
            font-weight: bold;
            font-size: 12px;
        }}
        .quantum-info .desc {{
            color: #aaa;
            font-size: 12px;
            margin-top: 4px;
        }}
        .formula {{
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 8px 12px;
            border-radius: 4px;
            color: #4f4;
            display: inline-block;
            margin: 5px 0;
        }}
        .chart-container {{
            margin-top: 15px;
        }}
        .chart-container h4 {{
            color: #888;
            font-size: 12px;
            margin: 0 0 8px 0;
        }}
    </style>
</head>
<body>
    <h1>Quantum Wavepacket Spreading</h1>
    <div class="subtitle">Free particle evolution via split-operator FFT method</div>

    <div class="main-container">
        <div class="wave-section">
            <canvas id="waveCanvas" width="650" height="400"></canvas>
            <div class="controls">
                <button id="playBtn">Pause</button>
                <button id="resetBtn">Reset</button>
                <div class="speed-control">
                    <span>Speed:</span>
                    <input type="range" id="speedSlider" min="1" max="10" value="3">
                    <span id="speedLabel">3x</span>
                </div>
            </div>
            <canvas id="widthCanvas" width="650" height="150" style="margin-top: 20px;"></canvas>
        </div>

        <div class="stats-panel">
            <div class="stat-group">
                <h3>Wavefunction</h3>
                <div class="stat-row">
                    <span class="stat-label">Time</span>
                    <span class="stat-value" id="timeValue">0.00</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Position &lt;x&gt;</span>
                    <span class="stat-value" id="posValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Width σ</span>
                    <span class="stat-value" id="widthValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Theory σ(t)</span>
                    <span class="stat-value" id="theoryValue">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Conservation</h3>
                <div class="stat-row">
                    <span class="stat-label">Norm ∫|ψ|²dx</span>
                    <span class="stat-value good" id="normValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Energy &lt;H&gt;</span>
                    <span class="stat-value good" id="energyValue">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Initial Parameters</h3>
                <div class="stat-row">
                    <span class="stat-label">Initial width σ₀</span>
                    <span class="stat-value">{sigma0:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Momentum k₀</span>
                    <span class="stat-value">{k0:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">ℏ, m</span>
                    <span class="stat-value">{hbar:.1f}, {m:.1f}</span>
                </div>
            </div>

            <div class="quantum-info">
                <div class="title">Uncertainty Principle</div>
                <div class="desc">A localized wavepacket contains a spread of momenta. Higher momenta travel faster, causing the packet to spread.</div>
                <div class="formula">σ(t) = σ₀√(1 + (ℏt/2mσ₀²)²)</div>
            </div>

            <div class="quantum-info">
                <div class="title">Split-Operator Method</div>
                <div class="desc">FFT transforms to momentum space where kinetic energy is diagonal. Exact unitary evolution!</div>
                <div class="formula">ψ(t+dt) = F⁻¹[e^(-iℏk²dt/2m)F[ψ]]</div>
            </div>
        </div>
    </div>

    <script>
        const x = {json.dumps(x_data)};
        const probHistory = {json.dumps(prob_data)};
        const normHistory = {json.dumps(norm_data)};
        const widthHistory = {json.dumps(width_data)};
        const xMeanHistory = {json.dumps(x_mean_data)};
        const energyHistory = {json.dumps(energy_data)};
        const widthTheory = {json.dumps(width_theory)};
        const sigma0 = {sigma0};
        const dt = {dt};
        const totalFrames = probHistory.length;

        let frame = 0;
        let playing = true;
        let speed = 3;

        const waveCanvas = document.getElementById('waveCanvas');
        const wctx = waveCanvas.getContext('2d');
        const widthCanvas = document.getElementById('widthCanvas');
        const wdctx = widthCanvas.getContext('2d');

        // Controls
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

            wctx.fillStyle = '#0a0a1a';
            wctx.fillRect(0, 0, waveCanvas.width, waveCanvas.height);

            // Grid
            wctx.strokeStyle = 'rgba(255,255,255,0.1)';
            wctx.lineWidth = 1;
            for (let i = 0; i <= 10; i++) {{
                const y = 50 + (300 * i / 10);
                wctx.beginPath();
                wctx.moveTo(50, y);
                wctx.lineTo(600, y);
                wctx.stroke();
            }}

            // Axes
            wctx.strokeStyle = '#555';
            wctx.lineWidth = 2;
            wctx.beginPath();
            wctx.moveTo(50, 350);
            wctx.lineTo(600, 350);
            wctx.moveTo(50, 50);
            wctx.lineTo(50, 350);
            wctx.stroke();

            // Labels
            wctx.fillStyle = '#888';
            wctx.font = '12px sans-serif';
            wctx.fillText('x', 610, 355);
            wctx.fillText('|ψ|²', 15, 45);
            wctx.fillText('0', 45, 365);

            // Initial wavepacket (faint)
            const prob0 = probHistory[0];
            wctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            wctx.lineWidth = 1;
            wctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 20) / 40) * 550;
                const py = 350 - (prob0[i] / maxProb) * 300;
                if (i === 0) wctx.moveTo(px, py);
                else wctx.lineTo(px, py);
            }}
            wctx.stroke();

            // Current wavepacket with gradient fill
            const gradient = wctx.createLinearGradient(0, 50, 0, 350);
            gradient.addColorStop(0, 'rgba(170, 68, 255, 0.8)');
            gradient.addColorStop(1, 'rgba(170, 68, 255, 0.1)');

            wctx.fillStyle = gradient;
            wctx.beginPath();
            wctx.moveTo(50, 350);
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 20) / 40) * 550;
                const py = 350 - (prob[i] / maxProb) * 300;
                wctx.lineTo(px, py);
            }}
            wctx.lineTo(600, 350);
            wctx.closePath();
            wctx.fill();

            // Current wavepacket line
            wctx.strokeStyle = '#a4f';
            wctx.lineWidth = 2;
            wctx.beginPath();
            for (let i = 0; i < x.length; i++) {{
                const px = 50 + ((x[i] + 20) / 40) * 550;
                const py = 350 - (prob[i] / maxProb) * 300;
                if (i === 0) wctx.moveTo(px, py);
                else wctx.lineTo(px, py);
            }}
            wctx.stroke();

            // Position marker
            const xMean = xMeanHistory[frame];
            const markerX = 50 + ((xMean + 20) / 40) * 550;
            wctx.strokeStyle = '#ff0';
            wctx.lineWidth = 2;
            wctx.setLineDash([5, 5]);
            wctx.beginPath();
            wctx.moveTo(markerX, 50);
            wctx.lineTo(markerX, 350);
            wctx.stroke();
            wctx.setLineDash([]);
            wctx.fillStyle = '#ff0';
            wctx.fillText('<x>', markerX - 10, 40);
        }}

        function drawWidthPlot() {{
            wdctx.fillStyle = '#0a0a1a';
            wdctx.fillRect(0, 0, widthCanvas.width, widthCanvas.height);

            const maxWidth = Math.max(...widthHistory, ...widthTheory) * 1.2;

            // Labels
            wdctx.fillStyle = '#888';
            wdctx.font = '12px sans-serif';
            wdctx.fillText('Width σ(t) vs Time', 280, 20);
            wdctx.fillText('σ', 15, 75);
            wdctx.fillText('t', 620, 130);

            // Axes
            wdctx.strokeStyle = '#555';
            wdctx.lineWidth = 1;
            wdctx.beginPath();
            wdctx.moveTo(50, 130);
            wdctx.lineTo(600, 130);
            wdctx.moveTo(50, 30);
            wdctx.lineTo(50, 130);
            wdctx.stroke();

            // Theoretical curve (full)
            wdctx.strokeStyle = 'rgba(255, 255, 255, 0.3)';
            wdctx.lineWidth = 2;
            wdctx.beginPath();
            for (let i = 0; i < widthTheory.length; i++) {{
                const px = 50 + (i / (widthTheory.length - 1)) * 550;
                const py = 130 - ((widthTheory[i] - sigma0) / (maxWidth - sigma0)) * 90;
                if (i === 0) wdctx.moveTo(px, py);
                else wdctx.lineTo(px, py);
            }}
            wdctx.stroke();
            wdctx.fillStyle = 'rgba(255,255,255,0.3)';
            wdctx.fillText('Theory', 560, 60);

            // Simulated width (up to current frame)
            wdctx.strokeStyle = '#a4f';
            wdctx.lineWidth = 2;
            wdctx.beginPath();
            for (let i = 0; i <= frame; i++) {{
                const px = 50 + (i / (totalFrames - 1)) * 550;
                const py = 130 - ((widthHistory[i] - sigma0) / (maxWidth - sigma0)) * 90;
                if (i === 0) wdctx.moveTo(px, py);
                else wdctx.lineTo(px, py);
            }}
            wdctx.stroke();

            // Current point
            const currentX = 50 + (frame / (totalFrames - 1)) * 550;
            const currentY = 130 - ((widthHistory[frame] - sigma0) / (maxWidth - sigma0)) * 90;
            wdctx.fillStyle = '#a4f';
            wdctx.beginPath();
            wdctx.arc(currentX, currentY, 5, 0, 2 * Math.PI);
            wdctx.fill();

            // Legend
            wdctx.fillStyle = '#a4f';
            wdctx.fillText('Simulation', 560, 80);
        }}

        function updateStats() {{
            const time = (frame / (totalFrames - 1)) * dt * (totalFrames - 1) * 10;
            document.getElementById('timeValue').textContent = time.toFixed(2);
            document.getElementById('posValue').textContent = xMeanHistory[frame].toFixed(3);
            document.getElementById('widthValue').textContent = widthHistory[frame].toFixed(4);
            document.getElementById('theoryValue').textContent = widthTheory[frame].toFixed(4);
            document.getElementById('normValue').textContent = normHistory[frame].toFixed(6);
            document.getElementById('energyValue').textContent = energyHistory[frame].toFixed(4);
        }}

        function animate() {{
            drawWavefunction();
            drawWidthPlot();
            updateStats();

            if (playing) {{
                frame = (frame + speed) % totalFrames;
            }}

            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
    """

    filepath = 'examples/wavepacket_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        # Quick test with smaller parameters
        run_simulation(N=256, L=30.0, sigma=1.0, k0=2.0, dt=0.02, steps=500)
    else:
        run_simulation()
