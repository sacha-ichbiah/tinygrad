"""
Kepler Problem - Planet orbiting a star (Level 1.3)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. The system automatically derives equations of motion via autograd
    3. Symplectic integrator preserves energy

Hamiltonian: H = |p|²/2m - GMm/|r|

This demonstrates the "compiler" approach - physics defined by energy alone.
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem
import json
import os


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def kepler_hamiltonian(GM: float = 1.0, m: float = 1.0, softening: float = 0.0):
    """
    Returns the Hamiltonian function for the Kepler problem.

    H = T + V = |p|²/2m - GMm/|r|

    The symplectic integrator will automatically derive:
        dq/dt = dH/dp = p/m           (velocity)
        dp/dt = -dH/dq = -GMm*r/|r|³  (gravitational force)

    No need to manually compute forces - autograd does it!
    """
    eps_sq = softening ** 2

    def H(q, p):
        # Kinetic energy: T = |p|²/2m
        T = (p * p).sum() / (2 * m)

        # Potential energy: V = -GMm/|r|
        # With optional softening to handle singularity
        r_sq = (q * q).sum()
        if softening > 0:
            r = (r_sq + eps_sq).sqrt()
        else:
            r = r_sq.sqrt()
        V = -GM * m / r

        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(integrator="yoshida4", dt=0.01, steps=5000, eccentricity=0.6):
    """
    Simulate the Kepler problem using the TinyPhysics compiler approach.

    User defines ONLY the Hamiltonian. Everything else is automatic.
    """
    # Physical constants (normalized: GM = 1)
    GM = 1.0
    m = 1.0
    a = 1.0  # Semi-major axis

    # Initial conditions for elliptical orbit (starting at aphelion)
    e = eccentricity
    r_aphelion = a * (1 + e)
    v_aphelion = np.sqrt(GM * (1 - e) / (a * (1 + e)))

    # Initial state
    q = Tensor([r_aphelion, 0.0], requires_grad=True)
    p = Tensor([0.0, m * v_aphelion], requires_grad=True)

    # Expected orbital period
    T_orbital = 2 * np.pi * np.sqrt(a**3 / GM)

    print("=" * 60)
    print("KEPLER PROBLEM - TinyPhysics Compiler Approach")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H(q, p) = |p|²/2m - GMm/|r|")
    print(f"\nEquations of motion derived automatically via autograd:")
    print(f"  dq/dt = +dH/dp = p/m")
    print(f"  dp/dt = -dH/dq = -GMm*r/|r|³")
    print(f"\nIntegrator: {integrator}")
    print(f"Eccentricity: {e}, Semi-major axis: {a}")
    print(f"Orbital period: {T_orbital:.4f}")
    print(f"Simulation: {steps} steps, dt={dt} ({dt*steps/T_orbital:.1f} orbits)")

    # CREATE THE HAMILTONIAN SYSTEM - This is the "compilation" step
    H = kepler_hamiltonian(GM=GM, m=m)
    system = HamiltonianSystem(H, integrator=integrator)

    # Initial energy
    E_start = system.energy(q, p)
    L_start = float((q.numpy()[0] * p.numpy()[1] - q.numpy()[1] * p.numpy()[0]))

    print(f"\nInitial Energy: {E_start:.6f}")
    print(f"Initial Angular Momentum: {L_start:.6f}")

    # EVOLVE THE SYSTEM
    q, p, history = system.evolve(q, p, dt=dt, steps=steps, record_every=10)

    # Final state
    E_end = system.energy(q, p)
    q_np, p_np = q.numpy(), p.numpy()
    L_end = float(q_np[0] * p_np[1] - q_np[1] * p_np[0])

    E_drift = abs(E_end - E_start) / abs(E_start)
    L_drift = abs(L_end - L_start) / abs(L_start)

    print(f"\nFinal Energy: {E_end:.6f}")
    print(f"Final Angular Momentum: {L_end:.6f}")
    print(f"Energy Drift: {E_drift:.2e}")
    print(f"Angular Momentum Drift: {L_drift:.2e}")

    # Generate viewer
    history_q = [h[0].tolist() for h in history]
    history_E = [h[2] for h in history]
    generate_viewer(history_q, a, e, history_E, E_start)

    return E_start, E_end, E_drift, L_drift


def generate_viewer(history_q, a, e, history_E, E_start):
    """Generate HTML visualization."""
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Kepler Problem - TinyPhysics</title>
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
        .orbit-section {{
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
            background: #4af;
            border: none;
            color: #000;
            padding: 10px 20px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
            transition: all 0.2s;
        }}
        button:hover {{ background: #5bf; transform: scale(1.05); }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
        }}
        input[type="range"] {{
            width: 100px;
            accent-color: #4af;
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
            color: #4af;
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
        .stat-value.warn {{ color: #fa4; }}
        .kepler-law {{
            background: rgba(74, 170, 255, 0.1);
            border-left: 3px solid #4af;
            padding: 10px 15px;
            margin: 10px 0;
            border-radius: 0 5px 5px 0;
        }}
        .kepler-law .title {{
            color: #4af;
            font-weight: bold;
            font-size: 12px;
        }}
        .kepler-law .desc {{
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
        .mini-chart {{
            background: #000;
            border-radius: 5px;
            height: 60px;
            position: relative;
            overflow: hidden;
        }}
    </style>
</head>
<body>
    <h1>Kepler Problem</h1>
    <div class="subtitle">Two-body gravitational orbit simulated from Hamiltonian mechanics</div>

    <div class="main-container">
        <div class="orbit-section">
            <canvas id="orbitCanvas" width="550" height="550"></canvas>
            <div class="controls">
                <button id="playBtn">Pause</button>
                <button id="resetBtn">Reset</button>
                <div class="speed-control">
                    <span>Speed:</span>
                    <input type="range" id="speedSlider" min="1" max="20" value="5">
                    <span id="speedLabel">5x</span>
                </div>
            </div>
        </div>

        <div class="stats-panel">
            <div class="stat-group">
                <h3>Current State</h3>
                <div class="stat-row">
                    <span class="stat-label">Position (x, y)</span>
                    <span class="stat-value" id="posValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Distance from Sun</span>
                    <span class="stat-value" id="distValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Speed</span>
                    <span class="stat-value" id="speedValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Orbit Progress</span>
                    <span class="stat-value" id="orbitValue">-</span>
                </div>
            </div>

            <div class="stat-group">
                <h3>Conservation Laws</h3>
                <div class="stat-row">
                    <span class="stat-label">Total Energy</span>
                    <span class="stat-value good" id="energyValue">-</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Energy Drift</span>
                    <span class="stat-value good" id="driftValue">-</span>
                </div>
                <div class="chart-container">
                    <h4>Energy over time</h4>
                    <canvas id="energyChart" class="mini-chart" width="280" height="60"></canvas>
                </div>
            </div>

            <div class="stat-group">
                <h3>Orbital Parameters</h3>
                <div class="stat-row">
                    <span class="stat-label">Semi-major axis (a)</span>
                    <span class="stat-value">{a:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Eccentricity (e)</span>
                    <span class="stat-value">{e:.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Perihelion</span>
                    <span class="stat-value">{a * (1 - e):.2f}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">Aphelion</span>
                    <span class="stat-value">{a * (1 + e):.2f}</span>
                </div>
            </div>

            <div class="kepler-law">
                <div class="title">How it works</div>
                <div class="desc">Only the Hamiltonian is defined:</div>
                <div class="formula">H = p²/2m - GMm/r</div>
                <div class="desc">Equations of motion derived via autograd. Symplectic integrator preserves energy.</div>
            </div>
        </div>
    </div>

    <script>
        const history = {json.dumps(history_q)};
        const energyHistory = {json.dumps(history_E)};
        const E_start = {E_start};
        const ecc = {e};
        const semiMajor = {a};

        // State
        let frame = 0;
        let playing = true;
        let speed = 5;

        // Canvas setup
        const canvas = document.getElementById('orbitCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 130;

        // Energy chart
        const eCanvas = document.getElementById('energyChart');
        const ectx = eCanvas.getContext('2d');

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

        function drawOrbit() {{
            // Clear
            ctx.fillStyle = '#0a0a1a';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Stars background
            ctx.fillStyle = '#fff';
            for (let i = 0; i < 100; i++) {{
                const x = (Math.sin(i * 123.456) * 0.5 + 0.5) * canvas.width;
                const y = (Math.cos(i * 654.321) * 0.5 + 0.5) * canvas.height;
                const size = Math.random() * 1.5;
                ctx.globalAlpha = 0.3 + Math.random() * 0.4;
                ctx.fillRect(x, y, size, size);
            }}
            ctx.globalAlpha = 1;

            // Expected ellipse (faint reference)
            const a_px = semiMajor * scale;
            const b_px = a_px * Math.sqrt(1 - ecc * ecc);
            const c_px = ecc * a_px;
            ctx.strokeStyle = 'rgba(100, 100, 100, 0.2)';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.ellipse(cx - c_px, cy, a_px, b_px, 0, 0, 2 * Math.PI);
            ctx.stroke();

            // Orbit trail (gradient fade)
            const trailLength = Math.min(frame + 1, 150);
            const startIdx = Math.max(0, frame - trailLength + 1);
            ctx.lineWidth = 2;
            for (let i = startIdx; i < frame; i++) {{
                const alpha = (i - startIdx) / trailLength;
                ctx.strokeStyle = `rgba(74, 170, 255, ${{alpha * 0.8}})`;
                ctx.beginPath();
                const x1 = cx + history[i][0] * scale;
                const y1 = cy - history[i][1] * scale;
                const x2 = cx + history[i + 1][0] * scale;
                const y2 = cy - history[i + 1][1] * scale;
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }}

            // Sun with glow
            const gradient = ctx.createRadialGradient(cx, cy, 0, cx, cy, 40);
            gradient.addColorStop(0, 'rgba(255, 255, 100, 1)');
            gradient.addColorStop(0.3, 'rgba(255, 200, 50, 0.8)');
            gradient.addColorStop(0.6, 'rgba(255, 150, 0, 0.3)');
            gradient.addColorStop(1, 'rgba(255, 100, 0, 0)');
            ctx.fillStyle = gradient;
            ctx.beginPath();
            ctx.arc(cx, cy, 40, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#ffee00';
            ctx.beginPath();
            ctx.arc(cx, cy, 12, 0, 2 * Math.PI);
            ctx.fill();

            // Planet
            const pos = history[frame];
            const px = cx + pos[0] * scale;
            const py = cy - pos[1] * scale;

            // Planet glow
            const pGrad = ctx.createRadialGradient(px, py, 0, px, py, 20);
            pGrad.addColorStop(0, 'rgba(74, 170, 255, 0.8)');
            pGrad.addColorStop(0.5, 'rgba(74, 170, 255, 0.2)');
            pGrad.addColorStop(1, 'rgba(74, 170, 255, 0)');
            ctx.fillStyle = pGrad;
            ctx.beginPath();
            ctx.arc(px, py, 20, 0, 2 * Math.PI);
            ctx.fill();

            ctx.fillStyle = '#4af';
            ctx.beginPath();
            ctx.arc(px, py, 8, 0, 2 * Math.PI);
            ctx.fill();

            // Velocity indicator
            if (frame < history.length - 1) {{
                const nextPos = history[Math.min(frame + 5, history.length - 1)];
                const vx = (nextPos[0] - pos[0]) * 15;
                const vy = (nextPos[1] - pos[1]) * 15;
                ctx.strokeStyle = '#4f4';
                ctx.lineWidth = 2;
                ctx.beginPath();
                ctx.moveTo(px, py);
                ctx.lineTo(px + vx * scale, py - vy * scale);
                ctx.stroke();
                // Arrow head
                const angle = Math.atan2(-vy, vx);
                ctx.beginPath();
                ctx.moveTo(px + vx * scale, py - vy * scale);
                ctx.lineTo(px + vx * scale - 8 * Math.cos(angle - 0.4), py - vy * scale + 8 * Math.sin(angle - 0.4));
                ctx.moveTo(px + vx * scale, py - vy * scale);
                ctx.lineTo(px + vx * scale - 8 * Math.cos(angle + 0.4), py - vy * scale + 8 * Math.sin(angle + 0.4));
                ctx.stroke();
            }}

            // Distance line to sun
            ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(cx, cy);
            ctx.lineTo(px, py);
            ctx.stroke();
            ctx.setLineDash([]);

            // Labels
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.fillText('Sun', cx + 15, cy - 15);
            ctx.fillStyle = '#4af';
            ctx.fillText('Planet', px + 12, py - 12);
        }}

        function drawEnergyChart() {{
            ectx.fillStyle = '#000';
            ectx.fillRect(0, 0, eCanvas.width, eCanvas.height);

            const visibleEnergy = energyHistory.slice(0, frame + 1);
            if (visibleEnergy.length < 2) return;

            const eMin = Math.min(...energyHistory);
            const eMax = Math.max(...energyHistory);
            const range = (eMax - eMin) || 1e-10;

            // Reference line (initial energy)
            const yRef = 50 - ((E_start - eMin) / range) * 40;
            ectx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
            ectx.beginPath();
            ectx.moveTo(0, yRef);
            ectx.lineTo(eCanvas.width, yRef);
            ectx.stroke();

            // Energy line
            ectx.strokeStyle = '#4f4';
            ectx.lineWidth = 1.5;
            ectx.beginPath();
            for (let i = 0; i < visibleEnergy.length; i++) {{
                const x = (i / energyHistory.length) * eCanvas.width;
                const y = 50 - ((visibleEnergy[i] - eMin) / range) * 40;
                if (i === 0) ectx.moveTo(x, y);
                else ectx.lineTo(x, y);
            }}
            ectx.stroke();
        }}

        function updateStats() {{
            const pos = history[frame];
            const dist = Math.sqrt(pos[0] * pos[0] + pos[1] * pos[1]);

            // Calculate velocity
            let speed_val = 0;
            if (frame < history.length - 1) {{
                const next = history[frame + 1];
                const dx = next[0] - pos[0];
                const dy = next[1] - pos[1];
                speed_val = Math.sqrt(dx * dx + dy * dy) * 100; // Scale for display
            }}

            const energy = energyHistory[frame];
            const drift = Math.abs(energy - E_start) / Math.abs(E_start);

            document.getElementById('posValue').textContent = `(${{pos[0].toFixed(2)}}, ${{pos[1].toFixed(2)}})`;
            document.getElementById('distValue').textContent = dist.toFixed(3);
            document.getElementById('speedValue').textContent = speed_val.toFixed(3);
            document.getElementById('orbitValue').textContent = `${{(frame / history.length * 100).toFixed(1)}}%`;
            document.getElementById('energyValue').textContent = energy.toFixed(6);
            document.getElementById('driftValue').textContent = drift.toExponential(2);

            // Color code drift
            const driftEl = document.getElementById('driftValue');
            if (drift < 1e-6) {{
                driftEl.className = 'stat-value good';
            }} else if (drift < 1e-3) {{
                driftEl.className = 'stat-value warn';
            }} else {{
                driftEl.className = 'stat-value';
                driftEl.style.color = '#f44';
            }}
        }}

        function animate() {{
            drawOrbit();
            drawEnergyChart();
            updateStats();

            if (playing) {{
                frame = (frame + speed) % history.length;
            }}

            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>
    """

    filepath = 'examples/kepler_viewer.html'
    with open(filepath, 'w') as f:
        f.write(html_content)
    print(f"\nViewer: {os.path.abspath(filepath)}")


def compare_integrators():
    """Compare integrators."""
    print("=" * 60)
    print("COMPARING SYMPLECTIC INTEGRATORS")
    print("=" * 60)

    results = {}
    for name in ["euler", "leapfrog", "yoshida4"]:
        print(f"\n{'='*60}")
        _, _, E_drift, L_drift = run_simulation(integrator=name, dt=0.01, steps=5000)
        results[name] = (E_drift, L_drift)

    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)
    print(f"{'Integrator':<12} {'Energy Drift':<15} {'Ang.Mom. Drift':<15}")
    print("-"*42)
    for name, (e_drift, l_drift) in results.items():
        print(f"{name:<12} {e_drift:<15.2e} {l_drift:<15.2e}")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--compare":
        compare_integrators()
    else:
        run_simulation("yoshida4")
