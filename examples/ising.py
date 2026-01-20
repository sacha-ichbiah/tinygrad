"""
Ising Model - The Hello World of Statistical Mechanics (Level 5)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H = -J Σ s_i s_j - h Σ s_i
    2. Use Monte Carlo (Metropolis/Wolff) to sample Boltzmann distribution
    3. Observe the ferromagnetic phase transition!

This demonstrates:
    - 2D Ising model on a square lattice
    - Metropolis and Wolff cluster algorithms
    - Phase transition at T_c ≈ 2.269 (Onsager's exact solution)
    - Critical phenomena: diverging susceptibility, specific heat
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import IsingSystem
import json
import os


def run_phase_transition():
    """Observe the ferromagnetic phase transition by sweeping temperature."""
    print("=" * 60)
    print("ISING MODEL - Phase Transition")
    print("=" * 60)

    L = 32  # Lattice size
    print(f"\nLattice: {L} × {L} = {L*L} spins")
    print(f"Hamiltonian: H = -J Σ s_i s_j")
    print(f"Critical temperature: T_c = 2/ln(1+√2) ≈ {IsingSystem.T_CRITICAL:.3f}")

    # Temperature sweep
    T_values = np.linspace(1.5, 3.5, 21)
    results = []

    system = IsingSystem(L=L, J=1.0, h=0.0, T=2.0, algorithm="wolff")
    spins = system.random_spins()

    print(f"\nSweeping temperature from T={T_values[0]:.1f} to T={T_values[-1]:.1f}...")
    print(f"Algorithm: {system.algorithm_name}")

    for T in T_values:
        system.set_temperature(T)
        # Measure thermal averages
        data = system.measure(spins, steps=500, warmup=200)
        results.append(data)
        print(f"  T = {T:.2f}: |m| = {data['M_mean']:.3f}, "
              f"E/N = {data['E_mean']:.3f}, χ = {data['susceptibility']:.2f}")

    print("\n" + "-" * 40)
    print("Near T_c, observe:")
    print("  - Magnetization drops from ~1 to ~0")
    print("  - Susceptibility χ diverges")
    print("  - Specific heat peaks")

    return results, L


def run_equilibration():
    """Watch the system equilibrate from a random (hot) start."""
    print("\n" + "=" * 60)
    print("ISING MODEL - Equilibration Dynamics")
    print("=" * 60)

    L = 48
    T = 2.0  # Below T_c (ordered phase)

    print(f"\nLattice: {L} × {L}")
    print(f"Temperature: T = {T:.2f} (below T_c ≈ 2.27)")
    print(f"Starting from random (hot) configuration...")

    system = IsingSystem(L=L, J=1.0, T=T, algorithm="wolff")
    spins = system.random_spins()

    E0 = system.energy_per_spin(spins)
    M0 = system.magnetization_abs(spins)
    print(f"\nInitial: E/N = {E0:.3f}, |m| = {M0:.3f}")

    # Evolve and record
    steps = 500
    spins_final, history = system.evolve(spins, steps=steps, record_every=1)

    Ef = system.energy_per_spin(spins_final)
    Mf = system.magnetization_abs(spins_final)
    print(f"Final:   E/N = {Ef:.3f}, |m| = {Mf:.3f}")

    print(f"\nThe system spontaneously magnetized!")
    print(f"This is spontaneous symmetry breaking.")

    return history, L, T


def run_critical_slowing():
    """Compare Metropolis vs Wolff at critical temperature."""
    print("\n" + "=" * 60)
    print("ISING MODEL - Critical Slowing Down")
    print("=" * 60)

    L = 32
    T = IsingSystem.T_CRITICAL

    print(f"\nAt T = T_c ≈ {T:.3f}, correlation length diverges.")
    print(f"Metropolis becomes slow (critical slowing down).")
    print(f"Wolff cluster algorithm remains efficient!")

    # Compare algorithms
    for algo in ["metropolis", "wolff"]:
        system = IsingSystem(L=L, J=1.0, T=T, algorithm=algo)
        spins = system.uniform_spins(1)  # Start ordered

        # Track decorrelation
        m_values = []
        for _ in range(200):
            spins = system.step(spins)
            m_values.append(system.magnetization(spins))

        m_arr = np.array(m_values)
        # Autocorrelation at lag 1
        autocorr = np.corrcoef(m_arr[:-1], m_arr[1:])[0, 1]
        print(f"  {algo:12s}: final |m| = {abs(m_values[-1]):.3f}, "
              f"autocorr(1) = {autocorr:.3f}")


def run_external_field():
    """Effect of external magnetic field."""
    print("\n" + "=" * 60)
    print("ISING MODEL - External Field")
    print("=" * 60)

    L = 32
    T = 3.0  # Above T_c

    print(f"\nAbove T_c (T = {T:.1f}), system is paramagnetic.")
    print(f"External field h aligns spins.")

    h_values = [0.0, 0.1, 0.5, 1.0]
    for h in h_values:
        system = IsingSystem(L=L, J=1.0, h=h, T=T, algorithm="metropolis")
        spins = system.random_spins()

        # Equilibrate
        for _ in range(300):
            spins = system.step(spins)

        m = system.magnetization(spins)
        print(f"  h = {h:.1f}: m = {m:+.3f}")


def generate_viewer(history, L, T, filename="examples/ising_viewer.html"):
    """Generate HTML visualization."""
    # Extract data
    spin_configs = [h[0].tolist() for h in history]
    energies = [h[1] for h in history]
    magnetizations = [h[2] for h in history]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ising Model - TinyPhysics</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            margin: 0;
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{ color: #fff; font-weight: 300; margin-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; font-size: 14px; }}
        .container {{ display: flex; gap: 30px; flex-wrap: wrap; justify-content: center; }}
        .panel {{
            background: rgba(0, 0, 0, 0.4);
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        canvas {{ border-radius: 5px; }}
        .info {{
            max-width: 350px;
        }}
        .info h3 {{ color: #f4a; margin: 0 0 15px 0; font-size: 14px; text-transform: uppercase; }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .stat-label {{ color: #888; }}
        .stat-value {{ color: #fff; font-family: monospace; }}
        .controls {{
            margin-top: 15px;
            display: flex;
            gap: 10px;
            align-items: center;
        }}
        button {{
            background: #f4a;
            border: none;
            color: #fff;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-weight: bold;
        }}
        button:hover {{ background: #f5b; }}
        .speed-control {{
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
        }}
        input[type="range"] {{ width: 100px; accent-color: #f4a; }}
        .legend {{
            display: flex;
            gap: 20px;
            margin: 15px 0;
            font-size: 14px;
        }}
        .legend span {{ display: flex; align-items: center; gap: 5px; }}
        .legend .spin-up {{ width: 15px; height: 15px; background: #ff6b6b; border-radius: 2px; }}
        .legend .spin-down {{ width: 15px; height: 15px; background: #4ecdc4; border-radius: 2px; }}
        .phase-info {{
            background: rgba(255, 68, 170, 0.1);
            border-left: 3px solid #f4a;
            padding: 10px 15px;
            margin: 15px 0;
            border-radius: 0 5px 5px 0;
        }}
        .phase-info .title {{ color: #f4a; font-weight: bold; font-size: 12px; }}
        .phase-info .desc {{ color: #aaa; font-size: 12px; margin-top: 4px; }}
        .formula {{
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 8px 12px;
            border-radius: 4px;
            color: #4f4;
            display: inline-block;
            margin: 5px 0;
        }}
    </style>
</head>
<body>
    <h1>2D Ising Model</h1>
    <div class="subtitle">Monte Carlo simulation of ferromagnetic spins</div>

    <div class="container">
        <div class="panel">
            <canvas id="spinCanvas" width="400" height="400"></canvas>
            <div class="legend">
                <span><div class="spin-up"></div> Spin +1</span>
                <span><div class="spin-down"></div> Spin -1</span>
            </div>
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

        <div class="panel info">
            <h3>Simulation</h3>
            <div class="stat-row">
                <span class="stat-label">Lattice</span>
                <span class="stat-value">{L} × {L}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Temperature T</span>
                <span class="stat-value">{T:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">T / T_c</span>
                <span class="stat-value">{T/IsingSystem.T_CRITICAL:.2f}</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Frame</span>
                <span class="stat-value" id="frameValue">0</span>
            </div>

            <h3 style="margin-top: 20px;">Observables</h3>
            <div class="stat-row">
                <span class="stat-label">Energy E/N</span>
                <span class="stat-value" id="energyValue">-</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Magnetization |m|</span>
                <span class="stat-value" id="magValue">-</span>
            </div>

            <div class="phase-info">
                <div class="title">Phase Transition</div>
                <div class="desc">
                    {"T < T_c: Ordered (ferromagnetic)" if T < IsingSystem.T_CRITICAL else "T > T_c: Disordered (paramagnetic)"}
                </div>
                <div class="formula">T_c = 2/ln(1+√2) ≈ 2.269</div>
            </div>

            <canvas id="plotCanvas" width="300" height="150" style="margin-top: 15px;"></canvas>
        </div>
    </div>

    <script>
        const spinConfigs = {json.dumps(spin_configs)};
        const energies = {json.dumps(energies)};
        const magnetizations = {json.dumps(magnetizations)};
        const L = {L};
        const totalFrames = spinConfigs.length;

        let frame = 0;
        let playing = true;
        let speed = 3;

        const spinCanvas = document.getElementById('spinCanvas');
        const ctx = spinCanvas.getContext('2d');
        const cellSize = spinCanvas.width / L;

        const plotCanvas = document.getElementById('plotCanvas');
        const pctx = plotCanvas.getContext('2d');

        document.getElementById('playBtn').onclick = () => {{
            playing = !playing;
            document.getElementById('playBtn').textContent = playing ? 'Pause' : 'Play';
        }};
        document.getElementById('resetBtn').onclick = () => {{ frame = 0; }};
        document.getElementById('speedSlider').oninput = (e) => {{
            speed = parseInt(e.target.value);
            document.getElementById('speedLabel').textContent = speed + 'x';
        }};

        function drawSpins() {{
            const spins = spinConfigs[frame];
            for (let i = 0; i < L; i++) {{
                for (let j = 0; j < L; j++) {{
                    ctx.fillStyle = spins[i][j] > 0 ? '#ff6b6b' : '#4ecdc4';
                    ctx.fillRect(j * cellSize, i * cellSize, cellSize, cellSize);
                }}
            }}
        }}

        function drawPlot() {{
            pctx.fillStyle = '#111';
            pctx.fillRect(0, 0, plotCanvas.width, plotCanvas.height);

            // Draw magnetization history
            pctx.strokeStyle = '#f4a';
            pctx.lineWidth = 2;
            pctx.beginPath();
            const maxFrames = Math.min(frame + 1, magnetizations.length);
            for (let i = 0; i < maxFrames; i++) {{
                const x = (i / totalFrames) * plotCanvas.width;
                const y = plotCanvas.height - magnetizations[i] * plotCanvas.height;
                if (i === 0) pctx.moveTo(x, y);
                else pctx.lineTo(x, y);
            }}
            pctx.stroke();

            // Labels
            pctx.fillStyle = '#888';
            pctx.font = '10px sans-serif';
            pctx.fillText('|m|', 5, 12);
            pctx.fillText('1', 5, 15);
            pctx.fillText('0', 5, plotCanvas.height - 3);
        }}

        function updateStats() {{
            document.getElementById('frameValue').textContent = frame + ' / ' + (totalFrames - 1);
            document.getElementById('energyValue').textContent = energies[frame].toFixed(3);
            document.getElementById('magValue').textContent = magnetizations[frame].toFixed(3);
        }}

        function animate() {{
            drawSpins();
            drawPlot();
            updateStats();
            if (playing) frame = (frame + speed) % totalFrames;
            requestAnimationFrame(animate);
        }}

        animate();
    </script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"\nViewer: {os.path.abspath(filename)}")


def generate_phase_viewer(results, L, filename="examples/ising_phase_viewer.html"):
    """Generate HTML visualization of phase transition."""
    T_values = [r["temperature"] for r in results]
    M_values = [r["M_mean"] for r in results]
    E_values = [r["E_mean"] for r in results]
    C_values = [r["specific_heat"] for r in results]
    chi_values = [r["susceptibility"] for r in results]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Ising Phase Transition - TinyPhysics</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            margin: 0;
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}
        h1 {{ color: #fff; font-weight: 300; margin-bottom: 5px; }}
        .subtitle {{ color: #888; margin-bottom: 20px; font-size: 14px; }}
        .container {{ display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }}
        canvas {{
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .info {{
            background: rgba(0,0,0,0.4);
            border-radius: 10px;
            padding: 20px;
            max-width: 400px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.5);
        }}
        .info h3 {{ color: #f4a; margin: 0 0 15px 0; }}
        .formula {{
            font-family: monospace;
            background: rgba(0,0,0,0.3);
            padding: 8px 12px;
            border-radius: 4px;
            color: #4f4;
            display: block;
            margin: 10px 0;
        }}
        .critical {{ color: #ff0; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Ising Model Phase Transition</h1>
    <div class="subtitle">2D square lattice, {L} × {L} spins</div>

    <div class="container">
        <canvas id="magCanvas" width="400" height="300"></canvas>
        <canvas id="susCanvas" width="400" height="300"></canvas>
    </div>

    <div class="info" style="margin-top: 20px; max-width: 820px;">
        <h3>The Onsager Solution (1944)</h3>
        <p>The 2D Ising model has an exact solution showing a continuous phase transition.</p>
        <div class="formula">T_c = 2J / ln(1 + √2) ≈ 2.269 J/k_B</div>
        <p>
            <strong>T &lt; T_c:</strong> Ordered phase (ferromagnetic), |m| ≈ 1<br>
            <strong>T &gt; T_c:</strong> Disordered phase (paramagnetic), m ≈ 0
        </p>
        <p>At T_c, the susceptibility χ and correlation length ξ diverge - this is the critical point.</p>
        <p class="critical">Critical temperature T_c ≈ {IsingSystem.T_CRITICAL:.3f}</p>
    </div>

    <script>
        const T = {json.dumps(T_values)};
        const M = {json.dumps(M_values)};
        const chi = {json.dumps(chi_values)};
        const Tc = {IsingSystem.T_CRITICAL};

        function drawPlot(canvasId, data, yLabel, color) {{
            const canvas = document.getElementById(canvasId);
            const ctx = canvas.getContext('2d');
            const w = canvas.width, h = canvas.height;
            const pad = 50;

            ctx.fillStyle = 'transparent';
            ctx.fillRect(0, 0, w, h);

            // Find data range
            const xMin = Math.min(...T), xMax = Math.max(...T);
            const yMin = 0, yMax = Math.max(...data) * 1.1;

            // Draw axes
            ctx.strokeStyle = '#444';
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(pad, pad);
            ctx.lineTo(pad, h - pad);
            ctx.lineTo(w - pad, h - pad);
            ctx.stroke();

            // Draw T_c vertical line
            const tcX = pad + ((Tc - xMin) / (xMax - xMin)) * (w - 2*pad);
            ctx.strokeStyle = '#ff0';
            ctx.setLineDash([5, 5]);
            ctx.beginPath();
            ctx.moveTo(tcX, pad);
            ctx.lineTo(tcX, h - pad);
            ctx.stroke();
            ctx.setLineDash([]);

            // Labels
            ctx.fillStyle = '#888';
            ctx.font = '12px sans-serif';
            ctx.fillText('T', w - pad + 10, h - pad + 5);
            ctx.fillText(yLabel, pad - 5, pad - 10);
            ctx.fillStyle = '#ff0';
            ctx.fillText('T_c', tcX - 10, pad - 5);

            // Draw data
            ctx.strokeStyle = color;
            ctx.lineWidth = 3;
            ctx.beginPath();
            for (let i = 0; i < T.length; i++) {{
                const x = pad + ((T[i] - xMin) / (xMax - xMin)) * (w - 2*pad);
                const y = h - pad - ((data[i] - yMin) / (yMax - yMin)) * (h - 2*pad);
                if (i === 0) ctx.moveTo(x, y);
                else ctx.lineTo(x, y);
            }}
            ctx.stroke();

            // Draw points
            ctx.fillStyle = color;
            for (let i = 0; i < T.length; i++) {{
                const x = pad + ((T[i] - xMin) / (xMax - xMin)) * (w - 2*pad);
                const y = h - pad - ((data[i] - yMin) / (yMax - yMin)) * (h - 2*pad);
                ctx.beginPath();
                ctx.arc(x, y, 4, 0, 2 * Math.PI);
                ctx.fill();
            }}
        }}

        drawPlot('magCanvas', M, '|m|', '#f4a');
        drawPlot('susCanvas', chi, 'χ', '#4af');
    </script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"\nPhase transition viewer: {os.path.abspath(filename)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "phase":
            results, L = run_phase_transition()
            generate_phase_viewer(results, L)
        elif mode == "equilibrate":
            history, L, T = run_equilibration()
            generate_viewer(history, L, T)
        elif mode == "critical":
            run_critical_slowing()
        elif mode == "field":
            run_external_field()
        else:
            print(f"Unknown mode: {mode}. Use: phase, equilibrate, critical, field")
            sys.exit(1)
    else:
        # Default: run equilibration demo with viewer
        history, L, T = run_equilibration()
        run_phase_transition()
        run_critical_slowing()
        run_external_field()
        generate_viewer(history, L, T)
