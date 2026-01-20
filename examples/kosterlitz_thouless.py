"""
Kosterlitz-Thouless Transition - Topological Phase Physics (Level 6)

THE TINYPHYSICS WAY:
    1. Define the XY Hamiltonian: H = -J Σ cos(θ_i - θ_j)
    2. Autograd/analytic gradients compute spin dynamics
    3. Vortices emerge as topological defects (winding number ≠ 0)

This demonstrates:
    - The 2D XY model on a lattice
    - Vortex detection via plaquette winding numbers
    - The KT transition: bound pairs (T < T_KT) → free vortices (T > T_KT)
    - Coulomb gas representation of vortices

The Kosterlitz-Thouless transition (Nobel Prize 2016) is special:
    - No symmetry breaking (Mermin-Wagner theorem)
    - Topological order: vortex-antivortex binding/unbinding
    - Universal jump in helicity modulus at T_KT ≈ 0.89 J
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import XYLatticeSystem, XYVortexGas, create_vortex_pair
import json
import os


def run_xy_lattice_below_kt():
    """XY model below T_KT - vortices remain bound."""
    print("=" * 60)
    print("XY MODEL - Below T_KT (T = 0.5, T_KT ≈ 0.89)")
    print("=" * 60)

    L = 32
    T = 0.5  # Below T_KT ≈ 0.89
    system = XYLatticeSystem(L=L, J=1.0, temperature=T, dynamics="langevin")

    print(f"\nLattice size: {L}×{L}")
    print(f"Temperature: T = {T} < T_KT ≈ {system.T_KT:.2f}")
    print(f"Expected: Vortices bound in pairs, quasi-long-range order")

    # Start from random configuration
    theta = system.random_state()

    print(f"\nInitial state (random):")
    print(f"  Energy per spin: {system.energy_per_spin(theta):.4f} (ground state: -2.0)")
    n_v, n_av = system.vortex_count(theta)
    print(f"  Vortices: {n_v}, Antivortices: {n_av}")

    # Equilibrate
    dt, steps = 0.1, 2000
    theta, history = system.evolve(theta, dt=dt, steps=steps, record_every=20)

    print(f"\nAfter {steps} steps:")
    print(f"  Energy per spin: {system.energy_per_spin(theta):.4f}")
    n_v, n_av = system.vortex_count(theta)
    print(f"  Vortices: {n_v}, Antivortices: {n_av}")
    m, _ = system.magnetization(theta)
    print(f"  |Magnetization|: {m:.4f}")

    return history, system


def run_xy_lattice_above_kt():
    """XY model above T_KT - free vortices proliferate."""
    print("\n" + "=" * 60)
    print("XY MODEL - Above T_KT (T = 1.5, T_KT ≈ 0.89)")
    print("=" * 60)

    L = 32
    T = 1.5  # Above T_KT
    system = XYLatticeSystem(L=L, J=1.0, temperature=T, dynamics="langevin")

    print(f"\nLattice size: {L}×{L}")
    print(f"Temperature: T = {T} > T_KT ≈ {system.T_KT:.2f}")
    print(f"Expected: Free vortices, exponential correlation decay")

    # Start from ordered configuration
    theta = system.ordered_state()

    print(f"\nInitial state (ordered):")
    print(f"  Energy per spin: {system.energy_per_spin(theta):.4f}")
    n_v, n_av = system.vortex_count(theta)
    print(f"  Vortices: {n_v}, Antivortices: {n_av}")

    # Evolve
    dt, steps = 0.1, 2000
    theta, history = system.evolve(theta, dt=dt, steps=steps, record_every=20)

    print(f"\nAfter {steps} steps:")
    print(f"  Energy per spin: {system.energy_per_spin(theta):.4f}")
    n_v, n_av = system.vortex_count(theta)
    print(f"  Vortices: {n_v}, Antivortices: {n_av}")
    print(f"  Vortex density: {system.vortex_density(theta):.4f}")

    return history, system


def run_vortex_pair_below_kt():
    """Vortex pair dynamics below T_KT - pair stays bound."""
    print("\n" + "=" * 60)
    print("VORTEX PAIR - Below T_KT (T = 0.3)")
    print("=" * 60)

    charges, z = create_vortex_pair(separation=3.0)
    system = XYVortexGas(charges, J=1.0, temperature=0.3, gamma=1.0)

    print(f"\nVortex-antivortex pair:")
    print(f"  Charges: {charges.numpy()}")
    print(f"  Temperature: T = 0.3 < T_KT = πJ/2 ≈ {np.pi/2:.2f}")
    print(f"  Initial separation: {system.pair_separation(z):.2f}")
    print(f"  Initial energy: {system.energy(z):.4f}")

    print("\nAt T < T_KT, the pair should remain bound (logarithmic attraction)")

    # Evolve
    dt, steps = 0.01, 2000
    z, history = system.evolve(z, dt=dt, steps=steps, record_every=10)

    print(f"\nAfter {steps} steps:")
    print(f"  Final separation: {system.pair_separation(z):.2f}")
    print(f"  Final energy: {system.energy(z):.4f}")

    return history, system


def run_vortex_pair_above_kt():
    """Vortex pair dynamics above T_KT - pair unbinds."""
    print("\n" + "=" * 60)
    print("VORTEX PAIR - Above T_KT (T = 2.0)")
    print("=" * 60)

    charges, z = create_vortex_pair(separation=3.0)
    system = XYVortexGas(charges, J=1.0, temperature=2.0, gamma=1.0)

    print(f"\nVortex-antivortex pair:")
    print(f"  Charges: {charges.numpy()}")
    print(f"  Temperature: T = 2.0 > T_KT = πJ/2 ≈ {np.pi/2:.2f}")
    print(f"  Initial separation: {system.pair_separation(z):.2f}")
    print(f"  Initial energy: {system.energy(z):.4f}")

    print("\nAt T > T_KT, entropy wins and the pair should unbind (diffuse apart)")

    # Evolve
    dt, steps = 0.01, 2000
    z, history = system.evolve(z, dt=dt, steps=steps, record_every=10)

    print(f"\nAfter {steps} steps:")
    print(f"  Final separation: {system.pair_separation(z):.2f}")
    print(f"  Final energy: {system.energy(z):.4f}")

    return history, system


def run_vortex_pair_dynamics():
    """Study vortex pair as a function of initial configuration."""
    print("\n" + "=" * 60)
    print("VORTEX PAIR - Zero temperature dynamics")
    print("=" * 60)

    charges, z = create_vortex_pair(separation=5.0)
    system = XYVortexGas(charges, J=1.0, temperature=0.0, gamma=1.0)

    print(f"\nVortex-antivortex pair at T = 0:")
    print(f"  Initial separation: {system.pair_separation(z):.2f}")
    print(f"  Total charge: {system.total_charge()} (neutral)")

    print("\nAt T = 0, the pair should collapse together (minimize energy)")

    dt, steps = 0.01, 500
    z, history = system.evolve(z, dt=dt, steps=steps, record_every=5)

    print(f"\nAfter {steps} steps:")
    print(f"  Final separation: {system.pair_separation(z):.2f}")

    return history, system


def generate_viewer(lattice_history, vortex_history, filename="examples/kosterlitz_thouless_viewer.html"):
    """Generate HTML visualization for KT physics."""

    # Process lattice history
    lattice_data = []
    for h in lattice_history:
        lattice_data.append({
            'theta': h['theta'].tolist(),
            'energy': h['energy'],
            'n_vortices': h['n_vortices'],
            'n_antivortices': h['n_antivortices'],
            'vortices': h['vortices'],
            'antivortices': h['antivortices'],
        })

    # Process vortex history
    vortex_data = []
    for h in vortex_history:
        vortex_data.append({
            'positions': h['positions'].tolist(),
            'energy': h['energy'],
            'separation': h['separation'],
        })

    L = len(lattice_data[0]['theta'])

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Kosterlitz-Thouless Transition - TinyPhysics</title>
    <style>
        body {{
            font-family: monospace;
            background: #0a0a1a;
            color: #eee;
            padding: 20px;
        }}
        .container {{ display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }}
        canvas {{ border: 1px solid #333; }}
        .panel {{
            background: #111;
            padding: 15px;
            border-radius: 5px;
            margin: 10px;
        }}
        .panel h3 {{ margin-top: 0; color: #4af; }}
        .controls {{ margin: 10px 0; }}
        button {{ padding: 8px 16px; margin: 0 5px; cursor: pointer; background: #333; color: #fff; border: 1px solid #555; }}
        button:hover {{ background: #444; }}
        .info {{ font-size: 12px; margin-top: 10px; }}
        code {{ color: #4f4; }}
        h1 {{ text-align: center; color: #4af; }}
        h2 {{ color: #f84; border-bottom: 1px solid #333; padding-bottom: 5px; }}
    </style>
</head>
<body>
    <h1>Kosterlitz-Thouless Transition</h1>

    <div class="panel" style="max-width: 800px; margin: 0 auto;">
        <h3>The Physics</h3>
        <p>The <b>XY Model</b> on a 2D lattice with Hamiltonian:</p>
        <code>H = -J Σ cos(θ_i - θ_j)</code>
        <p>Features a <b>topological phase transition</b> at T_KT ≈ 0.89 J:</p>
        <ul>
            <li><b>T &lt; T_KT:</b> Vortex-antivortex pairs bound, quasi-long-range order</li>
            <li><b>T &gt; T_KT:</b> Pairs unbind, free vortices proliferate, disorder</li>
        </ul>
        <p>Vortices are detected by the <b>winding number</b> around each plaquette.</p>
    </div>

    <div class="container">
        <div class="panel">
            <h2>XY Lattice Model (T = 0.5 &lt; T_KT)</h2>
            <div class="controls">
                <button onclick="togglePlay('lattice')">Play/Pause</button>
                <button onclick="reset('lattice')">Reset</button>
                <span id="latticeFrame">Frame: 0</span>
            </div>
            <canvas id="latticeCanvas" width="400" height="400"></canvas>
            <div class="info">
                <div>Energy: <span id="latticeEnergy">-</span></div>
                <div>Vortices: <span id="latticeVortices">-</span></div>
                <div><span style="color:#ff4444">●</span> Vortex (+1)
                     <span style="color:#4444ff">●</span> Antivortex (-1)</div>
            </div>
        </div>

        <div class="panel">
            <h2>Vortex Pair (T = 0, collapse)</h2>
            <div class="controls">
                <button onclick="togglePlay('vortex')">Play/Pause</button>
                <button onclick="reset('vortex')">Reset</button>
                <span id="vortexFrame">Frame: 0</span>
            </div>
            <canvas id="vortexCanvas" width="400" height="400"></canvas>
            <div class="info">
                <div>Separation: <span id="vortexSep">-</span></div>
                <div>Energy: <span id="vortexEnergy">-</span></div>
                <div>Logarithmic attraction: E ~ -πJ log(r)</div>
            </div>
        </div>
    </div>

    <script>
        const latticeData = {json.dumps(lattice_data)};
        const vortexData = {json.dumps(vortex_data)};
        const L = {L};

        // Animation state
        let latticeFrame = 0, vortexFrame = 0;
        let latticePlaying = true, vortexPlaying = true;

        // Lattice canvas
        const lCanvas = document.getElementById('latticeCanvas');
        const lCtx = lCanvas.getContext('2d');
        const lScale = lCanvas.width / L;

        // Vortex canvas
        const vCanvas = document.getElementById('vortexCanvas');
        const vCtx = vCanvas.getContext('2d');

        function hslToRgb(h, s, l) {{
            let r, g, b;
            if (s === 0) {{ r = g = b = l; }}
            else {{
                const hue2rgb = (p, q, t) => {{
                    if (t < 0) t += 1;
                    if (t > 1) t -= 1;
                    if (t < 1/6) return p + (q - p) * 6 * t;
                    if (t < 1/2) return q;
                    if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
                    return p;
                }};
                const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
                const p = 2 * l - q;
                r = hue2rgb(p, q, h + 1/3);
                g = hue2rgb(p, q, h);
                b = hue2rgb(p, q, h - 1/3);
            }}
            return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
        }}

        function drawLattice() {{
            const data = latticeData[latticeFrame];
            const theta = data.theta;

            // Draw spin angles as colors
            for (let i = 0; i < L; i++) {{
                for (let j = 0; j < L; j++) {{
                    const angle = theta[i][j];
                    // Map angle to hue (0-360 degrees)
                    const hue = angle / (2 * Math.PI);
                    const [r, g, b] = hslToRgb(hue, 0.8, 0.5);
                    lCtx.fillStyle = `rgb(${{r}},${{g}},${{b}})`;
                    lCtx.fillRect(j * lScale, i * lScale, lScale, lScale);
                }}
            }}

            // Draw vortices
            lCtx.strokeStyle = '#fff';
            lCtx.lineWidth = 2;

            data.vortices.forEach(([x, y]) => {{
                lCtx.fillStyle = '#ff4444';
                lCtx.beginPath();
                lCtx.arc((y + 0.5) * lScale, (x + 0.5) * lScale, 5, 0, 2 * Math.PI);
                lCtx.fill();
                lCtx.stroke();
            }});

            data.antivortices.forEach(([x, y]) => {{
                lCtx.fillStyle = '#4444ff';
                lCtx.beginPath();
                lCtx.arc((y + 0.5) * lScale, (x + 0.5) * lScale, 5, 0, 2 * Math.PI);
                lCtx.fill();
                lCtx.stroke();
            }});

            document.getElementById('latticeFrame').textContent = `Frame: ${{latticeFrame}}/${{latticeData.length - 1}}`;
            document.getElementById('latticeEnergy').textContent = data.energy.toFixed(4);
            document.getElementById('latticeVortices').textContent = `${{data.n_vortices}} + / ${{data.n_antivortices}} -`;
        }}

        function drawVortex() {{
            const data = vortexData[vortexFrame];
            const pos = data.positions;
            const cx = vCanvas.width / 2;
            const cy = vCanvas.height / 2;
            const scale = 40;

            // Clear
            vCtx.fillStyle = '#000';
            vCtx.fillRect(0, 0, vCanvas.width, vCanvas.height);

            // Draw trail
            vCtx.strokeStyle = '#333';
            vCtx.lineWidth = 1;
            for (let v = 0; v < 2; v++) {{
                vCtx.beginPath();
                for (let t = Math.max(0, vortexFrame - 50); t <= vortexFrame; t++) {{
                    const p = vortexData[t].positions[v];
                    const x = cx + p[0] * scale;
                    const y = cy - p[1] * scale;
                    if (t === Math.max(0, vortexFrame - 50)) vCtx.moveTo(x, y);
                    else vCtx.lineTo(x, y);
                }}
                vCtx.stroke();
            }}

            // Draw vortices
            const colors = ['#ff4444', '#4444ff'];
            for (let v = 0; v < 2; v++) {{
                const x = cx + pos[v][0] * scale;
                const y = cy - pos[v][1] * scale;
                vCtx.fillStyle = colors[v];
                vCtx.beginPath();
                vCtx.arc(x, y, 12, 0, 2 * Math.PI);
                vCtx.fill();
                vCtx.strokeStyle = '#fff';
                vCtx.lineWidth = 2;
                vCtx.stroke();

                // Label
                vCtx.fillStyle = '#fff';
                vCtx.font = '14px monospace';
                vCtx.fillText(v === 0 ? '+1' : '-1', x - 8, y + 4);
            }}

            // Draw connecting line
            vCtx.strokeStyle = '#666';
            vCtx.setLineDash([5, 5]);
            vCtx.beginPath();
            vCtx.moveTo(cx + pos[0][0] * scale, cy - pos[0][1] * scale);
            vCtx.lineTo(cx + pos[1][0] * scale, cy - pos[1][1] * scale);
            vCtx.stroke();
            vCtx.setLineDash([]);

            document.getElementById('vortexFrame').textContent = `Frame: ${{vortexFrame}}/${{vortexData.length - 1}}`;
            document.getElementById('vortexSep').textContent = data.separation.toFixed(3);
            document.getElementById('vortexEnergy').textContent = data.energy.toFixed(4);
        }}

        function animate() {{
            if (latticePlaying && latticeData.length > 0) {{
                latticeFrame = (latticeFrame + 1) % latticeData.length;
            }}
            if (vortexPlaying && vortexData.length > 0) {{
                vortexFrame = (vortexFrame + 1) % vortexData.length;
            }}
            drawLattice();
            drawVortex();
            requestAnimationFrame(animate);
        }}

        function togglePlay(which) {{
            if (which === 'lattice') latticePlaying = !latticePlaying;
            else vortexPlaying = !vortexPlaying;
        }}

        function reset(which) {{
            if (which === 'lattice') latticeFrame = 0;
            else vortexFrame = 0;
        }}

        animate();
    </script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"\nViewer saved: {os.path.abspath(filename)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "below":
            history, _ = run_xy_lattice_below_kt()
            vortex_history, _ = run_vortex_pair_dynamics()
        elif mode == "above":
            history, _ = run_xy_lattice_above_kt()
            vortex_history, _ = run_vortex_pair_above_kt()
        elif mode == "pair":
            history, _ = run_xy_lattice_below_kt()
            vortex_history, _ = run_vortex_pair_dynamics()
        else:
            print(f"Unknown mode: {mode}. Use: below, above, pair")
            sys.exit(1)
    else:
        # Default: run all demos
        print("\n" + "=" * 60)
        print("KOSTERLITZ-THOULESS PHYSICS")
        print("=" * 60)
        print("\nThe KT transition is a topological phase transition where")
        print("vortex-antivortex pairs unbind at T_KT ≈ 0.89 J.")
        print("\nNobel Prize 2016: Kosterlitz, Thouless, Haldane")

        # Run lattice simulation below T_KT
        lattice_history, _ = run_xy_lattice_below_kt()

        # Run above T_KT for comparison
        run_xy_lattice_above_kt()

        # Run vortex pair simulations
        vortex_history, _ = run_vortex_pair_dynamics()
        run_vortex_pair_below_kt()
        run_vortex_pair_above_kt()

        # Generate viewer
        generate_viewer(lattice_history, vortex_history)

    print("\n" + "=" * 60)
    print("KEY OBSERVATIONS")
    print("=" * 60)
    print("""
1. Below T_KT (T < 0.89 J):
   - Few free vortices
   - Vortex-antivortex pairs stay bound
   - Quasi-long-range order (power-law correlations)

2. Above T_KT (T > 0.89 J):
   - Many free vortices proliferate
   - Pairs unbind (entropy wins over energy)
   - Exponential decay of correlations

3. The transition is TOPOLOGICAL:
   - No symmetry breaking (Mermin-Wagner theorem)
   - Order parameter is the vortex density
   - Universal jump in helicity modulus at T_KT
""")
