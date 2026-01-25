"""
Point Vortex Dynamics - The Hello World of Fluid Mechanics (Level 3.1)

THE TINYPHYSICS WAY (Blueprint-Compliant):
    1. Define the Hamiltonian H = -1/(4π) Σ Γ_i Γ_j log|r_ij|
    2. Create PhysicalSystem with PointVortexStructure (Lie-Poisson)
    3. Compile to get a StructureProgram
    4. Autograd computes all pairwise interaction forces

This demonstrates:
    - Kirchhoff's point vortex equations
    - Conservation of energy, linear momentum, angular momentum
    - Classic configurations: co-rotating pair, vortex dipole, vortex street
"""

import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.physics import point_vortex_hamiltonian
from tinyphysics import PhysicalSystem
from tinyphysics.systems.vortices import PointVortexStructure
import json
import os


def run_vortex_pair():
    """Two co-rotating vortices orbit their center of vorticity."""
    print("=" * 60)
    print("VORTEX PAIR - Two co-rotating vortices")
    print("=" * 60)

    # Two vortices with equal circulation
    gamma = Tensor([1.0, 1.0])

    # Initial positions: symmetric about origin
    z = Tensor([
        -0.5, 0.0,  # vortex 0 at (-0.5, 0)
         0.5, 0.0,  # vortex 1 at (0.5, 0)
    ])

    # CREATE THE HAMILTONIAN
    H = point_vortex_hamiltonian(gamma)

    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H = -1/(4π) Σ Γ_i Γ_j log|r_ij|")
    print(f"\nStructure: PointVortexStructure (Lie-Poisson)")
    print(f"  bracket(z, grad) -> (grad_y/Γ, -grad_x/Γ)")
    print(f"\nKirchhoff's equations derived via autograd:")
    print(f"  Γ_i dx_i/dt = +∂H/∂y_i")
    print(f"  Γ_i dy_i/dt = -∂H/∂x_i")

    # Initial conserved quantities
    E0 = float(H(z).numpy())
    n_vortices = len(gamma.numpy())
    pos0 = z.numpy().reshape(n_vortices, 2)
    gamma_np = gamma.numpy()
    px0 = float((gamma_np * pos0[:, 0]).sum())
    py0 = float((gamma_np * pos0[:, 1]).sum())
    L0 = float((gamma_np * (pos0 * pos0).sum(axis=1)).sum())

    print(f"\nInitial Energy: {E0:.6f}")
    print(f"Initial Momentum: ({px0:.6f}, {py0:.6f})")
    print(f"Initial Angular Momentum: {L0:.6f}")

    # BLUEPRINT API: Create PhysicalSystem with explicit Structure
    system = PhysicalSystem(
        state=z,
        H_func=H,
        structure=PointVortexStructure(gamma)
    )

    # COMPILE: Returns a StructureProgram
    prog = system.compile()
    print(f"\nCompiled: {type(prog.program).__name__}")

    # EVOLVE with explicit history recording
    dt, steps, record_every = 0.05, 100, 2
    history = []

    for i in range(steps):
        if i % record_every == 0:
            z_np = z.numpy().copy()
            E_i = float(H(z).numpy())
            history.append((z_np, E_i))
        z = prog.step(z, dt)

    # Record final state
    z_np = z.numpy().copy()
    E_final = float(H(z).numpy())
    history.append((z_np, E_final))

    # Final state
    pos1 = z_np.reshape(n_vortices, 2)
    px_final = float((gamma_np * pos1[:, 0]).sum())
    py_final = float((gamma_np * pos1[:, 1]).sum())
    L_final = float((gamma_np * (pos1 * pos1).sum(axis=1)).sum())

    print(f"\nFinal Energy: {E_final:.6f}")
    print(f"Final Momentum: ({px_final:.6f}, {py_final:.6f})")
    print(f"Final Angular Momentum: {L_final:.6f}")
    E_drift = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-10 else abs(E_final - E0)
    L_drift = abs(L_final - L0) / abs(L0) if abs(L0) > 1e-10 else abs(L_final - L0)
    print(f"\nEnergy Drift: {E_drift:.2e}")
    print(f"Angular Momentum Drift: {L_drift:.2e}")

    return history, gamma.numpy()


def run_vortex_dipole():
    """Vortex-antivortex pair translates in a straight line."""
    print("\n" + "=" * 60)
    print("VORTEX DIPOLE - Vortex + Antivortex")
    print("=" * 60)

    # Opposite circulations
    gamma = Tensor([1.0, -1.0])

    # Symmetric about origin
    z = Tensor([
        0.0, -0.5,  # vortex at (0, -0.5)
        0.0,  0.5,  # antivortex at (0, 0.5)
    ])

    # CREATE THE HAMILTONIAN
    H = point_vortex_hamiltonian(gamma)

    print(f"\nA vortex-antivortex pair translates together.")
    print(f"The pair moves perpendicular to the line joining them.")

    E0 = float(H(z).numpy())
    n_vortices = len(gamma.numpy())
    pos0 = z.numpy().reshape(n_vortices, 2)
    gamma_np = gamma.numpy()
    px0 = float((gamma_np * pos0[:, 0]).sum())
    py0 = float((gamma_np * pos0[:, 1]).sum())

    print(f"\nInitial Energy: {E0:.6f}")
    print(f"Initial Momentum: ({px0:.6f}, {py0:.6f})")

    # BLUEPRINT API: Create PhysicalSystem with explicit Structure
    system = PhysicalSystem(
        state=z,
        H_func=H,
        structure=PointVortexStructure(gamma)
    )

    # COMPILE: Returns a StructureProgram
    prog = system.compile()

    # EVOLVE with explicit history recording
    dt, steps, record_every = 0.05, 100, 2
    history = []

    for i in range(steps):
        if i % record_every == 0:
            z_np = z.numpy().copy()
            E_i = float(H(z).numpy())
            history.append((z_np, E_i))
        z = prog.step(z, dt)

    # Record final state
    z_np = z.numpy().copy()
    E_final = float(H(z).numpy())
    history.append((z_np, E_final))

    print(f"Final Energy: {E_final:.6f}")
    E_drift = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-10 else abs(E_final - E0)
    print(f"Energy Drift: {E_drift:.2e}")

    # Show translation
    z0 = history[0][0].reshape(-1, 2)
    zf = z_np.reshape(-1, 2)
    print(f"\nVortex 0 moved from ({z0[0,0]:.2f}, {z0[0,1]:.2f}) to ({zf[0,0]:.2f}, {zf[0,1]:.2f})")

    return history, gamma.numpy()


def run_three_vortices():
    """Three vortices - complex quasi-periodic motion."""
    print("\n" + "=" * 60)
    print("THREE VORTICES - Complex dynamics")
    print("=" * 60)

    # Three vortices with equal circulations
    gamma = Tensor([1.0, 1.0, 1.0])

    # Equilateral triangle configuration
    r = 1.0
    z = Tensor([
        r * np.cos(0), r * np.sin(0),
        r * np.cos(2*np.pi/3), r * np.sin(2*np.pi/3),
        r * np.cos(4*np.pi/3), r * np.sin(4*np.pi/3),
    ])

    # CREATE THE HAMILTONIAN
    H = point_vortex_hamiltonian(gamma)

    print(f"\nThree equal vortices in equilateral triangle")
    print(f"They rotate rigidly about the center of vorticity")

    E0 = float(H(z).numpy())
    n_vortices = len(gamma.numpy())
    pos0 = z.numpy().reshape(n_vortices, 2)
    gamma_np = gamma.numpy()
    L0 = float((gamma_np * (pos0 * pos0).sum(axis=1)).sum())

    print(f"\nInitial Energy: {E0:.6f}")
    print(f"Initial Angular Momentum: {L0:.6f}")

    # BLUEPRINT API: Create PhysicalSystem with explicit Structure
    system = PhysicalSystem(
        state=z,
        H_func=H,
        structure=PointVortexStructure(gamma)
    )

    # COMPILE: Returns a StructureProgram
    prog = system.compile()

    # EVOLVE with explicit history recording
    dt, steps, record_every = 0.05, 100, 2
    history = []

    for i in range(steps):
        if i % record_every == 0:
            z_np = z.numpy().copy()
            E_i = float(H(z).numpy())
            history.append((z_np, E_i))
        z = prog.step(z, dt)

    # Record final state
    z_np = z.numpy().copy()
    E_final = float(H(z).numpy())
    history.append((z_np, E_final))

    # Final state
    pos1 = z_np.reshape(n_vortices, 2)
    L_final = float((gamma_np * (pos1 * pos1).sum(axis=1)).sum())

    print(f"\nFinal Energy: {E_final:.6f}")
    print(f"Final Angular Momentum: {L_final:.6f}")
    E_drift = abs(E_final - E0) / abs(E0) if abs(E0) > 1e-10 else abs(E_final - E0)
    L_drift = abs(L_final - L0) / abs(L0) if abs(L0) > 1e-10 else abs(L_final - L0)
    print(f"Energy Drift: {E_drift:.2e}")
    print(f"Angular Momentum Drift: {L_drift:.2e}")

    return history, gamma.numpy()


def generate_viewer(history, gamma, filename="examples/point_vortex_viewer.html"):
    """Generate HTML visualization."""
    n_vortices = len(gamma)
    # history is list of (z_numpy, energy) tuples
    trajectories = [h[0].reshape(n_vortices, 2).tolist() for h in history]
    energies = [h[1] for h in history]

    # Determine colors based on circulation sign
    colors = ['#ff4444' if g > 0 else '#4444ff' for g in gamma]

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Point Vortex Dynamics - TinyPhysics</title>
    <style>
        body {{
            font-family: monospace;
            background: #0a0a1a;
            color: #eee;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }}
        .container {{ display: flex; gap: 20px; flex-wrap: wrap; justify-content: center; }}
        canvas {{ border: 1px solid #333; background: #000; }}
        .info {{
            background: #111;
            padding: 15px;
            border-radius: 5px;
            max-width: 500px;
        }}
        .info h3 {{ margin-top: 0; color: #4af; }}
        code {{ color: #4f4; }}
        .controls {{ margin: 10px; }}
        button {{ padding: 8px 16px; margin: 0 5px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>Point Vortex Dynamics - Kirchhoff's Equations</h1>
    <div class="info">
        <h3>The TinyPhysics Way</h3>
        <p>Define ONLY the Hamiltonian:</p>
        <code>H = -1/(4π) Σ Γ_i Γ_j log|r_ij|</code>
        <p>Kirchhoff's equations derived automatically via autograd!</p>
        <p><span style="color:#ff4444">●</span> Positive circulation &nbsp;
           <span style="color:#4444ff">●</span> Negative circulation</p>
    </div>
    <div class="controls">
        <button onclick="togglePlay()">Play/Pause</button>
        <button onclick="reset()">Reset</button>
        <span id="frameInfo">Frame: 0</span>
    </div>
    <div class="container">
        <canvas id="vortexCanvas" width="600" height="600"></canvas>
        <canvas id="energyCanvas" width="300" height="200"></canvas>
    </div>
    <script>
        const trajectories = {json.dumps(trajectories)};
        const energies = {json.dumps(energies)};
        const colors = {json.dumps(colors)};
        const nVortices = {n_vortices};

        const canvas = document.getElementById('vortexCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 150;

        const eCanvas = document.getElementById('energyCanvas');
        const ectx = eCanvas.getContext('2d');

        let frame = 0;
        let playing = true;
        let trailLength = 50;

        function drawFrame() {{
            // Clear canvas
            ctx.fillStyle = '#000';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            // Draw trails
            const trailStart = Math.max(0, frame - trailLength);
            for (let v = 0; v < nVortices; v++) {{
                ctx.strokeStyle = colors[v];
                ctx.lineWidth = 2;
                ctx.globalAlpha = 0.6;
                ctx.beginPath();
                for (let t = trailStart; t <= frame; t++) {{
                    const x = cx + trajectories[t][v][0] * scale;
                    const y = cy - trajectories[t][v][1] * scale;
                    if (t === trailStart) ctx.moveTo(x, y);
                    else ctx.lineTo(x, y);
                }}
                ctx.stroke();
            }}

            // Draw current positions
            ctx.globalAlpha = 1.0;
            const current = trajectories[frame];
            for (let v = 0; v < nVortices; v++) {{
                const x = cx + current[v][0] * scale;
                const y = cy - current[v][1] * scale;
                ctx.fillStyle = colors[v];
                ctx.beginPath();
                ctx.arc(x, y, 10, 0, 2 * Math.PI);
                ctx.fill();
                ctx.strokeStyle = '#fff';
                ctx.lineWidth = 2;
                ctx.stroke();
            }}

            // Draw energy plot
            ectx.fillStyle = '#000';
            ectx.fillRect(0, 0, eCanvas.width, eCanvas.height);
            ectx.fillStyle = '#fff';
            ectx.font = '12px monospace';
            ectx.fillText('Energy Conservation', 10, 20);

            const eMin = Math.min(...energies);
            const eMax = Math.max(...energies);
            const range = eMax - eMin || 1e-10;

            ectx.strokeStyle = '#333';
            ectx.beginPath();
            for (let i = 0; i < energies.length; i++) {{
                const x = 10 + (i / energies.length) * 280;
                const y = 180 - ((energies[i] - eMin) / range) * 150;
                if (i === 0) ectx.moveTo(x, y);
                else ectx.lineTo(x, y);
            }}
            ectx.stroke();

            // Highlight current position on energy plot
            ectx.strokeStyle = '#4f4';
            ectx.lineWidth = 2;
            ectx.beginPath();
            for (let i = 0; i <= frame; i++) {{
                const x = 10 + (i / energies.length) * 280;
                const y = 180 - ((energies[i] - eMin) / range) * 150;
                if (i === 0) ectx.moveTo(x, y);
                else ectx.lineTo(x, y);
            }}
            ectx.stroke();

            const drift = Math.abs(energies[frame] - energies[0]) / (Math.abs(energies[0]) || 1e-10);
            ectx.fillText('Drift: ' + drift.toExponential(2), 10, 195);

            document.getElementById('frameInfo').textContent = 'Frame: ' + frame + '/' + (trajectories.length - 1);
        }}

        function animate() {{
            if (playing) {{
                frame = (frame + 1) % trajectories.length;
            }}
            drawFrame();
            requestAnimationFrame(animate);
        }}

        function togglePlay() {{
            playing = !playing;
        }}

        function reset() {{
            frame = 0;
        }}

        animate();
    </script>
</body>
</html>"""

    with open(filename, 'w') as f:
        f.write(html)
    print(f"\nViewer: {os.path.abspath(filename)}")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        mode = sys.argv[1]
        if mode == "pair":
            history, gamma = run_vortex_pair()
        elif mode == "dipole":
            history, gamma = run_vortex_dipole()
        elif mode == "three":
            history, gamma = run_three_vortices()
        else:
            print(f"Unknown mode: {mode}. Use: pair, dipole, three")
            sys.exit(1)
    else:
        # Default: run all three
        history, gamma = run_vortex_pair()
        run_vortex_dipole()
        run_three_vortices()

    generate_viewer(history, gamma)
