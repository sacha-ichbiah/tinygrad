"""Ideal Fluid (2D Euler) Simulation using TinyPhysics Structure Compiler.

This example demonstrates the Kelvin-Helmholtz instability using the
VorticityStructure, which is a proper Lie-Poisson structure that flows
through the universal physics compiler.
"""
import os
import numpy as np
from tinygrad.tensor import Tensor
import json

from tinyphysics.structures.vorticity import VorticityStructure
from tinyphysics.systems.vorticity import kelvin_helmholtz_ic, compute_enstrophy


def run_simulation():
    # Grid Parameters
    N = int(os.getenv("IDEAL_FLUID_N", "64"))
    L = 2 * np.pi

    # Initial Condition: Kelvin-Helmholtz Instability
    omega_init = kelvin_helmholtz_ic(N, L, delta=0.5, pert_amp=0.1)

    # Scale dt with grid size for CFL stability
    dt = float(os.getenv("IDEAL_FLUID_DT", str(0.02 * 64 / N)))
    steps = int(os.getenv("IDEAL_FLUID_STEPS", "5000"))
    record_every = int(os.getenv("IDEAL_FLUID_RECORD_EVERY", "50"))
    progress_every = int(os.getenv("IDEAL_FLUID_PROGRESS_EVERY", str(record_every * 5)))
    if progress_every % record_every != 0:
        progress_every = ((progress_every // record_every) + 1) * record_every

    # Use fewer midpoint iterations (3 instead of 5) - converges fast for smooth flows
    iters = int(os.getenv("IDEAL_FLUID_ITERS", "3"))

    print(f"Start Ideal Fluid (VorticityStructure + Unrolling) Simulation N={N}")

    # Create solver using the structure compiler
    solver = VorticityStructure(N, L=L, dealias=2.0/3.0, dtype=np.float32)

    # Run simulation with auto-unrolling
    W = Tensor(omega_init)
    history = []
    steps_done = 0

    while steps_done < steps:
        chunk = min(progress_every, steps - steps_done)
        W, chunk_history = solver.evolve(
            W, dt=dt, steps=chunk,
            record_every=record_every,
            method="midpoint",
            iters=iters
        )
        if history:
            chunk_history = chunk_history[1:]  # Avoid duplicating last frame
        history.extend(chunk_history)
        steps_done += chunk
        print(f"Progress: {steps_done}/{steps} steps")

    history_w = [frame.tolist() for frame in history]

    Z_start = compute_enstrophy(history[0], L, N)
    Z_end = compute_enstrophy(history[-1], L, N)
    print(f"Enstrophy Z: Start {Z_start:.4f}, End {Z_end:.4f}, Drift {abs(Z_end-Z_start)/abs(Z_start):.2e}")

    generate_viewer(history_w, N)


def generate_viewer(history, N):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Ideal Fluid (Kelvin-Helmholtz)</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #555; image-rendering: pixelated; width: 512px; height: 512px; }}
    </style>
</head>
<body>
    <h1>Ideal Fluid (Vorticity)</h1>
    <p>Red = Positive Vorticity, Blue = Negative.</p>
    <p>Frame: <span id="frame">0</span> / {len(history) - 1}</p>
    <canvas id="simCanvas" width="{N}" height="{N}"></canvas>
    <script>
        const history = {json.dumps(history)};
        const N = {N};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(N, N);
        const frameEl = document.getElementById('frame');
        const targetFPS = 30;

        let frame = 0;
        let lastTime = 0;

        function draw(ts) {{
            if (ts - lastTime < 1000 / targetFPS) {{
                requestAnimationFrame(draw);
                return;
            }}
            lastTime = ts;
            const W = history[frame];

            const data = imageData.data;

            for(let i=0; i<N; i++) {{
                for(let j=0; j<N; j++) {{
                    const idx = (i*N + j) * 4;
                    const val = W[i][j];

                    // Colormap: Blue -> Black -> Red
                    let r=0, g=0, b=0;
                    if(val > 0) {{
                        r = Math.min(255, val * 100);
                    }} else {{
                        b = Math.min(255, -val * 100);
                    }}

                    data[idx] = r;
                    data[idx+1] = g;
                    data[idx+2] = b;
                    data[idx+3] = 255;
                }}
            }}

            ctx.putImageData(imageData, 0, 0);

            frameEl.textContent = frame;
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}

        ctx.imageSmoothingEnabled = false;
        requestAnimationFrame(draw);
    </script>
</body>
</html>
    """

    with open('examples/ideal_fluid_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/ideal_fluid_viewer.html')}")


if __name__ == "__main__":
    run_simulation()
