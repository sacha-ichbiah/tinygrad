import numpy as np
from tinygrad.physics import IdealFluidVorticity2D
import json
import os

# --- Simulation ---

def run_simulation():
    # Grid Parameters
    N = 64 # Grid size
    L = 2 * np.pi
    
    # Coordinate Grid
    x = np.linspace(0, L, N, endpoint=False)
    y = np.linspace(0, L, N, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing='ij')
    
    # Initial Condition: Kelvin-Helmholtz Instability
    omega_init = np.zeros((N, N), dtype=np.float32)
    delta = 0.5
    pert = 0.1 * np.sin(X) # Perturbation
    
    # Strip 1 at y = L/4
    y1 = L/4
    omega_init += (1/delta) * (1.0 / np.cosh((Y - y1)/delta)**2) * (1 + pert)
    
    # Strip 2 at y = 3L/4 (Reverse sign)
    y2 = 3*L/4
    omega_init -= (1/delta) * (1.0 / np.cosh((Y - y2)/delta)**2) * (1 + pert)
    
    # Vorticity State
    W = omega_init

    dt = 0.02
    steps = int(os.getenv("IDEAL_FLUID_STEPS", "5000"))
    record_every = int(os.getenv("IDEAL_FLUID_RECORD_EVERY", "50"))
    progress_every = int(os.getenv("IDEAL_FLUID_PROGRESS_EVERY", str(record_every * 5)))
    if progress_every % record_every != 0:
        progress_every = ((progress_every // record_every) + 1) * record_every
    history_w = []
    
    print(f"Start Ideal Fluid (Symplectic Midpoint) Simulation N={N}")

    solver = IdealFluidVorticity2D(N, L=L, dealias=2.0/3.0, dtype=np.float32)
    history = []
    steps_done = 0
    while steps_done < steps:
        chunk = min(progress_every, steps - steps_done)
        W, chunk_history = solver.evolve(W, dt=dt, steps=chunk, record_every=record_every, method="midpoint", iters=5)
        if history:
            chunk_history = chunk_history[1:]
        history.extend(chunk_history)
        steps_done += chunk
        print(f"Progress: {steps_done}/{steps} steps")

    for frame in history:
        history_w.append(frame.tolist())

    Z_start = 0.5 * (history[0] ** 2).sum() * (L / N) ** 2
    Z_end = 0.5 * (history[-1] ** 2).sum() * (L / N) ** 2
    print(f"Enstrophy Z: Start {Z_start:.4f}, End {Z_end:.4f}, Drift {abs(Z_end-Z_start)/abs(Z_start):.2e}")
    
    generate_viewer(history_w, N)

def generate_viewer(history, N):
    # Flatten history for JS ? Or just keep 2D array
    # N is small (64), so 64x64 = 4096 data points per frame.
    # 50 frames = 200k floats.
    
    # Normalize data for visualization (0-255)
    # We do this in JS to maintain dynamic range
    
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
            // Find min/max for scaling? Or fixed
            // Vorticity ~ [-2, 2] roughly
            
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
