import numpy as np
from tinygrad.tensor import Tensor
from tinyphysics.compiler import UniversalSymplecticCompiler
import json
import os
from tinygrad.physics import point_vortex_hamiltonian

def run_simulation():
    # Parameters
    # N Point Vortices
    # Case 1: Vortex Dipole (Two vortices, opposite circulation) -> Translates
    # Case 2: Two same sign -> Rotate
    # Let's do 4 vortices: A dipole pair and another dipole pair colliding?
    # Or just random.
    # Let's do a "Leapfrogging" setup if possible, or just random N=10.
    
    np.random.seed(42)
    N = 4
    
    # Random Positions in [-1, 1]
    q_init = np.random.randn(N, 2).astype(np.float32)
    
    # Circulations Gamma
    # Random +/- 1
    gamma_np = np.random.choice([-1.0, 1.0], size=(N,)).astype(np.float32)
    # Enforce sum(Gamma) != 0 to make it interesting, or 0.
    
    # Tinygrad Tensors
    q = Tensor(q_init)
    Gamma = Tensor(gamma_np)
    
    dt = 0.01
    steps = 600
    record_every = 5
    
    print(f"Start Point Vortex Simulation (N={N})")
    print(f"Circulations: {Gamma.numpy()}")
    
    history_q = []
    
    H = point_vortex_hamiltonian(Gamma, softening=1e-2)
    system = UniversalSymplecticCompiler(kind="point_vortex", integrator="midpoint", gamma=Gamma, softening=1e-2)
    H_start = float(H(q).numpy())
    C_start = (q.numpy() * Gamma.numpy()[:, None]).sum(axis=0)
    L_start = (Gamma.numpy() * (q.numpy()**2).sum(axis=1)).sum()

    for i in range(steps):
        q = system.step(q, dt)
        if i % record_every == 0:
            history_q.append(q.numpy().tolist())
    H_end = float(H(q).numpy())
    C_end = (q.numpy() * Gamma.numpy()[:, None]).sum(axis=0)
    L_end = (Gamma.numpy() * (q.numpy()**2).sum(axis=1)).sum()
    
    print(f"Energy H: Start {H_start:.4f}, End {H_end:.4f}, Drift {abs(H_end-H_start)/abs(H_start):.2e}")
    print(f"Center C: Drift {np.linalg.norm(C_end - C_start):.2e}")
    print(f"AngMom L: Start {L_start:.4f}, End {L_end:.4f}, Drift {abs(L_end-L_start)/abs(L_start):.2e}")
    
    generate_viewer(history_q, Gamma.numpy())

def generate_viewer(history, gammas):
    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Point Vortices</title>
    <style>
        body {{ font-family: sans-serif; background: #222; color: #fff; display: flex; flex-direction: column; align-items: center; }}
        canvas {{ border: 1px solid #555; background: #000; }}
    </style>
</head>
<body>
    <h1>Point Vortex Dynamics</h1>
    <p>Red = Positive Circulation, Blue = Negative.</p>
    <canvas id="simCanvas" width="600" height="600"></canvas>
    <script>
        const history = {json.dumps(history)};
        const gammas = {json.dumps(gammas.tolist())};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const cx = canvas.width / 2;
        const cy = canvas.height / 2;
        const scale = 100; 
        
        let frame = 0;
        
        // Trail
        const trails = gammas.map(() => []);
        const maxTrail = 100;
        
        function draw() {{
            // Fade out
            ctx.fillStyle = 'rgba(0,0,0,0.2)';
            ctx.fillRect(0,0, canvas.width, canvas.height);
            
            const positions = history[frame];
            
            for(let i=0; i<gammas.length; i++) {{
                const x = cx + positions[i][0] * scale;
                const y = cy - positions[i][1] * scale;
                
                // Trail
                trails[i].push([x,y]);
                if(trails[i].length > maxTrail) trails[i].shift();
                
                ctx.beginPath();
                ctx.strokeStyle = gammas[i] > 0 ? 'rgba(255,100,100,0.5)' : 'rgba(100,100,255,0.5)';
                if(trails[i].length > 0) ctx.moveTo(trails[i][0][0], trails[i][0][1]);
                for(let p of trails[i]) ctx.lineTo(p[0], p[1]);
                ctx.stroke();
                
                // Vortex
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2*Math.PI);
                ctx.fillStyle = gammas[i] > 0 ? '#f55' : '#55f';
                ctx.fill();
            }}
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}
        
        draw();
    </script>
</body>
</html>
    """
    
    with open('examples/point_vortices_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/point_vortices_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
