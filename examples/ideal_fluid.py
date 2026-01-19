import numpy as np
from tinygrad.tensor import Tensor
import json
import os

# --- FFT Ops Wrapper ---
# Tinygrad doesn't have native FFT yet. 
# We perform FFT in Numpy. 
# For explicit time stepping, we can break the graph each step, 
# as we don't need to backprop through the solver for this demo.

def fft2(x: Tensor):
    # x: (N, N) Real Tensor
    # Returns: (N, N, 2) Complex Tensor (Real, Imag)
    # 1. Sync to CPU/Numpy
    dat = x.numpy()
    # 2. FFT
    ft = np.fft.fft2(dat)
    # 3. Stack Real/Imag
    out = np.stack([ft.real, ft.imag], axis=-1).astype(np.float32)
    # 4. Return new Tensor
    return Tensor(out)

def ifft2(x: Tensor):
    # x: (N, N, 2) Complex Tensor
    # Returns: (N, N) Real Tensor
    dat = x.numpy()
    # Reconstruct complex
    c = dat[..., 0] + 1j * dat[..., 1]
    # IFFT
    out = np.fft.ifft2(c)
    # Real part
    return Tensor(out.real.astype(np.float32))

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
    W = Tensor(omega_init, requires_grad=False)
    
    # Pre-compute Wave Numbers k
    kx = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    ky = np.fft.fftfreq(N, d=L/N) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing='ij')
    
    K2 = KX**2 + KY**2
    K2[0,0] = 1.0 # Avoid division by zero
    
    # Tensor Constants
    KX_T = Tensor(KX.astype(np.float32))
    KY_T = Tensor(KY.astype(np.float32))
    InvK2_T = Tensor((1.0/K2).astype(np.float32))
    
    # De-aliasing Cutoff (2/3 Rule)
    k_max = np.max(np.abs(kx))
    cutoff = (2.0/3.0) * k_max
    Mask = Tensor(((np.abs(KX) < cutoff) & (np.abs(KY) < cutoff)).astype(np.float32))
    
    dt = 0.05
    steps = 500
    history_w = []
    
    print(f"Start Ideal Fluid (Euler) Simulation N={N}")
    
    for step in range(steps):
        
        # RHS Calculation
        def compute_rhs(w_field):
            # w_field is Tensor
            
            # 1. FFT
            w_hat = fft2(w_field)
            
            # 2. Psi_hat = w_hat / k^2
            psi_hat = w_hat * InvK2_T.unsqueeze(-1)
            
            # 3. U_hat, V_hat
            psi_r, psi_i = psi_hat[..., 0], psi_hat[..., 1]
            u_hat_r = -KY_T * psi_i
            u_hat_i = KY_T * psi_r
            
            v_hat_r = KX_T * psi_i
            v_hat_i = -KX_T * psi_r
            
            u_hat = Tensor.stack([u_hat_r, u_hat_i], dim=-1)
            v_hat = Tensor.stack([v_hat_r, v_hat_i], dim=-1)
            
            # 4. IFFT to Real
            u = ifft2(u_hat)
            v = ifft2(v_hat)
            
            # 5. Grad W in Fourier
            w_r, w_i = w_hat[..., 0], w_hat[..., 1]
            dwdx_hat_r = -KX_T * w_i
            dwdx_hat_i = KX_T * w_r
            dwdx_hat = Tensor.stack([dwdx_hat_r, dwdx_hat_i], dim=-1)
            
            dwdy_hat_r = -KY_T * w_i
            dwdy_hat_i = KY_T * w_r
            dwdy_hat = Tensor.stack([dwdy_hat_r, dwdy_hat_i], dim=-1)
            
            dwdx = ifft2(dwdx_hat)
            dwdy = ifft2(dwdy_hat)
            
            # 6. Advection (Non-linear)
            advection = u * dwdx + v * dwdy
            
            # 7. Mask
            adv_hat = fft2(advection)
            adv_hat_masked = adv_hat * Mask.unsqueeze(-1)
            
            rhs = -ifft2(adv_hat_masked)
            return rhs

        # RK4 Integration
        # We need to be careful with breaking graphs.
        # W is the state.
        # k1, k2... calculate new tensors.
        
        k1 = compute_rhs(W)
        k2 = compute_rhs(W + 0.5*dt*k1)
        k3 = compute_rhs(W + 0.5*dt*k2)
        k4 = compute_rhs(W + dt*k3)
        
        W_new = W + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        W = W_new.realize()
        
        if step % 10 == 0:
            history_w.append(W.numpy().tolist())
            
        # Checks
        if step == 0:
            # Enstrophy
            Z_start = (0.5 * (W**2).sum()).numpy() * (L/N)**2
            
    Z_end = (0.5 * (W**2).sum()).numpy() * (L/N)**2
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
        canvas {{ border: 1px solid #555; image-rendering: pixelated; }}
    </style>
</head>
<body>
    <h1>Ideal Fluid (Vorticity)</h1>
    <p>Red = Positive Vorticity, Blue = Negative.</p>
    <canvas id="simCanvas" width="512" height="512"></canvas>
    <script>
        const history = {json.dumps(history)};
        const N = {N};
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const imageData = ctx.createImageData(N, N);
        
        // Scale canvas up via CSS
        
        let frame = 0;
        
        function draw() {{
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
            
            // Put image data to temp canvas then draw scaled?
            // Actually checking pixelated style works better if we drawImage.
            
            createImageBitmap(imageData).then(bmp => {{
                ctx.drawImage(bmp, 0, 0, canvas.width, canvas.height);
            }});
            
            frame = (frame + 1) % history.length;
            requestAnimationFrame(draw);
        }}
        
        draw();
    </script>
</body>
</html>
    """
    
    with open('examples/ideal_fluid_viewer.html', 'w') as f:
        f.write(html_content)
    print(f"Viewer generated: {os.path.abspath('examples/ideal_fluid_viewer.html')}")

if __name__ == "__main__":
    run_simulation()
