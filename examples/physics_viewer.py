import json
import os

class PhysicsViewer1D:
    def __init__(self, title="1D Physics Simulation", width=600, height=400):
        self.title = title
        self.width = width
        self.height = height

    def render(self, q_data, p_data, dt, output_file="physics_viewer.html"):
        # Ensure data is serializable (convert tensors/numpy to list if needed)
        # Assuming inputs are already lists of floats as per usage in harmonic_oscillator.py
        
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.title}</title>
    <style>
        body {{ font-family: sans-serif; display: flex; flex-direction: column; alignItems: center; justify-content: center; height: 100vh; margin: 0; background: #111; color: #eee; }}
        canvas {{ background: #222; border: 1px solid #444; }}
        .controls {{ margin-top: 10px; }}
        button {{ padding: 8px 16px; font-size: 16px; cursor: pointer; }}
    </style>
</head>
<body>
    <h1>{self.title}</h1>
    <canvas id="simCanvas" width="{self.width}" height="{self.height}"></canvas>
    <div class="controls">
        <button onclick="toggleAnimation()">Pause/Play</button>
        <button onclick="resetAnimation()">Reset</button>
    </div>
    <p>Step: <span id="stepDisplay">0</span> | q: <span id="qDisplay">0.00</span> | p: <span id="pDisplay">0.00</span></p>

    <script>
        const qData = {json.dumps(q_data)};
        const pData = {json.dumps(p_data)};
        const dt = {dt};
        
        const canvas = document.getElementById('simCanvas');
        const ctx = canvas.getContext('2d');
        const width = canvas.width;
        const height = canvas.height;
        const centerX = width / 2;
        const centerY = height / 2;
        const scale = 100; // Pixels per unit
        
        let frame = 0;
        let running = true;
        let animationId;

        function draw() {{
            ctx.clearRect(0, 0, width, height);

            // Draw axis
            ctx.strokeStyle = '#444';
            ctx.beginPath();
            ctx.moveTo(0, centerY);
            ctx.lineTo(width, centerY);
            ctx.stroke();

            // Draw Spring
            const x = qData[frame] * scale;
            ctx.strokeStyle = '#888';
            ctx.lineWidth = 2;
            ctx.beginPath();
            ctx.moveTo(centerX, centerY);
            // Simple zig-zag for spring
            const segments = 10;
            const springWidth = 10;
            for(let i=0; i<=segments; i++) {{
                const xPos = centerX + (x / segments) * i;
                const offset = (i%2 === 0 ? -1 : 1) * springWidth * (i>0 && i<segments ? 1 : 0);
                ctx.lineTo(xPos, centerY + offset);
            }}
            ctx.stroke();

            // Draw Mass
            ctx.fillStyle = '#00ccff';
            ctx.beginPath();
            ctx.arc(centerX + x, centerY, 15, 0, Math.PI * 2);
            ctx.fill();

            // Update Stats
            document.getElementById('stepDisplay').innerText = frame;
            document.getElementById('qDisplay').innerText = qData[frame].toFixed(4);
            document.getElementById('pDisplay').innerText = pData[frame].toFixed(4);

            if (running) {{
                frame++;
                if (frame >= qData.length) frame = 0;
                animationId = setTimeout(() => requestAnimationFrame(draw), 20); // Slow down slightly
            }}
        }}

        function toggleAnimation() {{
            running = !running;
            if (running) draw();
        }}

        function resetAnimation() {{
            frame = 0;
            if (!running) {{
                running = true;
                draw();
            }}
        }}

        draw();
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        print(f"\\nViewer generated at: {os.path.abspath(output_file)}")
