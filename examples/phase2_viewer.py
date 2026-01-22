import argparse
import os
import sys

sys.path.append(os.path.dirname(__file__))
from rigid_body_free import run_simulation as run_rigid
from heavy_top import simulate_heavy_top
from satellite_control import run_simulation as run_sat


def write_combined_viewer():
  html = """
<!DOCTYPE html>
<html>
<head>
  <title>TinyPhysics Phase 2 Viewer</title>
  <style>
    body { font-family: sans-serif; background: #111; color: #eee; margin: 0; }
    h1 { margin: 16px; }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; padding: 12px; }
    iframe { width: 100%; height: 500px; border: 1px solid #333; background: #000; }
    .full { grid-column: 1 / -1; }
  </style>
</head>
<body>
  <h1>Phase 2: Rigid Body, Heavy Top, Satellite Control</h1>
  <div class="grid">
    <iframe src="rigid_body_free_viewer.html"></iframe>
    <iframe src="heavy_top_viewer.html"></iframe>
    <iframe class="full" src="satellite_control_viewer.html"></iframe>
  </div>
</body>
</html>
  """
  out = 'examples/phase2_viewer.html'
  with open(out, 'w') as f:
    f.write(html)
  print(f"Combined viewer: {os.path.abspath(out)}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch", type=int, default=1)
  parser.add_argument("--viewer-batch", type=int, default=4)
  parser.add_argument("--rigid-steps", type=int, default=2000)
  parser.add_argument("--heavy-steps", type=int, default=800)
  parser.add_argument("--sat-steps", type=int, default=1500)
  parser.add_argument("--rigid-unroll", type=int, default=8)
  parser.add_argument("--heavy-unroll", type=int, default=8)
  parser.add_argument("--sat-unroll", type=int, default=10)
  parser.add_argument("--scan", action="store_true")
  args = parser.parse_args()

  run_rigid(batch_size=args.batch, steps=args.rigid_steps, unroll_steps=args.rigid_unroll, scan=args.scan, viewer_batch=args.viewer_batch)
  simulate_heavy_top(
    L0=[0.1, 0.0, 5.0],
    theta0=3.141592653589793/6,
    dt=0.002,
    steps=args.heavy_steps,
    method="splitting",
    batch_size=args.batch,
    unroll_steps=args.heavy_unroll,
    scan=args.scan,
    viewer_batch=args.viewer_batch,
    render=True,
  )
  run_sat(steps=args.sat_steps, unroll_steps=args.sat_unroll, batch_size=args.batch, scan=args.scan, viewer_batch=args.viewer_batch)
  write_combined_viewer()


if __name__ == "__main__":
  main()
