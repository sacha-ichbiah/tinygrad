# Double Pendulum Benchmark Learnings

Date: 2026-01-21

## Setup
- Script: `examples/double_pendulum_benchmark.py`
- Steps: 2000
- Repeats: 1 (single-run sweep)
- dt: 0.01

## Peak Result
Best observed speed:
- **Integrator:** `leapfrog`
- **Unroll:** `4`
- **JIT flag:** `--jit` not used (unroll path uses TinyJit internally)
- **Steps/s:** **9,493.2**
- **Command:** `python examples/double_pendulum_benchmark.py leapfrog --steps=2000 --repeats=1 --unroll=4`

## Sweep Notes
- `implicit` now supports fixed iteration counts via `--implicit-iters=N` (enables JIT/unroll).
- `--energy-drift` adds an accuracy metric (relative energy drift) per run.
- `--fast` enables `--jit`, sets `--unroll=4` (if not provided), and turns on `--energy-drift`.
- `--sweep-iters` auto-selects `implicit-iters` using drift and speed targets.
  - `--drift=1e-4` sets the drift target (relative).
  - `--min-steps=0` sets a minimum steps/s target.
  - `--iters-range=2,10` sets the candidate range.
  - Prints an iters table and the chosen iteration count.
- `leapfrog` and `yoshida4` improve significantly with unroll or `--jit`.
- Single-run results vary; use `--repeats=3` for more stable comparisons.

## Observed Results (steps/s)
- implicit, unroll=1, jit=False: 237.4
- implicit, unroll=1, jit=True: OK with `--implicit-iters`
- implicit, unroll=2/4/8: OK with `--implicit-iters`
- leapfrog, unroll=1, jit=False: 522.8
- leapfrog, unroll=1, jit=True: 3941.8
- leapfrog, unroll=2: 7074.9
- leapfrog, unroll=4: 9493.2  <-- best
- leapfrog, unroll=8: 7797.8
- yoshida4, unroll=1, jit=False: 150.9
- yoshida4, unroll=1, jit=True: 3053.1
- yoshida4, unroll=2: 4248.6
- yoshida4, unroll=4: 3947.7
- yoshida4, unroll=8: 3211.9

# Harmonic Oscillator Benchmark Learnings

Date: 2026-01-21

## Setup
- Script: `examples/harmonic_oscillator.py --bench`
- Steps: 2000
- Repeats: 1 (single-run sweep)
- dt: 0.01

## Peak Result
Best observed speed:
- **Integrator:** `euler`
- **Unroll:** `16`
- **JIT flag:** `--jit` not used (unroll path uses TinyJit internally)
- **Steps/s:** **35,265.0**
- **Command:** `python examples/harmonic_oscillator.py --bench euler --steps=2000 --repeats=1 --unroll=16`

## Sweep Notes
- Unrolling dominates performance for this small system.
- Peak speed depends on integrator cost; euler is fastest but least accurate.
- Single-run results vary; use `--repeats=3` for more stable comparisons.

## Observed Results (steps/s)
- euler, unroll=1, jit=False: 857.0
- euler, unroll=1, jit=True: 6091.0
- euler, unroll=2: 14880.0
- euler, unroll=4: 25231.0
- euler, unroll=8: 34917.0
- euler, unroll=16: 35265.0  <-- best
- leapfrog, unroll=1, jit=False: 542.0
- leapfrog, unroll=1, jit=True: 5832.0
- leapfrog, unroll=2: 13911.0
- leapfrog, unroll=4: 22781.0
- leapfrog, unroll=8: 30902.0
- leapfrog, unroll=16: 23221.0
- yoshida4, unroll=1, jit=False: 166.0
- yoshida4, unroll=1, jit=True: 5239.0
- yoshida4, unroll=2: 8441.0
- yoshida4, unroll=4: 16758.0
- yoshida4, unroll=8: 17292.0
- yoshida4, unroll=16: 7085.0
