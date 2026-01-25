# TinyPhysics Benchmarks

Quick sanity benchmarks for the structure‑preserving compiler and core systems.

## Run

```bash
PYTHONPATH=/workspaces/tinygrad python tinyphysics/bench/universal_physics_bench.py
```

## Benchmarks Included

- canonical: simple harmonic oscillator (symplectic)
- so3: rigid‑body style Lie‑Poisson flow
- quantum: split‑operator 1D wavepacket
- constraint: RATTLE projection on unit‑sphere
- dissipative: conformal damping
- fluid: 2D vorticity step
- thermostat: Langevin damping (optional)
- lj: Lennard‑Jones tensor‑bins (optional)
- barostat: Berendsen NPT (optional)
- lj_barostat: LJ + Berendsen (optional)

## Notes

- Times are wall‑clock for fixed step counts and small sizes.
- This is not a micro‑benchmark suite; it’s a regression sanity check.
- Use consistent CPU affinity if you compare across runs.
- Optional thresholds can be set via env vars like `TINYGRAD_BENCH_CANONICAL_MAX` (seconds).
- Optional thermostat bench: set `TINYGRAD_BENCH_THERMOSTAT=1` and `TINYGRAD_BENCH_THERMOSTAT_MAX`.
- Optional LJ bench: set `TINYGRAD_BENCH_LJ=1` and `TINYGRAD_BENCH_LJ_MAX`.
- Optional barostat bench: set `TINYGRAD_BENCH_BAROSTAT=1` and `TINYGRAD_BENCH_BAROSTAT_MAX`.
- Optional LJ barostat bench: set `TINYGRAD_BENCH_LJ_BAROSTAT=1` and `TINYGRAD_BENCH_LJ_BAROSTAT_MAX`.
  - Use `_MAX` env vars to enforce time thresholds in CI for NPT-related benches.

## Split-Operator Demo

A minimal split-operator example is available at `examples/split_operator_demo.py`.

## NPT Example

A minimal barostat example is available at `examples/lennard_jones_barostat.py`.

## Neighbor List Bench

Run the minimal cell‑linked neighbor list benchmark:

```bash
python tinyphysics/bench/neighbors_bench.py
```

Optional tensor-bins micro-bench:

```bash
TINYGRAD_BENCH_TENSOR_BINS=1 python tinyphysics/bench/neighbors_bench.py
```

Optional tensor-bins table build timing:

```bash
TINYGRAD_BENCH_TENSOR_BINS_TABLE=1 python tinyphysics/bench/neighbors_bench.py
```

Optional tensor-bins force-only timing (table prebuilt):

```bash
TINYGRAD_BENCH_TENSOR_BINS_FORCE=1 python tinyphysics/bench/neighbors_bench.py
```

CPU tensor-bins uses a small-N fallback to direct tensor pairwise when `N <= TINYGRAD_TENSOR_BINS_CPU_THRESHOLD` (default 512).

## N-Body Bench (Optional)

Enable from the universal runner:

```bash
TINYGRAD_BENCH_NBODY=1 python tinyphysics/bench/run_universal.py
```

Compare tensor-bins vs tensor vs neighbor:

```bash
TINYGRAD_BENCH_NBODY=1 TINYGRAD_BENCH_NBODY_COMPARE=1 python tinyphysics/bench/run_universal.py
```

## N-Body Bench

Run the minimal N-body benchmark (neighbor, naive, barnes-hut):

```bash
python tinyphysics/bench/nbody_bench.py
```

For tensor-bins, you can override the per-cell cap:

```bash
TINYGRAD_NBODY_MAX_PER=16 python tinyphysics/bench/nbody_bench.py
TINYGRAD_NBODY_MAX_PER=auto python tinyphysics/bench/nbody_bench.py
```

Recommended max_per ranges (rule of thumb):

- N <= 256, sparse: 8–16
- N <= 1024, moderate density: 16–32
- N >= 2048, dense: 32–64

Barnes‑Hut defaults: theta=0.5, softening=1e-2 (tune for accuracy vs. speed).
