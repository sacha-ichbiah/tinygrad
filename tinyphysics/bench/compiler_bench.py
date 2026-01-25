"""
Physics Compiler Benchmark

Measures the performance of the tinyphysics compiler across different
configurations and step counts.

Usage:
    python tinyphysics/bench/compiler_bench.py
    python tinyphysics/bench/compiler_bench.py --steps 10000
    python tinyphysics/bench/compiler_bench.py --profile fast
"""
import sys
from pathlib import Path

# Add tinygrad root to path
_root = Path(__file__).parent.parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import argparse
import time
import numpy as np

from tinygrad import Tensor
from tinygrad.physics import compile_symplectic_program, SymplecticPolicy


def harmonic_oscillator_H(q, p):
    """Simple harmonic oscillator: H = 0.5 * (p² + q²)"""
    return 0.5 * (p * p + q * q).sum()


def run_benchmark(steps: int, dt: float, record_every: int,
                  policy: SymplecticPolicy, unroll: int | None = None) -> dict:
    """Run a single benchmark and return results."""
    # Create tensors for compilation - these same tensors must be cloned for evolve
    q = Tensor([1.0])
    p = Tensor([0.0])

    prog = compile_symplectic_program(
        'canonical',
        H=harmonic_oscillator_H,
        policy=policy,
        sample_state=(q, p)
    )

    # Timed run - MUST use clone() of the same tensors from sample_state
    start = time.perf_counter()
    (q_final, p_final), history = prog.evolve(
        (q.clone(), p.clone()),
        dt=dt,
        steps=steps,
        record_every=record_every,
        unroll=unroll
    )
    q_final.realize()
    elapsed = time.perf_counter() - start

    result = q_final.numpy()[0]
    exact = np.cos(steps * dt)
    error = abs(result - exact)

    return {
        'steps': steps,
        'elapsed': elapsed,
        'steps_per_sec': steps / elapsed,
        'result': result,
        'exact': exact,
        'error': error,
        'history_len': len(history),
    }


def main():
    parser = argparse.ArgumentParser(description='Physics Compiler Benchmark')
    parser.add_argument('--steps', type=int, nargs='+', default=[1000, 10000, 100000],
                        help='Number of steps to benchmark')
    parser.add_argument('--dt', type=float, default=0.01, help='Time step')
    parser.add_argument('--record-every', type=int, default=None,
                        help='Record every N steps (default: same as steps)')
    parser.add_argument('--profile', type=str, default='fast',
                        choices=['fast', 'balanced', 'precise'],
                        help='Accuracy profile')
    parser.add_argument('--unroll', type=int, default=None,
                        help='Unroll factor (default: auto)')
    parser.add_argument('--no-scan', action='store_true',
                        help='Disable scan/unrolling')
    args = parser.parse_args()

    policy = SymplecticPolicy(
        accuracy=args.profile,
        scan=not args.no_scan
    )

    print('=' * 60)
    print('Physics Compiler Benchmark')
    print('=' * 60)
    print(f'Profile: {args.profile}')
    print(f'Scan: {not args.no_scan}')
    print(f'Unroll: {args.unroll or "auto"}')
    print(f'dt: {args.dt}')
    print()

    print(f'{"Steps":>10} {"Time (s)":>10} {"Steps/s":>12} {"Error":>12}')
    print('-' * 50)

    for steps in args.steps:
        record_every = args.record_every or steps
        result = run_benchmark(
            steps=steps,
            dt=args.dt,
            record_every=record_every,
            policy=policy,
            unroll=args.unroll
        )

        print(f'{result["steps"]:>10} {result["elapsed"]:>10.3f} '
              f'{result["steps_per_sec"]:>12,.0f} {result["error"]:>12.2e}')

    print()
    print('Notes:')
    print('- Error is |result - cos(steps * dt)|')
    print('- Steps/s includes JIT compilation on first run')
    print('- The compiler uses TinyJit for efficient batch execution')


if __name__ == '__main__':
    main()
