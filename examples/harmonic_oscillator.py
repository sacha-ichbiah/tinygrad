"""
Harmonic Oscillator - The simplest physics system (Level 1.1)

THE TINYPHYSICS WAY:
    1. Define the Hamiltonian H(q, p) - that's ALL the physics
    2. The system automatically derives equations of motion via autograd
    3. Symplectic integrator preserves energy

Hamiltonian: H = p²/2m + kq²/2

This is the "Hello World" of physics simulation.
"""

import numpy as np
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem
import os
import time

try:
    from examples.physics_viewer import PhysicsViewer1D
except ImportError:
    try:
        from physics_viewer import PhysicsViewer1D
    except ImportError:
        PhysicsViewer1D = None


# ============================================================================
# THE HAMILTONIAN - This is ALL the physics you need to define
# ============================================================================

def harmonic_hamiltonian(k: float = 1.0, m: float = 1.0):
    """
    Returns the Hamiltonian for a harmonic oscillator.

    H = T + V = p²/2m + kq²/2

    The symplectic integrator will automatically derive:
        dq/dt = dH/dp = p/m     (velocity)
        dp/dt = -dH/dq = -kq    (spring force)

    No need to manually compute F = -kx - autograd does it!
    """
    def H(q, p):
        T = (p * p).sum() / (2 * m)  # Kinetic energy
        V = k * (q * q).sum() / 2     # Potential energy
        return T + V

    return H


# ============================================================================
# SIMULATION
# ============================================================================

def run_simulation(integrator="yoshida4"):
    """
    Simulate a harmonic oscillator using the TinyPhysics compiler approach.
    """
    # Physics constants
    k = 1.0   # Spring constant
    m = 1.0   # Mass
    dt = 0.1
    steps = 100

    # Initial state
    q = Tensor([1.0], requires_grad=True)  # Start displaced
    p = Tensor([0.0], requires_grad=True)  # Start at rest

    # Expected period: T = 2*pi*sqrt(m/k)
    T_period = 2 * np.pi * np.sqrt(m / k)

    print("=" * 60)
    print("HARMONIC OSCILLATOR - TinyPhysics Compiler Approach")
    print("=" * 60)
    print(f"\nPhysics defined by Hamiltonian ONLY:")
    print(f"  H(q, p) = p²/2m + kq²/2")
    print(f"\nEquations of motion derived automatically via autograd:")
    print(f"  dq/dt = +dH/dp = p/m")
    print(f"  dp/dt = -dH/dq = -kq")
    print(f"\nIntegrator: {integrator}")
    print(f"Period: {T_period:.4f}, Simulation time: {dt*steps:.1f} ({dt*steps/T_period:.1f} periods)")

    # CREATE THE HAMILTONIAN SYSTEM
    H = harmonic_hamiltonian(k=k, m=m)
    system = HamiltonianSystem(H, integrator=integrator)

    # Initial energy
    E_start = system.energy(q, p)
    print(f"\nInitial: q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}, E={E_start:.6f}")

    # EVOLVE
    q, p, history = system.evolve(q, p, dt=dt, steps=steps, record_every=1)

    # Final state
    E_end = system.energy(q, p)
    E_drift = abs(E_end - E_start) / abs(E_start)

    print(f"Final:   q={q.numpy()[0]:.4f}, p={p.numpy()[0]:.4f}, E={E_end:.6f}")
    print(f"Energy Drift: {E_drift:.2e}")

    # Backprop through physics (differentiable simulation!)
    print(f"\nDifferentiable Physics Test:")
    target = 0.0
    loss = ((q - target)**2).sum()
    loss.backward()

    if p.grad is not None:
        print(f"dL/dp_initial = {p.grad.numpy()[0]:.4f}")
        print("Gradients flow through {steps} physics steps!")
    else:
        print("(Gradient not available in current mode)")

    # Generate viewer
    if PhysicsViewer1D is not None:
        history_q = [float(h[0][0]) for h in history]
        history_p = [float(h[1][0]) for h in history]
        viewer = PhysicsViewer1D(title=f"Harmonic Oscillator ({integrator})")
        viewer.render(history_q, history_p, dt, "examples/harmonic_oscillator_viewer.html")

    return E_start, E_end, E_drift


def compare_integrators():
    """Compare energy conservation across integrators."""
    print("=" * 60)
    print("COMPARING SYMPLECTIC INTEGRATORS")
    print("=" * 60)

    results = {}
    for name in ["euler", "leapfrog", "yoshida4"]:
        print(f"\n{'='*60}")
        _, _, drift = run_simulation(name)
        results[name] = drift

    print(f"\n{'='*60}")
    print("SUMMARY: Energy Drift")
    print("="*60)
    for name, drift in results.items():
        print(f"  {name:12s}: {drift:.2e}")


def benchmark(integrator: str = "yoshida4", steps: int = 20000, repeats: int = 5, jit: bool = False):
    """Micro-benchmark for autograd-based harmonic oscillator step speed."""
    k, m, dt = 1.0, 1.0, 0.01
    H = harmonic_hamiltonian(k=k, m=m)
    system = HamiltonianSystem(H, integrator=integrator)

    def bench_once(use_jit: bool) -> float:
        q = Tensor([1.0], requires_grad=True)
        p = Tensor([0.0], requires_grad=True)
        step = TinyJit(system.step) if use_jit else system.step
        if use_jit:
            q, p = step(q, p, dt)
        start = time.perf_counter()
        for _ in range(steps):
            q, p = step(q, p, dt)
        q.numpy()
        p.numpy()
        return time.perf_counter() - start

    print("=" * 60)
    print("HARMONIC OSCILLATOR BENCHMARK (AUTOGRAD)")
    print("=" * 60)
    print(f"Integrator: {integrator}, steps: {steps}, repeats: {repeats}")

    times = [bench_once(jit) for _ in range(repeats)]
    best = min(times)
    print(f"{'autograd' + (' + TinyJit' if jit else ''):28s}: {best*1e3:.2f} ms  ({steps/best:,.0f} steps/s)")


def _parse_integrator(args, default="yoshida4"):
    for arg in args:
        if not arg.startswith("--"):
            return arg
    return default


def _parse_int_flag(args, name: str, default: int) -> int:
    prefix = f"--{name}="
    for arg in args:
        if arg.startswith(prefix):
            return int(arg[len(prefix):])
    return default


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    if "--bench" in args:
        integrator = _parse_integrator(args)
        steps = _parse_int_flag(args, "steps", 20000)
        repeats = _parse_int_flag(args, "repeats", 5)
        benchmark(integrator=integrator, steps=steps, repeats=repeats, jit="--jit" in args)
    elif "--compare" in args:
        compare_integrators()
    else:
        integrator = _parse_integrator(args)
        run_simulation(integrator)
