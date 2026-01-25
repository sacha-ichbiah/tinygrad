# TinyPhysics: The Structure-Preserving Compiler

**Status:** v3 (Implementation ~85% complete)
**Philosophy:** One equation, many geometries.

---

## 1. The Key Insight

Classical mechanics, quantum mechanics, and rigid body dynamics look different. They're not.

Every physical system evolves by the same rule:

```
dz/dt = Lambda(z) * grad(H)
```

Where:
- `z` is the **State** (positions, momenta, wavefunctions, angular momentum)
- `H` is the **Generator** (energy, Hamiltonian)
- `Lambda` is the **Structure Map** (the geometry of phase space)

The only difference between a planet, an electron, and a spinning top is `Lambda`:

| System | Structure Map Lambda(z, g) | Result |
|--------|---------------------------|--------|
| Classical (q,p) | `(g_p, -g_q)` | Hamilton's equations |
| Quantum (psi) | `-i * g` | Schrodinger equation |
| Rigid Body (L) | `L x g` | Euler's equations |

**TinyPhysics compiles the structure bracket, not the physics.**

---

## 2. Why Structure-Preserving?

Standard integrators (RK4, Euler) break physics:

- **Energy drift**: A planet's orbit spirals inward or outward
- **Phase errors**: Oscillators speed up or slow down over time
- **Instability**: Long simulations explode

Structure-preserving integrators respect the geometry:

- **Symplectic**: Preserve phase-space volume (no energy drift)
- **Unitary**: Preserve probability (quantum normalization)
- **Lie-Poisson**: Preserve Casimirs (angular momentum magnitude)

The cost? Roughly the same as RK4. The benefit? Simulations that stay physical forever.

---

## 3. The Structure IR (Narrow Waist)

Every simulation reduces to three objects:

```
(State, Generator, Structure)
```

1. **State (z)**: Tensor data — `(q, p)`, `L`, `psi`, fields
2. **Generator (H)**: Scalar energy function or operator
3. **Structure**: Defines geometry via the `bracket()` method

This is the compiler's universal interface. All optimizations happen here.

---

## 4. The Structure Protocol

```python
class StructureKind(Enum):
    CANONICAL = auto()      # Classical (q, p)
    QUANTUM = auto()        # Unitary evolution
    LIE_POISSON = auto()    # Rigid bodies, vortices
    DISSIPATIVE = auto()    # Contact geometry

class Structure(Protocol):
    kind: StructureKind

    def bracket(self, state: Tensor, grad: Tensor) -> Tensor:
        """The universal evolution law: v = Lambda(z) * g"""
        ...

    def split(self, H_func: Callable) -> list[Callable] | None:
        """Optional: decompose H = T + V for splitting methods."""
        ...

    def constraints(self, state: Tensor) -> Callable | None:
        """Optional: manifold constraints for projection."""
        ...
```

**Compiler contract**: The kernel builder depends only on `state` and `bracket()`.  

---

## 5. Compiler Pipeline

### 5.1 Canonicalization Pass
- **Input**: User's Hamiltonian function `H(z)`
- **Action**: Trace computation graph, separate state from parameters
- **Output**: Clean graph for gradient computation

### 5.2 Structure Flow Builder
- **Compute gradient**: `grad = autograd(H, z)`
- **Apply structure**: `flow = structure.bracket(z, grad)`
- **Output**: Symbolic graph for `dz/dt`

### 5.3 Optimization Pass
- **Linearizer**: If flow is `M @ z`, use matrix exponentiation
- **FFT injection**: Replace convolutions with spectral methods
- **Split detection**: If `H = T + V`, use Strang splitting

### 5.4 Kernel Emission
- **Default**: Geometric predictor-corrector
- **Separable**: Leapfrog (2nd order), Yoshida (4th order)
- **Lie groups**: Cayley map, exponential map
- **Linear**: Direct matrix power

### 5.5 Optional Passes
- **Constraint projection**: RATTLE/SHAKE for manifold constraints
- **Shadow energy**: Diagnostic tracking for stability monitoring

---

## 6. Structure Implementations

### 6.1 Canonical (Symplectic)

```python
class CanonicalStructure(Structure):
    kind = StructureKind.CANONICAL

    def bracket(self, state, grad):
        n = state.shape[0] // 2
        grad_q, grad_p = grad[:n], grad[n:]
        return Tensor.cat(grad_p, -grad_q)  # (dH/dp, -dH/dq)
```

**Use cases**: Particles, springs, planets, molecular dynamics
**Integrator**: Leapfrog, Yoshida, implicit midpoint

### 6.2 Lie-Poisson (Geometric)

```python
class SO3Structure(Structure):
    kind = StructureKind.LIE_POISSON

    def bracket(self, state, grad):
        return state.cross(grad)  # L x grad(H)
```

**Use cases**: Rigid bodies, spinning tops, point vortices
**Integrator**: Lie-Trotter splitting, Cayley map

### 6.3 Quantum (Unitary)

```python
class QuantumStructure(Structure):
    kind = StructureKind.QUANTUM

    def bracket(self, state, grad):
        return -1j * grad  # -i * H|psi>
```

**Use cases**: Schrodinger equation, split-operator methods
**Integrator**: `exp(-iT*dt/2) * exp(-iV*dt) * exp(-iT*dt/2)`

### 6.4 Dissipative (Contact)

```python
class ConformalStructure(Structure):
    kind = StructureKind.DISSIPATIVE
    # Adds friction/damping while preserving structure
```

**Use cases**: Damped oscillators, thermostats
**Status**: Stub implementation

---

## 7. Shared Optimizations

Because everything flows through `bracket(z, grad)`, optimizations apply everywhere:

| Optimization | Classical | Quantum | Rigid Body |
|--------------|-----------|---------|------------|
| Linearizer (matrix exp) | Springs | Qubits | Small angles |
| FFT spectral methods | Wave equations | Split-operator | - |
| Operator splitting | T + V separation | Kinetic + Potential | Euler splitting |
| Kahan summation | All | All | All |

---

## 8. Module Layout

```
tinyphysics/
  core/
    structure.py      # StructureKind, Structure protocol, StructureProgram
    compiler.py       # StructureCompiler, compile_structure dispatch
  structures/
    canonical.py      # J bracket: (grad_p, -grad_q)
    lie_poisson.py    # Generic J(x) + SO3Structure
    commutator.py     # Quantum split-operator (641 lines, very complete)
    conformal.py      # Dissipative (contact stub)
  operators/
    spatial.py        # grad2, div2, curl2, laplacian2
    poisson.py        # FFT-based Poisson solver
  systems/
    canonical.py      # Particles, springs
    rigid_body.py     # SO3 rigid body wrapper
    vortices.py       # Point vortices with gamma weighting
    quantum.py        # Quantum system wrapper
    fluids.py         # 2D ideal fluid vorticity
  linear/
    symplectic.py     # Fast matrix-power for linear systems
  bench/
    universal_physics_bench.py
```

---

## 9. API Examples

```python
from tinyphysics import PhysicalSystem, CanonicalStructure, SO3Structure, QuantumStructure

# Classical: Two-body problem
planet = PhysicalSystem(
    state=(q, p),
    H_func=lambda q, p: p.dot(p)/(2*m) - 1/q.norm(),
    structure=CanonicalStructure(),
    project_every=4  # constraint projection cadence (if constraints are declared)
)
prog = planet.compile()
for _ in range(10000):
    (q, p), _ = prog.evolve((q, p), dt=0.01, steps=1)

# Rigid body: Spinning top
top = PhysicalSystem(
    state=L,
    H_func=lambda L: (L*L/I).sum(),
    structure=SO3Structure()
)
prog = top.compile()

# Quantum: Particle in a box
electron = PhysicalSystem(
    state=psi,
    H_func=None,  # Uses split-operator with T, V
    structure=SplitOperatorStructure(T, V, grid)
)
prog = electron.compile()
```

**Unified demo:** see `examples/universal_demo.py` for a compact tour of canonical, Lie‑Poisson, quantum, and dissipative flows.

---

## 10. Implementation Status

| Component | Status | Notes |
|-----------|--------|-------|
| Core compiler | Complete | Single entry point for all physics |
| Canonical structure | Complete | Leapfrog integration |
| Lie-Poisson structure | Complete | SO3 + generic J(x) |
| Quantum split-operator | Complete | 1D/2D/3D, FFT optimized, imaginary time |
| Dissipative structure | Partial | Contact stub present |
| Operator splitting | Partial | `split()` hooks + schedules |
| Constraint projection | Implemented | RATTLE/SHAKE in compiler |
| Benchmarks | Basic | Universal bench present |

**Estimated completion: ~85%**

---

## 10.1 Phase 3: Constraints & Thermostats (Plan)

**Goal:** Molecular‑dynamics workflows with structure‑preserving constraints and thermostats.

Planned steps:

1. **Constraint declaration API**
   - Provide a small `ConstrainedStructure` wrapper that injects constraints into the compiler.
   - Add a constrained demo + test that uses `project_every`.

2. **Thermostats**
   - Langevin (done): noise + diagnostics + regression test.
   - Nose-Hoover chain (done): minimal chain_len=1 with regression test.
   - Add a small benchmark entry (optional).

3. **Diagnostics**
   - Expose contact‑energy or entropy proxies in history for dissipative runs.

---

## 11. Non-Goals

- **No RK4**: Structure-preserving methods only
- **No backend-specific hacks**: Everything flows through `bracket()`
- **No special cases**: If it can't be expressed as a bracket, we don't want it
- **No premature optimization**: Clarity over cleverness

---

## 12. The Takeaway

There is no Classical vs Quantum distinction in this engine.

- **Classical** = constant antisymmetric bracket `J`
- **Quantum** = constant complex bracket `-i`
- **Rigid Body** = state-dependent bracket `L x`

By compiling the **structure bracket**, not the physics equations, TinyPhysics achieves:

1. **One codebase** for all Hamiltonian systems
2. **Shared optimizations** across domains
3. **Guaranteed stability** through geometric integration

**Hamiltonian in, structure-preserving simulation out.**

**Note:** A split‑operator demo lives at `examples/split_operator_demo.py`.
