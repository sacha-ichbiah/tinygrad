# TinyPhysics Roadmap

**Vision:** A universal physics compiler that scales from toy problems to production simulations.

---

## Current Status (Phase 1: Foundation)

| Component | Status | Notes |
|-----------|--------|-------|
| Core compiler | Done | Universal `bracket()` interface |
| Canonical structure | Done | Leapfrog integration |
| Lie-Poisson structure | Done | SO3 + generic J(x) |
| Quantum split-operator | Done | 1D/2D/3D, FFT optimized |
| Dissipative structure | Partial | Contact stub present |
| Constraint projection | Implemented | RATTLE/SHAKE in compiler |
| Operator splitting | Partial | `split()` hooks + schedules |
| Benchmarks | Basic | Universal bench present |
| PhysicalSystem frontend | Done | Thin wrapper over `compile_structure` |

**Completion: ~85%** of core architecture.

---

## Development Phases

### Phase 2: Scale
**Goal:** Handle 10^6+ particles efficiently

- [x] Neighbor lists with cell-linked sorting (CPU baseline)
- [x] Barnes-Hut tree for O(N log N) gravity (CPU baseline)
- [x] GPU kernel fusion for force computation (tensor-force path)
- [x] Batch processing for independent subsystems (batch-aware N-body)

**Status note:** A minimal CPU cell‑linked neighbor list benchmark is available at `tinyphysics/bench/neighbors_bench.py`. N‑body now includes
tensor‑bins and Barnes‑Hut paths, plus a `NBodySystem` wrapper for batch‑aware runs.

### Phase 3: Constraints & Thermostats
**Goal:** Support molecular dynamics workflows

- [ ] RATTLE/SHAKE for holonomic constraints (bonds, angles)
- [ ] Nose-Hoover chains for canonical ensemble (NVT)
- [ ] Langevin thermostat (dissipative structure)
- [ ] Berendsen barostat for NPT ensemble

### Phase 4: Long-Range Forces
**Goal:** Accurate electrostatics and gravity at scale

- [ ] Ewald summation for periodic systems
- [ ] Particle Mesh Ewald (PME) with FFT
- [ ] Fast Multipole Method (FMM) for open boundaries
- [ ] Gravitational softening options

### Phase 5: Multi-Physics
**Goal:** Coupled systems and hybrid methods

- [ ] Particle-in-cell (PIC) for plasmas
- [ ] Coupled particle-field evolution
- [ ] QM/MM hybrid quantum-classical
- [ ] Adaptive timestepping

---

## Challenging Use Cases

### 1. Galaxy Formation

**The Problem:** Simulate gravitational collapse and evolution of 10^6-10^9 particles over billions of years.

**Scale:**
- 10^6 particles (laptop) to 10^9 particles (cluster)
- 10^4 - 10^6 timesteps
- Softened gravity: `F = -Gm1m2 / (r^2 + eps^2)`

**Structure:** Canonical (q, p) with long-range forces

**Hamiltonian:**
```python
def H_galaxy(q, p, m, eps=0.01):
    T = (p * p / (2 * m)).sum()           # Kinetic
    r = pairwise_distance(q)              # N x N distances
    V = -G * (m[:, None] * m[None, :] / (r + eps)).sum() / 2
    return T + V
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Canonical structure | Done | Hamilton's equations |
| Barnes-Hut tree | Done | O(N log N) vs O(N^2) |
| Gravitational softening | Done | Prevent singularities |
| Adaptive timestep | Not started | Close encounters |
| GPU batching | Not started | Parallel force calculation |

**Success Metric:** Reproduce Plummer sphere collapse in < 1 minute for N=10^5.

---

### 2. Molecular Dynamics (Lennard-Jones Fluid)

**The Problem:** Simulate atoms interacting via Lennard-Jones potential, maintain temperature.

**Scale:**
- 10^3 - 10^6 atoms
- 10^6 - 10^9 timesteps (ns to us timescales)
- Cutoff radius for efficiency

**Structure:** Canonical + Dissipative (for thermostat)

**Hamiltonian:**
```python
def H_lj(q, p, m, sigma=1.0, epsilon=1.0, r_cut=2.5):
    T = (p * p / (2 * m)).sum()
    r = pairwise_distance(q)
    # Lennard-Jones: 4*eps * [(sigma/r)^12 - (sigma/r)^6]
    lj = 4 * epsilon * ((sigma/r)**12 - (sigma/r)**6)
    V = lj[r < r_cut * sigma].sum() / 2
    return T + V
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Canonical structure | Done | NVE ensemble |
| Neighbor lists | Not started | O(N) force calculation |
| Cutoff with shift | Not started | Energy conservation |
| Nose-Hoover thermostat | Not started | NVT ensemble |
| RATTLE constraints | Not started | Rigid water (TIP3P) |
| Periodic boundaries | Partial | Bulk simulation |

**Success Metric:** Reproduce liquid argon RDF and diffusion coefficient.

---

### 3. Quantum Chemistry (Time-Dependent Hartree-Fock)

**The Problem:** Propagate electronic wavefunction under mean-field Hamiltonian.

**Scale:**
- 10^2 - 10^4 basis functions
- Sparse Hamiltonians
- Femtosecond dynamics

**Structure:** Quantum (unitary)

**Hamiltonian:**
```python
def H_tdhf(C, F):
    # C: MO coefficients (n_basis x n_occ)
    # F: Fock matrix (n_basis x n_basis), depends on C
    return (C.T @ F @ C).trace()  # Electronic energy
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Quantum structure | Done | Unitary evolution |
| Sparse matrix support | Not started | Large basis sets |
| Magnus expansion | Not started | Time-dependent H |
| Self-consistent field | Not started | F depends on C |

**Success Metric:** Reproduce H2 dissociation dynamics.

---

### 4. Plasma Physics (Particle-in-Cell)

**The Problem:** Simulate charged particles + electromagnetic fields self-consistently.

**Scale:**
- 10^6 - 10^9 macro-particles
- 10^3 - 10^6 grid cells
- Coupled particle + field evolution

**Structure:** Canonical (particles) + Field equations

**Equations:**
```
Particles:  dq/dt = v,  dv/dt = (q/m)(E + v x B)
Fields:     dE/dt = curl(B) - J,  dB/dt = -curl(E)
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Canonical structure | Done | Particle dynamics |
| Spectral Poisson solver | Done | Field solve |
| Particle-mesh interpolation | Not started | Charge/current deposition |
| Charge conservation | Not started | Physical consistency |
| Boris pusher | Not started | Magnetic field integration |

**Success Metric:** Reproduce two-stream instability growth rate.

---

### 5. Turbulent Fluids (2D Euler)

**The Problem:** Simulate inviscid fluid with energy cascade across scales.

**Scale:**
- 10^6 - 10^9 grid points (1024^2 to 32768^2)
- Spectral accuracy
- Long-time statistics

**Structure:** Lie-Poisson (vorticity)

**Hamiltonian:**
```python
def H_euler(omega, grid):
    # omega: vorticity field
    # Invert Poisson: nabla^2 psi = -omega
    psi = poisson_solve_fft(omega, grid)
    # Energy = integral of omega * psi / 2
    return (omega * psi).sum() * grid.dx * grid.dy / 2
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Lie-Poisson structure | Done | Preserves enstrophy |
| FFT Poisson solver | Done | Spectral accuracy |
| Dealiasing (2/3 rule) | Not started | Prevent aliasing |
| Energy/enstrophy diagnostics | Not started | Verify conservation |

**Success Metric:** Reproduce inverse energy cascade spectrum E(k) ~ k^(-5/3).

---

### 6. Protein Folding (Implicit Solvent)

**The Problem:** Simulate protein dynamics with effective solvent, sample conformational space.

**Scale:**
- 10^4 - 10^5 atoms
- Microsecond timescales
- Langevin dynamics for sampling

**Structure:** Canonical + Dissipative

**Hamiltonian:**
```python
def H_protein(q, p, topology):
    T = kinetic_energy(p, topology.masses)
    V_bond = bond_energy(q, topology)
    V_angle = angle_energy(q, topology)
    V_dihedral = dihedral_energy(q, topology)
    V_nonbond = lennard_jones(q, topology) + coulomb(q, topology)
    V_implicit = gbsa_solvation(q, topology)  # Generalized Born
    return T + V_bond + V_angle + V_dihedral + V_nonbond + V_implicit
```

**Required Features:**
| Feature | Status | Why Needed |
|---------|--------|------------|
| Canonical structure | Done | Hamiltonian dynamics |
| Dissipative structure | Stub | Langevin thermostat |
| SHAKE constraints | Not started | Rigid H-bonds |
| Neighbor lists | Not started | Nonbonded forces |
| GBSA solvation | Not started | Implicit water |

**Success Metric:** Fold Trp-cage (20 residues) to native state.

---

## Technical Deep Dives

### Barnes-Hut Tree (Galaxy Formation)

```
Algorithm:
1. Build octree from particle positions - O(N log N)
2. For each particle:
   - Walk tree from root
   - If cell is "far" (theta criterion): use monopole
   - Else: recurse into children
3. Accumulate forces - O(N log N) total

Key parameter: theta = cell_size / distance
  theta = 0: exact (O(N^2))
  theta = 0.5: typical accuracy
  theta = 1.0: fast but less accurate
```

### RATTLE Algorithm (Molecular Dynamics)

```
Algorithm for constrained dynamics:
1. Unconstrained velocity half-step: p' = p - dt/2 * grad(V)
2. Unconstrained position step: q' = q + dt * p'/m
3. Project positions onto constraint manifold (SHAKE)
4. Update velocities: p'' = m * (q' - q) / dt
5. Unconstrained velocity half-step: p''' = p'' - dt/2 * grad(V(q'))
6. Project velocities onto constraint tangent space (RATTLE)

Constraint: |q_i - q_j| = d_ij (bond length)
```

### Nose-Hoover Thermostat (Temperature Control)

```
Extended Hamiltonian:
H_extended = H(q, p) + p_eta^2 / (2*Q) + g*kT*eta

Additional equations:
d(eta)/dt = p_eta / Q
d(p_eta)/dt = sum(p^2/m) - g*kT

where:
  eta: thermostat position
  p_eta: thermostat momentum
  Q: thermostat mass (controls coupling strength)
  g: degrees of freedom
```

---

## Success Metrics

| Use Case | Metric | Target |
|----------|--------|--------|
| Galaxy | Plummer collapse (N=10^5) | < 1 min |
| Molecular | Argon RDF error | < 5% |
| Quantum | H2 dissociation energy | < 0.01 Ha |
| Plasma | Two-stream growth rate | < 10% error |
| Fluids | Energy spectrum slope | k^(-5/3) ± 0.1 |
| Protein | Trp-cage RMSD | < 2 Å |

---

## Dependencies on tinygrad Core

| Feature | tinygrad Requirement | Status |
|---------|---------------------|--------|
| Tree traversal | Dynamic indexing | Needs work |
| Sparse matrices | Sparse tensor ops | Not available |
| Neighbor lists | Sorting primitives | Available |
| FFT | `Tensor.fft()` | Available |
| Complex numbers | Complex dtype | Available |
| Autograd | `Tensor.grad()` | Available |

---

## Timeline (Rough Estimates)

```
Phase 2 (Scale):           Enables galaxy, basic MD
Phase 3 (Constraints):     Enables realistic MD, proteins
Phase 4 (Long-Range):      Enables accurate electrostatics
Phase 5 (Multi-Physics):   Enables plasma, QM/MM
```

Each phase unlocks new use cases while keeping the core architecture unchanged. The structure-preserving approach means simulations stay physical regardless of complexity.

---

**The goal:** Run galaxy formation, molecular dynamics, and quantum chemistry with the same compiler, the same optimizations, and guaranteed stability.
