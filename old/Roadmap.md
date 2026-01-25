Here is the **TinyPhysics Development Roadmap**.

This roadmap is structured by **complexity of the primitives**, not just complexity of the physics. It starts with simple separable Hamiltonians (standard gradients) and progresses to complex geometric structures (Lie algebras) and infinite-dimensional fields (Fluids/Quantum).

---

# **Universal API (Hamiltonian in → Simulation out)**

```python
from tinyphysics import UniversalSymplecticCompiler

def H(q, p):
  return 0.5 * (q*q).sum() + 0.5 * (p*p).sum()

sim = UniversalSymplecticCompiler(kind="canonical", H=H, integrator="leapfrog")
(q, p), history = sim.evolve((q, p), dt=0.01, steps=100)
```

# **TinyPhysics: The Simulation Roadmap**

## **Phase 1: The Canonical Foundation**

**Goal:** Build the core `Tensor` class, the Autograd engine, and the Symplectic Integrator.
**Physics Domain:** Particle Mechanics (1D & 2D).
**Math:**  (Constant Matrix).

| Level | Simulation Challenge | New Primitive Needed | Implementation Difficulty |
| --- | --- | --- | --- |
| **1.1** | **Harmonic Oscillator** | `SymplecticEuler` Op | 🟢 Easy |
|  | *A single mass on a spring.* | Basic Autograd (). |  |
| **1.2** | **The Double Pendulum** | `ChainRule` (Dense) | 🟡 Medium |
|  | *Chaotic motion.* | Complex potential . |  |
| **1.3** | **Kepler Problem** | `Norm` / `Rsqrt` | 🟡 Medium |
|  | *Planet orbiting a star.* | Dealing with singularity at . |  |
| **1.4** | **N-Body Gravity** | `Broadcast` (Matrix) | 🔴 Hard (Perf) |
|  | *Solar System / Galaxy.* | Parallelizing  interactions. |  |

* **Success Metric:** Energy is conserved to  precision for 100,000 steps without drifting.

---

## **Phase 2: The Geometric World (Rigid Bodies)**

**Goal:** Break free from "Position/Momentum." Introduce **Lie-Poisson** structures.
**Physics Domain:** Rotational Dynamics.
**Math:**  is no longer constant; it depends on the state (Structure Constants).

| Level | Simulation Challenge | New Primitive Needed | Implementation Difficulty |
| --- | --- | --- | --- |
| **2.1** | **The Free Rigid Body** | `CrossProduct` | 🟡 Medium |
|  | *A tennis racket flipping in void.* | . |  |
| **2.2** | **The Heavy Top** | `ProductManifold` | 🔴 Hard |
|  | *A spinning top under gravity.* | Mixing SO(3) rotation with  translation. |  |
| **2.3** | **Satellite Attitude Control** | `ControlInput` | 🔴 Hard |
|  | *Reaction wheels stabilizing a sat.* | Adding external forcing terms. |  |

* **The Hurdle:** You can no longer just say `p -= grad`. You must implement the generalized flow .
* **Visualizing:**

---

## **Phase 3: The Continuum (Fluid Dynamics)**

**Goal:** Move from discrete particles () to continuous fields ( grids).
**Physics Domain:** Hydrodynamics.
**Math:** Infinite-dimensional Lie Algebras (Diffeomorphisms).

| Level | Simulation Challenge | New Primitive Needed | Implementation Difficulty |
| --- | --- | --- | --- |
| **3.1** | **Point Vortex Model** | `Gather`/`Scatter` | 🟡 Medium |
|  | *2D Cyclones interacting.* | Hamiltonian particle dynamics (Kirchhoff). |  |
| **3.2** | **Ideal Fluid (Euler Eq)** | `FFT` (Spectral) | 🔴 Hard |
|  | *Inviscid flow in a box.* | Solving Poisson eq for pressure (). |  |
| **3.3** | **Viscous Fluid (Navier-Stokes)** | `Dissipation` Op | 🔴 Very Hard |
|  | *Real water/honey.* | Adding non-Hamiltonian (entropic) terms. |  |
| **3.4** | **Shallow Water Waves** | `FiniteDifference` | 🔴 Very Hard |
|  | *Tsunamis / Ripples.* | Handling boundary conditions. |  |

* **The "Tiny" Insight:** The 2D Euler equation for fluids is actually just a Lie-Poisson system where the "variables" are the vorticity at every pixel. It uses the exact same solver code as the Rigid Body, just with a bigger matrix!
* **Visualizing:**

---

## **Phase 4: The Quantum Realm**

**Goal:** Introduce `Complex` numbers and Unitary Operators.
**Physics Domain:** Quantum Mechanics.
**Math:** Hilbert Spaces, Commutators, FFTs.

| Level | Simulation Challenge | New Primitive Needed | Implementation Difficulty |
| --- | --- | --- | --- |
| **4.1** | **Wavepacket Spreading** | `ComplexTensor` | 🟡 Medium |
|  | *Gaussian blur in free space.* |  evolution via FFT. |  |
| **4.2** | **Quantum Tunneling** | `SplitOperator` | 🟡 Medium |
|  | *Particle hitting a wall.* | . |  |
| **4.3** | **Quantum Harmonic Oscillator** | `ImaginaryTime` | 🔴 Hard |
|  | *Finding ground states.* | Wick rotation () to minimize energy. |  |
| **4.4** | **Gross-Pitaevskii Eq** | `NonLinear` | 🔴 Very Hard |
|  | *Bose-Einstein Condensates.* | Non-linear Schrödinger (self-interaction). |  |

* **The Hurdle:** Normalization. Unlike classical physics, if , physics breaks. You need a `Renormalize` op after steps.
* **Visualizing:**

---

## **Phase 5: The Grand Bosses (Multi-Physics)**

**Goal:** Coupling different solvers together.
**Physics Domain:** Complexity.

| Level | Simulation Challenge | Interaction | Difficulty |
| --- | --- | --- | --- |
| **5.1** | **Molecular Dynamics (MD)** | **Classical + Thermodynamics** | ☢️ Expert |
|  | *Protein folding.* | Thermostats (Nose-Hoover) are non-Hamiltonian hacks. |  |
| **5.2** | **Electro-Magnetism (PIC)** | **Particles + Fields** | ☢️ Expert |
|  | *Plasma Physics.* | Particles push fields (Maxwell), fields push particles. |  |
| **5.3** | **Semi-Classical Gravity** | **Quantum + Classical** | ☢️ Impossible? |
|  | *A quantum particle in a gravity well.* | Coupling Schrödinger to Newton. |  |

---

### **Recommended Progression Path**

If you are coding this alone, follow this strictly linear path to maintain sanity:

1. **Harmonic Oscillator** (Prove the Autograd works).
2. **N-Body Gravity** (Prove the GPU performance works).
3. **Rigid Body Top** (Prove the  structure works).
4. **1D Schrödinger** (Prove the Complex numbers work).
5. **2D Fluid (Vorticity)** (Prove the Scale works).
