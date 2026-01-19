Here is a reference document summarizing the architectural foundations of **TinyPhysics**.

---

# **Architectural Decision Record: The Physics Engine Core**

**Topic:** Core Formalism for Simulation (Hamiltonian vs. Lagrangian)
**Status:** Decided (Hamiltonian/Poisson)
**Context:** We are building a primitive-based, compilable physics engine inspired by `tinygrad`.

---

## **1. The Foundation: Symplectic Physics**

Classical mechanics is not just about "Force equals mass times acceleration." It is about the geometry of **Phase Space**.

* **Phase Space ():** The space of all possible states, usually defined by position  and momentum .
* **Symplecticity:** In a real physical system, information is never destroyed. If you take a blob of possible starting states and evolve them forward, the *shape* of the blob changes, but its *volume* remains exactly constant. This is **Liouville's Theorem**.

**Why this matters for a Solver:**
Standard numerical methods (like Runge-Kutta 4) do not respect this geometry. They accidentally "leak" energy or volume, causing simulations to dampen or explode over time. A **Symplectic Solver** is designed to respect this geometry perfectly, keeping the simulation stable for billions of steps.

---

## **2. The Tool: Poisson Brackets**

The Poisson Bracket is the "algebraic operator" that drives symplectic physics. It allows us to calculate how *any* quantity changes over time without writing specific force laws.

Given two functions  and  depending on phase space , the canonical Poisson Bracket is:

**The Universal Equation of Motion:**
The time evolution of *any* variable  (state, energy, momentum) is driven by the Hamiltonian (Energy) :

If we set  to be the state vector  itself, we get the evolution equation:



Where  is the **Symplectic Matrix** (or Structure Matrix).

---

## **3. Comparison: Poisson vs. Lagrangian**

We had two main mathematical frameworks to choose from for our architecture.

### **Option A: The Lagrangian (Variational) Approach**

* **Concept:** "Nature is lazy." Particles move along the path that minimizes the **Action** .


* **The Solver Implementation:** To find the next position , we must solve the discrete Euler-Lagrange equation.
* **The Problem:** This equation is **Implicit**.



To find , we must use a root-finding algorithm (like Newton-Raphson) *inside every single timestep*.

### **Option B: The Poisson (Hamiltonian) Approach (Selected)**

* **Concept:** "Energy flows." State flows through phase space along the contours of constant energy.
* **The Solver Implementation:** To find the next state , we compute the gradient of Energy and apply the structure matrix .
* **The Benefit:** This equation is **Explicit**.



We can calculate the next step directly from the current step. No guessing, no internal loops.

---

## **4. Why We Chose Poisson Brackets**

We selected the Poisson/Hamiltonian formulation for **TinyPhysics** for three engineering reasons:

### **I. Compiler Compatibility (The "Graph" Argument)**

* **Poisson:** The update step is a straight-line computational graph (Load  Grad  Mult  Store). This is trivial to fuse into a single GPU kernel.
* **Lagrangian:** The update step involves an iterative solver loop (`while error > epsilon`). Loops are notoriously difficult to compile efficiently on GPUs (due to thread divergence) and break the simple "Autograd Graph" paradigm.

### **II. State-Space Generalization**

* **Poisson:** The bracket formalism  works unchanged for:
* **Particles:** 
* **Rigid Bodies:**  (Lie-Poisson)
* **Quantum Mechanics:**  (Commutator)


* **Lagrangian:** Formulating fluids or quantum mechanics purely using  often requires complex, domain-specific coordinate patches or constrained optimization that doesn't fit a unified solver.

### **III. Native "Autograd" Alignment**

Deep Learning frameworks (Tinygrad, JAX, PyTorch) are effectively "Gradient Engines."

* Hamiltonian mechanics is defined by **Gradients of Energy** ().
* Lagrangian mechanics is defined by **Stationary Points of Action** ().

Therefore, Hamiltonian mechanics is "native" to AI hardware. We get the forces for free simply by calling `.backward()` on the Energy scalar.

---

## **Summary Table**

| Feature | **Poisson (Hamiltonian)** | **Variational (Lagrangian)** |
| --- | --- | --- |
| **Primary Object** | Energy  | Action  |
| **Math Type** | Explicit ODE () | Implicit Equation () |
| **Computational Cost** | **Low** (1 pass per step) | **High** (Iterative solver per step) |
| **GPU Suitability** | **Perfect** (Stream processing) | **Poor** (Branching/Loops) |
| **Quantum Extension** | Direct (Commutators) | Complex (Path Integrals) |
| **Our Verdict** | **SELECTED** | Rejected for Core |
