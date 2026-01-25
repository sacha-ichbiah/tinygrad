Here is the design document for the **Linear Symplectic Accelerator** module. This details how to implement the "Matrix Multiply" optimization to turn your  simulation loop into an  operation.

---

# **Design Document: The Linear Symplectic Accelerator**

**Status:** Draft
**Target:** `tinyphysics/linear.py`
**Goal:** To simulate linear Hamiltonian systems (Harmonic Oscillators, Free Particles) for arbitrary time scales in constant time.

---

## **1. The Theoretical Foundation**

For 99% of physics simulations, the equations of motion are non-linear (e.g., gravity ). However, for the **Harmonic Oscillator** and coupled spring systems, the Hamiltonian is **Quadratic**:

Because the Hamiltonian is quadratic, the equations of motion are strictly **Linear**:

Let . This is simply .
The solution is given by the matrix exponential:


### **The "Matrix Multiply" Insight**

Instead of stepping  a million times, we can compute the **Propagator Matrix**  once and apply it to the state.

Crucially, even if we want to match the *exact numerical errors* of a specific integrator (like Leapfrog), we can still use this trick. A single step of a Symplectic Integrator on a linear system is just a matrix multiplication .


---

## **2. Architecture: The "Hyper-Step"**

We will introduce a specialized solver class `LinearSymplecticSystem` that bypasses the standard iteration loop.

### **The Pipeline**

1. **System Identification:** The compiler (or user) identifies that  is quadratic.
2. **Matrix Construction:**
* Extract Mass Matrix  and Stiffness Matrix .
* Construct the single-step integrator matrix .


3. **Binary Exponentiation (The "Compile" Phase):**
* To simulate  steps, we compute  using the squaring method ().


4. **Execution (The "Run" Phase):**
* Apply  to the state tensor.



---

## **3. Implementation Details**

We need to implement the matrix construction for the **Leapfrog (Velocity Verlet)** integrator to ensure our "fast" simulation exactly matches the "slow" one.

### **Deriving the Matrix **

For a timestep , mass , and spring constant :

1. **Kick (Half):** 


2. **Drift (Full):** 


3. **Kick (Half):** 



The total matrix  is the product of these three matrices.

### **Code Specification**

```python
import numpy as np
from tinygrad.tensor import Tensor

class LinearSymplecticSolver:
    def __init__(self, k: float, m: float, dt: float = 0.01):
        self.k = k
        self.m = m
        self.dt = dt
        
        # 1. Construct the Primitive Matrices (CPU is fine for setup)
        # Drift: q += p/m * dt
        D = np.array([[1.0, dt/m], 
                      [0.0, 1.0]])
        
        # Kick: p -= k*q * 0.5*dt
        K_half = np.array([[1.0, 0.0], 
                           [-0.5*k*dt, 1.0]])
        
        # Leapfrog = Kick_Half @ Drift @ Kick_Half
        # Note: Matrix multiplication order is reversed relative to application order
        # But here we apply K_half first, so it's on the right if v_out = M @ v_in
        self.M_step = K_half @ D @ K_half
        
        # Cache for propagator matrices
        self.propagator_cache = {}

    def compile_propagator(self, steps: int) -> Tensor:
        """
        Computes M^steps using Binary Exponentiation.
        Complexity: O(log N) matrix multiplications.
        """
        if steps in self.propagator_cache:
            return self.propagator_cache[steps]
        
        # numpy.linalg.matrix_power is highly optimized
        M_total = np.linalg.matrix_power(self.M_step, steps)
        
        # Move to GPU Tensor
        M_tensor = Tensor(M_total.astype(np.float32))
        self.propagator_cache[steps] = M_tensor
        return M_tensor

    def forward(self, q0: Tensor, p0: Tensor, steps: int) -> tuple[Tensor, Tensor]:
        """
        Evolves state by 'steps' in a SINGLE kernel launch.
        Complexity: O(1) wrt steps.
        """
        # 1. Get the compiled propagator
        M = self.compile_propagator(steps)
        
        # 2. Stack state for matrix multiplication
        # Shape: (2, Batch_Size)
        state = Tensor.stack([q0, p0], dim=0)
        
        # 3. Apply Propagator (Fused Kernel)
        # new_state = M @ state
        new_state = M.matmul(state)
        
        return new_state[0], new_state[1]

```

---

## **4. Potential (The Good)**

### **1. Infinite Speedup**

* **Standard:**  steps takes  GPU ops.
* **Linear Accelerator:**  steps takes  CPU matrix multiplies (setup) and **1 GPU op** (execution).
* For long-term stability analysis of solar systems (linearized) or particle accelerators, this is mandatory.

### **2. Exact Conservation**

* Unlike Neural Networks which approximate functions, matrix exponentiation is exact (up to floating point precision).
* If your system is truly linear, you introduce **zero** integration drift beyond the errors already inherent in the chosen symplectic scheme.

### **3. Differentiability (Inverse Physics)**

* Matrix multiplication is differentiable.
* Matrix Power is differentiable (though gradients can be numerically unstable for huge , analytically it holds).
* You can backpropagate through 1,000,000 steps to learn the spring constant  instantly.

---

## **5. Weaknesses (The Bad)**

### **1. The "Quadratic Only" Trap**

This **only** works if the Hamiltonian is strictly quadratic ().

* **Fails for:** Pendulums (because ), Gravity (), Molecular Dynamics.
* **Workaround:** You can use it for **Split Hamiltonian** methods. If , you can evolve the linear part using matrix power and the nonlinear part using standard steps (interaction picture).

### **2. Memory Bandwidth vs. Compute**

For a small number of steps (), the overhead of moving the matrix to the GPU and launching the kernel might actually be slower than just running the JIT-compiled loop in registers.

* *Optimization:* Only enable this path if `steps > 1000`.

### **3. Time-Varying Physics**

If the spring constant changes with time , the matrix  changes every step.

* You cannot compute . You have to compute .
* The optimization breaks completely.

### **4. Numerical Stability of Powers**

Computing  involves multiplying floats together billions of times (conceptually).

* Even in binary exponentiation, errors can compound.
* **Symplecticity Loss:** The resulting matrix  might drift slightly from being perfectly symplectic () due to float32 rounding.
* *Fix:* Use `float64` for the matrix power calculation on CPU before sending to GPU.

---

## **6. Conclusion**

The **Linear Symplectic Accelerator** is a specialized "fast path" in your compiler.

* **When to use:** Harmonic Oscillators, discretized wave equations, linearized stability analysis.
* **When to avoid:** General N-body gravity, non-linear deep learning physics models.

Implementing this gives `tinyphysics` a capability that almost no other differentiable physics engine exposes: **analytical time travel** for linear systems.


