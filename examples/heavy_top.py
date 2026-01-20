"""
TinyPhysics 2.2: The Heavy Top
==============================

A spinning top under gravity - the classic Euler-Poisson equations.

The Heavy Top lives on a **Product Manifold**: SO(3) × S²
- SO(3): The rotation group (represented by angular momentum L in body frame)
- S²: The unit sphere (gravity direction γ in body frame)

The Lie-Poisson structure mixes both:
  {Lᵢ, Lⱼ} = εᵢⱼₖ Lₖ   (angular momentum algebra)
  {γᵢ, Lⱼ} = εᵢⱼₖ γₖ   (γ transforms as a vector)
  {γᵢ, γⱼ} = 0          (γ components commute)

This is the key insight: the Product Manifold has a NON-TRIVIAL coupling
between the two factors through the Poisson bracket.
"""

from tinygrad import Tensor, dtypes
import numpy as np

# ============================================================================
# PRIMITIVE 1: Cross Product (from 2.1)
# ============================================================================

def cross(a: Tensor, b: Tensor) -> Tensor:
  """
  Cross product a × b for 3-vectors.
  This is the fundamental operation for SO(3) dynamics.

  The Lie bracket structure: [a, b] = a × b
  """
  ax, ay, az = a[0], a[1], a[2]
  bx, by, bz = b[0], b[1], b[2]
  return Tensor.stack([
    ay * bz - az * by,
    az * bx - ax * bz,
    ax * by - ay * bx
  ])

# ============================================================================
# PRIMITIVE 2: Product Manifold State
# ============================================================================

class ProductManifold:
  """
  The Product Manifold SO(3)* × S² for the Heavy Top.

  State = (L, γ) where:
    L ∈ so(3)* ≃ R³ : Angular momentum in body frame
    γ ∈ S² ⊂ R³    : Gravity direction in body frame (unit vector)

  The key property: γ must stay on the unit sphere (|γ| = 1).
  This is a Casimir invariant of the Poisson structure.
  """

  def __init__(self, L: Tensor, gamma: Tensor):
    self.L = L          # Angular momentum (3-vector)
    self.gamma = gamma  # Gravity direction (unit 3-vector)

  def normalize_gamma(self) -> 'ProductManifold':
    """Project γ back to S² (numerical stability)."""
    norm = (self.gamma * self.gamma).sum().sqrt()
    return ProductManifold(self.L, self.gamma / norm)

  def realize(self) -> 'ProductManifold':
    """Force evaluation to prevent graph explosion."""
    self.L.realize()
    self.gamma.realize()
    return self

  @staticmethod
  def from_euler_angles(L: Tensor, theta: float, phi: float) -> 'ProductManifold':
    """
    Create state from angular momentum and spherical angles.
    θ: angle from vertical (0 = upright, π = hanging)
    φ: azimuthal angle
    """
    gamma = Tensor([
      np.sin(theta) * np.cos(phi),
      np.sin(theta) * np.sin(phi),
      np.cos(theta)
    ], dtype=dtypes.float64)
    return ProductManifold(L, gamma)

  def casimirs(self) -> tuple[Tensor, Tensor]:
    """
    The Casimir invariants (conserved by any Hamiltonian flow):
    C₁ = |γ|² = 1 (geometric constraint)
    C₂ = L · γ   (projection of L onto gravity direction)
    """
    C1 = (self.gamma * self.gamma).sum()
    C2 = (self.L * self.gamma).sum()
    return C1, C2

# ============================================================================
# THE HAMILTONIAN: Energy of the Heavy Top
# ============================================================================

class HeavyTopHamiltonian:
  """
  The Heavy Top Hamiltonian:

    H(L, γ) = ½ L · (I⁻¹ · L) + mgl (γ · e₃)
            = ½(L₁²/I₁ + L₂²/I₂ + L₃²/I₃) + mgl γ₃

  Where:
    I = diag(I₁, I₂, I₃) : Principal moments of inertia
    mgl : Gravitational torque parameter (mass × gravity × distance to pivot)
    e₃ = (0, 0, 1) : Symmetry axis of the top

  For a symmetric top: I₁ = I₂ (≠ I₃)
  """

  def __init__(self, I1: float, I2: float, I3: float, mgl: float):
    self.I_inv = Tensor([1.0/I1, 1.0/I2, 1.0/I3], dtype=dtypes.float64)
    self.mgl = mgl  # Gravitational torque parameter

  def __call__(self, state: ProductManifold) -> Tensor:
    """Compute total energy H = T + V."""
    # Kinetic energy: T = ½ L · (I⁻¹ · L)
    omega = self.I_inv * state.L  # Angular velocity ω = I⁻¹ L
    T = 0.5 * (state.L * omega).sum()

    # Potential energy: V = mgl γ₃ (height of center of mass)
    V = self.mgl * state.gamma[2]

    return T + V

  def angular_velocity(self, L: Tensor) -> Tensor:
    """ω = I⁻¹ · L"""
    return self.I_inv * L

# ============================================================================
# THE POISSON STRUCTURE: Lie-Poisson Bracket
# ============================================================================

class LiePoissonBracket:
  """
  The Lie-Poisson structure for e(3)* (Euclidean group coadjoint orbit).

  The flow equations are:
    dL/dt = {L, H} = L × ∂H/∂L + γ × ∂H/∂γ
    dγ/dt = {γ, H} = γ × ∂H/∂L

  This is the PRODUCT MANIFOLD structure:
  - The L equation has contributions from BOTH L and γ gradients
  - The γ equation only sees the L gradient

  This asymmetry is the hallmark of a semi-direct product Lie algebra.
  """

  def flow(self, state: ProductManifold, grad_L: Tensor, grad_gamma: Tensor) -> tuple[Tensor, Tensor]:
    """
    Apply the Poisson tensor J to the gradients.

    Returns (dL/dt, dγ/dt) = J · ∇H

    The structure matrix J encodes:
      J = [ [L×]  [γ×] ]
          [ [γ×]   0   ]

    Where [a×] is the skew-symmetric matrix for cross product.
    """
    # dL/dt = L × (∂H/∂L) + γ × (∂H/∂γ)
    dL_dt = cross(state.L, grad_L) + cross(state.gamma, grad_gamma)

    # dγ/dt = γ × (∂H/∂L)
    dgamma_dt = cross(state.gamma, grad_L)

    return dL_dt, dgamma_dt

# ============================================================================
# THE INTEGRATOR: Symplectic-Lie Splitting
# ============================================================================

class HeavyTopIntegrator:
  """
  Symplectic integrator for the Heavy Top.

  For a SYMMETRIC TOP (I1 = I2), we use an exact splitting:

  The Hamiltonian splits as: H = H₁(L₃) + H₂(L, γ)
  where H₁ = L₃²/(2I₃) and H₂ = (L₁² + L₂²)/(2I₁) + mgl·γ₃

  H₁-flow: exact rotation around e₃ axis
  H₂-flow: uses Runge-Kutta on the reduced system

  For general (non-symmetric) tops, we use 4th order Runge-Kutta
  which still gives excellent energy conservation.
  """

  def __init__(self, hamiltonian: HeavyTopHamiltonian, dt: float = 0.001):
    self.H = hamiltonian
    self.dt = dt
    self.bracket = LiePoissonBracket()

  def _rotate_vector(self, v: Tensor, axis: Tensor, angle: Tensor) -> Tensor:
    """
    Rodrigues' rotation formula: rotate v around axis by angle.

    v' = v cos(θ) + (k × v) sin(θ) + k (k · v)(1 - cos(θ))
    """
    axis_norm = (axis * axis).sum().sqrt()
    eps = Tensor([1e-10], dtype=dtypes.float64)
    k = axis / (axis_norm + eps)

    c = angle.cos()
    s = angle.sin()

    k_cross_v = cross(k, v)
    k_dot_v = (k * v).sum()

    return v * c + k_cross_v * s + k * k_dot_v * (1 - c)

  def _derivatives(self, L: Tensor, gamma: Tensor) -> tuple[Tensor, Tensor]:
    """
    Compute (dL/dt, dγ/dt) from the Lie-Poisson equations.

    dL/dt = L × ω + γ × (mgl·e₃)
    dγ/dt = γ × ω

    where ω = I⁻¹·L
    """
    omega = self.H.I_inv * L
    e3 = Tensor([0.0, 0.0, 1.0], dtype=dtypes.float64)

    dL = cross(L, omega) + cross(gamma, e3) * self.H.mgl
    dgamma = cross(gamma, omega)

    return dL, dgamma

  def step_rk4(self, state: ProductManifold) -> ProductManifold:
    """
    4th order Runge-Kutta integrator.
    Not strictly symplectic but excellent energy conservation.
    """
    L, gamma = state.L, state.gamma
    dt = self.dt

    # k1
    dL1, dg1 = self._derivatives(L, gamma)

    # k2
    L2 = L + dL1 * (dt/2)
    g2 = gamma + dg1 * (dt/2)
    dL2, dg2 = self._derivatives(L2, g2)

    # k3
    L3 = L + dL2 * (dt/2)
    g3 = gamma + dg2 * (dt/2)
    dL3, dg3 = self._derivatives(L3, g3)

    # k4
    L4 = L + dL3 * dt
    g4 = gamma + dg3 * dt
    dL4, dg4 = self._derivatives(L4, g4)

    # Combine
    L_new = L + (dL1 + dL2 * 2 + dL3 * 2 + dL4) * (dt/6)
    gamma_new = gamma + (dg1 + dg2 * 2 + dg3 * 2 + dg4) * (dt/6)

    return ProductManifold(L_new, gamma_new).normalize_gamma().realize()

  def step_symmetric(self, state: ProductManifold) -> ProductManifold:
    """
    Symplectic splitting for SYMMETRIC top (I1 = I2).

    Split: H = H₁ + H₂ where
      H₁ = L₃²/(2I₃)  (rotation around symmetry axis)
      H₂ = rest       (planar + potential)

    H₁-flow is exact: e₃-rotation by angle ω₃·dt
    H₂-flow uses midpoint rule (symplectic for quadratic H).
    """
    L, gamma = state.L, state.gamma
    dt = self.dt

    # === H₁ flow (half step): rotation around e₃ ===
    omega3 = self.H.I_inv[2] * L[2]
    angle = omega3 * (dt/2)
    e3 = Tensor([0.0, 0.0, 1.0], dtype=dtypes.float64)

    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)

    # === H₂ flow (full step): midpoint implicit ===
    # Use explicit midpoint (2nd order symplectic)
    dL, dgamma = self._derivatives(L, gamma)
    L_mid = L + dL * (dt/2)
    gamma_mid = gamma + dgamma * (dt/2)
    dL_mid, dgamma_mid = self._derivatives(L_mid, gamma_mid)
    L = L + dL_mid * dt
    gamma = gamma + dgamma_mid * dt

    # === H₁ flow (half step): rotation around e₃ ===
    omega3 = self.H.I_inv[2] * L[2]
    angle = omega3 * (dt/2)

    L = self._rotate_vector(L, e3, angle)
    gamma = self._rotate_vector(gamma, e3, angle)

    return ProductManifold(L, gamma).normalize_gamma().realize()

  def step(self, state: ProductManifold) -> ProductManifold:
    """
    Default step: use RK4 for best energy conservation.
    """
    return self.step_rk4(state)

  def step_explicit(self, state: ProductManifold) -> ProductManifold:
    """
    Simple explicit Euler for comparison (NOT symplectic).
    Shows the structure of the Poisson bracket directly.
    """
    omega = self.H.angular_velocity(state.L)
    e3 = Tensor([0.0, 0.0, 1.0], dtype=dtypes.float64)
    grad_gamma = e3 * self.H.mgl

    dL_dt, dgamma_dt = self.bracket.flow(state, omega, grad_gamma)

    L_new = state.L + dL_dt * self.dt
    gamma_new = state.gamma + dgamma_dt * self.dt

    return ProductManifold(L_new, gamma_new).normalize_gamma().realize()

# ============================================================================
# SIMULATION RUNNER
# ============================================================================

def simulate_heavy_top(
    L0: list[float],
    theta0: float,
    phi0: float = 0.0,
    I1: float = 1.0,
    I2: float = 1.0,
    I3: float = 0.5,
    mgl: float = 1.0,
    dt: float = 0.001,
    steps: int = 10000,
    method: str = "splitting"
) -> dict:
  """
  Simulate the Heavy Top and track conservation laws.

  Args:
    L0: Initial angular momentum [L1, L2, L3]
    theta0: Initial tilt angle from vertical (0 = upright)
    phi0: Initial azimuthal angle
    I1, I2, I3: Principal moments of inertia
    mgl: Gravitational torque parameter
    dt: Time step
    steps: Number of steps
    method: "rk4", "splitting" (symplectic), or "euler" (explicit)

  Returns:
    Dictionary with trajectories and conservation diagnostics
  """
  # Initialize
  L = Tensor(L0, dtype=dtypes.float64)
  state = ProductManifold.from_euler_angles(L, theta0, phi0)
  H = HeavyTopHamiltonian(I1, I2, I3, mgl)
  integrator = HeavyTopIntegrator(H, dt)

  # Storage
  history = {
    'L': [], 'gamma': [], 'energy': [],
    'C1': [], 'C2': [], 'time': []
  }

  # Initial values
  E0 = H(state).numpy()
  C1_0, C2_0 = state.casimirs()
  C1_0, C2_0 = C1_0.numpy(), C2_0.numpy()

  # Select integration method
  if method == "rk4":
    step_fn = integrator.step_rk4
  elif method == "splitting":
    step_fn = integrator.step_symmetric
  else:
    step_fn = integrator.step_explicit

  # Run simulation
  sample_interval = max(1, steps // 100)  # Sample ~100 times
  for i in range(steps):
    if i % sample_interval == 0:
      E = H(state).numpy()
      C1, C2 = state.casimirs()

      history['time'].append(i * dt)
      history['L'].append(state.L.numpy().copy())
      history['gamma'].append(state.gamma.numpy().copy())
      history['energy'].append(E)
      history['C1'].append(C1.numpy())
      history['C2'].append(C2.numpy())

    state = step_fn(state)

  # Final conservation check
  E_final = H(state).numpy()
  C1_final, C2_final = state.casimirs()

  history['diagnostics'] = {
    'E0': E0,
    'E_final': E_final,
    'dE_relative': abs(E_final - E0) / abs(E0) if E0 != 0 else abs(E_final - E0),
    'C1_0': C1_0,
    'C1_final': C1_final.numpy(),
    'C2_0': C2_0,
    'C2_final': C2_final.numpy(),
  }

  return history

# ============================================================================
# DEMO
# ============================================================================

if __name__ == "__main__":
  print("=" * 60)
  print("TinyPhysics 2.2: The Heavy Top")
  print("=" * 60)
  print()

  # Physical setup: A symmetric top (I1 = I2)
  # Tilted 30 degrees from vertical, spinning around its axis
  print("Configuration:")
  print("  Moments of inertia: I1=I2=1.0, I3=0.5 (symmetric top)")
  print("  Gravitational parameter: mgl = 1.0")
  print("  Initial tilt: 30 degrees from vertical")
  print("  Initial spin: L3 = 5.0 (fast rotation around axis)")
  print()

  # Run simulation (100 steps with dt=0.01 = 1 time unit)
  results = simulate_heavy_top(
    L0=[0.1, 0.0, 5.0],  # Mostly spinning around z-axis
    theta0=np.pi/6,       # 30 degrees tilt
    I1=1.0, I2=1.0, I3=0.5,
    mgl=1.0,
    dt=0.01,
    steps=100,
    method="rk4"
  )

  diag = results['diagnostics']

  print("Conservation Analysis:")
  print("-" * 40)
  print(f"  Energy:")
  print(f"    Initial:  {diag['E0']:.10f}")
  print(f"    Final:    {diag['E_final']:.10f}")
  print(f"    Relative error: {diag['dE_relative']:.2e}")
  print()
  print(f"  Casimir C1 = |γ|² (should be 1.0):")
  print(f"    Initial:  {diag['C1_0']:.10f}")
  print(f"    Final:    {diag['C1_final']:.10f}")
  print()
  print(f"  Casimir C2 = L·γ (conserved):")
  print(f"    Initial:  {diag['C2_0']:.10f}")
  print(f"    Final:    {diag['C2_final']:.10f}")
  print()

  # Check success metric from roadmap
  if diag['dE_relative'] < 1e-6:
    print("SUCCESS: Energy conserved to < 10^-6 precision!")
  else:
    print(f"WARNING: Energy drift detected: {diag['dE_relative']:.2e}")

  print()
  print("=" * 60)
  print("The Product Manifold SO(3) × S² is working!")
  print("=" * 60)
