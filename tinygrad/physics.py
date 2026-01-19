"""
TinyPhysics - A Compiler for the Laws of Physics

Core Philosophy:
- Physics = Energy + Geometry
- User defines ONLY the Hamiltonian H(q, p)
- Forces are derived automatically via autograd: F = -dH/dq
- The Poisson structure transforms gradients into equations of motion:
    dq/dt = +dH/dp
    dp/dt = -dH/dq

This is the "native language" of physics on AI hardware.
"""

from tinygrad.tensor import Tensor
from typing import Callable

# ============================================================================
# CORE: HAMILTONIAN MECHANICS VIA AUTOGRAD
# ============================================================================

def _grad_H(q: Tensor, p: Tensor, H_func) -> tuple[Tensor, Tensor]:
    """
    Compute gradients of Hamiltonian using tinygrad autograd.

    This is the ONLY place where forces are computed - derived from energy.

    The Poisson structure:
        dq/dt = +dH/dp  (velocity)
        dp/dt = -dH/dq  (force = negative gradient of energy)
    """
    q_grad = q.detach().requires_grad_(True)
    p_grad = p.detach().requires_grad_(True)

    H = H_func(q_grad, p_grad)
    H.backward()

    dHdq = q_grad.grad.detach() if q_grad.grad is not None else q * 0
    dHdp = p_grad.grad.detach() if p_grad.grad is not None else p * 0

    return dHdq, dHdp


# ============================================================================
# SYMPLECTIC INTEGRATORS
# ============================================================================

def symplectic_euler(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """1st-order Symplectic Euler (Kick-Drift). Order: O(dt)"""
    dHdq, _ = _grad_H(q, p, H_func)
    p_new = p - dt * dHdq
    _, dHdp = _grad_H(q, p_new, H_func)
    q_new = q + dt * dHdp
    return q_new.realize(), p_new.realize()


def leapfrog(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """2nd-order Leapfrog / Störmer-Verlet. Order: O(dt²), Symplectic, Time-reversible"""
    dHdq_1, _ = _grad_H(q, p, H_func)
    p_half = p - (0.5 * dt) * dHdq_1
    _, dHdp = _grad_H(q, p_half, H_func)
    q_new = q + dt * dHdp
    dHdq_2, _ = _grad_H(q_new, p_half, H_func)
    p_new = p_half - (0.5 * dt) * dHdq_2
    return q_new.realize(), p_new.realize()


# Yoshida 4th-order coefficients
_W0 = -2**(1/3) / (2 - 2**(1/3))
_W1 = 1 / (2 - 2**(1/3))


def yoshida4(q: Tensor, p: Tensor, H_func, dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """4th-order Yoshida symplectic integrator. Order: O(dt⁴)"""
    def substep(q, p, h):
        dHdq_1, _ = _grad_H(q, p, H_func)
        p_half = p - (0.5 * h) * dHdq_1
        _, dHdp = _grad_H(q, p_half, H_func)
        q_new = q + h * dHdp
        dHdq_2, _ = _grad_H(q_new, p_half, H_func)
        p_new = p_half - (0.5 * h) * dHdq_2
        return q_new, p_new

    q, p = substep(q, p, _W1 * dt)
    q, p = substep(q, p, _W0 * dt)
    q, p = substep(q, p, _W1 * dt)
    return q.realize(), p.realize()


def implicit_midpoint(q: Tensor, p: Tensor, H_func, dt: float = 0.01,
                      tol: float = 1e-10, max_iter: int = 10) -> tuple[Tensor, Tensor]:
    """Implicit Midpoint Rule - truly symplectic for ANY Hamiltonian. Order: O(dt²)"""
    dHdq, dHdp = _grad_H(q, p, H_func)
    q_next = q + dt * dHdp
    p_next = p - dt * dHdq

    for _ in range(max_iter):
        q_mid = 0.5 * (q + q_next)
        p_mid = 0.5 * (p + p_next)
        dHdq_mid, dHdp_mid = _grad_H(q_mid, p_mid, H_func)
        q_new = q + dt * dHdp_mid
        p_new = p - dt * dHdq_mid
        diff_q = (q_new - q_next).abs().max().numpy()
        diff_p = (p_new - p_next).abs().max().numpy()
        if diff_q < tol and diff_p < tol:
            break
        q_next, p_next = q_new, p_new

    return q_next.realize(), p_next.realize()


# ============================================================================
# HAMILTONIAN SYSTEM - The "Compiler" Interface
# ============================================================================

class HamiltonianSystem:
    """
    A physics system defined purely by its Hamiltonian.

    Example:
        def H(q, p): return 0.5 * (p**2).sum() + 0.5 * (q**2).sum()
        system = HamiltonianSystem(H)
        q, p = system.step(q, p, dt=0.01)
    """
    INTEGRATORS = {
        "euler": symplectic_euler,
        "leapfrog": leapfrog,
        "yoshida4": yoshida4,
        "implicit": implicit_midpoint,
    }

    def __init__(self, H_func, integrator: str = "leapfrog"):
        self.H = H_func
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step = self.INTEGRATORS[integrator]
        self.integrator_name = integrator

    def step(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        return self._step(q, p, self.H, dt)

    def energy(self, q: Tensor, p: Tensor) -> float:
        return float(self.H(q, p).numpy())

    def evolve(self, q: Tensor, p: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        q_history: list[Tensor] = []
        p_history: list[Tensor] = []

        for i in range(steps):
            if i % record_every == 0:
                q_history.append(q.detach())
                p_history.append(p.detach())
            q, p = self.step(q, p, dt)

        q_history.append(q.detach())
        p_history.append(p.detach())

        history = []
        for q_t, p_t in zip(q_history, p_history):
            q_np = q_t.numpy().copy()
            p_np = p_t.numpy().copy()
            e = float(self.H(q_t, p_t).numpy())
            history.append((q_np, p_np, e))

        return q, p, history


# ============================================================================
# FAST INTEGRATORS - Using analytical gradients (no autograd overhead)
# ============================================================================

def fast_leapfrog(q: Tensor, p: Tensor, dHdq_func: Callable, dHdp_func: Callable,
                  dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """Fast leapfrog using pre-computed gradient functions."""
    p_half = p - (0.5 * dt) * dHdq_func(q, p)
    q_new = q + dt * dHdp_func(q, p_half)
    p_new = p_half - (0.5 * dt) * dHdq_func(q_new, p_half)
    return q_new.realize(), p_new.realize()


def fast_yoshida4(q: Tensor, p: Tensor, dHdq_func: Callable, dHdp_func: Callable,
                  dt: float = 0.01) -> tuple[Tensor, Tensor]:
    """Fast Yoshida4 using pre-computed gradient functions."""
    def substep(q, p, h):
        p_half = p - (0.5 * h) * dHdq_func(q, p)
        q_new = q + h * dHdp_func(q, p_half)
        p_new = p_half - (0.5 * h) * dHdq_func(q_new, p_half)
        return q_new, p_new
    q, p = substep(q, p, _W1 * dt)
    q, p = substep(q, p, _W0 * dt)
    q, p = substep(q, p, _W1 * dt)
    return q.realize(), p.realize()


class FastHamiltonianSystem:
    """Fast Hamiltonian system using analytical gradients."""
    INTEGRATORS = {"leapfrog": fast_leapfrog, "yoshida4": fast_yoshida4}

    def __init__(self, H_func: Callable, dHdq_func: Callable, dHdp_func: Callable,
                 integrator: str = "leapfrog"):
        self.H, self.dHdq, self.dHdp = H_func, dHdq_func, dHdp_func
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step_func = self.INTEGRATORS[integrator]
        self.integrator_name = integrator

    def step(self, q: Tensor, p: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        return self._step_func(q, p, self.dHdq, self.dHdp, dt)

    def energy(self, q: Tensor, p: Tensor) -> float:
        return float(self.H(q, p).numpy())


# ============================================================================
# UTILITIES
# ============================================================================

def cross(a: Tensor, b: Tensor) -> Tensor:
    """Cross product of 3D vectors. Shape: (..., 3)."""
    return Tensor.stack([
        a[..., 1]*b[..., 2] - a[..., 2]*b[..., 1],
        a[..., 2]*b[..., 0] - a[..., 0]*b[..., 2],
        a[..., 0]*b[..., 1] - a[..., 1]*b[..., 0]
    ], dim=-1)


# ============================================================================
# LIE-POISSON MECHANICS (Phase 2: Rigid Body Dynamics)
# ============================================================================
#
# For systems on Lie algebras (like rigid body rotation), the Poisson structure
# is state-dependent: J = J(z). The equation of motion is:
#
#     dz/dt = J(z) · ∇H(z)
#
# For so(3) (angular momentum): J(L) acts via cross product
#     dL/dt = L × ∇H(L)
#
# This is Euler's equation when H = 0.5 * L · (I⁻¹ L)

def _grad_LP(z: Tensor, H_func) -> Tensor:
    """Compute gradient of Hamiltonian for Lie-Poisson systems."""
    z_grad = z.detach().requires_grad_(True)
    H = H_func(z_grad)
    H.backward()
    return z_grad.grad.detach() if z_grad.grad is not None else z * 0


def lie_poisson_euler_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """1st-order Lie-Poisson Euler for so(3): dL/dt = L × ∇H(L)"""
    dHdL = _grad_LP(L, H_func)
    return (L + dt * cross(L, dHdL)).realize()


def lie_poisson_midpoint_so3(L: Tensor, H_func, dt: float = 0.01,
                              tol: float = 1e-10, max_iter: int = 10) -> Tensor:
    """Implicit Midpoint for so(3). Preserves Casimirs (|L|²) to machine precision."""
    dHdL = _grad_LP(L, H_func)
    L_next = L + dt * cross(L, dHdL)

    for _ in range(max_iter):
        L_mid = 0.5 * (L + L_next)
        dHdL_mid = _grad_LP(L_mid, H_func)
        L_new = L + dt * cross(L_mid, dHdL_mid)
        if (L_new - L_next).abs().max().numpy() < tol:
            break
        L_next = L_new

    return L_next.realize()


def lie_poisson_rk4_so3(L: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """4th-order RK4 for so(3). Good accuracy, does NOT preserve Casimirs exactly."""
    def f(L_val):
        return cross(L_val, _grad_LP(L_val, H_func))

    k1 = f(L)
    k2 = f(L + 0.5 * dt * k1)
    k3 = f(L + 0.5 * dt * k2)
    k4 = f(L + dt * k3)
    return (L + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).realize()


def lie_poisson_splitting_so3(L: Tensor, I_inv: Tensor, dt: float = 0.01) -> Tensor:
    """
    Explicit splitting method for free rigid body.
    Preserves |L|² exactly (Casimir). 2nd order via Strang splitting.
    """
    def rotate_axis(L_val: Tensor, axis: int, angle: Tensor) -> Tensor:
        j, k = (axis + 1) % 3, (axis + 2) % 3
        c, s = angle.cos(), angle.sin()
        L_j_new = c * L_val[j] - s * L_val[k]
        L_k_new = s * L_val[j] + c * L_val[k]
        if axis == 0: return Tensor.stack([L_val[0], L_j_new, L_k_new])
        if axis == 1: return Tensor.stack([L_k_new, L_val[1], L_j_new])
        return Tensor.stack([L_j_new, L_k_new, L_val[2]])

    # Strang splitting: half-step forward, half-step reverse
    for axis in range(3):
        L = rotate_axis(L, axis, 0.5 * dt * L[axis] * I_inv[axis])
    for axis in [2, 1, 0]:
        L = rotate_axis(L, axis, 0.5 * dt * L[axis] * I_inv[axis])
    return L.realize()


class LiePoissonSystem:
    """
    Physics on a Lie algebra, defined by its Hamiltonian.

    For the free rigid body: H(L) = 0.5 * sum(L² / I)

    Example:
        I_inv = Tensor([1.0, 0.5, 0.333])
        def H(L): return 0.5 * (L * L * I_inv).sum()
        system = LiePoissonSystem(H, algebra="so3")
        L = system.step(L, dt=0.01)
    """
    INTEGRATORS = {
        "euler": lie_poisson_euler_so3,
        "midpoint": lie_poisson_midpoint_so3,
        "rk4": lie_poisson_rk4_so3,
    }

    def __init__(self, H_func, algebra: str = "so3", integrator: str = "midpoint"):
        if algebra != "so3":
            raise ValueError(f"Only so3 supported, got: {algebra}")
        self.H = H_func
        self.algebra = algebra
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step = self.INTEGRATORS[integrator]
        self.integrator_name = integrator

    def step(self, z: Tensor, dt: float = 0.01) -> Tensor:
        return self._step(z, self.H, dt)

    def energy(self, z: Tensor) -> float:
        return float(self.H(z).numpy())

    def casimir(self, z: Tensor) -> float:
        return float((z * z).sum().numpy())

    def evolve(self, z: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, list]:
        z_history: list[Tensor] = []
        for i in range(steps):
            if i % record_every == 0:
                z_history.append(z.detach())
            z = self.step(z, dt)
        z_history.append(z.detach())
        history = [(z_t.numpy().copy(), self.energy(z_t), self.casimir(z_t)) for z_t in z_history]
        return z, history


# ============================================================================
# RIGID BODY UTILITIES
# ============================================================================

def quaternion_multiply(q1: Tensor, q2: Tensor) -> Tensor:
    """Quaternion multiplication q1 * q2. Format: [w, x, y, z]."""
    w1, x1, y1, z1 = q1[0], q1[1], q1[2], q1[3]
    w2, x2, y2, z2 = q2[0], q2[1], q2[2], q2[3]
    return Tensor.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_normalize(q: Tensor) -> Tensor:
    """Normalize quaternion to unit length."""
    return q / (q * q).sum().sqrt()


def integrate_orientation(quat: Tensor, omega: Tensor, dt: float) -> Tensor:
    """Integrate orientation quaternion: dq/dt = 0.5 * q ⊗ [0, ω]"""
    omega_quat = Tensor.stack([Tensor(0.0), omega[0], omega[1], omega[2]])
    dq = quaternion_multiply(quat, omega_quat) * 0.5
    return quaternion_normalize(quat + dt * dq).realize()


class RigidBodySystem:
    """
    Complete rigid body simulation: Lie-Poisson dynamics (L) + quaternion kinematics (q).

    Example (Tennis Racket Theorem):
        system = RigidBodySystem(Tensor([1.0, 2.0, 3.0]), integrator="midpoint")
        L = Tensor([0.01, 2.0, 0.01])  # Near intermediate axis
        q = Tensor([1.0, 0.0, 0.0, 0.0])  # Identity
        for _ in range(1000): L, q = system.step(L, q, dt=0.01)
    """
    def __init__(self, I: Tensor, integrator: str = "midpoint"):
        self.I = I
        self.I_inv = 1.0 / I
        self.H = lambda L: 0.5 * (L * L * self.I_inv).sum()
        self.integrator_name = integrator
        if integrator == "splitting":
            self._step_L = lambda L, dt: lie_poisson_splitting_so3(L, self.I_inv, dt)
        elif integrator in LiePoissonSystem.INTEGRATORS:
            self._lp = LiePoissonSystem(self.H, algebra="so3", integrator=integrator)
            self._step_L = lambda L, dt: self._lp.step(L, dt)
        else:
            raise ValueError(f"Unknown integrator: {integrator}")

    def step(self, L: Tensor, q: Tensor, dt: float = 0.01) -> tuple[Tensor, Tensor]:
        L_new = self._step_L(L, dt)
        return L_new, integrate_orientation(q, L_new * self.I_inv, dt)

    def energy(self, L: Tensor) -> float:
        return float(self.H(L).numpy())

    def casimir(self, L: Tensor) -> float:
        return float((L * L).sum().numpy())

    def evolve(self, L: Tensor, q: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        history = []
        for i in range(steps):
            if i % record_every == 0:
                history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
            L, q = self.step(L, q, dt)
        history.append((L.numpy().copy(), q.numpy().copy(), self.energy(L), self.casimir(L)))
        return L, q, history


# ============================================================================
# POINT VORTEX DYNAMICS (Phase 3.1: Fluid Mechanics)
# ============================================================================
#
# Point vortices in 2D form a Hamiltonian system with non-standard symplectic
# structure. Each vortex has position (x_i, y_i) and circulation Γ_i.
#
# Hamiltonian: H = -1/(4π) Σ_{i<j} Γ_i Γ_j log|r_ij|
#
# The equations of motion (Kirchhoff):
#     Γ_i dx_i/dt = +∂H/∂y_i
#     Γ_i dy_i/dt = -∂H/∂x_i
#
# This is STILL autograd-friendly! We compute ∂H/∂z and apply the
# circulation-weighted Poisson structure.

def _grad_vortex(z: Tensor, H_func) -> Tensor:
    """Compute gradient of vortex Hamiltonian."""
    z_grad = z.detach().requires_grad_(True)
    H = H_func(z_grad)
    H.backward()
    return z_grad.grad.detach() if z_grad.grad is not None else z * 0


def point_vortex_hamiltonian(gamma: Tensor, softening: float = 1e-6):
    """
    Returns the Hamiltonian for N point vortices.

    H = -1/(4π) Σ_{i<j} Γ_i Γ_j log|r_ij|

    Args:
        gamma: Tensor of shape (N,) containing circulations
        softening: Small value to avoid log(0) singularity

    The gradient ∂H/∂z is computed via autograd.
    """
    import math
    n = gamma.shape[0]

    def H(z):
        # z has shape (2*N,) = [x0, y0, x1, y1, ...]
        x = z.reshape(n, 2)[:, 0]  # shape (N,)
        y = z.reshape(n, 2)[:, 1]  # shape (N,)

        # Compute pairwise distances: r_ij = sqrt((xi-xj)² + (yi-yj)²)
        dx = x.unsqueeze(1) - x.unsqueeze(0)  # (N, N)
        dy = y.unsqueeze(1) - y.unsqueeze(0)  # (N, N)
        r_sq = dx * dx + dy * dy + softening * softening  # avoid log(0)

        # Γ_i Γ_j matrix
        gamma_ij = gamma.unsqueeze(1) * gamma.unsqueeze(0)  # (N, N)

        # H = -1/(4π) Σ_{i<j} Γ_i Γ_j log(r_ij)
        # Use upper triangle (i < j) by masking diagonal and lower
        log_r = (r_sq.sqrt() + 1e-10).log()
        H_matrix = -gamma_ij * log_r / (4 * math.pi)

        # Sum upper triangle only (avoid double counting)
        # Mask: 1 for i < j, 0 otherwise
        mask_data = []
        for i in range(n):
            row = [1.0 if j > i else 0.0 for j in range(n)]
            mask_data.extend(row)
        mask = Tensor(mask_data).reshape(n, n)

        return (H_matrix * mask).sum()

    return H


def vortex_euler(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """
    1st-order Euler for point vortices.

    Equations: Γ_i dz_i/dt = J · ∇H where J = [[0, 1], [-1, 0]]
    """
    dHdz = _grad_vortex(z, H_func)
    n = gamma.shape[0]

    # Build the circulation-weighted Poisson update
    # dx_i/dt = (1/Γ_i) ∂H/∂y_i
    # dy_i/dt = -(1/Γ_i) ∂H/∂x_i
    dHdz_reshaped = dHdz.reshape(n, 2)  # (N, 2) = [[dH/dx0, dH/dy0], ...]
    dHdx = dHdz_reshaped[:, 0]
    dHdy = dHdz_reshaped[:, 1]

    # Velocity: v_i = (1/Γ_i) * J · ∇_i H
    vx = dHdy / gamma   # dx/dt = (1/Γ) ∂H/∂y
    vy = -dHdx / gamma  # dy/dt = -(1/Γ) ∂H/∂x

    v = Tensor.stack([vx, vy], dim=1).reshape(-1)  # flatten to (2N,)
    return (z + dt * v).realize()


def vortex_rk4(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01) -> Tensor:
    """4th-order Runge-Kutta for point vortices."""
    def velocity(z_val):
        dHdz = _grad_vortex(z_val, H_func)
        n = gamma.shape[0]
        dHdz_reshaped = dHdz.reshape(n, 2)
        dHdx = dHdz_reshaped[:, 0]
        dHdy = dHdz_reshaped[:, 1]
        vx = dHdy / gamma
        vy = -dHdx / gamma
        return Tensor.stack([vx, vy], dim=1).reshape(-1)

    k1 = velocity(z)
    k2 = velocity(z + 0.5 * dt * k1)
    k3 = velocity(z + 0.5 * dt * k2)
    k4 = velocity(z + dt * k3)
    return (z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)).realize()


def vortex_midpoint(z: Tensor, gamma: Tensor, H_func, dt: float = 0.01,
                    tol: float = 1e-10, max_iter: int = 10) -> Tensor:
    """Implicit midpoint for point vortices. Preserves Hamiltonian better."""
    def velocity(z_val):
        dHdz = _grad_vortex(z_val, H_func)
        n = gamma.shape[0]
        dHdz_reshaped = dHdz.reshape(n, 2)
        dHdx = dHdz_reshaped[:, 0]
        dHdy = dHdz_reshaped[:, 1]
        vx = dHdy / gamma
        vy = -dHdx / gamma
        return Tensor.stack([vx, vy], dim=1).reshape(-1)

    # Initial guess
    z_next = z + dt * velocity(z)

    for _ in range(max_iter):
        z_mid = 0.5 * (z + z_next)
        z_new = z + dt * velocity(z_mid)
        if (z_new - z_next).abs().max().numpy() < tol:
            break
        z_next = z_new

    return z_next.realize()


class PointVortexSystem:
    """
    Point vortex dynamics in 2D - Kirchhoff's equations.

    This is the "Hello World" of fluid mechanics: discrete vortices
    interacting via the Biot-Savart law.

    The Hamiltonian is defined automatically from circulations.
    Autograd computes all the interaction forces!

    Example:
        gamma = Tensor([1.0, 1.0, -1.0])  # 3 vortices
        z = Tensor([0.0, 1.0,   # vortex 0 at (0, 1)
                    1.0, 0.0,   # vortex 1 at (1, 0)
                    -1.0, 0.0]) # vortex 2 at (-1, 0)
        system = PointVortexSystem(gamma)
        z = system.step(z, dt=0.01)
    """
    INTEGRATORS = {
        "euler": vortex_euler,
        "rk4": vortex_rk4,
        "midpoint": vortex_midpoint,
    }

    def __init__(self, gamma: Tensor, integrator: str = "rk4", softening: float = 1e-6):
        self.gamma = gamma
        self.n_vortices = gamma.shape[0]
        self.H = point_vortex_hamiltonian(gamma, softening)
        if integrator not in self.INTEGRATORS:
            raise ValueError(f"Unknown integrator: {integrator}")
        self._step_func = self.INTEGRATORS[integrator]
        self.integrator_name = integrator

    def step(self, z: Tensor, dt: float = 0.01) -> Tensor:
        return self._step_func(z, self.gamma, self.H, dt)

    def energy(self, z: Tensor) -> float:
        return float(self.H(z).numpy())

    def momentum(self, z: Tensor) -> tuple[float, float]:
        """Linear impulse: P = Σ Γ_i r_i (conserved)."""
        n = self.gamma.shape[0]
        pos = z.reshape(n, 2)
        px = (self.gamma * pos[:, 0]).sum().numpy()
        py = (self.gamma * pos[:, 1]).sum().numpy()
        return float(px), float(py)

    def angular_momentum(self, z: Tensor) -> float:
        """Angular impulse: L = Σ Γ_i |r_i|² (conserved)."""
        n = self.gamma.shape[0]
        pos = z.reshape(n, 2)
        r_sq = (pos * pos).sum(axis=1)
        return float((self.gamma * r_sq).sum().numpy())

    def evolve(self, z: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, list]:
        z_history: list[Tensor] = []
        for i in range(steps):
            if i % record_every == 0:
                z_history.append(z.detach())
            z = self.step(z, dt)
        z_history.append(z.detach())

        history = []
        for z_t in z_history:
            z_np = z_t.numpy().copy()
            e = self.energy(z_t)
            px, py = self.momentum(z_t)
            L = self.angular_momentum(z_t)
            history.append((z_np, e, px, py, L))

        return z, history


# ============================================================================
# QUANTUM MECHANICS (Phase 4: Wavefunction Dynamics)
# ============================================================================
#
# Quantum mechanics uses complex wavefunctions ψ(x) evolving via Schrödinger:
#
#     iℏ ∂ψ/∂t = Ĥψ
#
# For a free particle: Ĥ = p̂²/2m = -ℏ²∇²/2m
#
# Solution: ψ(t) = e^(-iĤt/ℏ)ψ(0)
#
# In Fourier space, this becomes diagonal:
#     ψ̃(k,t) = e^(-iℏk²t/2m) ψ̃(k,0)
#
# This is the "split-operator" method - efficient and unitary!

import numpy as np


def fft(x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
    """
    1D FFT of complex tensor (real, imag) -> (real, imag).
    Uses numpy FFT internally.
    """
    c = x_real.numpy() + 1j * x_imag.numpy()
    ft = np.fft.fft(c)
    return Tensor(ft.real.copy()), Tensor(ft.imag.copy())


def ifft(x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
    """
    1D inverse FFT of complex tensor.
    """
    c = x_real.numpy() + 1j * x_imag.numpy()
    out = np.fft.ifft(c)
    return Tensor(out.real.copy()), Tensor(out.imag.copy())


def complex_mul(a_real: Tensor, a_imag: Tensor,
                b_real: Tensor, b_imag: Tensor) -> tuple[Tensor, Tensor]:
    """Complex multiplication: (a_r + i*a_i) * (b_r + i*b_i)"""
    real = a_real * b_real - a_imag * b_imag
    imag = a_real * b_imag + a_imag * b_real
    return real, imag


def complex_exp(phase: Tensor) -> tuple[Tensor, Tensor]:
    """e^(i*phase) = cos(phase) + i*sin(phase)"""
    return phase.cos(), phase.sin()


def wavefunction_norm(psi_real: Tensor, psi_imag: Tensor, dx: float) -> float:
    """Compute ∫|ψ|² dx (should equal 1 for normalized wavefunction)."""
    prob = psi_real * psi_real + psi_imag * psi_imag
    return float((prob.sum() * dx).numpy())


def normalize_wavefunction(psi_real: Tensor, psi_imag: Tensor,
                           dx: float) -> tuple[Tensor, Tensor]:
    """Normalize wavefunction so ∫|ψ|² dx = 1."""
    norm = np.sqrt(wavefunction_norm(psi_real, psi_imag, dx))
    return psi_real / norm, psi_imag / norm


def free_particle_propagator(k: Tensor, dt: float, hbar: float = 1.0,
                             m: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Free particle propagator in momentum space: e^(-i*ℏk²*dt/2m)

    This is the core of quantum evolution for a free particle.
    """
    phase = -hbar * k * k * dt / (2 * m)
    return complex_exp(phase)


def potential_propagator(V: Tensor, dt: float,
                         hbar: float = 1.0) -> tuple[Tensor, Tensor]:
    """
    Potential propagator in position space: e^(-i*V*dt/ℏ)

    For split-operator method with potentials.
    """
    phase = -V * dt / hbar
    return complex_exp(phase)


class QuantumSystem:
    """
    Quantum system for 1D wavefunction evolution.

    Uses split-operator method with FFT for efficient unitary evolution.

    The Schrödinger equation iℏ∂ψ/∂t = Ĥψ is solved via:
        ψ(t+dt) = e^(-iV̂dt/2ℏ) · FFT⁻¹[e^(-iT̂dt/ℏ) · FFT[e^(-iV̂dt/2ℏ)ψ(t)]]

    For free particle (V=0), this simplifies to momentum-space evolution.

    Example:
        system = QuantumSystem(N=512, L=20.0, m=1.0, hbar=1.0)
        psi_r, psi_i = system.gaussian_wavepacket(x0=0, sigma=1, k0=2)
        psi_r, psi_i = system.step(psi_r, psi_i, dt=0.01)
    """

    def __init__(self, N: int, L: float, m: float = 1.0, hbar: float = 1.0,
                 V: Tensor | None = None):
        """
        Initialize quantum system.

        Args:
            N: Number of grid points
            L: Domain size [-L/2, L/2]
            m: Particle mass
            hbar: Reduced Planck constant
            V: Potential energy V(x) as Tensor, or None for free particle
        """
        self.N = N
        self.L = L
        self.m = m
        self.hbar = hbar
        self.dx = L / N

        # Position grid
        self.x = Tensor(np.linspace(-L/2, L/2, N, endpoint=False))

        # Momentum grid (FFT frequencies)
        k_arr = np.fft.fftfreq(N, d=self.dx) * 2 * np.pi
        self.k = Tensor(k_arr)

        # Potential
        self.V = V

    def gaussian_wavepacket(self, x0: float = 0.0, sigma: float = 1.0,
                            k0: float = 0.0) -> tuple[Tensor, Tensor]:
        """
        Create a Gaussian wavepacket: ψ(x) = (2πσ²)^(-1/4) exp(-(x-x0)²/4σ² + ik0*x)

        Args:
            x0: Initial position
            sigma: Width of wavepacket
            k0: Initial momentum (wavenumber)

        Returns:
            (psi_real, psi_imag): Complex wavefunction as two real tensors
        """
        x_np = self.x.numpy()

        # Gaussian envelope
        norm = (2 * np.pi * sigma**2) ** (-0.25)
        envelope = norm * np.exp(-(x_np - x0)**2 / (4 * sigma**2))

        # Phase factor e^(ik0*x)
        phase = k0 * x_np

        psi_real = Tensor(envelope * np.cos(phase))
        psi_imag = Tensor(envelope * np.sin(phase))

        return psi_real, psi_imag

    def step(self, psi_real: Tensor, psi_imag: Tensor,
             dt: float) -> tuple[Tensor, Tensor]:
        """
        Evolve wavefunction by one time step using split-operator method.

        For free particle: ψ(t+dt) = FFT⁻¹[e^(-iℏk²dt/2m) · FFT[ψ(t)]]
        With potential: uses Strang splitting for 2nd order accuracy.
        """
        if self.V is None:
            # Free particle - pure momentum space evolution
            # FFT to momentum space
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)

            # Apply kinetic propagator
            prop_r, prop_i = free_particle_propagator(self.k, dt, self.hbar, self.m)
            psi_k_r, psi_k_i = complex_mul(psi_k_r, psi_k_i, prop_r, prop_i)

            # IFFT back to position space
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)
        else:
            # Strang splitting: V/2 -> T -> V/2
            # Half step in potential
            prop_r, prop_i = potential_propagator(self.V, dt/2, self.hbar)
            psi_real, psi_imag = complex_mul(psi_real, psi_imag, prop_r, prop_i)

            # Full step in kinetic (momentum space)
            psi_k_r, psi_k_i = fft(psi_real, psi_imag)
            prop_r, prop_i = free_particle_propagator(self.k, dt, self.hbar, self.m)
            psi_k_r, psi_k_i = complex_mul(psi_k_r, psi_k_i, prop_r, prop_i)
            psi_real, psi_imag = ifft(psi_k_r, psi_k_i)

            # Half step in potential
            prop_r, prop_i = potential_propagator(self.V, dt/2, self.hbar)
            psi_real, psi_imag = complex_mul(psi_real, psi_imag, prop_r, prop_i)

        return psi_real.realize(), psi_imag.realize()

    def probability_density(self, psi_real: Tensor, psi_imag: Tensor) -> Tensor:
        """Compute |ψ|² - probability density."""
        return psi_real * psi_real + psi_imag * psi_imag

    def expectation_x(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute <x> = ∫ψ*xψ dx"""
        prob = self.probability_density(psi_real, psi_imag)
        return float((prob * self.x).sum().numpy() * self.dx)

    def expectation_x2(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute <x²> = ∫ψ*x²ψ dx"""
        prob = self.probability_density(psi_real, psi_imag)
        return float((prob * self.x * self.x).sum().numpy() * self.dx)

    def width(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute wavepacket width: σ = √(<x²> - <x>²)"""
        x_mean = self.expectation_x(psi_real, psi_imag)
        x2_mean = self.expectation_x2(psi_real, psi_imag)
        var = x2_mean - x_mean**2
        return np.sqrt(max(var, 0))

    def energy(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """
        Compute total energy <H> = <T> + <V>.

        <T> is computed in momentum space for accuracy.
        """
        # Kinetic energy in momentum space
        # FFT normalization: |ψ̃|² needs to be divided by N² and multiplied by L
        psi_k_r, psi_k_i = fft(psi_real, psi_imag)
        prob_k = psi_k_r * psi_k_r + psi_k_i * psi_k_i
        # Normalize: dk = 2π/L, and FFT gives ψ̃ = Δx * Σ ψ * exp(...) so |ψ̃|² ~ Δx² * N
        # Correct normalization factor for <k²>
        norm_factor = self.dx * self.dx / self.L
        T = float((prob_k * self.hbar**2 * self.k * self.k / (2 * self.m)).sum().numpy() * norm_factor)

        # Potential energy
        if self.V is not None:
            prob_x = self.probability_density(psi_real, psi_imag)
            V_exp = float((prob_x * self.V).sum().numpy() * self.dx)
        else:
            V_exp = 0.0

        return T + V_exp

    def norm(self, psi_real: Tensor, psi_imag: Tensor) -> float:
        """Compute ∫|ψ|² dx (should be 1 for normalized wavefunction)."""
        return wavefunction_norm(psi_real, psi_imag, self.dx)

    def evolve(self, psi_real: Tensor, psi_imag: Tensor, dt: float, steps: int,
               record_every: int = 1) -> tuple[Tensor, Tensor, list]:
        """
        Evolve wavefunction for multiple steps.

        Returns:
            (psi_real, psi_imag, history) where history contains
            (x_array, prob_array, norm, width, x_mean, energy) for each recorded step.
        """
        history = []
        x_np = self.x.numpy()

        for i in range(steps):
            if i % record_every == 0:
                prob = self.probability_density(psi_real, psi_imag).numpy()
                n = self.norm(psi_real, psi_imag)
                w = self.width(psi_real, psi_imag)
                x_mean = self.expectation_x(psi_real, psi_imag)
                e = self.energy(psi_real, psi_imag)
                history.append((x_np.copy(), prob.copy(), n, w, x_mean, e))
            psi_real, psi_imag = self.step(psi_real, psi_imag, dt)

        # Record final state
        prob = self.probability_density(psi_real, psi_imag).numpy()
        n = self.norm(psi_real, psi_imag)
        w = self.width(psi_real, psi_imag)
        x_mean = self.expectation_x(psi_real, psi_imag)
        e = self.energy(psi_real, psi_imag)
        history.append((x_np.copy(), prob.copy(), n, w, x_mean, e))

        return psi_real, psi_imag, history


# Backward compatibility
def symplectic_step(q, p, force, dt=0.01, mass=1.0):
    """DEPRECATED: Use HamiltonianSystem."""
    return q + (p + force * dt) / mass * dt, p + force * dt

hamiltonian_step = leapfrog
hamiltonian_yoshida4 = yoshida4
