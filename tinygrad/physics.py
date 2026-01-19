from tinygrad.tensor import Tensor

def symplectic_step(q: Tensor, p: Tensor, force: Tensor, dt=0.01, mass=1.0):
    """
    Computes the next state (q_new, p_new) using semi-implicit Euler.
    
    1. p_{t+1} = p_t + F(q_t) * dt   (Kick)
    2. q_{t+1} = q_t + p_{t+1}/m * dt  (Drift)
    
    This function uses standard Tinygrad ops, so autograd works automatically.
    """
    # 1. Update Momentum (Kick)
    # p_new = p + F * dt
    p_new = p + force * dt
    
    # 2. Update Position (Drift)
    # q_new = q + (p_new / m) * dt
    q_new = q + (p_new / mass) * dt
    
    return q_new, p_new

def cross(a: Tensor, b: Tensor) -> Tensor:
    """
    Computes cross product of two 3D vectors a and b.
    Inputs must be shape (..., 3).
    """
    a1, a2, a3 = a[..., 0], a[..., 1], a[..., 2]
    b1, b2, b3 = b[..., 0], b[..., 1], b[..., 2]
    
    s1 = a2*b3 - a3*b2
    s2 = a3*b1 - a1*b3
    s3 = a1*b2 - a2*b1
    
    return Tensor.stack([s1, s2, s3], dim=-1)
