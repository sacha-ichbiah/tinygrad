import time
from tinygrad.tensor import Tensor
from tinygrad.physics import compile_symplectic_program


def main():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, 0.5, -0.25])
  p = Tensor([0.0, 0.1, -0.2])
  dt = 0.01
  steps = 5000

  prog = compile_symplectic_program("canonical", H=H, integrator="auto", sample_state=(q, p))
  start = time.perf_counter()
  (q_out, p_out), history = prog.evolve((q, p), dt=dt, steps=steps, record_every=100, unroll=50)
  elapsed = time.perf_counter() - start

  e0 = float(H(history[0][0], history[0][1]).numpy())
  e1 = float(H(q_out, p_out).numpy())
  drift = abs(e1 - e0) / max(abs(e0), 1e-12)

  print("Symplectic compiler demo")
  print(f"steps={steps} dt={dt} ops={[op.name for op in prog.ops]}")
  print(f"time={elapsed:.4f}s drift={drift:.3e}")


if __name__ == "__main__":
  main()
