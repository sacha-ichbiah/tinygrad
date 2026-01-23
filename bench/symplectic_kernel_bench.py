import time
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem, compile_symplectic_program


def main():
  def H(q, p):
    return (p * p).sum() / 2 + (q * q).sum() / 2

  q = Tensor([1.0, 0.5, -0.25, 0.2, -0.1, 0.3])
  p = Tensor([0.0, 0.1, -0.2, 0.05, 0.2, -0.15])
  dt = 0.01
  steps = 5000

  sys = HamiltonianSystem(H, integrator="leapfrog")
  prog = compile_symplectic_program("canonical", H=H, integrator="leapfrog", sample_state=(q, p))

  start = time.perf_counter()
  q1, p1, _ = sys.evolve(q, p, dt=dt, steps=steps, record_every=steps)
  t_sys = time.perf_counter() - start

  start = time.perf_counter()
  (q2, p2), _ = prog.evolve((q, p), dt=dt, steps=steps, record_every=steps, unroll=50)
  t_prog = time.perf_counter() - start

  print("symplectic kernel bench")
  print(f"steps={steps} dt={dt} ops={[op.name for op in prog.ops]}")
  print(f"system={t_sys:.4f}s compiler={t_prog:.4f}s")
  _ = (q1 + p1 + q2 + p2).sum().numpy()


if __name__ == "__main__":
  main()
