import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tinyphysics.bench.universal_physics_bench import (
  bench_canonical, bench_so3, bench_quantum, bench_constraint, bench_dissipative, bench_fluid, bench_thermostat
)


def main():
  t_can = bench_canonical()
  t_lp = bench_so3()
  t_q = bench_quantum()
  t_c = bench_constraint()
  t_d = bench_dissipative()
  t_f = bench_fluid()
  t_th = bench_thermostat() if os.getenv("TINYGRAD_BENCH_THERMOSTAT", "0") else None
  print(f"canonical: {t_can:.4f}s")
  print(f"so3: {t_lp:.4f}s")
  print(f"quantum: {t_q:.4f}s")
  print(f"constraint: {t_c:.4f}s")
  print(f"dissipative: {t_d:.4f}s")
  print(f"fluid: {t_f:.4f}s")
  if t_th is not None:
    print(f"thermostat: {t_th:.4f}s")
  if os.getenv("TINYGRAD_BENCH_NEIGHBORS", "0"):
    from tinyphysics.bench.neighbors_bench import bench_neighbors
    t_n, count = bench_neighbors()
    print(f"neighbors: {t_n:.4f}s, pairs={count}")
  if os.getenv("TINYGRAD_BENCH_NBODY", "0"):
    from tinyphysics.bench.nbody_bench import bench_nbody, bench_nbody_compare
    method = os.getenv("TINYGRAD_BENCH_NBODY_METHOD", "neighbor")
    if os.getenv("TINYGRAD_BENCH_NBODY_COMPARE", "0"):
      timings, pairs, max_per_auto = bench_nbody_compare()
      print(f"nbody_pairs: {pairs}, auto_max_per: {max_per_auto}")
      for method_name, (t_n, cap) in timings.items():
        suffix = "" if cap is None else f", max_per={cap}"
        print(f"nbody({method_name}): {t_n:.4f}s{suffix}")
    else:
      t_n = bench_nbody(method=method)
      print(f"nbody({method}): {t_n:.4f}s")


if __name__ == "__main__":
  main()
