import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tinyphysics.bench.universal_physics_bench import (
  bench_canonical, bench_so3, bench_quantum, bench_constraint, bench_dissipative, bench_fluid, bench_thermostat,
  bench_barostat, bench_lj_tensor_bins, bench_lj_barostat
)


def main():
  if os.getenv("CI", "") and os.getenv("TINYGRAD_BENCH_FAST", "") == "":
    os.environ["TINYGRAD_BENCH_FAST"] = "3"
  fast_level = int(os.getenv("TINYGRAD_BENCH_FAST", "0") or "0")
  fast = fast_level >= 1
  superfast = fast_level >= 2
  ultrafast = fast_level >= 3

  t_can = bench_canonical(steps=8 if ultrafast else 10 if superfast else 25 if fast else 100,
                          n=64 if ultrafast else 128 if superfast else 256 if fast else 1024)
  t_lp = bench_so3(steps=8 if ultrafast else 10 if superfast else 25 if fast else 100)
  t_q = bench_quantum(steps=2 if ultrafast else 5 if superfast else 10 if fast else 50,
                      n=16 if ultrafast else 32 if superfast else 64 if fast else 128)
  t_c = bench_constraint(steps=8 if ultrafast else 10 if superfast else 25 if fast else 100)
  t_d = bench_dissipative(steps=16 if ultrafast else 20 if superfast else 50 if fast else 200)
  t_f = bench_fluid(steps=1 if ultrafast else 2 if superfast else 5 if fast else 20,
                    n=8 if ultrafast else 16 if superfast else 32 if fast else 64)
  t_th = bench_thermostat(steps=16 if ultrafast else 20 if superfast else 50 if fast else 200) if os.getenv("TINYGRAD_BENCH_THERMOSTAT", "0") else None
  t_baro = bench_barostat(steps=8 if ultrafast else 10 if superfast else 25 if fast else 100) if os.getenv("TINYGRAD_BENCH_BAROSTAT", "0") else None
  t_lj = bench_lj_tensor_bins(steps=1 if ultrafast else 1 if superfast else 3 if fast else 10,
                              n=32 if ultrafast else 64 if superfast else 128 if fast else 512) if os.getenv("TINYGRAD_BENCH_LJ", "0") else None
  t_lj_baro = bench_lj_barostat(steps=1 if ultrafast else 1 if superfast else 3 if fast else 10,
                                n=16 if ultrafast else 32 if superfast else 64 if fast else 256) if os.getenv("TINYGRAD_BENCH_LJ_BAROSTAT", "0") else None
  print(f"canonical: {t_can:.4f}s")
  print(f"so3: {t_lp:.4f}s")
  print(f"quantum: {t_q:.4f}s")
  print(f"constraint: {t_c:.4f}s")
  print(f"dissipative: {t_d:.4f}s")
  print(f"fluid: {t_f:.4f}s")
  if t_th is not None:
    print(f"thermostat: {t_th:.4f}s")
  if t_baro is not None:
    print(f"barostat: {t_baro:.4f}s")
  if t_lj is not None:
    print(f"lj: {t_lj:.4f}s")
  if t_lj_baro is not None:
    print(f"lj_barostat: {t_lj_baro:.4f}s")
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
