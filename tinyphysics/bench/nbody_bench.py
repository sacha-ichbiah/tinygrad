import os
import sys
import time
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
  sys.path.insert(0, ROOT)

from tinygrad.tensor import Tensor
from tinyphysics.systems.nbody import NBodySystem
from tinyphysics.operators.neighbor import neighbor_pairs
from tinyphysics.operators.tensor_neighbors import autotune_max_per


def _parse_max_per(value: str | None):
  if value is None:
    return None
  if value.lower() == "auto":
    return "auto"
  return int(value)


def bench_nbody(n: int = 256, steps: int = 10, method: str = "neighbor",
                q: Tensor | None = None, p: Tensor | None = None, m: Tensor | None = None,
                max_per: int | None | str = None):
  if q is None or p is None or m is None:
    rng = np.random.default_rng(0)
    q = Tensor(rng.random((n, 3)).astype(np.float32) * 10.0)
    p = Tensor(rng.random((n, 3)).astype(np.float32))
    m = Tensor(np.ones((n,), dtype=np.float32))
  system = NBodySystem(mass=m, r_cut=1.0, box=10.0, method=method, max_per=max_per)
  prog = system.compile(q, p)
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0


def bench_nbody_auto(n: int = 256, steps: int = 10):
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((n, 3)).astype(np.float32) * 10.0)
  p = Tensor(rng.random((n, 3)).astype(np.float32))
  m = Tensor(np.ones((n,), dtype=np.float32))
  system = NBodySystem(mass=m, r_cut=1.0, box=10.0, method="auto")
  prog = system.compile(q, p)
  t0 = time.time()
  (q, p), _ = prog.evolve((q, p), 0.01, steps)
  _ = q.realize(); _ = p.realize()
  return time.time() - t0, system.last_method


def bench_nbody_compare(n: int = 256, steps: int = 10):
  rng = np.random.default_rng(0)
  q = Tensor(rng.random((n, 3)).astype(np.float32) * 10.0)
  p = Tensor(rng.random((n, 3)).astype(np.float32))
  m = Tensor(np.ones((n,), dtype=np.float32))
  q_np = q.numpy()
  pair_count = len(neighbor_pairs(q_np, box=10.0, r_cut=1.0))
  max_per_env = _parse_max_per(os.getenv("TINYGRAD_NBODY_MAX_PER"))
  max_per_auto = autotune_max_per(q, box=10.0, r_cut=1.0)
  max_per = max_per_env if max_per_env is not None else max_per_auto
  timings = {}
  timings["neighbor"] = (bench_nbody(n=n, steps=steps, method="neighbor", q=q, p=p, m=m), None)
  timings["tensor"] = (bench_nbody(n=n, steps=steps, method="tensor", q=q, p=p, m=m), None)
  timings["tensor_bins"] = (bench_nbody(n=n, steps=steps, method="tensor_bins", q=q, p=p, m=m, max_per=max_per), max_per)
  timings["auto"] = (bench_nbody_auto(n=n, steps=steps)[0], None)
  return timings, pair_count, max_per_auto


if __name__ == "__main__":
  if os.getenv("TINYGRAD_BENCH_NBODY_COMPARE", "0"):
    timings, pairs, max_per_auto = bench_nbody_compare()
    print(f"pairs: {pairs}, auto_max_per: {max_per_auto}")
    for method, (t, cap) in timings.items():
      suffix = "" if cap is None else f", max_per={cap}"
      print(f"{method}: {t:.4f}s{suffix}")
  else:
    max_per_env = _parse_max_per(os.getenv("TINYGRAD_NBODY_MAX_PER"))
    for method in ("auto", "neighbor", "naive", "barnes_hut", "tensor", "tensor_bins"):
      n = 128 if method == "tensor" else 256
      max_per = max_per_env
      if method == "tensor_bins" and max_per is None:
        max_per = "auto"
      if method == "auto":
        t, chosen = bench_nbody_auto(n=n)
        print(f"{method}: {t:.4f}s, chosen={chosen}")
      else:
        t = bench_nbody(n=n, method=method, max_per=max_per)
        suffix = "" if max_per is None or method != "tensor_bins" else f", max_per={max_per}"
        print(f"{method}: {t:.4f}s{suffix}")
