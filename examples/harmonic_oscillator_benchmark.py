"""
Harmonic Oscillator Benchmark Matrix

Comprehensive overview of integrator x JIT x unroll combinations.
"""

import json
import time
from tinygrad.tensor import Tensor
from tinygrad.physics import simulate_hamiltonian


def harmonic_hamiltonian(k: float = 1.0, m: float = 1.0):
  def H(q, p):
    T = (p * p).sum() / (2 * m)
    V = k * (q * q).sum() / 2
    return T + V
  return H


def coupled_hamiltonian():
  def H(q, p):
    coupling = q.sum()
    return (p * p).sum() / 2 + (coupling * coupling) / 2
  return H


def _make_state(size: int) -> tuple[Tensor, Tensor]:
  q = Tensor.ones(size).contiguous()
  p = Tensor.zeros(size).contiguous()
  return q, p


def _parse_list_flag(args, name: str, default: list[str]) -> list[str]:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      raw = arg[len(prefix):]
      return [v for v in raw.split(",") if v]
  return default


def _parse_int_flag(args, name: str, default: int) -> int:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return int(arg[len(prefix):])
  return default


def _parse_float_flag(args, name: str, default: float) -> float:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return float(arg[len(prefix):])
  return default


def _parse_str_flag(args, name: str, default: str | None) -> str | None:
  prefix = f"--{name}="
  for arg in args:
    if arg.startswith(prefix):
      return arg[len(prefix):]
  return default


def _load_json_runs(path: str) -> list[dict]:
  try:
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
  except FileNotFoundError:
    return []
  except json.JSONDecodeError:
    return []

  if isinstance(data, list):
    return data
  if isinstance(data, dict) and isinstance(data.get("runs"), list):
    return data["runs"]
  return []


def _write_json_runs(path: str, runs: list[dict]) -> None:
  with open(path, "w", encoding="utf-8") as f:
    json.dump({"runs": runs}, f, indent=2)


def _bench(H, steps: int, repeats: int, size: int) -> tuple[float, float]:
  def run_once() -> float:
    q, p = _make_state(size)
    start = time.perf_counter()
    q, p, _ = simulate_hamiltonian(H, q, p, dt=0.01, steps=steps, record_every=steps)
    q.numpy()
    p.numpy()
    return time.perf_counter() - start

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  return best, steps / best


def _energy_stats(history: list[tuple]) -> tuple[float, float]:
  if not history:
    return 0.0, 0.0
  e0 = float(history[0][2])
  denom = max(abs(e0), 1e-12)
  rel = [(abs(float(e) - e0) / denom) for (_, _, e) in history]
  max_rel = max(rel)
  rms_rel = (sum(r*r for r in rel) / len(rel)) ** 0.5
  return max_rel, rms_rel


def _scan_energy_history(H, q: Tensor, p: Tensor, dt: float, steps: int) -> tuple[list[tuple], Tensor, Tensor]:
  history = []
  q_cur, p_cur = q, p
  e0 = float(H(q_cur, p_cur).numpy())
  history.append((None, None, float(e0.numpy() if hasattr(e0, "numpy") else e0)))
  for _ in range(steps):
    q_cur, p_cur, _ = simulate_hamiltonian(H, q_cur, p_cur, dt=dt, steps=1, record_every=1)
    e = float(H(q_cur, p_cur).numpy())
    history.append((None, None, float(e.numpy() if hasattr(e, "numpy") else e)))
  return history, q_cur, p_cur


def _max_abs_diff(a: Tensor, b: Tensor) -> float:
  return float((a - b).abs().max().numpy())


def _bench_scan(H, steps: int, repeats: int, size: int) -> tuple[float, float]:
  return _bench(H, steps, repeats, size)


def main():
  import sys
  args = sys.argv[1:]

  integrators = ["auto"]
  unrolls = [1]
  steps = _parse_int_flag(args, "steps", 512)
  repeats = _parse_int_flag(args, "repeats", 3)
  size = _parse_int_flag(args, "size", 1)
  coupled_size = _parse_int_flag(args, "coupled-size", max(size, 1024))
  scan_unroll = _parse_int_flag(args, "scan-unroll", 1)
  scan_vec = _parse_int_flag(args, "scan-vec", 1)
  scan_inplace = "--scan-inplace" in args
  scan_tune = "--scan-tune" in args
  tune_unrolls = [int(v) for v in _parse_list_flag(args, "scan-unrolls", ["1", "2", "4", "8"])]
  tune_vecs = [int(v) for v in _parse_list_flag(args, "scan-vecs", ["1", "4"])]
  stability = "--stability" in args
  stability_steps = _parse_int_flag(args, "stability-steps", 512)
  stability_dt = _parse_float_flag(args, "stability-dt", 0.01)
  include_jit = False
  include_scan = False
  include_scan_coupled = False
  include_scan_coupled_fused = False
  include_coupled_rows = False
  use_coupled_hamiltonian = False
  json_path = _parse_str_flag(args, "json", None)

  print("=" * 72)
  print("HARMONIC OSCILLATOR BENCHMARK MATRIX (AUTOGRAD)")
  print("=" * 72)
  print(f"steps={steps} repeats={repeats} size={size} integrator=auto")
  print("-" * 72)
  print(f"{'integrator':12s} {'mode':12s} {'jit':5s} {'unroll':6s} {'ms':>10s} {'steps/s':>12s}")

  json_results = []
  stability_results = []
  for integrator in integrators:
    hamiltonian_name = "coupled" if use_coupled_hamiltonian else "harmonic"
    H = coupled_hamiltonian() if use_coupled_hamiltonian else harmonic_hamiltonian()
    for unroll in unrolls:
      if steps % unroll != 0:
        continue
      elapsed, steps_per_s = _bench(H, steps, repeats, size=size)
      time_per_step_s = elapsed / steps
      print(f"{integrator:12s} {'step':12s} {'-':5s} {unroll:6d} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")
      json_results.append({
        "integrator": integrator,
        "mode": "step",
        "hamiltonian": hamiltonian_name,
        "jit": False,
        "unroll": unroll,
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "time_per_step_s": time_per_step_s,
      })

    if include_coupled_rows:
      coupled_system = HamiltonianSystem(coupled_hamiltonian(), integrator=integrator)
      elapsed, steps_per_s = _bench(coupled_system, steps, repeats, jit=include_jit, unroll=1, size=coupled_size)
      time_per_step_s = elapsed / steps
      label_jit = "yes" if include_jit else "no"
      print(f"{integrator:12s} {'step_cpl':12s} {label_jit:5s} {1:6d} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")
      json_results.append({
        "integrator": integrator,
        "mode": "step_cpl",
        "hamiltonian": "coupled",
        "jit": include_jit,
        "unroll": 1,
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "time_per_step_s": time_per_step_s,
        "size": coupled_size,
      })

    if stability:
      q, p = _make_state(size)
      q_ref, p_ref, history = simulate_hamiltonian(H, q, p, dt=stability_dt, steps=stability_steps, record_every=1)
      max_rel, rms_rel = _energy_stats(history)
      stability_results.append({
        "integrator": integrator,
        "mode": "stability",
        "hamiltonian": hamiltonian_name,
        "max_rel_drift": max_rel,
        "rms_rel_drift": rms_rel,
        "steps": stability_steps,
        "dt": stability_dt,
      })
      if include_scan_coupled_fused and use_coupled_hamiltonian:
        q_scan, p_scan = _make_state(size)
        scan_history, q_scan, p_scan = _scan_energy_history(
          coupled_hamiltonian(), q_scan, p_scan, dt=stability_dt, steps=stability_steps)
        max_rel, rms_rel = _energy_stats(scan_history)
        stability_results.append({
          "integrator": integrator,
          "mode": "stability_scan_coupled_fused",
          "hamiltonian": "coupled",
          "max_rel_drift": max_rel,
          "rms_rel_drift": rms_rel,
          "max_abs_q": _max_abs_diff(q_ref, q_scan),
          "max_abs_p": _max_abs_diff(p_ref, p_scan),
          "steps": stability_steps,
          "dt": stability_dt,
        })

  if stability:
    print("-" * 72)
    print("NUMERICAL STABILITY (ENERGY DRIFT)")
    print("-" * 72)
    print(f"{'integrator':12s} {'mode':22s} {'max_rel_ppm':>12s} {'rms_rel_ppm':>12s}")
    for row in stability_results:
      print(f"{row['integrator']:12s} {row['mode']:22s} {row['max_rel_drift']*1e6:12.2f} {row['rms_rel_drift']*1e6:12.2f}")
    json_results.extend(stability_results)

  if json_path:
    run = {
      "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
      "steps": steps,
      "repeats": repeats,
      "size": size,
      "coupled_size": coupled_size,
      "integrators": integrators,
      "unrolls": unrolls,
      "include_jit": include_jit,
      "include_scan": include_scan,
      "include_scan_coupled": include_scan_coupled,
      "include_scan_coupled_fused": include_scan_coupled_fused,
      "include_coupled_rows": include_coupled_rows,
      "use_coupled_hamiltonian": use_coupled_hamiltonian,
      "scan_unroll": scan_unroll,
      "scan_vec": scan_vec,
      "scan_tune": scan_tune,
      "scan_inplace": scan_inplace,
      "scan_unrolls": tune_unrolls,
      "scan_vecs": tune_vecs,
      "stability": stability,
      "stability_steps": stability_steps,
      "stability_dt": stability_dt,
      "results": json_results,
    }
    runs = _load_json_runs(json_path)
    runs.append(run)
    _write_json_runs(json_path, runs)


if __name__ == "__main__":
  main()
