"""
Harmonic Oscillator Benchmark Matrix

Comprehensive overview of integrator x JIT x unroll combinations.
"""

import json
import time
from tinygrad import TinyJit
from tinygrad.tensor import Tensor
from tinygrad.physics import HamiltonianSystem


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


def _bench(system: HamiltonianSystem, steps: int, repeats: int, jit: bool, unroll: int, size: int) -> tuple[float, float]:
  def run_once() -> float:
    q, p = _make_state(size)
    if unroll > 1:
      step = system.compile_unrolled_step(0.01, unroll)
      q, p = step(q, p)
      steps_per_call = unroll
    else:
      step = TinyJit(system.step) if jit else system.step
      if jit:
        q, p = step(q, p, 0.01)
      steps_per_call = 1

    start = time.perf_counter()
    for _ in range(steps // steps_per_call):
      if unroll > 1:
        q, p = step(q, p)
      else:
        q, p = step(q, p, 0.01)
    q.numpy()
    p.numpy()
    return time.perf_counter() - start

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  return best, steps / best


def _bench_scan(system: HamiltonianSystem, steps: int, repeats: int, coupled: bool, size: int) -> tuple[float, float]:
  def run_once() -> float:
    q, p = _make_state(size)
    start = time.perf_counter()
    system.evolve_scan_kernel(q, p, dt=0.01, steps=steps, coupled=coupled)
    return time.perf_counter() - start

  times = [run_once() for _ in range(repeats)]
  best = min(times)
  return best, steps / best


def main():
  import sys
  args = sys.argv[1:]

  integrators = _parse_list_flag(args, "integrators", ["euler", "leapfrog", "yoshida4"])
  unrolls = [int(v) for v in _parse_list_flag(args, "unrolls", ["1", "2", "4", "8", "16"])]
  steps = _parse_int_flag(args, "steps", 512)
  repeats = _parse_int_flag(args, "repeats", 3)
  size = _parse_int_flag(args, "size", 1)
  coupled_size = _parse_int_flag(args, "coupled-size", max(size, 1024))
  include_jit = "--jit" in args
  include_scan = "--scan" in args
  include_scan_coupled = "--scan-coupled" in args
  include_coupled_rows = "--coupled-rows" in args
  use_coupled_hamiltonian = "--coupled-hamiltonian" in args
  json_path = _parse_str_flag(args, "json", None)

  print("=" * 72)
  print("HARMONIC OSCILLATOR BENCHMARK MATRIX (AUTOGRAD)")
  print("=" * 72)
  print(f"steps={steps} repeats={repeats} size={size} coupled_size={coupled_size} integrators={','.join(integrators)} "
        f"unrolls={','.join(map(str, unrolls))} jit={include_jit} scan={include_scan} scan_coupled={include_scan_coupled} "
        f"coupled_rows={include_coupled_rows} coupled_H={use_coupled_hamiltonian}")
  print("-" * 72)
  print(f"{'integrator':12s} {'mode':12s} {'jit':5s} {'unroll':6s} {'ms':>10s} {'steps/s':>12s}")

  json_results = []
  for integrator in integrators:
    hamiltonian_name = "coupled" if use_coupled_hamiltonian else "harmonic"
    H = coupled_hamiltonian() if use_coupled_hamiltonian else harmonic_hamiltonian()
    system = HamiltonianSystem(H, integrator=integrator)

    for unroll in unrolls:
      if steps % unroll != 0:
        continue
      if unroll > 1:
        jit = True
      else:
        jit = include_jit

      elapsed, steps_per_s = _bench(system, steps, repeats, jit=jit, unroll=unroll, size=size)
      time_per_step_s = elapsed / steps
      label_jit = "yes" if jit else "no"
      print(f"{integrator:12s} {'step':12s} {label_jit:5s} {unroll:6d} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")
      json_results.append({
        "integrator": integrator,
        "mode": "step",
        "hamiltonian": hamiltonian_name,
        "jit": jit,
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

    if integrator == "leapfrog" and include_scan and not use_coupled_hamiltonian:
      elapsed, steps_per_s = _bench_scan(system, steps, repeats, coupled=False, size=size)
      time_per_step_s = elapsed / steps
      print(f"{integrator:12s} {'scan':12s} {'-':5s} {'-':6s} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")
      json_results.append({
        "integrator": integrator,
        "mode": "scan",
        "hamiltonian": hamiltonian_name,
        "jit": False,
        "unroll": 0,
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "time_per_step_s": time_per_step_s,
      })

    if integrator == "leapfrog" and include_scan_coupled:
      coupled_system = system if use_coupled_hamiltonian else HamiltonianSystem(coupled_hamiltonian(), integrator=integrator)
      elapsed, steps_per_s = _bench_scan(coupled_system, steps, repeats, coupled=True, size=coupled_size)
      time_per_step_s = elapsed / steps
      print(f"{integrator:12s} {'scan_cpl':12s} {'-':5s} {'-':6s} {elapsed*1e3:10.2f} {steps_per_s:12,.0f}")
      json_results.append({
        "integrator": integrator,
        "mode": "scan_coupled",
        "hamiltonian": "coupled",
        "jit": False,
        "unroll": 0,
        "elapsed_s": elapsed,
        "steps_per_s": steps_per_s,
        "time_per_step_s": time_per_step_s,
        "size": coupled_size,
      })

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
      "include_coupled_rows": include_coupled_rows,
      "use_coupled_hamiltonian": use_coupled_hamiltonian,
      "results": json_results,
    }
    runs = _load_json_runs(json_path)
    runs.append(run)
    _write_json_runs(json_path, runs)


if __name__ == "__main__":
  main()
