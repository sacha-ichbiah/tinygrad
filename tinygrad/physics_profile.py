from dataclasses import dataclass
from tinygrad.helpers import getenv
from tinygrad.physics import SymplecticPolicy


@dataclass(frozen=True)
class PhysicsProfile:
  name: str
  policy: SymplecticPolicy


PROFILES = {
  "fast": PhysicsProfile("fast", SymplecticPolicy(accuracy="fast", scan=True, tune=False, max_unroll=16, min_unroll=4)),
  "balanced": PhysicsProfile("balanced", SymplecticPolicy(accuracy="balanced", scan=True, tune=False, max_unroll=16, min_unroll=2)),
  "precise": PhysicsProfile("precise", SymplecticPolicy(accuracy="precise", scan=True, tune=False, max_unroll=8, min_unroll=2, drift_target=1e-6)),
  "ultra_precise": PhysicsProfile("ultra_precise", SymplecticPolicy(accuracy="precise", scan=True, tune=False, max_unroll=4, min_unroll=2, drift_target=1e-8)),
  "tuned": PhysicsProfile("tuned", SymplecticPolicy(accuracy="balanced", scan=True, tune=True, max_unroll=16, min_unroll=2)),
}


def get_profile(name: str) -> PhysicsProfile:
  if name is None or name == "":
    name = "balanced"
  name = name.lower()
  if name not in PROFILES:
    raise ValueError(f"Unknown profile: {name}")
  prof = PROFILES[name]
  drift_env = getenv("TINYGRAD_PHYSICS_DRIFT_TARGET", "")
  if drift_env != "":
    try:
      drift = float(drift_env)
      return PhysicsProfile(prof.name, _clone_with_drift(prof.policy, drift))
    except Exception:
      pass
  return prof


def _clone_with_drift(policy: SymplecticPolicy, drift: float) -> SymplecticPolicy:
  return SymplecticPolicy(
    accuracy=policy.accuracy,
    scan=policy.scan,
    tune=policy.tune,
    max_unroll=policy.max_unroll,
    min_unroll=policy.min_unroll,
    drift_target=drift,
    budget_ms=policy.budget_ms,
  )


def get_default_profile() -> PhysicsProfile:
  name = getenv("TINYGRAD_PHYSICS_PROFILE", "balanced")
  return get_profile(name)
