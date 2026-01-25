# TinyPhysics TODO (Ordered)

Practical, implementation‑ready queue based on the current codebase and design docs. Each item is a small, testable step.

## Completed (recent)

- [x] **Expose tensor‑bins tuning knobs**
- [x] **Tensor‑bins performance bench**
- [x] **Barnes‑Hut validation**
- [x] **Contact diagnostics**
- [x] **Constraint cadence knob**
- [x] **Unified demo doc**
- [x] **Roadmap sync**
- [x] **CI bench gating**
- [x] **Roadmap accuracy pass**
- [x] **Constraint declaration API**
- [x] **Thermostat implementation**
- [x] **Tensor‑bins correctness check + micro‑bench**
- [x] **Blueprint Phase 3 section**

## Next (ordered)

- [x] **Replace CPU fallback with pure‑tensor bins on CPU**

- [x] **Tensor‑bins CPU performance**
  - Added a CPU small‑N fallback to direct tensor pairwise for `tensor_bins`.

- [x] **Auto method selection for N‑body**
  - Added `method="auto"` for CPU/GPU default routing.

- [x] **Re-run physics benches**
  - `universal_physics_bench.py`: canonical 0.0785s, so3 0.0562s, quantum 3.1232s, constraint 0.3263s, dissipative 0.5651s, fluid 3.7797s, thermostat 7.2518s.
  - `nbody_bench.py`: neighbor 0.5786s, tensor 0.1275s, tensor_bins 6.6801s (max_per=254, auto).

- [x] **Full tinyphysics test sweep**
  - `pytest -q test/test_contact_structure.py test/test_physical_system.py test/test_tensor_neighbors.py test/test_barnes_hut.py test/test_nbody_system.py test/test_split_schedule.py`
