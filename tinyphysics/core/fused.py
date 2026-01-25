"""
Fused Evolution Compiler - Builds entire evolution as single UOp DAG.

The key insight: Instead of calling realize() after each unroll batch,
build ONE massive computation graph for all steps and realize once at the end.

This eliminates N/unroll synchronization barriers, allowing tinygrad to:
- Fuse operations across time steps
- Optimize memory reuse
- Batch kernel launches
"""
from __future__ import annotations
from typing import Callable, Any

from tinygrad import Tensor
from tinygrad.engine.jit import TinyJit


def _snapshot(state: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
    """Mark state for history without forcing synchronization.

    Using .contiguous() tells the scheduler "I need this intermediate value"
    but doesn't force realize() - the scheduler keeps the value available
    while still fusing surrounding operations.
    """
    if isinstance(state, tuple):
        return tuple(s.contiguous() for s in state)
    return state.contiguous()


def _extract_history(snapshots: list) -> list:
    """Convert snapshots to numpy at end (single CPU-GPU sync)."""
    result = []
    for snap in snapshots:
        if isinstance(snap, tuple):
            result.append(tuple(s.numpy().copy() for s in snap))
        else:
            result.append(snap.numpy().copy())
    return result


class FusedEvolution:
    """Compiles entire evolution into single fused graph.

    Instead of:
        for i in range(steps // unroll):
            state = step(state)  # realize() per batch
            history.append(state.numpy())  # CPU sync per record

    We build:
        def fused_evolve(state):
            for i in range(steps):
                snapshots.append(state.contiguous())  # no sync
                state = step(state)  # pure graph building
            return state, snapshots
        # Single realize and numpy at end

    This is a compiler-level optimization that works for any Structure type.
    """

    def __init__(self, step_fn: Callable, max_fused: int = 4096):
        """Initialize fused evolution compiler.

        Args:
            step_fn: Function (state, dt) -> state that performs one integration step
            max_fused: Maximum steps to fuse before chunking (memory management)
        """
        self.step_fn = step_fn
        self.max_fused = max_fused
        self._cache: dict[tuple, TinyJit] = {}

    def compile(self, dt: float, steps: int, record_every: int,
                project_fn: Callable | None = None,
                project_every: int = 0) -> TinyJit:
        """Build JIT-compiled fused evolution function.

        Args:
            dt: Time step
            steps: Number of integration steps
            record_every: Record state every N steps
            project_fn: Optional projection function for constraints
            project_every: Apply projection every N steps

        Returns:
            JIT-compiled function (state) -> (final_state, snapshots)
        """
        key = (dt, steps, record_every, project_every)
        if key in self._cache:
            return self._cache[key]

        # Capture step_fn and project_fn in closure
        step = self.step_fn
        project = project_fn

        def fused_fn(state):
            snapshots = []
            for i in range(steps):
                if i % record_every == 0:
                    snapshots.append(_snapshot(state))
                state = step(state, dt)
                if project is not None and project_every > 0 and (i + 1) % project_every == 0:
                    state = project(state, dt)
            snapshots.append(_snapshot(state))
            # Single realize for entire graph
            if isinstance(state, tuple):
                for s in state:
                    s.realize()
            else:
                state.realize()
            return state, snapshots

        jit_fn = TinyJit(fused_fn)
        self._cache[key] = jit_fn
        return jit_fn

    def evolve(self, state: Any, dt: float, steps: int, record_every: int = 1,
               project_fn: Callable | None = None,
               project_every: int = 0) -> tuple[Any, list]:
        """Run fused evolution.

        Args:
            state: Initial state (Tensor or tuple of Tensors)
            dt: Time step
            steps: Number of integration steps
            record_every: Record state every N steps
            project_fn: Optional projection function for constraints
            project_every: Apply projection every N steps

        Returns:
            (final_state, history) where history is list of numpy arrays
        """
        if steps > self.max_fused:
            return self._chunked_evolve(state, dt, steps, record_every,
                                        project_fn, project_every)

        fused_fn = self.compile(dt, steps, record_every, project_fn, project_every)
        state, snapshots = fused_fn(state)
        history = _extract_history(snapshots)
        return state, history

    def _chunked_evolve(self, state: Any, dt: float, steps: int, record_every: int,
                        project_fn: Callable | None,
                        project_every: int) -> tuple[Any, list]:
        """For very long simulations, chunk to manage memory.

        The graph for 100k steps would be enormous. Instead, we chunk
        into max_fused-sized pieces with minimal synchronization overhead.
        """
        chunk = self.max_fused
        history = []
        remaining = steps
        step_idx = 0

        while remaining > 0:
            n = min(chunk, remaining)
            # Compute how many records in this chunk
            chunk_record = record_every if record_every <= n else n

            fused_fn = self.compile(dt, n, chunk_record, project_fn, project_every)
            state, snapshots = fused_fn(state)

            # Extract history (skip last which will be first of next chunk)
            chunk_hist = _extract_history(snapshots[:-1])
            history.extend(chunk_hist)

            remaining -= n
            step_idx += n

        # Final state
        final_snap = _extract_history([snapshots[-1]])
        history.append(final_snap[0])
        return state, history


__all__ = ["FusedEvolution", "_snapshot", "_extract_history"]
