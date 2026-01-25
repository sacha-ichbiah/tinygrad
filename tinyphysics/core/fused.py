"""
Deferred History Evolution - Delays numpy conversion until end.

The key optimization: Instead of calling .numpy() at each recording point
(which forces CPU-GPU sync), collect tensor references and convert at the end.

IMPORTANT: This module provides a deferred approach, but due to tinygrad's
lazy evaluation limitations, the existing SymplecticProgram.evolve() with
unrolling is often faster and more reliable.
"""
from __future__ import annotations
from typing import Callable, Any

from tinygrad import Tensor


def _snapshot(state: Tensor | tuple[Tensor, ...]) -> Tensor | tuple[Tensor, ...]:
    """Mark state using contiguous() to preserve intermediate values."""
    if isinstance(state, tuple):
        return tuple(s.contiguous() for s in state)
    return state.contiguous()


def _extract_history(snapshots: list) -> list:
    """Convert tensor snapshots to numpy arrays."""
    result = []
    for snap in snapshots:
        if isinstance(snap, tuple):
            result.append(tuple(s.numpy().copy() for s in snap))
        else:
            result.append(snap.numpy().copy())
    return result


class DeferredEvolution:
    """Evolution with deferred history extraction.

    Note: Due to tinygrad's lazy evaluation behavior, this approach
    may not provide speedups over the existing SymplecticProgram.evolve()
    with unrolling. Use with caution.
    """

    def __init__(self, step_fn: Callable):
        """Initialize deferred evolution.

        Args:
            step_fn: Function (state, dt) -> state that performs one integration step
        """
        self.step_fn = step_fn

    def evolve(self, state: Any, dt: float, steps: int, record_every: int = 1,
               project_fn: Callable | None = None,
               project_every: int = 0) -> tuple[Any, list]:
        """Evolve with deferred history extraction.

        Uses pure lazy evaluation without intermediate contiguous() calls,
        which produces correct results but may not be faster than the
        existing unrolled approach.
        """
        snapshots = []
        step = self.step_fn

        for i in range(steps):
            # Record snapshot (just store reference, no sync)
            if i % record_every == 0:
                # Store current state for later extraction
                if isinstance(state, tuple):
                    snapshots.append(tuple(s.contiguous() for s in state))
                else:
                    snapshots.append(state.contiguous())

            # Advance state (pure lazy evaluation)
            state = step(state, dt)

            # Apply projection if needed
            if project_fn is not None and project_every > 0 and (i + 1) % project_every == 0:
                state = project_fn(state, dt)

        # Final snapshot
        if isinstance(state, tuple):
            snapshots.append(tuple(s.contiguous() for s in state))
        else:
            snapshots.append(state.contiguous())

        # Extract all history at once (single CPU-GPU sync point)
        history = _extract_history(snapshots)
        return state, history


# Alias for backward compatibility
FusedEvolution = DeferredEvolution

__all__ = ["DeferredEvolution", "FusedEvolution", "_snapshot", "_extract_history"]
