"""
Optimized functions for CENOP simulation.

This module provides performance-critical functions that can be accelerated
with Numba when available, with pure numpy fallbacks.
"""

from __future__ import annotations

import numpy as np

# Try to import numba for JIT compilation
try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    # Create a no-op decorator when numba is not available
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


@njit(cache=True)
def _accumulate_psm_updates_numba(
    psm_buffer: np.ndarray,
    agent_indices: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    food_values: np.ndarray
) -> None:
    """
    Numba-accelerated PSM buffer accumulation.

    Updates the PSM buffer for each agent at their current grid position:
    - Channel 0: Increments tick count by 1
    - Channel 1: Adds food gained

    Args:
        psm_buffer: Shape (n_agents, rows, cols, 2) - PSM memory buffer
        agent_indices: Agent IDs to update
        y_coords: Y grid coordinates for each agent
        x_coords: X grid coordinates for each agent
        food_values: Food gained by each agent
    """
    n = len(agent_indices)
    for i in range(n):
        idx = agent_indices[i]
        y = y_coords[i]
        x = x_coords[i]
        food = food_values[i]

        # Increment tick count
        psm_buffer[idx, y, x, 0] += 1.0

        # Add food gained
        psm_buffer[idx, y, x, 1] += food


def _accumulate_psm_updates_numpy(
    psm_buffer: np.ndarray,
    agent_indices: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    food_values: np.ndarray
) -> None:
    """
    Pure numpy PSM buffer accumulation (fallback).

    Uses np.add.at for unbuffered in-place addition.

    Args:
        psm_buffer: Shape (n_agents, rows, cols, 2) - PSM memory buffer
        agent_indices: Agent IDs to update
        y_coords: Y grid coordinates for each agent
        x_coords: X grid coordinates for each agent
        food_values: Food gained by each agent
    """
    # Increment tick counts (channel 0)
    np.add.at(psm_buffer[:, :, :, 0], (agent_indices, y_coords, x_coords), 1.0)

    # Add food gained (channel 1)
    np.add.at(psm_buffer[:, :, :, 1], (agent_indices, y_coords, x_coords), food_values)


def accumulate_psm_updates(
    psm_buffer: np.ndarray,
    agent_indices: np.ndarray,
    y_coords: np.ndarray,
    x_coords: np.ndarray,
    food_values: np.ndarray
) -> None:
    """
    Accumulate PSM (Persistent Spatial Memory) updates for multiple agents.

    This function updates each agent's spatial memory at their current position,
    recording both the time spent (tick count) and food obtained.

    When Numba is available, uses JIT-compiled version for ~10x speedup.
    Otherwise falls back to numpy implementation.

    Args:
        psm_buffer: Shape (n_agents, rows, cols, 2) - PSM memory buffer
                   Channel 0: tick count, Channel 1: food accumulated
        agent_indices: Array of agent IDs to update (int32)
        y_coords: Y grid coordinates for each agent (int32)
        x_coords: X grid coordinates for each agent (int32)
        food_values: Food gained by each agent (float32)

    Example:
        >>> buffer = np.zeros((100, 20, 20, 2), dtype=np.float32)
        >>> agents = np.array([0, 1, 2], dtype=np.int32)
        >>> ys = np.array([5, 10, 15], dtype=np.int32)
        >>> xs = np.array([5, 10, 15], dtype=np.int32)
        >>> food = np.array([0.5, 0.3, 0.8], dtype=np.float32)
        >>> accumulate_psm_updates(buffer, agents, ys, xs, food)
    """
    if len(agent_indices) == 0:
        return

    if NUMBA_AVAILABLE:
        _accumulate_psm_updates_numba(
            psm_buffer, agent_indices, y_coords, x_coords, food_values
        )
    else:
        _accumulate_psm_updates_numpy(
            psm_buffer, agent_indices, y_coords, x_coords, food_values
        )


# Additional optimized functions can be added here

@njit(cache=True)
def _vectorized_distance_numba(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray
) -> np.ndarray:
    """Numba-accelerated pairwise distance calculation."""
    n = len(x1)
    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        dx = x1[i] - x2[i]
        dy = y1[i] - y2[i]
        result[i] = np.sqrt(dx * dx + dy * dy)
    return result


def vectorized_distance(
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray
) -> np.ndarray:
    """
    Calculate pairwise Euclidean distances.

    Args:
        x1, y1: First set of coordinates
        x2, y2: Second set of coordinates

    Returns:
        Array of distances
    """
    if NUMBA_AVAILABLE and len(x1) > 100:
        return _vectorized_distance_numba(
            x1.astype(np.float32),
            y1.astype(np.float32),
            x2.astype(np.float32),
            y2.astype(np.float32)
        )
    else:
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
