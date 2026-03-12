"""
Numba-accelerated simulation kernels.

Each kernel is a standalone @njit function that receives and returns flat NumPy
arrays.  No class references or Python objects.  Wrapper methods in population.py
unpack SoA arrays, call the kernel, and write results back.

Existing Numba functions in numba_helpers.py and __init__.py are NOT moved here;
this module holds NEW kernels extracted from population.py hot paths.
"""
from __future__ import annotations

import numpy as np

try:
    from numba import njit, prange
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

    def prange(*args):
        return range(*args)


@njit(cache=True)
def reflect_boundaries_kernel(
    new_x: np.ndarray,
    new_y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    world_w: int,
    world_h: int,
    mask: np.ndarray,
) -> None:
    """
    DEPONS-style bouncy borders — Numba kernel.

    Reflects positions that overshoot domain edges back into bounds
    and flips the displacement sign so heading recalculation points inward.
    Modifies arrays in-place.
    """
    max_x = float(world_w - 1)
    max_y = float(world_h - 1)
    n = new_x.shape[0]

    for i in range(n):
        if mask[i]:
            if new_x[i] < 0.0:
                new_x[i] = -new_x[i]
                dx[i] = -dx[i]
            elif new_x[i] > max_x:
                new_x[i] = 2.0 * max_x - new_x[i]
                dx[i] = -dx[i]
        # Safety clamp applied to all (matches NumPy np.clip behaviour)
        if new_x[i] < 0.0:
            new_x[i] = 0.0
        elif new_x[i] > max_x:
            new_x[i] = max_x

    for i in range(n):
        if mask[i]:
            if new_y[i] < 0.0:
                new_y[i] = -new_y[i]
                dy[i] = -dy[i]
            elif new_y[i] > max_y:
                new_y[i] = 2.0 * max_y - new_y[i]
                dy[i] = -dy[i]
        # Safety clamp applied to all (matches NumPy np.clip behaviour)
        if new_y[i] < 0.0:
            new_y[i] = 0.0
        elif new_y[i] > max_y:
            new_y[i] = max_y


def warmup_kernels():
    """Pre-compile all kernels with small dummy data to avoid first-call latency."""
    if not NUMBA_AVAILABLE:
        return False
    x = np.array([1.0, -1.0, 25.0], dtype=np.float64)
    y = np.array([1.0, -1.0, 25.0], dtype=np.float64)
    dx = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    dy = np.array([1.0, -1.0, 1.0], dtype=np.float64)
    mask = np.array([True, True, True])
    reflect_boundaries_kernel(x, y, dx, dy, 20, 20, mask)
    return True
