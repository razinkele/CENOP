"""
Optional numba-accelerated helpers for hot inner loops.
If Numba is not installed, fallback pure-Python implementations are used.
"""
from typing import Tuple

try:
    import numba
    from numba import njit
    import math

    @njit
    def weighted_direction_sum(dxs: 'float64[:]', dys: 'float64[:]', weights: 'float64[:]') -> Tuple[float, float, float]:
        ux = 0.0
        uy = 0.0
        sw = 0.0
        for i in range(weights.shape[0]):
            w = weights[i]
            if w <= 0.0:
                continue
            dx = dxs[i]
            dy = dys[i]
            dist = math.hypot(dx, dy) + 1e-6
            ux += (dx / dist) * w
            uy += (dy / dist) * w
            sw += w
        return ux, uy, sw

    has_numba = True
except Exception:
    has_numba = False

    def weighted_direction_sum(dxs, dys, weights):
        ux = 0.0
        uy = 0.0
        sw = 0.0
        for i in range(len(weights)):
            w = float(weights[i])
            if w <= 0.0:
                continue
            dx = float(dxs[i])
            dy = float(dys[i])
            dist = (dx*dx + dy*dy) ** 0.5 + 1e-6
            ux += (dx / dist) * w
            uy += (dy / dist) * w
            sw += w
        return ux, uy, sw


def warmup_numba():
    """Call a small helper to warm-up potential Numba-compiled functions.

    Safe no-op for fallback implementation.
    """
    try:
        import numpy as _np
        weighted_direction_sum(_np.array([0.0], dtype=_np.float64), _np.array([0.0], dtype=_np.float64), _np.array([0.0], dtype=_np.float64))
    except Exception:
        pass
    return True
