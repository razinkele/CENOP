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

    # Numba-accelerated accumulator for PSM updates
    @njit
    def accumulate_psm_updates(psm_buffer, idx0, y_arr, x_arr, food_arr):
        n = idx0.shape[0]
        for k in range(n):
            i = idx0[k]
            y = y_arr[k]
            x = x_arr[k]
            # Channel 0: ticks
            psm_buffer[i, y, x, 0] += 1.0
            # Channel 1: food
            psm_buffer[i, y, x, 1] += food_arr[k]

    # Numba-accelerated accumulator for pairwise social contributions
    @njit
    def accumulate_social_totals(count: 'int64', idx_i, idx_j,
                                 ux_i, uy_i, ux_j, uy_j, p_i, p_j,
                                 ux_total, uy_total, sw_total):
        """Accumulate contributions for canonical neighbor pairs into per-agent totals.
        All arrays are expected as NumPy arrays with appropriate dtypes.
        """
        n = idx_i.shape[0]
        for k in range(n):
            i = idx_i[k]
            j = idx_j[k]
            # i's contributions
            ux_total[i] += ux_i[k]
            uy_total[i] += uy_i[k]
            sw_total[i] += p_i[k]
            # j's contributions
            ux_total[j] += ux_j[k]
            uy_total[j] += uy_j[k]
            sw_total[j] += p_j[k]

    def warmup_numba():
        """Convenience wrapper to compile hot Numba functions used by the module."""
        if not has_numba:
            return False
        import numpy as np
        from math import hypot

        # Warm up weighted_direction_sum
        test_dx = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        test_dy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        test_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)
        _ = weighted_direction_sum(test_dx, test_dy, test_w)

        # Warm up PSM accumulator
        psm = np.zeros((3, 4, 4, 2), dtype=np.float32)
        idx = np.array([0, 1, 2], dtype=np.int32)
        ys = np.array([0, 1, 2], dtype=np.int32)
        xs = np.array([1, 2, 3], dtype=np.int32)
        food = np.array([0.5, 0.25, 0.75], dtype=np.float32)
        _ = accumulate_psm_updates(psm, idx, ys, xs, food)

        # Warm up social accumulator
        count = 8
        idx_i = np.array([0, 1, 2], dtype=np.int64)
        idx_j = np.array([3, 4, 5], dtype=np.int64)
        ux_ci = np.array([0.1, -0.2, 0.0], dtype=np.float64)
        uy_ci = np.array([0.2, 0.0, -0.1], dtype=np.float64)
        ux_cj = np.array([-0.1, 0.3, 0.2], dtype=np.float64)
        uy_cj = np.array([0.0, -0.3, 0.4], dtype=np.float64)
        p_i = np.array([0.5, 0.4, 0.6], dtype=np.float64)
        p_j = np.array([0.5, 0.2, 0.3], dtype=np.float64)
        ux_total = np.zeros(count, dtype=np.float64)
        uy_total = np.zeros(count, dtype=np.float64)
        sw_total = np.zeros(count, dtype=np.float64)
        _ = accumulate_social_totals(count, idx_i, idx_j, ux_ci, uy_ci, ux_cj, uy_cj, p_i, p_j, ux_total, uy_total, sw_total)
        return True

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

    # Pure-Python fallback accumulator (used when Numba unavailable)
    def accumulate_psm_updates(psm_buffer, idx0, y_arr, x_arr, food_arr):
        for k in range(len(idx0)):
            i = int(idx0[k])
            y = int(y_arr[k])
            x = int(x_arr[k])
            psm_buffer[i, y, x, 0] += 1.0
            psm_buffer[i, y, x, 1] += float(food_arr[k])

    def accumulate_social_totals(count, idx_i, idx_j,
                                 ux_i, uy_i, ux_j, uy_j, p_i, p_j,
                                 ux_total, uy_total, sw_total):
        """Pure-Python fallback: accumulates pairwise contributions into totals."""
        for k in range(len(idx_i)):
            i = int(idx_i[k])
            j = int(idx_j[k])
            ux_total[i] += float(ux_i[k])
            uy_total[i] += float(uy_i[k])
            sw_total[i] += float(p_i[k])
            ux_total[j] += float(ux_j[k])
            uy_total[j] += float(uy_j[k])
            sw_total[j] += float(p_j[k])
