"""
Optimization helpers for CENOP.

Optional numba-accelerated functions for performance-critical code paths.
"""

from cenop.optimizations.numba_helpers import weighted_direction_sum, has_numba, accumulate_psm_updates, accumulate_social_totals

__all__ = ['weighted_direction_sum', 'has_numba', 'warmup_numba', 'accumulate_psm_updates', 'accumulate_social_totals']


def warmup_numba():
    """
    Warm up Numba JIT-compiled functions to avoid compilation latency
    during early simulation ticks.
    """
    if not has_numba:
        return

    import numpy as np

    # Create small test arrays to trigger JIT compilation
    test_dx = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    test_dy = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    test_w = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    # Call the functions to compile them
    _ = weighted_direction_sum(test_dx, test_dy, test_w)

    # Warm up PSM accumulator
    psm = np.zeros((3, 4, 4, 2), dtype=np.float32)
    idx = np.array([0, 1, 2], dtype=np.int32)
    ys = np.array([0, 1, 2], dtype=np.int32)
    xs = np.array([1, 2, 3], dtype=np.int32)
    food = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    _ = accumulate_psm_updates(psm, idx, ys, xs, food)

    # Warm up social accumulator using a small canonical pairs example
    count = 6
    idx_i = np.array([0, 1], dtype=np.int64)
    idx_j = np.array([2, 3], dtype=np.int64)
    ux_ci = np.array([0.1, -0.2], dtype=np.float64)
    uy_ci = np.array([0.2, 0.0], dtype=np.float64)
    ux_cj = np.array([-0.1, 0.3], dtype=np.float64)
    uy_cj = np.array([0.0, -0.3], dtype=np.float64)
    p_i = np.array([0.5, 0.4], dtype=np.float64)
    p_j = np.array([0.5, 0.2], dtype=np.float64)
    ux_total = np.zeros(count, dtype=np.float64)
    uy_total = np.zeros(count, dtype=np.float64)
    sw_total = np.zeros(count, dtype=np.float64)

    _ = accumulate_social_totals(count, idx_i, idx_j, ux_ci, uy_ci, ux_cj, uy_cj, p_i, p_j, ux_total, uy_total, sw_total)
