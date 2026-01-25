import numpy as np
from cenop.optimizations import numba_helpers as nh


def test_accumulate_social_totals_matches_bincount():
    # Small synthetic example
    count = 8
    # Canonical pairs
    idx_i = np.array([0, 1, 2, 5], dtype=np.int64)
    idx_j = np.array([3, 4, 6, 7], dtype=np.int64)

    # Random contributions
    rng = np.random.RandomState(123)
    ux_ci = rng.normal(size=len(idx_i)).astype(np.float64)
    uy_ci = rng.normal(size=len(idx_i)).astype(np.float64)
    ux_cj = rng.normal(size=len(idx_i)).astype(np.float64)
    uy_cj = rng.normal(size=len(idx_i)).astype(np.float64)
    p_i = rng.uniform(0.0, 1.0, size=len(idx_i)).astype(np.float64)
    p_j = rng.uniform(0.0, 1.0, size=len(idx_i)).astype(np.float64)

    # Expected via bincount
    ux_expected = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([ux_ci, ux_cj]), minlength=count)
    uy_expected = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([uy_ci, uy_cj]), minlength=count)
    sw_expected = np.bincount(np.concatenate([idx_i, idx_j]), weights=np.concatenate([p_i, p_j]), minlength=count)

    ux_total = np.zeros(count, dtype=np.float64)
    uy_total = np.zeros(count, dtype=np.float64)
    sw_total = np.zeros(count, dtype=np.float64)

    # Call accumulator (works with Numba if available or fallback pure-Python)
    nh.accumulate_social_totals(count, idx_i, idx_j, ux_ci, uy_ci, ux_cj, uy_cj, p_i, p_j, ux_total, uy_total, sw_total)

    assert np.allclose(ux_total, ux_expected)
    assert np.allclose(uy_total, uy_expected)
    assert np.allclose(sw_total, sw_expected)
