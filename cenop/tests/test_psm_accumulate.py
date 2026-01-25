import numpy as np
from cenop.optimizations import accumulate_psm_updates


def test_accumulate_psm_updates_matches_np_add_at():
    rows, cols = 5, 5
    count = 4
    # buffer shape: (count, rows, cols, 2)
    buf1 = np.zeros((count, rows, cols, 2), dtype=np.float32)
    buf2 = np.zeros_like(buf1)

    idx = np.array([0, 1, 2, 3, 1], dtype=np.int32)
    ys = np.array([0, 1, 2, 3, 1], dtype=np.int32)
    xs = np.array([0, 1, 2, 3, 1], dtype=np.int32)
    food = np.array([0.5, 1.0, 0.2, 0.3, 0.1], dtype=np.float32)

    # Baseline using np.add.at
    np.add.at(buf1[:, :, :, 0], (idx, ys, xs), 1.0)
    np.add.at(buf1[:, :, :, 1], (idx, ys, xs), food)

    # Use accumulator
    accumulate_psm_updates(buf2, idx, ys, xs, food)

    assert np.allclose(buf1, buf2)
