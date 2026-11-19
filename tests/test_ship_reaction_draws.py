"""The per-ship reaction draws: only the needed rows are generated, with identical values."""

import numpy as np
import pytest

from cenop.agents.ship import _reaction_draw_rows


@pytest.mark.parametrize("rows", [[0], [5], [0, 1, 2], [3, 17, 18, 1999], [1999]])
def test_rows_equal_the_full_matrix(rows):
    n, steps = 2000, 30
    seed = [42, 1234, 77]
    full = np.random.default_rng(np.random.SeedSequence(seed)).random((n, steps))
    got = _reaction_draw_rows(seed, np.array(rows), steps)
    assert got.shape == (len(rows), steps)
    np.testing.assert_array_equal(got, full[rows])


def test_rows_do_not_depend_on_the_population_size():
    seed = [1, 2, 3]
    small = np.random.default_rng(np.random.SeedSequence(seed)).random((10, 30))
    np.testing.assert_array_equal(_reaction_draw_rows(seed, np.array([4, 9]), 30), small[[4, 9]])


def test_no_rows():
    assert _reaction_draw_rows([1, 2, 3], np.array([], dtype=np.intp), 30).shape == (0, 30)
