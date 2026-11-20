"""DEPONS-style food layers from seasonal maps (cenop.landscape.food_layers)."""

import numpy as np
import pytest

from cenop.landscape import food_layers as fl


def test_three_seasons_are_reproduced_at_their_months():
    spring, summer, autumn = np.array([0.2, 0.4]), np.array([0.6, 0.5]), np.array([0.3, 0.45])
    monthly = fl.monthly_from_seasons({4: spring, 7: summer, 10: autumn})
    assert monthly.shape == (12, 2)
    np.testing.assert_allclose(monthly[3], spring, atol=1e-12)
    np.testing.assert_allclose(monthly[6], summer, atol=1e-12)
    np.testing.assert_allclose(monthly[9], autumn, atol=1e-12)


def test_curve_is_cyclic_and_smooth():
    monthly = fl.monthly_from_seasons({4: np.array(0.2), 7: np.array(0.6), 10: np.array(0.3)})
    steps = np.abs(np.diff(np.concatenate([monthly, monthly[:1]])))
    assert steps.max() < 0.25  # no jump between December and January


def test_two_seasons_six_months_apart():
    monthly = fl.monthly_from_seasons({7: np.array([0.8]), 1: np.array([0.2])})
    assert monthly[6, 0] == pytest.approx(0.8)
    assert monthly[0, 0] == pytest.approx(0.2)
    assert monthly[3, 0] == pytest.approx(0.5)


def test_two_seasons_not_six_months_apart_are_refused():
    with pytest.raises(ValueError):
        fl.monthly_from_seasons({7: np.array(1.0), 9: np.array(0.5)})


def test_standardize_sets_the_depons_mean_on_water_and_keeps_food_positive():
    water = np.array([[True, True], [True, False]])
    prey = np.stack([np.array([[0.1, 0.2], [-0.05, 9.0]])] * 12)
    out = fl.standardize(prey, water)
    for m in range(12):
        assert out[m][water].mean() == pytest.approx(fl.DEPONS_MEAN_PREY)
        assert (out[m][water] > 0).all()
    assert out[0, 1, 1] == 9.0  # land untouched


def test_random_patches_share_and_placement():
    water = np.zeros((100, 100), dtype=bool)
    water[:, :60] = True
    p = fl.random_patches(water, share=0.016, seed=3)
    assert p.sum() == round(0.016 * 6000)
    assert not p[~water].any()
    assert np.array_equal(p, fl.random_patches(water, share=0.016, seed=3))


def test_fill_water_gaps_uses_the_nearest_value():
    v = np.array([[1.0, np.nan, np.nan, 4.0]])
    water = np.array([[True, True, True, False]])
    out = fl.fill_water_gaps(v, water)
    assert out[0, 1] == 1.0 and out[0, 2] == 4.0 and out[0, 3] == 4.0


def test_half_month_season_centres():
    monthly = fl.monthly_from_seasons({7.5: np.array([0.8]), 1.5: np.array([0.2])})
    assert monthly[6, 0] == pytest.approx(0.5 + 0.3 * np.cos(-np.pi / 12))
    assert monthly[6, 0] == pytest.approx(monthly[7, 0])  # July and August straddle the peak


def test_deep_gaps_get_a_low_quantile_and_shallow_gaps_stay_for_filling():
    values = np.array([[0.1, 0.2, 0.3, 0.4, np.nan, np.nan]])
    depth = np.array([[20.0, 20.0, 20.0, 20.0, 120.0, 30.0]])
    water = depth > 0
    out = fl.floor_deep_gaps(values, water, depth, deeper_than=80.0, quantile=0.0)
    assert out[0, 4] == pytest.approx(0.1)
    assert np.isnan(out[0, 5])
