"""Tests for D1: Compute Cell Indices Once.

Validates that _cell_xi/_cell_yi are pre-allocated, correctly computed,
and that get_depths_vectorized / get_salinities_vectorized / eat_food_vectorized
produce identical results with and without pre-computed indices.
"""

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData, LandscapeMetadata


@pytest.fixture
def params():
    """Minimal SimulationParameters-like object."""
    from cenop.parameters.simulation_params import SimulationParameters

    return SimulationParameters()


@pytest.fixture
def landscape(params):
    """Minimal CellData with depth + salinity."""
    cd = CellData("TestLandscape")
    w, h = params.world_width, params.world_height
    cd._depth = np.random.default_rng(42).uniform(1, 50, (h, w)).astype(np.float64)
    cd._salinity = np.random.default_rng(43).uniform(20, 35, (12, h, w)).astype(np.float64)
    cd._food_value = np.random.default_rng(44).uniform(0.1, 1.0, (h, w)).astype(np.float32)
    cd._food_prob = np.ones((h, w), dtype=np.float32)
    cd.metadata = LandscapeMetadata(ncols=w, nrows=h, xllcorner=0, yllcorner=0)
    cd._loaded = True
    cd._current_month = 6
    cd._demand_grid = None
    return cd


@pytest.fixture
def pop(params, landscape):
    """Small population with landscape."""
    return PorpoisePopulation(10, params, landscape=landscape)


# --- Existence & dtype tests ---


class TestCellIndicesInit:
    def test_arrays_exist(self, pop):
        assert hasattr(pop, "_cell_xi")
        assert hasattr(pop, "_cell_yi")

    def test_dtype_int32(self, pop):
        assert pop._cell_xi.dtype == np.int32
        assert pop._cell_yi.dtype == np.int32

    def test_shape_matches_count(self, pop):
        assert pop._cell_xi.shape == (pop.count,)
        assert pop._cell_yi.shape == (pop.count,)

    def test_values_match_manual_computation(self, pop):
        w = pop.landscape.width
        h = pop.landscape.height
        expected_xi = np.clip(pop.x.astype(np.int32), 0, w - 1)
        expected_yi = np.clip(pop.y.astype(np.int32), 0, h - 1)
        np.testing.assert_array_equal(pop._cell_xi, expected_xi)
        np.testing.assert_array_equal(pop._cell_yi, expected_yi)


# --- Edge-case tests ---


class TestCellIndicesEdgeCases:
    def test_position_at_zero(self, pop):
        pop.x[:] = 0.0
        pop.y[:] = 0.0
        pop._recompute_cell_indices()
        np.testing.assert_array_equal(pop._cell_xi, 0)
        np.testing.assert_array_equal(pop._cell_yi, 0)

    def test_position_at_max(self, pop):
        w = pop.landscape.width
        h = pop.landscape.height
        pop.x[:] = float(w - 1) + 0.9
        pop.y[:] = float(h - 1) + 0.9
        pop._recompute_cell_indices()
        # int32 truncation of (w-1)+0.9 = w-1 => clamped to w-1
        np.testing.assert_array_equal(pop._cell_xi, w - 1)
        np.testing.assert_array_equal(pop._cell_yi, h - 1)

    def test_position_beyond_max_clamped(self, pop):
        w = pop.landscape.width
        h = pop.landscape.height
        pop.x[:] = float(w + 10)
        pop.y[:] = float(h + 10)
        pop._recompute_cell_indices()
        np.testing.assert_array_equal(pop._cell_xi, w - 1)
        np.testing.assert_array_equal(pop._cell_yi, h - 1)

    def test_negative_position_clamped(self, pop):
        pop.x[:] = -5.0
        pop.y[:] = -3.0
        pop._recompute_cell_indices()
        np.testing.assert_array_equal(pop._cell_xi, 0)
        np.testing.assert_array_equal(pop._cell_yi, 0)


# --- Landscape API equivalence tests ---


class TestDepthsEquivalence:
    def test_with_and_without_indices(self, landscape):
        rng = np.random.default_rng(99)
        w, h = landscape.width, landscape.height
        n = 50
        positions = np.column_stack([rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)]).astype(
            np.float32
        )

        xi = np.clip(positions[:, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(positions[:, 1].astype(np.int32), 0, h - 1)

        result_without = landscape.get_depths_vectorized(positions)
        result_with = landscape.get_depths_vectorized(positions, xi=xi, yi=yi)
        np.testing.assert_array_equal(result_without, result_with)


class TestSalinitiesEquivalence:
    def test_with_and_without_indices(self, landscape):
        rng = np.random.default_rng(100)
        w, h = landscape.width, landscape.height
        n = 50
        positions = np.column_stack([rng.uniform(0, w - 1, n), rng.uniform(0, h - 1, n)]).astype(
            np.float32
        )

        xi = np.clip(positions[:, 0].astype(np.int32), 0, w - 1)
        yi = np.clip(positions[:, 1].astype(np.int32), 0, h - 1)

        result_without = landscape.get_salinities_vectorized(positions, month=3)
        result_with = landscape.get_salinities_vectorized(positions, month=3, xi=xi, yi=yi)
        np.testing.assert_array_equal(result_without, result_with)


class TestEatFoodEquivalence:
    def test_with_and_without_indices(self, landscape):
        rng = np.random.default_rng(101)
        w, h = landscape.width, landscape.height
        n = 20
        x_pos = rng.uniform(0, w - 1, n).astype(np.float32)
        y_pos = rng.uniform(0, h - 1, n).astype(np.float32)
        frac = rng.uniform(0.01, 0.1, n).astype(np.float32)

        xi = np.clip(x_pos.astype(np.int32), 0, w - 1)
        yi = np.clip(y_pos.astype(np.int32), 0, h - 1)

        # Save food state to compare
        food_backup = landscape._food_value.copy()

        result_without = landscape.eat_food_vectorized(x_pos, y_pos, frac)

        # Restore food state
        landscape._food_value[:] = food_backup

        result_with = landscape.eat_food_vectorized(x_pos, y_pos, frac, xi=xi, yi=yi)
        np.testing.assert_array_equal(result_without, result_with)
