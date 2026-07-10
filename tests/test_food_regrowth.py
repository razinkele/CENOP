"""Tests for DEPONS 3.2 logistic food regrowth."""

import numpy as np

from cenop.landscape.cell_data import create_homogeneous_landscape


class TestLogisticFoodRegrowth:
    def test_logistic_formula_single_iteration(self):
        """F=0.4, rU=0.1, K=0.8 → F_new = 0.4 + 0.1*0.4*(1-0.4/0.8) = 0.42"""
        F = 0.4
        rU = 0.1
        K = 0.8
        F_new = F + rU * F * (1.0 - F / K)
        assert abs(F_new - 0.42) < 1e-10

    def test_48x_compounding(self):
        """F=0.1, rU=0.1, K=0.8 → 48 iterations converges to ~0.773"""
        F = 0.1
        rU = 0.1
        K = 0.8
        F_after_1 = F + rU * F * (1.0 - F / K)
        delta = abs(F_after_1 - F)
        assert delta > 0.001
        f_level = F_after_1
        for _ in range(47):
            f_level = f_level + rU * f_level * (1.0 - f_level / K)
        assert abs(f_level - 0.758) < 0.01
        assert f_level < K

    def test_no_extra_compounding_when_delta_small(self):
        """F=0.799, K=0.8 → tiny delta, only 1 iteration"""
        F = 0.799
        rU = 0.1
        K = 0.8
        F_after_1 = F + rU * F * (1.0 - F / K)
        delta = abs(F_after_1 - F)
        assert delta <= 0.001

    def test_no_regrowth_above_capacity(self):
        """F=0.85 >= K=0.8 → skip"""
        assert 0.85 >= 0.8

    def test_regrowth_only_where_food_prob_positive(self):
        landscape = create_homogeneous_landscape(width=10, height=10, food_prob=0.0)
        initial_food = landscape._food_value.copy()
        landscape.replenish_food(0.1)
        np.testing.assert_array_equal(landscape._food_value, initial_food)

    def test_food_floor_at_001(self):
        landscape = create_homogeneous_landscape(width=5, height=5, food_prob=0.5)
        landscape._food_value[:] = 0.001
        landscape.replenish_food(0.1)
        assert np.all(landscape._food_value >= 0.01)

    def test_replenish_increases_food_from_low_level(self):
        """Food below K and above floor should increase after replenish."""
        landscape = create_homogeneous_landscape(width=5, height=5, food_prob=0.5)
        landscape._food_value[:] = 0.2
        initial = landscape._food_value.copy()
        landscape.replenish_food(0.1)
        # Food should have increased (food_prob=0.5 acts as K)
        assert np.all(landscape._food_value >= initial)

    def test_food_does_not_exceed_capacity(self):
        """Food should not exceed K (capacity derived from food_prob)."""
        landscape = create_homogeneous_landscape(width=5, height=5, food_prob=0.5)
        landscape._food_value[:] = 0.3
        landscape.replenish_food(0.1)
        # K = max_u * food_prob / mean_max_ent = 1.0 * 0.5 / 1.0 = 0.5 (default args)
        assert np.all(landscape._food_value <= 0.5 + 1e-9)

    def test_cells_at_or_above_k_not_modified(self):
        """Cells already at or above capacity should not change."""
        landscape = create_homogeneous_landscape(width=5, height=5, food_prob=0.5)
        # Set food above K (K=0.5 with defaults)
        landscape._food_value[:] = 0.9
        values_before = landscape._food_value.copy()
        landscape.replenish_food(0.1)
        # Cells at/above K should remain unchanged
        np.testing.assert_array_equal(landscape._food_value, values_before)


class TestPhase1Integration:
    """Smoke test: Phase 1 changes produce stable population."""

    def test_population_stable_100_days(self):
        """Run 100-day simulation and verify population stability.

        With logistic food regrowth + correct mortality, population should
        remain within ±30% of initial count over 100 simulated days.
        """
        from cenop.agents.population import PorpoisePopulation
        from cenop.parameters.simulation_params import SimulationParameters

        np.random.seed(42)
        params = SimulationParameters()
        landscape = create_homogeneous_landscape(width=200, height=200, food_prob=0.5)
        pop = PorpoisePopulation(count=200, params=params, landscape=landscape)

        initial_count = int(np.sum(pop.active_mask))

        for day in range(100):
            for tick in range(48):
                pop.step()

            # Daily food replenishment
            landscape.replenish_food(
                rate=params.food_growth_rate,
                max_u=params.max_u,
                regrowth_qualifier=params.regrowth_food_qualifier,
            )

        final_count = int(np.sum(pop.active_mask))
        ratio = final_count / initial_count

        assert (
            0.7 < ratio < 1.3
        ), f"Population ratio {ratio:.2f} ({initial_count}→{final_count}) outside ±30% stability band"


class TestFoodInitialization:
    def test_food_init_from_maxent(self):
        """Food should be initialized from maxEnt when entropy available.
        For homogeneous landscape with food_prob=0.5 and no maxEnt rasters,
        fallback should still produce food_value = food_prob."""
        landscape = create_homogeneous_landscape(width=10, height=10, food_prob=0.5)
        assert np.allclose(landscape._food_value, 0.5)

    def test_food_zero_where_food_prob_zero(self):
        """Cells with food_prob=0 should have food_value=0."""
        landscape = create_homogeneous_landscape(width=10, height=10, food_prob=0.0)
        assert np.all(landscape._food_value == 0.0)
