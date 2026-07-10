"""
Tests for energy budget module.

Tests the physiology/energy system including:
- DEPONS simple energy model
- JASMINE DEB model
- Energy state management
- Survival probability calculations
- Disturbance impact tracking
"""

import numpy as np
import pytest

from cenop.parameters import SimulationParameters
from cenop.physiology.energy_budget import (
    DEPONSEnergyModule,
    EnergyContext,
    EnergyMode,
    EnergyResult,
    EnergyState,
    JASMINEEnergyModule,
    create_energy_module,
)


class TestEnergyState:
    """Test EnergyState creation and properties."""

    def test_create_energy_state(self):
        """Should create energy state with correct shape."""
        state = EnergyState.create(100)

        assert state.energy.shape == (100,)
        assert state.body_mass.shape == (100,)
        assert state.body_condition.shape == (100,)
        assert state.disturbance_energy_cost.shape == (100,)

    def test_initial_energy_value(self):
        """Should use provided initial energy."""
        state = EnergyState.create(50, initial_energy=15.0)

        assert np.all(state.energy == 15.0)

    def test_default_body_mass(self):
        """Should have reasonable default body mass."""
        state = EnergyState.create(20)

        assert np.all(state.body_mass == 50.0)  # Adult porpoise ~50 kg


class TestEnergyContext:
    """Test EnergyContext creation."""

    def test_create_default_context(self):
        """Should create default context."""
        ctx = EnergyContext.create_default(50, month=6)

        assert ctx.food_available.shape == (50,)
        assert ctx.current_month == 6
        assert np.all(ctx.water_temperature == 10.0)

    def test_disturbance_flags(self):
        """Should initialize disturbance flags to False."""
        ctx = EnergyContext.create_default(20)

        assert not np.any(ctx.is_disturbed)


class TestDEPONSEnergyModule:
    """Test DEPONS energy module."""

    @pytest.fixture
    def params(self):
        return SimulationParameters(porpoise_count=20)

    @pytest.fixture
    def module(self, params):
        return DEPONSEnergyModule(params)

    @pytest.fixture
    def state(self):
        return EnergyState.create(20, initial_energy=10.0)

    @pytest.fixture
    def context(self):
        return EnergyContext.create_default(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_get_mode(self, module):
        """Should return DEPONS mode."""
        assert module.get_mode() == EnergyMode.DEPONS

    def test_compute_energy_update_returns_result(self, module, state, context, mask):
        """compute_energy_update should return EnergyResult."""
        result = module.compute_energy_update(state, context, mask)

        assert isinstance(result, EnergyResult)
        assert result.energy_intake.shape == (20,)
        assert result.energy_bmr.shape == (20,)
        assert result.net_energy_change.shape == (20,)

    def test_food_intake_equals_food_available(self, module, state, context, mask):
        """energy_intake should equal food_available directly (no hunger re-weighting).

        The hunger fraction is applied upstream in eat_food_vectorized before
        food_available reaches this module. DEPONSEnergyModule must NOT apply
        hunger again, otherwise the fraction would be applied twice.
        """
        context.food_available[:10] = 0.4  # Less food (e.g. hungry agents ate less)
        context.food_available[10:] = 0.8  # More food

        result = module.compute_energy_update(state, context, mask)

        # energy_intake should exactly mirror food_available
        np.testing.assert_allclose(result.energy_intake[:10], 0.4, rtol=1e-5)
        np.testing.assert_allclose(result.energy_intake[10:], 0.8, rtol=1e-5)

    def test_energy_clamped_to_bounds(self, module, state, context, mask):
        """Energy should stay within [0, 20]."""
        state.energy[:] = 19.0
        context.food_available[:] = 1.0

        result = module.compute_energy_update(state, context, mask)
        module.apply_result(state, result, mask)

        assert np.all(state.energy <= 20.0)
        assert np.all(state.energy >= 0.0)

    def test_survival_probability_depends_on_energy(self, module, state, mask):
        """Lower energy should give lower survival probability."""
        state.energy[:10] = 2.0  # Low energy
        state.energy[10:] = 15.0  # High energy

        surv_prob = module.compute_survival_probability(state, mask)

        # Low energy agents should have lower survival
        assert np.mean(surv_prob[:10]) < np.mean(surv_prob[10:])

    def test_zero_energy_zero_survival(self, module, state, mask):
        """Zero energy should give zero survival probability."""
        state.energy[:] = 0.0

        surv_prob = module.compute_survival_probability(state, mask)

        assert np.all(surv_prob == 0.0)

    def test_disturbance_increases_cost(self, params, state, context, mask):
        """DEPONS has no disturbance energy term by default (Finding #10);
        the JASMINE opt-in flag re-enables it."""
        context.is_disturbed[:] = True
        context.deterrence_magnitude[:] = 5.0

        # DEPONS default: disturbance adds nothing.
        depons_module = DEPONSEnergyModule(params)
        result_default = depons_module.compute_energy_update(state, context, mask)
        assert np.all(result_default.energy_disturbance == 0.0)

        # JASMINE opt-in flag: disturbance now drains energy.
        jasmine_params = SimulationParameters(porpoise_count=20, jasmine_disturbance_energy=True)
        flagged_module = DEPONSEnergyModule(jasmine_params)
        result_flagged = flagged_module.compute_energy_update(state, context, mask)
        assert np.all(result_flagged.energy_disturbance > 0.0)

    def test_lactation_increases_cost(self, module, state, context, mask):
        """Lactation should increase energy cost."""
        context.is_lactating[:10] = True

        result = module.compute_energy_update(state, context, mask)

        # Lactating agents should have higher BMR
        assert np.mean(result.energy_bmr[:10]) > np.mean(result.energy_bmr[10:])


class TestJASMINEEnergyModule:
    """Test JASMINE DEB energy module."""

    @pytest.fixture
    def params(self):
        return SimulationParameters(porpoise_count=20)

    @pytest.fixture
    def module(self, params):
        return JASMINEEnergyModule(params)

    @pytest.fixture
    def state(self):
        return EnergyState.create(20, initial_energy=10.0)

    @pytest.fixture
    def context(self):
        return EnergyContext.create_default(20)

    @pytest.fixture
    def mask(self):
        return np.ones(20, dtype=bool)

    def test_get_mode(self, module):
        """Should return JASMINE mode."""
        assert module.get_mode() == EnergyMode.JASMINE

    def test_compute_energy_update_returns_result(self, module, state, context, mask):
        """compute_energy_update should return EnergyResult."""
        result = module.compute_energy_update(state, context, mask)

        assert isinstance(result, EnergyResult)
        assert result.energy_thermoregulation.shape == (20,)

    def test_body_mass_affects_bmr(self, module, state, context, mask):
        """Larger body mass should have higher BMR."""
        state.body_mass[:10] = 30.0  # Smaller
        state.body_mass[10:] = 70.0  # Larger

        result = module.compute_energy_update(state, context, mask)

        # Larger animals have higher BMR
        assert np.mean(result.energy_bmr[10:]) > np.mean(result.energy_bmr[:10])

    def test_activity_state_affects_cost(self, module, state, context, mask):
        """Different behavioral states should have different costs."""
        # FORAGING (1) vs DISTURBED (5)
        context.behavioral_state[:10] = 1  # FORAGING
        context.behavioral_state[10:] = 5  # DISTURBED
        context.current_speed[:] = 1.0

        result = module.compute_energy_update(state, context, mask)

        # DISTURBED should have higher activity cost
        assert np.mean(result.energy_activity[10:]) > np.mean(result.energy_activity[:10])

    def test_thermoregulation_in_cold_water(self, module, state, context, mask):
        """Cold water should increase thermoregulation cost."""
        context.water_temperature[:10] = 15.0  # Within thermoneutral
        context.water_temperature[10:] = 2.0  # Below thermoneutral

        result = module.compute_energy_update(state, context, mask)

        # Cold water should increase thermoregulation cost
        assert np.mean(result.energy_thermoregulation[10:]) > np.mean(
            result.energy_thermoregulation[:10]
        )

    def test_body_condition_updates(self, module, state, context, mask):
        """Body condition should update based on energy."""
        state.energy[:] = 5.0  # Low energy

        result = module.compute_energy_update(state, context, mask)
        module.apply_result(state, result, mask)

        # Body condition should reflect energy level
        assert np.all(state.body_condition <= 1.0)
        assert np.all(state.body_condition >= 0.1)

    def test_disturbance_events_tracked(self, module, state, context, mask):
        """Disturbance events should be counted."""
        context.is_disturbed[:] = True
        context.deterrence_magnitude[:] = 5.0

        result = module.compute_energy_update(state, context, mask)
        module.apply_result(state, result, mask)

        # Disturbance events should be counted
        assert np.all(state.disturbance_events > 0)

    def test_get_fitness_metrics(self, module, state, mask):
        """Should return fitness metrics."""
        metrics = module.get_fitness_metrics(state, mask)

        assert "mean_body_condition" in metrics
        assert "total_disturbance_cost" in metrics
        assert "agents_in_deficit" in metrics


class TestFactoryFunction:
    """Test factory function."""

    def test_create_depons_module(self):
        """Factory should create DEPONS module."""
        params = SimulationParameters(porpoise_count=10)
        module = create_energy_module(params, EnergyMode.DEPONS)

        assert isinstance(module, DEPONSEnergyModule)
        assert module.get_mode() == EnergyMode.DEPONS

    def test_create_jasmine_module(self):
        """Factory should create JASMINE module."""
        params = SimulationParameters(porpoise_count=10)
        module = create_energy_module(params, EnergyMode.JASMINE)

        assert isinstance(module, JASMINEEnergyModule)
        assert module.get_mode() == EnergyMode.JASMINE


class TestEnergyResultProperties:
    """Test EnergyResult properties."""

    def test_total_cost_calculation(self):
        """total_cost should sum all cost components."""
        result = EnergyResult(
            energy_intake=np.array([1.0]),
            energy_bmr=np.array([0.1]),
            energy_activity=np.array([0.2]),
            energy_thermoregulation=np.array([0.05]),
            energy_reproduction=np.array([0.15]),
            energy_disturbance=np.array([0.1]),
            net_energy_change=np.array([0.4]),
            energy_balance=np.array([1.0]),
            survival_probability=np.array([0.99]),
        )

        total = result.total_cost
        expected = 0.1 + 0.2 + 0.05 + 0.15 + 0.1

        np.testing.assert_almost_equal(total[0], expected)


class TestEnergyStatistics:
    """Test energy statistics."""

    def test_get_statistics(self):
        """Should return meaningful statistics."""
        params = SimulationParameters(porpoise_count=20)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(20, initial_energy=10.0)
        mask = np.ones(20, dtype=bool)

        stats = module.get_statistics(state, mask)

        assert "mean_energy" in stats
        assert "min_energy" in stats
        assert "max_energy" in stats
        assert stats["mean_energy"] == 10.0

    def test_statistics_with_inactive_agents(self):
        """Statistics should only consider active agents."""
        params = SimulationParameters(porpoise_count=20)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(20, initial_energy=10.0)
        state.energy[10:] = 5.0  # Different energy for inactive

        mask = np.zeros(20, dtype=bool)
        mask[:10] = True  # Only first 10 active

        stats = module.get_statistics(state, mask)

        assert stats["mean_energy"] == 10.0  # Only active agents


class TestDoubleHungerFix:
    """Test that hunger fraction is not applied twice."""

    def test_food_intake_equals_food_available_in_depons(self):
        """In DEPONS mode, energy_intake should equal food_available directly.

        The hunger fraction is already applied during eat_food_vectorized,
        so DEPONSEnergyModule should NOT multiply by hunger again.
        """
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        state = EnergyState.create(5, initial_energy=15.0)
        mask = np.ones(5, dtype=bool)
        food = np.full(5, 0.3, dtype=np.float32)

        ctx = EnergyContext.create_default(5, month=6)
        ctx.food_available = food

        result = module.compute_energy_update(state, ctx, mask)

        np.testing.assert_allclose(
            result.energy_intake[mask],
            food[mask],
            rtol=1e-5,
            err_msg="energy_intake should equal food_available (already hunger-weighted)",
        )


class TestSeasonalScaling:
    """Test seasonal energy scaling matches Java Porpoise.java 3-state step function."""

    def test_warm_months_5_to_9(self):
        """Months 5-9 (May-Sep) should use e_warm=1.3."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        for month in [5, 6, 7, 8, 9]:
            scaling = module._get_seasonal_scaling(month, 1)
            assert float(scaling) == pytest.approx(
                params.e_warm, abs=1e-5
            ), f"Month {month} should use e_warm={params.e_warm}"

    def test_transition_months_4_and_10(self):
        """Months 4 (Apr) and 10 (Oct) should use 1.15 transition."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        for month in [4, 10]:
            scaling = module._get_seasonal_scaling(month, 1)
            assert float(scaling) == pytest.approx(
                1.15, abs=1e-5
            ), f"Month {month} should use transition scaling 1.15"

    def test_winter_months(self):
        """Months 1-3, 11-12 should use scaling=1.0."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        for month in [1, 2, 3, 11, 12]:
            scaling = module._get_seasonal_scaling(month, 1)
            assert float(scaling) == pytest.approx(
                1.0, abs=1e-5
            ), f"Month {month} should use winter scaling 1.0"

    def test_warm_water_bmr_uses_correct_months(self):
        """The warm-water multiplier should use months 5-9, not 6-10."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        state = EnergyState.create(1, initial_energy=10.0)
        mask = np.ones(1, dtype=bool)
        ctx = EnergyContext.create_default(1, month=5)
        ctx.food_available = np.zeros(1)

        result_may = module.compute_energy_update(state, ctx, mask)

        ctx_oct = EnergyContext.create_default(1, month=10)
        ctx_oct.food_available = np.zeros(1)
        result_oct = module.compute_energy_update(state, ctx_oct, mask)

        # May (warm, 1.3) > October (transition, 1.15) BMR
        assert (
            result_may.energy_bmr[0] > result_oct.energy_bmr[0]
        ), "May (warm) should have higher BMR than October (transition)"


class TestEnergyModuleSplit:
    """Test split energy computation: food intake → starvation check → BMR cost."""

    def test_compute_food_intake_returns_gain(self):
        """compute_food_intake should return only the food energy gained."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        state = EnergyState.create(3, initial_energy=5.0)
        mask = np.ones(3, dtype=bool)
        ctx = EnergyContext.create_default(3, month=6)
        ctx.food_available = np.array([0.5, 0.3, 0.0], dtype=np.float32)

        intake = module.compute_food_intake(state, ctx, mask)

        np.testing.assert_allclose(intake, [0.5, 0.3, 0.0], rtol=1e-5)

    def test_compute_bmr_cost_returns_cost(self):
        """compute_bmr_cost should return the total metabolic cost."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        state = EnergyState.create(2, initial_energy=10.0)
        mask = np.ones(2, dtype=bool)
        ctx = EnergyContext.create_default(2, month=6)
        ctx.current_speed = np.zeros(2, dtype=np.float32)
        ctx.is_disturbed = np.zeros(2, dtype=bool)
        ctx.is_lactating = np.zeros(2, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        expected_bmr = 0.001 * 1.3 * params.e_use_per_30_min
        np.testing.assert_allclose(cost[mask], expected_bmr, rtol=0.01)

    def test_starvation_check_between_food_and_bmr(self):
        """Starvation should be checked on post-food, pre-BMR energy."""
        params = SimulationParameters()
        module = DEPONSEnergyModule(params)

        state = EnergyState.create(1, initial_energy=0.1)
        mask = np.ones(1, dtype=bool)
        ctx = EnergyContext.create_default(1, month=1)
        ctx.food_available = np.array([5.0], dtype=np.float32)

        intake = module.compute_food_intake(state, ctx, mask)
        post_food_energy = state.energy.copy()
        post_food_energy[mask] += intake[mask]
        post_food_energy = np.clip(post_food_energy, 0, 20)

        assert post_food_energy[0] == pytest.approx(5.1, abs=0.01)


class TestBMRCostDEPONSPurity:
    """Finding #10: DEPONS compute_bmr_cost must be BMR-only by default.

    Authoritative DEPONS has E_USE_PER_KM=0.0 (no swimming term) and no
    disturbance energy term. The headless inline reference
    (population._apply_bmr_cost, energy_module is None) is BMR-only:
        total_cost = 0.001 * scaling * e_use_per_30_min
    The module path must match it under DEPONS defaults; the swimming +
    disturbance drains are JASMINE opt-ins gated behind params.
    """

    def _ctx(self, count=8, month=1):
        ctx = EnergyContext.create_default(count, month=month)
        # Non-lactating so BMR carries no e_lact multiplier (matches inline ref).
        ctx.is_lactating[:] = False
        # Nonzero speed + active deterrence: the (pre-fix) activity + disturbance
        # terms would fire here if they were still added unconditionally.
        ctx.current_speed[:] = 2.0
        ctx.is_disturbed[:] = True
        ctx.deterrence_magnitude[:] = 0.5
        return ctx

    def test_depons_default_params_exist(self):
        params = SimulationParameters(porpoise_count=8)
        assert params.e_use_per_km == 0.0
        assert params.jasmine_disturbance_energy is False

    def test_depons_default_is_bmr_only(self):
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)  # winter -> scaling 1.0
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        expected_bmr = 0.001 * 1.0 * params.e_use_per_30_min
        np.testing.assert_allclose(cost, expected_bmr, rtol=1e-6)

    def test_matches_inline_headless_reference(self):
        # Inline path (agents/population.py:1921): 0.001 * scaling * e_use_per_30_min
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=6)  # warm -> scaling e_warm (1.3)
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        inline_reference = 0.001 * params.e_warm * params.e_use_per_30_min
        np.testing.assert_allclose(cost, inline_reference, rtol=1e-6)

    def test_jasmine_flags_enable_extra_terms(self):
        params = SimulationParameters(
            porpoise_count=8, e_use_per_km=0.0001, jasmine_disturbance_energy=True
        )
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)
        mask = np.ones(8, dtype=bool)

        cost = module.compute_bmr_cost(state, ctx, mask)

        scaling = 1.0
        bmr = 0.001 * scaling * params.e_use_per_30_min
        activity = 2.0 * 0.0001 * scaling
        disturbance = 0.002 * 0.5 * scaling
        expected = bmr + activity + disturbance
        np.testing.assert_allclose(cost, expected, rtol=1e-5)
        assert float(cost[0]) > bmr + 1e-9

    def test_combined_path_activity_disturbance_zero_by_default(self):
        # Legacy combined path (compute_energy_update) must be gated too, so the
        # two DEPONS paths never diverge.
        params = SimulationParameters(porpoise_count=8)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(8, initial_energy=10.0)
        ctx = self._ctx(8, month=1)
        mask = np.ones(8, dtype=bool)

        result = module.compute_energy_update(state, ctx, mask)

        assert np.all(result.energy_activity == 0.0)
        assert np.all(result.energy_disturbance == 0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
