"""
Tests for energy budget module.

Tests the physiology/energy system including:
- DEPONS simple energy model
- JASMINE DEB model
- Energy state management
- Survival probability calculations
- Disturbance impact tracking
"""

import pytest
import numpy as np

from cenop.physiology.energy_budget import (
    EnergyMode,
    EnergyState,
    EnergyContext,
    EnergyResult,
    EnergyModule,
    DEPONSEnergyModule,
    JASMINEEnergyModule,
    create_energy_module,
)
from cenop.parameters import SimulationParameters


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

    def test_hungry_agents_eat_more(self, module, state, context, mask):
        """Hungry agents should have higher food intake."""
        # Set different hunger levels
        state.energy[:10] = 5.0   # Very hungry
        state.energy[10:] = 18.0  # Nearly full

        context.food_available[:] = 1.0  # Food available

        result = module.compute_energy_update(state, context, mask)

        # Hungry agents should have higher intake
        assert np.mean(result.energy_intake[:10]) > np.mean(result.energy_intake[10:])

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
        state.energy[:10] = 2.0   # Low energy
        state.energy[10:] = 15.0  # High energy

        surv_prob = module.compute_survival_probability(state, mask)

        # Low energy agents should have lower survival
        assert np.mean(surv_prob[:10]) < np.mean(surv_prob[10:])

    def test_zero_energy_zero_survival(self, module, state, mask):
        """Zero energy should give zero survival probability."""
        state.energy[:] = 0.0

        surv_prob = module.compute_survival_probability(state, mask)

        assert np.all(surv_prob == 0.0)

    def test_disturbance_increases_cost(self, module, state, context, mask):
        """Disturbance should increase energy cost."""
        # Without disturbance
        result1 = module.compute_energy_update(state, context, mask)

        # With disturbance
        context.is_disturbed[:] = True
        context.deterrence_magnitude[:] = 5.0

        result2 = module.compute_energy_update(state, context, mask)

        # Disturbance should add cost
        assert np.all(result2.energy_disturbance > result1.energy_disturbance)

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
        state.body_mass[:10] = 30.0   # Smaller
        state.body_mass[10:] = 70.0   # Larger

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
        context.water_temperature[10:] = 2.0   # Below thermoneutral

        result = module.compute_energy_update(state, context, mask)

        # Cold water should increase thermoregulation cost
        assert np.mean(result.energy_thermoregulation[10:]) > np.mean(result.energy_thermoregulation[:10])

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

        assert 'mean_body_condition' in metrics
        assert 'total_disturbance_cost' in metrics
        assert 'agents_in_deficit' in metrics


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

        assert 'mean_energy' in stats
        assert 'min_energy' in stats
        assert 'max_energy' in stats
        assert stats['mean_energy'] == 10.0

    def test_statistics_with_inactive_agents(self):
        """Statistics should only consider active agents."""
        params = SimulationParameters(porpoise_count=20)
        module = DEPONSEnergyModule(params)
        state = EnergyState.create(20, initial_energy=10.0)
        state.energy[10:] = 5.0  # Different energy for inactive

        mask = np.zeros(20, dtype=bool)
        mask[:10] = True  # Only first 10 active

        stats = module.get_statistics(state, mask)

        assert stats['mean_energy'] == 10.0  # Only active agents


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
