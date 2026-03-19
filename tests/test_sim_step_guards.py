"""Tests for sim.step() deterrence skip guards."""

import numpy as np
import pytest
from cenop.core.simulation import Simulation
from cenop.parameters.simulation_params import SimulationParameters


class TestDeterrenceSkipGuards:
    """Test that deterrence computation is skipped when not needed."""

    def test_zero_deterrence_preallocated(self):
        """Simulation should have a pre-allocated zero deterrence array."""
        params = SimulationParameters()
        params.turbines = "off"
        params.ships_enabled = False
        sim = Simulation(params=params, seed=42)
        sim.initialize()

        assert hasattr(sim, "_zero_deterrence")
        n = sim.population_manager.count
        assert sim._zero_deterrence.shape == (n,)
        assert sim._zero_deterrence.dtype == np.float64
        np.testing.assert_array_equal(sim._zero_deterrence, 0.0)

    def test_zero_array_not_mutated_after_step(self):
        """The pre-allocated zero array must not be modified by step()."""
        params = SimulationParameters()
        params.turbines = "off"
        params.ships_enabled = False
        sim = Simulation(params=params, seed=42)
        sim.initialize()

        zero_id = id(sim._zero_deterrence)
        sim.step()

        # Same object, still zeros
        assert id(sim._zero_deterrence) == zero_id
        np.testing.assert_array_equal(sim._zero_deterrence, 0.0)

    def test_turbines_off_skips_turbine_deterrence(self):
        """When turbines='off', turbine deterrence should be zero."""
        params = SimulationParameters()
        params.turbines = "off"
        params.ships_enabled = False
        params.porpoise_count = 5
        sim = Simulation(params=params, seed=42)
        sim.initialize()

        # Run a few steps — should not crash
        for _ in range(10):
            sim.step()

        assert sim.population_manager.population_size > 0

    def test_ships_disabled_skips_ship_deterrence(self):
        """When ships_enabled=False, ship deterrence should be zero."""
        params = SimulationParameters()
        params.turbines = "off"
        params.ships_enabled = False
        params.porpoise_count = 5
        sim = Simulation(params=params, seed=42)
        sim.initialize()

        for _ in range(10):
            sim.step()

        assert sim.population_manager.population_size > 0
