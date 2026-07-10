"""Tests for energy state sync elimination and clamp consolidation."""

import numpy as np

from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters
from cenop.physiology import EnergyMode, create_energy_module


def _make_pop(count=10):
    """Create a PorpoisePopulation with energy module wired up."""
    params = SimulationParameters()
    energy_module = create_energy_module(params, EnergyMode.DEPONS)
    pop = PorpoisePopulation(
        count=count, params=params, landscape=None, energy_module=energy_module
    )
    return pop


class TestEnergyStateView:

    def test_energy_state_is_shared_view(self):
        pop = _make_pop(10)
        assert pop._energy_state.energy is pop.energy

    def test_mutation_through_view_reflects_in_population(self):
        pop = _make_pop(10)
        pop._energy_state.energy[0] = 15.0
        assert pop.energy[0] == 15.0

    def test_mutation_through_population_reflects_in_view(self):
        pop = _make_pop(10)
        pop.energy[3] = 5.0
        assert pop._energy_state.energy[3] == 5.0


class TestEnergyClampConsolidation:

    def test_energy_clamped_after_step(self):
        from cenop.core.simulation import Simulation

        params = SimulationParameters()
        params.porpoise_count = 20
        params.turbines = "off"
        params.ships_enabled = False
        sim = Simulation(params=params, seed=42)
        sim.initialize()

        for _ in range(100):
            sim.step()
            assert np.all(sim.population_manager.energy >= 0.0)
            assert np.all(sim.population_manager.energy <= 20.0)

    def test_energy_matches_baseline(self):
        from cenop.core.simulation import Simulation

        params = SimulationParameters()
        params.porpoise_count = 50
        params.turbines = "off"
        params.ships_enabled = False
        sim = Simulation(params=params, seed=42)
        sim.initialize()
        for _ in range(200):
            sim.step()
        pop = sim.population_manager
        active = pop.active_mask
        final_pop = int(np.sum(active))
        final_mean_energy = float(np.mean(pop.energy[active]))
        assert final_pop > 0
        assert 0 < final_mean_energy < 20
