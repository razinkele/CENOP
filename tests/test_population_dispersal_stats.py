"""Dispersal statistics shown in the app's dispersal chart."""

import numpy as np

from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters


def _population(histories):
    pop = PorpoisePopulation(len(histories), SimulationParameters(random_seed=1))
    pop._energy_history[:] = 0.0
    for i, h in enumerate(histories):
        pop._energy_history[i, : len(h)] = h  # newest day first
    return pop


def test_max_declining_days_is_the_longest_current_run_of_drops():
    pop = _population(
        [
            [5.0, 6.0, 7.0, 8.0, 9.0],  # four drops in a row, then the zero padding
            [5.0, 6.0, 4.0, 8.0],  # one drop, then a rise
            [9.0, 8.0, 7.0],  # rising: no drop
        ]
    )
    assert pop.get_dispersal_stats()["max_declining_days"] == 4


def test_max_declining_days_ignores_inactive_slots():
    pop = _population([[5.0, 6.0, 7.0, 8.0, 9.0], [5.0, 6.0, 4.0, 8.0]])
    pop.active_mask[0] = False
    assert pop.get_dispersal_stats()["max_declining_days"] == 1


def test_max_declining_days_is_capped_by_the_history_length():
    days = 10
    pop = _population([np.arange(1.0, days + 1.0)])  # drops on every recorded day
    pop._energy_history[0, :] = np.arange(1.0, days + 1.0)
    assert pop.get_dispersal_stats()["max_declining_days"] == days - 1


def test_declining_count_counts_live_animals_that_meet_the_trigger_condition():
    pop = _population(
        [
            [5.0, 6.0, 7.0, 8.0],  # three drops: meets t_disp = 3
            [5.0, 6.0, 7.0],  # two drops (the third compares 7 with the zero padding)
            [5.0, 6.0, 7.0, 8.0, 9.0],  # four drops, but the slot is inactive
            [5.0, 6.0, 7.0, 8.0, 9.0, 10.0],  # five drops
        ]
    )
    pop.active_mask[2] = False
    assert pop.params.t_disp == 3
    assert pop.get_dispersal_stats()["declining_count"] == 2


def test_runner_passes_declining_count_to_the_dispersal_chart():
    from cenop.core.simulation import Simulation
    from cenop.landscape.cell_data import create_homogeneous_landscape
    from cenop.server.simulation_controller import SimulationRunner

    params = SimulationParameters(porpoise_count=10, sim_years=1, random_seed=1)
    sim = Simulation(params, cell_data=create_homogeneous_landscape(), seed=1)
    sim.initialize()
    entry = SimulationRunner(sim).step_ticks()
    assert "declining_count" in entry["dispersal_entry"]
