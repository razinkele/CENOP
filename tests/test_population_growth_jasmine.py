"""Array growth with the JASMINE modules: the cached avoidance result grows with the arrays."""

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.behavior import MemoryMode, create_memory_module
from cenop.core.time_manager import TimeMode
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.movement import MovementMode, create_movement_module
from cenop.parameters.simulation_params import SimulationParameters


def _jasmine_population(n=8):
    params = SimulationParameters(random_seed=3, porpoise_count=n, simulation_mode="JASMINE")
    land = create_homogeneous_landscape(width=60, height=60, food_prob=0.5)
    return PorpoisePopulation(
        n,
        params,
        land,
        movement_module=create_movement_module(
            params, TimeMode.JASMINE, MovementMode.JASMINE_PHYSICS
        ),
        memory_module=create_memory_module(params, MemoryMode.JASMINE),
    )


def test_step_after_growth_with_a_memory_module():
    pop = _jasmine_population()
    pop.step()  # the memory module caches an avoidance result for the next movement step
    assert pop._avoidance_result is not None
    pop._avoidance_result.avoidance_strength[:] = 0.5  # remembered disturbances to avoid
    pop._grow(5)
    pop.step()  # crashed: the cached result kept the old length (broadcast 13 vs 8)


def test_growth_keeps_avoidance_of_existing_animals_and_gives_none_to_new_slots():
    pop = _jasmine_population()
    pop.step()
    av = pop._avoidance_result
    av.avoidance_dx[:] = np.arange(8, dtype=av.avoidance_dx.dtype) + 1.0
    av.avoidance_strength[:] = 0.5

    pop._grow(5)

    grown = pop._avoidance_result
    assert grown.avoidance_dx.shape == (13,) and grown.cells_avoided.shape == (13,)
    assert grown.avoidance_dx[:8].tolist() == list(np.arange(8) + 1.0)
    assert not grown.avoidance_dx[8:].any() and not grown.avoidance_strength[8:].any()


@pytest.mark.slow
def test_full_jasmine_run_survives_the_first_array_growth():
    """All four JASMINE modules, as the app builds them, on Kattegat past the first births
    that outgrow the arrays (day 376 with seed 11; the run used to crash there)."""
    from cenop.behavior import create_behavior_fsm
    from cenop.core.simulation import Simulation
    from cenop.landscape.cell_data import CellData
    from cenop.physiology import EnergyMode, create_energy_module

    cd = CellData("Kattegat", "data")
    cd.load()
    params = SimulationParameters(
        porpoise_count=100,
        sim_years=2,
        random_seed=11,
        landscape="Kattegat",
        turbines="off",
        simulation_mode="JASMINE",
    )
    sim = Simulation(
        params,
        cell_data=cd,
        seed=11,
        time_mode=TimeMode.JASMINE,
        movement_module=create_movement_module(
            params, TimeMode.JASMINE, MovementMode.JASMINE_PHYSICS
        ),
        behavior_fsm=create_behavior_fsm(params, TimeMode.JASMINE),
        energy_module=create_energy_module(params, EnergyMode.JASMINE),
        memory_module=create_memory_module(params, MemoryMode.JASMINE),
    )
    sim.initialize()
    pm = sim.population_manager
    for _ in range(400 * 48):
        sim.step()
    assert pm.count > 100  # the arrays grew
    assert pm.active_mask.sum() > 0
