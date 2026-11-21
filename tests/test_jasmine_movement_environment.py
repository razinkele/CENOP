"""JASMINE movement draws DEPONS step lengths from the same environment as the CRW paths."""

import numpy as np

from cenop.agents.population import PorpoisePopulation
from cenop.core.time_manager import TimeMode
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.movement import MovementMode, create_movement_module
from cenop.parameters.simulation_params import SimulationParameters


def _mean_step(landscape_name):
    params = SimulationParameters(random_seed=5, porpoise_count=60, simulation_mode="JASMINE")
    land = create_homogeneous_landscape(width=200, height=200, food_prob=0.5)
    land.landscape_name = landscape_name
    land._salinity[:] = 10.0  # brackish map salinity
    pop = PorpoisePopulation(
        60,
        params,
        land,
        movement_module=create_movement_module(
            params, TimeMode.JASMINE, MovementMode.JASMINE_PHYSICS
        ),
    )
    steps = []
    for t in range(96):
        pop.step()
        if t >= 48:
            steps.append(pop._step_dist[pop.active_mask].mean())
    return float(np.mean(steps))


def test_kattegat_uses_depons_fixed_salinity_for_jasmine_steps():
    # DEPONS moves on Kattegat with salinity 34.07 whatever the map says (Porpoise.java:423-427):
    # mean step about 2 cells; the map's 10 would give about 11
    assert _mean_step("Kattegat") < 3.0


def test_other_landscapes_use_the_map_salinity():
    assert _mean_step("Lithuania") > 6.0
