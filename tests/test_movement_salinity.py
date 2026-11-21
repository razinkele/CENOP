"""Salinity in the CRW movement: fixed where the DEPONS movement model would be extrapolated.

DEPONS fixes it at 34.07 on Kattegat (Porpoise.java:339-345, 423-427; TRACE 3.2 section 3.2.2:
the movement model was fitted to North Sea tags). CENOP does the same on the Baltic landscapes.
"""

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import (
    FIXED_MOVEMENT_SALINITY,
    SimulationParameters,
    fixed_movement_salinity,
)


def test_default_is_auto_and_unknown_values_are_rejected():
    assert SimulationParameters().movement_salinity == "auto"
    with pytest.raises(ValueError, match="movement_salinity"):
        SimulationParameters(movement_salinity="sea")


@pytest.mark.parametrize(
    "mode, landscape, expected",
    [
        ("auto", "Kattegat", FIXED_MOVEMENT_SALINITY),
        ("auto", "Lithuania", FIXED_MOVEMENT_SALINITY),
        ("auto", "CentralBaltic", FIXED_MOVEMENT_SALINITY),
        ("auto", "NorthSea", None),
        ("auto", "Homogeneous", None),
        ("map", "Kattegat", None),
        ("map", "Lithuania", None),
        ("fixed", "NorthSea", FIXED_MOVEMENT_SALINITY),
    ],
)
def test_which_landscapes_move_with_the_fixed_salinity(mode, landscape, expected):
    assert fixed_movement_salinity(mode, landscape) == expected
    assert FIXED_MOVEMENT_SALINITY == 34.069105813295


def _mean_step(landscape_name, mode):
    params = SimulationParameters(random_seed=5, porpoise_count=60, movement_salinity=mode)
    land = create_homogeneous_landscape(width=200, height=200, food_prob=0.5)
    land.landscape_name = landscape_name
    land._salinity[:] = 7.0  # Baltic surface salinity
    pop = PorpoisePopulation(60, params, land)
    steps = []
    for t in range(96):
        pop.step()
        if t >= 48:
            steps.append(pop._step_dist[pop.active_mask & ~pop.is_dispersing].mean())
    return float(np.mean(steps))


def test_baltic_steps_use_the_fixed_salinity_by_default():
    # Salinity 7 gives about 10 cells (4 km) per 30 min; the fixed 34.07 about 2 (0.8 km)
    assert _mean_step("Lithuania", "auto") < 3.0
    assert _mean_step("Lithuania", "map") > 6.0
