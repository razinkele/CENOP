"""The seed passed to Simulation drives the whole run.

BatchRunner passes each replicate its own seed this way; the population used
params.random_seed instead, so replicates with a fixed random_seed were the same run (without
ships) and replicates without one could not be reproduced from their recorded seed.
"""

import numpy as np

from cenop.core.simulation import Simulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


def _positions(seed=None, random_seed=1, ticks=48):
    params = SimulationParameters(porpoise_count=30, sim_years=1, random_seed=random_seed)
    sim = Simulation(params=params, cell_data=create_homogeneous_landscape(), seed=seed)
    for _ in range(ticks):
        sim.step()
    pm = sim.population_manager
    return np.r_[pm.x, pm.y]


def test_seed_argument_drives_the_population():
    a, b, c = _positions(seed=5), _positions(seed=5), _positions(seed=6)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)  # was equal: the population used params.random_seed


def test_without_a_seed_argument_params_random_seed_is_used():
    assert np.array_equal(_positions(random_seed=3), _positions(random_seed=3))
    assert np.array_equal(_positions(random_seed=3), _positions(seed=3, random_seed=3))
