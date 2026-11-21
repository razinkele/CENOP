"""Learned avoidance enters the DEPONS heading as a fraction of the random walk's pull.

A fixed avoidance pull set the heading wherever the walk's weight (DEPONS crwContrib,
inertia_const + presMov * veTotal) was small, i.e. where an animal had found little food, and
pulled animals off their food (Kattegat with ships, 60 days: 438 alive and mean energy 7.47
with the memory, 499 and 10.50 without). Scaled like the social pull: 497 and 10.58.
"""

import numpy as np

from cenop.agents.population import PorpoisePopulation
from cenop.behavior import MemoryMode, create_memory_module
from cenop.behavior.disturbance_memory import AvoidanceResult
from cenop.core.time_manager import TimeMode
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.movement import MovementMode, create_movement_module
from cenop.parameters.simulation_params import SimulationParameters


def test_avoidance_does_not_set_the_heading_where_the_walk_is_weak():
    n = 200
    params = SimulationParameters(random_seed=9, porpoise_count=n, simulation_mode="JASMINE")
    land = create_homogeneous_landscape(width=300, height=300, food_prob=0.0)  # no food found
    pop = PorpoisePopulation(
        n,
        params,
        land,
        movement_module=create_movement_module(params, TimeMode.DEPONS, MovementMode.DEPONS_CRW),
        memory_module=create_memory_module(params, MemoryMode.JASMINE),
    )
    east = []
    for _ in range(6):
        pop._avoidance_result = AvoidanceResult(  # pull to the east before each step
            avoidance_dx=np.ones(n, dtype=np.float32),
            avoidance_dy=np.zeros(n, dtype=np.float32),
            avoidance_strength=np.full(n, 0.1, dtype=np.float32),
            cells_avoided=np.ones(n, dtype=np.int32),
        )
        pop.step()
        a = pop.active_mask
        off_east = np.abs((pop.heading[a] - 90.0 + 180.0) % 360.0 - 180.0)
        east.append(float((off_east < 45.0).mean()))
    # A fixed pull of 0.1 against a walk weight near inertia_const (0.001) pointed every heading
    # east; as a tenth of the walk's pull it only biases them (a quarter by chance)
    assert np.mean(east[1:]) < 0.45
