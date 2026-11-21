"""Learned avoidance calibrated to observed recovery after pile driving.

The memory stores a response between 0 and 1 (turbine deterrence strengths run to 60, which
kept a memory above the avoidance threshold for about two months). A full-strength memory now
drops below the threshold in about 3 days (porpoise acoustic activity stayed below normal for
24-72 h at 2.6 km, Brandt et al. 2011, doi:10.3354/meps08888), within 5 km (2.5 memory cells;
50% response distances 1.3-7.4 km, Graham et al. 2019, doi:10.1098/rsos.190335).
"""

import numpy as np

from cenop.behavior import MemoryMode, create_memory_module
from cenop.behavior.disturbance_memory import (
    DisturbanceMemoryContext,
    DisturbanceMemoryState,
    JASMINEMemoryModule,
)
from cenop.parameters.simulation_params import SimulationParameters


def test_stored_memory_is_capped_at_one():
    params = SimulationParameters()
    mm = create_memory_module(params, MemoryMode.JASMINE)
    state = DisturbanceMemoryState.create(2)
    ctx = DisturbanceMemoryContext(
        is_disturbed=np.array([True, True]),
        disturbance_intensity=np.array([45.0, 0.3]),
        disturbance_x=np.array([50.0, 60.0]),
        disturbance_y=np.array([50.0, 60.0]),
        agent_x=np.array([52.0, 62.0]),
        agent_y=np.array([52.0, 62.0]),
        current_tick=10,
    )
    mm.record_disturbance(state, ctx, np.array([True, True]))
    assert max(state.memory_grids[0].values()) == 1.0
    assert max(state.memory_grids[1].values()) == 0.3


def test_defaults_fade_a_full_memory_below_the_threshold_in_about_three_days():
    p = SimulationParameters()
    assert p.avoidance_radius == 2.5  # memory cells of 2 km: 5 km
    ticks = np.log(JASMINEMemoryModule.AVOIDANCE_THRESHOLD) / np.log(1.0 - p.memory_decay_rate)
    assert 2.5 * 48 <= ticks <= 3.5 * 48
    assert p.jasmine_memory_decay_rate == p.memory_decay_rate
    assert p.jasmine_avoidance_radius == p.avoidance_radius


def test_habituation_is_gone_rather_than_a_dead_setting():
    import pytest

    with pytest.raises(TypeError):
        SimulationParameters(habituation_enabled=True)
    assert not hasattr(JASMINEMemoryModule, "HABITUATION_RATE")


def test_a_deterred_animal_remembers_the_cell_it_is_in():
    from cenop.agents.population import PorpoisePopulation
    from cenop.landscape.cell_data import create_homogeneous_landscape

    params = SimulationParameters(random_seed=2, porpoise_count=3, simulation_mode="JASMINE")
    land = create_homogeneous_landscape(width=300, height=300, food_prob=0.5)
    pop = PorpoisePopulation(
        3, params, land, memory_module=create_memory_module(params, MemoryMode.JASMINE)
    )
    pop.x[:] = [100.0, 150.0, 200.0]
    pop.y[:] = [100.0, 150.0, 200.0]
    pop._dx[:] = [3.0, 0.0, 0.0]  # just moved 3 cells east; five steps back would be x = 85
    pop._dy[:] = 0.0
    pop.deter_strength[:] = [0.5, 0.0, 0.0]
    pop._update_disturbance_memory(pop.active_mask)
    cell = pop._memory_module._position_to_cell(100.0, 100.0)
    assert list(pop._memory_state.memory_grids[0]) == [cell]
