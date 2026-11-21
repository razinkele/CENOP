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
