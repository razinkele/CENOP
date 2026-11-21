"""The behaviour state machine inside the population follows the model's real dispersal."""

import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.behavior.hybrid_fsm import create_behavior_fsm
from cenop.behavior.states import BehaviorState
from cenop.core.time_manager import TimeMode
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters

FORAGING = BehaviorState.FORAGING.value
DISPERSING = BehaviorState.DISPERSING.value
DISTURBED = BehaviorState.DISTURBED.value
RESTING = BehaviorState.RESTING.value


def _population(time_mode):
    params = SimulationParameters(random_seed=1, porpoise_count=4)
    land = create_homogeneous_landscape(width=50, height=50, food_prob=0.5)
    pop = PorpoisePopulation(4, params, land, behavior_fsm=create_behavior_fsm(params, time_mode))
    pop.deter_strength[:] = 0.0
    pop.is_dispersing[:] = False
    pop.energy[:] = 15.0
    pop._step_dist[:] = 0.0
    pop._behavior_state.state[:] = FORAGING
    return pop


@pytest.mark.parametrize("time_mode", [TimeMode.DEPONS, TimeMode.JASMINE])
def test_states_follow_the_models_dispersal(time_mode):
    pop = _population(time_mode)
    pop.is_dispersing[:] = [True, False, True, False]
    pop.deter_strength[2] = 0.5  # deterred while dispersing: disturbance has priority
    pop._behavior_state.state[3] = DISPERSING  # the model ended its dispersal

    pop._update_behavior_fsm(pop.active_mask)

    assert pop._behavior_state.state.tolist() == [DISPERSING, FORAGING, DISTURBED, FORAGING]


@pytest.mark.parametrize("time_mode", [TimeMode.DEPONS, TimeMode.JASMINE])
def test_recovered_animal_that_is_dispersing_shows_dispersing(time_mode):
    pop = _population(time_mode)
    pop._behavior_state.state[0] = DISTURBED
    pop._last_disturbance_tick[0] = 0
    pop._global_tick = 500  # long past the recovery period, no deterrence now
    pop.is_dispersing[0] = True

    pop._update_behavior_fsm(pop.active_mask)

    assert pop._behavior_state.state[0] == DISPERSING


def test_jasmine_dispersing_wins_over_resting():
    pop = _population(TimeMode.JASMINE)
    pop.energy[:] = 2.0  # low energy would send a foraging animal to RESTING
    pop._behavior_state.state[1] = RESTING
    pop.is_dispersing[:2] = True

    pop._update_behavior_fsm(pop.active_mask)

    assert pop._behavior_state.state[:2].tolist() == [DISPERSING, DISPERSING]
    assert pop._behavior_state.state[2:].tolist() == [RESTING, RESTING]
