"""The behaviour state machine inside the population: dispersal starts need both conditions."""

from cenop.agents.population import PorpoisePopulation
from cenop.behavior.hybrid_fsm import create_behavior_fsm
from cenop.behavior.states import BehaviorState
from cenop.core.time_manager import TimeMode
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


def _population():
    params = SimulationParameters(random_seed=1, porpoise_count=4)
    land = create_homogeneous_landscape(width=50, height=50, food_prob=0.5)
    pop = PorpoisePopulation(
        4, params, land, behavior_fsm=create_behavior_fsm(params, TimeMode.DEPONS)
    )
    pop.deter_strength[:] = 0.0
    pop.is_dispersing[:] = False
    pop._behavior_state.state[:] = BehaviorState.FORAGING.value
    pop.psm_buffer[:] = 0.0
    return params, pop


def _remember(pop, i, n_cells):
    flat = pop.psm_buffer[i, :, :, 0].reshape(-1)
    flat[:n_cells] = 1.0
    pop.psm_buffer[i, :, :, 0] = flat.reshape(pop.psm_buffer.shape[1:3])


def test_dispersal_start_needs_declining_energy_and_enough_memory():
    params, pop = _population()
    t_disp, enough = int(params.t_disp), int(params.min_memory_cells)
    assert enough + 1 <= pop.psm_buffer.shape[1] * pop.psm_buffer.shape[2]
    pop.days_declining_energy[:] = [t_disp, t_disp - 1, t_disp, t_disp]
    for i, n in enumerate([enough, enough, enough - 1, enough]):
        _remember(pop, i, n)
    pop.is_dispersing[3] = True

    pop._update_behavior_fsm(pop.active_mask)

    dispersing = pop._behavior_state.state == BehaviorState.DISPERSING.value
    assert dispersing.tolist() == [True, False, False, False]
