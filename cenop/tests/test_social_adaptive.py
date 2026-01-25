from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation


def test_adaptive_low_disp_doubles():
    params = SimulationParameters()
    params.communication_recompute_interval = 4
    params.communication_recompute_min_interval = 1
    params.communication_recompute_max_interval = 16
    params.communication_recompute_adaptive = True

    pop = PorpoisePopulation(count=50, params=params)
    pop._current_recompute_interval = 4

    pop._update_neighbor_recompute_interval(mean_disp_m=0.0)
    assert pop._current_recompute_interval == 8

    # Doubling again
    pop._update_neighbor_recompute_interval(mean_disp_m=0.0)
    assert pop._current_recompute_interval == 16  # capped at max


def test_adaptive_high_disp_sets_min():
    params = SimulationParameters()
    params.communication_recompute_interval = 4
    params.communication_recompute_min_interval = 1
    params.communication_recompute_max_interval = 16
    params.communication_recompute_adaptive = True

    pop = PorpoisePopulation(count=50, params=params)
    pop._current_recompute_interval = 8

    pop._update_neighbor_recompute_interval(mean_disp_m=1000.0)
    assert pop._current_recompute_interval == params.communication_recompute_min_interval
