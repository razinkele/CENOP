import numpy as np
from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation


def test_social_cache_reuse():
    params = SimulationParameters()
    params.communication_range_km = 5.0
    params.communication_recompute_interval = 3

    pop = PorpoisePopulation(count=50, params=params)

    # initial step computes and sets cache
    mask = pop.active_mask.copy()
    soc_dx1, soc_dy1 = pop._compute_social_vectors(mask, ambient_rl=None)
    assert pop._social_cache is not None
    counter_after_first = pop._neighbor_recompute_counter
    assert counter_after_first == params.communication_recompute_interval

    # subsequent calls within interval should reuse cache (counter decrements)
    for i in range(1, params.communication_recompute_interval):
        prev_idx_i = pop._social_cache['idx_i'].copy()
        soc_dx2, soc_dy2 = pop._compute_social_vectors(mask, ambient_rl=None)
        # cache should be unchanged
        assert np.array_equal(prev_idx_i, pop._social_cache['idx_i'])
        # counter should have decremented
        assert pop._neighbor_recompute_counter == params.communication_recompute_interval - (i + 0)

    # After interval expires, next call should rebuild cache (counter resets)
    soc_dx3, soc_dy3 = pop._compute_social_vectors(mask, ambient_rl=None)
    assert pop._neighbor_recompute_counter == params.communication_recompute_interval
