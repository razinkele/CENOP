import numpy as np
from cenop.agents.population import PorpoisePopulation
from cenop.parameters import SimulationParameters


def test_social_attraction_moves_agents_closer():
    """Agents within communication range should be drawn closer after a step."""
    np.random.seed(42)
    params = SimulationParameters(porpoise_count=2)
    params.communication_enabled = True
    params.communication_range_km = 5.0  # plenty of range for the test
    params.social_weight = 0.9
    params.communication_source_level = 180.0
    params.communication_threshold = 100.0

    pop = PorpoisePopulation(count=2, params=params, landscape=None)

    # Place two agents 4 cells apart (should be within comm range)
    pop.x = np.array([10.0, 14.0], dtype=np.float32)
    pop.y = np.array([10.0, 10.0], dtype=np.float32)
    pop.prev_log_mov[:] = 1.0  # moderate step

    initial_dist = np.hypot(pop.x[0] - pop.x[1], pop.y[0] - pop.y[1])

    pop.step()  # single tick

    new_dist = np.hypot(pop.x[0] - pop.x[1], pop.y[0] - pop.y[1])

    assert new_dist <= initial_dist, "Social attraction should not increase distance between nearby agents"