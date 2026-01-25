import numpy as np
from cenop.behavior.sound import response_probability_from_rl
from cenop.parameters import SimulationParameters


def test_response_probability_basic():
    thr = 158.0
    slope = 0.2

    p_at_thr = response_probability_from_rl(thr, thr, slope)
    assert abs(p_at_thr - 0.5) < 1e-6

    p_high = response_probability_from_rl(thr + 20.0, thr, slope)
    p_low = response_probability_from_rl(thr - 20.0, thr, slope)

    assert p_high > 0.9
    assert p_low < 0.1


def test_turbine_probabilistic_scaling_matches_monotonicity():
    """Basic integration check: probability should decrease with distance (we test RL proxy)."""
    # RL values for near and far
    rl_near = np.array([170.0, 160.0, 150.0])
    thr = 158.0
    slope = 0.2

    p = response_probability_from_rl(rl_near, thr, slope)
    assert p[0] > p[1] > p[2]
