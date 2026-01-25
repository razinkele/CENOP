import numpy as np
from cenop.agents.population import PorpoisePopulation
from cenop.parameters import SimulationParameters


def test_ambient_masking_reduces_detection():
    np.random.seed(1)
    params = SimulationParameters(porpoise_count=2)
    params.communication_enabled = True
    params.communication_range_km = 5.0
    params.social_weight = 0.9
    params.communication_source_level = 180.0
    params.communication_threshold = 0.0  # threshold in SNR space
    params.communication_response_slope = 0.5

    pop = PorpoisePopulation(count=2, params=params, landscape=None)
    # Place two agents close
    pop.x = np.array([10.0, 12.0], dtype=np.float32)
    pop.y = np.array([10.0, 10.0], dtype=np.float32)
    pop.prev_log_mov[:] = 1.0

    mask = pop.active_mask

    # No ambient noise
    soc_dx_no, soc_dy_no = pop._compute_social_vectors(mask, ambient_rl=None)
    mag_no = np.hypot(soc_dx_no[0], soc_dy_no[0])

    # High ambient noise at listener (reduce SNR)
    ambient_rl = np.array([200.0, 200.0], dtype=np.float32)
    soc_dx_hi, soc_dy_hi = pop._compute_social_vectors(mask, ambient_rl=ambient_rl)
    mag_hi = np.hypot(soc_dx_hi[0], soc_dy_hi[0])

    assert mag_hi < mag_no, "Ambient noise should reduce social attraction magnitude"