"""The calf's spatial memory and preferred dispersal distance, per dispersal mode.

DEPONS (Porpoise.java:1180-1187, 731-733, 146-148): at birth the mother gets a `calfPsm`.
Its preferred distance is the mother's when `calfInheritsPsmDist()` (PSM types and off), else
a fresh draw (Undirected, InnerDanishWaters). While she nurses, every memory update of hers is
also made to the calf's memory when `calfHasPSM()` (PSM-Type2). The weaned calf starts with
that memory. An abandoned or unweaned calf's memory is discarded.
"""

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters


class _HighRoll:
    """Every uniform draw at 0.99: each weaned calf joins, nobody conceives."""

    def __init__(self, rng):
        self._rng = rng

    def __getattr__(self, name):
        return getattr(self._rng, name)

    def random(self, *args, **kwargs):
        out = self._rng.random(*args, **kwargs)
        if isinstance(out, np.ndarray):
            out[:] = 0.99
        return out


def _pop(mode, count=6):
    params = SimulationParameters(random_seed=9, porpoise_count=count, dispersal=mode)
    land = create_homogeneous_landscape(width=100, height=100, food_prob=1.0)
    pop = PorpoisePopulation(count=count, params=params, landscape=land)
    pop.pregnancy_status[:] = 0
    pop.days_since_mating[:] = -99
    pop.mating_day[:] = -99
    pop.with_calf[:] = False
    pop.energy[:] = 10.0
    pop.rng = _HighRoll(pop.rng)
    return pop


def _birth_then_wean(pop, mother=0, nursing_days=3):
    """Mother gives birth at tick 1 and weans about `nursing_days` later.

    Returns (mother's memory gained while nursing, index of the calf).
    """
    pop.active_mask[-1] = False  # a free slot for the calf
    pop.pregnancy_status[mother] = 1
    pop.days_since_mating[mother] = pop.params.gestation_time
    pop.step()  # the daily step also runs at tick 1: she gives birth here
    assert pop.with_calf[mother]
    at_birth = pop.psm_buffer[mother].copy()
    pop.days_since_birth[mother] = pop.params.nursing_time - nursing_days
    for _ in range(48 * (nursing_days + 3)):
        if not pop.with_calf[mother]:
            break
        pop.step()
    # the weaning tick's own feeding is still recorded for the calf (the daily step runs last)
    gained = pop.psm_buffer[mother] - at_birth
    calf = pop.count - 1
    assert pop.active_mask[calf] and not pop.with_calf[mother]
    return gained, calf


class TestPsmType2:
    def test_calf_starts_with_the_memory_its_mother_gained_while_nursing(self):
        pop = _pop("PSM-Type2")
        gained, calf = _birth_then_wean(pop)
        assert gained[..., 0].sum() > 100  # she fed on most of the nursing ticks
        np.testing.assert_allclose(pop.psm_buffer[calf], gained, rtol=1e-5)

    def test_calf_inherits_the_mothers_preferred_distance(self):
        pop = _pop("PSM-Type2")
        mothers = pop._psm_instances[0].preferred_distance
        _, calf = _birth_then_wean(pop)
        assert pop._psm_instances[calf].preferred_distance == pytest.approx(mothers)

    def test_abandoned_calfs_memory_is_discarded(self):
        pop = _pop("PSM-Type2")
        pop.pregnancy_status[0] = 1
        pop.days_since_mating[0] = pop.params.gestation_time
        for _ in range(60):
            pop.step()
        assert pop.with_calf[0]
        pop._calf_memory_consolidate()
        assert pop._calf_keys.size > 0
        pop.with_calf[0] = False  # abandoned (the kernels clear the flag in place)
        for _ in range(48):
            pop.step()
        pop._calf_memory_consolidate()
        assert pop._calf_keys.size == 0

    def test_memory_survives_array_growth(self):
        pop = _pop("PSM-Type2")
        pop.pregnancy_status[0] = 1
        pop.days_since_mating[0] = pop.params.gestation_time
        pop.step()  # birth at tick 1
        assert pop.with_calf[0]
        at_birth = pop.psm_buffer[0].copy()
        pop.days_since_birth[0] = pop.params.nursing_time - 2
        for _ in range(48):
            pop.step()
        pop._grow(5)
        for _ in range(48 * 5):
            if not pop.with_calf[0]:
                break
            pop.step()
        gained = pop.psm_buffer[0] - at_birth
        calves = np.flatnonzero(pop.active_mask[6:]) + 6
        assert calves.size == 1  # every original slot was taken, so the calf got a new one
        np.testing.assert_allclose(pop.psm_buffer[calves[0]], gained, rtol=1e-5)


class TestOtherModes:
    @pytest.mark.parametrize("mode", ["Undirected", "off"])
    def test_calf_starts_with_an_empty_memory(self, mode):
        pop = _pop(mode)
        gained, calf = _birth_then_wean(pop)
        assert gained[..., 0].sum() > 100
        assert not pop.psm_buffer[calf].any()
        pop._calf_memory_consolidate()
        assert pop._calf_keys.size == 0

    def test_off_inherits_the_distance_and_undirected_draws_a_new_one(self):
        pop = _pop("off")
        mothers = pop._psm_instances[0].preferred_distance
        _, calf = _birth_then_wean(pop)
        assert pop._psm_instances[calf].preferred_distance == pytest.approx(mothers)

        pop = _pop("Undirected")
        mothers = pop._psm_instances[0].preferred_distance
        _, calf = _birth_then_wean(pop)
        assert pop._psm_instances[calf].preferred_distance != pytest.approx(mothers)
