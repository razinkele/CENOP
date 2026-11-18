"""The dispersal setting (params.dispersal): validation, off, Undirected."""

import numpy as np
import pytest

import cenop.agents.population as popmod
from cenop.agents.population import PorpoisePopulation
from cenop.movement import JASMINEPhysicsMovement
from cenop.parameters.simulation_params import DISPERSAL_MODES, SimulationParameters


def test_modes_constant():
    assert DISPERSAL_MODES == ("off", "PSM-Type2", "Undirected")


@pytest.mark.parametrize("mode", ["off", "PSM-Type2", "Undirected"])
def test_valid_modes_accepted(mode):
    assert SimulationParameters(dispersal=mode).dispersal == mode


def test_inner_danish_waters_not_implemented():
    with pytest.raises(ValueError, match="InnerDanishWaters.*not implemented"):
        SimulationParameters(dispersal="InnerDanishWaters")


@pytest.mark.parametrize("bad", ["Off", "undirected", " off", "PSM-Typ2", ""])
def test_unknown_mode_lists_valid_values(bad):
    with pytest.raises(ValueError, match="off.*PSM-Type2.*Undirected"):
        SimulationParameters(dispersal=bad)


# --- population behaviour -------------------------------------------------------------


def _pop(count=4, landscape=None, **kw):
    params = SimulationParameters(random_seed=1, communication_enabled=False, **kw)
    return PorpoisePopulation(count, params, landscape)


def _disperser(pop, i, x, y, prev_heading, target_dist=500.0, traveled=0.0):
    pop.x[i], pop.y[i] = x, y
    pop.is_dispersing[i] = True
    pop.dispersal_start_x[i], pop.dispersal_start_y[i] = x, y
    pop.dispersal_target_distance[i] = target_dist
    pop.dispersal_distance_traveled[i] = traveled
    pop._prev_step_heading[i] = prev_heading
    pop.heading[i] = prev_heading


def _declining(pop):
    # newest day at index 0; strictly dropping for t_disp=3 days
    pop._energy_history[:, :4] = np.array([7.0, 8.0, 9.0, 10.0])


class TestOff:
    def test_type2_control_starts_dispersal(self):
        pop = _pop(count=20, dispersal="PSM-Type2")
        _declining(pop)
        pop._check_dispersal_trigger(pop.active_mask)
        assert pop.is_dispersing.any()

    def test_off_starts_none(self):
        pop = _pop(count=20, dispersal="off")
        _declining(pop)
        pop._check_dispersal_trigger(pop.active_mask)
        assert not pop.is_dispersing.any()

    def test_off_lets_existing_dispersers_finish(self):
        pop = _pop(dispersal="off", psm_angle=0.0)
        _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=500.0, traveled=0.0)
        pop._disperse_step(pop.active_mask)
        assert pop.is_dispersing[0]
        pop.dispersal_distance_traveled[0] = 0.95 * 500.0
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
