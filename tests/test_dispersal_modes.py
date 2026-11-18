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


T = 500.0


def _turns(mode):
    pop = _pop(count=2000, dispersal=mode)
    for i in range(2000):
        _disperser(pop, i, 100.0, 100.0, 90.0, target_dist=T, traveled=0.0)
    before = pop._prev_step_heading.copy()
    pop._apply_dispersal_heading(np.ones(2000, dtype=bool))
    return (pop.heading - before + 180) % 360 - 180


class TestUndirected:
    """DEPONS UndirectedDispersal (extends DispersalPSMType3)."""

    def test_ignores_memory(self, monkeypatch):
        pop = _pop(dispersal="Undirected")
        monkeypatch.setattr(pop, "_memory_dispersal_target", lambda idx: (1.0, 2.0))
        monkeypatch.setattr(pop, "_random_dispersal_target", lambda idx: (30.0, 40.0))
        assert pop._select_dispersal_target(0) == (30.0, 40.0)

    def test_type2_still_prefers_memory(self, monkeypatch):
        pop = _pop(dispersal="PSM-Type2")
        monkeypatch.setattr(pop, "_memory_dispersal_target", lambda idx: (1.0, 2.0))
        assert pop._select_dispersal_target(0) == (1.0, 2.0)

    def test_turn_is_undamped(self):
        # At distance 0 the Type2 logistic is 1 / (1 + exp(1.5 / 0.6)) = 0.0759
        t2, und = _turns("PSM-Type2"), _turns("Undirected")
        assert np.abs(t2).max() <= 40 * 0.08
        assert 35 < np.abs(und).max() <= 40
        np.testing.assert_allclose(t2, und * 0.0759, atol=0.05)  # same draws, same seed

    def test_stops_on_displacement_not_path(self):
        pop = _pop(dispersal="Undirected", psm_angle=0.0)
        _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=T, traveled=0.99 * T)
        _disperser(pop, 1, 100.0, 100.0, 90.0, target_dist=T)
        pop.x[1] = 100.0 + T  # displacement reached
        _disperser(pop, 2, 100.0, 100.0, 90.0, target_dist=T)
        pop.x[2] = 100.0 + 0.97 * T  # past Type2's 0.95, short of the full distance
        pop._disperse_step(pop.active_mask)
        assert pop.is_dispersing[0]  # long path, no displacement
        assert not pop.is_dispersing[1]
        assert pop.is_dispersing[2]

    def test_generic_module_path_uses_full_distance(self):
        params = SimulationParameters(
            random_seed=1, communication_enabled=False, dispersal="Undirected"
        )
        pop = PorpoisePopulation(4, params, movement_module=JASMINEPhysicsMovement(params))
        _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=T)
        pop.x[0] = 100.0 + 0.97 * T
        _disperser(pop, 1, 100.0, 300.0, 90.0, target_dist=T)
        pop.x[1] = 100.0 + T
        pop._update_dispersal(pop.active_mask)
        assert pop.is_dispersing[0]
        assert not pop.is_dispersing[1]

    def test_zero_target_completes_without_nan(self):
        pop = _pop(dispersal="Undirected")
        _disperser(pop, 0, 100.0, 100.0, 90.0, target_dist=0.0)
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0]
        assert np.isfinite(pop.heading).all()

    def test_failed_target_leaves_animal_untouched(self, monkeypatch):
        pop = _pop(dispersal="Undirected")
        monkeypatch.setattr(pop, "_random_dispersal_target", lambda idx: None)
        h = pop.heading[0]
        assert pop._start_dispersal(0) is False
        assert not pop.is_dispersing[0]
        assert pop.heading[0] == h

    def test_refused_on_jax(self):
        pytest.importorskip("jax")
        with pytest.raises(ValueError, match="Undirected.*JAX"):
            _pop(dispersal="Undirected", use_jax=True)

    def test_runs_when_jax_missing(self, monkeypatch):
        monkeypatch.setattr(popmod, "_HAS_JAX", False)
        pop = _pop(dispersal="Undirected", use_jax=True)
        assert pop._use_jax is False

    def test_off_allowed_on_jax(self):
        pytest.importorskip("jax")
        _pop(dispersal="off", use_jax=True)
