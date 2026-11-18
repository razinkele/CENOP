"""The dispersal setting (params.dispersal): validation, off, Undirected."""

import numpy as np
import pytest

import cenop.agents.population as popmod
from cenop.agents.population import PorpoisePopulation
from cenop.movement import JASMINEPhysicsMovement
from cenop.parameters.simulation_params import DISPERSAL_MODES, SimulationParameters


def test_modes_constant():
    assert DISPERSAL_MODES == ("off", "PSM-Type2", "Undirected", "InnerDanishWaters")


@pytest.mark.parametrize("mode", ["off", "PSM-Type2", "Undirected"])
def test_valid_modes_accepted(mode):
    assert SimulationParameters(dispersal=mode).dispersal == mode


def test_inner_danish_waters_accepted_on_kattegat():
    p = SimulationParameters(dispersal="InnerDanishWaters", landscape="Kattegat")
    assert p.dispersal == "InnerDanishWaters"


@pytest.mark.parametrize("landscape", ["Lithuania", "Homogeneous", "NorthSea"])
def test_inner_danish_waters_needs_kattegat(landscape):
    with pytest.raises(ValueError, match="InnerDanishWaters.*Kattegat"):
        SimulationParameters(dispersal="InnerDanishWaters", landscape=landscape)


def test_inner_danish_waters_default_landscape_rejected():
    with pytest.raises(ValueError, match="InnerDanishWaters.*Kattegat"):
        SimulationParameters(dispersal="InnerDanishWaters")


@pytest.mark.parametrize("bad", ["Off", "undirected", " off", "PSM-Typ2", ""])
def test_unknown_mode_lists_valid_values(bad):
    with pytest.raises(ValueError, match="off.*PSM-Type2.*Undirected.*InnerDanishWaters"):
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
        if not popmod._HAS_JAX:  # importable is not enough: a full GPU also disables it
            pytest.skip("JAX backend not usable")
        with pytest.raises(ValueError, match="Undirected.*JAX"):
            _pop(dispersal="Undirected", use_jax=True)

    def test_runs_when_jax_missing(self, monkeypatch):
        monkeypatch.setattr(popmod, "_HAS_JAX", False)
        pop = _pop(dispersal="Undirected", use_jax=True)
        assert pop._use_jax is False

    def test_off_allowed_on_jax(self):
        pytest.importorskip("jax")
        _pop(dispersal="off", use_jax=True)


def test_dispersal_events_file_records_the_mode():
    from types import SimpleNamespace

    from cenop.core.output_writer import OutputConfig, OutputWriter

    pop = _pop(dispersal="Undirected")
    sim = SimpleNamespace(
        population_manager=pop, params=pop.params, state=SimpleNamespace(tick=5, day=0)
    )
    writer = OutputWriter(OutputConfig(output_dir="unused", dispersal=True))
    events = []
    writer._write_dispersal_event = events.append
    writer._prev_dispersing = np.zeros(pop.count, dtype=bool)
    pop.is_dispersing[0] = True
    writer._detect_dispersal(sim)
    assert [e.dispersal_type for e in events] == ["Undirected"]


class TestInnerDanishWatersState:
    """Guards and per-animal state for dispersal="InnerDanishWaters" (wiring is tested in
    test_dispersal_idw_population.py)."""

    def _idw(self, **kw):
        params = SimulationParameters(
            random_seed=1,
            communication_enabled=False,
            dispersal="InnerDanishWaters",
            landscape="Kattegat",
            **kw,
        )
        return PorpoisePopulation(4, params)

    def test_constructs_on_the_default_backend(self):
        assert self._idw()._dispersal_mode == "InnerDanishWaters"

    def test_refused_on_jax(self):
        if not popmod._HAS_JAX:
            pytest.skip("JAX backend not usable")
        with pytest.raises(ValueError, match="InnerDanishWaters.*JAX"):
            self._idw(use_jax=True)

    def test_refused_with_non_crw_movement_module(self):
        params = SimulationParameters(
            random_seed=1,
            communication_enabled=False,
            dispersal="InnerDanishWaters",
            landscape="Kattegat",
        )
        with pytest.raises(ValueError, match="InnerDanishWaters.*movement"):
            PorpoisePopulation(4, params, movement_module=JASMINEPhysicsMovement(params))

    def test_initial_state(self):
        pop = _pop()
        assert pop.dispersal_phase.dtype == np.int8 and not pop.dispersal_phase.any()
        assert pop._daily_pos_x.shape == (pop.count, 10)
        assert (pop._daily_pos_x == 0.5).all() and (pop._daily_pos_y == 0.5).all()

    def test_daily_positions_pushed_at_the_day_boundary(self):
        pop = _pop()
        pop.x[:] = [10.0, 20.0, 30.0, 40.0]
        pop.y[:] = [11.0, 21.0, 31.0, 41.0]
        seen = {}
        pop._check_dispersal_trigger = lambda mask: seen.update(x0=pop._daily_pos_x[:, 0].copy())
        pop._global_tick = 48  # not tick 1, which has its own push
        pop._tick_counter = 47
        pop._update_energy_history(pop.active_mask)
        np.testing.assert_array_equal(seen["x0"], pop.x)  # pushed before the trigger
        np.testing.assert_array_equal(pop._daily_pos_y[:, 0], pop.y)
        assert (pop._daily_pos_x[:, 1] == 0.5).all()
        pop.x += 3.0
        pop._tick_counter = 47
        pop._global_tick += 1
        pop._update_energy_history(pop.active_mask)
        np.testing.assert_array_equal(pop._daily_pos_x[:, 1], [10.0, 20.0, 30.0, 40.0])
        np.testing.assert_array_equal(pop._daily_pos_x[:, 0], [13.0, 23.0, 33.0, 43.0])
        assert (pop._daily_pos_x[:, 2] == 0.5).all()

    def test_first_tick_also_pushes(self):
        # DEPONS runs the daily task once at tick 1 as well (PorpoiseSimBuilder.java:340,355)
        pop = _pop()
        pop.x[:] = 10.0
        pop.y[:] = 11.0
        pop._global_tick = 1
        pop._update_energy_history(pop.active_mask)
        assert (pop._daily_pos_x[:, 0] == 10.0).all() and (pop._daily_pos_y[:, 0] == 11.0).all()
        assert (pop._daily_pos_x[:, 1] == 0.5).all()
        assert not pop._energy_history.any()  # the energy history is not touched at tick 1

    def test_no_push_inside_a_day(self):
        pop = _pop()
        pop.x[:] = 10.0
        pop._tick_counter = 5
        pop._update_energy_history(pop.active_mask)
        assert (pop._daily_pos_x == 0.5).all()

    def test_end_dispersal_clears_the_phase(self):
        pop = _pop()
        pop.is_dispersing[0] = True
        pop.dispersal_phase[0] = 2
        pop._end_dispersal(np.array([0]))
        assert pop.dispersal_phase[0] == 0

    def test_recycled_slot_is_reset(self):
        pop = _pop()
        pop.dispersal_phase[1] = 2
        pop._daily_pos_x[1, :] = 77.0
        pop._daily_pos_y[1, :] = 78.0
        pop._reset_recycled_slots(np.array([1]))
        assert pop.dispersal_phase[1] == 0
        assert (pop._daily_pos_x[1] == 0.5).all() and (pop._daily_pos_y[1] == 0.5).all()
