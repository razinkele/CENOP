"""dispersal="InnerDanishWaters" wired into the population tick (Kattegat landscape)."""

import copy

import numpy as np
import pytest

from cenop.agents import dispersal_idw as idw
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters

py_disperse = idw.disperse.py_func if idw.HAS_NUMBA else idw.disperse


@pytest.fixture(scope="module")
def kattegat():
    cd = CellData("Kattegat", "data")
    cd.load()
    return cd


@pytest.fixture(scope="module")
def open_cells(kattegat):
    """Cells (x, y) in open, deep water away from the mode's geographic special cases."""
    depth, dtc, blocks = kattegat._depth, kattegat._dist_to_coast, kattegat._blocks
    ok = (depth > 25) & (dtc > 8000) & ~np.isin(blocks, [14, 31, 32])
    ys, xs = np.nonzero(ok)
    keep = ~((xs > 438) & (ys < 478)) & (xs > 50) & (xs < 550) & (ys > 50) & (ys < 950)
    cells = list(zip(xs[keep].tolist(), ys[keep].tolist()))
    assert len(cells) > 100
    return cells[:: len(cells) // 8][:8]


def make_pop(kattegat, count=4, **kw):
    params = SimulationParameters(
        random_seed=1,
        communication_enabled=False,
        dispersal="InnerDanishWaters",
        landscape="Kattegat",
        **kw,
    )
    pop = PorpoisePopulation(count, params, kattegat)
    return pop


def place(pop, i, cell, phase, heading=30.0, target=(250.5, 350.5)):
    """Put animal i at the centre of ``cell`` (CENOP coordinates) in the given phase."""
    pop.x[i], pop.y[i] = cell[0] + 0.5, cell[1] + 0.5
    pop.heading[i] = heading
    pop.is_dispersing[i] = phase > 0
    pop.dispersal_phase[i] = phase
    pop.dispersal_target_x[i], pop.dispersal_target_y[i] = target
    pop._daily_pos_x[i, :] = cell[0] + 0.5
    pop._daily_pos_y[i, :] = cell[1] + 0.5 - 100.0  # far away: no "moved too little" switch
    pop._enough_water_ahead[i] = True


def tick(pop):
    mask = pop.active_mask
    pop._recompute_cell_indices()
    pop._update_movement(mask, None, None)
    pop._handle_land_avoidance(mask)
    pop._apply_positions(mask)


def expected(pop, kattegat, i):
    """The module's answer for animal i's current state, back in CENOP coordinates."""
    f = np.float64
    res = py_disperse(
        int(pop.dispersal_phase[i]),
        float(pop.x[i]) - 0.5,
        float(pop.y[i]) - 0.5,
        float(pop.heading[i]),
        float(pop.dispersal_target_x[i]) - 0.5,
        float(pop.dispersal_target_y[i]) - 0.5,
        float(pop._daily_pos_x[i, 1]) - 0.5,
        float(pop._daily_pos_y[i, 1]) - 0.5,
        float(pop._daily_pos_x[i, 8]) - 0.5,
        float(pop._daily_pos_y[i, 8]) - 0.5,
        bool(pop._enough_water_ahead[i]),
        np.ascontiguousarray(kattegat._depth, dtype=f),
        np.ascontiguousarray(kattegat._dist_to_coast, dtype=f),
        np.ascontiguousarray(kattegat._blocks, dtype=f),
        pop.params.mean_disp_dist,
        pop.params.min_depth_dispersal,
        pop.params.min_depth,
    )
    return res[0], res[1] + 0.5, res[2] + 0.5, res[3]


class TestStart:
    def test_start_picks_a_block_centre_and_enters_phase_1(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 0)
        twin = copy.deepcopy(pop.rng)
        assert pop._start_dispersal(0) is True
        twin.random()
        assert twin.bit_generator.state == pop.rng.bit_generator.state  # exactly one draw
        assert pop.is_dispersing[0] and pop.dispersal_phase[0] == 1
        tx, ty = pop.dispersal_target_x[0] - 0.5, pop.dispersal_target_y[0] - 0.5
        assert tx in idw.BLOCK_CENTRES_X and ty in idw.BLOCK_CENTRES_Y
        dist = np.hypot(pop.dispersal_target_x[0] - pop.x[0], pop.dispersal_target_y[0] - pop.y[0])
        assert pop.dispersal_target_distance[0] == pytest.approx(dist, rel=1e-5)
        assert dist >= 249.5  # blocks nearer than 100 km are never chosen
        assert (pop.dispersal_start_x[0], pop.dispersal_start_y[0]) == (pop.x[0], pop.y[0])

    def test_trigger_starts_this_mode(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        for i in range(4):
            place(pop, i, open_cells[i], 0)
        pop._energy_history[:, :4] = np.array([7.0, 8.0, 9.0, 10.0])
        pop._check_dispersal_trigger(pop.active_mask)
        assert (pop.dispersal_phase == 1).all() and pop.is_dispersing.all()


class TestTick:
    @pytest.mark.parametrize("phase", [1, 2])
    def test_position_and_heading_follow_the_module(self, kattegat, open_cells, phase):
        pop = make_pop(kattegat, count=8)
        for i in range(8):
            place(pop, i, open_cells[i], phase, heading=40.0 * i)
        want = [expected(pop, kattegat, i) for i in range(8)]
        tick(pop)
        for i, (ph, x, y, hd) in enumerate(want):
            assert pop.dispersal_phase[i] == ph
            assert bool(pop.is_dispersing[i]) is (ph > 0)
            if ph > 0:  # stopped animals go on to a CRW step, checked elsewhere
                assert (pop.x[i], pop.y[i]) == pytest.approx((x, y), abs=1e-3)
                assert pop.heading[i] == pytest.approx(hd, abs=1e-3)
        assert sum(w[0] > 0 for w in want) >= 6  # the fixture is open water

    def test_landing_on_depth_zero_is_not_rolled_back(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 1)
        ph, x, y, _ = expected(pop, kattegat, 0)
        assert ph == 1
        yi, xi = int(y), int(x)
        saved = kattegat._depth[yi, xi]
        kattegat._depth[yi, xi] = 0.0
        pop._idw_grids = None  # drop the cached float64 copy
        try:
            tick(pop)
        finally:
            kattegat._depth[yi, xi] = saved
        assert pop.dispersal_phase[0] == 1
        assert (int(pop.x[0]), int(pop.y[0])) == (xi, yi)

    def test_stop_keeps_the_turned_heading_and_takes_a_crw_step(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 1, heading=200.0)
        x0, y0 = float(pop.x[0]), float(pop.y[0])
        pop._enough_water_ahead[0] = False
        ph, _, _, turned = expected(pop, kattegat, 0)
        assert ph == 0
        pop._disperse_step(pop.active_mask)
        assert not pop.is_dispersing[0] and pop.dispersal_phase[0] == 0
        assert pop.heading[0] == pytest.approx(turned, abs=1e-3)  # not restored to 200
        assert (float(pop.x[0]), float(pop.y[0])) == (x0, y0)
        assert not pop._idw_pending[0]
        place(pop, 1, open_cells[1], 1)
        pop._enough_water_ahead[1] = False
        tick(pop)
        assert not pop.is_dispersing[1]
        assert pop._step_dist[1] != pytest.approx(5.0)  # a CRW step, not the dispersal step

    def test_path_length_plays_no_part(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 1)
        pop.dispersal_target_distance[0] = 10.0
        pop.dispersal_distance_traveled[0] = 500.0
        tick(pop)
        assert pop.is_dispersing[0]

    def test_never_reported_complete_to_the_fsm(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 1)
        pop.dispersal_target_distance[0] = 10.0
        pop.dispersal_distance_traveled[0] = 500.0
        assert not pop._dispersal_complete_mask().any()

    def test_deterrence_and_recovery_clear_the_phase(self, kattegat, open_cells):
        pop = make_pop(kattegat)
        place(pop, 0, open_cells[0], 1)
        place(pop, 1, open_cells[1], 2)
        pop._turbine_deter_strength[0] = 1.0
        pop._energy_history[1, :8] = [9.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
        pop._tick_counter = 0
        pop._global_tick = 48
        pop._update_dispersal(pop.active_mask)
        assert not pop.is_dispersing[:2].any() and not pop.dispersal_phase[:2].any()


def test_psm_complete_mask_unchanged_for_other_modes():
    params = SimulationParameters(random_seed=1, communication_enabled=False)
    pop = PorpoisePopulation(4, params)
    pop.is_dispersing[0] = True
    pop.dispersal_target_distance[0] = 10.0
    pop.dispersal_distance_traveled[0] = 500.0
    assert pop._dispersal_complete_mask().tolist() == [True, False, False, False]


def test_thirty_days_on_kattegat(kattegat):
    from cenop.core.simulation import Simulation

    params = SimulationParameters(
        porpoise_count=400,
        sim_years=1,
        random_seed=42,
        landscape="Kattegat",
        ships_enabled=False,
        turbines="off",
        dispersal="InnerDanishWaters",
    )
    sim = Simulation(params=params, cell_data=kattegat, seed=42)
    sim.initialize()
    pm = sim.population_manager
    seen = set()
    for _ in range(30 * 48):
        sim.step()
        seen.update(np.unique(pm.dispersal_phase[pm.active_mask]).tolist())
    a = pm.active_mask
    assert seen == {0, 1, 2}
    for arr in (pm.x, pm.y, pm.heading, pm.energy):
        assert np.isfinite(arr[a]).all()
    assert ((pm.x[a] >= 0) & (pm.x[a] < 600) & (pm.y[a] >= 0) & (pm.y[a] < 1000)).all()
    assert (pm.dispersal_phase[a] > 0).tolist() == pm.is_dispersing[a].tolist()


def test_border_reflection_leaves_a_dispersing_animals_heading_alone(kattegat):
    # The mode bounces inside its own move (Java range -0.5 .. W - 0.5, CENOP 0 .. W) and
    # sets the heading; CENOP's reflection (range 0 .. W - 1) must not rewrite it.
    pop = make_pop(kattegat)
    w = kattegat.width
    pop.x[0], pop.y[0] = w - 3.0, 500.5
    pop.heading[0] = 77.0
    pop.is_dispersing[0] = True
    pop.dispersal_phase[0] = 1
    pop._idw_pending[0] = True
    pop._idw_x[0], pop._idw_y[0] = w - 0.2, 501.0
    pop._idw_write_step()
    pop._handle_land_avoidance(pop.active_mask)
    assert pop.heading[0] == 77.0


def test_stop_with_a_move_eats_at_the_start_of_tick_cell(kattegat, open_cells, monkeypatch):
    """DEPONS eats at posList.get(1), the position before disperse() moved the animal."""
    pop = make_pop(kattegat)
    place(pop, 0, open_cells[0], 2)
    x0, y0 = float(pop.x[0]), float(pop.y[0])

    def stop_three_cells_east(phase, xj, yj, heading, *rest):
        return 0, xj + 3.0, yj, heading, 1

    monkeypatch.setattr(idw, "disperse", stop_three_cells_east)
    tick(pop)
    assert not pop.is_dispersing[0]
    assert (float(pop._pre_move_x[0]), float(pop._pre_move_y[0])) == (x0, y0)
    assert (int(pop._pre_cell_xi[0]), int(pop._pre_cell_yi[0])) == (int(x0), int(y0))
    x1, y1 = float(pop.x[0]), float(pop.y[0])
    tick(pop)
    assert (float(pop._pre_move_x[0]), float(pop._pre_move_y[0])) == (x1, y1)
