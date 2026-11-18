"""DEPONS InnerDanishWatersDispersal port (cenop.agents.dispersal_idw). Java coordinates."""

import math

import numpy as np
import pytest

from cenop.agents import dispersal_idw as idw

W, H = 600, 1000


@pytest.fixture(scope="session")
def kattegat():
    from cenop.landscape.cell_data import CellData

    cd = CellData("Kattegat", "data")
    cd.load()
    return cd


def test_java_round_is_half_up():
    assert idw.java_round(2.5) == 3.0
    assert idw.java_round(3.5) == 4.0
    assert idw.java_round(-0.5) == 0.0


def test_tables_match_java():
    assert idw.BLOCK_CENTRES_X[:7].tolist() == [50, 150, 250, 350, 450, 550, 50]
    assert idw.BLOCK_CENTRES_Y[[0, 6, 59]].tolist() == [950, 850, 50]
    assert idw.BLOCK_VAL_KAT_1[[0, 17, 38, 59]].tolist() == [0, 0.4446, 1, 0]
    assert len(idw.BLOCK_VAL_KAT_1) == 60
    assert idw.BLOCK_VAL_KAT_1.sum() == pytest.approx(21.0125)


def test_cell_of_rounds_and_clamps():
    assert idw.cell_of(438.4, 477.5, W, H) == (438, 478)
    assert idw.cell_of(599.6, 999.7, W, H) == (599, 999)


def test_kattegat_nav_block_numbering(kattegat):
    # Java coords (150, 475) lie in block index 31 (centre 150, 450); the grid holds index + 1
    assert idw.grid_at(kattegat._blocks.astype(np.float64), 150.0, 475.0) == 32


def test_point_at_interior():
    px, py = idw.point_at(10.0, 10.0, 90.0, 5.0, W, H)
    assert (px, py) == pytest.approx((15.0, 10.0), abs=1e-9)
    px, py = idw.point_at(10.0, 10.0, 0.0, 5.0, W, H)
    assert (px, py) == pytest.approx((10.0, 15.0), abs=1e-9)


@pytest.mark.parametrize(
    "x, y, heading, expected",
    [
        (598.0, 10.0, 90.0, (596.0, 10.0)),  # 603 mirrors about 599.5
        (1.0, 10.0, 270.0, (3.0, 10.0)),  # -4 mirrors about -0.5
        (10.0, 998.0, 0.0, (10.0, 996.0)),  # 1003 mirrors about 999.5
        (10.0, 1.0, 180.0, (10.0, 3.0)),
    ],
)
def test_point_at_bounces_off_each_border(x, y, heading, expected):
    assert idw.point_at(x, y, heading, 5.0, W, H) == pytest.approx(expected, abs=1e-9)


def _blocks(value):
    return np.full((H, W), float(value))


def _reference_top12(xj, yj):
    q = idw.BLOCK_VAL_KAT_1.copy()
    for n in range(60):
        d = math.floor(math.hypot(xj - idw.BLOCK_CENTRES_X[n], yj - idw.BLOCK_CENTRES_Y[n]) + 0.5)
        if d < 250:
            q[n] = 0
    s = np.sort(q)
    out = []
    for i in range(12):
        hit = 0
        for j in range(60):
            if q[j] == s[59 - i]:
                hit = j
        out.append(hit)
    return out


def test_select_target_excludes_blocks_within_100km():
    # At the centre of block 38 (quality 1.0, the best) that block is excluded
    for u in np.linspace(0, 0.999, 24):
        assert idw.select_target(250.0, 350.0, u, _blocks(39)) != (250.0, 350.0)


def test_select_target_draw_index_and_last_match():
    expected = _reference_top12(0.0, 0.0)
    for k in range(12):
        got = idw.select_target(0.0, 0.0, (k + 0.5) / 12, _blocks(55))
        assert got == (idw.BLOCK_CENTRES_X[expected[k]], idw.BLOCK_CENTRES_Y[expected[k]])


def test_select_target_east_shift():
    xj, yj = 250.0, 900.0
    top = _reference_top12(xj, yj)
    assert 30 in top
    u = (top.index(30) + 0.5) / 12
    assert idw.select_target(xj, yj, u, _blocks(3)) == (250.0, 450.0)  # index 32
    assert idw.select_target(xj, yj, u, _blocks(20)) == (50.0, 450.0)  # index 30, no shift


# --- phase 1 / phase 2 -----------------------------------------------------------------

ANGLES17 = list(range(-80, 81, 10))


def grids(depth=30.0, dtc=10000.0, block=40.0):
    return np.full((H, W), depth), np.full((H, W), dtc), np.full((H, W), block)


def run(g, phase=1, pos=(300.0, 500.0), heading=0.0, target=(300.0, 800.0), **kw):
    d1 = kw.get("d1", (300.0, 100.0))
    d8 = kw.get("d8", (300.0, 100.0))
    depth, dtc, blocks = g
    return idw.disperse(
        phase, pos[0], pos[1], heading, target[0], target[1], d1[0], d1[1], d8[0], d8[1],
        kw.get("ew", True), depth, dtc, blocks, 2.0, 4.0, 1.0,
    )  # fmt: skip


def cell(x, y, heading, dist):
    return idw.cell_of(*idw.point_at(x, y, heading, dist, W, H), W, H)


def moved(res, pos=(300.0, 500.0)):
    return math.hypot(res[1] - pos[0], res[2] - pos[1])


class TestPhase1:
    def test_south_east_of_sealand_switches_and_moves_in_phase_2(self):
        res = run(grids(), pos=(450.0, 400.0))
        assert res[0] == 2 and res[4] == 1
        assert moved(res, (450.0, 400.0)) == pytest.approx(5.0)

    @pytest.mark.parametrize("block", [14.0, 31.0])
    def test_nav_blocks_14_and_31_switch(self, block):
        res = run(grids(block=block))
        assert res[0] == 2 and res[4] == 1 and moved(res) == pytest.approx(5.0)

    def test_nav_block_32_goes_one_cell_north_then_phase_2(self):
        g = grids(block=32.0)
        res = run(g, heading=123.0)
        after_north = (300.0, 501.0)
        expected = run(g, phase=2, pos=after_north, heading=0.0)
        assert res == expected
        assert res[0] == 2 and res[4] == 1

    def test_open_water_turns_twice_and_moves(self):
        res = run(grids(), heading=77.0)
        ex, ey = idw.point_at(300.0, 500.0, 50.0, 5.0, W, H)
        assert res == (1, pytest.approx(ex), pytest.approx(ey), 50.0, 1)

    def test_turns_to_deepest_of_five(self):
        g = grids()
        cx, cy = cell(300.0, 500.0, -20.0, 2.0)
        assert (cx, cy) != cell(300.0, 500.0, -10.0, 2.0)
        g[0][cy, cx] = 50.0
        assert run(g)[3] == 10.0  # -20 (deepest) then +30 (coast, last tie)

    def test_coast_turn_filtered_by_depth_read_before_the_first_turn(self):
        g = grids()
        cx, cy = cell(300.0, 500.0, 30.0, 16.0)  # +30 relative to the target-facing heading
        g[0][cy, cx] = -9999.0
        assert run(g)[3] == 40.0  # +20, then +20 because the +30 candidate is discarded

    def test_no_water_far_ahead_gives_plus_40(self):
        g = grids()
        for a in range(-30, 31, 10):
            cx, cy = cell(300.0, 500.0, a, 16.0)
            g[0][cy, cx] = -9999.0
        assert run(g)[3] == 60.0  # +20, then the trailing padding entry: +40

    @pytest.mark.parametrize(
        "kw",
        [
            {"d1": (300.0, 499.0)},
            {"d8": (300.0, 495.0)},
            {"target": (300.0, 530.0)},
        ],
    )
    def test_switches_to_phase_2_and_moves_in_the_same_call(self, kw):
        res = run(grids(), **kw)
        assert res[0] == 2 and res[4] == 1 and moved(res) == pytest.approx(5.0)

    def test_close_to_coast_switches(self):
        g = grids()
        g[1][500, 300] = 1500.0
        res = run(g)
        assert res[0] == 2 and res[4] == 1

    @pytest.mark.parametrize("dist", [4.0, 6.0, 8.0])
    def test_land_ahead_switches(self, dist):
        g = grids()
        cx, cy = cell(300.0, 500.0, 50.0, dist)
        g[0][cy, cx] = 0.0
        res = run(g)
        assert res[0] == 2 and res[4] == 1

    def test_not_enough_water_stops_without_moving(self):
        assert run(grids(), ew=False) == (0, 300.0, 500.0, 50.0, 0)

    def test_not_enough_water_overrides_a_switch(self):
        assert run(grids(), ew=False, d1=(300.0, 499.0)) == (0, 300.0, 500.0, 50.0, 0)

    def test_landing_on_land_steps_back_and_stops(self):
        g = grids()
        cx, cy = cell(300.0, 500.0, 50.0, 5.0)
        g[0][cy, cx] = -1.0
        res = run(g)
        assert res == (0, pytest.approx(300.0), pytest.approx(500.0), 50.0, 1)

    def test_landing_on_depth_zero_is_accepted(self):
        g = grids()
        cx, cy = cell(300.0, 500.0, 50.0, 5.0)
        g[0][cy, cx] = 0.0
        res = run(g)
        assert res[0] == 1 and moved(res) == pytest.approx(5.0)

    def test_origin_filled_daily_buffer_does_not_switch(self):
        assert run(grids(), d1=(0.0, 0.0), d8=(0.0, 0.0))[0] == 1


class TestPhase2:
    def test_faces_away_from_yesterday_then_smallest_difference_last_tie(self):
        res = run(grids(), phase=2, heading=200.0, d1=(300.0, 490.0))
        ex, ey = idw.point_at(300.0, 500.0, 80.0, 5.0, W, H)
        assert res == (2, pytest.approx(ex), pytest.approx(ey), 80.0, 1)

    def test_facing_is_skipped_when_yesterday_is_here(self):
        res = run(grids(), phase=2, heading=200.0, d1=(300.0, 500.0))
        assert res[3] == (200.0 + 180.0 + 80.0) % 360.0

    def test_middle_band_picks_most_alike(self):
        g = grids(dtc=9000.0)
        g[1][500, 300] = 3000.0
        target = cell(300.0, 500.0, -40.0, 2.0)
        g[1][target[1], target[0]] = 3000.0
        hits = [a for a in ANGLES17 if cell(300.0, 500.0, a, 2.0) == target]
        assert -40 in hits
        res = run(g, phase=2, d1=(300.0, 490.0))
        assert res[3] == hits[-1] % 360  # last tie

    def test_near_coast_picks_most_different_first_tie(self):
        g = grids(dtc=600.0)
        g[1][500, 300] = 500.0
        for a in (10.0, 60.0):
            cx, cy = cell(300.0, 500.0, a, 2.0)
            g[1][cy, cx] = 9000.0
        first = next(a for a in ANGLES17 if g[1][cell(300.0, 500.0, a, 2.0)[::-1]] == 9000.0)
        res = run(g, phase=2, d1=(300.0, 490.0))
        assert res[3] == first % 360

    def test_sentinel_band(self):
        g = grids()
        g[1][500, 300] = 1500.0
        res = run(g, phase=2, d1=(300.0, 490.0))
        assert res[3] == 81.0 and res[0] == 2  # -9999 degrees
        g[1][500, 300] = 1000.0
        assert run(g, phase=2, d1=(300.0, 490.0))[3] == 81.0
        g[1][500, 300] = 2000.0
        assert run(g, phase=2, d1=(300.0, 490.0))[3] != 81.0

    def test_not_enough_water_stops(self):
        assert run(grids(), phase=2, d1=(300.0, 490.0), ew=False) == (0, 300.0, 500.0, 80.0, 0)

    def test_shallow_ahead_stops(self):
        g = grids()
        cx, cy = cell(300.0, 500.0, 80.0, 5.0)
        g[0][cy, cx] = 1.0  # not above min_depth
        assert run(g, phase=2, d1=(300.0, 490.0)) == (0, 300.0, 500.0, 80.0, 0)


def test_not_dispersing_is_a_no_op():
    assert run(grids(), phase=0, heading=12.0) == (0, 300.0, 500.0, 12.0, 0)


def test_numba_and_python_agree(kattegat):
    if not idw.HAS_NUMBA:
        pytest.skip("Numba not installed")
    depth = np.ascontiguousarray(kattegat._depth, dtype=np.float64)
    dtc = np.ascontiguousarray(kattegat._dist_to_coast, dtype=np.float64)
    blocks = np.ascontiguousarray(kattegat._blocks, dtype=np.float64)
    rng = np.random.default_rng(7)
    ys, xs = np.nonzero(depth > 0)
    phases = set()
    for _ in range(3000):
        k = rng.integers(len(xs))
        args = (
            int(rng.integers(1, 3)), xs[k] + rng.uniform(-0.5, 0.5), ys[k] + rng.uniform(-0.5, 0.5),
            rng.uniform(0, 360), float(rng.choice(idw.BLOCK_CENTRES_X)),
            float(rng.choice(idw.BLOCK_CENTRES_Y)), xs[k] + rng.normal(0, 4), ys[k] + rng.normal(0, 4),
            xs[k] + rng.normal(0, 10), ys[k] + rng.normal(0, 10), bool(rng.random() < 0.9),
            depth, dtc, blocks, 2.0, 4.0, 1.0,
        )  # fmt: skip
        a = idw.disperse(*args)
        assert a == idw.disperse.py_func(*args)
        phases.add(a[0])
    assert phases == {0, 1, 2}
