"""Tests for the DEPONS land-avoidance kernel (`cenop.movement.land_check`).

DEPONS 3.2 Porpoise.stdMove 467-533 (100 rounds of checkDepth/avoidLand, then beh 5),
checkDepth 795-833, avoidLandTurn 885-904, avoidLand 918-985 (tiers and beh 4).

Every test runs twice: through the interpreted implementation (`land_check._py`, with the
helpers swapped for their pure-Python versions, NumPy's global RNG seeded) and through the
compiled one (`land_check.depons_land_check`, Numba's RNG seeded).

Shared geometry: agent at (30.5, 30.5) heading 0 (north, +y), presMov = 6, so the path cells
are (30, 31..36). First-tier escape points (+-40..50 deg, 6 cells) fall in x in {25, 26} or
{34, 35}, y in {34, 35}; second-tier points (+-70..80 deg) in x in {24} or {36}, y in {31, 32}.
These cell sets do not depend on the jitter.
"""

import math

import numpy as np
import pytest

import cenop.agents.population as popmod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.movement import land_check
from cenop.parameters.simulation_params import SimulationParameters

LOG6 = math.log10(6.0)
SEEDS = (0, 1, 2, 3)


@pytest.fixture(params=["py", "numba"])
def run(request, monkeypatch):
    """Return a callable running one implementation with its RNG seeded."""
    if request.param == "numba":
        if not land_check.HAS_NUMBA:
            pytest.skip("Numba not installed")
        from cenop.optimizations.kernels import seed_numba_rng

        def _run(a, depth, wmin=1.0, seed=0):
            seed_numba_rng(seed)
            land_check.depons_land_check(*_args(a, depth, wmin))

        return _run

    _interpret_helpers(monkeypatch)

    def _run(a, depth, wmin=1.0, seed=0):
        np.random.seed(seed)
        land_check._py(*_args(a, depth, wmin))

    return _run


def _interpret_helpers(monkeypatch):
    """Swap the compiled helpers for their Python functions so `_py` runs fully interpreted."""
    for name in ("_depth_at", "_reflect", "_look_depth", "_path_ok"):
        f = getattr(land_check, name)
        monkeypatch.setattr(land_check, name, getattr(f, "py_func", f))


def _grid(h=60, w=60, depth=30.0):
    return np.full((h, w), depth, dtype=np.float64)


def _agents(xs, ys, hs, log_movs, mask=None, mem_size=8):
    n = len(xs)
    return {
        "x": np.array(xs, dtype=np.float32),
        "y": np.array(ys, dtype=np.float32),
        "heading": np.array(hs, dtype=np.float32),
        "log_mov": np.array(log_movs, dtype=np.float64),
        "mask": np.ones(n, dtype=bool) if mask is None else np.array(mask, dtype=bool),
        "pos_hist_x": np.zeros((n, mem_size), dtype=np.float32),
        "pos_hist_y": np.zeros((n, mem_size), dtype=np.float32),
        "mem_ptr": np.zeros(n, dtype=np.int32),
        "mem_count": np.zeros(n, dtype=np.int32),
        "out_enough": np.zeros(n, dtype=bool),
    }


def _args(a, depth, wmin):
    return (
        a["x"],
        a["y"],
        a["heading"],
        a["log_mov"],
        a["mask"],
        depth,
        wmin,
        a["pos_hist_x"],
        a["pos_hist_y"],
        a["mem_ptr"],
        a["mem_count"],
        a["out_enough"],
    )


def _one(x=30.5, y=30.5, h=0.0, log_mov=LOG6, mem_size=8):
    return _agents([x], [y], [h], [log_mov], mem_size=mem_size)


def _remember(a, i, entries, mem_ptr):
    """Store `entries` (oldest first) ending just before index `mem_ptr` in the ring."""
    size = a["pos_hist_x"].shape[1]
    for k, (px, py) in enumerate(entries):
        b = (mem_ptr - len(entries) + k) % size
        a["pos_hist_x"][i, b] = px
        a["pos_hist_y"][i, b] = py
    a["mem_ptr"][i] = mem_ptr % size
    a["mem_count"][i] = len(entries)


def _dh(new, old):
    return (float(new) - float(old) + 180.0) % 360.0 - 180.0


def test_open_water_passes_unchanged(run):
    a = _one()
    run(a, _grid())
    assert a["heading"][0] == np.float32(0.0)
    assert a["x"][0] == np.float32(30.5) and a["y"][0] == np.float32(30.5)
    assert a["log_mov"][0] == LOG6
    assert a["out_enough"][0]


def test_lookahead_sees_land_four_steps_ahead(run):
    depth = _grid()
    depth[33, 30] = -5.0  # 2.5 cells ahead: the one-step destination (30.5, 32.0) is water
    assert depth[int(30.5 + 6.0 / 4.0), 30] > 0
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        assert abs(_dh(a["heading"][0], 0.0)) >= 40.0
        assert a["out_enough"][0]
        assert a["x"][0] == np.float32(30.5) and a["y"][0] == np.float32(30.5)
        assert a["log_mov"][0] == LOG6


def test_path_threshold_is_depth_above_zero(run):
    depth = _grid()
    depth[33, 30] = 0.0
    a = _one()
    run(a, depth)
    assert abs(_dh(a["heading"][0], 0.0)) >= 40.0

    depth[33, 30] = 0.5  # below wmin, but the path threshold is > 0
    a = _one()
    run(a, depth)
    assert a["heading"][0] == np.float32(0.0)
    assert a["out_enough"][0]


def test_ahead_point_threshold_is_depth_at_least_zero(run):
    # The ahead point only matters in a cell the 0.1-spaced samples skip. Heading 45 from
    # (30.5, 30.45) clips the corner of cell (31, 30) for < 0.1 cells; presMov = 0.7354 puts
    # the ahead point (31.02, 30.97) there while the 8 samples see only (30, 30) and (31, 31).
    x0, y0, pm = 30.5, 30.45, 0.7354
    r = math.radians(45.0)
    ys = float(np.float32(y0))
    assert (math.floor(x0 + math.sin(r) * pm), math.floor(ys + math.cos(r) * pm)) == (31, 30)
    samples = {
        (math.floor(x0 + 0.1 * k * math.sin(r)), math.floor(ys + 0.1 * k * math.cos(r)))
        for k in range(1, math.ceil(pm / 0.1) + 1)
    }
    assert samples == {(30, 30), (31, 31)}

    depth = _grid()
    depth[30, 31] = 0.0  # ahead point at depth 0: >= 0 passes
    a = _one(x=x0, y=y0, h=45.0, log_mov=math.log10(pm))
    run(a, depth)
    assert a["heading"][0] == np.float32(45.0)
    assert a["out_enough"][0]

    depth[30, 31] = -0.5  # ahead point on land: turn
    a = _one(x=x0, y=y0, h=45.0, log_mov=math.log10(pm))
    run(a, depth)
    assert abs(_dh(a["heading"][0], 45.0)) >= 40.0
    assert a["out_enough"][0]


def test_turns_toward_deeper_side(run):
    depth = _grid()
    depth[:, 33:] = 20.0
    depth[:, :28] = 5.0
    depth[33, 30] = -5.0
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        d = _dh(a["heading"][0], 0.0)
        assert 40.0 <= d < 50.0
        assert a["out_enough"][0]

    depth[:, 33:] = 5.0
    depth[:, :28] = 20.0
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        d = _dh(a["heading"][0], 0.0)
        assert -50.0 < d <= -40.0
        assert a["out_enough"][0]


def test_tie_turns_right(run):
    depth = _grid()
    depth[33, 30] = -5.0
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        assert 40.0 <= _dh(a["heading"][0], 0.0) < 50.0


def test_only_left_good_turns_left(run):
    depth = _grid()
    depth[33, 30] = -5.0
    depth[34:36, 34:36] = -5.0  # right first-tier cells are land
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        assert -50.0 < _dh(a["heading"][0], 0.0) <= -40.0


def test_second_tier(run):
    depth = _grid()
    depth[33, 30] = -5.0
    for cx in (25, 26, 34, 35):
        for cy in (34, 35):
            depth[cy, cx] = 0.5  # below wmin: not good escape points
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        d = abs(_dh(a["heading"][0], 0.0))
        assert 70.0 <= d < 80.0
        assert a["out_enough"][0]


def test_escape_point_threshold_is_at_least_wmin(run):
    depth = _grid()
    depth[33, 30] = -5.0
    for cx in (25, 26, 34, 35):
        for cy in (34, 35):
            depth[cy, cx] = 1.0  # exactly wmin: good
    a = _one()
    run(a, depth)
    assert 40.0 <= abs(_dh(a["heading"][0], 0.0)) < 50.0


def test_beh5_deepest_neighbour_sets_presmov_one(run):
    depth = _grid(depth=-1.0)
    depth[29:32, 29:32] = 0.5  # own cell and the 8 neighbours: water, shallow
    depth[29, 31] = 2.0  # deepest neighbour: cell (31, 29), centre (31.5, 29.5)
    a = _one()
    run(a, depth)
    assert a["log_mov"][0] == 0.0
    assert abs(_dh(a["heading"][0], 135.0)) < 1e-3
    assert not a["out_enough"][0]
    assert a["x"][0] == np.float32(30.5) and a["y"][0] == np.float32(30.5)


def test_beh5_first_maximum_wins(run):
    depth = _grid(depth=-1.0)
    depth[29:32, 29:32] = 0.5
    depth[30, 29] = 2.0  # (-1, 0): earlier in the neighbour order
    depth[31, 31] = 2.0  # (+1, +1): later, same depth
    a = _one()
    run(a, depth)
    assert abs(_dh(a["heading"][0], 270.0)) < 1e-3


def test_beh5_nan_neighbour_is_land(run):
    depth = _grid(depth=-1.0)
    depth[29:32, 29:32] = 0.5
    depth[29, 29] = np.nan  # (-1, -1): first in the neighbour order
    depth[31, 30] = 2.0  # (0, +1): deepest
    a = _one()
    run(a, depth)
    assert abs(_dh(a["heading"][0], 0.0)) < 1e-3
    assert a["log_mov"][0] == 0.0


def _pocket():
    """Land everywhere x >= 20; deep water x < 20. Agent cell (30, 30) is land."""
    depth = _grid()
    depth[:, 20:] = -5.0
    return depth


def test_beh4_walks_back_to_remembered_position(run):
    depth = _pocket()
    a = _one()
    _remember(a, 0, [(40.5, 40.5), (10.5, 30.5)], mem_ptr=2)
    run(a, depth)
    assert a["x"][0] == np.float32(10.5) and a["y"][0] == np.float32(30.5)
    assert abs(_dh(a["heading"][0], 270.0)) < 1e-3
    assert a["out_enough"][0]
    assert a["log_mov"][0] == LOG6


def test_beh4_newest_entry_index_wraps(run):
    depth = _pocket()
    a = _one(mem_size=8)
    a["pos_hist_x"][0, :] = 40.5
    a["pos_hist_y"][0, :] = 40.5
    a["pos_hist_x"][0, 7] = 10.5
    a["pos_hist_y"][0, 7] = 30.5
    a["mem_ptr"][0] = 0  # next write index 0 -> newest entry is index 7
    a["mem_count"][0] = 8
    run(a, depth)
    assert a["x"][0] == np.float32(10.5) and a["y"][0] == np.float32(30.5)


def test_beh4_current_position_entry_keeps_heading(run):
    # Water only in the column x = 30, rows 30..39. Heading 90 (east) fails, and every escape
    # point (+-40..130 deg off east) leaves the column. The only remembered position is the
    # current one: facing it is a no-op (heading stays 90), so the path still fails and beh 5
    # follows. Facing it with atan2(0, 0) = 0 would turn north up the column and pass.
    depth = _grid(depth=-5.0)
    depth[30:40, 30] = 30.0
    a = _one(h=90.0)
    _remember(a, 0, [(30.5, 30.5)], mem_ptr=1)
    run(a, depth)
    assert not a["out_enough"][0]
    assert a["log_mov"][0] == 0.0  # beh 5
    assert abs(_dh(a["heading"][0], 0.0)) < 1e-3  # the deepest neighbour is (30, 31)
    assert a["x"][0] == np.float32(30.5) and a["y"][0] == np.float32(30.5)


def _corridor():
    """Land, except water for x < 20 at rows >= 31 and the cells (8..10, 30)."""
    depth = _grid(depth=-5.0)
    depth[31:, :20] = 30.0
    depth[30, 8:11] = 30.0
    return depth


def test_beh4_exhausted_short_history_keeps_trying(run):
    # One remembered position, (10.5, 30.5), facing west (270): the path hits land at x < 8,
    # so the walk runs out of history without a pass. Fewer than 20 steps were taken, so DEPONS
    # stays in the round loop: from (10.5, 30.5) the right first-tier point (310..320 deg) is
    # water, the turn passes the path check, and beh 5 is not needed.
    depth = _corridor()
    for seed in SEEDS:
        a = _one()
        _remember(a, 0, [(10.5, 30.5)], mem_ptr=1)
        run(a, depth, seed=seed)
        assert a["x"][0] == np.float32(10.5) and a["y"][0] == np.float32(30.5)
        assert a["out_enough"][0]
        assert a["log_mov"][0] == LOG6
        assert 310.0 <= float(a["heading"][0]) < 320.0


def test_beh4_twenty_steps_end_the_rounds(run):
    # 25 remembered positions on row 30, x = 5.5 .. 29.5 (newest 29.5). Each step faces west.
    # None passes. The 20th step lands on (10.5, 30.5); DEPONS then forces the round loop to
    # end (j == 20), the final check fails, and beh 5 runs there. Without the cap, another round
    # would find the first-tier escape at (10.5, 30.5), as in the previous test.
    depth = _corridor()
    for seed in SEEDS:
        a = _one(mem_size=30)
        entries = [(5.5 + k, 30.5) for k in range(25)]
        _remember(a, 0, entries, mem_ptr=25)
        run(a, depth, seed=seed)
        assert a["x"][0] == np.float32(10.5) and a["y"][0] == np.float32(30.5)
        assert not a["out_enough"][0]
        assert a["log_mov"][0] == 0.0
        # beh 5: deepest neighbours of (10, 30) are all 30 m; first in order is (9, 30)
        assert abs(_dh(a["heading"][0], 270.0)) < 1e-3


def test_jitter_one_draw_per_round_shared_by_tiers(run):
    # Both modes draw np.random.uniform(0, 10) from MT19937 seeded with `seed`, so the draws
    # are known. Second tier in round 1: the turn is 70 + r1 (r1 also served the failed tier 1).
    depth = _grid()
    depth[33, 30] = -5.0
    for cx in (25, 26, 34, 35):
        for cy in (34, 35):
            depth[cy, cx] = 0.5
    for seed in SEEDS:
        r1 = np.random.RandomState(seed).uniform(0.0, 10.0)
        a = _one()
        run(a, depth, seed=seed)
        assert abs(_dh(a["heading"][0], 70.0 + r1)) < 1e-3

    # Exhausted short history: round 1 (tiers fail, walk) draws r1; round 2 turns by 40 + r2.
    depth = _corridor()
    for seed in SEEDS:
        rs = np.random.RandomState(seed)
        rs.uniform(0.0, 10.0)
        r2 = rs.uniform(0.0, 10.0)
        a = _one()
        _remember(a, 0, [(10.5, 30.5)], mem_ptr=1)
        run(a, depth, seed=seed)
        assert abs(_dh(a["heading"][0], 270.0 + 40.0 + r2)) < 1e-3


# Agent 1.5 cells from an edge, heading off the 60x60 grid. With presMov = 6 the look-ahead
# overshoots by 4.5 cells; DEPONS's BouncyBorders reflects it back, so the reflected samples
# cover the 4.5 cells behind the edge-side position. `land` is a cell reached only that way.
EDGES = [
    pytest.param(58.5, 30.5, 90.0, (30, 56), id="east"),
    pytest.param(1.5, 30.5, 270.0, (30, 3), id="west"),
    pytest.param(30.5, 58.5, 0.0, (56, 30), id="north"),
    pytest.param(30.5, 1.5, 180.0, (3, 30), id="south"),
]


@pytest.mark.parametrize("x0, y0, h0, land", EDGES)
def test_offgrid_lookahead_reflects_into_water(run, x0, y0, h0, land):
    a = _one(x=x0, y=y0, h=h0)
    run(a, _grid())
    assert a["heading"][0] == np.float32(h0)  # open-water edge: no turn
    assert a["out_enough"][0]


@pytest.mark.parametrize("x0, y0, h0, land", EDGES)
def test_offgrid_lookahead_reflects_onto_land(run, x0, y0, h0, land):
    depth = _grid()
    depth[land] = -5.0
    for seed in SEEDS:
        a = _one(x=x0, y=y0, h=h0)
        run(a, depth, seed=seed)
        assert 40.0 <= _dh(a["heading"][0], h0) < 50.0  # tie: first tier, right
        assert a["out_enough"][0]
        assert a["x"][0] == np.float32(x0) and a["y"][0] == np.float32(y0)


def test_lookahead_point_exactly_on_edge_reads_last_column(run):
    # presMov = 10 from x = 50 heading east: the ahead point and the last sample are exactly
    # x = 60.0 = W. BouncyBorders nudges that just below W (column 59), not off the grid.
    assert 50.0 + 0.1 * 100 == 60.0 and math.ceil(10.0 / 0.1) == 100
    a = _one(x=50.0, y=30.5, h=90.0, log_mov=1.0)
    run(a, _grid())
    assert a["heading"][0] == np.float32(90.0)
    assert a["out_enough"][0]

    depth = _grid()
    depth[30, 59] = -5.0
    a = _one(x=50.0, y=30.5, h=90.0, log_mov=1.0)
    run(a, depth)
    assert abs(_dh(a["heading"][0], 90.0)) >= 40.0


def test_beh5_offgrid_neighbour_is_land(run):
    # Agent in the corner cell (0, 0), boxed in. Off-grid neighbours read -9999 (no bounce):
    # the deepest in-grid neighbour wins even though it is shallow.
    depth = _grid(depth=-1.0)
    depth[0:2, 0:2] = 0.5
    depth[1, 0] = 0.8  # (0, +1): deepest in-grid neighbour
    a = _one(x=0.5, y=0.5)
    run(a, depth)
    assert a["log_mov"][0] == 0.0
    assert abs(_dh(a["heading"][0], 0.0)) < 1e-3


def test_nan_depth_is_land(run):
    depth = _grid()
    depth[33, 30] = np.nan
    a = _one()
    run(a, depth)
    assert abs(_dh(a["heading"][0], 0.0)) >= 40.0
    assert a["out_enough"][0]


def test_nan_escape_points_are_not_good(run):
    depth = _grid()
    depth[33, 30] = -5.0
    for cx in (25, 26, 34, 35):
        for cy in (34, 35):
            depth[cy, cx] = np.nan
    for seed in SEEDS:
        a = _one()
        run(a, depth, seed=seed)
        assert 70.0 <= abs(_dh(a["heading"][0], 0.0)) < 80.0


def test_long_lookahead_detects_far_land(run):
    log_mov = 1.73  # max_mov: presMov ~ 53.7, ~538 path samples
    x0, y0 = 80.5, 50.5
    depth = _grid(h=160, w=160)
    depth[100, 80] = -5.0  # 49.5..50.5 cells ahead
    for seed in SEEDS:
        a = _one(x=x0, y=y0, log_mov=log_mov)
        run(a, depth, seed=seed)
        assert 40.0 <= abs(_dh(a["heading"][0], 0.0)) < 50.0
        assert a["out_enough"][0]
        assert a["log_mov"][0] == log_mov

    depth = _grid(h=160, w=160)
    depth[105, 80] = -5.0  # 54.5 cells ahead: beyond presMov
    a = _one(x=x0, y=y0, log_mov=log_mov)
    run(a, depth)
    assert a["heading"][0] == np.float32(0.0)
    assert a["out_enough"][0]


def test_mask_respected(run):
    depth = _grid()
    depth[33, 30] = -5.0
    depth[33, 10] = -5.0
    a = _agents([30.5, 10.5], [30.5, 30.5], [0.0, 0.0], [LOG6, LOG6], mask=[False, True])
    a["out_enough"][0] = True  # sentinel: the real path check would give False
    run(a, depth)
    assert a["heading"][0] == np.float32(0.0)
    assert a["x"][0] == np.float32(30.5) and a["y"][0] == np.float32(30.5)
    assert a["log_mov"][0] == LOG6
    assert a["out_enough"][0]
    assert abs(_dh(a["heading"][1], 0.0)) >= 40.0
    assert a["out_enough"][1]


def test_float32_depth_grid(run):
    depth = _grid().astype(np.float32)
    depth[33, 30] = -5.0
    a = _one()
    run(a, depth)
    assert abs(_dh(a["heading"][0], 0.0)) >= 40.0
    assert a["out_enough"][0]


def test_heading_stays_in_range(run):
    depth = _grid()
    depth[33, 30] = -5.0
    depth[34:36, 34:36] = -5.0  # force a left turn from heading 0 -> ~310..320
    a = _one()
    run(a, depth)
    assert 0.0 <= float(a["heading"][0]) < 360.0


@pytest.mark.skipif(not land_check.HAS_NUMBA, reason="Numba not installed")
def test_python_and_numba_agree(monkeypatch):
    """Same seed, same scenario: both modes draw the same jitter (MT19937) and agree."""
    from cenop.optimizations.kernels import seed_numba_rng

    rng = np.random.default_rng(7)
    depth = rng.uniform(-3.0, 12.0, size=(80, 80))
    depth[rng.random((80, 80)) < 0.05] = np.nan
    n = 200
    a = _agents(
        rng.uniform(0.0, 80.0, n),
        rng.uniform(0.0, 80.0, n),
        rng.uniform(0.0, 360.0, n),
        rng.uniform(-0.5, 1.2, n),
        mask=rng.random(n) < 0.9,
        mem_size=30,
    )
    a["pos_hist_x"][:] = rng.uniform(0.0, 80.0, (n, 30)).astype(np.float32)
    a["pos_hist_y"][:] = rng.uniform(0.0, 80.0, (n, 30)).astype(np.float32)
    a["mem_ptr"][:] = rng.integers(0, 30, n)
    a["mem_count"][:] = rng.integers(0, 31, n)
    b = {k: v.copy() for k, v in a.items()}
    x_before = a["x"].copy()
    h_before = a["heading"].copy()

    seed_numba_rng(11)
    land_check.depons_land_check(*_args(a, depth, 1.0))

    _interpret_helpers(monkeypatch)
    np.random.seed(11)
    land_check._py(*_args(b, depth, 1.0))

    assert np.array_equal(a["out_enough"], b["out_enough"])
    assert np.array_equal(a["x"], b["x"]) and np.array_equal(a["y"], b["y"])
    assert np.allclose(a["heading"], b["heading"], atol=1e-4)
    assert np.array_equal(a["log_mov"], b["log_mov"])
    # the scenario exercises the interesting branches
    assert (a["mask"] & ~a["out_enough"]).any()  # beh 5
    assert (a["mask"] & (a["x"] != x_before)).any()  # beh 4 moves
    assert (a["mask"] & a["out_enough"] & (a["heading"] != h_before)).any()  # tier turns


# --- Population wiring (D15 Task 2) -------------------------------------------------------------


def _coast_landscape():
    """200 x 200, all water (30 m) except land (-10 m) at y >= 100. Renamed, so land avoidance
    is not skipped; the Homogeneous border land is overwritten so open water stays open."""
    land = create_homogeneous_landscape(200, 200)
    land._depth[:] = 30.0
    land._depth[100:, :] = -10.0
    land.landscape_name = "TestCoast"
    return land


def _coast_pop(count, landscape=None, movement_module=None, seed=7):
    params = SimulationParameters(random_seed=seed)
    land = _coast_landscape() if landscape is None else landscape
    pop = PorpoisePopulation(count, params, land, movement_module=movement_module)
    pop._skip_land_avoidance = False
    pop.is_dispersing[:] = False
    return pop


def _recorder(calls, label, fn):
    def wrapped(*args, **kwargs):
        calls.append(label)
        return fn(*args, **kwargs)

    return wrapped


class TestPopulationWiring:
    """`PorpoisePopulation._depons_land_check` on the CRW paths: after reference memory and the
    CRW turn, before heading composition; reflection-only after composition; beh 6 flip."""

    def test_land_check_runs_before_composition(self, monkeypatch):
        pop = _coast_pop(72)
        coast, open_, disp = np.arange(60), np.arange(60, 70), np.arange(70, 72)
        pop.x[coast] = np.linspace(20.5, 179.5, 60)
        pop.y[coast] = 98.5
        pop.heading[coast] = np.linspace(0.0, 360.0, 60, endpoint=False)
        pop.x[open_] = np.linspace(30.5, 170.5, 10)
        pop.y[open_] = 40.5
        pop.heading[open_] = np.linspace(0.0, 360.0, 10, endpoint=False)
        # Two dispersers in deep open water heading south: their look-ahead keeps them going
        pop.x[disp] = [60.5, 140.5]
        pop.y[disp] = 60.5
        pop.heading[disp] = 180.0
        pop._prev_step_heading[disp] = 180.0
        pop.is_dispersing[disp] = True
        pop.dispersal_target_distance[disp] = 50.0
        pop.dispersal_distance_traveled[disp] = 0.0

        calls, seen = [], {}
        orig_check = PorpoisePopulation._depons_land_check
        orig_refmem = PorpoisePopulation._update_reference_memory

        def check(self, mask):
            calls.append("land_check")
            seen["mask"] = mask.copy()
            seen["entry"] = self.heading.copy()
            orig_check(self, mask)
            seen["exit"] = self.heading.copy()

        monkeypatch.setattr(PorpoisePopulation, "_depons_land_check", check)
        monkeypatch.setattr(
            PorpoisePopulation,
            "_update_reference_memory",
            _recorder(calls, "ref_mem", orig_refmem),
        )
        for name in ("_heading_kernel", "compose_movement"):
            if hasattr(popmod, name):
                monkeypatch.setattr(
                    popmod, name, _recorder(calls, "compose", getattr(popmod, name))
                )

        pop._recompute_cell_indices()
        pop._update_movement(pop.active_mask, None, None)

        assert calls == ["ref_mem", "land_check", "compose"]
        assert pop.is_dispersing[disp].all()  # non-vacuous: the dispersers kept dispersing
        assert not seen["mask"][disp].any()
        assert seen["mask"][coast].all() and seen["mask"][open_].all()
        turn = np.abs((seen["exit"] - seen["entry"] + 180.0) % 360.0 - 180.0)
        assert turn[coast].max() >= 40.0
        np.testing.assert_array_equal(seen["exit"][open_], seen["entry"][open_])
        assert pop._enough_water_ahead[open_].all()

    def test_boxed_in_agent_records_no_water_and_sets_presmov_one(self):
        """beh 5 (boxed in by land, empty memory): _enough_water_ahead False, presMov = 1."""
        land = _coast_landscape()
        land._depth[:] = -10.0
        land._depth[50, 50] = 30.0  # a single water cell: every look-ahead point is land
        pop = _coast_pop(1, landscape=land)
        pop.x[0], pop.y[0], pop.heading[0] = 50.5, 50.5, 0.0
        pop._mem_count[0] = 0
        pop._mem_ptr[0] = 0

        pop._recompute_cell_indices()
        pop._update_movement(pop.active_mask, None, None)

        assert not pop._enough_water_ahead[0]
        assert pop._log_mov[0] == 0.0
        assert pop.prev_log_mov[0] == 0.0

    def test_no_post_composition_land_check_on_crw_path(self):
        """A destination at 0.5 m (0 < depth < wmin) is accepted: DEPONS forward() has no land
        check, and beh 6 only rolls back from depth <= 0."""
        land = _coast_landscape()
        land._depth[52, 50] = 0.5
        pop = _coast_pop(1, landscape=land)
        pop.x[0], pop.y[0], pop.heading[0] = 50.5, 50.5, 0.0
        pop._dx[0], pop._dy[0] = 0.0, 2.0
        mask = pop.active_mask.copy()

        pop._handle_land_avoidance(mask)
        pop._apply_positions(mask)

        assert (float(pop.x[0]), float(pop.y[0])) == (50.5, 52.5)
        assert float(pop.heading[0]) == 0.0

    def test_rollback_faces_back(self):
        """beh 6: a destination on depth <= 0 returns the agent to its pre-move position, facing
        back along the attempted move."""
        land = _coast_landscape()
        land._depth[:, 52:] = -10.0  # land east of x = 52
        pop = _coast_pop(1, landscape=land)
        pop.x[0], pop.y[0], pop.heading[0] = 50.5, 50.5, 180.0
        pop._log_mov[0] = math.log10(2.0)
        mask = pop.active_mask.copy()

        pop._depons_land_check(mask)  # open water to the south: nothing to avoid
        assert float(pop.heading[0]) == 180.0 and pop._enough_water_ahead[0]
        pop._dx[0], pop._dy[0] = 2.5, 0.0  # composition turned it east, onto land
        pop._handle_land_avoidance(mask)
        pop._apply_positions(mask)

        assert (float(pop.x[0]), float(pop.y[0])) == (50.5, 50.5)
        assert float(pop.heading[0]) == pytest.approx(270.0)

    def test_beh4_moved_agent_rolls_back_to_start_of_tick(self):
        """beh 4 moves the agent during the land check, but beh 6 returns it to the start-of-tick
        position (DEPONS posList.get(0)) and food is eaten there (posList.get(1))."""
        land = _coast_landscape()
        land._depth[:] = -10.0
        land._depth[:51, 49:52] = 30.0  # dead-end corridor: x in 49..51, y <= 50
        pop = _coast_pop(1, landscape=land)
        start, back = (50.5, 50.5), (50.5, 47.5)
        pop.x[0], pop.y[0], pop.heading[0] = start[0], start[1], 0.0
        pop._log_mov[0] = math.log10(2.0)
        # Ring buffer: an older position, then the start-of-tick one (stored by reference memory)
        pop._pos_history_x[0, :2] = [back[0], start[0]]
        pop._pos_history_y[0, :2] = [back[1], start[1]]
        pop._mem_ptr[0], pop._mem_count[0] = 2, 2
        mask = pop.active_mask.copy()

        pop._depons_land_check(mask)
        # Every escape point is land: beh 4 walked back onto the older position, facing it
        assert (float(pop.x[0]), float(pop.y[0])) == back
        assert float(pop.heading[0]) == pytest.approx(180.0)
        assert pop._enough_water_ahead[0]

        pop._dx[0], pop._dy[0] = 2.5, 0.0  # composition sends it east, onto land at (53, 47.5)
        pop._handle_land_avoidance(mask)
        pop._apply_positions(mask)

        assert (float(pop.x[0]), float(pop.y[0])) == start
        expected = math.degrees(math.atan2(start[0] - 53.0, start[1] - 47.5)) % 360.0
        assert float(pop.heading[0]) == pytest.approx(expected, abs=1e-4)
        # Eat cell: the start-of-tick cell, not the walked-back one
        assert (float(pop._pre_move_x[0]), float(pop._pre_move_y[0])) == start
        assert (int(pop._pre_cell_xi[0]), int(pop._pre_cell_yi[0])) == (50, 50)
        food_before = land._food_value.copy()
        pop._eat_food_vectorized(mask, np.full(1, 0.5, dtype=np.float32))
        assert np.argwhere(land._food_value != food_before).tolist() == [[50, 50]]

    @pytest.mark.skipif(not land_check.HAS_NUMBA, reason="Numba not installed")
    def test_python_fallback_seeds_numpy_rng(self, monkeypatch):
        """Without Numba the interpreted kernel draws its jitter from NumPy's global RNG,
        seeded from self.rng the same way, so both modes agree."""

        def run():
            pop = _coast_pop(60, seed=5)
            pop.x[:] = np.linspace(20.5, 179.5, 60)
            pop.y[:] = 98.5
            pop.heading[:] = np.linspace(0.0, 360.0, 60, endpoint=False)
            pop._log_mov[:] = np.linspace(0.2, 1.2, 60)
            h0 = pop.heading.copy()
            pop._depons_land_check(pop.active_mask.copy())
            return pop, h0

        a, h0 = run()
        _interpret_helpers(monkeypatch)
        monkeypatch.setattr(popmod, "_LAND_CHECK_NUMBA", False)
        monkeypatch.setattr(popmod, "_depons_land_check_kernel", land_check._py)
        b, _ = run()

        assert np.array_equal(a.x, b.x) and np.array_equal(a.y, b.y)
        assert np.allclose(a.heading, b.heading, atol=1e-4)
        assert np.array_equal(a._log_mov, b._log_mov)
        assert np.array_equal(a._enough_water_ahead, b._enough_water_ahead)
        turn = np.abs((a.heading - h0 + 180.0) % 360.0 - 180.0)
        assert (turn >= 40.0).any()  # non-vacuous: jittered tier turns happened

    def test_generic_module_path_keeps_old_land_avoidance(self):
        """Non-CRW modules keep the post-composition destination check (turn away from < wmin)."""
        from cenop.movement import JASMINEPhysicsMovement

        land = _coast_landscape()
        land._depth[52, 50] = 0.5
        params = SimulationParameters(random_seed=7)
        pop = _coast_pop(1, landscape=land, movement_module=JASMINEPhysicsMovement(params))
        pop.x[0], pop.y[0], pop.heading[0] = 50.5, 50.5, 0.0
        pop._step_dist[0] = 2.0
        pop._dx[0], pop._dy[0] = 0.0, 2.0
        mask = pop.active_mask.copy()

        pop._handle_land_avoidance(mask)

        xi, yi = int(pop._new_x[0]), int(pop._new_y[0])
        assert (xi, yi) != (50, 52)
        assert land._depth[yi, xi] >= 1.0

    @pytest.mark.parametrize("with_module", [False, True], ids=["inline", "crw_module"])
    def test_homogeneous_unchanged(self, monkeypatch, with_module):
        """Homogeneous grid (_skip_land_avoidance): no land check runs, the water state stays
        True, and a rollback onto the border land keeps the heading (no beh-6 flip)."""
        from cenop.movement import DEPONSCRWMovementVectorized

        params = SimulationParameters(random_seed=3)
        mod = DEPONSCRWMovementVectorized(params) if with_module else None
        pop = PorpoisePopulation(50, params, create_homogeneous_landscape(200, 200), mod)
        assert pop._skip_land_avoidance
        calls = []
        monkeypatch.setattr(popmod, "_depons_land_check_kernel", lambda *a: calls.append("kernel"))
        monkeypatch.setattr(
            PorpoisePopulation, "_depons_land_check", lambda self, m: calls.append("method")
        )

        for _ in range(5):
            pop.step()

        assert calls == []
        assert pop._enough_water_ahead.all()

        mask = np.zeros(pop.count, dtype=bool)
        mask[0] = True
        pop.x[0], pop.y[0], pop.heading[0] = 100.5, 10.5, 123.0
        pop._dx[0], pop._dy[0] = 0.0, -1.0  # onto the southern border land (y < 10)
        pop._handle_land_avoidance(mask)
        pop._apply_positions(mask)
        assert (float(pop.x[0]), float(pop.y[0])) == (100.5, 10.5)
        assert float(pop.heading[0]) == 123.0
