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
