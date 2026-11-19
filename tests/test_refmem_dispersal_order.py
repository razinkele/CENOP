"""D16: what a dispersing animal stores in its reference memory.

DEPONS (Porpoise.move): disperse() moves the animal first; refMemTurn then stores the food level
of the cell it is in *after* that move, and the position list still holds the position before
the move at that index. A foraging animal stores the food level of the cell it starts the tick in.
"""

import numpy as np
import pytest

import cenop.agents.population as pop_mod
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData
from cenop.parameters.simulation_params import SimulationParameters


@pytest.fixture(scope="module")
def kattegat():
    cd = CellData("Kattegat", "data")
    cd.load()
    return cd


@pytest.fixture
def numbered_food(kattegat):
    """Every cell gets its own food level, so a stored value identifies the cell it came from."""
    saved = kattegat._food_value
    h, w = saved.shape
    kattegat._food_value = (0.2 + 0.6 * np.arange(h * w).reshape(h, w) / (h * w)).astype(np.float32)
    yield kattegat
    kattegat._food_value = saved


def _open_cells(cd, n):
    ok = (cd._depth > 25) & (cd._dist_to_coast > 8000) & ~np.isin(cd._blocks, [14, 31, 32])
    ys, xs = np.nonzero(ok)
    keep = ~((xs > 438) & (ys < 478)) & (xs > 50) & (xs < 500) & (ys > 50) & (ys < 950)
    cells = list(zip(xs[keep].tolist(), ys[keep].tolist()))
    return cells[:: len(cells) // n][:n]


def _dispersers(cd, mode, n=8):
    params = SimulationParameters(
        random_seed=4, porpoise_count=2 * n, dispersal=mode, landscape="Kattegat"
    )
    pop = PorpoisePopulation(2 * n, params, cd)
    for i, (cx, cy) in enumerate(_open_cells(cd, 2 * n)):
        pop.x[i], pop.y[i] = cx + 0.5, cy + 0.5
    pop.energy[:] = 15.0
    d = np.arange(n)  # the first n disperse, the rest forage
    pop.heading[d] = 90.0
    pop._prev_step_heading[d] = 90.0
    pop.is_dispersing[d] = True
    pop.dispersal_start_x[d], pop.dispersal_start_y[d] = pop.x[d], pop.y[d]
    pop.dispersal_target_x[d], pop.dispersal_target_y[d] = pop.x[d] + 60.0, pop.y[d]
    pop.dispersal_target_distance[d] = 60.0
    pop.dispersal_distance_traveled[d] = 0.0
    pop._enough_water_ahead[:] = True
    if mode == "InnerDanishWaters":
        pop.dispersal_phase[d] = 1
        pop._daily_pos_x[d, :] = pop.x[d][:, None]
        pop._daily_pos_y[d, :] = pop.y[d][:, None] - 100.0
    pop._recompute_cell_indices()
    return pop, d


@pytest.mark.parametrize("mode", ["PSM-Type2", "InnerDanishWaters"])
def test_disperser_stores_the_food_of_the_cell_it_moved_to(monkeypatch, numbered_food, mode):
    monkeypatch.setattr(pop_mod, "_HAS_CYTHON", False)
    cd = numbered_food
    pop, d = _dispersers(cd, mode)
    food = cd._food_value.copy()
    x0, y0 = pop.x.copy(), pop.y.copy()

    pop.step()

    last = (pop._mem_ptr - 1) % pop._stored_util.shape[1]
    stored = pop._stored_util[np.arange(pop.count), last]
    moved = d[pop.is_dispersing[d] & (np.hypot(pop.x[d] - x0[d], pop.y[d] - y0[d]) > 3.0)]
    assert moved.size >= 4  # open water: most keep dispersing and make the 5-cell move
    after = food[pop.y[moved].astype(int), pop.x[moved].astype(int)]
    before = food[y0[moved].astype(int), x0[moved].astype(int)]
    assert (after != before).all()
    np.testing.assert_allclose(stored[moved], after, rtol=1e-6)
    # the position stored with it is the one before the move
    np.testing.assert_allclose(pop._pos_history_x[moved, last[moved]], x0[moved])
    np.testing.assert_allclose(pop._pos_history_y[moved, last[moved]], y0[moved])

    foragers = np.arange(d.size, pop.count)
    foragers = foragers[pop.active_mask[foragers] & ~pop.is_dispersing[foragers]]
    assert foragers.size >= 4
    np.testing.assert_allclose(
        stored[foragers], food[y0[foragers].astype(int), x0[foragers].astype(int)], rtol=1e-6
    )
