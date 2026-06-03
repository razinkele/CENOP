"""Round-3 codebase review fixes — resolution of disputed findings.

Three findings deferred from round 2 were resolved against the authoritative
DEPONS 3.2 source (parameters.xml + Porpoise.java):

  1. deter_max_distance = 1000.0 km  (parameters.xml dmax_deter, not the stale
     50*1000 field initializer) — asserted in test_deterrence.py.
  2. deter_ships_min_db = 80.0 dB    (parameters.xml Tships, not the 70.0
     unit-test default) — asserted in test_deterrence.py.
  3. Food is eaten at the PRE-move cell (the patch the porpoise "just has left",
     Java Porpoise.updEnergeticStatus eats at posList.get(1)), not the post-move
     cell — verified here.
"""

import numpy as np
import pytest

from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData, LandscapeMetadata
from cenop.parameters.simulation_params import SimulationParameters


@pytest.fixture
def params():
    return SimulationParameters()


@pytest.fixture
def landscape(params):
    """All-water landscape with uniform food = 1.0 everywhere."""
    cd = CellData("TestLandscape")
    w, h = params.world_width, params.world_height
    cd._depth = np.full((h, w), 20.0, dtype=np.float64)
    cd._salinity = np.full((12, h, w), 30.0, dtype=np.float64)
    cd._food_value = np.ones((h, w), dtype=np.float32)
    cd._food_prob = np.ones((h, w), dtype=np.float32)
    cd.metadata = LandscapeMetadata(ncols=w, nrows=h, xllcorner=0, yllcorner=0)
    cd._loaded = True
    cd._current_month = 6
    cd._demand_grid = None
    return cd


class TestFoodEatenAtPreMoveCell:
    """Finding 3: porpoise eats food at the cell it just left (pre-move)."""

    def test_eat_food_vectorized_uses_pre_move_cell(self, params, landscape):
        """Food depletion must hit the PRE-move cell, not the post-move cell.

        Java: Porpoise.updEnergeticStatus() →
              eatFood(ndPointToGridPoint(posList.get(1)), ...)
        posList.get(1) is the position the porpoise occupied before this step's
        move — the cell it "just has left".
        """
        pop = PorpoisePopulation(2, params, landscape=landscape)

        # Agent 0: pre-move cell (2, 2), post-move cell (7, 7)
        pop._pre_move_x[0] = 2.5
        pop._pre_move_y[0] = 2.5
        pop.x[0] = 7.5
        pop.y[0] = 7.5
        pop._recompute_cell_indices()                 # _cell_xi/_yi = post-move (7, 7)
        pop._snapshot_pre_move_cells(pop._pre_move_x, pop._pre_move_y)  # pre = (2, 2)
        pop.energy[0] = 10.0                           # hungry → fraction > 0

        pre_cell_before = float(landscape._food_value[2, 2])
        post_cell_before = float(landscape._food_value[7, 7])

        fract = np.full(pop.count, 0.5, dtype=np.float32)
        mask = np.array([True, False])
        eaten = pop._eat_food_vectorized(mask, fract, active_idx=np.array([0]))

        assert eaten[0] > 0, "Active hungry agent should consume food"
        assert landscape._food_value[2, 2] < pre_cell_before, \
            "Pre-move cell (2,2) should be depleted (eats where it just left)"
        assert landscape._food_value[7, 7] == pytest.approx(post_cell_before), \
            "Post-move cell (7,7) should be untouched"

    def test_pre_move_cell_buffers_exist(self, params, landscape):
        """Pre-move cell index buffers are allocated with correct dtype/shape."""
        pop = PorpoisePopulation(5, params, landscape=landscape)
        assert hasattr(pop, "_pre_cell_xi")
        assert hasattr(pop, "_pre_cell_yi")
        assert pop._pre_cell_xi.dtype == np.int32
        assert pop._pre_cell_yi.dtype == np.int32
        assert pop._pre_cell_xi.shape == (pop.count,)
        assert pop._pre_cell_yi.shape == (pop.count,)

    def test_snapshot_pre_move_cells_clamps(self, params, landscape):
        """_snapshot_pre_move_cells computes clamped int32 indices from positions."""
        pop = PorpoisePopulation(3, params, landscape=landscape)
        w, h = landscape.width, landscape.height
        pre_x = np.array([3.7, -2.0, w + 5.0], dtype=np.float32)
        pre_y = np.array([4.2, h + 9.0, -1.0], dtype=np.float32)
        pop._snapshot_pre_move_cells(pre_x, pre_y)
        np.testing.assert_array_equal(pop._pre_cell_xi, [3, 0, w - 1])
        np.testing.assert_array_equal(pop._pre_cell_yi, [4, h - 1, 0])
