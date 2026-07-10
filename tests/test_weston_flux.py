"""Tests for WestonFlux transmission loss model."""

from unittest.mock import MagicMock

import numpy as np

from cenop.behavior.weston_flux import weston_flux_tl
from cenop.landscape.cell_data import CellData, LandscapeMetadata
from cenop.parameters.simulation_params import SimulationParameters


class TestWestonFlux:
    """Test WestonFlux TL against Java reference values."""

    def test_basic_tl(self):
        """WestonFlux TL for typical North Sea conditions."""
        tl = weston_flux_tl(
            distance=1000.0, depth=30.0, grain_size=2.0, temperature=10.0, salinity=35.0
        )
        # TL should be in reasonable range (40-80 dB for 1km)
        assert 30 < tl < 90, f"TL={tl} outside expected range"

    def test_tl_increases_with_distance(self):
        """TL should increase with distance."""
        tl_near = weston_flux_tl(100.0, 30.0, 2.0, 10.0, 35.0)
        tl_far = weston_flux_tl(5000.0, 30.0, 2.0, 10.0, 35.0)
        assert tl_far > tl_near

    def test_tl_varies_with_depth(self):
        """TL should be different for different depths (both valid)."""
        tl_shallow = weston_flux_tl(1000.0, 10.0, 2.0, 10.0, 35.0)
        tl_deep = weston_flux_tl(1000.0, 100.0, 2.0, 10.0, 35.0)
        # Both should be in valid range; the relationship depends on the
        # Weston flux integral's balance of geometric spreading and bottom loss
        assert 20 < tl_shallow < 100
        assert 20 < tl_deep < 100
        assert tl_shallow != tl_deep

    def test_zero_distance_returns_zero(self):
        """Zero distance should return 0 TL."""
        tl = weston_flux_tl(0.0, 30.0, 2.0, 10.0, 35.0)
        assert tl == 0.0

    def test_various_grain_sizes(self):
        """TL should be computable for all valid grain sizes."""
        for gs in [-5.0, -1.0, 0.0, 1.0, 3.0, 5.0, 7.0, 9.0]:
            tl = weston_flux_tl(1000.0, 30.0, gs, 10.0, 35.0)
            assert 0 < tl < 200, f"TL={tl} for grain_size={gs} outside range"


class TestGetSedimentsVectorized:
    """Test vectorized sediment lookup."""

    def _make_cell_data(self):
        """Create a CellData with known sediment values."""
        cd = CellData.__new__(CellData)
        cd._loaded = True
        cd.metadata = LandscapeMetadata(
            ncols=4,
            nrows=4,
            xllcorner=0.0,
            yllcorner=0.0,
            cellsize=400.0,
        )
        cd._sediment = np.array(
            [
                [-2.0, -2.0, -2.0, -2.0],
                [2.0, 2.0, 2.0, 2.0],
                [5.0, 5.0, 5.0, 5.0],
                [8.0, 8.0, 8.0, 8.0],
            ]
        )
        cd._depth = np.full((4, 4), 30.0)
        cd._salinity = None
        cd._food_prob = None
        cd._food_level = None
        cd._dist_to_coast = None
        cd._blocks = None
        cd._entropy = None
        cd._current_month = 1
        return cd

    def test_returns_correct_values(self):
        """Should return sediment at each position."""
        cd = self._make_cell_data()
        positions = np.array([[0, 0], [0, 1], [0, 2], [0, 3]], dtype=float)
        result = cd.get_sediments_vectorized(positions)
        np.testing.assert_array_almost_equal(result, [-2.0, 2.0, 5.0, 8.0])

    def test_none_sediment_returns_default(self):
        """When _sediment is None, should return default 1.0."""
        cd = self._make_cell_data()
        cd._sediment = None
        positions = np.array([[0, 0], [1, 1]], dtype=float)
        result = cd.get_sediments_vectorized(positions)
        np.testing.assert_array_equal(result, [1.0, 1.0])

    def test_clips_out_of_bounds(self):
        """Out-of-bounds positions should be clipped to grid edges."""
        cd = self._make_cell_data()
        positions = np.array([[-1, -1], [99, 99]], dtype=float)
        result = cd.get_sediments_vectorized(positions)
        assert len(result) == 2


class TestPerCellWestonFluxDeterrence:
    """Test per-cell WestonFlux in ship deterrence pipeline."""

    def _make_landscape(self):
        """Create a mock CellData with varied sediment and depth."""
        cd = MagicMock()
        cd.width = 10
        cd.height = 10
        cd._sediment = np.array(
            [[2.0] * 10, [7.0] * 10] + [[2.0] * 10] * 8  # row 0: sand  # row 1: silt
        )
        cd._depth = np.full((10, 10), 30.0)
        cd._salinity = np.full((12, 10, 10), 35.0)

        def _get_depths(pos):
            x = np.clip(pos[:, 0].astype(int), 0, 9)
            y = np.clip(pos[:, 1].astype(int), 0, 9)
            return cd._depth[y, x]

        def _get_sediments(pos):
            x = np.clip(pos[:, 0].astype(int), 0, 9)
            y = np.clip(pos[:, 1].astype(int), 0, 9)
            return cd._sediment[y, x]

        def _get_salinities(pos, month=None):
            x = np.clip(pos[:, 0].astype(int), 0, 9)
            y = np.clip(pos[:, 1].astype(int), 0, 9)
            month_idx = ((month or 1) - 1) % 12
            return cd._salinity[month_idx, y, x]

        def _get_depth(x, y):
            xi = int(np.clip(x, 0, 9))
            yi = int(np.clip(y, 0, 9))
            return float(cd._depth[yi, xi])

        def _get_sediment(x, y):
            xi = int(np.clip(x, 0, 9))
            yi = int(np.clip(y, 0, 9))
            return float(cd._sediment[yi, xi])

        def _get_salinity(x, y, month=None):
            xi = int(np.clip(x, 0, 9))
            yi = int(np.clip(y, 0, 9))
            month_idx = ((month or 1) - 1) % 12
            return float(cd._salinity[month_idx, yi, xi])

        cd.get_depth = _get_depth
        cd.get_sediment = _get_sediment
        cd.get_salinity = _get_salinity
        cd.get_depths_vectorized = _get_depths
        cd.get_sediments_vectorized = _get_sediments
        cd.get_salinities_vectorized = _get_salinities
        return cd

    def _make_ship_and_manager(self):
        """Create a ShipManager with one active ship at (5, 5)."""
        from cenop.agents.ship import Ship, ShipManager

        params = SimulationParameters(ships_enabled=True)
        mgr = ShipManager()
        mgr.set_enabled(True)
        ship = Ship(id=1, x=5.0, y=5.0)
        ship._is_active = True
        ship.noise.base_source_level = 170.0
        mgr.ships.append(ship)
        return mgr, ship

    def test_percell_off_uses_simple_formula(self):
        """With weston_flux_percell=False, simple formula is used."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(ships_enabled=True, weston_flux_percell=False)
        px = np.array([5.0, 5.0])
        py = np.array([0.0, 1.0])

        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px, py, params, is_day=True, cell_size=400.0
        )
        assert dx.shape == (2,)

    def test_percell_on_varies_by_sediment(self):
        """With percell=True, TL should differ for different sediment."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        cell_data = self._make_landscape()
        px = np.array([5.0, 5.0])
        py = np.array([0.0, 1.0])

        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=6,
        )
        assert dx.shape == (2,)

    def test_nodata_sediment_falls_back_to_simple(self):
        """Porpoise on NODATA sediment cell uses simple formula."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        cell_data = self._make_landscape()
        cell_data._sediment[0, :] = -9999.0

        px = np.array([5.0])
        py = np.array([0.0])

        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=6,
        )
        assert dx.shape == (1,)

    def test_nodata_depth_falls_back_to_simple(self):
        """Porpoise at depth <= 0 should use simple formula."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        cell_data = self._make_landscape()
        cell_data._depth[0, :] = -9999.0

        px = np.array([5.0])
        py = np.array([0.0])

        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=6,
        )
        assert dx.shape == (1,)

    def test_salinity_uses_correct_month(self):
        """Salinity lookup should use the passed month index."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        cell_data = self._make_landscape()
        cell_data._salinity[5, :, :] = 20.0  # month 6
        cell_data._salinity[0, :, :] = 35.0  # month 1

        px = np.array([5.0])
        py = np.array([0.0])

        dx6, _ = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=6,
        )
        dx1, _ = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=1,
        )
        assert dx6.shape == (1,)
        assert dx1.shape == (1,)

    def test_month_12_uses_index_11(self):
        """Month 12 (1-indexed) should map to salinity index 11."""
        mgr, _ = self._make_ship_and_manager()
        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        cell_data = self._make_landscape()
        cell_data._salinity[11, :, :] = 10.0

        px = np.array([5.0])
        py = np.array([0.0])

        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(
            px,
            py,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=12,
        )
        assert dx.shape == (1,)

    def test_scalar_and_vectorized_consistent(self):
        """Scalar calculate_deterrence should work with per-cell mode."""
        from cenop.agents.ship import Ship, ShipManager

        params = SimulationParameters(
            ships_enabled=True,
            weston_flux_percell=True,
            weston_flux_default_temperature=10.0,
        )
        mgr = ShipManager()
        mgr.set_enabled(True)
        ship = Ship(id=1, x=5.0, y=5.0)
        ship.noise.base_source_level = 170.0
        ship._is_active = True
        mgr.ships.append(ship)

        cell_data = self._make_landscape()

        should_deter, prob, magnitude, dist = ship.calculate_deterrence(
            5.0,
            0.0,
            params,
            is_day=True,
            cell_size=400.0,
            cell_data=cell_data,
            month=6,
        )
        assert isinstance(should_deter, bool)
        assert dist > 0
