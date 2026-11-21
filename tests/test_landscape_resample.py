"""Landscapes with a cell size other than 400 m are resampled to 400 m when loaded.

Design: docs/superpowers/specs/2026-11-19-landscapes-400m-design.md
"""

from pathlib import Path

import numpy as np
import pytest

from cenop.landscape.cell_data import CellData, LandscapeMetadata
from cenop.landscape.loader import LandscapeLoader
from cenop.landscape.resample import resample_to_400m

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NODATA = -9999.0


def _data(cellsize, depth):
    h, w = depth.shape
    water = depth > 0
    return {
        "metadata": LandscapeMetadata(
            ncols=w, nrows=h, xllcorner=1000.0, yllcorner=2000.0, cellsize=cellsize
        ),
        "depth": depth.copy(),
        "dist_to_coast": np.where(water, 7.0, NODATA),
        "sediment": np.where(water, 1.0, NODATA),
        "food_prob": np.where(water, np.arange(h * w).reshape(h, w) / 100.0, NODATA),
        "blocks": np.arange(h * w).reshape(h, w),
        "entropy": np.stack([np.where(water, m + 0.5, NODATA) for m in range(12)]),
        "salinity": np.stack([np.where(water, 7.0 + m, NODATA) for m in range(12)]),
    }


@pytest.fixture
def coarse():
    # 1 km cells; row 0 is the southern row; one land cell in the south-west corner
    depth = np.array([[NODATA, 10.0], [20.0, 30.0]])
    return _data(1000.0, depth)


class TestResample:
    def test_400_m_input_is_returned_untouched(self):
        data = _data(400.0, np.array([[NODATA, 10.0], [20.0, 30.0]]))
        out = resample_to_400m(data)
        assert out is data

    def test_grid_and_metadata(self, coarse):
        out = resample_to_400m(coarse)
        meta = out["metadata"]
        assert (meta.ncols, meta.nrows, meta.cellsize) == (5, 5, 400.0)
        assert (meta.xllcorner, meta.yllcorner) == (1000.0, 2000.0)
        assert meta.nodata_value == NODATA
        for key in ("depth", "dist_to_coast", "sediment", "food_prob", "blocks"):
            assert out[key].shape == (5, 5)
        assert out["entropy"].shape == out["salinity"].shape == (12, 5, 5)

    def test_each_new_cell_takes_the_old_cell_its_centre_lies_in(self, coarse):
        out = resample_to_400m(coarse)
        # centres at 200, 600, 1000, 1400, 1800 m -> old index 0, 0, 1, 1, 1
        idx = np.array([0, 0, 1, 1, 1])
        np.testing.assert_array_equal(out["depth"], coarse["depth"][idx][:, idx])
        np.testing.assert_array_equal(out["blocks"], coarse["blocks"][idx][:, idx])
        np.testing.assert_array_equal(out["food_prob"], coarse["food_prob"][idx][:, idx])
        np.testing.assert_array_equal(out["salinity"][3], coarse["salinity"][3][idx][:, idx])
        assert out["blocks"].dtype == coarse["blocks"].dtype

    def test_layers_stay_consistent_with_the_water_mask(self, coarse):
        out = resample_to_400m(coarse)
        water = out["depth"] > 0
        for key in ("sediment", "food_prob", "dist_to_coast"):
            assert ((out[key] == NODATA) == ~water).all(), key
        assert ((out["entropy"] == NODATA) == ~water[None]).all()

    def test_distance_to_coast_is_recomputed_in_metres(self, coarse):
        out = resample_to_400m(coarse)
        dtc = out["dist_to_coast"]
        # land occupies new cells [0:2, 0:2]
        assert dtc[0, 2] == 400.0 and dtc[2, 0] == 400.0  # next to land
        assert dtc[0, 4] == 1200.0  # three cells east of it
        assert dtc[4, 4] == pytest.approx(400.0 * np.hypot(3, 3))
        assert (dtc[:2, :2] == NODATA).all()

    def test_finer_grid_is_coarsened(self):
        depth = np.full((8, 8), 10.0)
        depth[0, 0] = NODATA
        out = resample_to_400m(_data(250.0, depth))
        assert out["depth"].shape == (5, 5)
        assert out["metadata"].cellsize == 400.0

    def test_input_is_not_modified(self, coarse):
        before = coarse["depth"].copy()
        resample_to_400m(coarse)
        np.testing.assert_array_equal(coarse["depth"], before)
        assert coarse["metadata"].cellsize == 1000.0


class TestLoader:
    def test_lithuania_loads_on_a_400_m_grid(self):
        data = LandscapeLoader("Lithuania", DATA_DIR).load_all()
        meta = data["metadata"]
        assert (meta.ncols, meta.nrows, meta.cellsize) == (822, 952, 400.0)
        assert (meta.xllcorner, meta.yllcorner) == (4754000.0, 3482000.0)
        water_km2 = (data["depth"] > 0).sum() * 0.16
        assert water_km2 == pytest.approx(83873.0, rel=0.01)  # 83,873 water cells of 1 km2
        dtc = data["dist_to_coast"][data["depth"] > 0]
        assert dtc.min() == 400.0 and dtc.max() > 50_000

    def test_kattegat_is_not_resampled(self):
        data = LandscapeLoader("Kattegat", DATA_DIR).load_all()
        meta = data["metadata"]
        assert (meta.ncols, meta.nrows, meta.cellsize) == (600, 1000, 400.0)

    def test_celldata_reports_the_resampled_grid(self):
        cd = CellData("Lithuania", DATA_DIR)
        cd.load()
        assert (cd.width, cd.height) == (822, 952)
        assert cd._food_value.shape == (952, 822)


def test_lithuania_run_with_a_wind_farm_uses_the_400_m_grid():
    from cenop.core.simulation import Simulation
    from cenop.parameters.simulation_params import SimulationParameters

    cd = CellData("Lithuania", DATA_DIR)
    cd.load()
    params = SimulationParameters(
        porpoise_count=300,
        sim_years=1,
        random_seed=42,
        landscape="Lithuania",
        turbines="CuronianNord_35_15MW",
    )
    sim = Simulation(params=params, cell_data=cd, seed=42)
    sim.initialize()
    turbines = sim._turbine_manager.turbines
    tx = np.array([t.x for t in turbines])
    ty = np.array([t.y for t in turbines])
    assert len(turbines) == 35
    assert (cd._depth[ty.astype(int), tx.astype(int)] > 0).all()
    # the farm spans about 13 km: 33 cells of 400 m (it was 13 cells of 1 km)
    assert 25 < tx.max() - tx.min() < 40

    pm = sim.population_manager
    for _ in range(48):
        sim.step()
    a = pm.active_mask
    assert np.isfinite(pm.energy[a]).all()
    assert ((pm.x[a] >= 0) & (pm.x[a] < 822) & (pm.y[a] >= 0) & (pm.y[a] < 952)).all()


def test_central_baltic_is_resampled_to_400_m():
    if not (DATA_DIR / "CentralBaltic" / "bathy.asc").exists():
        pytest.skip("data/CentralBaltic not checked out (excluded from the CI sparse checkout)")
    data = LandscapeLoader("CentralBaltic", DATA_DIR).load_all()
    meta = data["metadata"]
    assert (meta.ncols, meta.nrows, meta.cellsize) == (1538, 1692, 400.0)  # from 615 x 677 at 1 km
    water_km2 = (data["depth"] > 0).sum() * 0.16
    assert water_km2 == pytest.approx(201414, rel=0.01)
    assert data["salinity"].shape == (12, 1692, 1538)
    assert data["food_prob"].shape == (1692, 1538)  # patches written on the 400 m grid
