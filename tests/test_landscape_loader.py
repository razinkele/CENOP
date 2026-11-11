"""Landscape loader: nodata on water cells is detected and reported.

A layer value of NODATA (-9999) on a water cell silently breaks the model there: salinity -9999
drives every CRW step to max_mov with ~90 degree turns, and food patches/prey -9999 mean no food.
Lithuania shipped with 19,767 such cells (27% of its water) after its bathymetry was enlarged in
7d50cfb without regenerating the other layers.
"""

import logging
from pathlib import Path

import numpy as np
import pytest

from cenop.landscape.loader import LandscapeLoader, nodata_on_water_counts

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
NODATA = -9999.0


def test_counts_only_water_cells_with_nodata():
    depth = np.array([[10.0, 10.0], [NODATA, 5.0]])
    layer = np.array([[NODATA, 1.0], [NODATA, 2.0]])  # nodata on one water + one land cell
    assert nodata_on_water_counts(depth, {"patches": layer}, NODATA) == {"patches": 1}


def test_monthly_layer_counts_a_cell_once_if_any_month_is_missing():
    depth = np.full((2, 2), 10.0)
    monthly = np.full((12, 2, 2), 7.5)
    monthly[0, 0, 0] = NODATA
    monthly[5, 0, 0] = NODATA
    monthly[11, 1, 1] = NODATA
    assert nodata_on_water_counts(depth, {"salinity": monthly}, NODATA) == {"salinity": 2}


def test_non_finite_values_on_water_count_as_missing():
    depth = np.full((1, 3), 10.0)
    layer = np.array([[np.nan, np.inf, 1.0]])
    assert nodata_on_water_counts(depth, {"prey": layer}, NODATA) == {"prey": 2}


def _write_asc(path, values):
    rows, cols = values.shape
    header = (
        f"NCOLS {cols}\nNROWS {rows}\nXLLCORNER 0\nYLLCORNER 0\nCELLSIZE 400\n"
        f"NODATA_value {int(NODATA)}\n"
    )
    body = "\n".join(" ".join(f"{v:.2f}" for v in row) for row in values)
    path.write_text(header + body + "\n")


def _tiny_landscape(root, salinity_gap):
    land = root / "Tiny"
    land.mkdir()
    depth = np.array([[NODATA, 10.0, 12.0], [8.0, 9.0, 11.0]])
    full = np.where(depth > 0, 1.0, NODATA)
    for name in ("bathy.asc", "disttocoast.asc", "sediment.asc", "patches.asc", "blocks.asc"):
        _write_asc(land / name, depth if name == "bathy.asc" else full)
    for month in range(1, 13):
        _write_asc(land / f"prey{month:02d}.asc", np.where(depth > 0, 0.5, NODATA))
        sal = np.where(depth > 0, 7.5, NODATA)
        if salinity_gap:
            sal[1, 2] = NODATA  # a water cell
        _write_asc(land / f"salinity{month:02d}.asc", sal)
    return root


def test_loader_warns_when_a_layer_has_nodata_on_water(tmp_path, caplog):
    root = _tiny_landscape(tmp_path, salinity_gap=True)
    with caplog.at_level(logging.WARNING, logger="cenop.landscape.loader"):
        LandscapeLoader("Tiny", root).load_all()
    messages = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Tiny" in m and "salinity" in m for m in messages), messages


def test_loader_is_silent_when_every_water_cell_has_data(tmp_path, caplog):
    root = _tiny_landscape(tmp_path, salinity_gap=False)
    with caplog.at_level(logging.WARNING, logger="cenop.landscape.loader"):
        LandscapeLoader("Tiny", root).load_all()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def _layers(data):
    return {
        "disttocoast": data["dist_to_coast"],
        "sediment": data["sediment"],
        "patches": data["food_prob"],
        "blocks": data["blocks"],
        "prey": data["entropy"],
        "salinity": data["salinity"],
    }


def test_lithuania_has_data_in_every_layer_on_every_water_cell():
    data = LandscapeLoader("Lithuania", DATA_DIR).load_all()
    layers = _layers(data)
    counts = nodata_on_water_counts(data["depth"], layers, data["metadata"].nodata_value)
    assert counts == dict.fromkeys(layers, 0)


@pytest.fixture(scope="module")
def central_baltic():
    if not (DATA_DIR / "CentralBaltic" / "bathy.asc").exists():
        pytest.skip("data/CentralBaltic not checked out (excluded from the CI sparse checkout)")
    return LandscapeLoader("CentralBaltic", DATA_DIR).load_all()


def test_central_baltic_has_data_in_every_layer_on_every_water_cell(central_baltic):
    layers = _layers(central_baltic)
    counts = nodata_on_water_counts(
        central_baltic["depth"], layers, central_baltic["metadata"].nodata_value
    )
    assert counts == dict.fromkeys(layers, 0)


def test_central_baltic_water_cells_carry_generator_values_not_land_values(central_baltic):
    """generate_central_baltic.py gives every water cell patches >= 0.3, sediment 1-3, a block
    id >= 1 and a positive distance to coast; 0 is its land value. The EMODnet bathymetry that
    later replaced bathy.asc has water where the generator had land (its 2% border band)."""
    water = central_baltic["depth"] > 0
    assert (central_baltic["food_prob"][water] >= 0.3).all()
    assert np.isin(central_baltic["sediment"][water], [1, 2, 3]).all()
    assert (central_baltic["blocks"][water] >= 1).all()
    assert (central_baltic["dist_to_coast"][water] > 0).all()
