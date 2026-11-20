"""Food layers (patches, prey) already on the 400 m grid of a landscape stored on another grid.

DEPONS food patches are single 400 m cells, so a landscape whose other layers are on 1 km or
250 m cells can carry its patches (and prey) as 400 m files with the same lower-left corner;
they are used as written and the other layers are resampled as before.
"""

import numpy as np
import pytest

from cenop.landscape.loader import LandscapeLoader

NODATA = -9999


def _write(path, arr, cellsize, xll=1000.0, yll=2000.0):
    h, w = arr.shape
    header = (
        f"ncols {w}\nnrows {h}\nxllcorner {xll}\nyllcorner {yll}\n"
        f"cellsize {cellsize}\nNODATA_value {NODATA}\n"
    )
    np.savetxt(path, np.flipud(arr), fmt="%g", header=header.rstrip("\n"), comments="")


def _landscape(tmp_path, patches=None, prey=None, prey_cell=1000.0):
    land = tmp_path / "Mixed"
    land.mkdir()
    depth = np.full((2, 2), 20.0)
    depth[0, 0] = NODATA  # south-west corner is land
    for name, value in (("bathy", depth), ("disttocoast", 5.0), ("sediment", 1.0), ("blocks", 1)):
        arr = depth if name == "bathy" else np.where(depth > 0, value, NODATA)
        _write(land / f"{name}.asc", arr, 1000.0)
    _write(
        land / "patches.asc",
        patches if patches is not None else np.ones((2, 2)),
        400.0 if patches is not None else 1000.0,
    )
    for m in range(1, 13):
        _write(land / f"salinity{m:02d}.asc", np.full((2, 2), 7.0), 1000.0)
        p = prey(m) if prey is not None else np.full((2, 2), 0.5)
        _write(land / f"prey{m:02d}.asc", p, prey_cell)
    return land


def test_400_m_patches_are_used_as_written(tmp_path):
    patches = np.zeros((5, 5))
    patches[2, 3] = 1.0  # a single 400 m patch
    _landscape(tmp_path, patches=patches)
    data = LandscapeLoader("Mixed", str(tmp_path)).load_all()
    assert data["depth"].shape == (5, 5)  # 2 km at 400 m cells
    assert np.array_equal(data["food_prob"], patches)


def test_400_m_prey_is_used_as_written(tmp_path):
    def prey(m):
        a = np.full((5, 5), 0.1 * m)
        a[4, 4] = 0.9
        return a

    _landscape(tmp_path, patches=np.ones((5, 5)), prey=prey, prey_cell=400.0)
    data = LandscapeLoader("Mixed", str(tmp_path)).load_all()
    assert data["entropy"].shape == (12, 5, 5)
    assert data["entropy"][6, 0, 0] == pytest.approx(0.7)
    assert data["entropy"][0, 4, 4] == pytest.approx(0.9)


def test_400_m_layer_on_the_wrong_grid_is_refused(tmp_path):
    _landscape(tmp_path, patches=np.ones((4, 5)))  # 4 rows, the 400 m grid has 5
    with pytest.raises(ValueError, match="patches.asc"):
        LandscapeLoader("Mixed", str(tmp_path)).load_all()


def test_native_grid_layers_are_still_resampled(tmp_path):
    _landscape(tmp_path)
    data = LandscapeLoader("Mixed", str(tmp_path)).load_all()
    assert data["food_prob"].shape == (5, 5)
    assert data["entropy"].shape == (12, 5, 5)
