"""Resample a loaded landscape to DEPONS's 400 m cells.

DEPONS is calibrated for 400 m cells and CENOP assumes them throughout (noise distances, the
random-walk step, dispersal distances, spatial memory, per-cell food). A landscape stored on
another grid is brought to 400 m when it is loaded, so the model code keeps one unit.

Design: docs/superpowers/specs/2026-11-19-landscapes-400m-design.md
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

CELL_M = 400.0

# Layers resampled by nearest neighbour; "dist_to_coast" is recomputed instead.
_NEAREST_LAYERS = ("depth", "sediment", "food_prob", "blocks", "entropy", "salinity")


def resample_to_400m(data: dict[str, Any]) -> dict[str, Any]:
    """Return ``data`` (the dict of ``LandscapeLoader.load_all``) on a 400 m grid.

    A landscape that already has 400 m cells is returned as is. Otherwise:

    - every layer except distance to coast is resampled by nearest neighbour: each new cell
      takes the value of the old cell its centre lies in, so the coastline, NODATA and the
      layers stay consistent with each other and no values are invented;
    - distance to coast is recomputed from the resampled water mask (depth > 0), in metres to
      the nearest non-water cell; non-water cells get NODATA;
    - the lower-left corner is kept and the grid size becomes the old size times
      ``cellsize / 400``, rounded.

    Arrays have row 0 in the south (as the loader returns them). The input is not modified.
    """
    meta = data["metadata"]
    if meta.cellsize == CELL_M:
        return data

    factor = meta.cellsize / CELL_M
    old_h, old_w = data["depth"].shape
    new_h, new_w = int(round(old_h * factor)), int(round(old_w * factor))
    rows = np.minimum(((np.arange(new_h) + 0.5) / factor).astype(np.intp), old_h - 1)
    cols = np.minimum(((np.arange(new_w) + 0.5) / factor).astype(np.intp), old_w - 1)

    out = dict(data)
    for key in _NEAREST_LAYERS:
        layer = data.get(key)
        if layer is not None:
            out[key] = np.ascontiguousarray(layer[..., rows[:, None], cols[None, :]])

    if data.get("dist_to_coast") is not None:
        from scipy import ndimage

        water = out["depth"] > 0
        dist = ndimage.distance_transform_edt(water) * CELL_M
        out["dist_to_coast"] = np.where(water, dist, meta.nodata_value).astype(
            data["dist_to_coast"].dtype
        )

    out["metadata"] = replace(meta, ncols=new_w, nrows=new_h, cellsize=CELL_M)
    logger.info(
        "Landscape resampled from %g m cells (%d x %d) to 400 m cells (%d x %d)",
        meta.cellsize,
        old_w,
        old_h,
        new_w,
        new_h,
    )
    return out
