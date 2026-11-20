"""Build DEPONS-style food layers (patches and monthly prey) from seasonal density maps.

DEPONS 3.2 (TRACE, section 2.6): food is found only in food patches, single 400 m cells
placed at random (1.6% of the water in Kattegat). The maximum food in a patch follows twelve
monthly maps derived from seasonal maps of relative porpoise density: the seasonal values are
placed at their months and the other months interpolated per cell with a curve
a + b sin(x) + c cos(x), x = 2 pi month / 12; each month is then scaled to a mean of
0.38592 over the water.

This module holds the arithmetic; scripts/build_food_layers.py reads source rasters and
writes the landscape files.
"""

from __future__ import annotations

import numpy as np

DEPONS_MEAN_PREY = 0.38592  # mean of every monthly prey map (TRACE 2.6)
DEPONS_PATCH_SHARE = 0.016  # share of water cells that are food patches (Kattegat)
MIN_PREY = 1e-3  # interpolation can undershoot zero; food must stay positive on water


def monthly_from_seasons(seasons: dict[float, np.ndarray]) -> np.ndarray:
    """Twelve monthly maps from maps placed at their months (1-12; fractions allowed, e.g. 7.5
    for the centre of a May-October average).

    Three or more months: least-squares fit of a + b sin(x) + c cos(x) per cell (exact for
    three). Two months six apart: a + A cos(x - x_first), exact at both. One month: constant.
    Returns an array of shape (12, ...), month 1 first.
    """
    months = sorted(seasons)
    vals = np.stack([np.asarray(seasons[m], dtype=np.float64) for m in months])
    x_all = 2.0 * np.pi * np.arange(1, 13) / 12.0
    if len(months) == 1:
        return np.repeat(vals, 12, axis=0)
    if len(months) == 2:
        m1, m2 = months
        if abs((m2 - m1) % 12 - 6) > 1e-9:
            raise ValueError("two seasonal maps must be six months apart")
        mean = 0.5 * (vals[0] + vals[1])
        amp = 0.5 * (vals[0] - vals[1])
        phase = 2.0 * np.pi * m1 / 12.0
        return mean[None] + amp[None] * np.cos(x_all - phase)[(...,) + (None,) * mean.ndim]
    x = 2.0 * np.pi * np.array(months) / 12.0
    design = np.column_stack([np.ones_like(x), np.sin(x), np.cos(x)])
    flat = vals.reshape(len(months), -1)
    coef, *_ = np.linalg.lstsq(design, flat, rcond=None)
    design_all = np.column_stack([np.ones(12), np.sin(x_all), np.cos(x_all)])
    return (design_all @ coef).reshape((12,) + vals.shape[1:])


def standardize(prey: np.ndarray, water: np.ndarray, mean: float = DEPONS_MEAN_PREY) -> np.ndarray:
    """Scale each month to ``mean`` over the water cells; values below MIN_PREY are raised.

    Cells outside the water are left unchanged (the caller writes NODATA there).
    """
    out = np.array(prey, dtype=np.float64)
    for m in range(out.shape[0]):
        month = np.maximum(out[m], MIN_PREY)
        scale = mean / month[water].mean()
        out[m] = np.where(water, month * scale, out[m])
    return out


def random_patches(
    water: np.ndarray, share: float = DEPONS_PATCH_SHARE, seed: int = 0
) -> np.ndarray:
    """0/1 map with ``round(share * water cells)`` patch cells chosen at random among water."""
    idx = np.flatnonzero(water)
    n = int(round(share * idx.size))
    chosen = np.random.default_rng(seed).choice(idx, size=n, replace=False)
    patches = np.zeros(water.shape, dtype=np.int8)
    patches.flat[chosen] = 1
    return patches


def floor_deep_gaps(
    values: np.ndarray, water: np.ndarray, depth: np.ndarray, deeper_than: float, quantile: float
) -> np.ndarray:
    """Water cells deeper than ``deeper_than`` without a value get the ``quantile`` of the valid
    water values (for a source that left deep water out of its model, as SAMBAH did > 80 m)."""
    valid = np.isfinite(values) & water
    if not valid.any():
        return values
    low = float(np.quantile(values[valid], quantile))
    return np.where(water & ~np.isfinite(values) & (depth > deeper_than), low, values)


def fill_water_gaps(values: np.ndarray, water: np.ndarray) -> np.ndarray:
    """Give water cells without a value (NaN) the value of the nearest cell that has one."""
    from scipy import ndimage

    missing = ~np.isfinite(values)
    if not (missing & water).any():
        return values
    if missing.all():
        raise ValueError("no valid values to fill from")
    _, (iy, ix) = ndimage.distance_transform_edt(missing, return_indices=True)
    filled = values[iy, ix]
    return np.where(water & missing, filled, values)
