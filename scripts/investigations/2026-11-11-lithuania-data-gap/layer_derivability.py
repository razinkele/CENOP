"""Can the padded Lithuania layers be regenerated from the new bathymetry?

For each non-bathy layer, on water cells that DO have data, test whether the value is a
deterministic function of depth, of position, or of distance to coast. Also check whether the
old (pre-7d50cfb) domain was clipped mid-sea at its edges (result: the west edge was 91% water).

Describes the data BEFORE scripts/fill_lithuania_data_gap.py (as of 42349a7). After the fill there
is no gap left; check out the pre-fill data/Lithuania files to reproduce.

Run: PYTHONPATH=src python3 -s scripts/investigations/2026-11-11-lithuania-data-gap/layer_derivability.py
"""

from pathlib import Path

import numpy as np
from scipy import ndimage

from cenop.landscape.cell_data import CellData

DATA = Path(__file__).resolve().parents[3] / "data" / "Lithuania"

land = CellData("Lithuania")
land.load()
depth = land._depth
water = depth > 0
gap = water & (land._salinity[0] <= -9000)
valid = water & ~gap
rows, cols = np.indices(depth.shape)
print(f"water {int(water.sum())}  valid {int(valid.sum())}  gap {int(gap.sum())}")


def r2_linear(y, *xs):
    X = np.column_stack([np.ones_like(y)] + list(xs))
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    return 1.0 - resid.var() / y.var() if y.var() > 0 else float("nan")


def depth_function_spread(vals, d):
    """Max std of the layer among cells sharing the same depth value (0 = pure function of depth)."""
    order = np.argsort(d)
    d_sorted, v_sorted = d[order], vals[order]
    _, starts = np.unique(d_sorted, return_index=True)
    groups = np.split(v_sorted, starts[1:])
    multi = [g for g in groups if len(g) > 1]
    return max((float(g.std()) for g in multi), default=0.0), len(multi)


layers = {
    "patches(food_prob)": land._food_prob,
    "disttocoast": land._dist_to_coast,
    "sediment": land._sediment,
    "blocks": land._blocks,
}
for m in (0, 6):
    layers[f"prey m{m + 1}"] = land._entropy[m]
    layers[f"salinity m{m + 1}"] = land._salinity[m]

d = depth[valid]
r, c = rows[valid].astype(float), cols[valid].astype(float)
for name, arr in layers.items():
    v = np.asarray(arr, dtype=np.float64)[valid]
    uniq = np.unique(v)
    spread, ngroups = depth_function_spread(v, d)
    print(
        f"{name:20s} unique {len(uniq):6d}  range [{v.min():.4g}, {v.max():.4g}]  "
        f"R2(depth) {r2_linear(v, d):.3f}  R2(row,col) {r2_linear(v, r, c):.3f}  "
        f"R2(depth,row,col,quad) {r2_linear(v, d, r, c, r * r, c * c, r * c):.3f}  "
        f"same-depth max std {spread:.4g} ({ngroups} groups)"
    )
    if len(uniq) <= 6:
        print(f"{'':20s} values: {uniq}")

# disttocoast: recompute from the new bathymetry and compare on valid cells
land_mask = depth <= 0
edt_cells = ndimage.distance_transform_edt(~land_mask)
dc = np.asarray(land._dist_to_coast, dtype=np.float64)
for label, cand in (
    ("EDT new bathy (cells)", edt_cells),
    ("EDT new bathy (cells-1)", edt_cells - 1),
):
    diff = cand[valid] - dc[valid]
    print(
        f"disttocoast vs {label:24s} corr {np.corrcoef(cand[valid], dc[valid])[0, 1]:.4f}  "
        f"mean diff {diff.mean():.3f}  max |diff| {np.abs(diff).max():.3f}  "
        f"share |diff|<=0.5 {(np.abs(diff) <= 0.5).mean():.3f}"
    )


# old domain: was its eastern edge water (clipped mid-sea)?
def read_asc(p):
    with open(p) as f:
        for _ in range(6):
            f.readline()
        return np.loadtxt(f)


old = read_asc(DATA / "bathy.asc.old")
for label, colidx in (("west edge col 0", 0), ("east edge col -1", -1)):
    edge = old[:, colidx]
    print(f"old bathy {label}: water share {(edge > 0).mean():.2f} of {len(edge)} rows")
print(
    f"old bathy north edge row 0: water share {(old[0] > 0).mean():.2f}; "
    f"south edge row -1: {(old[-1] > 0).mean():.2f}"
)

# Where does the gap sit relative to the new coastline?
edt_gap = edt_cells[gap]
print(
    f"gap cells: distance to new coast (cells) median {np.median(edt_gap):.1f}, "
    f"share within 5 km {(edt_gap <= 5).mean():.2f}, within 25 km {(edt_gap <= 25).mean():.2f}"
)
west = gap & (cols < 55)
east = gap & (cols >= 215 + 55)
inside = gap & ~west & ~east
print(
    f"gap split: west strip {int(west.sum())}, east of old footprint {int(east.sum())}, "
    f"inside old footprint (refined coast/lagoon) {int(inside.sum())}"
)
