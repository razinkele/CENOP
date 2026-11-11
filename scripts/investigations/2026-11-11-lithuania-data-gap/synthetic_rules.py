"""Are the Lithuania habitat layers iid synthetic noise? Recover their generating rules.

Checks, on water cells that have data: spatial autocorrelation with the right/lower neighbour,
flatness of the value histogram, per-month ranges, the exact disttocoast plane and what that
plane gives when extended over the gap cells.

Describes the data BEFORE scripts/fill_lithuania_data_gap.py (as of 42349a7). After the fill there
is no gap left; check out the pre-fill data/Lithuania files to reproduce.

Run: PYTHONPATH=src python3 -s scripts/investigations/2026-11-11-lithuania-data-gap/synthetic_rules.py
"""

import numpy as np
from scipy import ndimage

from cenop.agents.population import _main_water_body
from cenop.landscape.cell_data import CellData

land = CellData("Lithuania")
land.load()
depth = land._depth
water = depth > 0
gap = water & (land._salinity[0] <= -9000)
valid = water & ~gap
rows, cols = np.indices(depth.shape)


def neighbour_corr(a):
    out = []
    for sl_a, sl_b in (
        ((slice(None), slice(0, -1)), (slice(None), slice(1, None))),
        ((slice(0, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        both = valid[sl_a] & valid[sl_b]
        out.append(float(np.corrcoef(a[sl_a][both], a[sl_b][both])[0, 1]))
    return out


def flatness(v):
    _, counts = np.unique(v, return_counts=True)
    inner = counts[1:-1] if len(counts) > 2 else counts
    return float(inner.min() / inner.max()), counts[0], counts[-1], float(inner.mean())


for name, arr in (
    ("patches", land._food_prob),
    ("prey m1", land._entropy[0]),
    ("salinity m1", land._salinity[0]),
):
    a = np.asarray(arr, dtype=np.float64)
    v = a[valid]
    fr, c0, cn, cm = flatness(v)
    print(
        f"{name:12s} neighbour corr (right, down) {neighbour_corr(a)}  "
        f"histogram inner min/max {fr:.2f}; end bins {c0}, {cn} vs inner mean {cm:.0f}"
    )

print("\nper-month ranges on valid water (prey | salinity):")
for m in range(12):
    p = land._entropy[m][valid]
    s = land._salinity[m][valid]
    print(
        f"  m{m + 1:02d} prey [{p.min():.2f}, {p.max():.2f}] mean {p.mean():.3f} | "
        f"salinity [{s.min():.2f}, {s.max():.2f}] mean {s.mean():.3f}"
    )
same = all(
    np.array_equal(
        np.argsort(land._salinity[0][valid], kind="stable"),
        np.argsort(land._salinity[m][valid], kind="stable"),
    )
    for m in (3, 6)
)
ratio = land._salinity[3][valid] / land._salinity[0][valid]
print(
    f"salinity m4/m1 ratio range [{ratio.min():.4f}, {ratio.max():.4f}]  "
    f"(same ordering across months: {same})"
)
pd = land._entropy[6][valid] - land._entropy[0][valid]
print(f"prey m7 - m1 range [{pd.min():.4f}, {pd.max():.4f}]")

dc = np.asarray(land._dist_to_coast, dtype=np.float64)
X = np.column_stack([np.ones(int(valid.sum())), rows[valid], cols[valid]])
coef, *_ = np.linalg.lstsq(X, dc[valid], rcond=None)
resid = dc[valid] - X @ coef
print(
    f"\ndisttocoast = {coef[0]:.6f} + {coef[1]:.6f}*row + {coef[2]:.6f}*col   "
    f"max |resid| {np.abs(resid).max():.4f}"
)
ext = coef[0] + coef[1] * rows[gap] + coef[2] * cols[gap]
print(
    f"plane extended over gap cells: range [{ext.min():.2f}, {ext.max():.2f}]  "
    f"(existing valid range [{dc[valid].min():.2f}, {dc[valid].max():.2f}])"
)

edt = ndimage.distance_transform_edt(depth > 0)
main = _main_water_body(depth >= 1.0)
west = gap & (cols < 55)
east = gap & (cols >= 270)
inside = gap & ~west & ~east
for label, sub in (
    ("west strip", west),
    ("east of old footprint", east),
    ("inside old footprint", inside),
):
    print(
        f"{label:22s} {int(sub.sum()):6d} cells; in main body {int((sub & main).sum()):6d}; "
        f"dist to new coast median {np.median(edt[sub]):.0f} km"
    )
