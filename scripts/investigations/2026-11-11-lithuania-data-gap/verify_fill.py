"""Verify scripts/fill_lithuania_data_gap.py against the committed (pre-fill) files.

For every filled layer: existing tokens byte-identical to the pre-fill commit (BASE), header
identical, only water gap cells changed, filled-value distribution vs existing (mean, min, max, max
histogram gap), and no spatial correlation introduced.

Run from the repo root: python3 scripts/investigations/2026-11-11-lithuania-data-gap/verify_fill.py
"""

import subprocess
from pathlib import Path

import numpy as np

BASE = "cc80096~1"  # last commit before the fill (cc80096 committed the filled data)
REPO = Path(__file__).resolve().parents[3]
LAND = REPO / "data" / "Lithuania"
LAYERS = (
    ["patches.asc", "sediment.asc", "blocks.asc", "disttocoast.asc"]
    + [f"prey{m:02d}.asc" for m in range(1, 13)]
    + [f"salinity{m:02d}.asc" for m in range(1, 13)]
)


def split(raw):
    lines = raw.replace(b"\r\n", b"\n").rstrip(b"\n").split(b"\n")
    return lines[:6], np.array([ln.split(b" ") for ln in lines[6:]], dtype=object)


_, bathy = split((LAND / "bathy.asc").read_bytes())
water = bathy.astype(float) > 0

worst = 0.0
for name in LAYERS:
    old_raw = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE}:data/Lithuania/{name}"],
        capture_output=True,
        check=True,
    ).stdout
    old_hdr, old = split(old_raw)
    new_hdr, new = split((LAND / name).read_bytes())
    changed = old != new
    gap = water & (old == b"-9999")
    assert old_hdr == new_hdr, f"{name}: header changed"
    assert np.array_equal(changed, gap), f"{name}: changed cells != water gap cells"
    assert not (water & (new == b"-9999")).any(), f"{name}: NODATA left on water"
    existing = old[water & ~gap].astype(float)
    filled = new[gap].astype(float)
    line = (
        f"{name:16s} changed {int(changed.sum()):6d}  existing mean {existing.mean():8.3f} "
        f"[{existing.min():.2f},{existing.max():.2f}]  filled mean {filled.mean():8.3f} "
        f"[{filled.min():.2f},{filled.max():.2f}]"
    )
    if name != "disttocoast.asc":
        vals = np.union1d(existing, filled)
        h_old = np.array([(existing == v).mean() for v in vals])
        h_new = np.array([(filled == v).mean() for v in vals])
        gap_h = float(np.abs(h_old - h_new).max())
        worst = max(worst, gap_h)
        nv = new.astype(float)
        both = gap[:, :-1] & gap[:, 1:]
        r = (
            float(np.corrcoef(nv[:, :-1][both], nv[:, 1:][both])[0, 1])
            if existing.std() > 0
            else 0.0
        )
        line += f"  max bin diff {gap_h:.4f}  neighbour corr in gap {r:+.4f}"
    print(line)
print(f"all {len(LAYERS)} layers OK; worst histogram bin difference {worst:.4f}")
