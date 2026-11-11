"""Verify scripts/fill_centralbaltic_data_gap.py against the pre-fill commit.

For every filled layer: header identical, changed cells == water cells in the generator band,
and the filled values follow the generator's rules (residuals against its deterministic part,
with the expected noise level).

Run from the repo root: python3 scripts/investigations/2026-11-11-centralbaltic-data-gap/verify_fill.py
"""

import subprocess
from pathlib import Path

import numpy as np

BASE = "a39cc08"  # CENOP-JASMINE before the CentralBaltic fill
REPO = Path(__file__).resolve().parents[3]
LAND = REPO / "data" / "CentralBaltic"


def split(raw):
    lines = raw.rstrip(b"\n").split(b"\n")
    return lines[:6], np.array([ln.split(b" ") for ln in lines[6:]], dtype=object)


def old_new(name):
    old_raw = subprocess.run(
        ["git", "-C", str(REPO), "show", f"{BASE}:data/CentralBaltic/{name}"],
        capture_output=True,
        check=True,
    ).stdout
    return split(old_raw), split((LAND / name).read_bytes())


_, bathy = split((LAND / "bathy.asc").read_bytes())
depth = bathy.astype(float)
nrows, ncols = depth.shape
water = depth > 0
X, Y = np.meshgrid(np.linspace(0, 1, ncols), np.linspace(0, 1, nrows))
edge = np.minimum(np.minimum(X, 1 - X), np.minimum(Y, 1 - Y))
gap = water & (edge < 0.02)
rows, cols = np.indices(depth.shape)
print(f"gap cells (water in generator band): {int(gap.sum())}")

filled = {}
names = ["patches.asc", "sediment.asc", "disttocoast.asc", "blocks.asc"]
names += [f"{p}{m:02d}.asc" for m in range(1, 13) for p in ("salinity", "prey")]
for name in names:
    (old_hdr, old), (new_hdr, new) = old_new(name)
    assert old_hdr == new_hdr, f"{name}: header changed"
    changed = old != new
    assert np.array_equal(changed, gap), f"{name}: changed cells != gap cells"
    filled[name] = new[gap].astype(float)
print(f"all {len(names)} layers: header identical, only the {int(gap.sum())} gap cells changed")

d = depth[gap]
p = filled["patches.asc"]
boost = np.maximum((1 - edge[gap] / 0.15) * 0.3, 0)
resid_p = p - np.minimum(0.3 + boost, 1)
print(
    f"patches: range [{p.min():.3f}, {p.max():.3f}]; patches - (0.3 + boost) in "
    f"[{resid_p.min():.3f}, {resid_p.max():.3f}] (expect U(0, 0.5), capped at 1)"
)
sed_rule = np.select([d > 50, d > 20], [1.0, 2.0], 3.0)
print(
    f"sediment == depth rule: {(filled['sediment.asc'] == sed_rule).mean():.3f}; "
    f"disttocoast == 50*depth (6 dp): "
    f"{np.isclose(filled['disttocoast.asc'], np.round(50 * d, 6)).mean():.3f}"
)
blocks_rule = (rows[gap] // 50) * (ncols // 50) + cols[gap] // 50 + 1
print(f"blocks == 50x50 numbering: {(filled['blocks.asc'] == blocks_rule).mean():.3f}")
base = 8 + 10 * (1 - X[gap]) * (1 - Y[gap])
for m in (1, 4, 7, 10):
    s = filled[f"salinity{m:02d}.asc"]
    rs = s - np.clip(base * (1 - 0.15 * np.sin((m - 4) * np.pi / 6)), 2, 25)
    pr = filled[f"prey{m:02d}.asc"]
    rp = pr - np.clip(p * (0.7 + 0.3 * np.sin((m - 3) * np.pi / 6)), 0, 1)
    inner = (pr > 0) & (pr < 1)
    print(
        f"m{m:02d} salinity resid mean {rs.mean():+.3f} sd {rs.std():.3f} (expect 0, 0.5) | "
        f"prey resid (unclipped cells) mean {rp[inner].mean():+.3f} sd {rp[inner].std():.3f} "
        f"(expect ~0, <=0.1)"
    )
