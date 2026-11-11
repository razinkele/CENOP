"""Compare food-relevant maps between the v3.0 (CENOP data/Kattegat) and v3.2 landscapes."""

from pathlib import Path

import numpy as np

V30 = Path("/home/razinka/cenjas/CENOP/data/Kattegat")
V32 = Path("/home/razinka/cenjas/CENOP/.depons-cache/landscape/Kattegat")


def read_asc(p):
    hdr = {}
    with open(p) as fh:
        lines = fh.readlines()
    i = 0
    while lines[i].split()[0][0].isalpha():
        k, v = lines[i].split()[:2]
        hdr[k.lower()] = float(v)
        i += 1
    a = np.loadtxt(lines[i:], dtype=float)
    nod = hdr.get("nodata_value")
    if nod is not None:
        a = np.where(a == nod, np.nan, a)
    return hdr, a


def stats(a, mask=None):
    v = a[~np.isnan(a)] if mask is None else a[mask & ~np.isnan(a)]
    nz = v[v > 0]
    return f"n={v.size:6d} nz={nz.size:6d} sum={v.sum():10.2f} mean_nz={nz.mean() if nz.size else 0:.4f} max={v.max():.4f}"


for name in ("patches.asc", "bathy.asc"):
    h0, a0 = read_asc(V30 / name)
    h2, a2 = read_asc(V32 / name)
    print(f"{name:12s} v3.0 shape={a0.shape} {stats(a0)}")
    print(f"{'':12s} v3.2 shape={a2.shape} {stats(a2)}")
    if a0.shape == a2.shape:
        both = ~np.isnan(a0) & ~np.isnan(a2)
        d = (a0 != a2) & both
        print(f"{'':12s} cells differing: {d.sum()}  (nan-mask equal: {np.array_equal(np.isnan(a0), np.isnan(a2))})")
    vals0, vals2 = np.unique(a0[~np.isnan(a0)]), np.unique(a2[~np.isnan(a2)])
    if vals0.size < 20:
        print(f"{'':12s} unique v3.0 {vals0}  v3.2 {vals2[:20]}")

_, p0 = read_asc(V30 / "patches.asc")
_, p2 = read_asc(V32 / "patches.asc")
print("\nmonth | prey sum over patch cells (v3.0 vs v3.2) | prey sum all cells | ratio v3.2/v3.0")
for m in range(1, 13):
    _, a0 = read_asc(V30 / f"prey0000_{m:02d}.asc")
    _, a2 = read_asc(V32 / f"prey0000_{m:02d}_XX.asc")
    s0p = np.nansum(np.where(p0 > 0, a0, 0))
    s2p = np.nansum(np.where(p2 > 0, a2, 0))
    s0, s2 = np.nansum(a0), np.nansum(a2)
    print(f"  {m:2d}  | {s0p:10.2f} vs {s2p:10.2f} | {s0:10.2f} vs {s2:10.2f} | patch {s2p/s0p if s0p else float('nan'):.3f}  all {s2/s0:.3f}"
          f" | max {np.nanmax(a0):.3f} vs {np.nanmax(a2):.3f}")
