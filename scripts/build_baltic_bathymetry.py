"""Build bathy.asc, disttocoast.asc and blocks.asc for a Baltic landscape from EMODnet.

Downloads the EMODnet Bathymetry digital terrain model (WCS coverage emodnet:mean, mean
elevation, WGS84, about 115 m) in tiles, merges them and reprojects them onto an EPSG:3035 grid
with average resampling. Water is where the mean elevation is below 0; depth = -elevation;
land is NODATA. Distance to coast is the Euclidean distance from each water cell to the nearest
non-water cell, in metres. Blocks are squares of --block-km numbered 1, 2, ... over the grid
(only water cells carry a block), for counting animals per area.

EMODnet Bathymetry: CC BY 4.0, "EMODnet Bathymetry Consortium (year): EMODnet Digital Bathymetry
(DTM)". DEPONS 3.2 used the same product (TRACE 2.6).

Usage (from the repository root):

  python3 scripts/build_baltic_bathymetry.py --out data/CentralBaltic \\
      --lon 13 22 --lat 53.9 59.5 --cellsize 1000 --tiles-dir <scratch dir>
"""

from __future__ import annotations

import argparse
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

WCS = "https://ows.emodnet-bathymetry.eu/wcs"
NODATA = -9999


def fetch_tile(lon0, lon1, lat0, lat1, path: Path) -> None:
    if path.exists() and path.stat().st_size > 1000:
        return
    query = urllib.parse.urlencode(
        [
            ("service", "WCS"),
            ("version", "2.0.1"),
            ("request", "GetCoverage"),
            ("CoverageId", "emodnet:mean"),
            ("format", "image/tiff"),
            ("subset", f"Long({lon0},{lon1})"),
            ("subset", f"Lat({lat0},{lat1})"),
        ]
    )
    for attempt in range(4):
        try:
            with urllib.request.urlopen(f"{WCS}?{query}", timeout=300) as r:
                data = r.read()
            if data[:2] in (b"II", b"MM"):
                path.write_bytes(data)
                return
            raise RuntimeError(f"not a GeoTIFF: {data[:200]!r}")
        except Exception as e:  # network hiccups: retry with a pause
            if attempt == 3:
                raise
            print(f"   retry {lon0},{lat0}: {e}")
            time.sleep(10 * (attempt + 1))


def write_asc(path: Path, arr: np.ndarray, xll: float, yll: float, cell: float, fmt: str) -> None:
    """``arr`` has row 0 in the north (as the ASC file)."""
    h, w = arr.shape
    header = (
        f"ncols {w}\nnrows {h}\nxllcorner {xll:.1f}\nyllcorner {yll:.1f}\n"
        f"cellsize {cell:g}\nNODATA_value {NODATA}"
    )
    np.savetxt(path, arr, fmt=fmt, header=header, comments="")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lon", type=float, nargs=2, required=True)
    ap.add_argument("--lat", type=float, nargs=2, required=True)
    ap.add_argument("--cellsize", type=float, default=1000.0)
    ap.add_argument("--block-km", type=float, default=40.0)
    ap.add_argument("--tile-deg", type=float, default=1.5)
    ap.add_argument("--tiles-dir", required=True)
    ap.add_argument(
        "--fill-gaps-of",
        default=None,
        metavar="BATHY_ASC",
        help="keep this existing bathy.asc (its grid and every water cell) and add only the "
        "cells it lacks where EMODnet has water",
    )
    args = ap.parse_args()

    import rasterio
    from pyproj import Transformer
    from rasterio.merge import merge
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject
    from scipy import ndimage

    tiles = Path(args.tiles_dir)
    tiles.mkdir(parents=True, exist_ok=True)
    lon0, lon1 = args.lon
    lat0, lat1 = args.lat
    pad = 0.3  # degrees beyond the box, so the reprojected corners are covered
    paths = []
    lons = np.arange(lon0 - pad, lon1 + pad, args.tile_deg)
    lats = np.arange(lat0 - pad, lat1 + pad, args.tile_deg)
    for a in lons:
        for b in lats:
            p = tiles / f"emodnet_mean_{a:.2f}_{b:.2f}.tif"
            fetch_tile(round(a, 4), round(min(a + args.tile_deg, lon1 + pad), 4),
                       round(b, 4), round(min(b + args.tile_deg, lat1 + pad), 4), p)  # fmt: skip
            paths.append(p)
            print(f"tile {p.name}: {p.stat().st_size / 1e6:.1f} MB")

    existing = None
    if args.fill_gaps_of:
        with open(args.fill_gaps_of) as fh:
            head = dict(next(fh).split() for _ in range(6))
        head = {k.lower(): float(v) for k, v in head.items()}
        existing = np.loadtxt(args.fill_gaps_of, skiprows=6)  # row 0 is north
    # Target grid: the box's EPSG:3035 envelope, snapped outwards to whole cells
    t = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    edge_lon = np.r_[
        np.linspace(lon0, lon1, 60),
        np.full(60, lon1),
        np.linspace(lon1, lon0, 60),
        np.full(60, lon0),
    ]
    edge_lat = np.r_[np.full(60, lat0), np.linspace(lat0, lat1, 60), np.full(60, lat1), np.linspace(lat1, lat0, 60)]  # fmt: skip
    ex, ey = t.transform(edge_lon, edge_lat)
    c = args.cellsize
    xll, yll = math.floor(ex.min() / c) * c, math.floor(ey.min() / c) * c
    w, h = math.ceil((ex.max() - xll) / c), math.ceil((ey.max() - yll) / c)
    if existing is not None:
        xll, yll, c = head["xllcorner"], head["yllcorner"], head["cellsize"]
        w, h = int(head["ncols"]), int(head["nrows"])

    srcs = [rasterio.open(p) for p in paths]
    mosaic, mosaic_tr = merge(srcs, nodata=np.nan)
    elev = np.full((h, w), np.nan, dtype=np.float32)
    reproject(
        source=mosaic[0],
        destination=elev,
        src_transform=mosaic_tr,
        src_crs=srcs[0].crs,
        src_nodata=np.nan,
        dst_transform=from_origin(xll, yll + h * c, c, c),
        dst_crs="EPSG:3035",
        dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    for s in srcs:
        s.close()

    # Water is where the depth survives being written with two decimals (a depth under 5 mm
    # would be written as 0.00, which the model reads as land)
    water = np.isfinite(elev) & (elev <= -0.005)
    depth = np.where(water, -elev, NODATA)
    if existing is not None:
        had = existing > 0
        added = water & ~had
        depth = np.where(had, existing, depth)
        water = had | added
        print(f"kept {had.sum()} water cells, added {added.sum()} from EMODnet")
    dist = np.where(water, ndimage.distance_transform_edt(water) * c, NODATA)
    nb = int(round(args.block_km * 1000 / c))
    rows, cols = np.indices((h, w))
    ncb = math.ceil(w / nb)
    blocks = np.where(water, (rows // nb) * ncb + cols // nb + 1, NODATA)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_asc(out / "bathy.asc", depth, xll, yll, c, "%.2f")
    write_asc(out / "disttocoast.asc", dist, xll, yll, c, "%.0f")
    write_asc(out / "blocks.asc", blocks, xll, yll, c, "%d")
    print(
        f"grid {w} x {h} cells of {c:g} m from ({xll:.0f}, {yll:.0f}), EPSG:3035; "
        f"water {water.sum()} cells, depth max {depth[water].max():.0f} m, "
        f"median {np.median(depth[water]):.0f} m; blocks of {args.block_km:g} km: "
        f"{len(np.unique(blocks[water]))}"
    )


if __name__ == "__main__":
    main()
