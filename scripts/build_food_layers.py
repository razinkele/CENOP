"""Write DEPONS-style food layers (patches.asc, prey01-12.asc) for a landscape from seasonal maps.

Follows DEPONS 3.2 (TRACE 2.6): food patches are single 400 m cells chosen at random (1.6% of
the water); the monthly prey maps are seasonal maps of relative porpoise density placed at
their months, interpolated per cell with a + b sin(x) + c cos(x), and scaled to a mean of
0.38592 over the water. patches.asc is written on the landscape's 400 m grid (the grid the
loader resamples the landscape to), which the loader uses as written; the prey files on the
coarser of the landscape's own grid and 400 m.

Usage (from the repository root):

  python3 scripts/build_food_layers.py --landscape Lithuania --crs EPSG:3035 \\
      --season 7=summer.tif --season 1=winter.tif [--band 1] [--seed 0] \\
      [--out data/Lithuania] [--note "source and licence"]

--season MONTH=PATH places a raster (any CRS GDAL reads) at a month; give one, two (six months
apart) or three or more. Values are relative; only their pattern matters after scaling.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cenop.landscape import food_layers as fl  # noqa: E402
from cenop.landscape.loader import LandscapeLoader  # noqa: E402
from cenop.landscape.resample import resample_to_400m  # noqa: E402

NODATA = -9999


def reproject_to_grid(path: str, band: int, crs: str, meta, shape) -> np.ndarray:
    """Bilinear reprojection of one raster band onto the landscape grid; row 0 is south."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    h, w = shape
    dst = np.full((h, w), np.nan, dtype=np.float64)
    c = meta.cellsize
    transform = from_origin(meta.xllcorner, meta.yllcorner + h * c, c, c)
    with rasterio.open(path) as src:
        reproject(
            source=rasterio.band(src, band),
            destination=dst,
            dst_transform=transform,
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
    return np.flipud(dst)


def write_asc(path: Path, arr: np.ndarray, meta, fmt: str) -> None:
    """Write an ESRI ASCII grid; ``arr`` has row 0 in the south (loader convention)."""
    h, w = arr.shape
    header = (
        f"ncols {w}\nnrows {h}\nxllcorner {meta.xllcorner:.6f}\nyllcorner {meta.yllcorner:.6f}\n"
        f"cellsize {meta.cellsize:g}\nNODATA_value {NODATA}"
    )
    np.savetxt(path, np.flipud(arr), fmt=fmt, header=header, comments="")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--landscape", required=True)
    ap.add_argument("--crs", required=True, help="CRS of the landscape grid, e.g. EPSG:3035")
    ap.add_argument("--season", action="append", default=[], metavar="MONTH=PATH")
    ap.add_argument(
        "--uniform",
        action="store_true",
        help="prey at the DEPONS mean on every water cell (no source maps)",
    )
    ap.add_argument("--band", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="seed for the patch placement")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--out", default=None, help="output folder (default: the landscape folder)")
    ap.add_argument("--note", default="", help="source description for FOOD_PROVENANCE.txt")
    ap.add_argument(
        "--floor-deeper-than",
        type=float,
        default=None,
        help="water deeper than this (m) without a source value gets --floor-quantile",
    )
    ap.add_argument("--floor-quantile", type=float, default=0.1)
    args = ap.parse_args()

    if args.uniform == bool(args.season):
        ap.error("give --season maps or --uniform, not both or neither")
    loader = LandscapeLoader(args.landscape, args.data_dir)
    # Only the depth is needed: the 400 m grid the model runs on is the depth resampled as the
    # loader does it (so the landscape's other files need not fit yet)
    native_depth, native_meta = loader._load_asc("bathy.asc")
    on_400 = resample_to_400m({"metadata": native_meta, "depth": native_depth})
    meta = on_400["metadata"]
    depth = on_400["depth"]
    water = depth > 0
    # Prey maps go on the coarser of the landscape's own grid and 400 m: patches must be single
    # 400 m cells, but prey maps from ~1 km sources gain nothing from 400 m files.
    if native_meta.cellsize > meta.cellsize:
        prey_meta, prey_depth = native_meta, native_depth
    else:
        prey_meta, prey_depth = meta, depth
    prey_water = prey_depth > 0
    seasons = {}
    coverage = {}
    for item in args.season:
        month, path = item.split("=", 1)
        grid = reproject_to_grid(path, args.band, args.crs, prey_meta, prey_depth.shape)
        covered = np.isfinite(grid) & prey_water
        coverage[float(month)] = covered.sum() / prey_water.sum()
        print(f"month {float(month):4.1f}: {path} covers {coverage[float(month)]:.1%} of the water")
        if args.floor_deeper_than is not None:
            grid = fl.floor_deep_gaps(
                grid, prey_water, prey_depth, args.floor_deeper_than, args.floor_quantile
            )
        seasons[float(month)] = fl.fill_water_gaps(grid, prey_water)

    if args.uniform:
        prey = np.full((12,) + prey_depth.shape, fl.DEPONS_MEAN_PREY)
    else:
        prey = fl.standardize(fl.monthly_from_seasons(seasons), prey_water)
    patches = fl.random_patches(water, seed=args.seed)

    out = Path(args.out or Path(args.data_dir) / args.landscape)
    out.mkdir(parents=True, exist_ok=True)
    write_asc(out / "patches.asc", np.where(water, patches.astype(np.int32), NODATA), meta, "%d")
    for m in range(12):
        write_asc(
            out / f"prey{m + 1:02d}.asc", np.where(prey_water, prey[m], NODATA), prey_meta, "%.5f"
        )
    lines = [
        f"Food layers written {date.today().isoformat()} by scripts/build_food_layers.py",
        f"Grid: patches on 400 m cells, {meta.ncols} x {meta.nrows}; prey on "
        f"{prey_meta.cellsize:g} m cells, {prey_meta.ncols} x {prey_meta.nrows}; both from "
        f"({meta.xllcorner:.0f}, {meta.yllcorner:.0f}), {args.crs}",
        f"Patches: {int(patches.sum())} cells = {patches.sum() / water.sum():.2%} of the water, "
        f"random, seed {args.seed}",
    ]
    if args.uniform:
        lines.append(
            f"Prey: uniform, {fl.DEPONS_MEAN_PREY} (the DEPONS mean) on every water cell in every "
            "month"
        )
    else:
        maps = ", ".join(
            f"{m:g} ({Path(p).parent.name}/{Path(p).name}, {coverage[m]:.1%} of the water)"
            for m, p in sorted((float(x.split("=", 1)[0]), x.split("=", 1)[1]) for x in args.season)
        )
        floor = (
            f"water deeper than {args.floor_deeper_than:g} m without a value set to the "
            f"{args.floor_quantile:g} quantile; "
            if args.floor_deeper_than is not None
            else ""
        )
        lines.append(
            f"Prey: seasonal maps at months {maps}; band {args.band}; {floor}other gaps filled "
            "from the nearest value; interpolated with a + b sin(x) + c cos(x); scaled to mean "
            f"{fl.DEPONS_MEAN_PREY} over the water each month"
        )
    lines.append(f"Source: {args.note}")
    (out / "FOOD_PROVENANCE.txt").write_text("\n".join(lines) + "\n")
    print(f"wrote patches.asc, prey01-12.asc and FOOD_PROVENANCE.txt to {out}")


if __name__ == "__main__":
    main()
