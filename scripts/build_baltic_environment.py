"""Write sediment.asc and salinity01-12.asc for a Baltic landscape from open data.

Salinity: twelve monthly surface fields from a netCDF file, either the Copernicus Baltic
reanalysis BALTICSEA_MULTIYEAR_PHY_003_011 (variable so; what DEPONS used, monthly means for
2021, TRACE 2.6; scripts/fetch_copernicus_salinity.py, needs a Copernicus Marine account) or
the open Baltic & North Sea Climatology (BNSC v1.1, University of Hamburg ICDC; decadal
interpolated monthly means 2006-2015, 0.25 degree; CC BY 4.0). Bilinear onto the landscape
grid; water cells without a value take the nearest value.

Sediment: median grain size in phi from EMODnet Geology seabed substrate (Folk classes;
CC BY 4.0), with the values DEPONS's Kattegat sediment.asc uses per class (TRACE Appendix 1,
Table S1). Polygons are applied coarsest first, so finer maps overwrite coarser ones; the Folk-7
class is used where a polygon has one, otherwise the Folk-5 class. "Mud to muddy Sand" without
a Folk-7 split has no value in the DEPONS table and gets the sandy-Mud value (the middle of the
three classes it spans; a choice made here). Water cells no map covers (lagoons) take the
nearest value.

Usage (from the repository root):

  python3 scripts/build_baltic_environment.py --landscape CentralBaltic --crs EPSG:3035 \\
      --salinity BNSC_salinity_OAN_2006-2015.nc \\
      --substrate substrate_1m.geojson substrate_250k.geojson substrate_100k.geojson
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cenop.landscape.food_layers import fill_water_gaps  # noqa: E402
from cenop.landscape.loader import LandscapeLoader  # noqa: E402

NODATA = -9999
# Median grain size (phi) per EMODnet Folk class, as in DEPONS's Kattegat sediment.asc
FOLK7_PHI = {11: 7.142, 12: 4.40, 13: 3.155, 2: 2.277, 3: 1.218, 4: 1.495, 5: -8.0}
FOLK5_PHI = {1: 4.40, 2: 2.277, 3: 1.218, 4: 1.495, 5: -8.0}


def write_asc(path: Path, arr: np.ndarray, meta, fmt: str) -> None:
    """``arr`` has row 0 in the south (loader convention)."""
    h, w = arr.shape
    header = (
        f"ncols {w}\nnrows {h}\nxllcorner {meta.xllcorner:.1f}\nyllcorner {meta.yllcorner:.1f}\n"
        f"cellsize {meta.cellsize:g}\nNODATA_value {NODATA}"
    )
    np.savetxt(path, np.flipud(arr), fmt=fmt, header=header, comments="")


def grid_transform(meta, shape):
    from rasterio.transform import from_origin

    h, _ = shape
    c = meta.cellsize
    return from_origin(meta.xllcorner, meta.yllcorner + h * c, c, c)


def salinity_layers(path: str, crs: str, meta, depth: np.ndarray) -> np.ndarray:
    import xarray as xr
    from rasterio.transform import from_origin
    from rasterio.warp import Resampling, reproject

    ds = xr.open_dataset(path)
    # BNSC names the variable salinity_*, Copernicus "so"; coordinates lat/lon or latitude/longitude
    var = next(v for v in ds.data_vars if "salinity" in v or v == "so")
    da = ds[var]
    if "depth" in da.dims:
        da = da.isel(depth=0)  # surface (top level)
    da = da.rename({k: v for k, v in (("latitude", "lat"), ("longitude", "lon")) if k in da.dims})
    if da.sizes["time"] != 12:
        raise ValueError(f"{path}: expected 12 monthly fields, found {da.sizes['time']}")
    lat = da["lat"].values
    lon = da["lon"].values
    if lat[0] < lat[-1]:
        da = da.isel(lat=slice(None, None, -1))
        lat = lat[::-1]
    dlon, dlat = float(lon[1] - lon[0]), float(lat[0] - lat[1])
    src_tr = from_origin(lon[0] - dlon / 2, lat[0] + dlat / 2, dlon, dlat)
    water = depth > 0
    out = np.empty((12,) + depth.shape)
    for m in range(12):
        dst = np.full(depth.shape, np.nan)
        reproject(
            source=da.isel(time=m).values.astype(np.float64),
            destination=dst,
            src_transform=src_tr,
            src_crs="EPSG:4326",
            src_nodata=np.nan,
            dst_transform=grid_transform(meta, depth.shape),
            dst_crs=crs,
            dst_nodata=np.nan,
            resampling=Resampling.bilinear,
        )
        dst = np.flipud(dst)
        out[m] = fill_water_gaps(dst, water)
    return out


def sediment_layer(paths: list[str], crs: str, meta, depth: np.ndarray) -> np.ndarray:
    import geopandas as gpd
    from rasterio.features import rasterize

    os.environ.setdefault("OGR_GEOJSON_MAX_OBJ_SIZE", "0")
    phi = np.full(depth.shape, np.nan)
    for p in paths:  # coarsest first: finer maps overwrite
        g = gpd.read_file(p).to_crs(crs)
        f7 = g["folk_7cl"].astype(float) if "folk_7cl" in g else np.nan
        f5 = g["folk_5cl"].astype(float)
        val = np.where(
            np.isin(f7, list(FOLK7_PHI)), [FOLK7_PHI.get(int(v), np.nan) if np.isfinite(v) else np.nan for v in f7], np.nan
        )  # fmt: skip
        val = np.where(np.isfinite(val), val, [FOLK5_PHI.get(int(v), np.nan) for v in f5])
        keep = np.isfinite(val)
        shapes = list(zip(g.geometry[keep], val[keep]))
        if not shapes:
            continue
        layer = rasterize(
            shapes,
            out_shape=depth.shape,
            transform=grid_transform(meta, depth.shape),
            fill=np.nan,
            dtype="float64",
        )
        layer = np.flipud(layer)
        phi = np.where(np.isfinite(layer), layer, phi)
        print(f"{Path(p).name}: {keep.sum()} polygons with a class")
    water = depth > 0
    covered = np.isfinite(phi) & water
    print(f"substrate maps cover {covered.sum() / water.sum():.1%} of the water")
    return fill_water_gaps(phi, water)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--landscape", required=True)
    ap.add_argument("--crs", required=True)
    ap.add_argument("--salinity", required=True)
    ap.add_argument("--substrate", nargs="+", default=[], help="coarsest first")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--salinity-only", action="store_true", help="write the salinity files only")
    args = ap.parse_args()

    loader = LandscapeLoader(args.landscape, args.data_dir)
    depth, meta = loader._load_asc("bathy.asc")  # the landscape's own grid
    water = depth > 0
    out = Path(args.data_dir) / args.landscape

    sal = salinity_layers(args.salinity, args.crs, meta, depth)
    for m in range(12):
        write_asc(out / f"salinity{m + 1:02d}.asc", np.where(water, sal[m], NODATA), meta, "%.3f")
    print(
        f"salinity: January mean {sal[0][water].mean():.2f}, July mean {sal[6][water].mean():.2f}, "
        f"range {sal[:, water].min():.2f}-{sal[:, water].max():.2f}"
    )
    if args.salinity_only:
        return
    sed = sediment_layer(args.substrate, args.crs, meta, depth)
    write_asc(out / "sediment.asc", np.where(water, sed, NODATA), meta, "%.3f")
    vals, counts = np.unique(np.round(sed[water], 3), return_counts=True)
    print(
        "sediment phi shares:", {float(v): round(c / water.sum(), 3) for v, c in zip(vals, counts)}
    )


if __name__ == "__main__":
    main()
