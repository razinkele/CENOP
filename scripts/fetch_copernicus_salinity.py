"""Download monthly surface salinity from the Copernicus Baltic Sea physics reanalysis.

The source DEPONS 3.2 used for its Kattegat salinity (TRACE 2.6): product
BALTICSEA_MULTIYEAR_PHY_003_011 (formerly ..._REANALYSIS_...), dataset cmems_mod_bal_phy_my_P1M-m,
variable so (sea water salinity), monthly means, here the top model level (0.5 m) of one year.

Needs a Copernicus Marine account: run `copernicusmarine login` once beforehand (it stores the
credentials under ~/.copernicusmarine/); this script uses them and never asks for or prints them.
Copernicus Marine products are free of charge under the Copernicus Marine licence.

Usage (from the repository root):

  python3 scripts/fetch_copernicus_salinity.py --year 2021 --lon 12.5 24 --lat 53.3 60.1 \\
      --out copernicus_bal_so_2021.nc

The output works with scripts/build_baltic_environment.py --salinity.
"""

from __future__ import annotations

import argparse
from pathlib import Path

DATASET = "cmems_mod_bal_phy_my_P1M-m"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--lon", type=float, nargs=2, required=True)
    ap.add_argument("--lat", type=float, nargs=2, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import copernicusmarine

    out = Path(args.out)
    copernicusmarine.subset(
        dataset_id=DATASET,
        variables=["so"],
        minimum_longitude=args.lon[0],
        maximum_longitude=args.lon[1],
        minimum_latitude=args.lat[0],
        maximum_latitude=args.lat[1],
        start_datetime=f"{args.year}-01-01T00:00:00",
        end_datetime=f"{args.year}-12-31T23:59:59",
        minimum_depth=0.0,
        maximum_depth=1.0,
        output_directory=str(out.parent),
        output_filename=out.name,
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
