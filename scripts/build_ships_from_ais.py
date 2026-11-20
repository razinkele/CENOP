"""Write a DEPONS ships.json for a landscape from raw AIS positions (see fetch_ais_huetten.py).

Every route is one passage of one ship sampled at 30-minute ticks, placed in time by its
first-buoy pause and padded by its last-buoy pause to a fixed cycle, as in DEPONS's Kattegat
file (cenop.agents.ais_routes). AIS ship-type codes become JOMOPANS classes. Lengths come
from --lengths (CSV with columns mmsi,length) where known; otherwise each class gets the
median length of that class in DEPONS's Kattegat ship file (real Danish AIS lengths).

Usage (from the repository root):

  python3 scripts/build_ships_from_ais.py --ais ais.parquet --landscape Lithuania \\
      --start 2024-08-12 --days 15 [--lengths lengths.csv] [--note "source"]
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cenop.agents import ais_routes  # noqa: E402


def kattegat_median_lengths() -> dict:
    ships = json.loads((ROOT / "data" / "Kattegat" / "ships.json").read_text())["ships"]
    by_type = defaultdict(list)
    for s in ships:
        by_type[s["type"]].append(s["length"])
    return {k: float(np.median(v)) for k, v in by_type.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--ais", required=True, help="parquet from fetch_ais_huetten.py")
    ap.add_argument("--landscape", required=True)
    ap.add_argument("--start", required=True, help="UTC date of tick 0, e.g. 2024-08-12")
    ap.add_argument("--days", type=int, required=True, help="length of the cycle in days")
    ap.add_argument("--lengths", default=None, help="CSV mmsi,length of known ship lengths")
    ap.add_argument("--data-dir", default=str(ROOT / "data"))
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    import pandas as pd

    df = pd.read_parquet(args.ais)
    # Keep positions on the landscape's water (400 m grid): elsewhere, e.g. in harbours or where
    # the landscape has no depth data, a ship emits nothing (DEPONS: no depth, no sound), and a
    # passage that leaves the water is split there by the gap rule.
    from cenop.landscape.loader import LandscapeLoader

    land = LandscapeLoader(args.landscape, args.data_dir).load_all()
    grid, depth = land["metadata"], land["depth"]
    ci = np.floor((df.x.to_numpy() - grid.xllcorner) / grid.cellsize).astype(int)
    ri = np.floor((df.y.to_numpy() - grid.yllcorner) / grid.cellsize).astype(int)
    inside = (ci >= 0) & (ri >= 0) & (ci < depth.shape[1]) & (ri < depth.shape[0])
    on_water = np.zeros(len(df), dtype=bool)
    on_water[inside] = depth[ri[inside], ci[inside]] > 0
    n_all, n_vessels_all = len(df), df.mmsi.nunique()
    df = df[on_water]
    t0 = dt.datetime.fromisoformat(args.start).replace(tzinfo=dt.timezone.utc).timestamp()
    cycle = args.days * 48 + 1
    lengths = {}
    if args.lengths:
        known = pd.read_csv(args.lengths)
        lengths = dict(zip(known.mmsi.astype(int), known.length.astype(float)))
    defaults = kattegat_median_lengths()
    out = ais_routes.build_ships(
        df.mmsi.to_numpy(np.int64),
        df.timestamp.to_numpy(np.float64),
        df.x.to_numpy(np.float64),
        df.y.to_numpy(np.float64),
        df.sog.to_numpy(np.float64),
        df.type.to_numpy(np.float64),
        t0,
        cycle_ticks=cycle,
        lengths=lengths,
        default_lengths=defaults,
    )
    dest = Path(args.data_dir) / args.landscape
    (dest / "ships.json").write_text(json.dumps(out, separators=(",", ":")))

    ships = out["ships"]
    types = Counter(s["type"] for s in ships)
    known_len = sum(1 for s in ships if int(s["name"].split("-")[0]) in lengths)
    speeds = np.array(
        [b["speed"] for r in out["routes"] for b in r["route"][1:-1] if b["speed"] >= 1.0]
    )
    vessels = len({s["name"].split("-")[0] for s in ships})
    summary = (
        f"Ship file written {dt.date.today().isoformat()} by scripts/build_ships_from_ais.py\n"
        f"Window: {args.start} 00:00 UTC, {args.days} days; routes span {cycle} ticks and repeat\n"
        f"Input: {Path(args.ais).name}: {n_all} positions of {n_vessels_all} vessels, of which "
        f"{len(df)} ({len(df) / n_all:.0%}) on the landscape's water\n"
        f"Output: {len(ships)} passages of {vessels} vessels (moored ships and passages "
        "shorter than one hour dropped)\n"
        f"Types: {dict(types.most_common())}\n"
        f"Lengths: {known_len} passages from --lengths, the rest the median length of their "
        "class in DEPONS's Kattegat ship file\n"
        f"Speed while moving (knots): median {np.median(speeds):.1f}, "
        f"p10 {np.quantile(speeds, 0.1):.1f}, p90 {np.quantile(speeds, 0.9):.1f}\n"
        f"Source: {args.note}\n"
    )
    (dest / "SHIPS_PROVENANCE.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
