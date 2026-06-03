#!/usr/bin/env python3
"""Generate Kattegat reference (regression baseline) runs.

Headless runner that loads the real Kattegat landscape from ``data/Kattegat``,
runs an undisturbed simulation (DEPONS defaults: turbines off, ships off), and
writes DEPONS-format output files (Population/Energy/Mortality/Dispersal.txt)
to be diffed as a regression baseline.

Seed: DEPONS parameters.xml ``randomSeed`` is ``__NULL__`` (DEPONS randomizes
per run), so for a deterministic baseline we use CENOP's default fallback
seed=42 — the faithful stand-in for an unconfigured DEPONS seed.

Usage:
    python3 scripts/run_kattegat_reference.py \
        --count 2000 --years 5 --seed 42 --out output/kattegat_ref
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from repo root or CENOP/
_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from cenop.core.output_writer import OutputConfig, OutputWriter  # noqa: E402
from cenop.core.simulation import Simulation  # noqa: E402
from cenop.landscape.cell_data import CellData  # noqa: E402
from cenop.parameters.simulation_params import SimulationParameters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Kattegat reference run generator")
    parser.add_argument("--count", type=int, default=2000, help="Initial porpoise count")
    parser.add_argument("--years", type=int, default=5, help="Simulation years")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (deterministic)")
    parser.add_argument("--data-dir", default="data", help="Base data directory (contains Kattegat/)")
    parser.add_argument("--out", default="output/kattegat_ref", help="Output directory")
    parser.add_argument(
        "--ships", action="store_true",
        help="Enable ship traffic from data/Kattegat/ships.json (disturbed run; "
             "exercises deter_ships_min_db). Default off, matching DEPONS.",
    )
    parser.add_argument(
        "--turbines", default="off",
        help="Wind-farm scenario name (file in data/wind-farms/, e.g. 'Kattegat-test'); "
             "exercises turbine deterrence + deter_max_distance. Default 'off'.",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    log = logging.getLogger("kattegat_ref")

    params = SimulationParameters(
        porpoise_count=args.count,
        sim_years=args.years,
        random_seed=args.seed,
        landscape="Kattegat",
        ships_enabled=args.ships,
        turbines=args.turbines,
    )

    log.info("Loading Kattegat landscape from %s/Kattegat ...", args.data_dir)
    cell_data = CellData("Kattegat", args.data_dir)
    cell_data.load()
    log.info("Landscape loaded: %dx%d cells", cell_data.width, cell_data.height)

    sim = Simulation(params=params, cell_data=cell_data, seed=args.seed)
    sim.initialize()

    if args.ships:
        # Replace the sample-ship fallback with the real Kattegat routes,
        # mapped via the landscape's UTM origin (xll/yll from bathy.asc header).
        meta = cell_data.metadata
        ships_json = Path(args.data_dir) / "Kattegat" / "ships.json"
        sim._ship_manager.load_from_json(
            str(ships_json),
            utm_origin_x=getattr(meta, "xllcorner", 3976618.0),
            utm_origin_y=getattr(meta, "yllcorner", 3363923.0),
            cell_size=getattr(meta, "cellsize", 400.0),
        )
        sim._ship_manager.set_enabled(True)
        log.info("Loaded %d ships from %s", sim._ship_manager.count, ships_json)

    total_ticks = sim.max_ticks
    log.info(
        "Running Kattegat reference: N=%d, %d yr (%d ticks), seed=%d, ships=%s, turbines=%s, "
        "%d active turbines -> %s",
        args.count, args.years, total_ticks, args.seed, args.ships, args.turbines,
        len(sim._turbine_manager.turbines), args.out,
    )

    config = OutputConfig(output_dir=args.out)
    with OutputWriter(config) as writer:
        for tick in range(total_ticks):
            sim.step()
            writer.record_tick(sim)
            if tick % 5000 == 0:
                log.info("Progress: %.1f%% (tick %d/%d)", tick / total_ticks * 100, tick, total_ticks)

    log.info("Done. Output written to %s", Path(args.out).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
