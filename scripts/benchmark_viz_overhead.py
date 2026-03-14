#!/usr/bin/env python3
"""Benchmark simulation with and without visualization overhead.

Measures the time difference between:
- Full mode: position extraction + coordinate conversion each update tick
- Headless mode: pure simulation engine only

Usage:
    python3 scripts/benchmark_viz_overhead.py [--ticks N] [--population N]
"""

import argparse
import time
import sys
import numpy as np


def run_benchmark(n_ticks: int, population: int, landscape: str):
    """Run simulation for n_ticks and measure time with/without viz overhead."""
    from cenop.core.simulation import Simulation
    from cenop.parameters.simulation_params import SimulationParameters

    print(f"Benchmark: {n_ticks} ticks, {population} porpoises, {landscape}")
    print("=" * 60)

    # --- Headless run (no position extraction) ---
    params = SimulationParameters(
        porpoise_count=population,
        sim_years=1,
        landscape=landscape,
    )
    sim = Simulation(params)
    sim.initialize()

    print(f"Setup complete. Grid: {sim._cell_data.width}x{sim._cell_data.height}")
    print(f"Actual population: {np.sum(sim.population_manager.active_mask)}")
    print()

    t0 = time.perf_counter()
    for _ in range(n_ticks):
        sim.step()
    t_headless = time.perf_counter() - t0

    print(f"Headless (pure engine):  {t_headless:.3f}s "
          f"({t_headless / n_ticks * 1000:.2f} ms/tick)")

    # --- Full viz run (position extraction + coordinate conversion) ---
    params2 = SimulationParameters(
        porpoise_count=population,
        sim_years=1,
        landscape=landscape,
    )
    sim2 = Simulation(params2)
    sim2.initialize()
    meta = sim2._cell_data.metadata

    # Import coordinate conversion
    try:
        from cenop.server.main import grid_to_lonlat
        from cenop.ui.sidebar import LANDSCAPE_CRS
        crs = LANDSCAPE_CRS.get(landscape, "EPSG:3035")
        can_convert = True
    except ImportError:
        can_convert = False

    t0 = time.perf_counter()
    for tick in range(n_ticks):
        sim2.step()
        # Simulate viz overhead: extract positions + coordinate conversion
        raw_pos = sim2.get_porpoise_positions()
        if raw_pos.size > 0 and can_convert and meta is not None:
            lons, lats = grid_to_lonlat(
                raw_pos[:, 1], raw_pos[:, 2], meta, crs
            )
            converted = np.column_stack((
                raw_pos[:, 0], lons, lats,
                raw_pos[:, 3], raw_pos[:, 4],
                raw_pos[:, 5], raw_pos[:, 6],
            ))
            _ = converted.tolist()  # serialization cost
    t_viz = time.perf_counter() - t0

    print(f"With viz overhead:      {t_viz:.3f}s "
          f"({t_viz / n_ticks * 1000:.2f} ms/tick)")
    print()

    overhead = t_viz - t_headless
    pct = (overhead / t_headless) * 100 if t_headless > 0 else 0
    print(f"Viz overhead:           {overhead:.3f}s "
          f"({overhead / n_ticks * 1000:.2f} ms/tick)")
    print(f"Speedup (headless):     {pct:.1f}% faster")
    print(f"Ratio:                  {t_viz / t_headless:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark simulation viz overhead"
    )
    parser.add_argument(
        "--ticks", type=int, default=480,
        help="Number of ticks to simulate (default: 480 = 10 days)",
    )
    parser.add_argument(
        "--population", type=int, default=500,
        help="Initial porpoise population (default: 500)",
    )
    parser.add_argument(
        "--landscape", type=str, default="Homogeneous",
        help="Landscape name (default: Homogeneous)",
    )
    args = parser.parse_args()
    run_benchmark(args.ticks, args.population, args.landscape)


if __name__ == "__main__":
    main()
