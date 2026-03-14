#!/usr/bin/env python3
"""Benchmark simulation with and without visualization overhead.

Measures the time difference between:
- Headless: pure simulation engine (no position extraction)
- Viz pipeline: engine + position extraction + coordinate conversion + serialization

Runs multiple iterations with warm-up for statistical reliability.

Usage:
    python3 scripts/benchmark_viz_overhead.py
    python3 scripts/benchmark_viz_overhead.py --ticks 960 --population 1000 --runs 5
    python3 scripts/benchmark_viz_overhead.py --landscape Kattegat --all-scenarios
"""

import argparse
import statistics
import time

import numpy as np


def run_single(n_ticks, population, landscape, with_viz):
    """Run one benchmark iteration. Returns elapsed seconds."""
    from cenop.core.simulation import Simulation
    from cenop.parameters.simulation_params import SimulationParameters

    params = SimulationParameters(
        porpoise_count=population,
        sim_years=1,
        landscape=landscape,
    )
    sim = Simulation(params)
    sim.initialize()

    if with_viz:
        meta = sim._cell_data.metadata
        try:
            from cenop.server.main import grid_to_lonlat
            from cenop.ui.sidebar import LANDSCAPE_CRS

            crs = LANDSCAPE_CRS.get(landscape, "EPSG:3035")
            can_convert = meta is not None
        except ImportError:
            can_convert = False

    t0 = time.perf_counter()
    for _ in range(n_ticks):
        sim.step()
        if with_viz:
            raw_pos = sim.get_porpoise_positions()
            if raw_pos.size > 0 and can_convert:
                lons, lats = grid_to_lonlat(
                    raw_pos[:, 1], raw_pos[:, 2], meta, crs
                )
                converted = np.column_stack((
                    raw_pos[:, 0],
                    lons,
                    lats,
                    raw_pos[:, 3],
                    raw_pos[:, 4],
                    raw_pos[:, 5],
                    raw_pos[:, 6],
                ))
                _ = converted.tolist()
    return time.perf_counter() - t0


def run_benchmark(n_ticks, population, landscape, n_runs, warmup):
    """Run benchmark with multiple iterations and report statistics."""
    print(f"\n{'=' * 65}")
    print(f"  {population} porpoises | {landscape} | "
          f"{n_ticks} ticks ({n_ticks / 48:.0f} days)")
    print(f"  {n_runs} runs + {warmup} warm-up")
    print(f"{'=' * 65}")

    # Warm-up (JIT compilation, caches)
    for i in range(warmup):
        print(f"  warm-up {i + 1}/{warmup}...", end="\r")
        run_single(min(n_ticks, 48), population, landscape, with_viz=False)
    if warmup:
        print(f"  warm-up complete       ")

    # Headless runs
    headless_times = []
    for i in range(n_runs):
        print(f"  headless run {i + 1}/{n_runs}...", end="\r")
        t = run_single(n_ticks, population, landscape, with_viz=False)
        headless_times.append(t)
    print(f"  headless runs complete    ")

    # Viz runs
    viz_times = []
    for i in range(n_runs):
        print(f"  viz run {i + 1}/{n_runs}...", end="\r")
        t = run_single(n_ticks, population, landscape, with_viz=True)
        viz_times.append(t)
    print(f"  viz runs complete         ")

    # Statistics
    h_mean = statistics.mean(headless_times)
    h_stdev = statistics.stdev(headless_times) if n_runs > 1 else 0.0
    v_mean = statistics.mean(viz_times)
    v_stdev = statistics.stdev(viz_times) if n_runs > 1 else 0.0

    overhead = v_mean - h_mean
    overhead_pct = (overhead / h_mean * 100) if h_mean > 0 else 0
    ratio = v_mean / h_mean if h_mean > 0 else 1.0

    ms_per_tick_h = h_mean / n_ticks * 1000
    ms_per_tick_v = v_mean / n_ticks * 1000
    ms_per_tick_o = overhead / n_ticks * 1000

    print()
    print(f"  {'Mode':<25} {'Total (s)':>10} {'± (s)':>8} {'ms/tick':>10}")
    print(f"  {'-' * 55}")
    print(f"  {'Headless (engine only)':<25} {h_mean:>10.3f} {h_stdev:>8.3f} "
          f"{ms_per_tick_h:>10.2f}")
    print(f"  {'With viz pipeline':<25} {v_mean:>10.3f} {v_stdev:>8.3f} "
          f"{ms_per_tick_v:>10.2f}")
    print(f"  {'-' * 55}")
    print(f"  {'Viz overhead':<25} {overhead:>10.3f} {'':>8} "
          f"{ms_per_tick_o:>10.2f}")
    print(f"  {'Overhead %':<25} {overhead_pct:>9.1f}%")
    print(f"  {'Speedup ratio':<25} {ratio:>9.2f}x")

    return {
        "landscape": landscape,
        "population": population,
        "ticks": n_ticks,
        "headless_mean": h_mean,
        "headless_stdev": h_stdev,
        "viz_mean": v_mean,
        "viz_stdev": v_stdev,
        "overhead_pct": overhead_pct,
        "ratio": ratio,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark simulation visualization overhead"
    )
    parser.add_argument(
        "--ticks",
        type=int,
        default=480,
        help="Ticks to simulate per run (default: 480 = 10 days)",
    )
    parser.add_argument(
        "--population",
        type=int,
        default=500,
        help="Porpoise population (default: 500)",
    )
    parser.add_argument(
        "--landscape",
        type=str,
        default="Kattegat",
        help="Landscape name (default: Kattegat)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=3,
        help="Number of measurement runs (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Number of warm-up runs (default: 1)",
    )
    parser.add_argument(
        "--all-scenarios",
        action="store_true",
        help="Run across multiple population sizes and landscapes",
    )
    args = parser.parse_args()

    print("CENOP Visualization Overhead Benchmark")
    print("=" * 65)

    if args.all_scenarios:
        scenarios = [
            (480, 100, "Homogeneous"),
            (480, 500, "Homogeneous"),
            (480, 100, "Kattegat"),
            (480, 500, "Kattegat"),
            (480, 1000, "Kattegat"),
        ]
        results = []
        for ticks, pop, land in scenarios:
            r = run_benchmark(ticks, pop, land, args.runs, args.warmup)
            results.append(r)

        # Summary table
        print(f"\n\n{'=' * 65}")
        print("  SUMMARY")
        print(f"{'=' * 65}")
        print(f"  {'Scenario':<30} {'Overhead':>10} {'Ratio':>8}")
        print(f"  {'-' * 50}")
        for r in results:
            label = f"{r['population']}p / {r['landscape']}"
            print(f"  {label:<30} {r['overhead_pct']:>9.1f}% "
                  f"{r['ratio']:>7.2f}x")
    else:
        run_benchmark(
            args.ticks, args.population, args.landscape,
            args.runs, args.warmup,
        )


if __name__ == "__main__":
    main()
