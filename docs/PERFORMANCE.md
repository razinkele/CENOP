# Performance Optimizations — CENOP

This document summarizes the key performance work completed to speed up the CENOP porpoise model and how to reproduce/measure performance.

## Key optimizations implemented

- Numba-accelerated PSM accumulator
  - Implemented `accumulate_psm_updates` (Numba `njit`) to replace hot `np.add.at` code paths that updated the per-agent PSM buffer.
  - Added a pure-Python fallback to preserve behavior when Numba isn't available.
  - Result: reduced memory churn and improved update speed (measured improvement in instrumented runs).

- Numba-accelerated social aggregator
  - Implemented `accumulate_social_totals` (Numba `njit`) to accumulate per-pair contributions into per-agent totals efficiently (avoids repeated `bincount` or big sparse matrix creations).
  - Provides a pure-Python fallback for portability.

- KD-tree neighbor-list change
  - Replaced `sparse_distance_matrix` creation with `scipy.spatial.cKDTree.query_ball_tree` to get neighbor lists, then build canonical neighbor pairs (rows < cols).
  - Avoids allocating large sparse matrices and reduces memory allocation overhead.

- Warm-up Numba at startup
  - Added `warmup_numba()` to pre-compile hot Numba functions during `Simulation.initialize()` to avoid first-run JIT latency in interactive/short runs.

## Profiling & measurement

- Recommended tool: `py-spy` for low-overhead sampling profiling. Example:

  py-spy record -o pyspy_flame.svg --format flamegraph -- python -u scripts/instrumented_run.py --ticks 2000 --porpoise 1000 --seed 42 --log instrumented_log_profile.txt --warn 200

- Benchmarking script: `scripts/ci_benchmark.py` runs a short instrumented simulation and writes `benchmark_result.json` and `ci/perf/perf_history.csv`.

- CI perf check: `.github/workflows/ci-perf-check.yml` runs the short benchmark and uploads artifacts for review.

## How to reproduce locally

1. Create and activate the `shiny` conda env (see `cenop/requirements.txt`).
2. Warm up Numba:

   python scripts/warmup_numba.py

3. Run instrumented benchmark for a representative case:

   python scripts/ci_benchmark.py --ticks 500 --porpoise 500

4. Inspect `benchmark_result.json` and `ci/perf/perf_history.csv` for timings.

## Next steps (recommended)

- Add scheduled nightly perf runs to collect long-term trends (implemented: nightly aggregator publishes `ci/perf` to GitHub Pages).
- Add a small set of micro-benchmarks and a perf dashboard (CSV-to-plot published as `ci/perf/perf_plot.png` and `index.html`) to visualize regressions over time.
- Investigate additional hotspots (neighbor weighting remains a dominant cost) and consider Numba-accelerating distance→RL→probability computations as a next step.

See the published perf dashboard at [https://your-org.github.io/your-repo/](https://your-org.github.io/your-repo/) after the nightly job runs (files under `ci/perf` are published to `gh-pages` branch). **Contact:** Arturas Razinkovas-Baziukas ([arturas.razinkovas-baziukas@ku.lt](mailto:arturas.razinkovas-baziukas@ku.lt)), Marine Research Institute, Klaipeda University — [ORCID: 0000-0001-5060-5532](https://orcid.org/0000-0001-5060-5532). For more details about specific changes see code references in `src/cenop/optimizations` and `src/cenop/agents/population.py`.
