CENOP Performance Dashboard
===========================

This folder is published nightly to the project's GitHub Pages site (branch `gh-pages`) and contains:

- `perf_history.csv` — a running history of benchmark runs (timestamp, ticks, porpoise, elapsed, etc.)
- `perf_plot.png` — an image with the elapsed time per run and a rolling median
- `perf_history.html` — a tiny HTML wrapper that displays the plot

How to interpret
----------------
- Metric: **elapsed** — total wall-clock time (seconds) for a representative short run (default: 500 ticks, 500 porpoise).
- On PRs the CI perf check will compare the measured elapsed time to the baseline. A small comment is posted to the PR with:
  - ticks, porpoise, elapsed (s)
  - baseline (s), threshold (s)
  - regression: YES/NO

Baseline & update policy
------------------------
- **Baseline**: a reference elapsed time stored in `short_run_baseline.json`.
- **Regression threshold** (for CI failures): elapsed > baseline * 1.25 + 2s (25% slower + 2s tolerance).
- **Nightly baseline update rule** (conservative): the nightly job computes the median elapsed of the most recent 7 successful runs; the baseline is updated only when the median shows a stable improvement of at least 3% (i.e., median < baseline * 0.97) and at least 7 recent runs are available. Updates are made via an automated PR so they can be reviewed before merging.

If you want to change thresholds or the window size, edit `scripts/update_baseline.py` or the workflow `.github/workflows/nightly-perf.yml`.

Contact & notes
----------------
**Contact:** Arturas Razinkovas-Baziukas, Marine Research Institute, Klaipeda University — [arturas.razinkovas-baziukas@ku.lt](mailto:arturas.razinkovas-baziukas@ku.lt) — [ORCID: 0000-0001-5060-5532](https://orcid.org/0000-0001-5060-5532)

If you see consistent performance regressions or spurious updates, raise an issue and we can tune the sensitivity or add more representative benchmarks.
