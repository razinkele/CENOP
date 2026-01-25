"""Update baseline based on recent perf history.

Reads `ci/perf/perf_history.csv`, computes the median elapsed over the last N successful runs,
and if the median indicates a stable improvement (e.g., median < baseline * 0.98 and at
least `min_runs` available), writes a new baseline JSON file for review.

Usage:
    python scripts/update_baseline.py --history ci/perf/perf_history.csv --baseline ci/perf/short_run_baseline.json --min-runs 5 --window 5
"""
import argparse
import csv
import json
from statistics import median
from pathlib import Path


def compute_recent_median(history_path: Path, window: int = 5):
    rows = []
    with history_path.open('r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            # Skip entries that failed or missing elapsed
            try:
                if r.get('regression','').lower() in ('true','1'):
                    continue
                elapsed = float(r.get('elapsed', 'nan'))
                rows.append(elapsed)
            except Exception:
                continue
    if not rows:
        return None
    recent = rows[-window:]
    return median(recent)


def update_baseline_if_improved(history: str, baseline: str, min_runs: int = 7, window: int = 7, improvement_factor: float = 0.97):
    """Update baseline if recent median shows stable improvement.

    Defaults are conservative: require at least 7 runs and a 3% improvement
    (median < baseline * 0.97) over a window of 7 runs.
    """
    history_p = Path(history)
    baseline_p = Path(baseline)
    if not history_p.exists():
        print(f"History file not found: {history}")
        return False

    med = compute_recent_median(history_p, window=window)
    if med is None:
        print("No valid recent runs found")
        return False

    # Require at least min_runs entries in file
    with history_p.open('r', encoding='utf-8') as f:
        count = sum(1 for _ in f) - 1  # minus header
    if count < min_runs:
        print(f"Not enough runs in history ({count} < {min_runs})")
        return False

    # Load existing baseline
    if baseline_p.exists():
        with baseline_p.open('r', encoding='utf-8') as f:
            data = json.load(f)
        old = float(data.get('elapsed', 0.0))
    else:
        old = float('inf')
        data = {}

    # Update if median is a meaningful improvement
    if med < old * improvement_factor:
        data['ticks'] = data.get('ticks', 500)
        data['porpoise'] = data.get('porpoise', 500)
        data['elapsed'] = float(f"{med:.4f}")
        data['note'] = f"Updated by nightly aggregator: median of last {window} runs = {med:.4f}"
        baseline_p.parent.mkdir(parents=True, exist_ok=True)
        with baseline_p.open('w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
        print(f"Baseline updated: {baseline} -> {med:.4f}")
        return True

    print(f"No baseline update needed: median={med:.4f}, baseline={old}")
    return False


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='ci/perf/perf_history.csv')
    parser.add_argument('--baseline', type=str, default='ci/perf/short_run_baseline.json')
    parser.add_argument('--min-runs', type=int, default=5)
    parser.add_argument('--window', type=int, default=5)
    args = parser.parse_args()

    updated = update_baseline_if_improved(args.history, args.baseline, args.min_runs, args.window)
    if updated:
        # Exit with 0; the workflow will commit and open PR
        exit(0)
    else:
        exit(0)
