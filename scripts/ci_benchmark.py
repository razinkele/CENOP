"""Run a short CI benchmark and produce a JSON report.

Usage:
    python scripts/ci_benchmark.py --ticks 500 --porpoise 500 --baseline ci/perf/short_run_baseline.json

The script warms up numba, runs `scripts/instrumented_run.py` and records elapsed time. If a baseline exists it compares and (conservatively) fails the run when elapsed > baseline*2 + 5s.
"""
import argparse
import json
import subprocess
import time
import sys
import csv
from datetime import datetime
from pathlib import Path

from cenop.optimizations import warmup_numba


def write_history_entry(result: dict, path: str = 'ci/perf/perf_history.csv') -> None:
    """Append a benchmark result as a CSV row into `path`.
    Creates parent dir and header if missing.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    header = [
        'timestamp', 'ticks', 'porpoise', 'seed', 'elapsed', 'returncode',
        'baseline_elapsed', 'threshold', 'regression'
    ]
    row = {
        'timestamp': result.get('timestamp', datetime.utcnow().isoformat() + 'Z'),
        'ticks': result.get('ticks'),
        'porpoise': result.get('porpoise'),
        'seed': result.get('seed'),
        'elapsed': result.get('elapsed'),
        'returncode': result.get('returncode'),
        'baseline_elapsed': result.get('baseline_elapsed', ''),
        'threshold': result.get('threshold', ''),
        'regression': result.get('regression', False)
    }

    write_header = not p.exists()
    with p.open('a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def compute_threshold(baseline_elapsed: float) -> float:
    """Compute strict regression threshold from baseline elapsed time.
    Current policy: fail if elapsed > baseline * 1.25 + 2s (25% slower + small tolerance).
    """
    return float(baseline_elapsed) * 1.25 + 2.0


def run_benchmark(ticks: int, porpoise: int, seed: int = 42, baseline_path: str | None = None):
    # Warm up numba-compiled functions to avoid compilation time in measurement
    try:
        warmup_numba()
    except Exception as e:
        print(f"Warm-up failed: {e}")

    cmd = [sys.executable, "scripts/instrumented_run.py", "--ticks", str(ticks), "--porpoise", str(porpoise), "--seed", str(seed), "--log", "ci_instrumented_log.txt", "--warn", "1000"]

    print("Running benchmark:", " ".join(cmd))
    start = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - start

    # Save console output as part of the artifact
    with open("ci_instrumented_stdout.txt", "w", encoding="utf-8") as f:
        f.write(res.stdout)
        f.write("\n\n--- STDERR ---\n\n")
        f.write(res.stderr)

    result = {
        "ticks": ticks,
        "porpoise": porpoise,
        "seed": seed,
        "elapsed": elapsed,
        "returncode": res.returncode,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }

    comparison = None
    if baseline_path:
        try:
            with open(baseline_path, "r", encoding="utf-8") as f:
                baseline = json.load(f)
            baseline_elapsed = float(baseline.get("elapsed", 0.0))
            # Tightened threshold: fail if more than 25% slower + 2s tolerance
            threshold = compute_threshold(baseline_elapsed)
            result["baseline_elapsed"] = baseline_elapsed
            result["threshold"] = threshold
            result["regression"] = elapsed > threshold
            comparison = result["regression"]
        except FileNotFoundError:
            print(f"Baseline not found at {baseline_path}; uploading results for manual inspection.")
        except Exception as e:
            print(f"Error reading baseline file: {e}")

    # Write result json
    with open("benchmark_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    # Append to perf history CSV (for trend tracking)
    try:
        write_history_entry(result, path='ci/perf/perf_history.csv')
    except Exception as e:
        print(f"Warning: failed to write perf history CSV: {e}")

    # Print a summary
    print(f"Benchmark finished: elapsed={elapsed:.2f}s, returncode={res.returncode}")
    if comparison is True:
        print("Regression detected (elapsed > threshold). Failing job.")
        return 2
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ticks', type=int, default=500)
    parser.add_argument('--porpoise', type=int, default=500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--baseline', type=str, default='ci/perf/short_run_baseline.json')
    args = parser.parse_args()

    rc = run_benchmark(args.ticks, args.porpoise, args.seed, args.baseline)
    sys.exit(rc)
