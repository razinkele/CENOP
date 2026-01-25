"""Generate a perf plot from the perf_history CSV.

Writes `ci/perf/perf_plot.png` and `ci/perf/perf_history.html` (simple HTML with embedded PNG)
"""
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def make_plot(history_path: Path, out_png: Path, show_baseline: bool = True):
    df = pd.read_csv(history_path, parse_dates=['timestamp'])
    if df.empty:
        print('No data to plot')
        return False

    # Only plot successful runs
    df = df[df['returncode'] == 0]
    df['elapsed'] = pd.to_numeric(df['elapsed'], errors='coerce')
    df = df.dropna(subset=['elapsed'])
    if df.empty:
        print('No valid elapsed times to plot')
        return False

    df = df.sort_values('timestamp')

    fig, ax = plt.subplots(figsize=(8,3))
    ax.plot(df['timestamp'], df['elapsed'], marker='o', linestyle='-')

    # Rolling median for smoothing
    df['rolling_med'] = df['elapsed'].rolling(window=5, min_periods=1).median()
    ax.plot(df['timestamp'], df['rolling_med'], color='orange', linestyle='--', label='rolling median')

    # Baseline line if present
    if 'baseline_elapsed' in df.columns and df['baseline_elapsed'].notnull().any():
        # Use last non-empty baseline
        baseline_vals = pd.to_numeric(df['baseline_elapsed'], errors='coerce')
        last_baseline = baseline_vals[baseline_vals.notnull()].iloc[-1] if baseline_vals.notnull().any() else None
        if last_baseline and show_baseline:
            ax.axhline(last_baseline, color='red', linestyle=':', label=f'baseline {last_baseline:.2f}s')

    ax.set_ylabel('Elapsed (s)')
    ax.set_xlabel('Timestamp')
    ax.set_title('CENOP perf: elapsed time over time')
    ax.grid(True, alpha=0.3)
    ax.legend()

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)

    # Write a tiny HTML wrapper
    html = f"""<html><body><h3>Performance history</h3><img src='{out_png.name}' alt='perf plot' /></body></html>"""
    out_html = out_png.parent / 'perf_history.html'
    out_html.write_text(html, encoding='utf-8')
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--history', type=str, default='ci/perf/perf_history.csv')
    parser.add_argument('--out', type=str, default='ci/perf/perf_plot.png')
    args = parser.parse_args()

    ok = make_plot(Path(args.history), Path(args.out))
    if not ok:
        raise SystemExit(1)
    print('Plot generated:', args.out)
