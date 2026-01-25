from pathlib import Path
from scripts.update_baseline import update_baseline_if_improved
import json


def write_history(tmp_path, values):
    p = tmp_path / 'perf_history.csv'
    p.write_text('timestamp,ticks,porpoise,seed,elapsed,returncode,baseline_elapsed,threshold,regression\n')
    with p.open('a') as f:
        for v in values:
            # Provide explicit placeholders for baseline_elapsed, threshold and regression
            f.write(f'2026-01-01T00:00:00Z,500,500,42,{v},0,, ,False\n')
    return p


def test_update_baseline_if_improved(tmp_path):
    history = write_history(tmp_path, [10.0, 9.8, 9.7, 9.6, 9.5])
    baseline = tmp_path / 'short_run_baseline.json'
    baseline.write_text(json.dumps({'elapsed': 11.0}))

    updated = update_baseline_if_improved(str(history), str(baseline), min_runs=3, window=3)
    assert updated is True
    data = json.loads(baseline.read_text())
    assert float(data['elapsed']) < 11.0
