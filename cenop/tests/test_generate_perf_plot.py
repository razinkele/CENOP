from pathlib import Path
from scripts.generate_perf_plot import make_plot
import pandas as pd


def test_make_plot(tmp_path):
    p = tmp_path / 'perf' 
    p.mkdir()
    csv = p / 'perf_history.csv'
    df = pd.DataFrame({
        'timestamp': ['2026-01-01T00:00:00Z','2026-01-02T00:00:00Z','2026-01-03T00:00:00Z'],
        'ticks': [500,500,500],
        'porpoise': [500,500,500],
        'seed': [42,42,42],
        'elapsed': [10.0,9.8,9.6],
        'returncode': [0,0,0],
        'baseline_elapsed': [11.0,11.0,11.0],
        'threshold': [11.5,11.5,11.5],
        'regression': [False, False, False]
    })
    df.to_csv(csv, index=False)
    out = p / 'perf_plot.png'
    ok = make_plot(csv, out)
    assert ok is True
    assert out.exists()
