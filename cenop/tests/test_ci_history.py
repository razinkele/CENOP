import csv
from pathlib import Path
from scripts.ci_benchmark import write_history_entry


def test_write_history_entry(tmp_path):
    out = tmp_path / 'perf_history.csv'
    result = {
        'timestamp': '2026-01-25T00:00:00Z',
        'ticks': 50,
        'porpoise': 20,
        'seed': 123,
        'elapsed': 1.23,
        'returncode': 0,
        'baseline_elapsed': 1.0,
        'threshold': 1.25,
        'regression': False
    }
    write_history_entry(result, path=str(out))
    assert out.exists()
    with out.open('r', encoding='utf-8') as f:
        reader = list(csv.DictReader(f))
    assert len(reader) == 1
    row = reader[0]
    assert row['ticks'] == '50'
    assert row['porpoise'] == '20'
    assert float(row['elapsed']) == 1.23
