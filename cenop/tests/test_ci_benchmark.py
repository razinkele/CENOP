from scripts.ci_benchmark import compute_threshold


def test_compute_threshold():
    baseline = 10.0
    expected = baseline * 1.25 + 2.0
    assert compute_threshold(baseline) == expected
