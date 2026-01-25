def test_warmup_numba_returns_true():
    from cenop.optimizations.numba_helpers import warmup_numba
    assert warmup_numba() is True
