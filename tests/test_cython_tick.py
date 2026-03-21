"""Tests for Cython tick acceleration module."""

import numpy as np
import pytest
import sys
import os

def _try_import_cython():
    try:
        from cenop.optimizations.tick_cython import cython_available
        return True
    except ImportError:
        return False

CYTHON_OK = _try_import_cython()


@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonAvailable:
    def test_module_loads(self):
        from cenop.optimizations.tick_cython import cython_available
        assert cython_available() is True
