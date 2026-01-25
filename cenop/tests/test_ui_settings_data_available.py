import pytest

from cenop.ui.tabs import settings


def test_data_available_panel_runs():
    """Ensure the `_data_available_panel` constructs without raising."""
    panel = settings._data_available_panel()
    assert panel is not None
    # Basic structural check: should have title "Data Available" in representation
    repr_str = repr(panel)
    assert "Data Available" in repr_str or "Data Available" in str(panel)
