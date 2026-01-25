from cenop.ui.tabs import settings


def test_data_available_has_refresh_button():
    panel = settings._data_available_panel()
    rep = repr(panel)
    assert "refresh_data_available" in rep or "refresh_data_available" in str(panel)
    assert "data_available_table" in rep or "data_available_table" in str(panel)
    # Ensure the UI includes a placeholder for per-landscape details
    assert "Details" in repr(panel) or "Details" in str(panel)
