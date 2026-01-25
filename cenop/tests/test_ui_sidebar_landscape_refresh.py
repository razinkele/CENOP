from cenop.ui import sidebar


def test_sidebar_contains_refresh_and_selector():
    s = sidebar.create_sidebar()
    rep = repr(s)
    assert "refresh_landscapes" in rep or "refresh_landscapes" in str(s)
    assert "landscape_selector" in rep or "landscape_selector" in str(s)
