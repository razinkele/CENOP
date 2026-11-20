"""Navbar extras: dark-mode switch, share (bookmark) button, and the UI as a function."""

from cenop.ui.layout import app_ui


def test_ui_is_a_function_of_the_request_for_bookmarks():
    html = str(app_ui(None))
    assert "bslib-input-dark-mode" in html
    assert 'id="._bookmark_"' in html
    assert 'id="help_btn"' in html


def test_dark_mode_overrides_the_light_shell_colours():
    html = str(app_ui(None))
    assert '[data-bs-theme="dark"]' in html
    assert "--coastal-sidebar: #101c27" in html
