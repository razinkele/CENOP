# Ensure the local `src` package dir is available during tests
# This helps pytest discover `cenop` when running tests without an editable install.
import sys
import os
import typing

# Patch coverage.types for numba compatibility (numba 0.64 expects attrs
# that were renamed/removed in newer coverage versions).
try:
    import coverage.types as _ct
    if not hasattr(_ct, 'Tracer') and hasattr(_ct, 'TracerCore'):
        _ct.Tracer = _ct.TracerCore
    if not hasattr(_ct, 'TTracer') and hasattr(_ct, 'TracerCore'):
        _ct.TTracer = _ct.TracerCore
    for _attr in ('TShouldTraceFn', 'TShouldStartContextFn'):
        if not hasattr(_ct, _attr):
            setattr(_ct, _attr, typing.Any)
except ImportError:
    pass

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC = os.path.join(ROOT, 'src')
if SRC not in sys.path:
    sys.path.insert(0, SRC)

# ---------------------------------------------------------------------------
# Mock shiny_deckgl — the package is not pip-installed in the test env.
# This must happen before any cenop module tries to import it.
# ---------------------------------------------------------------------------
import types
from unittest.mock import MagicMock

if "shiny_deckgl" not in sys.modules:
    _shiny_deckgl = types.ModuleType("shiny_deckgl")

    def _fake_layer(layer_id, data, **kwargs):
        """Return a dict mimicking a real shiny_deckgl layer."""
        return {"id": layer_id, "data": data, "visible": kwargs.get("visible", True), **kwargs}

    _shiny_deckgl.icon_layer = _fake_layer
    _shiny_deckgl.grid_layer = _fake_layer
    _shiny_deckgl.scatterplot_layer = _fake_layer
    _shiny_deckgl.bitmap_layer = _fake_layer
    _shiny_deckgl.trips_layer = _fake_layer

    # Widget / control helpers used by server code
    _shiny_deckgl.deck_legend_control = MagicMock(name="deck_legend_control")
    _shiny_deckgl.scale_widget = MagicMock(name="scale_widget")
    _shiny_deckgl.zoom_widget = MagicMock(name="zoom_widget")
    _shiny_deckgl.compass_widget = MagicMock(name="compass_widget")
    _shiny_deckgl.fullscreen_widget = MagicMock(name="fullscreen_widget")
    _shiny_deckgl.head_includes = MagicMock(name="head_includes")

    # MapWidget mock with .ui(), .update(), .fly_to() methods
    # .ui() must return valid HTML (not a MagicMock) since Shiny validates tag children
    from htmltools import tags as _html_tags

    class _MockMapWidget:
        def __init__(self, *args, **kwargs):
            self._update = MagicMock(name="MapWidget.update")
            self._fly_to = MagicMock(name="MapWidget.fly_to")

        def ui(self, **kwargs):
            return _html_tags.div("mock-map-widget", id="mock-map")

        async def update(self, *args, **kwargs):
            return self._update(*args, **kwargs)

        async def fly_to(self, *args, **kwargs):
            return self._fly_to(*args, **kwargs)

    _shiny_deckgl.MapWidget = _MockMapWidget

    # Style constants
    _shiny_deckgl.CARTO_DARK = "carto-dark"
    _shiny_deckgl.CARTO_POSITRON = "carto-positron"

    sys.modules["shiny_deckgl"] = _shiny_deckgl
