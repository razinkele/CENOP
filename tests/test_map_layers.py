"""Tests for map layer builders.

shiny_deckgl is not pip-installed in the test environment, so we mock its
layer-builder functions before importing cenop.server.map_layers.
"""

import sys
import types
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Mock shiny_deckgl before importing map_layers
# ---------------------------------------------------------------------------

_shiny_deckgl = types.ModuleType("shiny_deckgl")


def _fake_layer(layer_id, data, **kwargs):
    """Return a dict mimicking a real shiny_deckgl layer."""
    return {"id": layer_id, "data": data, "visible": kwargs.get("visible", True), **kwargs}


_shiny_deckgl.icon_layer = _fake_layer
_shiny_deckgl.grid_layer = _fake_layer
_shiny_deckgl.scatterplot_layer = _fake_layer
sys.modules["shiny_deckgl"] = _shiny_deckgl

from cenop.server.map_layers import (  # noqa: E402
    build_porpoise_layer,
    build_depth_heatmap,
    build_foraging_heatmap,
    build_noise_construction_layer,
    build_noise_operational_layer,
    build_turbine_pole_layer,
    build_turbine_blade_layer,
)


class TestBuildPorpoiseLayer:
    def test_empty_returns_invisible(self):
        layer = build_porpoise_layer([])
        assert layer["id"] == "porpoises"
        assert layer["visible"] is False

    def test_with_positions_returns_visible(self):
        positions = [
            {"position": [21.0, 55.5], "heading": 90, "color": [0, 150, 255], "radius": 200},
        ]
        layer = build_porpoise_layer(positions)
        assert layer["id"] == "porpoises"
        assert layer["data"] == positions

    def test_layer_id_is_stable(self):
        """Layer ID must be stable for partial_update matching."""
        layer1 = build_porpoise_layer([])
        layer2 = build_porpoise_layer([{"position": [0, 0], "heading": 0, "color": [0, 0, 0], "radius": 100}])
        assert layer1["id"] == layer2["id"] == "porpoises"


class TestBuildDepthHeatmap:
    def test_empty_returns_invisible(self):
        layer = build_depth_heatmap([])
        assert layer["id"] == "depth-heatmap"
        assert layer["visible"] is False

    def test_with_points(self):
        points = [[21.0, 55.5, 42.0]]
        layer = build_depth_heatmap(points)
        assert layer["id"] == "depth-heatmap"
        assert layer["data"] == points


class TestBuildForagingHeatmap:
    def test_empty_returns_invisible(self):
        layer = build_foraging_heatmap([])
        assert layer["id"] == "foraging-heatmap"
        assert layer["visible"] is False


class TestBuildNoiseLayers:
    def test_construction_empty(self):
        layer = build_noise_construction_layer([])
        assert layer["id"] == "noise-construction"
        assert layer["visible"] is False

    def test_operational_empty(self):
        layer = build_noise_operational_layer([])
        assert layer["id"] == "noise-operational"
        assert layer["visible"] is False


class TestBuildTurbineLayers:
    def test_poles_empty(self):
        layer = build_turbine_pole_layer([])
        assert layer["id"] == "turbine-poles"
        assert layer["visible"] is False

    def test_blades_empty(self):
        layer = build_turbine_blade_layer([])
        assert layer["id"] == "turbine-blades"
        assert layer["visible"] is False

    def test_blades_with_rotation(self):
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, rotation=45.0)
        assert layer["id"] == "turbine-blades"
        assert layer["data"] == data

    def test_layer_ids_are_distinct(self):
        """All layer IDs must be unique for partial_update matching."""
        ids = {
            build_porpoise_layer([])["id"],
            build_depth_heatmap([])["id"],
            build_foraging_heatmap([])["id"],
            build_noise_construction_layer([])["id"],
            build_noise_operational_layer([])["id"],
            build_turbine_pole_layer([])["id"],
            build_turbine_blade_layer([])["id"],
        }
        assert len(ids) == 7, "All layer IDs must be unique"
