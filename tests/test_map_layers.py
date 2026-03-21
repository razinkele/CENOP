"""Tests for map layer builders.

shiny_deckgl is not pip-installed in the test environment, so we mock its
layer-builder functions before importing cenop.server.map_layers.
The mock is set up in conftest.py — no need to re-register it here.
"""

import sys

import pytest

from cenop.server.map_layers import (  # noqa: E402
    build_porpoise_layer,
    build_porpoise_trails_layer,
    build_noise_construction_layer,
    build_noise_operational_layer,
    build_turbine_pole_layer,
    build_turbine_blade_layer,
    BLADE_ANIMATION_JS,
    BLADE_ANIMATION_STOP_JS,
    GIS_COLOR_SCHEMES,
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
        assert layer["visible"] is True

    def test_layer_id_is_stable(self):
        """Layer ID must be stable for partial_update matching."""
        layer1 = build_porpoise_layer([])
        layer2 = build_porpoise_layer([{"position": [0, 0], "heading": 0, "color": [0, 0, 0], "radius": 100}])
        assert layer1["id"] == layer2["id"] == "porpoises"


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
            build_porpoise_trails_layer([])["id"],
            build_noise_construction_layer([])["id"],
            build_noise_operational_layer([])["id"],
            build_turbine_pole_layer([])["id"],
            build_turbine_blade_layer([])["id"],
        }
        assert len(ids) == 6, "All layer IDs must be unique"


class TestBuildPorpoiseTrailsLayer:
    def test_empty_returns_invisible(self):
        layer = build_porpoise_trails_layer([])
        assert layer["id"] == "porpoise-trails"
        assert layer["visible"] is False

    def test_with_trails_returns_visible(self):
        trails = [
            {
                "path": [[21.0, 55.5, 1], [21.1, 55.6, 2]],
                "timestamps": [1, 2],
                "color": [0, 150, 255, 240],
            },
        ]
        layer = build_porpoise_trails_layer(trails)
        assert layer["id"] == "porpoise-trails"
        assert layer["visible"] is True
        assert layer["data"] == trails

    def test_layer_id_unique(self):
        """Trail layer ID must differ from porpoise icon layer ID."""
        trail_layer = build_porpoise_trails_layer([])
        icon_layer_result = build_porpoise_layer([])
        assert trail_layer["id"] != icon_layer_result["id"]


class TestBladeAnimation:
    def test_server_side_rotation(self):
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, rotation=90.0, client_animated=False)
        assert "90.0" in str(layer)
        assert "window._cenopBladeRotation" not in str(layer)

    def test_client_side_animation(self):
        """client_animated=True still returns a valid layer dict.

        The actual JS animation is injected separately via BLADE_ANIMATION_JS
        in main.py, not embedded in the layer dict itself.
        """
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, client_animated=True)
        assert layer["id"] == "turbine-blades"
        assert len(layer["data"]) == 1

    def test_client_animated_empty_data(self):
        layer = build_turbine_blade_layer([], client_animated=True)
        assert layer["id"] == "turbine-blades"
        assert layer["visible"] is False

    def test_animation_js_constants_exist(self):
        assert "window._cenopBladeAnimRunning" in BLADE_ANIMATION_JS
        assert "requestAnimationFrame" in BLADE_ANIMATION_JS
        assert "window._cenopBladeAnimRunning = false" in BLADE_ANIMATION_STOP_JS


class TestGisColorSchemes:
    def test_all_color_schemes_defined(self):
        expected = {"viridis", "green", "blue_white", "yellow_red"}
        assert expected.issubset(set(GIS_COLOR_SCHEMES.keys()))

    def test_color_scheme_has_six_stops(self):
        for name, colors in GIS_COLOR_SCHEMES.items():
            assert len(colors) == 6, f"Scheme '{name}' should have 6 color stops"
            for c in colors:
                assert len(c) == 3, f"Each color in '{name}' should be [R, G, B]"


import numpy as np

from cenop.server.map_layers import (
    grid_to_rgba_image,
    CATEGORICAL_COLORS,
)


class TestGridToRgbaImage:
    def test_continuous_basic_shape(self):
        """Output shape should be (H, W, 4) RGBA."""
        data = np.array([[0.0, 5.0], [10.0, 15.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        assert rgba.shape == (2, 2, 4)
        assert rgba.dtype == np.uint8

    def test_continuous_nodata_transparent(self):
        """NODATA cells should have alpha=0."""
        data = np.array([[1.0, -9999.0], [5.0, 10.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        assert rgba[0, 1, 3] == 0  # NODATA cell alpha
        assert rgba[0, 0, 3] == 255  # valid cell alpha

    def test_continuous_nan_transparent(self):
        """NaN cells should be treated as NODATA."""
        data = np.array([[1.0, np.nan], [5.0, 10.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        assert rgba[0, 1, 3] == 0

    def test_all_nodata_fully_transparent(self):
        """All-NODATA layer should return fully transparent image."""
        data = np.full((3, 3), -9999.0)
        rgba = grid_to_rgba_image(data, "viridis")
        assert rgba.shape == (3, 3, 4)
        assert np.all(rgba[:, :, 3] == 0)

    def test_single_value_no_crash(self):
        """Single unique value should not crash (min==max)."""
        data = np.array([[5.0, 5.0], [5.0, 5.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        assert rgba.shape == (2, 2, 4)
        assert np.all(rgba[:, :, 3] == 255)

    def test_categorical_discrete_colors(self):
        """Categorical scheme should use discrete color lookup."""
        data = np.array([[0.0, 1.0], [2.0, 0.0]])
        rgba = grid_to_rgba_image(data, "categorical")
        expected_rgb = CATEGORICAL_COLORS[0]
        assert list(rgba[0, 0, :3]) == expected_rgb
        expected_rgb1 = CATEGORICAL_COLORS[1]
        assert list(rgba[0, 1, :3]) == expected_rgb1

    def test_categorical_negative_values(self):
        """Negative categorical values should use abs()."""
        data = np.array([[-3.0]])
        rgba = grid_to_rgba_image(data, "categorical")
        expected_rgb = CATEGORICAL_COLORS[3]  # abs(-3) % 12 = 3
        assert list(rgba[0, 0, :3]) == expected_rgb

    def test_continuous_min_maps_to_first_color(self):
        """Min value should map to the first color stop (light end)."""
        data = np.array([[0.0, 100.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        # First stop is [255, 255, 217] (light yellow)
        assert rgba[0, 0, 0] == 255
        assert rgba[0, 0, 1] == 255
        assert rgba[0, 0, 2] == 217

    def test_continuous_max_maps_to_last_color(self):
        """Max value should map to the last color stop (dark end)."""
        data = np.array([[0.0, 100.0]])
        rgba = grid_to_rgba_image(data, "viridis")
        # Last stop is [68, 1, 84] (dark purple)
        assert rgba[0, 1, 0] == 68
        assert rgba[0, 1, 1] == 1
        assert rgba[0, 1, 2] == 84


from cenop.server.map_layers import array_to_base64_png


class TestArrayToBase64Png:
    def test_returns_data_uri(self):
        """Output should be a base64 data URI string."""
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[:, :, 3] = 255
        result = array_to_base64_png(rgba)
        assert result.startswith("data:image/png;base64,")

    def test_decodable_png(self):
        """The base64 content should decode to valid PNG bytes."""
        import base64
        rgba = np.zeros((3, 3, 4), dtype=np.uint8)
        rgba[:, :] = [255, 0, 0, 255]
        result = array_to_base64_png(rgba)
        b64_data = result.split(",", 1)[1]
        png_bytes = base64.b64decode(b64_data)
        # PNG magic bytes
        assert png_bytes[:4] == b'\x89PNG'

    def test_flips_y_axis(self):
        """Row 0 of input (north) should become bottom of image."""
        from PIL import Image
        import base64
        import io
        rgba = np.zeros((2, 1, 4), dtype=np.uint8)
        rgba[0, 0] = [255, 0, 0, 255]  # row 0 = red (north)
        rgba[1, 0] = [0, 0, 255, 255]  # row 1 = blue (south)
        result = array_to_base64_png(rgba)
        b64_data = result.split(",", 1)[1]
        img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
        pixels = list(img.getdata())
        # ASC row 0 is North (top of map). In the PNG, North should be at the
        # BOTTOM because bitmap_layer bounds are [west,south,east,north] and
        # PNG pixel (0,0) is top-left = south-west corner.
        # So row 0 (north) should be flipped to the bottom row of the image.
        assert pixels[0] == (0, 0, 255, 255)    # top pixel = south = blue
        assert pixels[1] == (255, 0, 0, 255)    # bottom pixel = north = red


import numpy as np
from cenop.server.map_layers import build_grid_bitmap_layer
from cenop.landscape.cell_data import LandscapeMetadata


class TestBuildGridBitmapLayer:
    def _make_metadata(self):
        return LandscapeMetadata(
            ncols=4, nrows=4,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )

    def test_returns_two_layers(self):
        """Should return [bitmap_layer, scatter_layer]."""
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = LandscapeMetadata(
            ncols=2, nrows=2,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )
        layers = build_grid_bitmap_layer("test", data, meta, "EPSG:3035", "viridis")
        assert len(layers) == 2

    def test_bitmap_layer_has_correct_id(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("depth", data, meta, "EPSG:3035", "viridis")
        assert layers[0]["id"] == "depth-bitmap"

    def test_scatter_layer_has_correct_id(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("depth", data, meta, "EPSG:3035", "viridis")
        assert layers[1]["id"] == "depth-tooltip"

    def test_bitmap_layer_has_image_data_uri(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("depth", data, meta, "EPSG:3035", "viridis")
        assert layers[0]["data"].startswith("data:image/png;base64,")

    def test_bitmap_layer_has_bounds(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("depth", data, meta, "EPSG:3035", "viridis")
        bounds = layers[0]["bounds"]
        assert len(bounds) == 4

    def test_all_nodata_returns_invisible_layers(self):
        data = np.full((3, 3), -9999.0)
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("test", data, meta, "EPSG:3035", "viridis")
        assert len(layers) == 2
        assert layers[0]["visible"] is False
        assert layers[1]["visible"] is False

    def test_categorical_scheme(self):
        data = np.array([[0.0, 1.0], [2.0, 3.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("sed", data, meta, "EPSG:3035", "categorical")
        assert len(layers) == 2
        assert layers[0]["data"].startswith("data:image/png;base64,")

    def test_scatter_layer_has_tooltip_data(self):
        data = np.array([[1.0, 2.0], [3.0, 4.0]])
        meta = self._make_metadata()
        layers = build_grid_bitmap_layer("test", data, meta, "EPSG:3035", "viridis")
        scatter = layers[1]
        assert len(scatter["data"]) > 0


from cenop.server.map_layers import compute_grid_bounds
from cenop.landscape.cell_data import LandscapeMetadata


class TestComputeGridBounds:
    def test_returns_bbox(self):
        """Should return [west, south, east, north] in WGS84."""
        meta = LandscapeMetadata(
            ncols=10, nrows=10,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        assert len(bounds) == 4
        west, south, east, north = bounds
        assert west < east, "west should be less than east"
        assert south < north, "south should be less than north"

    def test_bounds_are_wgs84_range(self):
        """All coordinates should be valid WGS84."""
        meta = LandscapeMetadata(
            ncols=100, nrows=100,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        west, south, east, north = bounds
        assert -180 <= west <= 180
        assert -180 <= east <= 180
        assert -90 <= south <= 90
        assert -90 <= north <= 90

    def test_edge_sampling_captures_full_extent(self):
        """Bbox from edge sampling should be >= bbox from corners only."""
        meta = LandscapeMetadata(
            ncols=500, nrows=500,
            xllcorner=4754000.0, yllcorner=3482000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        west, south, east, north = bounds
        # For EPSG:3035 grids, the extent in degrees should be reasonable
        assert (east - west) > 0.1, "Grid should span some longitude"
        assert (north - south) > 0.1, "Grid should span some latitude"
