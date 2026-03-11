"""Tests for map layer builders.

shiny_deckgl is not pip-installed in the test environment, so we mock its
layer-builder functions before importing cenop.server.map_layers.
The mock is set up in conftest.py — no need to re-register it here.
"""

import sys

import pytest

from cenop.server.map_layers import (  # noqa: E402
    build_porpoise_layer,
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
            build_noise_construction_layer([])["id"],
            build_noise_operational_layer([])["id"],
            build_turbine_pole_layer([])["id"],
            build_turbine_blade_layer([])["id"],
        }
        assert len(ids) == 5, "All layer IDs must be unique"


class TestBladeAnimation:
    def test_server_side_rotation(self):
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, rotation=90.0, client_animated=False)
        assert "90.0" in str(layer)
        assert "window._cenopBladeRotation" not in str(layer)

    def test_client_side_animation(self):
        data = [{"position": [21.0, 55.5], "radius": 300, "phase": "operational"}]
        layer = build_turbine_blade_layer(data, client_animated=True)
        assert "window._cenopBladeRotation" in str(layer)

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
    def test_returns_four_corners(self):
        """Should return [[SW], [NW], [NE], [SE]] in WGS84."""
        meta = LandscapeMetadata(
            ncols=10, nrows=10,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        assert len(bounds) == 4
        for corner in bounds:
            assert len(corner) == 2, f"Each corner should be [lon, lat], got {corner}"
        sw, nw, ne, se = bounds
        assert sw[1] < nw[1], "SW lat should be less than NW lat"
        assert se[0] > sw[0], "SE lon should be greater than SW lon"

    def test_corners_are_wgs84_range(self):
        """All corner coordinates should be valid WGS84."""
        meta = LandscapeMetadata(
            ncols=100, nrows=100,
            xllcorner=4321000.0, yllcorner=3210000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        for corner in bounds:
            lon, lat = corner
            assert -180 <= lon <= 180, f"Longitude {lon} out of range"
            assert -90 <= lat <= 90, f"Latitude {lat} out of range"

    def test_laea_corners_are_not_axis_aligned(self):
        """EPSG:3035 (LAEA) grids should produce rotated corners in WGS84."""
        meta = LandscapeMetadata(
            ncols=500, nrows=500,
            xllcorner=4754000.0, yllcorner=3482000.0,
            cellsize=400.0,
        )
        bounds = compute_grid_bounds(meta, "EPSG:3035")
        sw, nw, ne, se = bounds
        # Bottom edge (SW to SE) should NOT have equal latitudes
        assert abs(sw[1] - se[1]) > 0.01, (
            f"LAEA bottom edge should be tilted, but SW lat={sw[1]:.4f} ~ SE lat={se[1]:.4f}"
        )
