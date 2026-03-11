"""Map layer builders for the shiny-deckgl MapWidget.

Each function returns a layer dict compatible with MapWidget.update().
"""

import numpy as np
from shiny_deckgl import icon_layer, scatterplot_layer, bitmap_layer


# -- Porpoise arrow icon (base64 SVG) --
PORPOISE_ICON_ATLAS = (
    "data:image/svg+xml;base64,"
    "PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjxzdmcgeG1s"
    "bnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMzIiIGhlaWdo"
    "dD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiI+DQogIDxnIGZpbGw9Im5vbmUiIHN0"
    "cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJv"
    "dW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4NCiAgICA8IS0tIEFycm93IHNo"
    "YWZ0IC0tPg0KICAgIDxsaW5lIHgxPSIxNiIgeTE9IjI2IiB4Mj0iMTYiIHkyPSI4"
    "IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIgLz4NCiAgICA8IS0t"
    "IEFycm93IGhlYWQgLS0+DQogICAgPHBvbHlsaW5lIHBvaW50cz0iOCwxMiAxNiw0"
    "IDI0LDEyIiBmaWxsPSIjZmZmZmZmIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13"
    "aWR0aD0iMyIvPg0KICA8L2c+DQo8L3N2Zz4="
)
PORPOISE_ICON_MAPPING = {
    "arrow": {"x": 0, "y": 0, "width": 32, "height": 32,
              "anchorY": 16, "anchorX": 16},
}

# -- Turbine pole icon --
TURBINE_POLE_ATLAS = (
    "data:image/svg+xml;utf8,"
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" '
    'viewBox="0 0 48 48"><g fill="white">'
    '<rect x="22" y="14" width="4" height="32"/>'
    '<circle cx="24" cy="14" r="3"/></g></svg>'
)
TURBINE_POLE_MAPPING = {
    "pole": {"x": 0, "y": 0, "width": 48, "height": 48,
             "anchorY": 14, "anchorX": 24},
}

# -- Turbine blade icon --
TURBINE_BLADE_ATLAS = (
    "data:image/svg+xml;utf8,"
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" '
    'viewBox="0 0 48 48"><g fill="white">'
    '<path d="M22 14 L24 1 L26 14 Z"/>'
    '<path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/>'
    '<path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/>'
    '</g></svg>'
)
TURBINE_BLADE_MAPPING = {
    "blade": {"x": 0, "y": 0, "width": 48, "height": 48,
              "anchorY": 14, "anchorX": 24},
}


# -- Standard color ramps for GIS layers (6-stop gradients) --
GIS_COLOR_SCHEMES = {
    "viridis": [
        [255, 255, 217], [253, 231, 37], [94, 201, 98],
        [33, 145, 140], [59, 82, 139], [68, 1, 84],
    ],
    "green": [
        [8, 48, 20], [20, 100, 40], [40, 160, 60],
        [80, 200, 80], [140, 230, 100], [200, 255, 140],
    ],
    "blue_white": [
        [245, 250, 255], [200, 220, 240], [140, 180, 230],
        [90, 140, 215], [50, 100, 200], [20, 60, 180],
    ],
    "yellow_red": [
        [255, 255, 178], [254, 204, 92], [253, 141, 60],
        [240, 59, 32], [189, 0, 38], [128, 0, 38],
    ],
}

CATEGORICAL_COLORS = [
    [31,119,180],[255,127,14],[44,160,44],[214,39,40],
    [148,103,189],[140,86,75],[227,119,194],[127,127,127],
    [188,189,34],[23,190,207],[174,199,232],[255,187,120],
]


def grid_to_rgba_image(
    data: np.ndarray,
    scheme: str,
    nodata: float = -9999.0,
) -> np.ndarray:
    """Convert a 2D grid array to an RGBA image array.

    Each cell becomes one pixel. Continuous schemes use linear interpolation
    through the color gradient; categorical schemes use discrete lookup.

    Args:
        data: 2D NumPy array of grid values.
        scheme: Color scheme name — key in GIS_COLOR_SCHEMES for continuous,
                or "categorical" for discrete lookup.
        nodata: Value treated as missing (rendered transparent).

    Returns:
        uint8 NumPy array of shape (H, W, 4) — RGBA pixels.
    """
    h, w = data.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)

    # Build mask of valid (non-NODATA, non-NaN) cells
    valid = (data != nodata) & ~np.isnan(data)

    if not np.any(valid):
        return rgba  # fully transparent

    if scheme == "categorical":
        indices = np.abs(np.round(data)).astype(int) % len(CATEGORICAL_COLORS)
        palette = np.array(CATEGORICAL_COLORS, dtype=np.uint8)
        for row in range(h):
            for col in range(w):
                if valid[row, col]:
                    rgba[row, col, :3] = palette[indices[row, col]]
                    rgba[row, col, 3] = 255
    else:
        colors = np.array(
            GIS_COLOR_SCHEMES.get(scheme, GIS_COLOR_SCHEMES["viridis"]),
            dtype=np.float64,
        )
        n_stops = len(colors)
        valid_vals = data[valid]
        vmin = float(np.min(valid_vals))
        vmax = float(np.max(valid_vals))

        if vmin == vmax:
            # Single value — map to midpoint color
            mid = colors[n_stops // 2].astype(np.uint8)
            rgba[valid, :3] = mid
            rgba[valid, 3] = 255
        else:
            # Normalize to [0, 1] then interpolate through color stops
            norm = (data - vmin) / (vmax - vmin)
            norm = np.clip(norm, 0.0, 1.0)
            # Map [0,1] to color stop index
            t = norm * (n_stops - 1)
            idx = np.floor(t).astype(int)
            idx = np.clip(idx, 0, n_stops - 2)
            frac = t - idx

            for row in range(h):
                for col in range(w):
                    if valid[row, col]:
                        i = idx[row, col]
                        f = frac[row, col]
                        c = colors[i] * (1 - f) + colors[i + 1] * f
                        rgba[row, col, :3] = c.astype(np.uint8)
                        rgba[row, col, 3] = 255

    return rgba


def array_to_base64_png(rgba: np.ndarray) -> str:
    """Encode an RGBA image array as a base64 PNG data URI.

    The Y-axis is flipped so that array row 0 (north in ASC convention)
    becomes the bottom row of the PNG (since bitmap_layer uses
    [west, south, east, north] bounds and PNG pixel (0,0) is top-left).

    Args:
        rgba: uint8 NumPy array of shape (H, W, 4).

    Returns:
        Base64-encoded PNG as a ``data:image/png;base64,...`` string.
    """
    import base64
    import io
    from PIL import Image

    # Flip vertically: ASC row 0 = north → PNG bottom row
    flipped = np.flipud(rgba)
    img = Image.fromarray(flipped, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_porpoise_layer(positions: list[dict]) -> dict:
    """Build porpoise icon layer from position data."""
    if not positions:
        return icon_layer("porpoises", [], visible=False)

    return icon_layer(
        "porpoises",
        positions,
        iconAtlas=PORPOISE_ICON_ATLAS,
        iconMapping=PORPOISE_ICON_MAPPING,
        getIcon="arrow",
        getPosition="@@=d.position",
        getSize="@@=Math.max(8, (d.radius || 200) / 20)",
        sizeScale=1,
        getAngle="@@=d.heading || 0",
        getColor="@@=d.color || [0,150,255]",
        opacity=0.95,
        pickable=True,
    )


def build_noise_construction_layer(noise_points: list) -> dict:
    """Build construction noise scatterplot layer."""
    if not noise_points:
        return scatterplot_layer("noise-construction", [], visible=False)

    return scatterplot_layer(
        "noise-construction",
        noise_points,
        getPosition="@@=d.position",
        getRadius="@@=d.radius || 4000",
        getFillColor=[255, 60, 60, 80],
        getLineColor=[255, 60, 60, 160],
        lineWidthMinPixels=1,
        stroked=True,
        filled=True,
        radiusMinPixels=5,
        pickable=False,
    )


def build_noise_operational_layer(noise_points: list) -> dict:
    """Build operational noise scatterplot layer."""
    if not noise_points:
        return scatterplot_layer("noise-operational", [], visible=False)

    return scatterplot_layer(
        "noise-operational",
        noise_points,
        getPosition="@@=d.position",
        getRadius="@@=d.radius || 500",
        getFillColor=[255, 200, 60, 50],
        getLineColor=[255, 200, 60, 120],
        lineWidthMinPixels=1,
        stroked=True,
        filled=True,
        radiusMinPixels=3,
        pickable=False,
    )


def build_turbine_pole_layer(turbine_data: list) -> dict:
    """Build turbine pole icon layer."""
    if not turbine_data:
        return icon_layer("turbine-poles", [], visible=False)

    return icon_layer(
        "turbine-poles",
        turbine_data,
        iconAtlas=TURBINE_POLE_ATLAS,
        iconMapping=TURBINE_POLE_MAPPING,
        getIcon="pole",
        getPosition="@@=d.position",
        getSize="@@=Math.max(20, Math.min(64, (d.radius || 300) / 15))",
        getColor="@@=d.phase === 'construction' ? [255,70,40,220] : d.phase === 'operational' ? [50,160,240,220] : d.phase === 'planned' ? [180,180,180,180] : d.color || [255,140,60]",
        pickable=True,
        opacity=0.95,
    )


def build_turbine_blade_layer(turbine_data: list, rotation: float = 0,
                               client_animated: bool = False) -> dict:
    """Build turbine blade icon layer with rotation angle.

    Args:
        turbine_data: List of turbine position dicts.
        rotation: Server-side rotation angle (used when client_animated=False).
        client_animated: If True, use JS-side animation variable for angle.
    """
    if not turbine_data:
        return icon_layer("turbine-blades", [], visible=False)

    if client_animated:
        angle_expr = "@@=d.phase === 'operational' ? (window._cenopBladeRotation || 0) : 0"
    else:
        angle_expr = f"@@=d.phase === 'operational' ? {rotation} : 0"

    return icon_layer(
        "turbine-blades",
        turbine_data,
        iconAtlas=TURBINE_BLADE_ATLAS,
        iconMapping=TURBINE_BLADE_MAPPING,
        getIcon="blade",
        getPosition="@@=d.position",
        getSize="@@=Math.max(20, Math.min(64, (d.radius || 300) / 15))",
        getAngle=angle_expr,
        getColor="@@=d.phase === 'construction' ? [255,70,40,220] : d.phase === 'operational' ? [50,160,240,220] : d.phase === 'planned' ? [180,180,180,180] : d.color || [255,140,60]",
        pickable=False,
        opacity=0.95,
    )


BLADE_ANIMATION_JS = """
<script>
(function() {
    // Stop any existing animation loop before starting a new one
    window._cenopBladeAnimRunning = false;
    window._cenopBladeRotation = window._cenopBladeRotation || 0;
    // Allow previous loop to exit on next frame, then start fresh
    setTimeout(function() {
        window._cenopBladeAnimRunning = true;

        function animateBlades() {
            if (!window._cenopBladeAnimRunning) return;
        window._cenopBladeRotation = (window._cenopBladeRotation + 1.5) % 360;
        var widget = document.querySelector('[data-widget-id="sim_map"]');
        if (widget && widget.__deckgl_instance) {
            var inst = widget.__deckgl_instance;
            if (inst.lastLayers) {
                var bladeIdx = inst.lastLayers.findIndex(function(l) {
                    return l.id === 'turbine-blades';
                });
                if (bladeIdx >= 0) {
                    inst.overlay.setProps({layers: inst.overlay.props.layers});
                }
            }
        }
        requestAnimationFrame(animateBlades);
    }
    requestAnimationFrame(animateBlades);
    }, 20);
})();
</script>
"""

BLADE_ANIMATION_STOP_JS = """
<script>
window._cenopBladeAnimRunning = false;
</script>
"""


def compute_grid_bounds(metadata, source_crs: str) -> list[list[float]]:
    """Compute WGS84 corner coordinates for an ASC grid.

    Transforms the grid's four corners from the native CRS to WGS84
    and returns them in deck.gl's four-corner format for bitmap_layer.
    This correctly handles grids in projections like EPSG:3035 (LAEA)
    that are rotated relative to WGS84 latitude/longitude axes.

    Args:
        metadata: LandscapeMetadata with xllcorner, yllcorner, ncols,
                  nrows, cellsize.
        source_crs: EPSG string for the grid's native CRS.

    Returns:
        Four corners as [[SW_lon, SW_lat], [NW_lon, NW_lat],
        [NE_lon, NE_lat], [SE_lon, SE_lat]] in WGS84 degrees.
    """
    from cenop.server.main import _get_transformer

    transformer = _get_transformer(source_crs)

    # Grid corners in projected CRS
    x_min = metadata.xllcorner
    y_min = metadata.yllcorner
    x_max = metadata.xllcorner + metadata.ncols * metadata.cellsize
    y_max = metadata.yllcorner + metadata.nrows * metadata.cellsize

    # Transform corners: SW, NW, NE, SE (deck.gl bitmap_layer order)
    lons, lats = transformer.transform(
        [x_min, x_min, x_max, x_max],
        [y_min, y_max, y_max, y_min],
    )

    return [[lons[i], lats[i]] for i in range(4)]


def build_grid_bitmap_layer(
    layer_id: str,
    data: np.ndarray,
    metadata,
    source_crs: str,
    scheme: str,
    nodata: float = -9999.0,
) -> list[dict]:
    """Build a bitmap + tooltip scatter layer pair for grid data.

    Replaces build_depth_heatmap, build_foraging_heatmap, build_gis_cell_layer,
    and categorical scatterplot rendering with a single unified pipeline.

    Args:
        layer_id: Base layer identifier (e.g. "depth", "gis-sediment").
        data: 2D NumPy array of grid values.
        metadata: LandscapeMetadata with grid geometry.
        source_crs: EPSG string for the grid's native CRS.
        scheme: Color scheme — key in GIS_COLOR_SCHEMES or "categorical".
        nodata: NODATA sentinel value.

    Returns:
        List of two layer dicts: [bitmap_layer, scatter_tooltip_layer].
    """
    valid = (data != nodata) & ~np.isnan(data)

    if not np.any(valid):
        return [
            bitmap_layer(f"{layer_id}-bitmap", "", visible=False),
            scatterplot_layer(f"{layer_id}-tooltip", [], visible=False),
        ]

    # Generate RGBA image and encode as PNG
    rgba = grid_to_rgba_image(data, scheme, nodata=nodata)
    image_uri = array_to_base64_png(rgba)
    bounds = compute_grid_bounds(metadata, source_crs)

    bmp = bitmap_layer(
        f"{layer_id}-bitmap",
        image_uri,
        bounds=bounds,
        opacity=0.85,
        pickable=False,
    )

    # Build tooltip scatter layer with adaptive sampling
    from cenop.server.main import grid_to_lonlat

    h, w = data.shape
    total_cells = h * w
    max_points = 15000
    sample_step = max(1, int((total_cells / max_points) ** 0.5))

    tooltip_data = []
    for row in range(0, h, sample_step):
        for col in range(0, w, sample_step):
            val = float(data[row, col])
            if not valid[row, col]:
                continue
            lon, lat = grid_to_lonlat(col, row, metadata, source_crs)
            tooltip_data.append({
                "position": [lon, lat],
                "value": round(val, 4),
            })

    scatter = scatterplot_layer(
        f"{layer_id}-tooltip",
        tooltip_data,
        getPosition="@@=d.position",
        getRadius=max(100, metadata.cellsize / 2),
        getFillColor=[0, 0, 0, 1],
        stroked=False,
        filled=True,
        radiusMinPixels=3,
        pickable=True,
    )

    return [bmp, scatter]
