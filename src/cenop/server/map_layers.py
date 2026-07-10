"""Map layer builders for the shiny-deckgl MapWidget.

Each function returns a layer dict compatible with MapWidget.update().
"""

import numpy as np
from shiny_deckgl import bitmap_layer, icon_layer, scatterplot_layer, trips_layer
from shiny_deckgl.ibm import ICON_ATLAS as IBM_ICON_ATLAS
from shiny_deckgl.ibm import ICON_MAPPING as IBM_ICON_MAPPING

# -- Porpoise icon: use IBM marine species sprite atlas --
PORPOISE_ICON_ATLAS = IBM_ICON_ATLAS
PORPOISE_ICON_MAPPING = IBM_ICON_MAPPING

# -- Turbine pole icon --
TURBINE_POLE_ATLAS = (
    "data:image/svg+xml;utf8,"
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" '
    'viewBox="0 0 48 48"><g fill="white">'
    '<rect x="22" y="14" width="4" height="32"/>'
    '<circle cx="24" cy="14" r="3"/></g></svg>'
)
TURBINE_POLE_MAPPING = {
    "pole": {"x": 0, "y": 0, "width": 48, "height": 48, "anchorY": 14, "anchorX": 24},
}

# -- Turbine blade icon --
TURBINE_BLADE_ATLAS = (
    "data:image/svg+xml;utf8,"
    '<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" '
    'viewBox="0 0 48 48"><g fill="white">'
    '<path d="M22 14 L24 1 L26 14 Z"/>'
    '<path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/>'
    '<path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/>'
    "</g></svg>"
)
TURBINE_BLADE_MAPPING = {
    "blade": {"x": 0, "y": 0, "width": 48, "height": 48, "anchorY": 14, "anchorX": 24},
}


# -- Standard color ramps for GIS layers (6-stop gradients) --
GIS_COLOR_SCHEMES = {
    "viridis": [
        [255, 255, 217],
        [253, 231, 37],
        [94, 201, 98],
        [33, 145, 140],
        [59, 82, 139],
        [68, 1, 84],
    ],
    "green": [
        [8, 48, 20],
        [20, 100, 40],
        [40, 160, 60],
        [80, 200, 80],
        [140, 230, 100],
        [200, 255, 140],
    ],
    "blue_white": [
        [245, 250, 255],
        [200, 220, 240],
        [140, 180, 230],
        [90, 140, 215],
        [50, 100, 200],
        [20, 60, 180],
    ],
    "yellow_red": [
        [255, 255, 178],
        [254, 204, 92],
        [253, 141, 60],
        [240, 59, 32],
        [189, 0, 38],
        [128, 0, 38],
    ],
}

CATEGORICAL_COLORS = [
    [31, 119, 180],
    [255, 127, 14],
    [44, 160, 44],
    [214, 39, 40],
    [148, 103, 189],
    [140, 86, 75],
    [227, 119, 194],
    [127, 127, 127],
    [188, 189, 34],
    [23, 190, 207],
    [174, 199, 232],
    [255, 187, 120],
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

    The Y-axis is flipped so that array row 0 (south, after loader flipud)
    becomes the bottom row of the PNG.  PNG pixel (0,0) is top-left,
    so after the flip PNG row 0 = north, matching the bitmap_layer bounds
    [west, south, east, north] where the top of the image is north.

    Args:
        rgba: uint8 NumPy array of shape (H, W, 4).

    Returns:
        Base64-encoded PNG as a ``data:image/png;base64,...`` string.
    """
    import base64
    import io

    from PIL import Image

    # Flip vertically: row 0 = south → PNG bottom row, so PNG top = north
    flipped = np.flipud(rgba)
    img = Image.fromarray(flipped, mode="RGBA")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def build_porpoise_layer(positions: list[dict]) -> dict:
    """Build porpoise icon layer using IBM harbour porpoise sprite."""
    if not positions:
        return icon_layer("porpoises", [], visible=False)

    return icon_layer(
        "porpoises",
        positions,
        iconAtlas=PORPOISE_ICON_ATLAS,
        iconMapping=PORPOISE_ICON_MAPPING,
        getIcon="Harbour porpoise",
        getPosition="@@d.position",
        getSize=40,
        sizeMinPixels=16,
        sizeMaxPixels=48,
        getAngle="@@d.heading",
        getColor="@@d.color",
        opacity=0.9,
        pickable=True,
        visible=True,
    )


def build_porpoise_trails_layer(trails: list[dict], current_time: float = 0) -> dict:
    """Build porpoise trace layer using TripsLayer with decaying trails.

    Each trail is a dict with 'path' (list of [lon, lat, timestamp] coords)
    and 'color' ([r, g, b] or [r, g, b, a]).
    """
    if not trails:
        return trips_layer("porpoise-trails", [], visible=False)
    return trips_layer(
        "porpoise-trails",
        trails,
        getPath="@@d.path",
        getColor="@@d.color",
        currentTime=current_time,
        trailLength=180,
        fadeTrail=True,
        widthMinPixels=2,
        widthMaxPixels=4,
        opacity=0.7,
        visible=True,
    )


def build_noise_construction_layer(noise_points: list) -> dict:
    """Build construction noise scatterplot layer."""
    if not noise_points:
        return scatterplot_layer("noise-construction", [], visible=False)

    return scatterplot_layer(
        "noise-construction",
        noise_points,
        getPosition="@@d.position",
        getRadius="@@d.radius",
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
        getPosition="@@d.position",
        getRadius="@@d.radius",
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

    # Compute size and color server-side (was JS ternary expressions)
    _PHASE_COLORS = {
        "construction": [255, 70, 40, 220],
        "operational": [50, 160, 240, 220],
        "planned": [180, 180, 180, 180],
    }
    for t in turbine_data:
        t["size"] = max(20, min(64, (t.get("radius", 300)) / 15))
        t["color"] = _PHASE_COLORS.get(t.get("phase"), t.get("color", [255, 140, 60]))

    return icon_layer(
        "turbine-poles",
        turbine_data,
        iconAtlas=TURBINE_POLE_ATLAS,
        iconMapping=TURBINE_POLE_MAPPING,
        getIcon="pole",
        getPosition="@@d.position",
        getSize="@@d.size",
        getColor="@@d.color",
        pickable=True,
        opacity=0.95,
    )


def build_turbine_blade_layer(turbine_data: list, rotation: float = 0) -> dict:
    """Build turbine blade icon layer with rotation angle.

    Args:
        turbine_data: List of turbine position dicts.
        rotation: Server-side rotation angle applied to operational turbines.
    """
    if not turbine_data:
        return icon_layer("turbine-blades", [], visible=False)

    # Compute size, color, and angle server-side
    _PHASE_COLORS = {
        "construction": [255, 70, 40, 220],
        "operational": [50, 160, 240, 220],
        "planned": [180, 180, 180, 180],
    }
    for t in turbine_data:
        t["size"] = max(20, min(64, (t.get("radius", 300)) / 15))
        t["color"] = _PHASE_COLORS.get(t.get("phase"), t.get("color", [255, 140, 60]))
        t["angle"] = rotation if t.get("phase") == "operational" else 0

    return icon_layer(
        "turbine-blades",
        turbine_data,
        iconAtlas=TURBINE_BLADE_ATLAS,
        iconMapping=TURBINE_BLADE_MAPPING,
        getIcon="blade",
        getPosition="@@d.position",
        getSize="@@d.size",
        getAngle="@@d.angle",
        getColor="@@d.color",
        pickable=False,
        opacity=0.95,
    )


def compute_grid_bounds(metadata, source_crs: str) -> list[float]:
    """Compute WGS84 bounding box for an ASC grid.

    Samples many points along all four grid edges to find the accurate
    WGS84 bounding box, accounting for non-linear projection distortion.

    Args:
        metadata: LandscapeMetadata with xllcorner, yllcorner, ncols,
                  nrows, cellsize.
        source_crs: EPSG string for the grid's native CRS.

    Returns:
        [west, south, east, north] in WGS84 degrees.
    """
    from cenop.server.main import _get_transformer

    transformer = _get_transformer(source_crs)

    x_min = metadata.xllcorner
    y_min = metadata.yllcorner
    x_max = metadata.xllcorner + metadata.ncols * metadata.cellsize
    y_max = metadata.yllcorner + metadata.nrows * metadata.cellsize

    # Sample along all 4 edges for accurate bbox (edges curve in WGS84)
    n = 100
    edge_xs = np.concatenate(
        [
            np.linspace(x_min, x_max, n),  # south edge
            np.linspace(x_min, x_max, n),  # north edge
            np.full(n, x_min),  # west edge
            np.full(n, x_max),  # east edge
        ]
    )
    edge_ys = np.concatenate(
        [
            np.full(n, y_min),
            np.full(n, y_max),
            np.linspace(y_min, y_max, n),
            np.linspace(y_min, y_max, n),
        ]
    )
    lons, lats = transformer.transform(edge_xs, edge_ys)

    return [float(np.min(lons)), float(np.min(lats)), float(np.max(lons)), float(np.max(lats))]


def reproject_grid_to_wgs84(
    data: np.ndarray,
    metadata,
    source_crs: str,
    nodata: float = -9999.0,
) -> tuple[np.ndarray, list[float]]:
    """Reproject grid data to Web Mercator-aligned WGS84 pixel space.

    For each pixel in the output grid, inverse-transforms to the source CRS
    and samples the input via nearest-neighbor. Pixels are spaced equally in
    Web Mercator Y (not WGS84 latitude) so that deck.gl's linear texture
    interpolation in Mercator screen space reproduces the correct positions.

    The output array follows the same convention as the input: row 0 = south.

    Args:
        data: 2D array in source CRS (row 0 = south after loader flipud).
        metadata: LandscapeMetadata with grid geometry.
        source_crs: EPSG string for the grid's native CRS.
        nodata: NODATA sentinel value.

    Returns:
        (reprojected_data, bounds) where bounds is [west, south, east, north].
    """
    from pyproj import Transformer

    inv = Transformer.from_crs("EPSG:4326", source_crs, always_xy=True)

    # Compute WGS84 bounding box
    bounds = compute_grid_bounds(metadata, source_crs)
    west, south, east, north = bounds

    # Output resolution: match source cellsize at grid center latitude
    center_lat = (south + north) / 2
    m_per_deg_lat = 111320.0
    m_per_deg_lon = 111320.0 * np.cos(np.radians(center_lat))

    res_lat = metadata.cellsize / m_per_deg_lat
    res_lon = metadata.cellsize / m_per_deg_lon

    out_h = max(1, int(np.ceil((north - south) / res_lat)))
    out_w = max(1, int(np.ceil((east - west) / res_lon)))

    # Sample latitudes at equal Web Mercator Y intervals so the bitmap
    # aligns with deck.gl's Mercator-projected basemap.  Longitude is
    # unaffected (Mercator X is linear with longitude).
    def _lat_to_merc_y(lat):
        return np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))

    def _merc_y_to_lat(y):
        return np.degrees(2 * np.arctan(np.exp(y)) - np.pi / 2)

    merc_south = _lat_to_merc_y(south)
    merc_north = _lat_to_merc_y(north)
    merc_ys = np.linspace(merc_south, merc_north, out_h + 1)
    # Pixel centers = midpoints of Mercator Y bins
    merc_centers = (merc_ys[:-1] + merc_ys[1:]) / 2
    out_lats = _merc_y_to_lat(merc_centers)

    out_lons = np.linspace(west + res_lon / 2, east - res_lon / 2, out_w)

    lon_grid, lat_grid = np.meshgrid(out_lons, out_lats)

    # Inverse transform: WGS84 → source CRS
    src_x, src_y = inv.transform(lon_grid.ravel(), lat_grid.ravel())
    src_x = src_x.reshape(out_h, out_w)
    src_y = src_y.reshape(out_h, out_w)

    # Source grid indices (nearest neighbor)
    x_min = metadata.xllcorner
    y_min = metadata.yllcorner
    col_f = (src_x - x_min) / metadata.cellsize - 0.5
    row_f = (src_y - y_min) / metadata.cellsize - 0.5

    col_nn = np.floor(col_f + 0.5).astype(int)
    row_nn = np.floor(row_f + 0.5).astype(int)

    in_bounds = (
        (col_nn >= 0) & (col_nn < metadata.ncols) & (row_nn >= 0) & (row_nn < metadata.nrows)
    )

    out_data = np.full((out_h, out_w), nodata, dtype=data.dtype)
    # Clip indices to valid range (in_bounds already filters, clip is safety)
    r_safe = np.clip(row_nn, 0, data.shape[0] - 1)
    c_safe = np.clip(col_nn, 0, data.shape[1] - 1)
    out_data[in_bounds] = data[r_safe[in_bounds], c_safe[in_bounds]]

    return out_data, bounds


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

    # Reproject to WGS84 pixel space to eliminate projection distortion
    reproj_data, bounds = reproject_grid_to_wgs84(
        data,
        metadata,
        source_crs,
        nodata=nodata,
    )

    # Generate RGBA image from reprojected data and encode as PNG
    rgba = grid_to_rgba_image(reproj_data, scheme, nodata=nodata)
    image_uri = array_to_base64_png(rgba)

    bmp = bitmap_layer(
        f"{layer_id}-bitmap",
        image_uri,
        bounds=bounds,
        opacity=0.85,
        pickable=False,
    )

    # Build tooltip scatter layer with adaptive sampling
    # (uses original grid + per-point transform — no distortion issue)
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
            tooltip_data.append(
                {
                    "position": [lon, lat],
                    "value": round(val, 4),
                }
            )

    scatter = scatterplot_layer(
        f"{layer_id}-tooltip",
        tooltip_data,
        getPosition="@@d.position",
        getRadius=max(100, metadata.cellsize / 2),
        getFillColor=[0, 0, 0, 1],
        stroked=False,
        filled=True,
        radiusMinPixels=3,
        pickable=True,
    )

    return [bmp, scatter]
