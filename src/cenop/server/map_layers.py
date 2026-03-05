"""Map layer builders for the shiny-deckgl MapWidget.

Each function returns a layer dict compatible with MapWidget.update().
"""

from shiny_deckgl import icon_layer, heatmap_layer, scatterplot_layer


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


def build_depth_heatmap(depth_points: list) -> dict:
    """Build depth heatmap layer from [lon, lat, depth] triples."""
    if not depth_points:
        return heatmap_layer("depth-heatmap", [], visible=False)

    return heatmap_layer(
        "depth-heatmap",
        depth_points,
        getPosition="@@d",
        getWeight="@@=d[2] || 1",
        radiusPixels=40,
        intensity=1,
        threshold=0.03,
        colorRange=[
            [1, 31, 75, 200],
            [3, 56, 108, 200],
            [15, 94, 156, 180],
            [46, 134, 193, 160],
            [86, 180, 233, 140],
            [166, 216, 247, 120],
        ],
    )


def build_foraging_heatmap(foraging_points: list) -> dict:
    """Build foraging/food heatmap from [lon, lat, food_prob] triples."""
    if not foraging_points:
        return heatmap_layer("foraging-heatmap", [], visible=False)

    return heatmap_layer(
        "foraging-heatmap",
        foraging_points,
        getPosition="@@d",
        getWeight="@@=d[2] || 1",
        radiusPixels=30,
        intensity=1.2,
        threshold=0.05,
        colorRange=[
            [8, 48, 20, 100],
            [20, 100, 40, 140],
            [40, 160, 60, 160],
            [80, 200, 80, 180],
            [140, 230, 100, 200],
            [200, 255, 140, 220],
        ],
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


def build_turbine_blade_layer(turbine_data: list, rotation: float = 0) -> dict:
    """Build turbine blade icon layer with rotation angle."""
    if not turbine_data:
        return icon_layer("turbine-blades", [], visible=False)

    return icon_layer(
        "turbine-blades",
        turbine_data,
        iconAtlas=TURBINE_BLADE_ATLAS,
        iconMapping=TURBINE_BLADE_MAPPING,
        getIcon="blade",
        getPosition="@@=d.position",
        getSize="@@=Math.max(20, Math.min(64, (d.radius || 300) / 15))",
        getAngle=f"@@=d.phase === 'operational' ? {rotation} : 0",
        getColor="@@=d.phase === 'construction' ? [255,70,40,220] : d.phase === 'operational' ? [50,160,240,220] : d.phase === 'planned' ? [180,180,180,180] : d.color || [255,140,60]",
        pickable=False,
        opacity=0.95,
    )
