"""Layer legend for the dashboard map.

The legend is shiny_deckgl's own ``layer_legend_widget`` (a deck.gl widget). This module holds
what is specific to CENOP: which layers are listed, in what order and with what swatch, which
layers follow another one's checkbox, and how a user's choice is kept across layer updates.

The checkbox toggles the layer in the browser and reports
``{"layer_id": ..., "visible": ...}`` through ``input[widget.legend_visibility_input_id]``; the
server keeps that choice and applies it to every layer it sends afterwards, because the next
update would otherwise undo it.
"""

from __future__ import annotations

# The basemap is not a deck.gl layer. An empty layer with this id stands in for it, so that
# the legend has something to tick; the server swaps the map style when it is toggled.
BASEMAP_ID = "basemap"
BLANK_STYLE = {"version": 8, "sources": {}, "layers": []}

# The depth layer is drawn with map_layers.GIS_COLOR_SCHEMES["viridis"] (shallow pale yellow
# to deep purple); the legend must use the same stops (tests/test_turbine_noise_radius.py)
_BATHYMETRY_RAMP = [
    [255, 255, 217],
    [253, 231, 37],
    [94, 201, 98],
    [33, 145, 140],
    [59, 82, 139],
    [68, 1, 84],
]
_FORAGING_RAMP = [
    [8, 48, 20],
    [20, 100, 40],
    [40, 160, 60],
    [80, 200, 80],
    [140, 230, 100],
    [200, 255, 140],
]

_SPECS: dict[str, dict] = {
    BASEMAP_ID: {"label": "Background map", "shape": "rect", "color": [176, 190, 197]},
    "depth-bitmap": {"label": "Bathymetry", "shape": "gradient", "colors": _BATHYMETRY_RAMP},
    "foraging-bitmap": {"label": "Foraging", "shape": "gradient", "colors": _FORAGING_RAMP},
    "noise-construction": {
        "label": "Construction noise",
        "shape": "circle",
        "color": [255, 60, 60, 160],
    },
    "noise-operational": {
        "label": "Operational noise",
        "shape": "circle",
        "color": [255, 200, 60, 120],
    },
    "turbine-poles": {"label": "Wind turbines", "shape": "rect", "color": [50, 160, 240]},
    "porpoises": {"label": "Porpoises", "shape": "circle", "color": [0, 150, 255]},
    "porpoise-trails": {"label": "Porpoise traces", "shape": "line", "color": [0, 150, 255]},
}

#: Legend order, top to bottom.
LEGEND_ORDER: tuple[str, ...] = tuple(_SPECS)

#: Layers without a legend row that follow another layer's checkbox.
COMPANIONS: dict[str, tuple[str, ...]] = {
    "depth-bitmap": ("depth-tooltip",),
    "foraging-bitmap": ("foraging-tooltip",),
    "turbine-poles": ("turbine-blades",),
}


def legend_entries(loaded: set[str]) -> list[dict]:
    """Entries for ``layer_legend_widget``: the basemap, then the loaded layers in order."""
    return [
        {"layer_id": layer_id, **_SPECS[layer_id]}
        for layer_id in LEGEND_ORDER
        if layer_id == BASEMAP_ID or layer_id in loaded
    ]


def toggled_ids(layer_id: str) -> list[str]:
    """The layer a checkbox names, followed by the layers that follow it."""
    return [layer_id, *COMPANIONS.get(layer_id, ())]


def active_choice(choice: dict[str, bool], loaded: set[str]) -> dict[str, bool]:
    """The part of the user's choice that applies now: the basemap and the loaded layers."""
    live = {BASEMAP_ID}
    for layer_id in loaded:
        live.update(toggled_ids(layer_id))
    return {layer_id: visible for layer_id, visible in choice.items() if layer_id in live}


def with_choice(layer: dict, choice: dict[str, bool]) -> dict:
    """``layer`` with the user's visibility choice applied; the input is not changed."""
    layer_id = layer.get("id")
    if layer_id not in choice or layer.get("visible", True) == choice[layer_id]:
        return layer
    return {**layer, "visible": choice[layer_id]}
