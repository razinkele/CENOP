"""
GIS Editor renderers.

Handles loading spatial data layers and rendering them via the
shiny-deckgl MapWidget in the Landscape tab.
"""

import logging
import numpy as np
from shiny import render, ui, reactive
from shiny_deckgl import deck_legend_control, scale_widget

from cenop.ui.tabs.landscape_editor import gis_map
from cenop.server.map_layers import (
    build_grid_bitmap_layer,
    GIS_COLOR_SCHEMES,
    CATEGORICAL_COLORS,
)

logger = logging.getLogger("CENOP")

# Layer name → (attribute on CellData, color scheme, is_monthly)
LAYER_CONFIG = {
    "bathymetry":     ("_depth",          "viridis",     False),
    "dist_to_coast":  ("_dist_to_coast",  "yellow_red",  False),
    "sediment":       ("_sediment",       "categorical", False),
    "prey":           ("_entropy",        "green",       True),
    "salinity":       ("_salinity",       "blue_white",  True),
    "blocks":         ("_blocks",         "categorical", False),
    "food_prob":      ("_food_prob",      "green",       False),
}

LAYER_DISPLAY_NAMES = {
    "bathymetry": "Bathymetry",
    "dist_to_coast": "Distance to Coast",
    "sediment": "Sediment",
    "prey": "Prey (MaxEnt)",
    "salinity": "Salinity",
    "blocks": "Blocks",
    "food_prob": "Food Probability",
}


def register_gis_editor_renderers(input, output, session, state):
    """Register all renderers for the GIS editor tab."""

    # Cache to avoid re-loading CellData on every click
    _gis_cell_data_cache = {"landscape": None, "cell_data": None}

    @render.ui
    def gis_month_control():
        """Show month slider only for monthly layers (Prey, Salinity)."""
        layer = input.gis_layer()
        _, _, is_monthly = LAYER_CONFIG.get(layer, (None, None, False))
        if is_monthly:
            return ui.input_slider("gis_month", "Month", min=1, max=12, value=1, step=1)
        return ui.div()

    @reactive.effect
    @reactive.event(input.gis_load, ignore_none=True)
    async def _gis_load_layer():
        """Load selected layer from CellData and render via MapWidget."""
        layer_key = input.gis_layer()
        if layer_key not in LAYER_CONFIG:
            return

        attr_name, scheme, is_monthly = LAYER_CONFIG[layer_key]
        display_name = LAYER_DISPLAY_NAMES.get(layer_key, layer_key)

        landscape_name = input.landscape()

        # Load CellData (cached per landscape)
        if _gis_cell_data_cache["landscape"] != landscape_name or _gis_cell_data_cache["cell_data"] is None:
            try:
                from cenop.landscape import CellData, create_homogeneous_landscape
                if landscape_name == "Homogeneous":
                    cell_data = create_homogeneous_landscape()
                else:
                    cell_data = CellData(landscape_name)
                    cell_data.load()
                _gis_cell_data_cache["landscape"] = landscape_name
                _gis_cell_data_cache["cell_data"] = cell_data
                logger.info("GIS editor: loaded CellData for '%s'", landscape_name)
            except Exception as e:
                logger.error("GIS editor: failed to load landscape '%s': %s", landscape_name, e)
                ui.notification_show(f"Error loading landscape: {e}", type="error")
                return

        cell_data = _gis_cell_data_cache["cell_data"]
        raw = getattr(cell_data, attr_name, None)
        if raw is None:
            ui.notification_show(f"Layer '{display_name}' not available for this landscape.", type="warning")
            return

        # For monthly layers, select the month slice
        if is_monthly:
            month = input.gis_month() if hasattr(input, 'gis_month') else 1
            try:
                month = int(month)
            except (TypeError, ValueError):
                month = 1
            month_idx = max(0, min(11, month - 1))
            if raw.ndim == 3 and raw.shape[0] >= month_idx + 1:
                data_array = raw[month_idx]
            else:
                data_array = raw
            display_name = f"{display_name} (Month {month})"
        else:
            data_array = raw

        grid_height, grid_width = data_array.shape

        from cenop.ui.sidebar import LANDSCAPE_CRS, LANDSCAPE_BOUNDS
        source_crs = LANDSCAPE_CRS.get(landscape_name, "EPSG:3035")
        meta = cell_data.metadata

        bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
        lat_min, lat_max, lon_min, lon_max = bounds

        # Compute stats
        valid_mask = ~np.isnan(data_array) & (data_array != -9999.0)
        valid_vals = data_array[valid_mask]
        if len(valid_vals) > 0:
            d_min = float(np.min(valid_vals))
            d_max = float(np.max(valid_vals))
        else:
            d_min, d_max = 0.0, 1.0

        # Build unified bitmap + tooltip layers
        layers = build_grid_bitmap_layer(
            f"gis-{layer_key}", data_array, meta, source_crs, scheme,
        )

        # Count tooltip points for stats display
        tooltip_layer = layers[1]
        sampled_count = len(tooltip_layer.get("data", []))

        state.gis_stats.set({
            "min": d_min,
            "max": d_max,
            "mean": float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "std": float(np.std(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "coverage": f"{len(valid_vals)}/{data_array.size}",
            "sampled": sampled_count,
            "layer": display_name,
        })

        # Build legend
        if scheme == "categorical":
            legend_entries = [{"label": display_name, "color": [31, 119, 180], "shape": "rect"}]
        else:
            colors = GIS_COLOR_SCHEMES.get(scheme, GIS_COLOR_SCHEMES["viridis"])
            legend_entries = [
                {"label": f"{d_min:.2f}", "color": colors[0], "shape": "rect"},
                {"label": f"{d_max:.2f}", "color": colors[-1], "shape": "rect"},
            ]

        # Push layers + widgets to map
        await gis_map.update(
            session,
            layers=layers,
            widgets=[scale_widget(placement="bottom-left")],
        )

        # Legend is a MapLibre control — must use set_controls, not update
        await gis_map.set_controls(session, [
            deck_legend_control(
                legend_entries,
                position="bottom-left",
                title=display_name,
            ),
        ])

        # Fly to landscape bounds
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        await gis_map.fly_to(session, longitude=center_lon, latitude=center_lat, zoom=7)

        logger.info("GIS layer '%s': bitmap rendered, %d tooltip points", display_name, sampled_count)

    @render.ui
    def gis_layer_stats():
        """Display statistics for the currently loaded layer."""
        stats = state.gis_stats()
        if not stats:
            return ui.p("No layer loaded yet.", class_="text-muted small")

        return ui.HTML(f'''
        <table class="table table-sm table-borderless" style="font-size:0.8rem;">
            <tr><td class="text-muted">Layer</td><td><b>{stats["layer"]}</b></td></tr>
            <tr><td class="text-muted">Min</td><td>{stats["min"]:.4f}</td></tr>
            <tr><td class="text-muted">Max</td><td>{stats["max"]:.4f}</td></tr>
            <tr><td class="text-muted">Mean</td><td>{stats["mean"]:.4f}</td></tr>
            <tr><td class="text-muted">Std Dev</td><td>{stats["std"]:.4f}</td></tr>
            <tr><td class="text-muted">Coverage</td><td>{stats["coverage"]}</td></tr>
            <tr><td class="text-muted">Displayed</td><td>{stats["sampled"]} pts</td></tr>
        </table>
        ''')
