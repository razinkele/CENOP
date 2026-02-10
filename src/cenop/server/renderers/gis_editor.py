"""
GIS Editor renderers.

Handles loading spatial data layers and sending them to the deck.gl
iframe in the Landscape tab via postMessage.
"""

import json
import logging
import numpy as np
from shiny import render, ui, reactive

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

    @render.ui
    @reactive.event(input.gis_load, ignore_none=True)
    def gis_data_sender():
        """Load selected layer from CellData and send to iframe via postMessage."""
        layer_key = input.gis_layer()
        if layer_key not in LAYER_CONFIG:
            return ui.div()

        attr_name, scheme, is_monthly = LAYER_CONFIG[layer_key]
        display_name = LAYER_DISPLAY_NAMES.get(layer_key, layer_key)

        # Always use the current sidebar selection (not the dashboard-loaded one)
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
                return ui.div()

        cell_data = _gis_cell_data_cache["cell_data"]
        raw = getattr(cell_data, attr_name, None)
        if raw is None:
            ui.notification_show(f"Layer '{display_name}' not available for this landscape.", type="warning")
            return ui.div()

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

        # Adaptive sample to cap ~15k points for performance
        total_cells = grid_height * grid_width
        max_points = 15000
        sample_step = max(1, int((total_cells / max_points) ** 0.5))

        # Get landscape bounds for coordinate conversion
        from cenop.ui.sidebar import LANDSCAPE_BOUNDS
        bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
        lat_min, lat_max, lon_min, lon_max = bounds

        # Build data points
        points = []
        for row in range(0, grid_height, sample_step):
            for col in range(0, grid_width, sample_step):
                val = float(data_array[row, col])
                if np.isnan(val) or val == -9999.0:
                    continue
                lat = lat_min + (row / grid_height) * (lat_max - lat_min)
                lon = lon_min + (col / grid_width) * (lon_max - lon_min)
                points.append({"position": [lon, lat], "value": round(val, 4)})

        if not points:
            ui.notification_show("No valid data found for this layer.", type="warning")
            return ui.div()

        # Compute stats for the full array (not just sampled)
        valid_mask = ~np.isnan(data_array) & (data_array != -9999.0)
        valid_vals = data_array[valid_mask]
        if len(valid_vals) > 0:
            d_min = float(np.min(valid_vals))
            d_max = float(np.max(valid_vals))
        else:
            d_min, d_max = 0.0, 1.0

        # Store stats in reactive state for the stats panel
        state.gis_stats.set({
            "min": d_min,
            "max": d_max,
            "mean": float(np.mean(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "std": float(np.std(valid_vals)) if len(valid_vals) > 0 else 0.0,
            "coverage": f"{len(valid_vals)}/{data_array.size}",
            "sampled": len(points),
            "layer": display_name,
        })

        cell_deg_lat = (lat_max - lat_min) / grid_height * sample_step
        cell_deg_lon = (lon_max - lon_min) / grid_width * sample_step

        data_json = json.dumps(points)

        js_code = f'''
        <script>
            (function() {{
                function sendGISData() {{
                    var iframe = document.getElementById('gis-editor-map-frame');
                    if (iframe && iframe.contentWindow) {{
                        iframe.contentWindow.postMessage({{
                            type: 'setGISBounds',
                            latMin: {lat_min}, latMax: {lat_max},
                            lonMin: {lon_min}, lonMax: {lon_max}
                        }}, '*');
                        iframe.contentWindow.postMessage({{
                            type: 'setGISLayerData',
                            data: {data_json},
                            scheme: '{scheme}',
                            dataMin: {d_min},
                            dataMax: {d_max},
                            cellDegLat: {cell_deg_lat},
                            cellDegLon: {cell_deg_lon},
                            layerName: '{display_name}'
                        }}, '*');
                        console.log('GIS data sent:', {len(points)}, 'points');
                    }} else {{
                        setTimeout(sendGISData, 100);
                    }}
                }}
                setTimeout(sendGISData, 300);
            }})();
        </script>
        '''
        return ui.HTML(js_code)

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
