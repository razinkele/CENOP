"""
Landscape GIS Editor Tab UI

Read-only viewer for spatial data layers (bathymetry, prey, salinity, etc.)
using a shiny-deckgl MapWidget.
"""

from shiny import ui
from shiny_deckgl import MapWidget, CARTO_POSITRON, fullscreen_widget

# Module-level widget instance — shared between UI and server
gis_map = MapWidget(
    "gis_map",
    view_state={
        "longitude": 11.0,
        "latitude": 55.0,
        "zoom": 5,
        "pitch": 0,
        "bearing": 0,
    },
    style=CARTO_POSITRON,
    tooltip={
        "html": "<b>{layerType}</b><br/>Value: {value}",
        "style": {"backgroundColor": "#fff", "color": "#333", "fontSize": "12px"},
    },
)


def landscape_editor_tab():
    """Create the Landscape GIS editor tab for viewing spatial layers."""
    return ui.nav_panel(
        "Landscape",
        ui.layout_columns(
            # Left: controls
            ui.card(
                ui.card_header("Layer Controls"),
                ui.input_select(
                    "gis_layer", "Data Layer",
                    choices={
                        "bathymetry": "Bathymetry (Depth)",
                        "dist_to_coast": "Distance to Coast",
                        "sediment": "Sediment Type",
                        "prey": "Prey (MaxEnt)",
                        "salinity": "Salinity",
                        "blocks": "Blocks",
                        "food_prob": "Food Probability",
                    },
                    selected="bathymetry"
                ),
                ui.output_ui("gis_month_control"),
                ui.input_action_button("gis_load", "Load Layer",
                                       class_="btn-primary w-100 mt-2"),
                ui.tags.hr(),
                ui.tags.h6("Layer Statistics"),
                ui.output_ui("gis_layer_stats"),
                height="auto"
            ),
            # Right: map
            ui.card(
                ui.card_header("Spatial Viewer"),
                gis_map.ui(width="100%", height="560px"),
                height="620px"
            ),
            col_widths=[3, 9]
        )
    )
