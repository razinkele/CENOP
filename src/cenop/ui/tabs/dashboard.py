"""
Dashboard Tab UI
"""

from shiny import ui
from shiny_deckgl import (
    MapWidget, CARTO_POSITRON,
    zoom_widget, compass_widget, scale_widget, fullscreen_widget,
    legend_control,
)

# Legend entries for the layer control (matches deck.gl layer IDs in server)
LEGEND_ENTRIES = [
    {
        "layer_id": "depth-bitmap",
        "label": "Bathymetry",
        "colors": [
            [1, 31, 75], [3, 56, 108], [15, 94, 156],
            [46, 134, 193], [86, 180, 233], [166, 216, 247],
        ],
        "shape": "rect",
    },
    {
        "layer_id": "foraging-bitmap",
        "label": "Foraging",
        "colors": [
            [8, 48, 20], [20, 100, 40], [40, 160, 60],
            [80, 200, 80], [140, 230, 100], [200, 255, 140],
        ],
        "shape": "rect",
    },
    {
        "layer_id": "noise-construction",
        "label": "Construction noise",
        "color": [255, 60, 60, 160],
        "shape": "circle",
    },
    {
        "layer_id": "noise-operational",
        "label": "Operational noise",
        "color": [255, 200, 60, 120],
        "shape": "circle",
    },
    {
        "layer_id": "turbine-poles",
        "label": "Wind turbines",
        "color": [50, 160, 240],
        "shape": "rect",
    },
    {
        "layer_id": "porpoises",
        "label": "Porpoises",
        "color": [0, 150, 255],
        "shape": "circle",
    },
]

# Module-level widget instance — shared between UI and server
sim_map = MapWidget(
    "sim_map",
    view_state={
        "longitude": 21.1,
        "latitude": 55.7,
        "zoom": 6,
        "pitch": 0,
        "bearing": 0,
    },
    style=CARTO_POSITRON,
    tooltip={
        "html": "<b>{layerType}</b><br/>{info}",
        "style": {"backgroundColor": "#fff", "color": "#333", "fontSize": "12px"},
    },
    # Only legend control as initial MapLibre control (no default nav — deck.gl widgets handle that)
    controls=[
        legend_control(
            {entry["layer_id"]: entry["label"] for entry in LEGEND_ENTRIES},
            position="bottom-right",
            show_checkbox=True,
            show_default=True,
            title="Layers",
        ),
    ],
)


def dashboard_tab():
    """Create the Dashboard tab with full-width map and collapsible chart panel."""
    return ui.nav_panel(
        "Dashboard",
        # Compact stat row styling (using ocean theme)
        ui.tags.style("""
            .stat-row { display: flex; gap: 6px; margin-bottom: 6px; }
            .stat-chip {
                flex: 1; text-align: center; padding: 2px 8px;
                border-radius: 6px; font-size: 0.75rem; line-height: 1.3;
                color: #fff; font-weight: 500;
                box-shadow: 0 2px 6px rgba(0,0,0,0.15);
            }
            .stat-chip .stat-label { font-weight: 400; opacity: 0.85; }
            .stat-chip .stat-val { font-weight: 700; font-size: 0.85rem; }
        """),
        # Top row: 4 compact stat chips
        ui.div(
            ui.div(
                ui.span("Population ", class_="stat-label"),
                ui.span(ui.output_text("current_population", inline=True), class_="stat-val"),
                class_="stat-chip stat-pop"
            ),
            ui.div(
                ui.span("Year ", class_="stat-label"),
                ui.span(ui.output_text("current_year", inline=True), class_="stat-val"),
                class_="stat-chip stat-year"
            ),
            ui.div(
                ui.span("Births ", class_="stat-label"),
                ui.span(ui.output_text("total_births", inline=True), class_="stat-val"),
                class_="stat-chip stat-birth"
            ),
            ui.div(
                ui.span("Deaths ", class_="stat-label"),
                ui.span(ui.output_text("total_deaths", inline=True), class_="stat-val"),
                class_="stat-chip stat-death"
            ),
            class_="stat-row"
        ),
        # Full-width map card (dark zone) — now using shiny-deckgl MapWidget
        ui.card(
            ui.card_header(
                ui.div(
                    "Spatial Distribution",
                    ui.input_switch(
                        "blade_animation", "Animate blades",
                        value=True,
                    ),
                    style="display: flex; justify-content: space-between; align-items: center; width: 100%;",
                ),
            ),
            sim_map.ui(width="100%", height="620px"),
            height="calc(100vh - 280px)",
            class_="ocean-card"
        ),
        # Collapsible chart toggle bar
        ui.div(
            ui.HTML('<span id="chart-toggle-icon">▼</span> Charts'),
            id="chart-toggle",
            class_="chart-toggle-bar",
            onclick="toggleChartPanel()"
        ),
        # Collapsible chart panel: 3 charts in a row
        ui.div(
            ui.layout_columns(
                ui.card(
                    ui.output_ui("population_plot"),
                    height="190px",
                    class_="ocean-card"
                ),
                ui.card(
                    ui.output_ui("life_death_plot"),
                    height="190px",
                    class_="ocean-card"
                ),
                ui.card(
                    ui.output_ui("energy_balance_plot"),
                    height="190px",
                    class_="ocean-card"
                ),
                col_widths=[4, 4, 4]
            ),
            id="chart-panel",
            class_="chart-panel"
        ),
        # Toggle script
        ui.tags.script("""
            function toggleChartPanel() {
                var panel = document.getElementById('chart-panel');
                var icon = document.getElementById('chart-toggle-icon');
                panel.classList.toggle('collapsed');
                icon.textContent = panel.classList.contains('collapsed') ? '▶' : '▼';
            }
        """),
    )
