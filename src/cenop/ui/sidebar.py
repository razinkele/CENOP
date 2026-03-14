"""
Sidebar component for CENOP Shiny app.
CENOP - CETacean Noise-Population Model
"""

from shiny import ui


# Landscape-turbine compatibility mapping with display labels
# Note: Only landscapes with available bathymetry data are included
# Format: {landscape: {turbine_key: display_label}} — used by Shiny input_select
LANDSCAPE_TURBINE_COMPATIBILITY: dict[str, dict[str, str]] = {
    "Homogeneous": {"off": "No turbines"},
    "CentralBaltic": {"off": "No turbines"},
    "Kattegat": {"off": "No turbines"},
    "Lithuania": {
        "off": "No turbines",
        "CuronianNord_35_15MW": "Curonian Nord 35x15 MW",
        "CuronianNord_60_10MW": "Curonian Nord 60x10 MW",
    },
    "NorthSea": {
        "off": "No turbines",
        "NorthSea_scenario1": "Scenario 1",
        "NorthSea_scenario2": "Scenario 2",
        "NorthSea_scenario3": "Scenario 3",
    },
}

# Geographic bounds for each landscape (lat_min, lat_max, lon_min, lon_max)
# Used for map centering only (NOT for coordinate transforms — use grid_to_lonlat)
LANDSCAPE_BOUNDS = {
    "Homogeneous": (53.27, 54.79, 4.83, 7.13),  # Default North Sea bounds
    "CentralBaltic": (51.2, 55.3, 16.2, 23.7),  # Central Baltic - actual grid extent
    "Kattegat": (53.90, 57.41, 9.45, 13.49),    # Kattegat / Inner Danish Waters - 600x1000 @ 400m
    "Lithuania": (54.20, 57.28, 17.49, 21.70),   # Lithuanian EEZ expanded - 215x375 grid at 1km
    "NorthSea": (50.62, 59.06, -1.95, 9.89),    # DEPONS North Sea - 2088x2175 @ 400m
}

# Source CRS for each landscape's ASC grid files
# Most DEPONS landscapes use EPSG:3035 (ETRS89-LAEA), Kattegat uses EPSG:25832 (ETRS89/UTM 32N)
LANDSCAPE_CRS = {
    "Homogeneous": "EPSG:3035",
    "CentralBaltic": "EPSG:3035",
    "Kattegat": "EPSG:25832",
    "Lithuania": "EPSG:3035",
    "NorthSea": "EPSG:3035",
    "Gemini": "EPSG:3035",
    "DanTysk": "EPSG:3035",
}

# Tooltips for sidebar parameters
SIDEBAR_TOOLTIPS = {
    "porpoise_count": "Initial number of porpoises at simulation start. DEPONS typically uses 1000-5000 for realistic population dynamics.",
    "sim_years": "Duration of the simulation in years. Each year = 360 days = 17,280 ticks (30-min steps).",
    "landscape": "Geographic area for the simulation. Each landscape has specific bathymetry, food availability, and compatible turbine scenarios.",
    "turbines": "Wind turbine scenario to simulate. Construction scenarios include pile-driving noise that deters porpoises.",
    "sim_speed": "Controls simulation speed. 1% = slow (0.3s per day), 100% = maximum speed (no delay between steps).",
    "simulation_mode": "DEPONS = regulatory-compatible empirical models (validated against DEPONS 3.0). JASMINE = research-grade physics and DEB models with learned behaviors.",
}


def create_sidebar():
    """Create the simulation control sidebar with setup controls."""
    return ui.sidebar(
        ui.h5("Simulation Control", class_="mb-3"),
        
        # Simulation Setup
        ui.div(
            ui.div(
                ui.tags.label(
                    "Initial Population ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["porpoise_count"], 
                                 style="cursor: help; color: #0d7377;"),
                    **{"for": "porpoise_count"}
                ),
                ui.input_numeric("porpoise_count", None, value=100, min=1, max=50000, step=1),
                class_="mb-2"
            ),
            ui.div(
                ui.tags.label(
                    "Simulation Years ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["sim_years"],
                                 style="cursor: help; color: #0d7377;"),
                    **{"for": "sim_years"}
                ),
                ui.input_numeric("sim_years", None, value=5, min=1, max=100),
                class_="mb-2"
            ),
            ui.div(
                ui.tags.label(
                    "Simulation Mode ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["simulation_mode"],
                                 style="cursor: help; color: #0d7377;"),
                    **{"for": "simulation_mode"}
                ),
                ui.input_select("simulation_mode", None,
                    choices={"DEPONS": "DEPONS (Regulatory)", "JASMINE": "JASMINE (Research)"},
                    selected="DEPONS"),
                class_="mb-2"
            ),
            ui.div(
                ui.tags.label(
                    "Landscape ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["landscape"],
                                 style="cursor: help; color: #0d7377;"),
                    **{"for": "landscape"}
                ),
                ui.input_select("landscape", None,
                    choices=["Homogeneous", "Lithuania", "CentralBaltic", "Kattegat", "NorthSea"],
                    selected="Lithuania"),
                class_="mb-2"
            ),
            ui.input_action_button("load_landscape", "Load Landscape", class_="btn-outline-secondary w-100 mt-1 mb-1"),
            ui.output_text("landscape_status"),
            # Turbine scenario - dynamically filtered based on landscape
            ui.output_ui("turbine_selector"),
            ui.input_action_button("load_turbines", "Load Turbines", class_="btn-outline-secondary w-100 mt-1 mb-1"),
            ui.output_text("turbine_status"),
            class_="mb-3"
        ),
        
        ui.tags.hr(),
        
        # Progress section
        ui.div(
            ui.output_ui("progress_bar"),
            ui.output_text("progress_text", inline=True),
            class_="mb-3"
        ),
        
        # Run controls
        ui.div(
            ui.input_action_button("run_sim", "Run Simulation", class_="btn-primary w-100 mb-2"),
            ui.input_action_button("stop_sim", "Stop", class_="btn-danger w-100 mb-2"),
            ui.input_action_button("reset_sim", "Reset", class_="btn-secondary w-100"),
            class_="mb-3"
        ),
        
        # Speed control with tooltip
        ui.div(
            ui.tags.label(
                "Simulation Speed ",
                ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["sim_speed"], 
                             style="cursor: help; color: #0d7377;"),
            ),
            ui.input_slider(
                "sim_speed", 
                None,
                min=1, 
                max=100, 
                value=50,
                step=1,
                post=" %"
            ),
            ui.p("1% = slowest, 100% = fastest", class_="text-muted small mb-0"),
            class_="mb-3"
        ),

        # Skip visualization for faster headless runs
        ui.div(
            ui.input_checkbox(
                "skip_viz", "Skip visualization (fast run)", value=False
            ),
            class_="mb-3",
        ),

        # Porpoise trace controls
        ui.div(
            ui.input_checkbox(
                "show_traces", "Show porpoise traces", value=False
            ),
            ui.panel_conditional(
                "input.show_traces",
                ui.input_slider(
                    "trace_length_days",
                    "Trace history (days)",
                    min=1,
                    max=7,
                    value=2,
                    step=1,
                ),
            ),
            class_="mb-3",
        ),

        ui.tags.hr(),

        # Social Communication controls
        ui.accordion(
            ui.accordion_panel(
                "Social Communication",
                ui.input_checkbox("communication_enabled", "Enable Social Calls", value=False),
                ui.input_numeric("communication_range_km", "Detection Range (km)", value=1.0, min=0.1, max=50.0, step=0.5),
                ui.input_numeric("communication_source_level", "Source Level (dB)", value=130.0, min=80.0, max=200.0, step=1.0),
                ui.input_numeric("communication_threshold", "Detection Threshold (dB)", value=80.0, min=40.0, max=160.0, step=1.0),
                ui.input_numeric("communication_response_slope", "Response Slope", value=0.1, min=0.01, max=1.0, step=0.01),
                ui.input_slider("social_weight", "Social Weight", min=0.0, max=1.0, value=0.3, step=0.05),
            ),
            id="social_comm_accordion",
            open=False,
        ),

        ui.tags.hr(),

        ui.p("Advanced parameters in 'Model Settings' tab. JASMINE-specific settings available when JASMINE mode is selected.", class_="text-muted small"),
        
        width=280,
        bg="#e8eff5"
    )

