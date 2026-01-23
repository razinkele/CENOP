"""
Sidebar component for CENOP Shiny app.
CENOP - CETacean Noise-Population Model
"""

from shiny import ui


# Landscape-turbine compatibility mapping
# Note: Only landscapes with available bathymetry data are included
# Kattegat, InnerDanishWaters, DanTysk, Gemini data files are not available in this distribution
LANDSCAPE_TURBINE_COMPATIBILITY = {
    "Homogeneous": ["off"],  # Synthetic uniform landscape for testing
    "NorthSea": ["off", "NorthSea_scenario1", "NorthSea_scenario2", "NorthSea_scenario3"],  # Uses UserDefined data
    "UserDefined": ["off", "User-def"],  # DEPONS default 400x400 North Sea grid
    "CentralBaltic": ["off"],  # Central Baltic Sea (Lithuania to Sweden) - 400x400 at 1km resolution
}

# Geographic bounds for each landscape (lat_min, lat_max, lon_min, lon_max)
# Used for map centering and coordinate transformations
LANDSCAPE_BOUNDS = {
    "Homogeneous": (53.27, 54.79, 4.83, 7.13),  # Default North Sea bounds
    "NorthSea": (53.27, 54.79, 4.83, 7.13),     # DEPONS North Sea area
    "UserDefined": (53.27, 54.79, 4.83, 7.13),  # Same as NorthSea
    "CentralBaltic": (53.9, 59.5, 13.0, 22.0),  # Central Baltic Sea - 450x460 grid (covers Oder mouth)
}

# Tooltips for sidebar parameters
SIDEBAR_TOOLTIPS = {
    "porpoise_count": "Initial number of porpoises at simulation start. DEPONS typically uses 1000-5000 for realistic population dynamics.",
    "sim_years": "Duration of the simulation in years. Each year = 360 days = 17,280 ticks (30-min steps).",
    "landscape": "Geographic area for the simulation. Each landscape has specific bathymetry, food availability, and compatible turbine scenarios.",
    "turbines": "Wind turbine scenario to simulate. Construction scenarios include pile-driving noise that deters porpoises.",
    "sim_speed": "Controls simulation speed. 1% = slow (0.3s per day), 100% = maximum speed (no delay between steps).",
}


def create_sidebar():
    """Create the simulation control sidebar with setup controls."""
    return ui.sidebar(
        ui.h5("🎮 Simulation Control", class_="mb-3"),
        
        # Simulation Setup
        ui.div(
            ui.div(
                ui.tags.label(
                    "Initial Population ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["porpoise_count"], 
                                 style="cursor: help; color: #0d6efd;"),
                    **{"for": "porpoise_count"}
                ),
                ui.input_numeric("porpoise_count", None, value=1000, min=1, max=50000, step=1),
                class_="mb-2"
            ),
            ui.div(
                ui.tags.label(
                    "Simulation Years ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["sim_years"], 
                                 style="cursor: help; color: #0d6efd;"),
                    **{"for": "sim_years"}
                ),
                ui.input_numeric("sim_years", None, value=5, min=1, max=100),
                class_="mb-2"
            ),
            ui.div(
                ui.tags.label(
                    "Landscape ",
                    ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["landscape"], 
                                 style="cursor: help; color: #0d6efd;"),
                    **{"for": "landscape"}
                ),
                ui.input_select("landscape", None, 
                    choices=["Homogeneous", "NorthSea", "CentralBaltic", "UserDefined"], 
                    selected="NorthSea"),
                class_="mb-2"
            ),
            ui.input_action_button("load_landscape", "🗺️ Load Landscape", class_="btn-outline-secondary w-100 mt-1 mb-1"),
            ui.output_text("landscape_status"),
            # Turbine scenario - dynamically filtered based on landscape
            ui.output_ui("turbine_selector"),
            ui.input_action_button("load_turbines", "🌬️ Load Turbines", class_="btn-outline-secondary w-100 mt-1 mb-1"),
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
            ui.input_action_button("run_sim", "▶ Run Simulation", class_="btn-primary w-100 mb-2"),
            ui.input_action_button("stop_sim", "⏹ Stop", class_="btn-danger w-100 mb-2"),
            ui.input_action_button("reset_sim", "🔄 Reset", class_="btn-secondary w-100"),
            class_="mb-3"
        ),
        
        # Speed control with tooltip
        ui.div(
            ui.tags.label(
                "⚡ Simulation Speed ",
                ui.tags.span("ⓘ", title=SIDEBAR_TOOLTIPS["sim_speed"], 
                             style="cursor: help; color: #0d6efd;"),
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
        
        ui.tags.hr(),
        
        ui.p("Advanced parameters in 'Model Settings' tab.", class_="text-muted small"),
        
        width=280,
        bg="#f8f9fa"
    )

