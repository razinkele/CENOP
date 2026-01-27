"""
Model Settings Tab UI
"""

from shiny import ui


# Tooltip definitions for all parameters
TOOLTIPS = {
    # Basic settings
    "random_seed": "Set a specific random seed for reproducible simulations. Use 0 for automatic random seed each run.",
    "tracked_porpoise_count": "Number of individual porpoises to track in detail for movement analysis and debugging.",
    "ships_enabled": "Enable/disable ship traffic in the simulation. Ships create underwater noise that can disturb porpoises.",
    "bycatch_prob": "Annual probability (0-1) that a porpoise dies from fishing net entanglement. DEPONS default: 0.018 for Kattegat.",
    
    # Movement/CRW parameters
    "param_k": "Inertia constant: controls how much the previous heading influences the next. Higher = straighter paths. DEPONS default: 0.001",
    "param_a0": "Autoregressive parameter for log₁₀(distance/100). Controls step length persistence. DEPONS default: 0.35",
    "param_a1": "Effect of water depth on step length. Positive = longer steps in deeper water. DEPONS default: 0.0005",
    "param_a2": "Effect of salinity on step length. Negative = shorter steps in high salinity. DEPONS default: -0.02",
    "param_b0": "Autoregressive parameter for turning angle. Controls heading persistence. DEPONS default: -0.024",
    "param_b1": "Effect of water depth on turning angle. Negative = less turning in deeper water. DEPONS default: -0.008",
    "param_b2": "Effect of salinity on turning angle. Higher = more turning in high salinity. DEPONS default: 0.93",
    "param_b3": "Intercept for turning angle model. DEPONS default: -14.0",
    
    # Dispersal settings
    "dispersal": "Dispersal behavior type: 'off' = no dispersal, 'PSM-Type2' = memory-based with heading dampening, 'Undirected' = random, 'InnerDanishWaters' = region-specific.",
    "tdisp": "Number of consecutive days of declining energy that triggers dispersal behavior. DEPONS default: 3 days.",
    
    # PSM parameters
    "psm_log": "Logistic increase rate for PSM memory cells. Controls how fast food memories strengthen. DEPONS default: 0.6",
    "psm_dist": "Preferred dispersal distance distribution. Format: N(mean;std) in km. DEPONS default: N(300;100) = mean 300km, std 100km.",
    "psm_tol": "Tolerance distance (km) for reaching dispersal target. Dispersal ends when within this distance. DEPONS default: 5km.",
    "psm_angle": "Maximum turning angle (degrees) during PSM dispersal. Limits heading changes per step. DEPONS default: 20°.",
    
    # Energy parameters
    "param_rS": "Satiation memory decay rate. Higher = faster forgetting of food satisfaction. DEPONS default: 0.04",
    "param_rR": "Reference memory decay rate. Higher = faster forgetting of remembered food locations. DEPONS default: 0.04",
    "param_rU": "Food replenishment rate. How fast depleted food patches recover. DEPONS default: 0.1",

    # JASMINE Mode settings
    "time_mode_override": "Override time subsystem: None = follow main mode, DEPONS = fixed 30-min ticks, JASMINE = variable timesteps with events.",
    "movement_mode_override": "Override movement subsystem: None = follow main mode, DEPONS = empirical CRW, JASMINE = physics-based with velocity.",
    "fsm_mode_override": "Override behavior FSM: None = follow main mode, DEPONS = simple state machine, JASMINE = utility-based decisions.",
    "energy_mode_override": "Override energy subsystem: None = follow main mode, DEPONS = simple tracking, JASMINE = full DEB model.",
    "memory_mode_override": "Override memory subsystem: None = follow main mode, DEPONS = no disturbance memory, JASMINE = learned avoidance.",

    # JASMINE Physics parameters
    "jasmine_mass_kg": "Body mass in kg for physics calculations. Affects inertia and acceleration. Default: 50 kg (adult porpoise).",
    "jasmine_drag_coeff": "Hydrodynamic drag coefficient. Higher = more resistance. Default: 0.01",
    "jasmine_max_thrust": "Maximum thrust force in Newtons. Limits acceleration. Default: 100 N",
    "jasmine_current_weight": "Weight of ocean current advection (0-1). 0 = ignore currents, 1 = fully advected. Default: 0.5",

    # JASMINE DEB parameters
    "jasmine_bmr_scale": "Basal metabolic rate scale factor. 1.0 = standard, >1 = higher base costs. Default: 1.0",
    "jasmine_activity_cost": "Activity cost multiplier. Higher = more energy used during active movement. Default: 2.0",
    "jasmine_disturbance_cost": "Extra energy cost during disturbance. Multiplier on base costs. Default: 1.5",

    # JASMINE Memory parameters
    "jasmine_memory_decay_rate": "Memory decay rate per tick. Lower = longer memory retention. Default: 0.001",
    "jasmine_avoidance_strength": "Maximum learned avoidance strength (0-1). Higher = stronger aversion. Default: 0.8",
    "jasmine_avoidance_radius": "Avoidance influence radius in grid cells. Larger = wider effect. Default: 20 cells",
}


def _tooltip_wrapper(input_element, tooltip_id: str):
    """Wrap an input element with a tooltip icon."""
    tooltip_text = TOOLTIPS.get(tooltip_id, "No description available.")
    return ui.div(
        input_element,
        ui.span(
            "ℹ️",
            title=tooltip_text,
            style="cursor: help; margin-left: 5px; font-size: 0.9em;",
            class_="tooltip-icon"
        ),
        # Add CSS for better tooltip styling
        ui.tags.style("""
            .tooltip-icon {
                position: relative;
            }
            .tooltip-icon:hover::after {
                content: attr(title);
                position: absolute;
                left: 20px;
                top: -5px;
                background: #333;
                color: white;
                padding: 8px 12px;
                border-radius: 6px;
                white-space: normal;
                width: 250px;
                z-index: 1000;
                font-size: 12px;
                line-height: 1.4;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }
        """),
        style="display: flex; align-items: center; margin-bottom: 10px;"
    )


def _input_with_tooltip(input_fn, input_id: str, label: str, **kwargs):
    """Create an input with an integrated tooltip in the label."""
    tooltip_text = TOOLTIPS.get(input_id, "")
    if tooltip_text:
        # Add tooltip icon to label
        label_with_tip = ui.span(
            label,
            ui.span(
                " ⓘ",
                title=tooltip_text,
                style="cursor: help; color: #6c757d; font-size: 0.85em;",
            )
        )
        # For numeric/select inputs, we need to use HTML tags
        return ui.div(
            ui.tags.label(
                label,
                ui.tags.span(
                    " ⓘ",
                    title=tooltip_text,
                    style="cursor: help; color: #0d6efd; font-size: 0.9em;",
                    **{"data-bs-toggle": "tooltip", "data-bs-placement": "right"}
                ),
                **{"for": input_id}
            ),
            input_fn(input_id, None, **kwargs),  # None label since we made our own
            class_="mb-2"
        )
    return input_fn(input_id, label, **kwargs)


def settings_tab():
    """Create the Model Settings tab with all parameter inputs."""
    return ui.nav_panel(
        "Model Settings",
        # Add Bootstrap tooltip initialization script
        ui.tags.script("""
            // Initialize Bootstrap tooltips
            document.addEventListener('DOMContentLoaded', function() {
                var tooltipTriggerList = [].slice.call(document.querySelectorAll('[title]'));
                tooltipTriggerList.forEach(function(el) {
                    el.style.cursor = 'help';
                });
            });
        """),
        ui.navset_card_tab(
            _basic_settings_panel(),
            _movement_settings_panel(),
            _dispersal_settings_panel(),
            _energy_settings_panel(),
            _jasmine_settings_panel()
        )
    )


def _basic_settings_panel():
    """Basic simulation settings - advanced options."""
    return ui.nav_panel(
        "Basic",
        ui.layout_columns(
            ui.card(
                ui.card_header("🎯 Advanced Setup"),
                ui.div(
                    ui.tags.label(
                        "Random Seed (0 = auto) ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["random_seed"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "random_seed"}
                    ),
                    ui.input_numeric("random_seed", None, value=0, min=0),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Tracked Porpoises ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["tracked_porpoise_count"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "tracked_porpoise_count"}
                    ),
                    ui.input_numeric("tracked_porpoise_count", None, value=1, min=0, max=100),
                    class_="mb-3"
                ),
                ui.p("Main simulation settings (population, years, landscape, turbines) are in the left sidebar.", 
                     class_="text-muted small mt-3"),
            ),
            ui.card(
                ui.card_header("⚠️ Disturbance & Threats"),
                ui.p("Wind turbines are selected in the left sidebar (filtered by landscape).", 
                     class_="text-muted small"),
                ui.div(
                    ui.tags.label(
                        "Ship Traffic Enabled ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["ships_enabled"], 
                                     style="cursor: help; color: #0d6efd;"),
                    ),
                    ui.input_switch("ships_enabled", None, value=False),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Annual Bycatch Probability ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["bycatch_prob"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "bycatch_prob"}
                    ),
                    ui.input_numeric("bycatch_prob", None, value=0.0, step=0.001, min=0.0, max=1.0),
                    class_="mb-3"
                ),
            ),
            col_widths=[6, 6]
        )
    )


def _movement_settings_panel():
    """Movement/CRW settings."""
    return ui.nav_panel(
        "Movement",
        ui.card(
            ui.card_header("🧭 Correlated Random Walk (CRW) Parameters"),
            ui.p("These parameters control porpoise movement behavior based on the DEPONS CRW model.", 
                 class_="text-muted mb-3"),
            ui.layout_column_wrap(
                ui.div(
                    ui.tags.label("k - Inertia constant ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_k"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_k"}),
                    ui.input_numeric("param_k", None, value=0.001, step=0.001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("a0 - AutoReg for log₁₀(d/100) ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_a0"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_a0"}),
                    ui.input_numeric("param_a0", None, value=0.35, step=0.01),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("a1 - Water depth effect ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_a1"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_a1"}),
                    ui.input_numeric("param_a1", None, value=0.0005, step=0.0001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("a2 - Salinity effect ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_a2"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_a2"}),
                    ui.input_numeric("param_a2", None, value=-0.02, step=0.01),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("b0 - AutoReg for turning ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_b0"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_b0"}),
                    ui.input_numeric("param_b0", None, value=-0.024, step=0.001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("b1 - Depth on turning ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_b1"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_b1"}),
                    ui.input_numeric("param_b1", None, value=-0.008, step=0.001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("b2 - Salinity on turning ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_b2"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_b2"}),
                    ui.input_numeric("param_b2", None, value=0.93, step=0.01),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("b3 - Intercept ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_b3"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_b3"}),
                    ui.input_numeric("param_b3", None, value=-14.0, step=1.0),
                    class_="mb-2"
                ),
                width=1/4
            )
        )
    )


def _dispersal_settings_panel():
    """Dispersal settings."""
    return ui.nav_panel(
        "Dispersal",
        ui.layout_columns(
            ui.card(
                ui.card_header("🌊 Dispersal Settings"),
                ui.div(
                    ui.tags.label("Dispersal Type ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["dispersal"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "dispersal"}),
                    ui.input_select("dispersal", None, 
                        choices=["off", "PSM-Type2", "Undirected", "InnerDanishWaters"], 
                        selected="PSM-Type2"),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label("Days to Dispersal (tDisp) ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["tdisp"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "tdisp"}),
                    ui.input_numeric("tdisp", None, value=3, min=1),
                    class_="mb-3"
                ),
            ),
            ui.card(
                ui.card_header("📢 PSM Parameters (Persistent Spatial Memory)"),
                ui.div(
                    ui.tags.label("PSM_log - Logistic increase ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["psm_log"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "psm_log"}),
                    ui.input_numeric("psm_log", None, value=0.6, step=0.1),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label("PSM_dist - Preferred distance ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["psm_dist"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "psm_dist"}),
                    ui.input_text("psm_dist", None, value="N(300;100)"),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label("PSM_tol - Tolerance (km) ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["psm_tol"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "psm_tol"}),
                    ui.input_numeric("psm_tol", None, value=5.0, step=0.5),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label("PSM_angle - Max turn (deg) ", 
                                  ui.tags.span("ⓘ", title=TOOLTIPS["psm_angle"], 
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "psm_angle"}),
                    ui.input_numeric("psm_angle", None, value=20.0, step=1.0),
                    class_="mb-3"
                ),
            ),
            col_widths=[6, 6]
        )
    )


def _energy_settings_panel():
    """Energy settings."""
    return ui.nav_panel(
        "Energy",
        ui.card(
            ui.card_header("⚡ Energy & Memory Parameters"),
            ui.p("Memory decay rates and food replenishment control how porpoises remember food locations.",
                 class_="text-muted mb-3"),
            ui.layout_column_wrap(
                ui.div(
                    ui.tags.label("rS - Satiation memory decay ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_rS"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_rS"}),
                    ui.input_numeric("param_rS", None, value=0.04, step=0.01),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("rR - Reference memory decay ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_rR"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_rR"}),
                    ui.input_numeric("param_rR", None, value=0.04, step=0.01),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("rU - Food replenishment rate ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["param_rU"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "param_rU"}),
                    ui.input_numeric("param_rU", None, value=0.1, step=0.01),
                    class_="mb-2"
                ),
                width=1/3
            )
        )
    )


def _jasmine_settings_panel():
    """JASMINE-specific settings panel."""
    return ui.nav_panel(
        "JASMINE",
        ui.div(
            ui.p(
                ui.tags.strong("JASMINE Mode Settings"),
                " - These parameters are only active when JASMINE mode is selected in the sidebar. "
                "JASMINE provides physics-based movement, Dynamic Energy Budget (DEB) modeling, "
                "and learned avoidance behavior.",
                class_="alert alert-info"
            ),
        ),
        ui.layout_columns(
            # Subsystem Mode Overrides
            ui.card(
                ui.card_header("🔧 Subsystem Mode Overrides"),
                ui.p("Override individual subsystems independently of the main simulation mode.",
                     class_="text-muted small mb-3"),
                ui.div(
                    ui.tags.label("Time Mode ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["time_mode_override"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "time_mode_override"}),
                    ui.input_select("time_mode_override", None,
                        choices={"": "Follow main mode", "DEPONS": "DEPONS", "JASMINE": "JASMINE"},
                        selected=""),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Movement Mode ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["movement_mode_override"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "movement_mode_override"}),
                    ui.input_select("movement_mode_override", None,
                        choices={"": "Follow main mode", "DEPONS": "DEPONS", "JASMINE": "JASMINE"},
                        selected=""),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Behavior FSM Mode ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["fsm_mode_override"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "fsm_mode_override"}),
                    ui.input_select("fsm_mode_override", None,
                        choices={"": "Follow main mode", "DEPONS": "DEPONS", "JASMINE": "JASMINE"},
                        selected=""),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Energy Mode ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["energy_mode_override"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "energy_mode_override"}),
                    ui.input_select("energy_mode_override", None,
                        choices={"": "Follow main mode", "DEPONS": "DEPONS", "JASMINE": "JASMINE"},
                        selected=""),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Memory Mode ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["memory_mode_override"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "memory_mode_override"}),
                    ui.input_select("memory_mode_override", None,
                        choices={"": "Follow main mode", "DEPONS": "DEPONS", "JASMINE": "JASMINE"},
                        selected=""),
                    class_="mb-2"
                ),
            ),
            # Physics Parameters
            ui.card(
                ui.card_header("🔬 Physics Parameters"),
                ui.p("Parameters for JASMINE physics-based movement model.",
                     class_="text-muted small mb-3"),
                ui.div(
                    ui.tags.label("Body Mass (kg) ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_mass_kg"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_mass_kg"}),
                    ui.input_numeric("jasmine_mass_kg", None, value=50.0, min=10.0, max=100.0, step=1.0),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Drag Coefficient ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_drag_coeff"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_drag_coeff"}),
                    ui.input_numeric("jasmine_drag_coeff", None, value=0.01, min=0.001, max=0.1, step=0.001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Max Thrust (N) ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_max_thrust"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_max_thrust"}),
                    ui.input_numeric("jasmine_max_thrust", None, value=100.0, min=10.0, max=500.0, step=10.0),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Current Weight (0-1) ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_current_weight"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_current_weight"}),
                    ui.input_numeric("jasmine_current_weight", None, value=0.5, min=0.0, max=1.0, step=0.1),
                    class_="mb-2"
                ),
            ),
            col_widths=[6, 6]
        ),
        ui.layout_columns(
            # DEB Parameters
            ui.card(
                ui.card_header("⚡ DEB Energy Parameters"),
                ui.p("Dynamic Energy Budget parameters for JASMINE mode.",
                     class_="text-muted small mb-3"),
                ui.div(
                    ui.tags.label("BMR Scale Factor ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_bmr_scale"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_bmr_scale"}),
                    ui.input_numeric("jasmine_bmr_scale", None, value=1.0, min=0.5, max=2.0, step=0.1),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Activity Cost Multiplier ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_activity_cost"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_activity_cost"}),
                    ui.input_numeric("jasmine_activity_cost", None, value=2.0, min=1.0, max=5.0, step=0.1),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Disturbance Cost Multiplier ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_disturbance_cost"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_disturbance_cost"}),
                    ui.input_numeric("jasmine_disturbance_cost", None, value=1.5, min=1.0, max=3.0, step=0.1),
                    class_="mb-2"
                ),
            ),
            # Memory Parameters
            ui.card(
                ui.card_header("🧠 Learned Avoidance Parameters"),
                ui.p("Disturbance memory and avoidance behavior parameters.",
                     class_="text-muted small mb-3"),
                ui.div(
                    ui.tags.label("Memory Decay Rate ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_memory_decay_rate"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_memory_decay_rate"}),
                    ui.input_numeric("jasmine_memory_decay_rate", None, value=0.001, min=0.0001, max=0.01, step=0.0001),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Avoidance Strength (0-1) ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_avoidance_strength"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_avoidance_strength"}),
                    ui.input_numeric("jasmine_avoidance_strength", None, value=0.8, min=0.0, max=1.0, step=0.1),
                    class_="mb-2"
                ),
                ui.div(
                    ui.tags.label("Avoidance Radius (cells) ",
                                  ui.tags.span("ⓘ", title=TOOLTIPS["jasmine_avoidance_radius"],
                                               style="cursor: help; color: #0d6efd;"),
                                  **{"for": "jasmine_avoidance_radius"}),
                    ui.input_numeric("jasmine_avoidance_radius", None, value=20.0, min=5.0, max=50.0, step=1.0),
                    class_="mb-2"
                ),
            ),
            col_widths=[6, 6]
        )
    )
