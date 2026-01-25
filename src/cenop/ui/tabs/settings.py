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
    "communication_enabled": "Enable or disable social communication (calls) and cohesion among porpoises.",
    "communication_range_km": "Communication detection range in km (typical porpoise call detection tens of km depending on source and propagation).",
    "communication_source_level": "Porpoise call source level (dB re 1 µPa @1m).",
    "communication_threshold": "Received level at which detection probability is 50% (dB).",
    "communication_response_slope": "Steepness (per dB) for the logistic detection function.",
    "social_weight": "Weight (0-1) of social attraction influence on movement.",
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
            _data_available_panel()
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
                ui.div(
                    ui.tags.label(
                        "Enable Social Communication ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["communication_enabled"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "communication_enabled"}
                    ),
                    ui.input_switch("communication_enabled", None, value=True),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Communication Range (km) ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["communication_range_km"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "communication_range_km"}
                    ),
                    ui.input_numeric("communication_range_km", None, value=10.0, step=1.0, min=0.1),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Social Weight ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["social_weight"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "social_weight"}
                    ),
                    ui.input_slider("social_weight", None, min=0.0, max=1.0, value=0.3, step=0.05),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Call SL (dB) ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["communication_source_level"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "communication_source_level"}
                    ),
                    ui.input_numeric("communication_source_level", None, value=160.0, step=1.0),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Call detect. threshold (dB) ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["communication_threshold"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "communication_threshold"}
                    ),
                    ui.input_numeric("communication_threshold", None, value=120.0, step=1.0),
                    class_="mb-3"
                ),
                ui.div(
                    ui.tags.label(
                        "Call detect. slope ",
                        ui.tags.span("ⓘ", title=TOOLTIPS["communication_response_slope"], 
                                     style="cursor: help; color: #0d6efd;"),
                        **{"for": "communication_response_slope"}
                    ),
                    ui.input_numeric("communication_response_slope", None, value=0.2, step=0.01),
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


def create_preview_pydeck_map():
    """
    Create a static pydeck map for data preview.
    Reuses the dashboard pattern - map is created once, data updated via JavaScript.
    """
    html_content = '''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>
    <style>
        html, body {
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }
        #deck-container {
            width: 100%;
            height: 100%;
            position: absolute;
        }
        .info-box {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 10px 15px;
            border-radius: 8px;
            color: white;
            font-size: 12px;
            font-family: 'Segoe UI', Arial, sans-serif;
            z-index: 1000;
            border: 1px solid #444;
        }
        .info-box h4 {
            margin: 0 0 5px 0;
            font-size: 13px;
            color: #4fc3f7;
        }
        .stat { margin: 3px 0; }
        .label { color: #aaa; }
        .value { font-weight: bold; color: #4fc3f7; }
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div class="info-box">
        <h4 id="file-name">No data loaded</h4>
        <div class="stat"><span class="label">Points:</span> <span class="value" id="point-count">0</span></div>
        <div class="stat"><span class="label">Min:</span> <span class="value" id="data-min">-</span></div>
        <div class="stat"><span class="label">Max:</span> <span class="value" id="data-max">-</span></div>
    </div>
    
    <script>
        const {DeckGL, TileLayer, BitmapLayer, ColumnLayer} = deck;
        
        // EMODnet Bathymetry tiles - clear coastlines
        const BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png';
        
        // Default center (updated when data loads)
        let CENTER_LAT = 54.03;
        let CENTER_LON = 5.98;
        
        // Current preview data
        let previewData = [];
        let dataMin = 0;
        let dataMax = 100;
        let fileName = 'No data';
        
        // Static bathymetry base layer
        const bathymetryLayer = new TileLayer({
            id: 'bathymetry-layer',
            data: BATHYMETRY_URL,
            minZoom: 0,
            maxZoom: 12,
            tileSize: 256,
            renderSubLayers: props => {
                const { bbox: {west, south, east, north} } = props.tile;
                return new BitmapLayer(props, {
                    data: null,
                    image: props.data,
                    bounds: [west, south, east, north]
                });
            }
        });
        
        // Color mapping based on data type
        function getColor(value, min, max, dataType) {
            if (value === null || value === undefined) return [0, 0, 0, 0];
            
            const t = (value - min) / (max - min || 1);
            
            // Bathymetry: Blues (negative depths are deeper)
            if (dataType === 'bathy') {
                if (value > 0) return [139, 90, 43, 180]; // Land (brown)
                const depth = Math.abs(value);
                const depthT = Math.min(1, depth / 50);  // 0-50m range
                const r = Math.floor(26 + (66 - 26) * depthT);
                const g = Math.floor(107 + (165 - 107) * depthT);
                const b = Math.floor(133 + (246 - 133) * depthT);
                return [r, g, b, 180];
            }
            
            // Salinity: viridis-like
            if (dataType === 'sal') {
                let r, g, b;
                if (t < 0.5) {
                    const s = t / 0.5;
                    r = Math.floor(68 + (49 - 68) * s);
                    g = Math.floor(1 + (104 - 1) * s);
                    b = Math.floor(84 + (142 - 84) * s);
                } else {
                    const s = (t - 0.5) / 0.5;
                    r = Math.floor(49 + (253 - 49) * s);
                    g = Math.floor(104 + (231 - 104) * s);
                    b = Math.floor(142 + (37 - 142) * s);
                }
                return [r, g, b, 180];
            }
            
            // Prey/Food: Yellow-Orange-Red
            if (dataType === 'prey') {
                const r = Math.floor(255);
                const g = Math.floor(255 * (1 - t * 0.7));
                const b = Math.floor(50 * (1 - t));
                return [r, g, b, 180];
            }
            
            // Temperature: RdYlBu reversed (blue=cold, red=warm)
            if (dataType === 'temp') {
                let r, g, b;
                if (t < 0.5) {
                    const s = t / 0.5;
                    r = Math.floor(49 + (255 - 49) * s);
                    g = Math.floor(130 + (255 - 130) * s);
                    b = Math.floor(189 + (191 - 189) * s);
                } else {
                    const s = (t - 0.5) / 0.5;
                    r = Math.floor(255);
                    g = Math.floor(255 * (1 - s * 0.6));
                    b = Math.floor(191 * (1 - s * 0.9));
                }
                return [r, g, b, 180];
            }
            
            // Sediment/Default: terrain-like
            let r, g, b;
            if (t < 0.33) {
                const s = t / 0.33;
                r = Math.floor(208 + (140 - 208) * s);
                g = Math.floor(186 + (200 - 186) * s);
                b = Math.floor(145 + (100 - 145) * s);
            } else if (t < 0.67) {
                const s = (t - 0.33) / 0.34;
                r = Math.floor(140 + (90 - 140) * s);
                g = Math.floor(200 + (170 - 200) * s);
                b = Math.floor(100 + (80 - 100) * s);
            } else {
                const s = (t - 0.67) / 0.33;
                r = Math.floor(90 + (50 - 90) * s);
                g = Math.floor(170 + (120 - 170) * s);
                b = Math.floor(80 + (60 - 80) * s);
            }
            return [r, g, b, 180];
        }
        
        // Create data layer
        function createDataLayer(data, dataType) {
            if (!data || data.length === 0) return null;
            
            return new ColumnLayer({
                id: 'preview-data-layer',
                data: data,
                diskResolution: 4,
                radius: 1800,
                extruded: false,
                getPosition: d => d.position,
                getFillColor: d => getColor(d.value, dataMin, dataMax, dataType),
                pickable: true,
                opacity: 0.7
            });
        }
        
        // Build layers
        function buildLayers(dataType) {
            const layers = [bathymetryLayer];
            const dataLayer = createDataLayer(previewData, dataType);
            if (dataLayer) layers.push(dataLayer);
            return layers;
        }
        
        // Initialize deck.gl
        const deckgl = new DeckGL({
            container: 'deck-container',
            initialViewState: {
                latitude: CENTER_LAT,
                longitude: CENTER_LON,
                zoom: 8,
                pitch: 0,
                bearing: 0
            },
            controller: true,
            layers: buildLayers('bathy'),
            parameters: {
                clearColor: [0.05, 0.1, 0.15, 1]
            }
        });
        
        console.log('[IFRAME DEBUG] Deck.gl initialized successfully');
        
        window.deckgl = deckgl;
        
        // Add error handler
        window.addEventListener('error', function(event) {
            console.error('[IFRAME DEBUG] Window error:', event.error || event.message);
        });
        
        window.addEventListener('unhandledrejection', function(event) {
            console.error('[IFRAME DEBUG] Unhandled promise rejection:', event.reason);
        });
        
        // Update data function
        window.setPreviewData = function(data, min, max, name, dataType, centerLat, centerLon) {
            console.log('[IFRAME DEBUG] setPreviewData called:', {
                dataPoints: data?.length || 0,
                min, max, name, dataType, centerLat, centerLon
            });
            
            previewData = data || [];
            dataMin = min || 0;
            dataMax = max || 100;
            fileName = name || 'Unknown';
            CENTER_LAT = centerLat || CENTER_LAT;
            CENTER_LON = centerLon || CENTER_LON;
            
            // Update info box
            document.getElementById('file-name').textContent = fileName;
            document.getElementById('point-count').textContent = previewData.length;
            document.getElementById('data-min').textContent = min.toFixed(2);
            document.getElementById('data-max').textContent = max.toFixed(2);
            
            console.log('[IFRAME DEBUG] Updating deck.gl layers');
            
            // Update layers
            try {
                deckgl.setProps({ 
                    layers: buildLayers(dataType),
                    initialViewState: {
                        latitude: CENTER_LAT,
                        longitude: CENTER_LON,
                        zoom: 8,
                        pitch: 0,
                        bearing: 0
                    }
                });
                console.log('[IFRAME DEBUG] Deck.gl layers updated successfully');
            } catch (error) {
                console.error('[IFRAME DEBUG] Error updating deck.gl:', error);
            }
            
            console.log('[IFRAME DEBUG] Preview data update complete:', previewData.length, 'points');
        };
        
        // Listen for messages
        window.addEventListener('message', function(event) {
            console.log('[IFRAME DEBUG] Message received:', event.data?.type);
            
            if (event.data && event.data.type === 'setPreviewData') {
                console.log('[IFRAME DEBUG] Processing setPreviewData message');
                window.setPreviewData(
                    event.data.data,
                    event.data.min,
                    event.data.max,
                    event.data.name,
                    event.data.dataType,
                    event.data.centerLat,
                    event.data.centerLon
                );
            }
        });
        
        console.log('[IFRAME DEBUG] Preview map iframe initialized successfully');
    </script>
</body>
</html>
'''
    return ui.tags.iframe(
        id="preview-map-frame",
        srcdoc=html_content,
        style="width: 100%; height: 100%; border: none; border-radius: 8px;",
    )


def _data_available_panel():
    """Data Available panel: show files available per landscape in data directory.

    This UI is lightweight and reactive content is rendered server-side into
    `output.data_available_table`. A refresh button triggers re-scan via the
    server render function so that inspecting landscapes does not happen at
    module import time.
    """

    return ui.nav_panel(
        "Data Available",
        ui.div(
            ui.input_action_button("refresh_data_available", "🔄 Refresh", class_="btn-outline-secondary mb-2"),
            ui.output_text("data_available_refreshed", inline=True),
            class_="mb-2"
        ),
        ui.output_ui("data_available_table"),
        ui.hr(),
        ui.card(
            ui.card_header("🗺️ Data Preview"),
            ui.div(
                ui.input_select(
                    "preview_landscape",
                    "Landscape:",
                    choices=["Homogeneous", "CentralBaltic", "DanTysk", "Gemini", "Kattegat", "NorthSea", "UserDefined"],
                    selected="CentralBaltic",
                    width="200px"
                ),
                ui.output_ui("data_preview_controls"),
                style="display: flex; gap: 10px; margin-bottom: 10px;"
            ),
            create_preview_pydeck_map(),
            ui.output_ui("preview_stats_text"),
            # Handler for custom messages from server - cleaner than script injection
            ui.tags.script("""
                $(document).ready(function() {
                    Shiny.addCustomMessageHandler('preview_data_update', function(message) {
                        console.log('[PREVIEW] Received data update via custom message');
                        var iframe = document.getElementById('preview-map-frame');
                        
                        function sendData() {
                            if (iframe && iframe.contentWindow) {
                                iframe.contentWindow.postMessage({
                                    type: 'setPreviewData',
                                    data: message.points,
                                    min: message.min,
                                    max: message.max,
                                    name: message.name,
                                    dataType: message.dataType,
                                    centerLat: message.centerLat,
                                    centerLon: message.centerLon
                                }, '*');
                                console.log('[PREVIEW] Data sent to iframe');
                            } else {
                                // Retry if iframe not ready
                                setTimeout(sendData, 500);
                            }
                        }
                        
                        sendData();
                    });
                });
            """),
            height="700px"
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
