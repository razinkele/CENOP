"""
Dashboard Tab UI
"""

from shiny import ui


def create_static_pydeck_map():
    """
    Create a static pydeck map that updates via JavaScript messaging.
    Following the DEPONS-master pattern where the map is created once
    and only the scatter overlay is updated via deckgl.setProps().
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
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        #deck-container {
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }
        .info-panel {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 10px;
            border-radius: 10px;
            color: white;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            max-width: 180px;
            transition: all 0.3s ease;
            cursor: move;
        }
        .info-panel.collapsed {
            padding: 8px 12px;
            max-width: 50px;
        }
        .info-panel.collapsed .panel-content {
            display: none;
        }
        .info-panel h4 {
            margin: 0 0 8px 0;
            font-size: 13px;
            color: #4fc3f7;
            border-bottom: 1px solid #444;
            padding-bottom: 6px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .collapse-btn {
            background: none;
            border: none;
            color: #4fc3f7;
            cursor: pointer;
            font-size: 14px;
            padding: 0 4px;
        }
        .collapse-btn:hover {
            color: #fff;
        }
        .stat-row {
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
        }
        .stat-label { color: #aaa; }
        .stat-value { font-weight: bold; color: #4fc3f7; }
        .legend {
            position: absolute;
            bottom: 20px;
            left: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 10px 12px;
            border-radius: 8px;
            color: white;
            font-size: 10px;
            z-index: 1000;
            border: 1px solid #444;
            transition: all 0.3s ease;
            cursor: move;
        }
        .legend.collapsed {
            padding: 6px 10px;
        }
        .legend.collapsed .legend-content {
            display: none;
        }
        .legend-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 6px;
            font-size: 11px;
            color: #4fc3f7;
            font-weight: bold;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 3px 0;
        }
        .legend-dot {
            width: 12px;
            height: 12px;
            margin-right: 6px;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.3);
        }
        .view-info {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 6px 10px;
            border-radius: 5px;
            color: #ccc;
            font-family: monospace;
            font-size: 10px;
            z-index: 1000;
        }

        /* Spinner animation for operational turbine legend — rotates around the hub */
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
        .turbine-spinner { animation: spin 1s linear infinite; transform-origin: 24px 14px; transform-box: view-box; }
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div class="info-panel" id="info-panel">
        <h4>
            <span>🐬 Layers</span>
            <button class="collapse-btn" onclick="togglePanel('info-panel')">−</button>
        </h4>
        <div class="panel-content">
            <div class="stat-row">
                <span class="stat-label">Visible:</span>
                <span class="stat-value" id="point-count">0</span>
            </div>
            <div class="stat-row">
                <span class="stat-label">Depth:</span>
                <label class="switch" style="margin-left: 5px;">
                    <input type="checkbox" id="depth-toggle">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="stat-row">
                <span class="stat-label">Turbines:</span>
                <label class="switch" style="margin-left: 5px;">
                    <input type="checkbox" id="turbine-toggle">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="stat-row">
                <span class="stat-label">Noise:</span>
                <label class="switch" style="margin-left: 5px;">
                    <input type="checkbox" id="noise-toggle">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="stat-row">
                <span class="stat-label">Foraging:</span>
                <label class="switch" style="margin-left: 5px;">
                    <input type="checkbox" id="foraging-toggle">
                    <span class="slider"></span>
                </label>
            </div>
            <div class="stat-row">
                <span class="stat-label">Ships:</span>
                <label class="switch" style="margin-left: 5px;">
                    <input type="checkbox" id="ships-toggle">
                    <span class="slider"></span>
                </label>
            </div>
        </div>
    </div>
    <div class="legend" id="legend-panel" id="legend-panel">
        <div class="legend-header">
            <span>Legend</span>
            <button class="collapse-btn" onclick="togglePanel('legend-panel')">−</button>
        </div>
        <div class="legend-content">
            <div class="legend-item">
                <svg width="16" height="16" viewBox="0 0 32 32" style="margin-right:8px;">
                    <path fill="#2ecc71" d="M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15 C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2 C12.8 19 13.2 22 14 24.5 C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8 C15.2 26.5 15.6 27 16 27.2 C16.4 27 16.8 26.5 17.2 25.8 C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5 C18.8 22 19.2 19 19.5 16.2 C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15 C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>
                </svg>
                <span>Juvenile (&lt;2 yr)</span>
            </div>
            <div class="legend-item">
                <svg width="16" height="16" viewBox="0 0 32 32" style="margin-right:8px;">
                    <path fill="#3498db" d="M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15 C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2 C12.8 19 13.2 22 14 24.5 C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8 C15.2 26.5 15.6 27 16 27.2 C16.4 27 16.8 26.5 17.2 25.8 C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5 C18.8 22 19.2 19 19.5 16.2 C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15 C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>
                </svg>
                <span>Young Adult (2-6 yr)</span>
            </div>
            <div class="legend-item">
                <svg width="16" height="16" viewBox="0 0 32 32" style="margin-right:8px;">
                    <path fill="#2980b9" d="M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15 C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2 C12.8 19 13.2 22 14 24.5 C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8 C15.2 26.5 15.6 27 16 27.2 C16.4 27 16.8 26.5 17.2 25.8 C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5 C18.8 22 19.2 19 19.5 16.2 C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15 C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>
                </svg>
                <span>Mature (6-12 yr)</span>
            </div>
            <div class="legend-item">
                <svg width="16" height="16" viewBox="0 0 32 32" style="margin-right:8px;">
                    <path fill="#95a5a6" d="M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15 C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2 C12.8 19 13.2 22 14 24.5 C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8 C15.2 26.5 15.6 27 16 27.2 C16.4 27 16.8 26.5 17.2 25.8 C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5 C18.8 22 19.2 19 19.5 16.2 C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15 C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>
                </svg>
                <span>Older (&gt;12 yr)</span>
            </div>
            <div class="legend-item">
                <svg width="16" height="16" viewBox="0 0 32 32" style="margin-right:8px;">
                    <path fill="#ff3232" d="M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15 C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2 C12.8 19 13.2 22 14 24.5 C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8 C15.2 26.5 15.6 27 16 27.2 C16.4 27 16.8 26.5 17.2 25.8 C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5 C18.8 22 19.2 19 19.5 16.2 C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15 C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>
                </svg>
                <span>Disturbed</span>
            </div>
            <div class="legend-item" style="margin-top:6px;padding-top:6px;border-top:1px solid #444;">
                <div style="width:16px;height:8px;margin-right:8px;background:linear-gradient(to right, rgba(52,152,219,0.5), rgba(52,152,219,1));border-radius:2px;"></div>
                <span style="font-size:9px;color:#888;">Opacity = Energy</span>
            </div>
            <div class="legend-item" style="align-items:flex-start;">
                <div style="display:flex;flex-direction:column;margin-right:8px;gap:4px;">
                    <!-- Construction (static) -->
                    <svg width="18" height="18" viewBox="0 0 48 48"><g fill="#ff4630" stroke="#000" stroke-width="0.6"><rect x="22" y="14" width="4" height="32"/><circle cx="24" cy="14" r="3"/><path d="M22 14 L24 1 L26 14 Z"/><path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/><path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/></g></svg>
                    <!-- Operational (only blades spin) -->
                    <svg width="18" height="18" viewBox="0 0 48 48"><g stroke="#000" stroke-width="0.6"><rect x="22" y="14" width="4" height="32" fill="#32b0f0"/><circle cx="24" cy="14" r="3" fill="#32b0f0"/><g class="turbine-spinner" fill="#32b0f0"><path d="M22 14 L24 1 L26 14 Z"/><path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/><path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/></g></g></svg>
                    <!-- Planned (static) -->
                    <svg width="18" height="18" viewBox="0 0 48 48"><g fill="#b0b0b0" stroke="#000" stroke-width="0.6"><rect x="22" y="14" width="4" height="32"/><circle cx="24" cy="14" r="3"/><path d="M22 14 L24 1 L26 14 Z"/><path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/><path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/></g></svg>
                </div>
                <div>
                    <div style="font-weight:bold;margin-bottom:4px;">Turbines (<span id="turbine-count">0</span>)</div>
                    <div style="font-size:11px;color:#ccc;">
                        <span style="display:inline-block;width:10px;height:10px;background:#ff4630;margin-right:6px;border-radius:2px;border:1px solid rgba(0,0,0,0.2);"></span> Construction<br>
                        <span style="display:inline-block;width:10px;height:10px;background:#32b0f0;margin-right:6px;border-radius:2px;border:1px solid rgba(0,0,0,0.2);"></span> Operational<br>
                        <span style="display:inline-block;width:10px;height:10px;background:#b0b0b0;margin-right:6px;border-radius:2px;border:1px solid rgba(0,0,0,0.2);"></span> Planned
                    </div>
                </div>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: linear-gradient(to right, #1a237e, #0288d1, #4fc3f7);"></div>
                <span>Depth</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgb(255, 30, 30);"></div>
                <span>Construction Noise</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgb(255, 180, 50);"></div>
                <span>Operational Noise</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: linear-gradient(to right, rgba(0,200,0,0.2), rgba(0,255,0,0.8));"></div>
                <span>Foraging</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgb(128, 0, 180);"></div>
                <span>Ship (<span id="ship-count">0</span>)</span>
            </div>
        </div>
    </div>
    <div class="view-info" id="view-info">Loading...</div>
    
    <style>
        .switch {
            position: relative;
            display: inline-block;
            width: 32px;
            height: 16px;
        }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider {
            position: absolute;
            cursor: pointer;
            top: 0; left: 0; right: 0; bottom: 0;
            background-color: #555;
            transition: .3s;
            border-radius: 16px;
        }
        .slider:before {
            position: absolute;
            content: "";
            height: 12px;
            width: 12px;
            left: 2px;
            bottom: 2px;
            background-color: white;
            transition: .3s;
            border-radius: 50%;
        }
        input:checked + .slider { background-color: #4fc3f7; }
        input:checked + .slider:before { transform: translateX(16px); }
    </style>
    
    <script>
        const {DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, ColumnLayer, IconLayer, SolidPolygonLayer} = deck;
        
        // EMODnet Bathymetry tiles - STATIC base layer
        const BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png';
        
        // DEPONS landscape extent in lat/lon (calculated from EPSG:3035 bounds)
        // Actual bounds from bathy.asc: XLLCORNER=3976618, YLLCORNER=3363923, 400x400 cells @ 400m
        // These can be updated dynamically for different landscapes
        let LAT_MIN = 53.27, LAT_MAX = 54.79;
        let LON_MIN = 4.83, LON_MAX = 7.13;
        
        // North Sea center (center of DEPONS area) - updated dynamically
        let CENTER_LAT = 54.03;  // Center of DEPONS area
        let CENTER_LON = 5.98;   // Center of DEPONS area
        
        // Grid dimensions (will be updated from depth data)
        let GRID_WIDTH = 400;
        let GRID_HEIGHT = 400;
        let DEPTH_RADIUS = 1800;  // Render radius for depth cells (metres)
        let CELL_DEG_LAT = 0.005;  // Cell extent in degrees (for square grid)
        let CELL_DEG_LON = 0.005;

        // Current data
        let porpoiseData = [];
        let depthData = [];
        let turbineData = [];
        let noiseData = [];  // Noise propagation contours
        let foragingData = [];  // Food availability/foraging patches
        let FORAGING_RADIUS = 1800;  // Render radius for foraging cells (metres)
        let shipData = [];  // Ship positions
        let showDepthLayer = false;  // Off until loaded
        // Show turbine layer by default so turbines are visible on load
        let showTurbineLayer = true;
        // Rotor animation state for operational turbines
        let turbineRotation = 0;
        let turbineAnimationRunning = false;
        let turbineAnimationFrameId = null;
        let showNoiseLayer = false;
        let showForagingLayer = false;
        let showShipLayer = false;
        
        // Noise propagation parameters (from DEPONS)
        const NOISE_THRESHOLD = 158.0;  // RT: deterrence threshold in dB
        const BETA_HAT = 20.0;  // Spreading loss factor
        const ALPHA_HAT = 0.0;  // Absorption coefficient
        
        // Static bathymetry layer - created once, never recreated
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
        
        // Color scale for depth - HIGH CONTRAST viridis-like
        function getDepthColor(depth, minDepth, maxDepth) {
            if (depth <= 0) return [139, 90, 43, 180]; // Brown for land
            // Normalize 0-1 (shallow to deep)
            const t = Math.max(0, Math.min(1, (depth - minDepth) / (maxDepth - minDepth)));
            // Viridis-inspired: yellow (shallow) -> cyan -> blue -> purple (deep)
            let r, g, b;
            if (t < 0.25) {
                // Yellow to green
                const s = t / 0.25;
                r = Math.floor(255 * (1 - s) + 32 * s);
                g = Math.floor(255 * (1 - s * 0.3));
                b = Math.floor(100 * s);
            } else if (t < 0.5) {
                // Green to cyan  
                const s = (t - 0.25) / 0.25;
                r = Math.floor(32 * (1 - s));
                g = Math.floor(178 + 77 * s);
                b = Math.floor(100 + 155 * s);
            } else if (t < 0.75) {
                // Cyan to blue
                const s = (t - 0.5) / 0.25;
                r = Math.floor(30 * s);
                g = Math.floor(255 * (1 - s) + 100 * s);
                b = 255;
            } else {
                // Blue to purple (deep)
                const s = (t - 0.75) / 0.25;
                r = Math.floor(30 + 90 * s);
                g = Math.floor(100 * (1 - s) + 20 * s);
                b = Math.floor(255 * (1 - s * 0.3));
            }
            return [r, g, b, 160];
        }
        
        // Create depth overlay layer — square grid cells via SolidPolygonLayer
        function createDepthLayer(data) {
            if (!data || data.length === 0 || !showDepthLayer) {
                return null;
            }

            // Find depth range for coloring (only water cells)
            const waterDepths = data.filter(d => d.depth > 0).map(d => d.depth);
            const minDepth = Math.min(...waterDepths) || 0;
            const maxDepth = Math.max(...waterDepths) || 50;

            const halfLon = CELL_DEG_LON * 0.5;
            const halfLat = CELL_DEG_LAT * 0.5;

            return new SolidPolygonLayer({
                id: 'depth-layer',
                data: data,
                getPolygon: d => {
                    const [lon, lat] = d.position;
                    return [[lon-halfLon, lat-halfLat], [lon+halfLon, lat-halfLat],
                            [lon+halfLon, lat+halfLat], [lon-halfLon, lat+halfLat]];
                },
                getFillColor: d => getDepthColor(d.depth, minDepth, maxDepth),
                pickable: false,
                extruded: false,
            });
        }
        
        // Create PORPOISE icon layer — harbour porpoise dorsal-view silhouette (pointing up/north)
        // Torpedo body, small pectoral fins, dorsal fin bump, horizontal tail fluke
        const PORPOISE_SVG = 'data:image/svg+xml;utf8,' + encodeURIComponent(
            '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
            // Body: streamlined torpedo, widest at ~1/3 from nose
            + '<path fill="white" d="'
            + 'M16 1 C14.8 3 13.5 6 13 9 C12.6 11 12.5 13 12.5 15'
            // Left pectoral fin
            + ' C12 15.5 10 16.5 9 17.5 C10 17.5 11.5 17 12.5 16.2'
            // Body continues tapering to peduncle
            + ' C12.8 19 13.2 22 14 24.5'
            // Left tail fluke
            + ' C13 26 11 28 9.5 29 C11 29 13.5 27.5 14.8 25.8'
            + ' C15.2 26.5 15.6 27 16 27.2'
            // Right tail fluke
            + ' C16.4 27 16.8 26.5 17.2 25.8'
            + ' C18.5 27.5 21 29 22.5 29 C21 28 19 26 18 24.5'
            // Right body back up
            + ' C18.8 22 19.2 19 19.5 16.2'
            + ' C20.5 17 22 17.5 23 17.5 C22 16.5 20 15.5 19.5 15'
            // Right body to nose
            + ' C19.5 13 19.4 11 19 9 C18.5 6 17.2 3 16 1Z"/>'
            // Dorsal fin (small bump on upper-right of body)
            + '<path fill="white" d="M18.5 10 C19 9 20.5 8 21 8.5 C20 9.5 19 11 18.8 12Z"/>'
            + '</svg>'
        );

        function createScatterLayer(data) {
            if (!data || data.length === 0) return null;

            return new IconLayer({
                id: 'porpoise-layer',
                data: data,
                iconAtlas: PORPOISE_SVG,
                iconMapping: {
                    'porpoise': { x: 0, y: 0, width: 32, height: 32, anchorY: 16, anchorX: 16 }
                },
                getIcon: d => 'porpoise',
                getPosition: d => d.position,
                // Size varies by age: juveniles smaller, adults larger
                getSize: d => {
                    const age = d.age || 5;
                    if (age < 1) return 14;      // Calves - smallest
                    if (age < 2) return 18;      // Juveniles
                    if (age < 6) return 22;      // Young adults
                    if (age < 15) return 24;     // Prime adults
                    return 20;                   // Older - slightly smaller
                },
                sizeScale: 1,
                sizeMinPixels: 8,
                sizeMaxPixels: 32,
                // Heading: 0 = North, 90 = East, etc. (matches DEPONS convention)
                getAngle: d => -(d.heading || 0),  // Negate for deck.gl rotation direction
                getColor: d => {
                    // Priority: disturbed > age-based (colors match legend)
                    if (d.is_disturbed) return [255, 50, 50, 255];  // #ff3232 red

                    const age = d.age || 5;
                    const energy = d.energy || 10;
                    const energyAlpha = Math.floor(200 + 55 * Math.min(1, energy / 15));

                    if (age < 2) return [46, 204, 113, energyAlpha];   // #2ecc71 emerald green - juveniles
                    if (age < 6) return [52, 152, 219, energyAlpha];   // #3498db bright blue - young adults
                    if (age < 12) return [41, 128, 185, energyAlpha];  // #2980b9 darker blue - mature
                    return [149, 165, 166, energyAlpha];               // #95a5a6 gray - older
                },
                pickable: true,
                billboard: false,  // Keep flat on map
                alphaCutoff: 0.05,
                // Smooth transitions
                transitions: {
                    getPosition: 300,
                    getSize: 200,
                    getAngle: 250,
                    getColor: 200
                }
            });
        }
        
        // Turbine pole+hub layer (static) - poles and hubs do not rotate
        function createTurbinePoleLayer(data) {
            if (!data || data.length === 0 || !showTurbineLayer) return null;
            return new IconLayer({
                id: 'turbine-pole-layer',
                data: data,
                // Pole+hub icon: hub at (24,14), pole extends downward. Anchor at hub.
                iconAtlas: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><g fill="white"><rect x="22" y="14" width="4" height="32"/><circle cx="24" cy="14" r="3"/></g></svg>',
                iconMapping: {
                    'pole': { x: 0, y: 0, width: 48, height: 48, anchorY: 14, anchorX: 24 }
                },
                getIcon: d => 'pole',
                getPosition: d => d.position,
                getSize: d => Math.max(20, Math.min(64, (d.radius || 300) / 15)),
                getColor: d => {
                    if (d.phase === 'construction') return [255, 70, 48, 220];
                    if (d.phase === 'operational') return [50, 176, 240, 220];
                    if (d.phase === 'planned') return [176, 176, 176, 180];
                    return d.color || [255, 140, 60];
                },
                pickable: true,
                opacity: 0.95
            });
        }

        // Turbine blades layer (rotating around the hub)
        function createTurbineBladeLayer(data) {
            if (!data || data.length === 0 || !showTurbineLayer) return null;
            return new IconLayer({
                id: 'turbine-blade-layer',
                data: data,
                // Three blades radiating from hub at (24,14). Anchor at hub for correct rotation.
                iconAtlas: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><g fill="white"><path d="M22 14 L24 1 L26 14 Z"/><path d="M25 15.7 L12.7 20.5 L23 12.3 Z"/><path d="M25 12.3 L35.3 20.5 L23 15.7 Z"/></g></svg>',
                iconMapping: {
                    'blade': { x: 0, y: 0, width: 48, height: 48, anchorY: 14, anchorX: 24 }
                },
                getIcon: d => 'blade',
                getPosition: d => d.position,
                getSize: d => Math.max(20, Math.min(64, (d.radius || 300) / 15)),
                getAngle: d => (d.phase === 'operational') ? turbineRotation : 0,
                getColor: d => {
                    // Blades follow same color mapping for visibility
                    if (d.phase === 'construction') return [255, 70, 48, 220];
                    if (d.phase === 'operational') return [50, 176, 240, 220];
                    if (d.phase === 'planned') return [176, 176, 176, 180];
                    return d.color || [255, 140, 60];
                },
                pickable: false,
                opacity: 0.95,
                billboard: true
            });
        }
        
        // Create noise propagation layers (construction + operational)
        // data is now {construction: [], operational: []} or legacy array format
        function createNoiseLayers(data) {
            const layers = [];
            if (!showNoiseLayer) return layers;
            
            // Handle both new format {construction, operational} and legacy array format
            let constructionData = [];
            let operationalData = [];
            
            if (Array.isArray(data)) {
                // Legacy format - treat all as construction noise
                constructionData = data;
            } else if (data && typeof data === 'object') {
                constructionData = data.construction || [];
                operationalData = data.operational || [];
            }
            
            // Construction noise layer (RED - high impact pile-driving, 234 dB)
            if (constructionData.length > 0) {
                console.log('Creating construction noise layer with', constructionData.length, 'turbines');
                layers.push(new ScatterplotLayer({
                    id: 'noise-construction-layer',
                    data: constructionData,
                    getPosition: d => d.position,
                    getRadius: d => d.radius || 6000,
                    getFillColor: [255, 30, 30, 100],
                    getLineColor: [255, 50, 50, 180],
                    lineWidthMinPixels: 1,
                    stroked: true,
                    filled: true,
                    pickable: false,
                    opacity: 0.6,
                    radiusMinPixels: 8,
                    radiusMaxPixels: 200
                }));
            }
            
            // Operational noise layer (YELLOW/ORANGE - low impact, ~145 dB)
            if (operationalData.length > 0) {
                console.log('Creating operational noise layer with', operationalData.length, 'points');
                layers.push(new ScatterplotLayer({
                    id: 'noise-operational-layer',
                    data: operationalData,
                    getPosition: d => d.position,
                    getRadius: d => d.radius || 1800,
                    getFillColor: [255, 180, 50, 60],
                    getLineColor: [255, 200, 80, 120],
                    lineWidthMinPixels: 1,
                    stroked: true,
                    filled: true,
                    pickable: false,
                    opacity: 0.4,
                    radiusMinPixels: 3,
                    radiusMaxPixels: 60
                }));
            }
            
            return layers;
        }
        
        // Create foraging layer (shows food availability / patches)
        function createForagingLayer(data) {
            if (!data || data.length === 0 || !showForagingLayer) return null;
            
            console.log('Creating foraging layer with', data.length, 'points');
            
            return new ScatterplotLayer({
                id: 'foraging-layer',
                data: data,
                getPosition: d => d.position,
                getRadius: FORAGING_RADIUS,
                getFillColor: d => {
                    // Green gradient based on food probability (0-1)
                    // Higher food = brighter green
                    const foodProb = d.food || 0;
                    if (foodProb <= 0.1) return [0, 0, 0, 0];  // Skip very low food areas
                    const t = Math.min(1, foodProb);
                    const alpha = Math.floor(40 + 150 * t);  // 40-190 alpha
                    const green = Math.floor(150 + 105 * t);  // 150-255 green
                    return [0, green, 50, alpha];
                },
                pickable: false,
                opacity: 0.5,
                radiusMinPixels: 3,
                radiusMaxPixels: 30
            });
        }


        
        // Create ship layer (shows vessel traffic)
        function createShipLayer(data) {
            if (!data || data.length === 0 || !showShipLayer) return null;
            
            console.log('Creating ship layer with', data.length, 'ships');
            
            return new ScatterplotLayer({
                id: 'ship-layer',
                data: data,
                getPosition: d => d.position,
                getRadius: d => 1500 + (d.size || 0) * 200,  // Size based on vessel class
                getFillColor: d => {
                    // Purple for ships - distinct from porpoises (blue)
                    // Opacity varies by speed - faster = more visible
                    const speed = d.speed || 1;
                    const alpha = Math.floor(150 + 80 * Math.min(1, speed / 20));
                    return [128, 0, 180, alpha];
                },
                getLineColor: [255, 255, 255, 200],
                getLineWidth: 2,
                stroked: true,
                pickable: true,
                opacity: 0.9,
                radiusMinPixels: 6,
                radiusMaxPixels: 20
            });
        }
        
        // Build layers array
        function buildLayers() {
            const layers = [bathymetryLayer];
            const depthLayer = createDepthLayer(depthData);
            if (depthLayer) layers.push(depthLayer);
            const foragingLayer = createForagingLayer(foragingData);
            if (foragingLayer) layers.push(foragingLayer);
            // Noise layers (construction + operational)
            const noiseLayers = createNoiseLayers(noiseData);
            layers.push(...noiseLayers);
            const poleLayer = createTurbinePoleLayer(turbineData);
            if (poleLayer) layers.push(poleLayer);
            const bladeLayer = createTurbineBladeLayer(turbineData);
            if (bladeLayer) layers.push(bladeLayer);
            const shipLayer = createShipLayer(shipData);
            if (shipLayer) layers.push(shipLayer);
            const porpoiseLayer = createScatterLayer(porpoiseData);
            if (porpoiseLayer) layers.push(porpoiseLayer);
            return layers;
        }
        
        // Initialize deck.gl ONCE
        const deckgl = new DeckGL({
            container: 'deck-container',
            initialViewState: {
                latitude: CENTER_LAT,
                longitude: CENTER_LON,
                zoom: 8,  // Higher zoom for smaller DEPONS area
                pitch: 0,
                bearing: 0
            },
            controller: true,
            layers: buildLayers(),
            parameters: {
                clearColor: [0.05, 0.1, 0.15, 1]
            },
            onViewStateChange: ({viewState}) => {
                document.getElementById('view-info').innerHTML = 
                    'Lat: ' + viewState.latitude.toFixed(3) + 
                    ' | Lon: ' + viewState.longitude.toFixed(3) + 
                    ' | Zoom: ' + viewState.zoom.toFixed(1);
            }
        });
        
        // Store global reference for updates (DEPONS pattern)
        window.deckgl = deckgl;
        window.bathymetryLayer = bathymetryLayer;
        
        // Toggle depth layer
        document.getElementById('depth-toggle').addEventListener('change', function(e) {
            e.stopPropagation();
            showDepthLayer = e.target.checked;
            deckgl.setProps({ layers: buildLayers() });
        });
        
        // Toggle turbine layer
        document.getElementById('turbine-toggle').addEventListener('change', function(e) {
            e.stopPropagation();
            showTurbineLayer = e.target.checked;
            deckgl.setProps({ layers: buildLayers() });
        });
        
        // Toggle noise layer
        document.getElementById('noise-toggle').addEventListener('change', function(e) {
            e.stopPropagation();
            showNoiseLayer = e.target.checked;
            deckgl.setProps({ layers: buildLayers() });
        });
        
        // Toggle foraging layer
        document.getElementById('foraging-toggle').addEventListener('change', function(e) {
            e.stopPropagation();
            showForagingLayer = e.target.checked;
            deckgl.setProps({ layers: buildLayers() });
        });
        
        // Toggle ship layer
        document.getElementById('ships-toggle').addEventListener('change', function(e) {
            e.stopPropagation();
            showShipLayer = e.target.checked;
            deckgl.setProps({ layers: buildLayers() });
        });
        
        // Update function - only updates scatter layer via setProps
        window.updatePorpoiseData = function(newData) {
            porpoiseData = newData || [];
            console.log('Porpoise data loaded (first 10):', porpoiseData.slice(0,10));
            // Use setProps to update ONLY the layers, not the whole map
            deckgl.setProps({ layers: buildLayers() });
            document.getElementById('point-count').textContent = porpoiseData.length;
        };
        
        // Set depth data (called once at startup)
        window.setDepthData = function(data, gridWidth, gridHeight, radius, cellDegLat, cellDegLon) {
            GRID_WIDTH = gridWidth || 400;
            GRID_HEIGHT = gridHeight || 400;
            DEPTH_RADIUS = radius || 1800;
            CELL_DEG_LAT = cellDegLat || 0.005;
            CELL_DEG_LON = cellDegLon || 0.005;
            depthData = data;
            // Auto-enable depth layer so users see the actual simulation grid
            if (!showDepthLayer && data && data.length > 0) {
                showDepthLayer = true;
                var toggle = document.getElementById('depth-toggle');
                if (toggle) toggle.checked = true;
            }
            deckgl.setProps({ layers: buildLayers() });
            console.log('Depth layer loaded:', data.length, 'cells, cellDeg:', CELL_DEG_LAT.toFixed(4), CELL_DEG_LON.toFixed(4));
        };
        
        // Set turbine data (called when turbine scenario is selected)
        window.setTurbineData = function(data) {
            turbineData = data || [];
            deckgl.setProps({ layers: buildLayers() });
            console.log('Turbine layer loaded:', turbineData.length, 'turbines');
            // Update turbine count if element exists
            const el = document.getElementById('turbine-count');
            if (el) el.textContent = turbineData.length;

            // Start rotor animation if any turbine is operational
            const hasOperational = turbineData.some(d => d.phase === 'operational');
            if (hasOperational && !turbineAnimationRunning) {
                turbineAnimationRunning = true;
                function animate() {
                    turbineRotation = (turbineRotation + 4) % 360; // rotation speed
                    deckgl.setProps({ layers: buildLayers() });
                    turbineAnimationFrameId = requestAnimationFrame(animate);
                }
                turbineAnimationFrameId = requestAnimationFrame(animate);
            } else if (!hasOperational && turbineAnimationRunning) {
                turbineAnimationRunning = false;
                if (turbineAnimationFrameId) {
                    cancelAnimationFrame(turbineAnimationFrameId);
                    turbineAnimationFrameId = null;
                }
                turbineRotation = 0;
                deckgl.setProps({ layers: buildLayers() });
            }
        }; 
        
        // Set noise propagation data (calculated from turbines)
        window.setNoiseData = function(data) {
            noiseData = data || [];
            // Auto-enable noise layer when data arrives
            const hasNoise = (Array.isArray(data) && data.length > 0)
                || (data && (data.construction && data.construction.length > 0 || data.operational && data.operational.length > 0));
            if (hasNoise && !showNoiseLayer) {
                showNoiseLayer = true;
                var toggle = document.getElementById('noise-toggle');
                if (toggle) toggle.checked = true;
            }
            deckgl.setProps({ layers: buildLayers() });
            const nc = Array.isArray(data) ? data.length : ((data && data.construction ? data.construction.length : 0) + (data && data.operational ? data.operational.length : 0));
            console.log('Noise layer loaded:', nc, 'sources');
        };
        
        // Set foraging data (food probability / patches)
        window.setForagingData = function(data, radius) {
            foragingData = data || [];
            FORAGING_RADIUS = radius || 1800;
            deckgl.setProps({ layers: buildLayers() });
            console.log('Foraging layer loaded:', foragingData.length, 'food cells, radius:', FORAGING_RADIUS);
        };
        
        // Set ship data (vessel traffic positions)
        window.setShipData = function(data) {
            shipData = data || [];
            deckgl.setProps({ layers: buildLayers() });
            console.log('Ship layer loaded:', shipData.length, 'vessels');
            // Update ship count if element exists
            const el = document.getElementById('ship-count');
            if (el) el.textContent = shipData.length;
        };
        
        // Listen for messages from parent (Shiny)
        window.addEventListener('message', function(event) {
            if (event.data && event.data.type === 'updatePorpoises') {
                window.updatePorpoiseData(event.data.data);
            }
            if (event.data && event.data.type === 'setDepthData') {
                window.setDepthData(event.data.data, event.data.gridWidth, event.data.gridHeight, event.data.radius, event.data.cellDegLat, event.data.cellDegLon);
            }
            if (event.data && event.data.type === 'setTurbineData') {
                window.setTurbineData(event.data.data);
            }
            if (event.data && event.data.type === 'setNoiseData') {
                window.setNoiseData(event.data.data);
            }
            if (event.data && event.data.type === 'setForagingData') {
                window.setForagingData(event.data.data, event.data.radius);
            }
            if (event.data && event.data.type === 'setShipData') {
                window.setShipData(event.data.data);
            }
            if (event.data && event.data.type === 'setLandscapeBounds') {
                // Update bounds for different landscapes
                LAT_MIN = event.data.latMin;
                LAT_MAX = event.data.latMax;
                LON_MIN = event.data.lonMin;
                LON_MAX = event.data.lonMax;
                CENTER_LAT = (LAT_MIN + LAT_MAX) / 2;
                CENTER_LON = (LON_MIN + LON_MAX) / 2;
                console.log('Landscape bounds updated:', {LAT_MIN, LAT_MAX, LON_MIN, LON_MAX});
                // Re-center the map
                if (deckgl) {
                    deckgl.setProps({
                        initialViewState: {
                            longitude: CENTER_LON,
                            latitude: CENTER_LAT,
                            zoom: 6,
                            pitch: 0,
                            bearing: 0
                        }
                    });
                }
            }
        });
        
        // Initial view info
        document.getElementById('view-info').innerHTML = 
            'Lat: ' + CENTER_LAT.toFixed(3) + 
            ' | Lon: ' + CENTER_LON.toFixed(3) + 
            ' | Zoom: 6.0';
        
        // Toggle collapse for panels
        function togglePanel(panelId) {
            const panel = document.getElementById(panelId);
            const btn = panel.querySelector('.collapse-btn');
            panel.classList.toggle('collapsed');
            btn.textContent = panel.classList.contains('collapsed') ? '+' : '−';
        }
        window.togglePanel = togglePanel;
        
        // Make panels draggable
        function makeDraggable(elmnt) {
            let pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
            elmnt.onmousedown = dragMouseDown;
            
            function dragMouseDown(e) {
                if (e.target.tagName === 'INPUT' || e.target.tagName === 'BUTTON') return;
                e.preventDefault();
                pos3 = e.clientX;
                pos4 = e.clientY;
                document.onmouseup = closeDragElement;
                document.onmousemove = elementDrag;
            }
            
            function elementDrag(e) {
                e.preventDefault();
                pos1 = pos3 - e.clientX;
                pos2 = pos4 - e.clientY;
                pos3 = e.clientX;
                pos4 = e.clientY;
                elmnt.style.top = (elmnt.offsetTop - pos2) + 'px';
                elmnt.style.left = (elmnt.offsetLeft - pos1) + 'px';
                elmnt.style.right = 'auto';
                elmnt.style.bottom = 'auto';
            }
            
            function closeDragElement() {
                document.onmouseup = null;
                document.onmousemove = null;
            }
        }
        
        // Initialize draggable panels
        makeDraggable(document.getElementById('info-panel'));
        makeDraggable(document.getElementById('legend-panel'));
    </script>
</body>
</html>
'''
    return ui.tags.iframe(
        id="porpoise-map-frame",
        srcdoc=html_content,
        style="width: 100%; height: 620px; min-height: 500px; border: none; border-radius: 8px;",
    )


def dashboard_tab():
    """Create the Dashboard tab with value boxes and main visualizations."""
    return ui.nav_panel(
        "Dashboard",
        # Compact stat row styling
        ui.tags.style("""
            .stat-row { display: flex; gap: 6px; margin-bottom: 6px; }
            .stat-chip {
                flex: 1; text-align: center; padding: 2px 8px;
                border-radius: 4px; font-size: 0.75rem; line-height: 1.3;
                color: #fff;
            }
            .stat-chip .stat-label { font-weight: 400; opacity: 0.85; }
            .stat-chip .stat-val { font-weight: 700; font-size: 0.85rem; }
            .stat-pop { background: #0d6efd; }
            .stat-year { background: #0dcaf0; color: #000; }
            .stat-birth { background: #198754; }
            .stat-death { background: #fd7e14; }
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
        # Main content: map on left, charts on right
        ui.layout_columns(
            # Left: Large map
            ui.card(
                ui.card_header("Spatial Distribution"),
                create_static_pydeck_map(),
                ui.output_ui("depth_data_initializer"),
                ui.output_ui("foraging_data_initializer"),
                ui.output_ui("ship_data_initializer"),
                ui.output_ui("turbine_data_initializer"),
                ui.output_ui("turbine_data_updater"),
                ui.output_ui("noise_data_initializer"),
                ui.output_ui("porpoise_data_updater"),
                height="660px"
            ),
            # Right: 3 SVG charts stacked
            ui.div(
                ui.card(
                    ui.output_ui("population_plot"),
                    height="190px"
                ),
                ui.card(
                    ui.output_ui("life_death_plot"),
                    height="190px"
                ),
                ui.card(
                    ui.output_ui("energy_balance_plot"),
                    height="190px"
                ),
            ),
            col_widths=[7, 5]
        )
    )
