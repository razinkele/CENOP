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
                <div class="legend-dot" style="background: rgb(0, 150, 255);"></div>
                <span>Porpoise</span>
            </div>
            <div class="legend-item">
                <div class="legend-dot" style="background: rgb(255, 100, 50);"></div>
                <span>Turbine (<span id="turbine-count">0</span>)</span>
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
        const {DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, ColumnLayer} = deck;
        
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
        
        // Current data
        let porpoiseData = [];
        let depthData = [];
        let turbineData = [];
        let noiseData = [];  // Noise propagation contours
        let foragingData = [];  // Food availability/foraging patches
        let shipData = [];  // Ship positions
        let showDepthLayer = false;  // Off until loaded
        let showTurbineLayer = false;
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
        
        // Create depth overlay layer
        function createDepthLayer(data) {
            if (!data || data.length === 0 || !showDepthLayer) {
                return null;
            }
            
            // Calculate cell size in degrees
            const cellWidth = (LON_MAX - LON_MIN) / GRID_WIDTH;
            const cellHeight = (LAT_MAX - LAT_MIN) / GRID_HEIGHT;
            
            // Find depth range for coloring (only water cells)
            const waterDepths = data.filter(d => d.depth > 0).map(d => d.depth);
            const minDepth = Math.min(...waterDepths) || 0;
            const maxDepth = Math.max(...waterDepths) || 50;
            
            return new ColumnLayer({
                id: 'depth-layer',
                data: data,
                diskResolution: 4,
                radius: 1800,  // meters - roughly matches 400m cells at this zoom
                extruded: false,
                getPosition: d => d.position,
                getFillColor: d => getDepthColor(d.depth, minDepth, maxDepth),
                pickable: false,
                opacity: 0.5
            });
        }
        
        // Create scatter layer with current data
        function createScatterLayer(data) {
            return new ScatterplotLayer({
                id: 'porpoise-layer',
                data: data,
                getPosition: d => d.position,
                getRadius: d => d.radius || 400,
                getFillColor: d => d.color || [0, 150, 255, 200],
                pickable: true,
                opacity: 0.9,
                stroked: true,
                getLineColor: [255, 255, 255, 150],
                lineWidthMinPixels: 1,
                radiusMinPixels: 4,
                radiusMaxPixels: 20,
                transitions: {
                    getPosition: 200,
                    getRadius: 200
                }
            });
        }
        
        // Create turbine layer (orange-red markers for wind turbines)
        function createTurbineLayer(data) {
            if (!data || data.length === 0 || !showTurbineLayer) return null;
            return new ScatterplotLayer({
                id: 'turbine-layer',
                data: data,
                getPosition: d => d.position,
                getRadius: 800,
                getFillColor: [255, 100, 50, 220],  // Orange-red
                pickable: true,
                opacity: 1.0,
                stroked: true,
                getLineColor: [255, 255, 255, 255],
                lineWidthMinPixels: 2,
                radiusMinPixels: 5,
                radiusMaxPixels: 20
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
                console.log('Creating construction noise layer with', constructionData.length, 'points');
                layers.push(new ScatterplotLayer({
                    id: 'noise-construction-layer',
                    data: constructionData,
                    getPosition: d => d.position,
                    getRadius: 2500,  // Larger radius for high-impact noise
                    getFillColor: d => {
                        // Red gradient - construction noise is very loud
                        const levelAbove = d.level - NOISE_THRESHOLD;
                        if (levelAbove <= 0) return [0, 0, 0, 0];
                        const t = Math.min(1, levelAbove / 48);
                        const alpha = Math.floor(120 + 100 * t);  // 120-220 alpha
                        return [255, 30, 30, alpha];  // Bright red
                    },
                    pickable: false,
                    opacity: 0.7,
                    radiusMinPixels: 5,
                    radiusMaxPixels: 50
                }));
            }
            
            // Operational noise layer (YELLOW/ORANGE - low impact, ~145 dB)
            if (operationalData.length > 0) {
                console.log('Creating operational noise layer with', operationalData.length, 'points');
                layers.push(new ScatterplotLayer({
                    id: 'noise-operational-layer',
                    data: operationalData,
                    getPosition: d => d.position,
                    getRadius: 1500,  // Smaller radius for low-impact noise
                    getFillColor: d => {
                        // Yellow/orange gradient - operational noise is quieter
                        const level = d.level || 140;
                        const t = Math.min(1, (level - 130) / 20);  // Scale from 130-150 dB
                        const alpha = Math.floor(40 + 80 * t);  // 40-120 alpha
                        return [255, 180, 50, alpha];  // Yellow-orange
                    },
                    pickable: false,
                    opacity: 0.4,
                    radiusMinPixels: 2,
                    radiusMaxPixels: 25
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
                getRadius: 1800,  // meters - matches depth cells
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
            const turbineLayer = createTurbineLayer(turbineData);
            if (turbineLayer) layers.push(turbineLayer);
            const shipLayer = createShipLayer(shipData);
            if (shipLayer) layers.push(shipLayer);
            layers.push(createScatterLayer(porpoiseData));
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
            porpoiseData = newData;
            // Use setProps to update ONLY the layers, not the whole map
            deckgl.setProps({ layers: buildLayers() });
            document.getElementById('point-count').textContent = porpoiseData.length;
        };
        
        // Set depth data (called once at startup)
        window.setDepthData = function(data, gridWidth, gridHeight) {
            GRID_WIDTH = gridWidth || 400;
            GRID_HEIGHT = gridHeight || 400;
            depthData = data;
            deckgl.setProps({ layers: buildLayers() });
            console.log('Depth layer loaded:', data.length, 'cells');
        };
        
        // Set turbine data (called when turbine scenario is selected)
        window.setTurbineData = function(data) {
            turbineData = data || [];
            deckgl.setProps({ layers: buildLayers() });
            console.log('Turbine layer loaded:', turbineData.length, 'turbines');
            // Update turbine count if element exists
            const el = document.getElementById('turbine-count');
            if (el) el.textContent = turbineData.length;
        };
        
        // Set noise propagation data (calculated from turbines)
        window.setNoiseData = function(data) {
            noiseData = data || [];
            deckgl.setProps({ layers: buildLayers() });
            console.log('Noise layer loaded:', noiseData.length, 'cells above threshold');
        };
        
        // Set foraging data (food probability / patches)
        window.setForagingData = function(data) {
            foragingData = data || [];
            deckgl.setProps({ layers: buildLayers() });
            console.log('Foraging layer loaded:', foragingData.length, 'food cells');
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
                window.setDepthData(event.data.data, event.data.gridWidth, event.data.gridHeight);
            }
            if (event.data && event.data.type === 'setTurbineData') {
                window.setTurbineData(event.data.data);
            }
            if (event.data && event.data.type === 'setNoiseData') {
                window.setNoiseData(event.data.data);
            }
            if (event.data && event.data.type === 'setForagingData') {
                window.setForagingData(event.data.data);
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
        # Main content: map on left, stats + charts on right
        ui.layout_columns(
            # Left: Large map
            ui.card(
                ui.card_header("Spatial Distribution"),
                create_static_pydeck_map(),
                ui.output_ui("depth_data_initializer"),
                ui.output_ui("foraging_data_initializer"),
                ui.output_ui("ship_data_initializer"),
                ui.output_ui("turbine_data_initializer"),
                ui.output_ui("noise_data_initializer"),
                ui.output_ui("porpoise_data_updater"),
                height="720px"
            ),
            # Right: Compact stats + charts stacked
            ui.div(
                # Population value box on its own
                ui.value_box(
                    "Population",
                    ui.output_text("current_population"),
                    showcase=ui.span("🐬", style="font-size: 1.5rem;"),
                    theme="primary",
                    height="65px",
                    style="margin-bottom: 8px;"
                ),
                # Year, Births, Deaths in one row - compact
                ui.layout_column_wrap(
                    ui.value_box(
                        "Year",
                        ui.div(ui.output_text("current_year"), style="font-size: 1.1rem;"),
                        showcase=ui.span("📅", style="font-size: 1rem;"),
                        theme="info",
                        height="55px"
                    ),
                    ui.value_box(
                        "Births",
                        ui.div(ui.output_text("total_births"), style="font-size: 1.1rem;"),
                        showcase=ui.span("🎂", style="font-size: 1rem;"),
                        theme="success",
                        height="55px"
                    ),
                    ui.value_box(
                        "Deaths",
                        ui.div(ui.output_text("total_deaths"), style="font-size: 1.1rem;"),
                        showcase=ui.span("💀", style="font-size: 1rem;"),
                        theme="warning",
                        height="55px"
                    ),
                    width=1/3,
                    heights_equal=True
                ),
                # Charts below the value boxes
                ui.card(
                    ui.card_header("Population Size", style="padding: 4px 10px; font-size: 0.9rem;"),
                    ui.output_ui("population_plot"),
                    height="170px"
                ),
                ui.card(
                    ui.card_header("Life and Death", style="padding: 4px 10px; font-size: 0.9rem;"),
                    ui.output_ui("life_death_plot"),
                    height="170px"
                ),
                ui.card(
                    ui.card_header("Energy Balance", style="padding: 4px 10px; font-size: 0.9rem;"),
                    ui.output_ui("energy_balance_plot"),
                    height="170px"
                )
            ),
            col_widths=[7, 5]
        )
    )
