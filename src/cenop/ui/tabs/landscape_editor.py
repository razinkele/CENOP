"""
Landscape GIS Editor Tab UI

Read-only viewer for spatial data layers (bathymetry, prey, salinity, etc.)
using a deck.gl SolidPolygonLayer rendered in an iframe.
"""

from shiny import ui


def create_gis_editor_map():
    """
    Create a self-contained deck.gl iframe for viewing GIS layers.
    Uses the same postMessage pattern as create_static_pydeck_map() in dashboard.py.
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
            margin: 0; padding: 0;
            width: 100%; height: 100%;
            overflow: hidden;
            font-family: 'Segoe UI', Arial, sans-serif;
        }
        #deck-container {
            width: 100%; height: 100%;
            position: absolute; top: 0; left: 0;
        }
        .legend {
            position: absolute;
            bottom: 12px; left: 12px;
            background: rgba(30,30,30,0.92);
            padding: 10px 14px;
            border-radius: 8px;
            color: white;
            font-size: 11px;
            z-index: 1000;
            border: 1px solid #444;
            min-width: 120px;
        }
        .legend-title {
            font-weight: bold;
            color: #4fc3f7;
            margin-bottom: 6px;
            font-size: 12px;
        }
        .legend-bar {
            width: 100%;
            height: 14px;
            border-radius: 3px;
            margin: 4px 0 2px 0;
        }
        .legend-labels {
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #aaa;
        }
        .tooltip {
            position: absolute;
            pointer-events: none;
            background: rgba(20,20,20,0.92);
            color: white;
            padding: 6px 10px;
            border-radius: 5px;
            font-size: 11px;
            z-index: 2000;
            display: none;
            border: 1px solid #555;
        }
        .view-info {
            position: absolute;
            bottom: 10px; right: 10px;
            background: rgba(30,30,30,0.9);
            padding: 6px 10px;
            border-radius: 5px;
            color: #ccc;
            font-family: monospace;
            font-size: 10px;
            z-index: 1000;
        }
        .placeholder-msg {
            position: absolute;
            top: 50%; left: 50%;
            transform: translate(-50%, -50%);
            color: #888;
            font-size: 16px;
            text-align: center;
        }
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div id="placeholder" class="placeholder-msg">Select a layer and click <b>Load Layer</b></div>
    <div class="legend" id="legend" style="display:none;">
        <div class="legend-title" id="legend-title">Layer</div>
        <div class="legend-bar" id="legend-bar"></div>
        <div class="legend-labels">
            <span id="legend-min">0</span>
            <span id="legend-max">1</span>
        </div>
    </div>
    <div class="tooltip" id="tooltip"></div>
    <div class="view-info" id="view-info">Loading...</div>

    <script>
        const {DeckGL, TileLayer, BitmapLayer, SolidPolygonLayer} = deck;

        const BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png';

        // Defaults are overridden by setGISBounds message from Shiny
        let LAT_MIN = 50.0, LAT_MAX = 60.0;
        let LON_MIN = -2.0, LON_MAX = 24.0;
        let CENTER_LAT = 55.0;
        let CENTER_LON = 11.0;
        let CELL_DEG_LAT = 0.005;
        let CELL_DEG_LON = 0.005;

        let layerData = [];
        let currentScheme = 'viridis';
        let dataMin = 0, dataMax = 1;

        const bathymetryLayer = new TileLayer({
            id: 'base-tiles',
            data: BATHYMETRY_URL,
            minZoom: 0, maxZoom: 12, tileSize: 256,
            renderSubLayers: props => {
                const {bbox: {west, south, east, north}} = props.tile;
                return new BitmapLayer(props, {
                    data: null, image: props.data,
                    bounds: [west, south, east, north]
                });
            }
        });

        // ---- Color ramps ----
        function viridisColor(t) {
            let r, g, b;
            if (t < 0.25) {
                const s = t / 0.25;
                r = Math.floor(255*(1-s) + 32*s);
                g = Math.floor(255*(1-s*0.3));
                b = Math.floor(100*s);
            } else if (t < 0.5) {
                const s = (t-0.25)/0.25;
                r = Math.floor(32*(1-s));
                g = Math.floor(178 + 77*s);
                b = Math.floor(100 + 155*s);
            } else if (t < 0.75) {
                const s = (t-0.5)/0.25;
                r = Math.floor(30*s);
                g = Math.floor(255*(1-s) + 100*s);
                b = 255;
            } else {
                const s = (t-0.75)/0.25;
                r = Math.floor(30 + 90*s);
                g = Math.floor(100*(1-s) + 20*s);
                b = Math.floor(255*(1-s*0.3));
            }
            return [r, g, b, 180];
        }

        function greenSeq(t) {
            return [Math.floor(20 + 40*t), Math.floor(80 + 175*t), Math.floor(20 + 30*t), 180];
        }

        function blueWhiteSeq(t) {
            return [Math.floor(20 + 235*t), Math.floor(60 + 195*t), Math.floor(180 + 75*t), 180];
        }

        function yellowRedSeq(t) {
            return [Math.floor(255), Math.floor(230*(1-t) + 30*t), Math.floor(30), 180];
        }

        const CATEGORICAL_COLORS = [
            [31,119,180,180],[255,127,14,180],[44,160,44,180],[214,39,40,180],
            [148,103,189,180],[140,86,75,180],[227,119,194,180],[127,127,127,180],
            [188,189,34,180],[23,190,207,180],[174,199,232,180],[255,187,120,180]
        ];

        function categoricalColor(val) {
            const idx = Math.abs(Math.round(val)) % CATEGORICAL_COLORS.length;
            return CATEGORICAL_COLORS[idx];
        }

        function getColor(value, scheme, mn, mx) {
            if (scheme === 'categorical') return categoricalColor(value);
            const range = mx - mn;
            const t = range > 0 ? Math.max(0, Math.min(1, (value - mn) / range)) : 0.5;
            switch (scheme) {
                case 'viridis': return viridisColor(t);
                case 'green': return greenSeq(t);
                case 'blue_white': return blueWhiteSeq(t);
                case 'yellow_red': return yellowRedSeq(t);
                default: return viridisColor(t);
            }
        }

        // CSS gradients for legend
        const SCHEME_GRADIENTS = {
            viridis: 'linear-gradient(to right, rgb(255,255,0), rgb(32,178,100), rgb(0,100,255), rgb(120,20,178))',
            green: 'linear-gradient(to right, rgb(20,80,20), rgb(60,255,50))',
            blue_white: 'linear-gradient(to right, rgb(20,60,180), rgb(255,255,255))',
            yellow_red: 'linear-gradient(to right, rgb(255,230,30), rgb(255,30,30))',
            categorical: 'linear-gradient(to right, #1f77b4, #ff7f0e, #2ca02c, #d62728, #9467bd, #8c564b)'
        };

        function buildLayers() {
            const layers = [bathymetryLayer];
            if (layerData.length > 0) {
                const halfLon = CELL_DEG_LON * 0.5;
                const halfLat = CELL_DEG_LAT * 0.5;
                layers.push(new SolidPolygonLayer({
                    id: 'gis-layer',
                    data: layerData,
                    getPolygon: d => {
                        const [lon, lat] = d.position;
                        return [[lon-halfLon,lat-halfLat],[lon+halfLon,lat-halfLat],
                                [lon+halfLon,lat+halfLat],[lon-halfLon,lat+halfLat]];
                    },
                    getFillColor: d => getColor(d.value, currentScheme, dataMin, dataMax),
                    pickable: true,
                    extruded: false,
                    updateTriggers: {
                        getFillColor: [currentScheme, dataMin, dataMax]
                    }
                }));
            }
            return layers;
        }

        const deckgl = new DeckGL({
            container: 'deck-container',
            initialViewState: {
                latitude: CENTER_LAT, longitude: CENTER_LON,
                zoom: 5, pitch: 0, bearing: 0
            },
            controller: true,
            layers: buildLayers(),
            parameters: { clearColor: [0.05, 0.1, 0.15, 1] },
            getTooltip: ({object}) => {
                if (!object) return null;
                return {
                    html: '<b>Value:</b> ' + (typeof object.value === 'number' ? object.value.toFixed(4) : object.value),
                    style: {background: 'rgba(20,20,20,0.92)', color: '#fff', fontSize: '12px', padding: '6px 10px', borderRadius: '5px'}
                };
            },
            onViewStateChange: ({viewState}) => {
                document.getElementById('view-info').innerHTML =
                    'Lat: ' + viewState.latitude.toFixed(3) +
                    ' | Lon: ' + viewState.longitude.toFixed(3) +
                    ' | Zoom: ' + viewState.zoom.toFixed(1);
            }
        });

        window.deckgl = deckgl;

        // Message handler for receiving GIS layer data from Shiny
        window.addEventListener('message', function(event) {
            const d = event.data;
            if (!d) return;

            if (d.type === 'setGISLayerData') {
                layerData = d.data || [];
                currentScheme = d.scheme || 'viridis';
                dataMin = d.dataMin != null ? d.dataMin : 0;
                dataMax = d.dataMax != null ? d.dataMax : 1;
                CELL_DEG_LAT = d.cellDegLat || 0.005;
                CELL_DEG_LON = d.cellDegLon || 0.005;

                deckgl.setProps({ layers: buildLayers() });

                // Update legend
                const legend = document.getElementById('legend');
                legend.style.display = layerData.length > 0 ? 'block' : 'none';
                document.getElementById('legend-title').textContent = d.layerName || 'Layer';
                document.getElementById('legend-bar').style.background =
                    SCHEME_GRADIENTS[currentScheme] || SCHEME_GRADIENTS.viridis;
                document.getElementById('legend-min').textContent =
                    currentScheme === 'categorical' ? 'Classes' : dataMin.toFixed(2);
                document.getElementById('legend-max').textContent =
                    currentScheme === 'categorical' ? '' : dataMax.toFixed(2);

                // Hide placeholder
                document.getElementById('placeholder').style.display =
                    layerData.length > 0 ? 'none' : 'block';

                console.log('GIS layer loaded:', layerData.length, 'cells, scheme:', currentScheme);
            }

            if (d.type === 'setGISBounds') {
                LAT_MIN = d.latMin; LAT_MAX = d.latMax;
                LON_MIN = d.lonMin; LON_MAX = d.lonMax;
                CENTER_LAT = (LAT_MIN + LAT_MAX) / 2;
                CENTER_LON = (LON_MIN + LON_MAX) / 2;
                deckgl.setProps({
                    initialViewState: {
                        longitude: CENTER_LON, latitude: CENTER_LAT,
                        zoom: 7, pitch: 0, bearing: 0
                    }
                });
            }
        });

        document.getElementById('view-info').innerHTML =
            'Lat: ' + CENTER_LAT.toFixed(3) +
            ' | Lon: ' + CENTER_LON.toFixed(3) +
            ' | Zoom: 5.0';
    </script>
</body>
</html>
'''
    return ui.tags.iframe(
        id="gis-editor-map-frame",
        srcdoc=html_content,
        style="width: 100%; height: 560px; min-height: 400px; border: none; border-radius: 8px;",
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
                create_gis_editor_map(),
                ui.output_ui("gis_data_sender"),
                height="620px"
            ),
            col_widths=[3, 9]
        )
    )
