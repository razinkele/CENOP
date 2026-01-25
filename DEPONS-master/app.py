import pydeck as pdk
import shinyswatch
import math
import json
from shiny import reactive, render, ui as shiny_ui
from shiny.express import input, ui


# Sample Data
DATA_URL = "https://raw.githubusercontent.com/visgl/deck.gl-data/master/examples/3d-heatmap/heatmap-data.csv"

ui.page_opts(title="Pydeck in Shiny (Fixed)", fillable=True, theme=shinyswatch.theme.darkly)

# Add CSS for full height maps
ui.tags.style("""
    .nav-panel-content { height: calc(100vh - 120px) !important; }
    .tab-content { height: 100% !important; }
    .tab-pane { height: 100% !important; }
    .bslib-page-fill { height: 100% !important; }
""")

with ui.sidebar():
    ui.input_slider("elevation", "Elevation Scale", min=0, max=100, value=50)
    ui.input_slider("radius", "Hexagon Radius", min=100, max=2000, step=100, value=1000)
    ui.hr()
    ui.h5("Animation Controls")
    ui.input_switch("animate", "Enable Animation", value=False)
    ui.input_slider("anim_speed", "Animation Speed (ms)", min=50, max=500, step=50, value=100)
    ui.input_select(
        "anim_layer",
        "Animate Layer",
        choices={"scatter": "Scatterplot", "hex": "Hexagons"},
        selected="scatter"
    )
    ui.hr()
    ui.h5("Ocean & Earth Map")
    ui.input_select(
        "basemap_style",
        "Basemap",
        choices={
            "maplibre_satellite": "MapLibre Satellite",
            "maplibre_streets": "MapLibre Streets",
            "maplibre_terrain": "MapLibre Terrain",
            "mapbox_terrain": "Mapbox Terrain",
            "emodnet_bathymetry": "EMODnet Bathymetry",
            "esri_ocean": "ESRI Ocean Basemap",
            "esri_ocean_reference": "ESRI Ocean + Labels",
            "gebco": "GEBCO Bathymetry",
            "esri_imagery": "ESRI World Imagery",
            "osm": "OpenStreetMap"
        },
        selected="maplibre_satellite"
    )
    ui.input_select(
        "overlay_layer",
        "Overlay Layer",
        choices={
            "emodnet_bathymetry_colors": "EMODnet Bathymetry Colors",
            "none": "None",
            "emodnet_seabed": "EMODnet Seabed Habitats",
            "emodnet_biology": "EMODnet Biology",
            "emodnet_human": "EMODnet Human Activities",
            "copernicus_sst": "Sea Surface Temperature",
            "copernicus_chlorophyll": "Chlorophyll"
        },
        selected="emodnet_bathymetry_colors"
    )
    ui.input_slider("overlay_opacity", "Overlay Opacity", min=0, max=1, step=0.1, value=0.7)
    ui.hr()
    ui.h5("3D Controls")
    ui.input_slider("terrain_exaggeration", "Terrain Exaggeration", min=1, max=50, step=1, value=10)
    ui.input_slider("map_pitch", "Camera Pitch", min=0, max=85, step=5, value=45)
    ui.input_slider("map_bearing", "Camera Bearing", min=0, max=360, step=15, value=0)
    ui.input_switch("enable_3d", "Enable 3D Terrain", value=True)

with ui.navset_tab(id="main_tabs"):
    with ui.nav_panel("Hexagon Map"):
        shiny_ui.panel_absolute(
            shiny_ui.div(
                shiny_ui.h5("Layer Controls", style="margin-top: 0; color: white;"),
                shiny_ui.input_slider("opacity", "Layer Opacity", min=0, max=1, step=0.1, value=0.6),
                shiny_ui.input_checkbox("show_hex", "Show Hexagons", value=True),
                shiny_ui.input_checkbox("show_scatter", "Show Scatterplot", value=True),
                style="padding: 15px; background-color: rgba(30,30,30,0.9); border-radius: 8px; border: 1px solid #555; box-shadow: 0 4px 6px rgba(0,0,0,0.3);"
            ),
            top="80px", right="20px", width="250px", draggable=True
        )

        @render.ui
        def my_map():
            # Get current settings (map only re-renders when these change, NOT for animation)
            base_elevation = input.elevation()
            base_radius = input.radius()
            opacity = input.opacity()
            show_hex = input.show_hex()
            show_scatter = input.show_scatter()
            animate = input.animate()
            anim_speed = input.anim_speed()
            anim_layer = input.anim_layer()

            # Build layers - animation happens in JavaScript, not Python
            layers_config = []
            
            if show_hex:
                layers_config.append({
                    "type": "HexagonLayer",
                    "id": "hex-layer",
                    "data": DATA_URL,
                    "getPosition": "d => [d.lng, d.lat]",
                    "elevationScale": base_elevation,
                    "radius": base_radius,
                    "elevationRange": [0, 3000],
                    "pickable": True,
                    "extruded": True,
                    "opacity": opacity,
                    "colorRange": [
                        [255, 255, 178],
                        [254, 217, 118],
                        [254, 178, 76],
                        [253, 141, 60],
                        [240, 59, 32],
                        [189, 0, 38],
                    ],
                })

            if show_scatter:
                layers_config.append({
                    "type": "ScatterplotLayer",
                    "id": "scatter-layer", 
                    "data": DATA_URL,
                    "getPosition": "d => [d.lng, d.lat]",
                    "getRadius": base_radius / 5,
                    "getFillColor": [0, 200, 255],
                    "pickable": True,
                    "opacity": opacity,
                })

            # Generate HTML with embedded JavaScript animation
            html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <script src="https://unpkg.com/deck.gl@latest/dist.min.js"></script>
    <script src="https://d3js.org/d3.v7.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; }}
        #container {{ width: 100vw; height: 100vh; }}
        .legend {{
            position: absolute;
            bottom: 30px;
            left: 20px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px 18px;
            border-radius: 10px;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            min-width: 180px;
        }}
        .legend h4 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #4fc3f7;
            border-bottom: 1px solid #444;
            padding-bottom: 8px;
        }}
        .legend-section {{
            margin-bottom: 12px;
        }}
        .legend-section-title {{
            font-size: 11px;
            color: #aaa;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-color {{
            width: 18px;
            height: 14px;
            margin-right: 10px;
            border-radius: 3px;
            border: 1px solid rgba(255,255,255,0.2);
        }}
        .legend-gradient {{
            width: 100%;
            height: 18px;
            border-radius: 4px;
            margin: 6px 0;
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 10px;
            color: #888;
        }}
        .legend-value {{
            font-size: 11px;
            color: #ccc;
        }}
    </style>
</head>
<body>
    <div id="container"></div>
    <div class="legend" id="legend">
        <h4>📊 Map Legend</h4>
        <div class="legend-section" id="hex-legend" style="display: {('block' if show_hex else 'none')};">
            <div class="legend-section-title">Hexagon Density</div>
            <div class="legend-gradient" style="background: linear-gradient(to right, rgb(255,255,178), rgb(254,217,118), rgb(254,178,76), rgb(253,141,60), rgb(240,59,32), rgb(189,0,38));"></div>
            <div class="legend-labels">
                <span>Low</span>
                <span>Medium</span>
                <span>High</span>
            </div>
            <div class="legend-item" style="margin-top: 8px;">
                <span class="legend-value">Elevation: {base_elevation}x | Radius: {base_radius}m</span>
            </div>
        </div>
        <div class="legend-section" id="scatter-legend" style="display: {('block' if show_scatter else 'none')};">
            <div class="legend-section-title">Scatter Points</div>
            <div class="legend-item">
                <div class="legend-color" style="background: rgb(0, 200, 255); border-radius: 50%;"></div>
                <span class="legend-value">Data Points (r={int(base_radius/5)}m)</span>
            </div>
        </div>
        <div class="legend-section" style="border-top: 1px solid #444; padding-top: 10px; margin-top: 10px;">
            <div class="legend-section-title">Settings</div>
            <div class="legend-value">Opacity: {opacity}</div>
            <div class="legend-value">Animation: {('ON' if animate else 'OFF')}</div>
        </div>
    </div>
    <script>
        const DATA_URL = "{DATA_URL}";
        const ANIMATE = {str(animate).lower()};
        const ANIM_SPEED = {anim_speed};
        const ANIM_LAYER = "{anim_layer}";
        const BASE_ELEVATION = {base_elevation};
        const BASE_RADIUS = {base_radius};
        const OPACITY = {opacity};
        const SHOW_HEX = {str(show_hex).lower()};
        const SHOW_SCATTER = {str(show_scatter).lower()};
        
        let animationFrame = 0;
        let data = [];
        
        // Load data
        d3.csv(DATA_URL).then(rawData => {{
            data = rawData.map(d => ({{
                position: [parseFloat(d.lng), parseFloat(d.lat)]
            }}));
            render();
            if (ANIMATE) animate();
        }});
        
        function getScatterColor(frame) {{
            const r = Math.floor(128 + 127 * Math.sin(frame * 0.1));
            const g = Math.floor(128 + 127 * Math.cos(frame * 0.1));
            const b = Math.floor(128 + 127 * Math.sin(frame * 0.05 + 1.57));
            return [r, g, b, 200];
        }}
        
        function getScatterRadius(frame) {{
            const phase = Math.sin(frame * 0.1);
            return (BASE_RADIUS / 5) * (0.5 + 0.5 * phase);
        }}
        
        function getHexElevation(frame) {{
            const phase = Math.sin(frame * 0.1);
            return BASE_ELEVATION * (0.5 + 0.5 * phase);
        }}
        
        function render() {{
            const layers = [];
            
            if (SHOW_HEX) {{
                const hexElevation = (ANIMATE && ANIM_LAYER === 'hex') ? getHexElevation(animationFrame) : BASE_ELEVATION;
                layers.push(
                    new deck.HexagonLayer({{
                        id: 'hex-layer',
                        data: data,
                        getPosition: d => d.position,
                        elevationScale: hexElevation,
                        radius: BASE_RADIUS,
                        elevationRange: [0, 3000],
                        pickable: true,
                        extruded: true,
                        opacity: OPACITY,
                        colorRange: [
                            [255, 255, 178],
                            [254, 217, 118],
                            [254, 178, 76],
                            [253, 141, 60],
                            [240, 59, 32],
                            [189, 0, 38],
                        ],
                    }})
                );
            }}
            
            if (SHOW_SCATTER) {{
                const scatterColor = (ANIMATE && ANIM_LAYER === 'scatter') ? getScatterColor(animationFrame) : [0, 200, 255, 200];
                const scatterRadius = (ANIMATE && ANIM_LAYER === 'scatter') ? getScatterRadius(animationFrame) : BASE_RADIUS / 5;
                layers.push(
                    new deck.ScatterplotLayer({{
                        id: 'scatter-layer',
                        data: data,
                        getPosition: d => d.position,
                        getRadius: scatterRadius,
                        getFillColor: scatterColor,
                        pickable: true,
                        opacity: OPACITY,
                        radiusMinPixels: 2,
                    }})
                );
            }}
            
            if (!window.deckgl) {{
                window.deckgl = new deck.DeckGL({{
                    container: 'container',
                    initialViewState: {{
                        longitude: -1.415,
                        latitude: 52.23,
                        zoom: 6,
                        pitch: 45,
                        bearing: 0
                    }},
                    controller: true,
                    layers: layers
                }});
            }} else {{
                window.deckgl.setProps({{ layers: layers }});
            }}
        }}
        
        function animate() {{
            animationFrame++;
            render();
            setTimeout(() => requestAnimationFrame(animate), ANIM_SPEED);
        }}
    </script>
</body>
</html>
'''

            return shiny_ui.tags.iframe(
                srcdoc=html_content,
                style="width: 100%; height: calc(100vh - 120px); min-height: 600px; border: none;",
            )

    with ui.nav_panel("Ocean & Earth Map"):
        @render.ui
        def natural_earth_map():
            basemap_style = input.basemap_style()
            terrain_exaggeration = input.terrain_exaggeration()
            map_pitch = input.map_pitch()
            map_bearing = input.map_bearing()
            enable_3d = input.enable_3d()
            overlay_layer = input.overlay_layer()
            overlay_opacity = input.overlay_opacity()
            
            basemap_names = {
                "emodnet_bathymetry": "EMODnet Bathymetry",
                "esri_ocean": "ESRI Ocean Basemap",
                "esri_ocean_reference": "ESRI Ocean + Labels",
                "gebco": "GEBCO Bathymetry",
                "esri_imagery": "ESRI World Imagery",
                "osm": "OpenStreetMap",
                "maplibre_streets": "MapLibre Streets",
                "maplibre_satellite": "MapLibre Satellite",
                "maplibre_terrain": "MapLibre Terrain",
                "mapbox_terrain": "Mapbox Terrain"
            }
            
            basemap_name = basemap_names.get(basemap_style, "Ocean Map")
            
            # Generate working deck.gl HTML with TileLayer
            html_content = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <script src="https://unpkg.com/deck.gl@^9.0.0/dist.min.js"></script>
    <style>
        html, body {{
            margin: 0;
            padding: 0;
            width: 100%;
            height: 100%;
            overflow: hidden;
        }}
        #deck-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .controls {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            max-width: 280px;
        }}
        .controls h4 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #4fc3f7;
        }}
        .controls p {{
            margin: 5px 0;
            font-size: 11px;
            color: #ccc;
        }}
        .legend {{
            position: absolute;
            bottom: 30px;
            left: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px 18px;
            border-radius: 10px;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            min-width: 200px;
        }}
        .legend h4 {{
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #4fc3f7;
            border-bottom: 1px solid #444;
            padding-bottom: 8px;
        }}
        .legend-gradient {{
            width: 100%;
            height: 18px;
            border-radius: 4px;
            margin: 6px 0;
            border: 1px solid rgba(255,255,255,0.1);
        }}
        .legend-labels {{
            display: flex;
            justify-content: space-between;
            font-size: 9px;
            color: #888;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-color {{
            width: 18px;
            height: 14px;
            margin-right: 10px;
            border-radius: 3px;
        }}
        .view-info {{
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 8px 12px;
            border-radius: 6px;
            color: white;
            font-family: monospace;
            font-size: 10px;
            z-index: 1000;
        }}
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div class="controls">
        <h4>🌊 {basemap_name}</h4>
        <p><strong>3D Mode:</strong> {"Enabled" if enable_3d else "Disabled"}</p>
        <p><strong>Terrain Scale:</strong> {terrain_exaggeration}x</p>
        <p><strong>Pitch:</strong> {map_pitch}°</p>
        <p><strong>Bearing:</strong> {map_bearing}°</p>
        <p><strong>Overlay:</strong> {"EMODnet Bathymetry" if overlay_layer == "emodnet_bathymetry_colors" else overlay_layer} ({int(overlay_opacity * 100)}%)</p>
        <p style="font-size: 10px; color: #888; margin-top: 10px;">
            <strong>Controls:</strong> Drag to pan, Scroll to zoom<br>
            Ctrl+Drag to rotate, Shift+Drag to tilt
        </p>
    </div>
    <div class="legend">
        <h4>📊 {"3D Terrain" if enable_3d else "Flat Map"} Legend</h4>
        {"<div style='margin-bottom: 10px;'><div class='legend-section-title' style='font-size: 10px; color: #4fc3f7;'>EMODnet Bathymetry Overlay</div></div>" if overlay_layer == "emodnet_bathymetry_colors" else ""}
        <div class="legend-gradient" style="background: linear-gradient(to right, #08306b, #2171b5, #4292c6, #6baed6, #9ecae1, #c6dbef, #228B22, #8B4513);"></div>
        <div class="legend-labels">
            <span>-6000m</span>
            <span>0m</span>
            <span>+3000m</span>
        </div>
        <div style="margin-top: 10px;">
            <div class="legend-item">
                <div class="legend-color" style="background: #08306b;"></div>
                <span>Deep Ocean</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #6baed6;"></div>
                <span>Continental Shelf</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #228B22;"></div>
                <span>Lowlands</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: #8B4513;"></div>
                <span>Mountains</span>
            </div>
        </div>
        {"<p style='margin-top: 10px; font-size: 10px; color: #4fc3f7;'>✓ 3D terrain + EMODnet bathymetry colors</p>" if enable_3d and overlay_layer == "emodnet_bathymetry_colors" else "<p style='margin-top: 10px; font-size: 10px; color: #4fc3f7;'>✓ 3D terrain elevation active</p>" if enable_3d else "<p style='margin-top: 10px; font-size: 10px; color: #888;'>Enable 3D for terrain elevation</p>"}
    </div>
    <div class="view-info" id="view-info">Loading...</div>
    
    <script>
        const {{DeckGL, TileLayer, BitmapLayer, TerrainLayer}} = deck;
        
        const BASEMAP_STYLE = "{basemap_style}";
        const ENABLE_3D = {str(enable_3d).lower()};
        const MAP_PITCH = {map_pitch};
        const MAP_BEARING = {map_bearing};
        const TERRAIN_EXAGGERATION = {terrain_exaggeration};
        const OVERLAY_LAYER = "{overlay_layer}";
        const OVERLAY_OPACITY = {overlay_opacity};
        
        // Choose tile URL based on basemap style
        let tileUrl;
        switch(BASEMAP_STYLE) {{
            case 'emodnet_bathymetry':
                tileUrl = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{{z}}/{{x}}/{{y}}.png';
                break;
            case 'esri_ocean':
            case 'esri_ocean_reference':
                tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}';
                break;
            case 'gebco':
                tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{{z}}/{{y}}/{{x}}';
                break;
            case 'esri_imagery':
                tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}';
                break;
            case 'maplibre_streets':
                // OpenFreeMap Streets (free, no API key)
                tileUrl = 'https://tiles.openfreemap.org/styles/liberty/{{z}}/{{x}}/{{y}}.png';
                break;
            case 'maplibre_satellite':
                // ESRI World Imagery (free, no API key required)
                tileUrl = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{{z}}/{{y}}/{{x}}';
                break;
            case 'maplibre_terrain':
                // OpenTopoMap (terrain style, free)
                tileUrl = 'https://tile.opentopomap.org/{{z}}/{{x}}/{{y}}.png';
                break;
            case 'mapbox_terrain':
                // Stamen Terrain (now hosted by Stadia Maps, free tier)
                tileUrl = 'https://tiles.stadiamaps.com/tiles/stamen_terrain/{{z}}/{{x}}/{{y}}.png';
                break;
            case 'osm':
            default:
                tileUrl = 'https://tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png';
                break;
        }}
        
        // EMODnet Bathymetry overlay URL
        const EMODNET_BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{{z}}/{{x}}/{{y}}.png';
        
        // AWS Terrain Tiles (open, no API key required) for elevation data
        const TERRAIN_URL = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{{z}}/{{x}}/{{y}}.png';
        
        // Build layers based on 3D mode
        let layers = [];
        
        if (ENABLE_3D) {{
            // Use TerrainLayer for true 3D elevation rendering
            const terrainLayer = new TerrainLayer({{
                id: 'terrain-layer',
                minZoom: 0,
                maxZoom: 15,
                elevationDecoder: {{
                    // Terrarium encoding: height = (red * 256 + green + blue / 256) - 32768
                    rScaler: 256,
                    gScaler: 1,
                    bScaler: 1/256,
                    offset: -32768
                }},
                elevationData: TERRAIN_URL,
                texture: tileUrl,
                wireframe: false,
                color: [255, 255, 255],
                meshMaxError: 4.0,
                operation: 'terrain+draw',
                // Apply terrain exaggeration
                elevationScale: TERRAIN_EXAGGERATION
            }});
            layers.push(terrainLayer);
            
            // Add bathymetry overlay if selected (for ocean coloring)
            if (OVERLAY_LAYER === 'emodnet_bathymetry_colors') {{
                const bathymetryOverlay = new TileLayer({{
                    id: 'bathymetry-overlay',
                    data: EMODNET_BATHYMETRY_URL,
                    minZoom: 0,
                    maxZoom: 19,
                    tileSize: 256,
                    opacity: OVERLAY_OPACITY,
                    renderSubLayers: props => {{
                        const {{
                            bbox: {{west, south, east, north}}
                        }} = props.tile;
                        
                        return new BitmapLayer(props, {{
                            data: null,
                            image: props.data,
                            bounds: [west, south, east, north],
                            opacity: OVERLAY_OPACITY
                        }});
                    }}
                }});
                layers.push(bathymetryOverlay);
            }}
        }} else {{
            // Use flat TileLayer when 3D is disabled
            const tileLayer = new TileLayer({{
                id: 'tile-layer',
                data: tileUrl,
                minZoom: 0,
                maxZoom: 19,
                tileSize: 256,
                renderSubLayers: props => {{
                    const {{
                        bbox: {{west, south, east, north}}
                    }} = props.tile;
                    
                    return new BitmapLayer(props, {{
                        data: null,
                        image: props.data,
                        bounds: [west, south, east, north]
                    }});
                }}
            }});
            layers.push(tileLayer);
            
            // Add bathymetry overlay if selected (for ocean coloring)
            if (OVERLAY_LAYER === 'emodnet_bathymetry_colors') {{
                const bathymetryOverlay = new TileLayer({{
                    id: 'bathymetry-overlay',
                    data: EMODNET_BATHYMETRY_URL,
                    minZoom: 0,
                    maxZoom: 19,
                    tileSize: 256,
                    opacity: OVERLAY_OPACITY,
                    renderSubLayers: props => {{
                        const {{
                            bbox: {{west, south, east, north}}
                        }} = props.tile;
                        
                        return new BitmapLayer(props, {{
                            data: null,
                            image: props.data,
                            bounds: [west, south, east, north],
                            opacity: OVERLAY_OPACITY
                        }});
                    }}
                }});
                layers.push(bathymetryOverlay);
            }}
        }}
        
        const INITIAL_VIEW_STATE = {{
            latitude: 55.4,
            longitude: 21.1,
            zoom: 9,
            pitch: ENABLE_3D ? MAP_PITCH : 0,
            bearing: ENABLE_3D ? MAP_BEARING : 0,
            maxPitch: 85
        }};
        
        const deckgl = new DeckGL({{
            container: 'deck-container',
            initialViewState: INITIAL_VIEW_STATE,
            controller: {{
                touchRotate: true,
                keyboard: true,
                dragRotate: true
            }},
            layers: layers,
            parameters: {{
                clearColor: [0.1, 0.15, 0.2, 1]
            }},
            onViewStateChange: ({{viewState}}) => {{
                document.getElementById('view-info').innerHTML = 
                    'Lat: ' + viewState.latitude.toFixed(3) + 
                    ' | Lon: ' + viewState.longitude.toFixed(3) + 
                    ' | Zoom: ' + viewState.zoom.toFixed(1) +
                    ' | 3D: ' + (ENABLE_3D ? 'ON (x' + TERRAIN_EXAGGERATION + ')' : 'OFF');
            }}
        }});
        
        // Initial view info
        document.getElementById('view-info').innerHTML = 
            'Lat: ' + INITIAL_VIEW_STATE.latitude.toFixed(3) + 
            ' | Lon: ' + INITIAL_VIEW_STATE.longitude.toFixed(3) + 
            ' | Zoom: ' + INITIAL_VIEW_STATE.zoom.toFixed(1) +
            ' | 3D: ' + (ENABLE_3D ? 'ON (x' + TERRAIN_EXAGGERATION + ')' : 'OFF');
    </script>
</body>
</html>
'''
            return shiny_ui.tags.iframe(
                srcdoc=html_content,
                style="width: 100%; height: calc(100vh - 120px); min-height: 600px; border: none;",
            )

    with ui.nav_panel("Dynamic Scatter"):
        @render.ui
        def dynamic_scatter_map():
            # Generate HTML with animated scatter points on bathymetry
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
            top: 0;
            left: 0;
        }
        .controls {
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            max-width: 280px;
        }
        .controls h4 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #4fc3f7;
        }
        .controls p {
            margin: 5px 0;
            font-size: 11px;
            color: #ccc;
        }
        .stats {
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #444;
        }
        .stats-value {
            font-size: 18px;
            font-weight: bold;
            color: #4fc3f7;
        }
        .legend {
            position: absolute;
            bottom: 30px;
            left: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px 18px;
            border-radius: 10px;
            color: white;
            font-family: 'Segoe UI', Arial, sans-serif;
            font-size: 11px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            min-width: 180px;
        }
        .legend h4 {
            margin: 0 0 12px 0;
            font-size: 14px;
            color: #4fc3f7;
            border-bottom: 1px solid #444;
            padding-bottom: 8px;
        }
        .legend-item {
            display: flex;
            align-items: center;
            margin: 6px 0;
        }
        .legend-color {
            width: 18px;
            height: 18px;
            margin-right: 10px;
            border-radius: 50%;
            border: 2px solid rgba(255,255,255,0.3);
        }
        .view-info {
            position: absolute;
            bottom: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.9);
            padding: 8px 12px;
            border-radius: 6px;
            color: white;
            font-family: monospace;
            font-size: 10px;
            z-index: 1000;
        }
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div class="controls">
        <h4>🔵 Dynamic Scatter & Areas</h4>
        <p><strong>Base Map:</strong> EMODnet Bathymetry</p>
        <p><strong>Animation:</strong> Active</p>
        <p><strong>Update Rate:</strong> 100ms</p>
        <div class="stats">
            <p>Active Points: <span class="stats-value" id="point-count">0</span></p>
            <p>Active Areas: <span class="stats-value" id="area-count">0</span></p>
        </div>
        <p style="font-size: 10px; color: #888; margin-top: 10px;">
            Points and areas randomly change
        </p>
    </div>
    <div class="legend">
        <h4>📊 Layer Legend</h4>
        <div style="margin-bottom: 8px; font-size: 10px; color: #aaa;">POINTS</div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(255, 100, 100);"></div>
            <span>High Activity</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(100, 255, 100);"></div>
            <span>Medium Activity</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgb(100, 100, 255);"></div>
            <span>Low Activity</span>
        </div>
        <div style="margin: 10px 0 8px 0; padding-top: 8px; border-top: 1px solid #444; font-size: 10px; color: #aaa;">AREAS</div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(255, 150, 50, 0.5); border-radius: 4px;"></div>
            <span>Zone A (Orange)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(50, 200, 255, 0.5); border-radius: 4px;"></div>
            <span>Zone B (Cyan)</span>
        </div>
        <div class="legend-item">
            <div class="legend-color" style="background: rgba(200, 50, 255, 0.5); border-radius: 4px;"></div>
            <span>Zone C (Purple)</span>
        </div>
    </div>
    <div class="view-info" id="view-info">Loading...</div>
    
    <script>
        const {DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, PolygonLayer} = deck;
        
        // EMODnet Bathymetry tiles
        const BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{z}/{x}/{y}.png';
        
        // Curonian Lagoon center
        const CENTER_LAT = 55.4;
        const CENTER_LON = 21.1;
        
        // Area color palettes
        const AREA_COLORS = [
            [255, 150, 50, 120],   // Orange
            [50, 200, 255, 120],   // Cyan
            [200, 50, 255, 120],   // Purple
            [50, 255, 150, 120],   // Green
            [255, 50, 150, 120]    // Pink
        ];
        
        // Generate random polygon around a center point
        function generateRandomPolygon(centerLon, centerLat, size, sides) {
            const polygon = [];
            for (let i = 0; i < sides; i++) {
                const angle = (i / sides) * Math.PI * 2;
                const radiusVariation = 0.7 + Math.random() * 0.6;
                const r = size * radiusVariation;
                polygon.push([
                    centerLon + r * Math.cos(angle),
                    centerLat + r * Math.sin(angle) * 0.6  // Adjust for latitude
                ]);
            }
            polygon.push(polygon[0]); // Close the polygon
            return polygon;
        }
        
        // Generate random area features
        function generateRandomAreas(count) {
            const areas = [];
            for (let i = 0; i < count; i++) {
                const centerLon = CENTER_LON + (Math.random() - 0.5) * 1.2;
                const centerLat = CENTER_LAT + (Math.random() - 0.5) * 0.6;
                const size = 0.02 + Math.random() * 0.08;
                const sides = 5 + Math.floor(Math.random() * 6);  // 5-10 sides
                const colorIndex = Math.floor(Math.random() * AREA_COLORS.length);
                
                areas.push({
                    polygon: generateRandomPolygon(centerLon, centerLat, size, sides),
                    color: [...AREA_COLORS[colorIndex]],
                    elevation: 0
                });
            }
            return areas;
        }
        
        // Generate random points around Curonian Lagoon
        function generateRandomPoints(count) {
            const points = [];
            for (let i = 0; i < count; i++) {
                points.push({
                    position: [
                        CENTER_LON + (Math.random() - 0.5) * 1.5,
                        CENTER_LAT + (Math.random() - 0.5) * 0.8
                    ],
                    radius: 200 + Math.random() * 800,
                    color: [
                        50 + Math.floor(Math.random() * 205),
                        50 + Math.floor(Math.random() * 205),
                        50 + Math.floor(Math.random() * 205),
                        150 + Math.floor(Math.random() * 105)
                    ]
                });
            }
            return points;
        }
        
        let scatterData = generateRandomPoints(150);
        let areaData = generateRandomAreas(8);
        let animationFrame = 0;
        
        // Bathymetry base layer
        const bathymetryLayer = new TileLayer({
            id: 'bathymetry-layer',
            data: BATHYMETRY_URL,
            minZoom: 0,
            maxZoom: 19,
            tileSize: 256,
            renderSubLayers: props => {
                const {
                    bbox: {west, south, east, north}
                } = props.tile;
                
                return new BitmapLayer(props, {
                    data: null,
                    image: props.data,
                    bounds: [west, south, east, north]
                });
            }
        });
        
        // Update area data with animation
        function updateAreaData() {
            areaData = areaData.map(area => {
                // 15% chance to regenerate polygon
                if (Math.random() < 0.15) {
                    const centerLon = CENTER_LON + (Math.random() - 0.5) * 1.2;
                    const centerLat = CENTER_LAT + (Math.random() - 0.5) * 0.6;
                    const size = 0.02 + Math.random() * 0.08;
                    const sides = 5 + Math.floor(Math.random() * 6);
                    const colorIndex = Math.floor(Math.random() * AREA_COLORS.length);
                    
                    return {
                        polygon: generateRandomPolygon(centerLon, centerLat, size, sides),
                        color: [...AREA_COLORS[colorIndex]],
                        elevation: 0
                    };
                }
                
                // Slightly animate color opacity
                const newAlpha = Math.min(180, Math.max(60, area.color[3] + Math.floor((Math.random() - 0.5) * 30)));
                return {
                    ...area,
                    color: [area.color[0], area.color[1], area.color[2], newAlpha]
                };
            });
            
            // Occasionally add or remove areas
            if (Math.random() < 0.05 && areaData.length < 12) {
                const centerLon = CENTER_LON + (Math.random() - 0.5) * 1.2;
                const centerLat = CENTER_LAT + (Math.random() - 0.5) * 0.6;
                const size = 0.02 + Math.random() * 0.08;
                const sides = 5 + Math.floor(Math.random() * 6);
                const colorIndex = Math.floor(Math.random() * AREA_COLORS.length);
                
                areaData.push({
                    polygon: generateRandomPolygon(centerLon, centerLat, size, sides),
                    color: [...AREA_COLORS[colorIndex]],
                    elevation: 0
                });
            } else if (Math.random() < 0.05 && areaData.length > 5) {
                areaData.pop();
            }
        }
        
        function updateScatterData() {
            // Randomly update some points each frame
            scatterData = scatterData.map(point => {
                if (Math.random() < 0.3) {
                    // 30% chance to update each point
                    return {
                        position: [
                            CENTER_LON + (Math.random() - 0.5) * 1.5,
                            CENTER_LAT + (Math.random() - 0.5) * 0.8
                        ],
                        radius: 200 + Math.random() * 800,
                        color: [
                            50 + Math.floor(Math.random() * 205),
                            50 + Math.floor(Math.random() * 205),
                            50 + Math.floor(Math.random() * 205),
                            150 + Math.floor(Math.random() * 105)
                        ]
                    };
                }
                // Slightly animate existing points
                return {
                    ...point,
                    radius: point.radius * (0.9 + Math.random() * 0.2),
                    color: [
                        Math.min(255, Math.max(50, point.color[0] + Math.floor((Math.random() - 0.5) * 20))),
                        Math.min(255, Math.max(50, point.color[1] + Math.floor((Math.random() - 0.5) * 20))),
                        Math.min(255, Math.max(50, point.color[2] + Math.floor((Math.random() - 0.5) * 20))),
                        point.color[3]
                    ]
                };
            });
            
            // Occasionally add or remove points
            if (Math.random() < 0.1 && scatterData.length < 200) {
                scatterData.push({
                    position: [
                        CENTER_LON + (Math.random() - 0.5) * 1.5,
                        CENTER_LAT + (Math.random() - 0.5) * 0.8
                    ],
                    radius: 200 + Math.random() * 800,
                    color: [
                        50 + Math.floor(Math.random() * 205),
                        50 + Math.floor(Math.random() * 205),
                        50 + Math.floor(Math.random() * 205),
                        150 + Math.floor(Math.random() * 105)
                    ]
                });
            } else if (Math.random() < 0.1 && scatterData.length > 100) {
                scatterData.pop();
            }
        }
        
        function render() {
            const polygonLayer = new PolygonLayer({
                id: 'polygon-layer',
                data: areaData,
                getPolygon: d => d.polygon,
                getFillColor: d => d.color,
                getLineColor: d => [d.color[0], d.color[1], d.color[2], 200],
                getLineWidth: 2,
                lineWidthMinPixels: 1,
                pickable: true,
                stroked: true,
                filled: true,
                wireframe: false,
                opacity: 0.7
            });
            
            const scatterLayer = new ScatterplotLayer({
                id: 'scatter-layer',
                data: scatterData,
                getPosition: d => d.position,
                getRadius: d => d.radius,
                getFillColor: d => d.color,
                pickable: true,
                opacity: 0.8,
                stroked: true,
                getLineColor: [255, 255, 255, 100],
                lineWidthMinPixels: 1,
                radiusMinPixels: 3,
                radiusMaxPixels: 50
            });
            
            if (!window.deckgl) {
                window.deckgl = new DeckGL({
                    container: 'deck-container',
                    initialViewState: {
                        latitude: CENTER_LAT,
                        longitude: CENTER_LON,
                        zoom: 9,
                        pitch: 0,
                        bearing: 0
                    },
                    controller: true,
                    layers: [bathymetryLayer, polygonLayer, scatterLayer],
                    parameters: {
                        clearColor: [0.1, 0.15, 0.2, 1]
                    },
                    onViewStateChange: ({viewState}) => {
                        document.getElementById('view-info').innerHTML = 
                            'Lat: ' + viewState.latitude.toFixed(3) + 
                            ' | Lon: ' + viewState.longitude.toFixed(3) + 
                            ' | Zoom: ' + viewState.zoom.toFixed(1);
                    }
                });
            } else {
                window.deckgl.setProps({ layers: [bathymetryLayer, polygonLayer, scatterLayer] });
            }
            
            // Update stats display
            document.getElementById('point-count').textContent = scatterData.length;
            document.getElementById('area-count').textContent = areaData.length;
        }
        
        function animate() {
            animationFrame++;
            updateAreaData();
            updateScatterData();
            render();
            setTimeout(() => requestAnimationFrame(animate), 100);
        }
        
        // Initial render and start animation
        render();
        animate();
        
        // Initial view info
        document.getElementById('view-info').innerHTML = 
            'Lat: ' + CENTER_LAT.toFixed(3) + 
            ' | Lon: ' + CENTER_LON.toFixed(3) + 
            ' | Zoom: 9.0';
    </script>
</body>
</html>
'''
            return shiny_ui.tags.iframe(
                srcdoc=html_content,
                style="width: 100%; height: calc(100vh - 120px); min-height: 600px; border: none;",
            )