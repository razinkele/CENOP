"""
Chart Helper Functions for CENOP

Provides standardized Plotly chart creation with DEPONS-style formatting.
Eliminates duplicate chart configuration code across renderers.
"""

from typing import List, Dict, Optional, Any, Tuple
import pandas as pd
import plotly.graph_objects as go
from shiny import ui


# DEPONS color scheme
DEPONS_COLORS = {
    'primary': 'blue',
    'secondary': 'red',
    'success': 'green',
    'warning': 'orange',
    'background': 'rgb(192, 192, 192)',
    'grid': 'white'
}


def _apply_depons_layout(fig: go.Figure, title: str, height: int, 
                         x_title: str = "", y_title: str = "",
                         show_legend: bool = True) -> go.Figure:
    """
    Apply standard DEPONS styling to a Plotly figure.
    
    Args:
        fig: Plotly figure to style
        title: Chart title
        height: Chart height in pixels
        x_title: X-axis label
        y_title: Y-axis label
        show_legend: Whether to show legend
        
    Returns:
        Styled figure
    """
    legend_config = dict(yanchor="top", y=0.99, xanchor="right", x=0.99) if show_legend else {}
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        legend=legend_config,
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor=DEPONS_COLORS['background'],
        paper_bgcolor='white'
    )
    fig.update_xaxes(showgrid=True, gridcolor=DEPONS_COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=DEPONS_COLORS['grid'])
    
    return fig


def create_time_series_chart(
    df: pd.DataFrame,
    x_col: str,
    y_cols: List[str],
    colors: List[str],
    names: List[str],
    title: str,
    x_title: str = "",
    y_title: str = "",
    height: int = 180
) -> ui.HTML:
    """
    Create a standardized time series chart.
    
    Args:
        df: DataFrame with the data
        x_col: Column name for x-axis
        y_cols: List of column names for y-axis traces
        colors: List of colors for each trace
        names: List of legend names for each trace
        title: Chart title
        x_title: X-axis label
        y_title: Y-axis label
        height: Chart height in pixels
        
    Returns:
        Shiny HTML element containing the chart
    """
    fig = go.Figure()
    
    for col, color, name in zip(y_cols, colors, names):
        if col in df.columns:
            fig.add_trace(go.Scatter(
                x=df[x_col],
                y=df[col],
                mode='lines',
                name=name,
                line=dict(color=color, width=2)
            ))
    
    _apply_depons_layout(fig, title, height, x_title, y_title)
    
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


def create_histogram_chart(
    data: List[float],
    title: str,
    x_title: str,
    y_title: str,
    x_range: Tuple[float, float],
    nbins: int = 30,
    color: str = 'red',
    height: int = 300
) -> ui.HTML:
    """
    Create a standardized histogram chart.
    
    Args:
        data: List of values to histogram
        title: Chart title
        x_title: X-axis label
        y_title: Y-axis label
        x_range: Tuple of (min, max) for x-axis
        nbins: Number of bins
        color: Bar color
        height: Chart height in pixels
        
    Returns:
        Shiny HTML element containing the chart
    """
    fig = go.Figure()
    
    bin_size = (x_range[1] - x_range[0]) / nbins
    
    fig.add_trace(go.Histogram(
        x=data,
        nbinsx=nbins,
        xbins=dict(start=x_range[0], end=x_range[1], size=bin_size),
        marker_color=color,
        name=x_title
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_title,
        yaxis_title=y_title,
        height=height,
        xaxis=dict(range=list(x_range)),
        margin=dict(l=50, r=20, t=40, b=40),
        plot_bgcolor=DEPONS_COLORS['background'],
        bargap=0.1
    )
    fig.update_xaxes(showgrid=True, gridcolor=DEPONS_COLORS['grid'])
    fig.update_yaxes(showgrid=True, gridcolor=DEPONS_COLORS['grid'])
    
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


def create_map_figure(
    lats: List[float],
    lons: List[float],
    center_lat: float = 55.5,
    center_lon: float = 7.5,
    zoom: int = 5,
    height: int = 650
) -> ui.HTML:
    """
    Create a map with scatter points for porpoise positions.
    
    Args:
        lats: List of latitude values
        lons: List of longitude values
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Map zoom level
        height: Map height in pixels
        
    Returns:
        Shiny HTML element containing the map
    """
    if lats and lons:
        fig = go.Figure(go.Scattermapbox(
            lat=lats,
            lon=lons,
            mode='markers',
            marker=dict(size=6, color='blue', opacity=0.7),
            name='Porpoises'
        ))
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        fig = go.Figure()
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=zoom
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False
    )
    
    return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))


def create_pydeck_map(
    lats: List[float],
    lons: List[float],
    center_lat: float = 55.5,
    center_lon: float = 7.5,
    zoom: int = 6,
    height: str = "600px",
    map_id: str = "porpoise-map",
    _cache: Dict[str, Any] = {},  # mutable default = module-level cache
) -> ui.Tag:
    """
    Create an interactive pydeck/deck.gl map with scatter points.
    
    Uses EMODnet bathymetry tiles and deck.gl ScatterplotLayer
    for high-performance rendering of porpoise positions.
    
    Optimized: caches HTML output keyed by data hash to avoid
    redundant JSON serialization and HTML rebuilds.
    
    Args:
        lats: List of latitude values
        lons: List of longitude values
        center_lat: Map center latitude
        center_lon: Map center longitude
        zoom: Map zoom level
        height: Map height as CSS string (e.g., "600px")
        map_id: Unique ID for the map container
        
    Returns:
        Shiny iframe element containing the interactive map
    """
    import json
    import hashlib
    
    # Prepare point data for JavaScript — vectorized path
    points_data = []
    if lats and lons:
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
        # Build list-of-dicts via zip (avoids repeated dict() calls in loop)
        points_data = [
            {"position": [lon, lat], "radius": 400, "color": [0, 150, 255, 200]}
            for lat, lon in zip(lats, lons)
        ]
    
    points_json = json.dumps(points_data)
    point_count = len(points_data)
    
    # Cache key: hash of data + view parameters
    data_hash = hashlib.md5(points_json.encode()).hexdigest()[:8]
    cache_key = f"{map_id}:{data_hash}:{zoom}:{height}"
    
    if cache_key in _cache:
        return _cache[cache_key]
    
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
            font-family: 'Segoe UI', Arial, sans-serif;
        }}
        #deck-container {{
            width: 100%;
            height: 100%;
            position: absolute;
            top: 0;
            left: 0;
        }}
        .info-panel {{
            position: absolute;
            top: 10px;
            right: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 15px;
            border-radius: 10px;
            color: white;
            font-size: 12px;
            z-index: 1000;
            border: 1px solid #444;
            box-shadow: 0 4px 15px rgba(0,0,0,0.4);
            max-width: 200px;
        }}
        .info-panel h4 {{
            margin: 0 0 10px 0;
            font-size: 14px;
            color: #4fc3f7;
            border-bottom: 1px solid #444;
            padding-bottom: 8px;
        }}
        .stat-row {{
            display: flex;
            justify-content: space-between;
            margin: 6px 0;
        }}
        .stat-label {{
            color: #aaa;
        }}
        .stat-value {{
            font-weight: bold;
            color: #4fc3f7;
        }}
        .legend {{
            position: absolute;
            bottom: 20px;
            left: 10px;
            background: rgba(30, 30, 30, 0.95);
            padding: 12px 15px;
            border-radius: 8px;
            color: white;
            font-size: 11px;
            z-index: 1000;
            border: 1px solid #444;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            margin: 4px 0;
        }}
        .legend-dot {{
            width: 14px;
            height: 14px;
            margin-right: 8px;
            border-radius: 50%;
            border: 1px solid rgba(255,255,255,0.3);
        }}
        .legend-turbine {{
            width: 14px;
            height: 14px;
            margin-right: 8px;
            background: rgb(255, 100, 50);
            border: 2px solid white;
        }}
        .view-info {{
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
        }}
    </style>
</head>
<body>
    <div id="deck-container"></div>
    <div class="info-panel">
        <h4>🐬 Porpoise Distribution</h4>
        <div class="stat-row">
            <span class="stat-label">Porpoises:</span>
            <span class="stat-value" id="point-count">{point_count}</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Turbines:</span>
            <span class="stat-value" id="turbine-count">0</span>
        </div>
        <div class="stat-row">
            <span class="stat-label">Base Map:</span>
            <span class="stat-value" style="font-size: 10px;">EMODnet</span>
        </div>
    </div>
    <div class="legend">
        <div class="legend-item">
            <div class="legend-dot" style="background: rgb(0, 150, 255);"></div>
            <span>Porpoise</span>
        </div>
        <div class="legend-item">
            <div class="legend-turbine"></div>
            <span>Wind Turbine</span>
        </div>
    </div>
    <div class="view-info" id="view-info">Loading...</div>
    
    <!-- Data element for updates - following DEPONS pattern -->
    <script id="porpoise-data" type="application/json">{points_json}</script>
    
    <script>
        const {{DeckGL, TileLayer, BitmapLayer, ScatterplotLayer, IconLayer}} = deck;
        
        // EMODnet Bathymetry tiles (ocean-focused basemap) - STATIC, created once
        const BATHYMETRY_URL = 'https://tiles.emodnet-bathymetry.eu/2020/baselayer/web_mercator/{{z}}/{{x}}/{{y}}.png';
        
        // Center coordinates
        const CENTER_LAT = {center_lat};
        const CENTER_LON = {center_lon};
        
        // Parse porpoise data from embedded JSON - use let so it can be updated
        let porpoiseData = JSON.parse(document.getElementById('porpoise-data').textContent);
        console.log('Initial porpoise data (first 10):', porpoiseData.slice(0,10));
        
        // Static bathymetry base layer - never recreated
        const bathymetryLayer = new TileLayer({{
            id: 'bathymetry-layer',
            data: BATHYMETRY_URL,
            minZoom: 0,
            maxZoom: 12,
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
        
        // Dynamic porpoise icon layer - updated via setProps pattern from DEPONS
        function createScatterLayer(data) {{
            if (!data || data.length === 0) return null;
            return new IconLayer({{
                id: 'porpoise-layer',
                data: data,
                // Embedded base64 SVG data URL
                iconAtlas: 'data:image/svg+xml;base64,PD94bWwgdmVyc2lvbj0iMS4wIiBlbmNvZGluZz0idXRmLTgiPz4NCjxzdmcgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB3aWR0aD0iMzIiIGhlaWdodD0iMzIiIHZpZXdCb3g9IjAgMCAzMiAzMiI+DQogIDxnIGZpbGw9Im5vbmUiIHN0cm9rZT0iYmxhY2siIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIj4NCiAgICA8IS0tIEFycm93IHNoYWZ0IC0tPg0KICAgIDxsaW5lIHgxPSIxNiIgeTE9IjI2IiB4Mj0iMTYiIHkyPSI4IiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIgLz4NCiAgICA8IS0tIEFycm93IGhlYWQgLS0+DQogICAgPHBvbHlsaW5lIHBvaW50cz0iOCwxMiAxNiw0IDI0LDEyIiBmaWxsPSIjZmZmZmZmIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMyIvPg0KICA8L2c+DQo8L3N2Zz4=',
                iconMapping: {{ 'arrow': {{ x: 0, y: 0, width: 32, height: 32, anchorY: 16, anchorX: 16 }} }},
                getIcon: d => 'arrow',
                getPosition: d => d.position,
                getSize: d => Math.max(8, (d.radius || 200) / 20),
                sizeScale: 1,
                getAngle: d => d.heading || 0,
                getColor: d => {{
                    if (d.color && d.color.length >= 3) return d.color;
                    if (d.is_disturbed) return [255, 40, 40];
                    const age = d.age || 0;
                    if (age < 2) return [60, 180, 75];
                    if (age < 12) return [0, 150, 255];
                    return [160, 160, 160];
                }},

                pickable: true,
                opacity: 0.95,
                transitions: {{ getPosition: 300, getSize: 300, getAngle: 300, getColor: 300 }}
            }});
        }}
        
        // Turbine pole layer (static) for server-rendered charts
        let turbineData = [];
        function createTurbinePoleLayer(data) {{
            return new IconLayer({{
                id: 'turbine-pole-layer',
                data: data,
                iconAtlas: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><g fill="white"><circle cx="24" cy="20" r="3"/><rect x="22" y="10" width="4" height="32" fill="white"/></g></svg>',
                iconMapping: {{ 'pole': {{ x: 0, y: 0, width: 48, height: 48, anchorY: 42, anchorX: 24 }} }},
                getIcon: d => 'pole',
                getPosition: d => d.position,
                getSize: d => Math.max(20, Math.min(64, (d.radius || 300) / 15)),
                getColor: d => {{
                    if (d.phase === 'construction') return [255, 70, 40, 220];
                    if (d.phase === 'operational') return [50, 160, 240, 220];
                    if (d.phase === 'planned') return [180, 180, 180, 180];
                    return d.color || [255, 140, 60];
                }},
                pickable: true,
                opacity: 0.95,
            }});
        }}

        function createTurbineBladeLayer(data) {{
            return new IconLayer({{
                id: 'turbine-blade-layer',
                data: data,
                iconAtlas: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="48" height="48" viewBox="0 0 48 48"><g fill="white"><path d="M24 4 L27 24 L24 27 L21 24 Z"/><path d="M42 14 L27 21 L24 18 L39 11 Z"/><path d="M6 14 L21 21 L24 18 L9 11 Z"/></g></svg>',
                iconMapping: {{ 'blade': {{ x: 0, y: 0, width: 48, height: 48, anchorY: 42, anchorX: 24 }} }},
                getIcon: d => 'blade',
                getPosition: d => d.position,
                getSize: d => Math.max(20, Math.min(64, (d.radius || 300) / 15)),
                getAngle: d => 0,
                getColor: d => {{
                    if (d.phase === 'construction') return [255, 70, 40, 220];
                    if (d.phase === 'operational') return [50, 160, 240, 220];
                    if (d.phase === 'planned') return [180, 180, 180, 180];
                    return d.color || [255, 140, 60];
                }},
                pickable: false,
                opacity: 0.95,
            }});
        }}

        // Function to update all layers
        function updateAllLayers() {{
            deckgl.setProps({{
                layers: [
                    bathymetryLayer, 
                    createTurbinePoleLayer(turbineData),  // poles below blades
                    createTurbineBladeLayer(turbineData), // blades above poles
                    createScatterLayer(porpoiseData)
                ]
            }});
        }}

        // Initialize deck.gl with all layers
        const deckgl = new DeckGL({{
            container: 'deck-container',
            initialViewState: {{
                latitude: CENTER_LAT,
                longitude: CENTER_LON,
                zoom: {zoom},
                pitch: 0,
                bearing: 0
            }},
            controller: true,
            layers: [bathymetryLayer, createTurbinePoleLayer([]), createTurbineBladeLayer([]), createScatterLayer(porpoiseData)],
            parameters: {{
                clearColor: [0.05, 0.1, 0.15, 1]
            }},
            onViewStateChange: ({{viewState}}) => {{
                document.getElementById('view-info').innerHTML = 
                    'Lat: ' + viewState.latitude.toFixed(3) + 
                    ' | Lon: ' + viewState.longitude.toFixed(3) + 
                    ' | Zoom: ' + viewState.zoom.toFixed(1);
            }}
        }});
        
        // Store reference for updates (DEPONS pattern)
        window.deckgl = deckgl;
        window.bathymetryLayer = bathymetryLayer;
        
        // Function to update only the scatter layer (called when data changes)
        window.updatePorpoiseLayer = function(newData) {{
            porpoiseData = newData;
            updateAllLayers();
            document.getElementById('point-count').textContent = newData.length;
        }};
        
        // Function to update turbine layer
        window.updateTurbineLayer = function(newData) {{
            turbineData = newData;
            updateAllLayers();
            document.getElementById('turbine-count').textContent = newData.length;
        }};
        
        // Listen for postMessage updates from parent
        window.addEventListener('message', function(event) {{
            if (event.data && event.data.type === 'setTurbineData') {{
                console.log('Received turbine data:', event.data.data.length, 'turbines');
                window.updateTurbineLayer(event.data.data);
            }}
            if (event.data && event.data.type === 'setPorpoiseData') {{
                console.log('Received porpoise data:', event.data.data.length, 'porpoises');
                window.updatePorpoiseLayer(event.data.data);
            }}
        }});
        
        // Initial view info
        document.getElementById('view-info').innerHTML = 
            'Lat: ' + CENTER_LAT.toFixed(3) + 
            ' | Lon: ' + CENTER_LON.toFixed(3) + 
            ' | Zoom: {zoom}.0';
    </script>
</body>
</html>
'''
    # Use a fixed name attribute to help prevent iframe recreation flicker
    result = ui.tags.iframe(
        srcdoc=html_content,
        name=f"porpoise-map-{map_id}",
        style=f"width: 100%; height: {height}; min-height: 500px; border: none; border-radius: 8px;",
    )
    
    # Cache for next call with same data
    _cache[cache_key] = result
    # Limit cache size to prevent memory growth
    if len(_cache) > 20:
        oldest_key = next(iter(_cache))
        del _cache[oldest_key]
    
    return result


def no_data_placeholder(message: str = "No data yet. Run simulation to see results.") -> ui.Tag:
    """
    Create a standardized placeholder for empty charts.
    
    Args:
        message: Message to display
        
    Returns:
        Shiny paragraph element
    """
    return ui.p(message, class_="text-muted text-center mt-5")
