"""
Simple test script for debugging CENOP app
"""

import sys
print(f"Python: {sys.version}")

# Test 1: Check imports
print("\n=== Test 1: Imports ===")
try:
    from shiny import App, render, ui, reactive
    print("✓ Shiny imports OK")
except ImportError as e:
    print(f"✗ Shiny import failed: {e}")

try:
    from shinywidgets import output_widget, render_widget
    from ipyleaflet import Map, CircleMarker, LayerGroup, basemaps
    print("✓ shinywidgets/ipyleaflet imports OK")
except ImportError as e:
    print(f"✗ Widget import failed: {e}")

try:
    from cenop import Simulation, SimulationParameters
    from cenop.landscape import CellData, create_homogeneous_landscape
    from cenop.parameters import SimulationConstants
    print("✓ CENOP core imports OK")
except ImportError as e:
    print(f"✗ CENOP core import failed: {e}")

# Test 2: Create simulation
print("\n=== Test 2: Create Simulation ===")
try:
    params = SimulationParameters(
        porpoise_count=10,
        sim_years=1,
        landscape="Homogeneous"
    )
    print(f"✓ Parameters created: {params}")
    
    landscape = create_homogeneous_landscape()
    print(f"✓ Landscape created: {landscape.width}x{landscape.height}")
    
    sim = Simulation(params, cell_data=landscape)
    print(f"✓ Simulation created with {len(list(sim.agents))} agents")
except Exception as e:
    print(f"✗ Simulation creation failed: {e}")
    import traceback
    traceback.print_exc()

# Test 3: Run one step
print("\n=== Test 3: Simulation Step ===")
try:
    sim.step()
    agents = list(sim.agents)
    print(f"✓ Step completed. Agents: {len(agents)}")
    if agents:
        a = agents[0]
        print(f"  Agent 0: x={a.x:.2f}, y={a.y:.2f}, energy={getattr(a, 'energy_level', 'N/A')}")
except Exception as e:
    print(f"✗ Simulation step failed: {e}")
    import traceback
    traceback.print_exc()

# Test 4: Create a leaflet map
print("\n=== Test 4: Leaflet Map ===")
try:
    m = Map(center=(55.5, 4.0), zoom=6, basemap=basemaps.Esri.OceanBasemap)
    marker = CircleMarker(location=(55.5, 4.0), radius=5, color='green')
    layer = LayerGroup(layers=[marker])
    m.add(layer)
    print("✓ Leaflet map created with marker")
except Exception as e:
    print(f"✗ Leaflet map failed: {e}")
    import traceback
    traceback.print_exc()

# Test 5: Create plotly chart
print("\n=== Test 5: Plotly Chart ===")
try:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[1, 2, 3], y=[4, 5, 6], mode='lines', name='Test'))
    html = fig.to_html(include_plotlyjs="cdn", full_html=False)
    print(f"✓ Plotly chart created ({len(html)} chars)")
except Exception as e:
    print(f"✗ Plotly failed: {e}")
    import traceback
    traceback.print_exc()

print("\n=== All Tests Complete ===")
