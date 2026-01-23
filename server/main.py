"""
Main Server Function for CENOP Shiny App
CENOP - CETacean Noise-Population Model

This module contains the main server function and all render callbacks.
"""

from shiny import render, ui, reactive
import pandas as pd
import numpy as np
import logging
import threading
import queue

from server.reactive_state import SimulationState
from server.simulation_controller import create_simulation_from_inputs, SimulationRunner
from server.renderers.chart_helpers import (
    create_time_series_chart,
    create_histogram_chart,
    create_map_figure,
    create_pydeck_map,
    no_data_placeholder
)

logger = logging.getLogger("CENOP")


def run_simulation_loop(runner, result_queue, stop_event, throttle_value):
    """Background thread worker for simulation loop.
    
    Args:
        runner: SimulationRunner instance
        result_queue: Queue for sending updates to main thread
        stop_event: Threading event to signal stop
        throttle_value: List with single float [0.0-1.0] for speed control (mutable for thread sharing)
    """
    import time
    print(f"[DEBUG] run_simulation_loop STARTED - max_ticks={runner.max_ticks}")
    loop_count = 0
    try:
        while not runner.is_complete and not stop_event.is_set():
            loop_count += 1
            # Step one day
            entry = runner.step_day()
            
            if loop_count <= 5 or loop_count % 50 == 0:
                print(f"[DEBUG] Loop #{loop_count}: tick={runner.tick}, pop={entry.get('population', '?')}, year={entry.get('year', '?')}, speed={throttle_value[0]:.2f}")
            
            # Send update to main thread
            update = {
                "type": "update",
                "progress": runner.progress_percent,
                "entry": entry,
                "total_births": runner.total_births,
                "total_deaths": runner.total_deaths,
                "should_update_map": runner.should_update_map,
                "sim": runner.sim 
            }
            result_queue.put(update)
            
            # Dynamic sleep based on throttle value
            # throttle_value[0] is 0.0 (slowest) to 1.0 (fastest)
            # Use exponential scaling for more responsive control:
            # At 1.0 (100%): sleep = 0 (as fast as possible)
            # At 0.5 (50%): sleep = ~0.05
            # At 0.0 (1%): sleep = 0.3 (slow but not frozen)
            speed = throttle_value[0]
            if speed >= 0.99:
                sleep_time = 0  # Maximum speed - no delay
            else:
                # Exponential: slower at low speeds, faster at high speeds
                sleep_time = 0.3 * ((1.0 - speed) ** 2)
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
        if runner.is_complete:
            print(f"[DEBUG] Simulation COMPLETE after {loop_count} iterations, years={runner.sim.state.year}")
            result_queue.put({"type": "complete", "years": runner.sim.state.year})
            
    except Exception as e:
        print(f"[DEBUG] Simulation ERROR: {e}")
        import traceback
        traceback.print_exc()
        logger.error(f"Simulation error: {e}", exc_info=True)
        result_queue.put({"type": "error", "message": str(e)})


def server(input, output, session):
    """Main server function for CENOP Shiny app."""
    logger.info("Server function initialized")
    
    # Centralized reactive state
    state = SimulationState()
    
    # Internal state for background thread management
    sim_thread: threading.Thread | None = None
    stop_event = threading.Event()
    result_queue = queue.Queue()
    # Shared throttle value as a mutable list [0.0-1.0] for thread-safe updates
    # 0.0 = slowest (1%), 1.0 = fastest (100%)
    throttle_value = [0.49]  # Default 50%
    
    # =========================================================================
    # Help Modal
    # =========================================================================
    
    @reactive.effect
    @reactive.event(input.help_btn)
    def show_help_modal():
        """Show the help modal when help button is clicked."""
        from ui.layout import create_help_modal
        ui.modal_show(create_help_modal())
    
    # =========================================================================
    # Landscape Loading
    # =========================================================================
    
    @reactive.effect
    @reactive.event(input.load_landscape)
    def trigger_load_landscape():
        """Trigger landscape loading when button is clicked."""
        # Increment counter to trigger depth_data_initializer
        current = state.landscape_load_counter()
        state.landscape_load_counter.set(current + 1)
        state.landscape_loaded_name.set(input.landscape())
        print(f"[DEBUG] Load Landscape button clicked, loading '{input.landscape()}'")
    
    @render.text
    def landscape_status():
        """Show landscape loading status."""
        loaded_name = state.landscape_loaded_name()
        landscape_info = state.landscape_info()
        if loaded_name:
            if landscape_info:
                return f"✓ {loaded_name} ({landscape_info})"
            return f"✓ Loaded: {loaded_name}"
        return ""
    
    @render.ui
    def turbine_selector():
        """Render turbine selector filtered by landscape compatibility."""
        from shiny import ui as shiny_ui
        
        # Landscape-turbine compatibility mapping with descriptions
        # Only landscapes with available data are included
        # NorthSea scenarios differ in construction timing, not turbine count
        LANDSCAPE_TURBINE_COMPATIBILITY = {
            "Homogeneous": {"off": "No turbines"},
            "NorthSea": {
                "off": "No turbines",
                "NorthSea_scenario1": "Scenario 1",
                "NorthSea_scenario2": "Scenario 2",
                "NorthSea_scenario3": "Scenario 3"
            },
            "UserDefined": {
                "off": "No turbines",
                "User-def": "User Defined Scenario"
            },
            "CentralBaltic": {
                "off": "No turbines"
            },
        }
        
        landscape = input.landscape()
        # Get compatible turbines for selected landscape
        compatible = LANDSCAPE_TURBINE_COMPATIBILITY.get(landscape, {"off": "No turbines"})
        
        return shiny_ui.input_select(
            "turbines", 
            "Wind Turbines", 
            choices=compatible, 
            selected=list(compatible.keys())[0] if compatible else "off"
        )
    
    @reactive.effect
    @reactive.event(input.load_turbines)
    def trigger_load_turbines():
        """Trigger turbine loading when button is clicked."""
        # Increment counter to trigger turbine and noise data initializers
        current = state.turbine_load_counter()
        state.turbine_load_counter.set(current + 1)
        state.turbine_loaded_name.set(input.turbines())
        print(f"[DEBUG] Load Turbines button clicked, loading '{input.turbines()}'")
    
    @render.text
    def turbine_status():
        """Show turbine loading status."""
        loaded_name = state.turbine_loaded_name()
        turbine_count = state.turbine_count()
        if loaded_name and loaded_name != "off":
            if turbine_count > 0:
                return f"✓ {turbine_count} turbines loaded"
            return f"✓ Loaded: {loaded_name}"
        elif loaded_name == "off":
            return ""
        return ""
    
    # =========================================================================
    # Simulation Control Effects
    # =========================================================================
    
    @reactive.effect
    @reactive.event(input.run_sim)
    def start_simulation():
        """Start the simulation in a background thread."""
        nonlocal sim_thread
        print("[DEBUG] start_simulation() TRIGGERED")
        if state.running():
            print("[DEBUG] Already running, skipping")
            return
        
        print("[DEBUG] Creating simulation from inputs...")
        sim = create_simulation_from_inputs(input)
        print(f"[DEBUG] Simulation created, is_initialized={sim._is_initialized}")
        sim.initialize()
        print(f"[DEBUG] Simulation initialized, pop_size={sim.population_size}, max_ticks={sim.max_ticks}")
        
        runner = SimulationRunner(sim)
        print(f"[DEBUG] SimulationRunner created, max_ticks={runner.max_ticks}")
        
        # Reset queue and event
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
        stop_event.clear()
        
        state.simulation.set(sim)
        state.running.set(True)
        state.population_history.set([])
        state.birth_count.set(0)
        state.death_count.set(0)
        state.progress.set(0.0)
        state.progress_message.set("Running simulation...")
        
        # Update throttle from current slider value
        speed_percent = input.sim_speed()
        throttle_value[0] = (speed_percent - 1) / 99.0  # Convert 1-100 to 0.0-1.0
        
        # Start background thread
        sim_thread = threading.Thread(
            target=run_simulation_loop,
            args=(runner, result_queue, stop_event, throttle_value),
            daemon=True
        )
        sim_thread.start()
        
        # Start polling immediately
        reactive.invalidate_later(0.1)
    
    @reactive.effect
    @reactive.event(input.sim_speed)
    def update_throttle():
        """Update throttle value when slider changes during simulation."""
        speed_percent = input.sim_speed()
        new_throttle = (speed_percent - 1) / 99.0  # Convert 1-100 to 0.0-1.0
        throttle_value[0] = new_throttle
        print(f"[DEBUG] Speed slider changed: {speed_percent}% -> throttle={new_throttle:.3f}")
    
    @reactive.effect
    def poll_simulation():
        """Poll for updates from the background simulation thread."""
        running = state.running()
        if not running:
            return
        
        # Poll frequently to keep up with simulation
        reactive.invalidate_later(0.05)
        
        try:
            # Process multiple updates per poll to keep up
            has_updates = False
            last_sim_ref = None
            msgs_processed = 0
            entries_batch = []
            
            # Drain queue - process all available messages
            while True:
                msg = result_queue.get_nowait()
                has_updates = True
                msgs_processed += 1
                
                if msg["type"] == "error":
                    state.running.set(False)
                    ui.notification_show(f"Error: {msg['message']}", type="error")
                    return
                
                if msg["type"] == "complete":
                    state.running.set(False)
                    state.progress.set(100.0)
                    state.progress_message.set(f"Complete! {msg['years']} years simulated")
                    # Final history update
                    if entries_batch:
                        current_hist = state.population_history()
                        state.population_history.set(current_hist + entries_batch)
                    print(f"[DEBUG] Poll: Simulation COMPLETE, final history len={len(state.population_history())}")
                    return
                
                if msg["type"] == "update":
                    state.progress.set(msg["progress"])
                    state.birth_count.set(msg["total_births"])
                    state.death_count.set(msg["total_deaths"])
                    
                    entry = msg["entry"]
                    state.progress_message.set(f"Year {entry['year']}, Day {entry['day'] % 360}")
                    
                    # Batch entries instead of updating one-by-one
                    entries_batch.append(entry)
                    
                    if msg["should_update_map"]:
                        state.map_update_counter.set(state.map_update_counter() + 1)
                    
                    last_sim_ref = msg["sim"]
            
            # Batch update history once per poll (not per message)
            if entries_batch:
                current_hist = state.population_history()
                state.population_history.set(current_hist + entries_batch)
                if len(current_hist) == 0 or len(current_hist) % 100 == 0:
                    print(f"[DEBUG] Poll: added {len(entries_batch)} entries, total history len={len(state.population_history())}")
            
            if last_sim_ref:
                state.simulation.set(last_sim_ref)
                
        except queue.Empty:
            pass
    
    @reactive.effect
    @reactive.event(input.stop_sim)
    def stop_simulation():
        """Stop the running simulation."""
        stop_event.set()
        state.running.set(False)

    
    @reactive.effect
    @reactive.event(input.reset_sim)
    def reset_simulation():
        """Reset the simulation."""
        stop_event.set()
        # Clear queue to release refs if thread stopped
        while not result_queue.empty():
            try:
                result_queue.get_nowait()
            except queue.Empty:
                break
        state.reset()
    
    # =========================================================================
    # Progress Renderers
    # =========================================================================
    
    @render.ui
    def progress_bar():
        pct = state.progress()
        is_running = state.running()
        
        if pct >= 100:
            color = "#198754"  # green
        elif is_running:
            color = "#0d6efd"  # blue
        else:
            color = "#6c757d"  # gray
            
        return ui.div(
            ui.div(
                style=f"width: {pct}%; height: 24px; background-color: {color}; border-radius: 4px; transition: width 0.3s;"
            ),
            style="width: 100%; height: 24px; background-color: #e9ecef; border-radius: 4px; overflow: hidden;"
        )
    
    @render.text
    def progress_text():
        return state.progress_message()
    
    # =========================================================================
    # Value Box Renderers
    # =========================================================================
    
    @render.text
    def current_population():
        # Use history for reactive updates
        history = state.population_history()
        if history:
            return str(history[-1].get('population', 0))
        sim = state.simulation()
        return str(sim.state.population if sim else 0)
    
    @render.text
    def current_year():
        # Use history for reactive updates
        history = state.population_history()
        if history:
            return str(history[-1].get('year', 0))
        sim = state.simulation()
        return str(sim.state.year if sim else 0)
    
    @render.text
    def total_births():
        # Trigger on history updates
        _ = state.population_history()
        return str(state.birth_count())
    
    @render.text
    def total_deaths():
        # Trigger on history updates
        _ = state.population_history()
        return str(state.death_count())
    
    # =========================================================================
    # Dashboard Chart Renderers
    # =========================================================================
    
    @render.ui
    def population_plot():
        """Porpoise Population Size chart."""
        history = state.population_history()
        hist_len = len(history) if history else 0
        # Only log occasionally to reduce spam
        if hist_len == 0 or hist_len % 50 == 0:
            print(f"[DEBUG] population_plot: history length={hist_len}")
        if not history:
            return no_data_placeholder()
        
        df = pd.DataFrame(history)
        if 'tick' not in df.columns or 'population' not in df.columns:
            print(f"[DEBUG] population_plot: MISSING COLUMNS! Available: {list(df.columns)}")
            return no_data_placeholder("Missing required data columns")
        
        result = create_time_series_chart(
            df=df,
            x_col='tick',
            y_cols=['population', 'lact_calf'],
            colors=['blue', 'red'],
            names=['Total Count', 'Lactating + Calf'],
            title='Porpoise Population Size',
            x_title='Tick Count',
            y_title='Population Size',
            height=180
        )
        return result
    
    @render.ui
    def life_death_plot():
        """Life and Death chart."""
        history = state.population_history()
        if not history:
            return no_data_placeholder()
        
        df = pd.DataFrame(history)
        df['daily_births'] = df['births'].diff().fillna(0)
        df['daily_deaths'] = df['deaths'].diff().fillna(0)
        
        return create_time_series_chart(
            df=df,
            x_col='tick',
            y_cols=['daily_births', 'daily_deaths'],
            colors=['blue', 'red'],
            names=['Births', 'Deaths'],
            title='Life and Death',
            x_title='Tick Count',
            y_title='Count',
            height=180
        )
    
    @render.ui
    def energy_balance_plot():
        """Food consumption and expenditure chart."""
        history = state.energy_history()
        if not history:
            return no_data_placeholder("No energy data yet.")
        
        df = pd.DataFrame(history)
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['avg_food_eaten', 'avg_energy_expended'],
            colors=['blue', 'red'],
            names=['Avg Food Eaten', 'Avg Energy Expended'],
            title='Food Consumption and Expenditure',
            x_title='Day',
            y_title='Energy',
            height=180
        )
    
    # Cache for depth data - will be updated when Load Landscape is clicked or simulation starts
    _depth_data_cache = None
    _depth_landscape_name = None  # Track which landscape is cached
    
    @render.ui
    def depth_data_initializer():
        """
        Hidden output that sends depth data to the map.
        Updates when Load Landscape button is clicked or simulation starts.
        The landscape depth grid is shown as a static overlay on the map.
        """
        import json
        nonlocal _depth_data_cache, _depth_landscape_name
        
        # React to load landscape button clicks
        load_counter = state.landscape_load_counter()
        
        # Also react to simulation state changes (when sim starts)
        sim = state.simulation()
        
        # Get the landscape to load
        landscape_name = state.landscape_loaded_name() if state.landscape_loaded_name() else input.landscape()
        
        # Use simulation's landscape if available
        if sim is not None and hasattr(sim, 'landscape') and sim.landscape is not None:
            landscape_name = getattr(sim.landscape, 'landscape_name', landscape_name)
        
        # Only load if button was clicked or sim started (not just on dropdown change)
        if load_counter == 0 and sim is None:
            # No load yet - return empty
            return ui.div()
        
        # Recompute if landscape changed
        if _depth_data_cache is None or _depth_landscape_name != landscape_name:
            try:
                from cenop.landscape import CellData, create_homogeneous_landscape, create_landscape_from_depons
                
                print(f"[DEBUG] Loading landscape '{landscape_name}' for depth overlay...")
                
                # Create landscape matching the simulation
                if landscape_name == "Homogeneous":
                    landscape = create_homogeneous_landscape()
                elif landscape_name == "NorthSea":
                    # NorthSea uses DEPONS data files directly
                    landscape = create_landscape_from_depons()
                else:
                    # Other named landscapes (CentralBaltic, InnerDanishWaters, etc.)
                    landscape = CellData(landscape_name)
                    landscape.load()  # Explicitly load data
                
                depth_array = landscape._depth
                if depth_array is not None:
                    # Sample depth data (every Nth cell to reduce data size)
                    # 400x400 = 160,000 cells is too many, sample every 5th = 6,400 cells
                    sample_step = 5
                    depth_array = landscape._depth
                    grid_height, grid_width = depth_array.shape
                    
                    # Get landscape-specific bounds
                    from ui.sidebar import LANDSCAPE_BOUNDS
                    bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
                    lat_min, lat_max, lon_min, lon_max = bounds
                    
                    depth_points = []
                    for row in range(0, grid_height, sample_step):
                        for col in range(0, grid_width, sample_step):
                            depth = float(depth_array[row, col])
                            # Convert grid to lat/lon
                            # Array is flipped during loading: row 0 = SOUTH, row max = NORTH
                            lat = lat_min + (row / grid_height) * (lat_max - lat_min)
                            lon = lon_min + (col / grid_width) * (lon_max - lon_min)
                            depth_points.append({
                                "position": [lon, lat],
                                "depth": depth
                            })
                    
                    _depth_data_cache = {
                        "points": depth_points,
                        "width": grid_width,
                        "height": grid_height
                    }
                    _depth_landscape_name = landscape_name
                    state.landscape_info.set(f"{grid_width}x{grid_height} grid")
                    print(f"[DEBUG] Depth data cached for '{landscape_name}': {len(depth_points)} points from {grid_width}x{grid_height} grid")
                else:
                    _depth_data_cache = {"points": [], "width": 400, "height": 400}
                    _depth_landscape_name = landscape_name
            except Exception as e:
                print(f"[DEBUG] Error loading depth data for '{landscape_name}': {e}")
                import traceback
                traceback.print_exc()
                _depth_data_cache = {"points": [], "width": 400, "height": 400}
                _depth_landscape_name = landscape_name
        
        if not _depth_data_cache["points"]:
            return ui.div()
        
        depth_json = json.dumps(_depth_data_cache["points"])
        
        # Also send the landscape bounds to update map center
        from ui.sidebar import LANDSCAPE_BOUNDS
        bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
        lat_min, lat_max, lon_min, lon_max = bounds
        
        js_code = f'''
        <script>
            (function() {{
                // Wait for iframe to be ready
                function sendDepthData() {{
                    var iframe = document.getElementById('porpoise-map-frame');
                    if (iframe && iframe.contentWindow) {{
                        // First update the landscape bounds
                        iframe.contentWindow.postMessage({{
                            type: 'setLandscapeBounds',
                            latMin: {lat_min},
                            latMax: {lat_max},
                            lonMin: {lon_min},
                            lonMax: {lon_max}
                        }}, '*');
                        console.log('Landscape bounds sent:', {{latMin: {lat_min}, latMax: {lat_max}, lonMin: {lon_min}, lonMax: {lon_max}}});
                        
                        // Then send depth data
                        var data = {depth_json};
                        iframe.contentWindow.postMessage({{
                            type: 'setDepthData',
                            data: data,
                            gridWidth: {_depth_data_cache["width"]},
                            gridHeight: {_depth_data_cache["height"]}
                        }}, '*');
                        console.log('Depth data sent to map:', data.length, 'points');
                    }} else {{
                        setTimeout(sendDepthData, 100);
                    }}
                }}
                // Small delay to ensure iframe is loaded
                setTimeout(sendDepthData, 500);
            }})();
        </script>
        '''
        return ui.HTML(js_code)
    
    # Cache for foraging data
    _foraging_data_cache = None
    _foraging_landscape_name = None
    
    @render.ui
    def foraging_data_initializer():
        """
        Hidden output that sends foraging/food data to the map.
        Shows food probability (patches) as a green overlay.
        Updates when Load Landscape button is clicked.
        """
        import json
        nonlocal _foraging_data_cache, _foraging_landscape_name
        
        # React to load landscape button clicks
        load_counter = state.landscape_load_counter()
        
        # Also react to simulation state changes (when sim starts)
        sim = state.simulation()
        
        # Get the landscape to load
        landscape_name = state.landscape_loaded_name() if state.landscape_loaded_name() else input.landscape()
        
        # Use simulation's landscape if available
        if sim is not None and hasattr(sim, 'landscape') and sim.landscape is not None:
            landscape_name = getattr(sim.landscape, 'landscape_name', landscape_name)
        
        # Only load if button was clicked or sim started
        if load_counter == 0 and sim is None:
            return ui.div()
        
        # Recompute if landscape changed
        if _foraging_data_cache is None or _foraging_landscape_name != landscape_name:
            try:
                from cenop.landscape import CellData, create_homogeneous_landscape, create_landscape_from_depons
                
                print(f"[DEBUG] Loading foraging data for '{landscape_name}'...")
                
                # Create landscape matching the simulation
                if landscape_name == "Homogeneous":
                    landscape = create_homogeneous_landscape()
                elif landscape_name == "NorthSea":
                    landscape = create_landscape_from_depons()
                else:
                    landscape = CellData(landscape_name)
                    landscape.load()
                
                food_array = landscape._food_prob
                if food_array is not None:
                    # Sample food data (every Nth cell to reduce data size)
                    sample_step = 5
                    grid_height, grid_width = food_array.shape
                    
                    # Get landscape-specific bounds
                    from ui.sidebar import LANDSCAPE_BOUNDS
                    bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
                    lat_min, lat_max, lon_min, lon_max = bounds
                    
                    food_points = []
                    for row in range(0, grid_height, sample_step):
                        for col in range(0, grid_width, sample_step):
                            food = float(food_array[row, col])
                            if food > 0.1:  # Only include cells with significant food
                                lat = lat_min + (row / grid_height) * (lat_max - lat_min)
                                lon = lon_min + (col / grid_width) * (lon_max - lon_min)
                                food_points.append({
                                    "position": [lon, lat],
                                    "food": food
                                })
                    
                    _foraging_data_cache = food_points
                    _foraging_landscape_name = landscape_name
                    print(f"[DEBUG] Foraging data cached for '{landscape_name}': {len(food_points)} food cells")
                else:
                    _foraging_data_cache = []
                    _foraging_landscape_name = landscape_name
            except Exception as e:
                print(f"[DEBUG] Error loading foraging data for '{landscape_name}': {e}")
                import traceback
                traceback.print_exc()
                _foraging_data_cache = []
                _foraging_landscape_name = landscape_name
        
        if not _foraging_data_cache:
            return ui.div()
        
        foraging_json = json.dumps(_foraging_data_cache)
        
        js_code = f'''
        <script>
            (function() {{
                function sendForagingData() {{
                    var iframe = document.getElementById('porpoise-map-frame');
                    if (iframe && iframe.contentWindow) {{
                        var data = {foraging_json};
                        iframe.contentWindow.postMessage({{
                            type: 'setForagingData',
                            data: data
                        }}, '*');
                        console.log('Foraging data sent to map:', data.length, 'cells');
                    }} else {{
                        setTimeout(sendForagingData, 100);
                    }}
                }}
                setTimeout(sendForagingData, 600);  // After depth data
            }})();
        </script>
        '''
        return ui.HTML(js_code)
    
    # =========================================================================
    # Ship Data Layer
    # =========================================================================
    
    @render.ui
    def ship_data_initializer():
        """
        Hidden output that sends ship traffic data to the map.
        Shows ships as moving markers when Ship Traffic is enabled.
        Updates during simulation to show ship movement.
        """
        import json
        
        # React to simulation state changes
        sim = state.simulation()
        
        # Also react to map updates during simulation
        map_counter = state.map_update_counter()
        
        # Check if ships are enabled
        ships_enabled_val = input.ships_enabled() if hasattr(input, 'ships_enabled') else False
        
        # If no simulation or ships not enabled, send empty data
        if sim is None or not ships_enabled_val:
            return ui.HTML('''
            <script>
                (function() {
                    function clearShips() {
                        var iframe = document.getElementById('porpoise-map-frame');
                        if (iframe && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({
                                type: 'setShipData',
                                data: []
                            }, '*');
                        }
                    }
                    setTimeout(clearShips, 500);
                })();
            </script>
            ''')
        
        try:
            # Get ship data from simulation
            ship_manager = getattr(sim, '_ship_manager', None)
            if ship_manager is None or len(ship_manager.get_all_ships()) == 0:
                return ui.div()
            
            ships = ship_manager.get_all_ships()
            
            # Get landscape-specific bounds
            from ui.sidebar import LANDSCAPE_BOUNDS
            landscape_name = state.landscape_loaded_name() if state.landscape_loaded_name() else input.landscape()
            bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
            lat_min, lat_max, lon_min, lon_max = bounds
            
            # Get landscape grid dimensions
            grid_width = 400
            grid_height = 400
            if hasattr(sim, 'landscape') and sim.landscape is not None:
                grid_width = getattr(sim.landscape, 'width', 400)
                grid_height = getattr(sim.landscape, 'height', 400)
            
            ship_points = []
            for ship in ships:
                # Get ship position (grid coordinates)
                x, y = ship.x, ship.y
                
                # Convert grid to lat/lon
                lat = lat_min + (y / grid_height) * (lat_max - lat_min)
                lon = lon_min + (x / grid_width) * (lon_max - lon_min)
                
                ship_points.append({
                    "position": [lon, lat],
                    "name": ship.name,
                    "speed": ship.speed,
                    "size": ship.vessel_class.value if hasattr(ship.vessel_class, 'value') else 1
                })
            
            print(f"[DEBUG] Sending {len(ship_points)} ships to map (update #{map_counter})")
            
            if not ship_points:
                return ui.div()
            
            ship_json = json.dumps(ship_points)
            
            js_code = f'''
            <script>
                (function() {{
                    function sendShipData() {{
                        var iframe = document.getElementById('porpoise-map-frame');
                        if (iframe && iframe.contentWindow) {{
                            var data = {ship_json};
                            iframe.contentWindow.postMessage({{
                                type: 'setShipData',
                                data: data
                            }}, '*');
                            console.log('Ship data sent to map:', data.length, 'vessels');
                        }} else {{
                            setTimeout(sendShipData, 100);
                        }}
                    }}
                    setTimeout(sendShipData, 700);  // After other layers
                }})();
            </script>
            '''
            return ui.HTML(js_code)
            
        except Exception as e:
            print(f"[DEBUG] Error loading ship data: {e}")
            import traceback
            traceback.print_exc()
            return ui.div()
    
    # Cache for turbine data
    _turbine_data_cache = None
    _turbine_scenario_name = None
    
    @render.ui
    def turbine_data_initializer():
        """
        Hidden output that sends turbine data to the map.
        Updates when Load Turbines button is clicked.
        """
        import json
        import os
        from pathlib import Path
        nonlocal _turbine_data_cache, _turbine_scenario_name
        
        # React to Load Turbines button clicks
        turbine_counter = state.turbine_load_counter()
        turbine_scenario = input.turbines() if hasattr(input, 'turbines') else "off"
        
        # Skip if no turbines or button not clicked
        if turbine_scenario == "off" or turbine_counter == 0:
            _turbine_data_cache = None
            _turbine_scenario_name = turbine_scenario
            state.turbine_count.set(0)
            # Send empty turbine data to clear the layer
            return ui.HTML('''
            <script>
                (function() {
                    function clearTurbines() {
                        var iframe = document.getElementById('porpoise-map-frame');
                        if (iframe && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({
                                type: 'setTurbineData',
                                data: []
                            }, '*');
                        }
                    }
                    setTimeout(clearTurbines, 500);
                })();
            </script>
            ''')
        
        # Only reload if scenario changed
        if _turbine_data_cache is not None and _turbine_scenario_name == turbine_scenario:
            turbine_json = json.dumps(_turbine_data_cache)
        else:
            # Load turbine data from file
            try:
                from pyproj import Transformer
                
                # Find the wind-farms directory
                possible_paths = [
                    Path("../DEPONS-master/data/wind-farms"),
                    Path("../../DEPONS-master/data/wind-farms"),
                    Path("DEPONS-master/data/wind-farms"),
                    Path("data/wind-farms"),
                ]
                wind_farms_dir = None
                for p in possible_paths:
                    if p.exists():
                        wind_farms_dir = p
                        break
                
                if wind_farms_dir is None:
                    print(f"[DEBUG] Wind farms directory not found")
                    return ui.div()
                
                turbine_file = wind_farms_dir / f"{turbine_scenario}.txt"
                if not turbine_file.exists():
                    print(f"[DEBUG] Turbine file not found: {turbine_file}")
                    return ui.div()
                
                print(f"[DEBUG] Loading turbines from {turbine_file}")
                
                # Transform from EPSG:3035 to WGS84
                transformer = Transformer.from_crs('EPSG:3035', 'EPSG:4326', always_xy=True)
                
                turbines = []
                with open(turbine_file, 'r') as f:
                    header = f.readline().strip().split('\t')
                    # Handle different column names
                    x_col = 'x' if 'x' in header else 'x.coordinate'
                    y_col = 'y' if 'y' in header else 'y.coordinate'
                    x_idx = header.index(x_col)
                    y_idx = header.index(y_col)
                    id_idx = header.index('id')
                    impact_idx = header.index('impact') if 'impact' in header else None
                    start_idx = header.index('start') if 'start' in header else None
                    end_idx = header.index('end') if 'end' in header else None
                    
                    for line in f:
                        parts = line.strip().split('\t')
                        if len(parts) < max(x_idx, y_idx) + 1:
                            continue
                        
                        try:
                            x_3035 = float(parts[x_idx])
                            y_3035 = float(parts[y_idx])
                            turbine_id = parts[id_idx]
                            impact = int(float(parts[impact_idx])) if impact_idx else 200
                            # Construction start/end ticks (pile-driving phase)
                            start_tick = int(float(parts[start_idx])) if start_idx else 0
                            end_tick = int(float(parts[end_idx])) if end_idx else start_tick + 4
                            
                            # Transform to lat/lon
                            lon, lat = transformer.transform(x_3035, y_3035)
                            
                            turbines.append({
                                "id": turbine_id,
                                "position": [lon, lat],
                                "impact": impact,  # Construction noise: 234 dB SEL
                                "start": start_tick,  # Pile-driving start tick
                                "end": end_tick,  # Pile-driving end tick
                                "radius": 600,  # Turbine marker size
                                "color": [255, 100, 50, 220]  # Orange-red for turbines
                            })
                        except (ValueError, IndexError) as e:
                            continue
                
                _turbine_data_cache = turbines
                _turbine_scenario_name = turbine_scenario
                state.turbine_count.set(len(turbines))
                print(f"[DEBUG] Loaded {len(turbines)} turbines from {turbine_scenario}")
                
            except Exception as e:
                print(f"[DEBUG] Error loading turbine data: {e}")
                import traceback
                traceback.print_exc()
                _turbine_data_cache = []
                _turbine_scenario_name = turbine_scenario
        
        if not _turbine_data_cache:
            return ui.div()
        
        turbine_json = json.dumps(_turbine_data_cache)
        turbine_count = len(_turbine_data_cache)
        
        js_code = f'''
        <script>
            (function() {{
                function sendTurbineData() {{
                    var iframe = document.getElementById('porpoise-map-frame');
                    if (iframe && iframe.contentWindow) {{
                        var data = {turbine_json};
                        iframe.contentWindow.postMessage({{
                            type: 'setTurbineData',
                            data: data,
                            count: {turbine_count},
                            scenario: '{turbine_scenario}'
                        }}, '*');
                        console.log('Turbine data sent to map:', data.length, 'turbines');
                    }} else {{
                        setTimeout(sendTurbineData, 100);
                    }}
                }}
                setTimeout(sendTurbineData, 600);
            }})();
        </script>
        '''
        return ui.HTML(js_code)
    
    # Cache for noise propagation data
    _noise_data_cache = None
    _noise_scenario_name = None
    _noise_tick = None  # Track tick for dynamic updates
    
    # Operational noise level (dB SEL) - much lower than construction
    OPERATIONAL_NOISE_LEVEL = 145.0  # Typical operational turbine noise
    
    @render.ui
    def noise_data_initializer():
        """
        Hidden output that sends noise propagation data to the map.
        
        Two types of noise:
        1. Construction noise (red): 234 dB during pile-driving (start <= tick <= end)
        2. Operational noise (yellow): ~145 dB after construction (tick > end)
        
        Before simulation: Shows operational noise for all turbines (preview mode)
        During simulation: Shows construction + operational based on current tick
        """
        import json
        import numpy as np
        
        nonlocal _noise_data_cache, _noise_scenario_name, _noise_tick
        
        # Depend on turbine load button clicks
        turbine_scenario = input.turbines() if hasattr(input, 'turbines') else "off"
        turbine_counter = state.turbine_load_counter()
        
        # Also react to simulation state for tick-based updates
        sim = state.simulation()
        map_counter = state.map_update_counter()  # Triggers on map updates
        
        # Get current tick from simulation
        current_tick = 0
        if sim is not None and hasattr(sim, 'state') and hasattr(sim.state, 'tick'):
            current_tick = sim.state.tick
        
        # Only show noise when turbines are loaded (not "off" and button clicked)
        if not turbine_scenario or turbine_scenario == "off" or turbine_counter == 0:
            _noise_data_cache = {"construction": [], "operational": []}
            _noise_scenario_name = turbine_scenario
            _noise_tick = current_tick
            # Clear noise layer when no turbines
            return ui.HTML('''
            <script>
                (function() {
                    function clearNoise() {
                        var iframe = document.getElementById('porpoise-map-frame');
                        if (iframe && iframe.contentWindow) {
                            iframe.contentWindow.postMessage({
                                type: 'setNoiseData',
                                data: {construction: [], operational: []}
                            }, '*');
                        } else {
                            setTimeout(clearNoise, 100);
                        }
                    }
                    setTimeout(clearNoise, 300);
                })();
            </script>
            ''')
        
        # Recalculate if scenario changed or tick changed significantly
        tick_changed = _noise_tick is None or abs(current_tick - _noise_tick) >= 48  # Update every ~day
        scenario_changed = _noise_scenario_name != turbine_scenario
        
        if scenario_changed or _noise_data_cache is None or tick_changed:
            try:
                # Get turbine data first
                if _turbine_data_cache is None or len(_turbine_data_cache) == 0:
                    _noise_data_cache = {"construction": [], "operational": []}
                    _noise_scenario_name = turbine_scenario
                    _noise_tick = current_tick
                    return ui.div()
                
                # DEPONS noise propagation parameters
                beta_hat = 20.0  # Spreading loss factor
                alpha_hat = 0.0  # Absorption coefficient
                deter_threshold = 158.0  # RT: deterrence threshold (dB)
                operational_threshold = 140.0  # Lower threshold for operational display
                
                # Grid parameters
                grid_step = 5
                grid_width = 400
                grid_height = 400
                
                # Get landscape-specific bounds
                from ui.sidebar import LANDSCAPE_BOUNDS
                landscape_name = state.landscape_loaded_name() if state.landscape_loaded_name() else input.landscape()
                bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
                lat_min, lat_max, lon_min, lon_max = bounds
                
                # Separate turbines by phase based on current tick
                constructing_turbines = []  # Currently pile-driving
                operational_turbines = []   # Already built (tick > end)
                
                for t in _turbine_data_cache:
                    start = t.get('start', 0)
                    end = t.get('end', start + 4)
                    
                    if current_tick == 0:
                        # Preview mode (no simulation): show all as operational
                        operational_turbines.append(t)
                    elif start <= current_tick <= end:
                        # Currently constructing (pile-driving)
                        constructing_turbines.append(t)
                    elif current_tick > end:
                        # Construction complete, now operational
                        operational_turbines.append(t)
                    # If tick < start, turbine not yet built - no noise
                
                print(f"[DEBUG] Tick {current_tick}: {len(constructing_turbines)} constructing, {len(operational_turbines)} operational")
                
                # Calculate construction noise (high impact, red)
                construction_points = []
                if constructing_turbines:
                    c_lons = np.array([t['position'][0] for t in constructing_turbines])
                    c_lats = np.array([t['position'][1] for t in constructing_turbines])
                    c_impacts = np.array([t.get('impact', 234) for t in constructing_turbines])
                    
                    for row in range(0, grid_height, grid_step):
                        for col in range(0, grid_width, grid_step):
                            lat = lat_min + (row / grid_height) * (lat_max - lat_min)
                            lon = lon_min + (col / grid_width) * (lon_max - lon_min)
                            
                            lat_scale = 111000
                            lon_scale = 111000 * np.cos(np.radians(lat))
                            
                            dlat = (lat - c_lats) * lat_scale
                            dlon = (lon - c_lons) * lon_scale
                            distances = np.maximum(np.sqrt(dlat**2 + dlon**2), 1.0)
                            
                            transmission_loss = beta_hat * np.log10(distances) + alpha_hat * distances
                            received_levels = c_impacts - transmission_loss
                            max_rl = np.max(received_levels)
                            
                            if max_rl > deter_threshold:
                                construction_points.append({
                                    "position": [float(lon), float(lat)],
                                    "level": float(max_rl),
                                    "type": "construction"
                                })
                
                # Calculate operational noise (low impact, yellow)
                operational_points = []
                if operational_turbines:
                    o_lons = np.array([t['position'][0] for t in operational_turbines])
                    o_lats = np.array([t['position'][1] for t in operational_turbines])
                    # Operational noise is much lower - typically 145 dB
                    o_impacts = np.full(len(operational_turbines), OPERATIONAL_NOISE_LEVEL)
                    
                    for row in range(0, grid_height, grid_step):
                        for col in range(0, grid_width, grid_step):
                            lat = lat_min + (row / grid_height) * (lat_max - lat_min)
                            lon = lon_min + (col / grid_width) * (lon_max - lon_min)
                            
                            lat_scale = 111000
                            lon_scale = 111000 * np.cos(np.radians(lat))
                            
                            dlat = (lat - o_lats) * lat_scale
                            dlon = (lon - o_lons) * lon_scale
                            distances = np.maximum(np.sqrt(dlat**2 + dlon**2), 1.0)
                            
                            transmission_loss = beta_hat * np.log10(distances) + alpha_hat * distances
                            received_levels = o_impacts - transmission_loss
                            max_rl = np.max(received_levels)
                            
                            if max_rl > operational_threshold:
                                operational_points.append({
                                    "position": [float(lon), float(lat)],
                                    "level": float(max_rl),
                                    "type": "operational"
                                })
                
                _noise_data_cache = {
                    "construction": construction_points,
                    "operational": operational_points
                }
                _noise_scenario_name = turbine_scenario
                _noise_tick = current_tick
                print(f"[DEBUG] Noise calculated: {len(construction_points)} construction cells, {len(operational_points)} operational cells")
                
            except Exception as e:
                print(f"[DEBUG] Error calculating noise propagation: {e}")
                import traceback
                traceback.print_exc()
                _noise_data_cache = {"construction": [], "operational": []}
                _noise_scenario_name = turbine_scenario
                _noise_tick = current_tick
        
        if not _noise_data_cache or (not _noise_data_cache.get("construction") and not _noise_data_cache.get("operational")):
            return ui.div()
        
        noise_json = json.dumps(_noise_data_cache)
        
        js_code = f'''
        <script>
            (function() {{
                function sendNoiseData() {{
                    var iframe = document.getElementById('porpoise-map-frame');
                    if (iframe && iframe.contentWindow) {{
                        var data = {noise_json};
                        iframe.contentWindow.postMessage({{
                            type: 'setNoiseData',
                            data: data
                        }}, '*');
                        console.log('Noise data sent to map: construction=' + data.construction.length + ', operational=' + data.operational.length);
                    }} else {{
                        setTimeout(sendNoiseData, 100);
                    }}
                }}
                setTimeout(sendNoiseData, 700);
            }})();
        </script>
        '''
        return ui.HTML(js_code)
    
    @render.ui
    def porpoise_data_updater():
        """
        Hidden output that sends porpoise data to the static map via JavaScript.
        Following the DEPONS pattern where the map is created once and only
        the scatter layer is updated via deckgl.setProps().
        """
        import json
        
        map_counter = state.map_update_counter()  # Depend on counter for updates
        sim = state.simulation()
        
        # Get landscape-specific bounds
        from ui.sidebar import LANDSCAPE_BOUNDS
        landscape_name = state.landscape_loaded_name() if state.landscape_loaded_name() else input.landscape()
        bounds = LANDSCAPE_BOUNDS.get(landscape_name, (53.27, 54.79, 4.83, 7.13))
        lat_min, lat_max, lon_min, lon_max = bounds
        
        points_data = []
        
        if sim is not None:
            # Use new DataFrame access for performance (Vectorized Phase 3)
            if hasattr(sim, 'agents_df'):
                df = sim.agents_df
                if not df.empty:
                    # Limit to 1000 for plotting performance
                    df_plot = df.head(1000)
                    
                    # Get actual grid dimensions from landscape (via cell_data property)
                    if hasattr(sim, 'cell_data') and sim.cell_data is not None:
                        world_width = sim.cell_data.width
                        world_height = sim.cell_data.height
                    elif hasattr(sim, '_cell_data') and sim._cell_data is not None:
                        world_width = sim._cell_data.width
                        world_height = sim._cell_data.height
                    else:
                        world_width = getattr(sim.params, 'world_width', 400)
                        world_height = getattr(sim.params, 'world_height', 400)
                    
                    # Vectorized coordinate conversion
                    # DEPONS convention: y=0 is SOUTH (low lat), y=max is NORTH (high lat)
                    # So: lat = lat_min + (y/height) * range
                    lats = lat_min + (df_plot['y'] / world_height) * (lat_max - lat_min)
                    lons = lon_min + (df_plot['x'] / world_width) * (lon_max - lon_min)
                    
                    # Debug: show porpoise grid and latlon ranges
                    print(f"[DEBUG] Porpoise grid: x=[{df_plot['x'].min():.1f}-{df_plot['x'].max():.1f}], y=[{df_plot['y'].min():.1f}-{df_plot['y'].max():.1f}]")
                    print(f"[DEBUG] Porpoise latlon: lat=[{lats.min():.4f}-{lats.max():.4f}], lon=[{lons.min():.4f}-{lons.max():.4f}]")
                    print(f"[DEBUG] World size: {world_width}x{world_height}, lat bounds: {lat_min}-{lat_max}, lon bounds: {lon_min}-{lon_max}")
                    
                    for lat, lon in zip(lats, lons):
                        points_data.append({
                            "position": [float(lon), float(lat)],
                            "radius": 400,
                            "color": [0, 150, 255, 200]
                        })
            elif hasattr(sim, '_porpoises') and sim._porpoises:
                # Fallback for legacy
                if hasattr(sim, 'cell_data') and sim.cell_data is not None:
                    world_width = sim.cell_data.width
                    world_height = sim.cell_data.height
                elif hasattr(sim, '_cell_data') and sim._cell_data is not None:
                    world_width = sim._cell_data.width
                    world_height = sim._cell_data.height
                else:
                    world_width = getattr(sim.params, 'world_width', 400)
                    world_height = getattr(sim.params, 'world_height', 400)
                
                for p in sim._porpoises[:1000]:
                    if hasattr(p, 'alive') and p.alive:
                        # DEPONS convention: y=0 is SOUTH (low lat)
                        lat = lat_min + (p.y / world_height) * (lat_max - lat_min)
                        lon = lon_min + (p.x / world_width) * (lon_max - lon_min)
                        points_data.append({
                            "position": [lon, lat],
                            "radius": 400,
                            "color": [0, 150, 255, 200]
                        })
        
        points_json = json.dumps(points_data)
        
        # Return a script that sends data to the iframe via postMessage
        # This is the DEPONS pattern - update only the overlay, not the whole map
        js_code = f'''
        <script>
            (function() {{
                var iframe = document.getElementById('porpoise-map-frame');
                if (iframe && iframe.contentWindow) {{
                    var data = {points_json};
                    iframe.contentWindow.postMessage({{
                        type: 'updatePorpoises',
                        data: data
                    }}, '*');
                }}
            }})();
        </script>
        '''
        return ui.HTML(js_code)
    
    # =========================================================================
    # Population Tab Renderers
    # =========================================================================
    
    @render.ui
    def age_histogram():
        """Age distribution histogram."""
        sim = state.simulation()
        if sim is None:
            return no_data_placeholder("No data available.")
        
        if hasattr(sim, 'agents_df'):
            df = sim.agents_df
            if df.empty:
                return no_data_placeholder("No age data.")
            ages = df['age'].tolist()
        else:
            agents_list = list(sim.agents) if sim.agents else []
            ages = [a.age for a in agents_list if hasattr(a, 'age')]
        
        if not ages:
            return no_data_placeholder("No age data.")
        
        return create_histogram_chart(
            data=ages,
            title='Porpoise Age Distribution',
            x_title='Age (years)',
            y_title='Count',
            x_range=(0, 30),
            nbins=30,
            color='red',
            height=300
        )
    
    @render.ui
    def energy_histogram():
        """Energy level histogram."""
        sim = state.simulation()
        if sim is None:
            return no_data_placeholder("No data available.")
        
        if hasattr(sim, 'agents_df'):
            df = sim.agents_df
            if df.empty:
                return no_data_placeholder("No energy data.")
            energies = df['energy'].tolist()
        else:
            agents_list = list(sim.agents) if sim.agents else []
            energies = [getattr(a, 'energy_level', 0) for a in agents_list]
        
        if not energies:
            return no_data_placeholder("No energy data.")
        
        return create_histogram_chart(
            data=energies,
            title='Energy Level Distribution',
            x_title='Energy',
            y_title='Porpoise Count',
            x_range=(0, 20),
            nbins=20,
            color='red',
            height=300
        )
    
    @render.ui
    def landscape_energy_plot():
        """Landscape energy level over time."""
        history = state.energy_history()
        if not history:
            return no_data_placeholder("No landscape data yet.")
        
        df = pd.DataFrame(history)
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['landscape_energy'],
            colors=['blue'],
            names=['Landscape Energy'],
            title='Landscape Energy Level',
            x_title='Day',
            y_title='Energy',
            height=300
        )
    
    @render.ui
    def movement_plot():
        """Average porpoise movement chart."""
        history = state.movement_history()
        if not history:
            return no_data_placeholder("No movement data yet.")
        
        df = pd.DataFrame(history)
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['avg_daily_movement'],
            colors=['blue'],
            names=['Average Daily Movement'],
            title='Average Porpoise Movement',
            x_title='Day',
            y_title='Moved Cells Daily',
            height=300
        )
    
    @render.data_frame
    def vital_stats_table():
        sim = state.simulation()
        if sim is None:
            return pd.DataFrame()
        
        try:
            stats = sim.get_statistics()
            df = pd.DataFrame([
                {"Statistic": k, "Value": f"{v:.2f}" if isinstance(v, float) else str(v)}
                for k, v in stats.items()
            ])
            return df
        except Exception:
            return pd.DataFrame()
    
    # =========================================================================
    # Disturbance Tab Renderers
    # =========================================================================
    
    @render.ui
    def dispersal_plot():
        """Porpoise Dispersal chart."""
        history = state.dispersal_history()
        if not history:
            return no_data_placeholder("No dispersal data yet.")
        
        df = pd.DataFrame(history)
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['dispersing_1', 'dispersing_2', 'dispersing_3'],
            colors=['blue', 'green', 'orange'],
            names=['Dispersing 1', 'Dispersing 2', 'Dispersing 3'],
            title='Porpoise Dispersal',
            x_title='Day',
            y_title='# Porpoises',
            height=350
        )
    
    @render.ui
    def deterrence_plot():
        """Deterrence events display."""
        sim = state.simulation()
        if sim is None:
            return no_data_placeholder("Deterrence data will appear when turbines/ships are active.")
        
        deterred = 0
        if hasattr(sim, 'population_manager'):
            # Vectorized count
            # Assuming deter_strength is maintained in population manager
            # Although my initial implementation of PorpoisePopulation didn't fully propagate 
            # deter_strength from step() back to the array. 
            # I need to ensure step() updates self.deter_strength
            pop = sim.population_manager
            if hasattr(pop, 'deter_strength'):
                deterred = np.sum(pop.deter_strength > 0)
        else:
            agents_list = list(sim.agents) if sim.agents else []
            deterred = sum(1 for a in agents_list if getattr(a, 'deter_strength', 0) > 0)
        
        if deterred == 0:
            return no_data_placeholder("No deterrence events detected. Enable turbines or ships.")
        
        return ui.p(f"Currently deterred: {deterred} porpoises", class_="text-center mt-5")
    
    @render.ui
    def noise_map():
        """Noise exposure map placeholder."""
        return no_data_placeholder("Noise exposure map will appear when disturbance sources are active.")
    
    # =========================================================================
    # Export
    # =========================================================================
    
    @render.download(filename="cenop_results.csv")
    def download_data():
        history = state.population_history()
        if history:
            df = pd.DataFrame(history)
            return df.to_csv(index=False)
        return ""
