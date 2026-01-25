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
from pathlib import Path

from .reactive_state import SimulationState
try:
    # Importing SimulationRunner at module import time can create circular imports
    # in unit test environments. Import lazily where needed instead.
    from .simulation_controller import create_simulation_from_inputs, SimulationRunner
except ImportError:
    create_simulation_from_inputs = None
    SimulationRunner = None
from .renderers.chart_helpers import (
    create_time_series_chart,
    create_histogram_chart,
    create_map_figure,
    create_pydeck_map,
    no_data_placeholder
)

logger = logging.getLogger("CENOP")


def run_simulation_loop(runner, result_queue, stop_event, throttle_value, throttle_lock, ticks_per_update_value, ticks_lock):
    """Background thread worker for simulation loop.

    Args:
        runner: SimulationRunner instance
        result_queue: Queue for sending updates to main thread
        stop_event: Threading event to signal stop
        throttle_value: List with single float [0.0-1.0] for speed control (mutable for thread sharing)
        throttle_lock: Threading lock to protect throttle_value access
        ticks_per_update_value: List with single int [1-48] for ticks per update (mutable for thread sharing)
        ticks_lock: Threading lock to protect ticks_per_update_value access
    """
    import time
    print(f"[DEBUG] run_simulation_loop STARTED - max_ticks={runner.max_ticks}")
    loop_count = 0
    try:
        while not runner.is_complete and not stop_event.is_set():
            loop_count += 1
            
            # Read ticks_per_update with lock protection
            with ticks_lock:
                current_ticks = ticks_per_update_value[0]
            runner.set_ticks_per_update(current_ticks)
            
            # Step configured number of ticks
            entry = runner.step_ticks()

            # Read throttle value with lock protection
            with throttle_lock:
                current_speed = throttle_value[0]

            if loop_count <= 5 or loop_count % 500 == 0:
                print(f"[DEBUG] Loop #{loop_count}: tick={runner.tick}, pop={entry.get('population', '?')}, year={entry.get('year', '?')}, speed={current_speed:.2f}, ticks_per_update={current_ticks}")

            # Send update to main thread
            # Only include sim reference when map needs update (reduces memory pressure)
            # Build lightweight update payload; avoid sending full Simulation objects
            porpoise_positions = None
            if runner.should_update_map:
                try:
                    porpoise_positions = runner.sim.get_porpoise_positions().tolist()
                except Exception:
                    porpoise_positions = None

            update = {
                "type": "update",
                "progress": runner.progress_percent,
                "entry": entry,
                "total_births": runner.total_births,
                "total_deaths": runner.total_deaths,
                "should_update_map": runner.should_update_map,
                "porpoise_positions": porpoise_positions
            }
            result_queue.put(update)

            # Dynamic sleep based on throttle value
            # throttle_value[0] is 0.0 (slowest) to 1.0 (fastest)
            # Use exponential scaling for more responsive control:
            # At 1.0 (100%): sleep = 0 (as fast as possible)
            # At 0.5 (50%): sleep = ~0.05
            # At 0.0 (1%): sleep = 0.3 (slow but not frozen)
            if current_speed >= 0.99:
                sleep_time = 0  # Maximum speed - no delay
            else:
                # Exponential: slower at low speeds, faster at high speeds
                sleep_time = 0.3 * ((1.0 - current_speed) ** 2)

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


# =========================================================================
# Helper functions for testability (defined at module level)
# =========================================================================

def _build_landscape_table_rows(landscapes):
    """Build table row data for each landscape (pure helper for testing).
    
    Args:
        landscapes: List of landscape names
        
    Returns:
        List of dicts with keys: name, core_icons, prey_months, salinity_months, error
    """
    from cenop.landscape.loader import LandscapeLoader
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Building table rows for {len(landscapes)} landscapes")
    rows = []
    for i, name in enumerate(sorted(landscapes), 1):
        try:
            logger.debug(f"  Processing landscape {i}/{len(landscapes)}: {name}")
            loader = LandscapeLoader(name)
            info = loader.list_files()
            
            # Core file presence icons
            core_files = ["bathy.asc", "disttocoast.asc", "sediment.asc", "patches.asc", "blocks.asc"]
            core_icons = ["✅" if info.get(k, False) else "❌" for k in core_files]
            
            prey = info.get("prey_months", [])
            sal = info.get("salinity_months", [])
            
            rows.append({
                'name': name,
                'core_icons': core_icons,
                'prey_months': prey,
                'salinity_months': sal,
                'error': None
            })
        except Exception as e:
            logger.warning(f"  Error processing landscape {name}: {e}")
            rows.append({
                'name': name,
                'core_icons': [],
                'prey_months': [],
                'salinity_months': [],
                'error': str(e)
            })
    logger.info(f"Completed building {len(rows)} table rows")
    return rows
    return rows


def _build_details_modal_content(name, info, warnings):
    """Build the HTML content for the landscape details modal (pure helper for testing).
    
    Args:
        name: Landscape name
        info: Dict from loader.list_files() with keys like bathy.asc, prey_months, etc.
        warnings: List of warning strings from loader.load_all()
        
    Returns:
        Shiny UI div containing the modal content
    """
    from shiny import ui as shiny_ui
    
    core_files = ["bathy.asc", "disttocoast.asc", "sediment.asc", "patches.asc", "blocks.asc"]
    core_list = [shiny_ui.tags.li(f"{'✅' if info.get(f, False) else '❌'} {f}") for f in core_files]
    prey = info.get('prey_months', [])
    sal = info.get('salinity_months', [])

    warn_nodes = []
    if warnings:
        warn_nodes.append(shiny_ui.h5("Loader warnings", class_="text-danger"))
        warn_nodes.append(shiny_ui.tags.ul(*[shiny_ui.tags.li(w) for w in warnings]))
    else:
        warn_nodes.append(shiny_ui.p("No loader warnings reported."))

    return shiny_ui.div(
        shiny_ui.h4(f"Landscape: {name}"),
        shiny_ui.h5("Core files"),
        shiny_ui.tags.ul(*core_list),
        shiny_ui.h5("Monthly files"),
        shiny_ui.p(f"Prey months: {prey if prey else '—'}"),
        shiny_ui.p(f"Salinity months: {sal if sal else '—'}"),
        shiny_ui.hr(),
        *warn_nodes
    )


def server(input, output, session):
    """Main server function for CENOP Shiny app."""
    logger.info("Server function initialized")
    
    # Centralized reactive state
    state = SimulationState()
    
    # Initialize preview with default landscape
    state.selected_preview_file.set({
        'landscape': 'CentralBaltic',
        'file': 'bathy.asc'
    })
    
    # Internal state for background thread management
    sim_thread: threading.Thread | None = None
    stop_event = threading.Event()
    result_queue = queue.Queue()
    # Shared throttle value as a mutable list [0.0-1.0] for thread-safe updates
    # 0.0 = slowest (1%), 1.0 = fastest (100%)
    throttle_value = [1.0]  # Default 100% (maximum speed)
    throttle_lock = threading.Lock()  # Protects throttle_value access
    # Shared ticks per update value [1-48] for map update frequency
    ticks_per_update_value = [1]  # Default 1 tick (every tick)
    ticks_lock = threading.Lock()  # Protects ticks_per_update_value access
    
    # =========================================================================
    # Help Modal
    # =========================================================================
    
    @reactive.effect
    @reactive.event(input.help_btn)
    def show_help_modal():
        """Show the help modal when help button is clicked."""
        from ..ui.layout import create_help_modal
        ui.modal_show(create_help_modal())
    
    # =========================================================================
    # Landscape Loading
    # =========================================================================
    
    @reactive.effect
    @reactive.event(input.load_landscape)
    def trigger_load_landscape():
        """Trigger landscape loading when button is clicked."""
        try:
            landscape_name = input.landscape()
            if not landscape_name:
                ui.notification_show("Please select a landscape", type="warning")
                return
            # Increment counter to trigger depth_data_initializer
            current = state.landscape_load_counter()
            state.landscape_load_counter.set(current + 1)
            state.landscape_loaded_name.set(landscape_name)
            logger.info(f"Load Landscape button clicked, loading '{landscape_name}'")
            ui.notification_show(f"Loading landscape: {landscape_name}...", type="info", duration=3)
        except Exception as e:
            logger.error(f"Error loading landscape: {e}", exc_info=True)
            ui.notification_show(f"Error loading landscape: {str(e)}", type="error")
    
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
        
        # Get current landscape with error handling
        landscape = "Homogeneous"
        try:
            landscape = input.landscape()
        except Exception:
            pass
            
        # Get compatible turbines for selected landscape
        compatible = LANDSCAPE_TURBINE_COMPATIBILITY.get(landscape, {"off": "No turbines"})
        
        return shiny_ui.input_select(
            "turbines", 
            "Wind Turbines", 
            choices=compatible, 
            selected=list(compatible.keys())[0] if compatible else "off"
        )

    # ---------------------------------------------------------------------
    # Dynamic Landscape Selector (refreshable)
    # Renders the Landscape select input server-side so the list of choices
    # can be refreshed on demand via the 'refresh_landscapes' button.
    @render.ui
    def landscape_selector():
        """Render the landscape selector with current available landscapes."""
        from shiny import ui as shiny_ui
        # Use the refresh button as an event to re-run this renderer
        _ = None
        try:
            # referencing event input.refresh_landscapes() causes reactivity
            _ = input.refresh_landscapes()
        except Exception:
            # Older sessions might not have the input bound; ignore
            pass

        try:
            from cenop.landscape.loader import LandscapeLoader
            lands = LandscapeLoader.list_landscapes()
            logger.info(f"Landscape selector: found {len(lands)} landscapes")
        except Exception as e:
            lands = []
            logger.error(f"Could not list landscapes: {e}", exc_info=True)

        choices = ["Homogeneous"] + sorted(lands)
        
        # Keep currently selected landscape if still available
        current = None
        try:
            current = input.landscape()
        except Exception:
            pass
            
        selected = current if (current and current in choices) else (choices[0] if choices else "Homogeneous")

        return shiny_ui.input_select("landscape", None, choices=choices, selected=selected)
    
    @reactive.effect
    @reactive.event(input.load_turbines)
    def trigger_load_turbines():
        """Trigger turbine loading when button is clicked."""
        try:
            turbine_scenario = input.turbines()
            if not turbine_scenario:
                ui.notification_show("Please select a turbine scenario", type="warning")
                return
            # Increment counter to trigger turbine and noise data initializers
            current = state.turbine_load_counter()
            state.turbine_load_counter.set(current + 1)
            state.turbine_loaded_name.set(turbine_scenario)
            logger.info(f"Load Turbines button clicked, loading '{turbine_scenario}'")
            if turbine_scenario != "off":
                ui.notification_show(f"Loading turbines: {turbine_scenario}...", type="info", duration=3)
        except Exception as e:
            logger.error(f"Error loading turbines: {e}", exc_info=True)
            ui.notification_show(f"Error loading turbines: {str(e)}", type="error")
    
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

    # -----------------------------------------------------------------
    # Data Available table (refreshable)
    
    @render.text
    def data_available_refreshed():
        """Show last refresh time for data available table."""
        ts = None
        try:
            ts = state.last_refreshed()
        except Exception:
            pass
        if ts:
            return f"Last refreshed: {ts}"
        return ""
    
    @render.ui
    def data_available_table():
        """Render a compact table summarizing files available per landscape.

        Shows a notification toast with timestamp when the refresh button is used,
        and provides a 'Details' button per landscape that triggers a server-side
        modal with loader warnings and file lists (loaded on demand).
        """
        from shiny import ui as shiny_ui
        from datetime import datetime
        
        # Check if refresh button was clicked
        refresh_clicked = False
        try:
            _ = input.refresh_data_available()
            refresh_clicked = True
        except Exception:
            pass

        try:
            from cenop.landscape.loader import LandscapeLoader
            import os
            logger.info(f"Loading landscapes from current directory: {os.getcwd()}")
            landscapes = LandscapeLoader.list_landscapes()
            logger.info(f"Found {len(landscapes)} landscapes: {landscapes}")
            
            # Show notification after refresh
            if refresh_clicked:
                ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                try:
                    state.last_refreshed.set(ts)
                except Exception:
                    pass
                ui.notification_show(
                    f"Data refreshed at {ts} - Found {len(landscapes)} landscape(s)", 
                    type="info", 
                    duration=5
                )
        except Exception as e:
            logger.error(f"Data available: could not list landscapes: {e}", exc_info=True)
            ui.notification_show(f"Error loading landscapes: {str(e)}", type="error", duration=10)
            landscapes = []

        # Build table rows using helper function for testability
        row_data = _build_landscape_table_rows(landscapes)
        
        thead = shiny_ui.tags.thead(shiny_ui.tags.tr(
            shiny_ui.tags.th("Landscape"),
            shiny_ui.tags.th(
                shiny_ui.span("Core files ", shiny_ui.tags.small("(B=Bathy, D=DistCoast, S=Sediment, Pat=Patches, Blk=Blocks)", style="font-weight: normal; color: #6c757d;"))
            ),
            shiny_ui.tags.th("Prey months"),
            shiny_ui.tags.th("Salinity months"),
            shiny_ui.tags.th("Actions")
        ))
        rows = []
        if not row_data:
            rows.append(shiny_ui.tags.tr(shiny_ui.tags.td("No landscapes found", colspan=5)))
        else:
            for data in row_data:
                name = data['name']
                if data['error']:
                    rows.append(shiny_ui.tags.tr(
                        shiny_ui.tags.td(name),
                        shiny_ui.tags.td(f"Error: {data['error']}", colspan=4)
                    ))
                    continue
                
                # Make core files more readable
                core_files = ["B", "D", "S", "Pat", "Blk"]  # Bathy, DistToCoast, Sediment, Patches, Blocks
                core_icons_with_labels = []
                for i, (icon, label) in enumerate(zip(data['core_icons'], core_files)):
                    core_icons_with_labels.append(f"{icon}{label}")
                core_cell = shiny_ui.tags.td(" ".join(core_icons_with_labels), style="font-size: 0.85em;")
                prey_cell = shiny_ui.tags.td(str(data['prey_months']) if data['prey_months'] else "—")
                sal_cell = shiny_ui.tags.td(str(data['salinity_months']) if data['salinity_months'] else "—")
                
                # Details button: sets input.detail_landscape to landscape name (server event)
                btn = shiny_ui.tags.button(
                    "Details",
                    type="button",
                    class_="btn btn-sm btn-outline-secondary",
                    onclick=f"event.stopPropagation(); Shiny.setInputValue('detail_landscape','{name}', {{priority: 'event'}});"
                )

                # Make row clickable - select bathy.asc file for preview (removed to prevent reload)
                rows.append(shiny_ui.tags.tr(
                    shiny_ui.tags.td(shiny_ui.tags.strong(name)),
                    core_cell,
                    prey_cell,
                    sal_cell,
                    shiny_ui.tags.td(btn, onclick="event.stopPropagation();")
                ))

        table = shiny_ui.tags.table(thead, shiny_ui.tags.tbody(*rows), class_="table table-sm table-striped")
        return shiny_ui.card(shiny_ui.card_header("Data Available"), shiny_ui.div(table))

    # -----------------------------------------------------------------
    # Details modal: load full info and warnings for a single landscape
    @reactive.effect
    @reactive.event(input.detail_landscape)
    def show_landscape_details():
        """Show details modal when a landscape's Details button is clicked."""
        try:
            name = input.detail_landscape()
            if not name:
                return
        except Exception:
            return

        try:
            from cenop.landscape.loader import LandscapeLoader
            loader = LandscapeLoader(name)
            info = loader.list_files()
            # load_all may return loader warnings (heavy operation but single-landscape)
            loaded = loader.load_all()
            warnings = loaded.get('warnings', [])
        except Exception as e:
            ui.notification_show(f"Error loading details for {name}: {e}", type="error")
            return

        # Build modal content using helper function for testability
        from shiny import ui as shiny_ui
        detail_ui = _build_details_modal_content(name, info, warnings)
        ui.modal_show(shiny_ui.modal_dialog(detail_ui, title=f"Details: {name}", easy_close=True))

    # -----------------------------------------------------------------
    # Data Preview Pane
    
    @reactive.effect
    @reactive.event(input.preview_landscape, ignore_none=True)
    def _handle_preview_landscape():
        """Update preview when landscape selector changes."""
        try:
            selected_landscape = input.preview_landscape()
            logger.info(f"[PREVIEW DEBUG] _handle_preview_landscape fired: {selected_landscape}")
            if selected_landscape:
                # Default to bathymetry file
                state.selected_preview_file.set({
                    'landscape': selected_landscape,
                    'file': 'bathy.asc'
                })
                logger.info(f"[PREVIEW DEBUG] Preview landscape changed to: {selected_landscape}, file reset to bathy.asc")
        except Exception as e:
            logger.error(f"[PREVIEW DEBUG] Error handling preview landscape change: {e}", exc_info=True)
    
    @render.ui
    @reactive.event(input.preview_landscape, ignore_none=True)
    def data_preview_controls():
        """Generate file selector dropdown based on selected landscape.
        
        Only re-renders when preview_landscape changes, not when files are selected.
        """
        try:
            selected_landscape = input.preview_landscape()
            logger.info(f"[PREVIEW DEBUG] data_preview_controls rendering for landscape: {selected_landscape}")
            
            if not selected_landscape:
                logger.info(f"[PREVIEW DEBUG] No landscape selected, returning empty div")
                return ui.div()
            
            # Get available files for this landscape
            module_dir = Path(__file__).resolve().parent.parent.parent.parent
            data_dir = module_dir / "data" / selected_landscape
            
            if not data_dir.exists():
                logger.warning(f"[PREVIEW DEBUG] Data directory not found: {data_dir}")
                return ui.span("No data directory found", class_="text-muted")
            
            # List all .asc files
            files = sorted([f.name for f in data_dir.glob("*.asc")])
            logger.info(f"[PREVIEW DEBUG] Found {len(files)} .asc files in {selected_landscape}")
            
            if not files:
                logger.warning(f"[PREVIEW DEBUG] No .asc files found in {selected_landscape}")
                return ui.span("No data files found", class_="text-muted")
            
            # Categorize files
            core_files = []
            monthly_files = {}
            
            for f in files:
                fname_lower = f.lower()
                # Monthly pattern: prey0000_01.asc or prey01.asc
                monthly_match = any(pattern in fname_lower for pattern in [
                    'prey', 'sal', 'temp', 'sed'
                ])
                if monthly_match and any(char.isdigit() for char in f[-6:-4]):
                    # Extract type and month
                    for prefix in ['prey', 'sal', 'temp', 'sed']:
                        if prefix in fname_lower:
                            month_str = ''.join(filter(str.isdigit, f[-6:-4]))
                            if month_str:
                                month = int(month_str)
                                if prefix not in monthly_files:
                                    monthly_files[prefix] = []
                                monthly_files[prefix].append((month, f))
                            break
                else:
                    core_files.append(f)
            
            # Build choices
            choices = {}
            
            # Core files first
            if core_files:
                for f in core_files:
                    label = f.replace('.asc', '').replace('_', ' ').title()
                    choices[f] = label
            
            # Monthly files grouped
            for file_type in ['prey', 'sal', 'temp', 'sed']:
                if file_type in monthly_files:
                    type_label = {'prey': 'Prey', 'sal': 'Salinity', 'temp': 'Temperature', 'sed': 'Sediment'}[file_type]
                    sorted_months = sorted(monthly_files[file_type])
                    for month, fname in sorted_months:
                        choices[fname] = f"{type_label} (Month {month})"
            
            # Get current selection
            # Use isolate to prevent reactive loop when state is updated
            with reactive.isolate():
                current = state.selected_preview_file()
            
            current_file = None
            if current and current.get('landscape') == selected_landscape:
                current_file = current.get('file')
            
            # Default to first file if no selection
            if not current_file and choices:
                current_file = list(choices.keys())[0]
            
            logger.info(f"[PREVIEW DEBUG] Returning file selector with {len(choices)} choices, current: {current_file}")
            
            return ui.input_select(
                "file_selector",
                "File:",
                choices=choices,
                selected=current_file,
                width="300px"
            )
        except Exception as e:
            logger.error(f"[PREVIEW DEBUG] Exception in data_preview_controls: {e}", exc_info=True)
            return ui.div(
                ui.p(f"Error loading file list: {str(e)}", class_="text-danger")
            )
    
    @reactive.effect
    @reactive.event(input.file_selector, ignore_none=True, ignore_init=True)
    def _handle_file_selector_change():
        """Update preview when file selector changes."""
        try:
            selected_file = input.file_selector()
            logger.info(f"[PREVIEW DEBUG] _handle_file_selector_change fired: {selected_file}")
            
            # Get current landscape from preview_landscape selector
            # Use isolate to avoid triggering on landscape changes
            with reactive.isolate():
                landscape = input.preview_landscape()
            
            if landscape and selected_file:
                logger.info(f"[PREVIEW DEBUG] Setting selected_preview_file to {landscape}/{selected_file}")
                state.selected_preview_file.set({
                    'landscape': landscape,
                    'file': selected_file
                })
                logger.info(f"[PREVIEW DEBUG] File selector changed to: {selected_file}")
            else:
                logger.warning(f"[PREVIEW DEBUG] Skipping update - landscape: {landscape}, file: {selected_file}")
        except Exception as e:
            logger.error(f"[PREVIEW DEBUG] Error handling file selector change: {e}", exc_info=True)
    
    # Cache for preview data to avoid reloading
    _preview_data_cache = None
    _preview_cache_key = None
    
    # Track render count to detect loops
    _preview_loader_render_count = [0]
    
    @reactive.calc
    def preview_data_source():
        """Load ASC file data and return structured dict.
        
        Optimized for large files like Central Baltic (26MB, 400x400 grid).
        Uses smart downsampling and caching.
        """
        import json
        import numpy as np
        nonlocal _preview_data_cache, _preview_cache_key, _preview_loader_render_count
        
        _preview_loader_render_count[0] += 1
        
        file_info = state.selected_preview_file()
        logger.info(f"[PREVIEW DEBUG] preview_data_calc triggered for {file_info}")
        
        if not file_info:
            return None
        
        landscape = file_info.get('landscape')
        filename = file_info.get('file')
        
        if not landscape or not filename:
            return None
        
        cache_key = f"{landscape}/{filename}"
        
        # Use cache if available
        if _preview_data_cache is not None and _preview_cache_key == cache_key:
            logger.info(f"[PREVIEW DEBUG] Using cached data for {cache_key}")
            return _preview_data_cache
            
        logger.info(f"[PREVIEW DEBUG] Loading fresh data for {cache_key}")
        try:
            # Load ASC file
            module_dir = Path(__file__).resolve().parent.parent.parent.parent
            file_path = module_dir / "data" / landscape / filename
            
            if not file_path.exists():
                    logger.error(f"[PREVIEW DEBUG] File not found: {file_path}")
                    return None

            logger.info(f"Loading preview data: {file_path}")
            
            # Read ASC file metadata
            with open(file_path, 'r') as f:
                metadata = {}
                for _ in range(6):  # Read 6 header lines
                    line = f.readline().strip().split()
                    if len(line) == 2:
                        key = line[0].lower()
                        value = float(line[1])
                        metadata[key] = value
            
            ncols = int(metadata['ncols'])
            nrows = int(metadata['nrows'])
            nodata_value = metadata.get('nodata_value', -9999)
            
            # Read data array
            data_array = np.loadtxt(file_path, skiprows=6)
            
            # Flip array (ESRI ASC format has top row first)
            data_array = np.flipud(data_array)
            
            # Get landscape-specific bounds (WGS84 lat/lon)
            from ..ui.sidebar import LANDSCAPE_BOUNDS
            bounds = LANDSCAPE_BOUNDS.get(landscape, (53.27, 54.79, 4.83, 7.13))
            lat_min, lat_max, lon_min, lon_max = bounds
            
            # Calculate center from bounds
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            
            # Calculate data type from filename
            fname_lower = filename.lower()
            if 'bathy' in fname_lower or 'depth' in fname_lower:
                data_type = 'bathy'
            elif 'sal' in fname_lower:
                data_type = 'sal'
            elif 'prey' in fname_lower or 'food' in fname_lower:
                data_type = 'prey'
            elif 'temp' in fname_lower:
                data_type = 'temp'
            else:
                data_type = 'other'
            
            # Mask nodata values
            valid_mask = data_array != nodata_value
            valid_data = data_array[valid_mask]
            
            if len(valid_data) == 0:
                logger.warning(f"No valid data in {filename}")
                return None
            
            # Calculate bounds
            data_min = float(np.min(valid_data))
            data_max = float(np.max(valid_data))
            
            # Smart downsampling based on grid size
            total_cells = nrows * ncols
            valid_cells = np.sum(valid_mask)
            
            # Adaptive sampling: larger files = more aggressive sampling
            if total_cells > 100000:  # Very large (e.g., 400x400 = 160,000)
                max_points = 3000
                sample_step = max(1, int(np.sqrt(valid_cells / max_points)))
            elif total_cells > 40000:  # Large
                max_points = 5000
                sample_step = max(1, int(np.sqrt(valid_cells / max_points)))
            else:  # Medium/small
                max_points = 10000
                sample_step = 1 if valid_cells < max_points else int(np.sqrt(valid_cells / max_points))
            
            logger.info(f"Sampling {filename}: {nrows}x{ncols} grid, step={sample_step}, target ~{max_points} points")
            
            # Build point array with proper coordinate mapping
            points = []
            for row in range(0, nrows, sample_step):
                for col in range(0, ncols, sample_step):
                    value = float(data_array[row, col])
                    if value != nodata_value:
                        lat = lat_min + (row / nrows) * (lat_max - lat_min)
                        lon = lon_min + (col / ncols) * (lon_max - lon_min)
                        points.append({
                            "position": [lon, lat],
                            "value": value
                        })
            
            preview_data = {
                "points": points,
                "min": data_min,
                "max": data_max,
                "name": f"{landscape} / {filename}",
                "dataType": data_type,
                "centerLat": center_lat,
                "centerLon": center_lon,
                "gridInfo": f"{nrows}x{ncols}, {len(points)} points sampled"
            }
            
            # Cache the result
            _preview_data_cache = preview_data
            _preview_cache_key = cache_key
            
            return preview_data
            
        except Exception as e:
            logger.error(f"[PREVIEW DEBUG] Error loading preview data {landscape}/{filename}: {e}", exc_info=True)
            return None

    @reactive.effect
    async def preview_map_updater():
        """Send preview data to map trigger by data source."""
        data = preview_data_source()
        if data:
            logger.info(f"[PREVIEW DEBUG] Sending {len(data['points'])} points via custom message")
            await session.send_custom_message("preview_data_update", data)
            import gc
            gc.collect()

    @render.ui
    def preview_stats_text():
        """Show metadata for the previewed file."""
        data = preview_data_source()
        if data:
            return ui.HTML(f'''
            <div class="text-muted small mt-2" style="text-align: center;">
                📍 {data["gridInfo"]} | 
                Range: {data["min"]:.2f} to {data["max"]:.2f}
            </div>
            ''')
        return ui.div()

    
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
        # Import simulation controller (always import fresh to avoid stale references)
        from .simulation_controller import create_simulation_from_inputs as create_sim, SimulationRunner as Runner
        sim = create_sim(input)
        print(f"[DEBUG] Simulation created, is_initialized={sim._is_initialized}")
        sim.initialize()
        print(f"[DEBUG] Simulation initialized, pop_size={sim.population_size}, max_ticks={sim.max_ticks}")
        
        runner = Runner(sim)
        print(f"[DEBUG] SimulationRunner created, max_ticks={runner.max_ticks}")
        
        # Reset queue and event - use idiomatic pattern to avoid TOCTOU race
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
        stop_event.clear()
        
        state.simulation.set(sim)
        state.running.set(True)
        state.population_history.set([])
        state.energy_history.set([])  # Reset energy history
        state.dispersal_history.set([])  # Reset dispersal history
        state.birth_count.set(0)
        state.death_count.set(0)
        state.progress.set(0.0)
        state.progress_message.set("Running simulation...")
        
        # Update throttle from current slider value (with lock protection)
        speed_percent = input.sim_speed()
        with throttle_lock:
            throttle_value[0] = (speed_percent - 1) / 99.0  # Convert 1-100 to 0.0-1.0
        
        # Update ticks per update from slider value
        ticks_val = input.ticks_per_update()
        with ticks_lock:
            ticks_per_update_value[0] = ticks_val

        # Start background thread
        sim_thread = threading.Thread(
            target=run_simulation_loop,
            args=(runner, result_queue, stop_event, throttle_value, throttle_lock, ticks_per_update_value, ticks_lock),
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
        with throttle_lock:
            throttle_value[0] = new_throttle
        print(f"[DEBUG] Speed slider changed: {speed_percent}% -> throttle={new_throttle:.3f}")
    
    @reactive.effect
    @reactive.event(input.ticks_per_update)
    def update_ticks_per_update():
        """Update ticks per update when slider changes during simulation."""
        ticks_val = input.ticks_per_update()
        with ticks_lock:
            ticks_per_update_value[0] = ticks_val
        print(f"[DEBUG] Ticks per update changed: {ticks_val}")
    
    @reactive.effect
    def poll_simulation():
        """Poll for updates from the background simulation thread."""
        running = state.running()
        if not running:
            return
        
        # Poll at reasonable interval (200ms) to avoid overwhelming connection
        reactive.invalidate_later(0.2)
        
        # Process multiple updates per poll to keep up
        has_updates = False
        last_positions = None
        msgs_processed = 0
        entries_batch = []
        energy_entries_batch = []
        dispersal_entries_batch = []
        
        # Drain queue - process all available messages
        while True:
            try:
                msg = result_queue.get_nowait()
            except queue.Empty:
                break  # No more messages, exit loop and process batch
                
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
                if energy_entries_batch:
                    current_energy = state.energy_history()
                    state.energy_history.set(current_energy + energy_entries_batch)
                if dispersal_entries_batch:
                    current_dispersal = state.dispersal_history()
                    state.dispersal_history.set(current_dispersal + dispersal_entries_batch)
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
                
                # Extract and batch energy entry if present
                if 'energy_entry' in entry:
                    energy_entries_batch.append(entry['energy_entry'])
                
                # Extract and batch dispersal entry if present
                if 'dispersal_entry' in entry and entry['dispersal_entry'] is not None:
                    dispersal_entries_batch.append(entry['dispersal_entry'])
                
                if msg["should_update_map"]:
                    state.map_update_counter.set(state.map_update_counter() + 1)
                    # Extract porpoise positions snapshot (lightweight) instead of whole sim
                    if msg.get("porpoise_positions") is not None:
                        try:
                            state.porpoise_positions.set(msg.get("porpoise_positions"))
                        except Exception:
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
        # Clear queue to release refs - use idiomatic pattern to avoid TOCTOU race
        try:
            while True:
                result_queue.get_nowait()
        except queue.Empty:
            pass
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
                    # Surface loader warnings to UI if any
                    if getattr(landscape, '_load_warnings', None):
                        msgs = landscape._load_warnings
                        for m in msgs:
                            ui.notification_show(f"Landscape load: {m}", type="warning")

                    # Sample depth data (every Nth cell to reduce data size)
                    # 400x400 = 160,000 cells is too many, sample every 5th = 6,400 cells
                    sample_step = 5
                    depth_array = landscape._depth
                    grid_height, grid_width = depth_array.shape
                    
                    # Get landscape-specific bounds
                    from ..ui.sidebar import LANDSCAPE_BOUNDS
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
        from ..ui.sidebar import LANDSCAPE_BOUNDS
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
                    # Surface loader warnings to UI if any
                    if getattr(landscape, '_load_warnings', None):
                        msgs = landscape._load_warnings
                        for m in msgs:
                            ui.notification_show(f"Landscape load: {m}", type="warning")

                    # Sample food data (every Nth cell to reduce data size)
                    sample_step = 5
                    grid_height, grid_width = food_array.shape
                    
                    # Get landscape-specific bounds
                    from ..ui.sidebar import LANDSCAPE_BOUNDS
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
            from ..ui.sidebar import LANDSCAPE_BOUNDS
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
                from ..ui.sidebar import LANDSCAPE_BOUNDS
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
        from ..ui.sidebar import LANDSCAPE_BOUNDS
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
                    
                    # Debug: show porpoise grid and latlon ranges (only first time)
                    if map_counter <= 1:
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
        # React to population history to update during simulation
        _ = state.population_history()
        
        sim = state.simulation()
        if sim is None:
            return no_data_placeholder("Run simulation to see age distribution.")
        
        # Use population_manager directly for vectorized access
        if hasattr(sim, 'population_manager'):
            pm = sim.population_manager
            if hasattr(pm, 'age') and hasattr(pm, 'active_mask'):
                active = pm.active_mask
                if np.any(active):
                    ages = pm.age[active].tolist()
                else:
                    return no_data_placeholder("No active porpoises.")
            else:
                return no_data_placeholder("No age data available.")
        elif hasattr(sim, 'agents_df'):
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
        # React to population history to update during simulation
        _ = state.population_history()
        
        sim = state.simulation()
        if sim is None:
            return no_data_placeholder("Run simulation to see energy distribution.")
        
        # Use population_manager directly for vectorized access
        if hasattr(sim, 'population_manager'):
            pm = sim.population_manager
            if hasattr(pm, 'energy') and hasattr(pm, 'active_mask'):
                active = pm.active_mask
                if np.any(active):
                    energies = pm.energy[active].tolist()
                else:
                    return no_data_placeholder("No active porpoises.")
            else:
                return no_data_placeholder("No energy data available.")
        elif hasattr(sim, 'agents_df'):
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
        """Landscape energy level over time - uses avg porpoise energy as proxy."""
        history = state.energy_history()
        if not history:
            return no_data_placeholder("Run simulation to see energy trends.")
        
        df = pd.DataFrame(history)
        # Use avg_food_eaten as landscape energy proxy
        if 'avg_food_eaten' not in df.columns:
            return no_data_placeholder("No landscape energy data.")
        
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['avg_food_eaten'],
            colors=['blue'],
            names=['Avg Porpoise Energy'],
            title='Average Porpoise Energy Level',
            x_title='Day',
            y_title='Energy',
            height=300
        )
    
    @render.ui
    def movement_plot():
        """Average porpoise movement chart - uses dispersal data."""
        history = state.dispersal_history()
        if not history:
            return no_data_placeholder("Run simulation to see movement data.")
        
        df = pd.DataFrame(history)
        if 'dispersing_count' not in df.columns:
            return no_data_placeholder("No movement data available.")
        
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['dispersing_count'],
            colors=['blue'],
            names=['Dispersing Porpoises'],
            title='Porpoise Dispersal Activity',
            x_title='Day',
            y_title='Count',
            height=300
        )
    
    @render.data_frame
    def vital_stats_table():
        # React to population history to update during simulation
        _ = state.population_history()
        
        sim = state.simulation()
        if sim is None:
            return pd.DataFrame()
        
        try:
            stats = sim.get_statistics()
            # Add more stats from population_manager if available
            if hasattr(sim, 'population_manager'):
                pm = sim.population_manager
                active = pm.active_mask
                if np.any(active):
                    stats['avg_age'] = float(np.mean(pm.age[active]))
                    stats['avg_energy'] = float(np.mean(pm.energy[active]))
                    stats['females'] = int(np.sum(pm.is_female[active]))
                    stats['with_calf'] = int(np.sum(pm.with_calf[active]))
            
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
            return no_data_placeholder("No dispersal data yet. Run simulation to see results.")
        
        df = pd.DataFrame(history)
        # Use actual columns from dispersal_entry
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['dispersing_count', 'max_declining_days'],
            colors=['blue', 'orange'],
            names=['Dispersing Porpoises', 'Max Declining Days'],
            title='Porpoise Dispersal Behavior',
            x_title='Day',
            y_title='Count',
            height=350
        )
    
    @render.ui
    def deterrence_plot():
        """Deterrence events display."""
        # Use population history which includes deterred_count
        history = state.population_history()
        if not history:
            return no_data_placeholder("Deterrence data will appear when simulation runs with turbines/ships.")
        
        # Extract deterred counts from history
        deterred_data = [{'day': h['day'], 'deterred': h.get('deterred_count', 0)} for h in history]
        df = pd.DataFrame(deterred_data)
        
        # Check if any deterrence occurred
        if df['deterred'].sum() == 0:
            return no_data_placeholder("No deterrence events detected. Enable turbines or ships.")
        
        return create_time_series_chart(
            df=df,
            x_col='day',
            y_cols=['deterred'],
            colors=['red'],
            names=['Deterred Porpoises'],
            title='Deterrence Events Over Time',
            x_title='Day',
            y_title='# Deterred',
            height=350
        )
    
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
