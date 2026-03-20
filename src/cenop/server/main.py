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
    create_svg_chart,
    no_data_placeholder
)

from shiny_deckgl import zoom_widget, compass_widget, scale_widget, fullscreen_widget, bitmap_layer, scatterplot_layer
from cenop.ui.tabs.dashboard import sim_map
from cenop.server.map_layers import (
    build_porpoise_layer,
    build_porpoise_trails_layer,
    build_grid_bitmap_layer,
    build_noise_construction_layer,
    build_noise_operational_layer,
    build_turbine_pole_layer,
    build_turbine_blade_layer,
)

logger = logging.getLogger("CENOP")


def _safe_input(input_obj, name: str, default=None):
    """Safely read a Shiny input that may not be bound yet."""
    try:
        return getattr(input_obj, name)()
    except Exception:
        return default


# ---------------------------------------------------------------------------
# CRS coordinate transformer: grid cell indices → WGS84 lon/lat
# ---------------------------------------------------------------------------
_transformers: dict = {}  # cache: source_crs → pyproj.Transformer


def _get_transformer(source_crs: str):
    """Get or create a cached pyproj Transformer from source_crs to WGS84."""
    if source_crs not in _transformers:
        from pyproj import Transformer
        _transformers[source_crs] = Transformer.from_crs(
            source_crs, "EPSG:4326", always_xy=True
        )
    return _transformers[source_crs]


def grid_to_lonlat(
    grid_x, grid_y, metadata, source_crs: str
):
    """Convert grid cell coordinates to WGS84 (lon, lat).

    Works with scalars or numpy arrays.

    Args:
        grid_x: column index (or array of column indices)
        grid_y: row index (or array of row indices)
        metadata: LandscapeMetadata with xllcorner, yllcorner, cellsize
        source_crs: EPSG string for the landscape's native CRS

    Returns:
        (lon, lat) — scalars or arrays matching input type
    """
    # Grid cell centre → projected CRS coordinates
    proj_x = metadata.xllcorner + (grid_x + 0.5) * metadata.cellsize
    proj_y = metadata.yllcorner + (grid_y + 0.5) * metadata.cellsize
    transformer = _get_transformer(source_crs)
    lon, lat = transformer.transform(proj_x, proj_y)
    return lon, lat


def run_simulation_loop(
    runner,
    result_queue,
    stop_event,
    throttle_value,
    throttle_lock,
    ticks_per_update_value,
    ticks_lock,
    trace_enabled_value,
    trace_length_value,
    trace_lock,
    skip_viz_value,
    skip_viz_lock,
):
    """Background thread worker for simulation loop.

    Args:
        runner: SimulationRunner instance
        result_queue: Queue for sending updates to main thread
        stop_event: Threading event to signal stop
        throttle_value: List with single float [0.0-1.0] for speed control (mutable for thread sharing)
        throttle_lock: Threading lock to protect throttle_value access
        ticks_per_update_value: List with single int [1-48] for ticks per update (mutable for thread sharing)
        ticks_lock: Threading lock to protect ticks_per_update_value access
        trace_enabled_value: List with single bool for trace toggle (mutable for thread sharing)
        trace_length_value: List with single int for trace length in days (mutable for thread sharing)
        trace_lock: Threading lock to protect trace settings access
    """
    import time
    from collections import deque

    logger.debug("run_simulation_loop STARTED - max_ticks=%s", runner.max_ticks)
    loop_count = 0
    trail_history: dict[int, deque] = {}
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
                logger.debug("Loop #%d: tick=%s, pop=%s, year=%s, speed=%.2f, ticks_per_update=%s",
                             loop_count, runner.tick, entry.get('population', '?'),
                             entry.get('year', '?'), current_speed, current_ticks)

            # Send update to main thread
            # Check skip-viz flag
            with skip_viz_lock:
                viz_skipped = skip_viz_value[0]

            # Build lightweight update payload; avoid sending full Simulation objects
            porpoise_positions = None
            if runner.should_update_map and not viz_skipped:
                try:
                    raw_pos = runner.sim.get_porpoise_positions()  # (N,7)
                    if raw_pos.size > 0:
                        sim = runner.sim
                        meta = sim._cell_data.metadata if sim._cell_data else None
                        if meta is not None:
                            from cenop.ui.sidebar import LANDSCAPE_CRS
                            crs = LANDSCAPE_CRS.get(
                                sim.params.landscape, "EPSG:3035"
                            )
                            lons, lats = grid_to_lonlat(
                                raw_pos[:, 1], raw_pos[:, 2], meta, crs
                            )
                            # [id, lon, lat, energy, heading, age, is_dispersing]
                            converted = np.column_stack((
                                raw_pos[:, 0], lons, lats,
                                raw_pos[:, 3], raw_pos[:, 4],
                                raw_pos[:, 5], raw_pos[:, 6],
                            ))
                            porpoise_positions = converted.tolist()
                        else:
                            porpoise_positions = raw_pos.tolist()
                    else:
                        porpoise_positions = []
                except (ImportError, ValueError, RuntimeError) as e:
                    logger.debug("Coordinate transform failed: %s", e)
                    porpoise_positions = None

            # Collect trail data if traces enabled
            trail_data = None
            if runner.should_update_map:
                with trace_lock:
                    traces_on = trace_enabled_value[0]
                    # Limit trail to N map-update points (not sim ticks).
                    # Each map update adds 1 point; keep short tails to avoid clutter.
                    max_points = trace_length_value[0] * 3  # short tail per agent
                if traces_on and porpoise_positions:
                    for row in porpoise_positions:
                        pid = int(row[0])
                        lon, lat = row[1], row[2]
                        if pid not in trail_history:
                            trail_history[pid] = deque(maxlen=max_points)
                        elif trail_history[pid].maxlen != max_points:
                            trail_history[pid] = deque(
                                trail_history[pid], maxlen=max_points
                            )
                        trail_history[pid].append((lon, lat))
                    # Remove dead porpoises
                    alive_ids = {int(row[0]) for row in porpoise_positions}
                    dead_ids = set(trail_history.keys()) - alive_ids
                    for pid in dead_ids:
                        del trail_history[pid]
                    # Serialize for queue
                    trail_data = []
                    for pid, trail in list(trail_history.items())[:1000]:
                        if len(trail) >= 2:
                            trail_data.append({
                                "pid": pid,
                                "path": [
                                    [t[0], t[1]] for t in trail
                                ],
                            })
                elif not traces_on:
                    trail_history.clear()

            update = {
                "type": "update",
                "progress": runner.progress_percent,
                "entry": entry,
                "total_births": runner.total_births,
                "total_deaths": runner.total_deaths,
                "should_update_map": runner.should_update_map,
                "porpoise_positions": porpoise_positions,
                "porpoise_trails": trail_data,
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
            logger.debug("Simulation COMPLETE after %d iterations, years=%s", loop_count, runner.sim.state.year)
            result_queue.put({"type": "complete", "years": runner.sim.state.year})
            
    except Exception as e:
        logger.error("Simulation error: %s", e, exc_info=True)
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

    # --- shiny-deckgl layer cache ---
    _layer_cache: dict[str, dict] = {}
    _loaded_data_layers: set[str] = set()

    async def _push_all_layers():
        """Combine cached layers and push to MapWidget."""
        layers = [
            _layer_cache.get("depth-bitmap", bitmap_layer("depth-bitmap", "", [], visible=False)),
            _layer_cache.get("depth-tooltip", scatterplot_layer("depth-tooltip", [], visible=False)),
            _layer_cache.get("foraging-bitmap", bitmap_layer("foraging-bitmap", "", [], visible=False)),
            _layer_cache.get("foraging-tooltip", scatterplot_layer("foraging-tooltip", [], visible=False)),
            _layer_cache.get("noise-construction", build_noise_construction_layer([])),
            _layer_cache.get("noise-operational", build_noise_operational_layer([])),
            _layer_cache.get("turbine-poles", build_turbine_pole_layer([])),
            _layer_cache.get("turbine-blades", build_turbine_blade_layer([])),
            _layer_cache.get("porpoises", build_porpoise_layer([])),
            _layer_cache.get("porpoise-trails", build_porpoise_trails_layer([])),
        ]
        await sim_map.update(
            session,
            layers=layers,
            widgets=[
                zoom_widget(placement="top-right"),
                compass_widget(placement="top-right"),
                scale_widget(placement="bottom-left"),
                fullscreen_widget(placement="top-left"),
            ],
        )

    async def _push_dynamic_layers(*layer_ids: str):
        """Partial layer push — only sends specified layers by ID.

        Uses partial_update() which merges by layer ID on the JS side,
        leaving static layers (depth, foraging) untouched.
        """
        layers = [_layer_cache[lid] for lid in layer_ids if lid in _layer_cache]
        if layers:
            await sim_map.partial_update(session, layers=layers)

    async def _sync_legend():
        """Send current legend entries to JS based on loaded layers."""
        entries = [
            {
                "id": "basemap",
                "label": "Background map",
                "color": "#b0bec5",
                "shape": "rect",
                "checked": True,
            }
        ]

        if "depth-bitmap" in _loaded_data_layers:
            entries.append({
                "id": "depth-bitmap",
                "label": "Bathymetry",
                "colors": [
                    [1, 31, 75], [3, 56, 108], [15, 94, 156],
                    [46, 134, 193], [86, 180, 233], [166, 216, 247],
                ],
                "shape": "rect",
                "checked": True,
            })

        if "foraging-bitmap" in _loaded_data_layers:
            foraging_lyr = _layer_cache.get("foraging-bitmap", {})
            entries.append({
                "id": "foraging-bitmap",
                "label": "Foraging",
                "colors": [
                    [8, 48, 20], [20, 100, 40], [40, 160, 60],
                    [80, 200, 80], [140, 230, 100], [200, 255, 140],
                ],
                "shape": "rect",
                "checked": foraging_lyr.get("visible") is not False,
            })

        if "noise-construction" in _loaded_data_layers:
            entries.append({
                "id": "noise-construction",
                "label": "Construction noise",
                "color": "rgba(255,60,60,0.63)",
                "shape": "circle",
                "checked": True,
            })

        if "noise-operational" in _loaded_data_layers:
            entries.append({
                "id": "noise-operational",
                "label": "Operational noise",
                "color": "rgba(255,200,60,0.47)",
                "shape": "circle",
                "checked": True,
            })

        if "turbine-poles" in _loaded_data_layers:
            entries.append({
                "id": "turbine-poles",
                "label": "Wind turbines",
                "color": "rgb(50,160,240)",
                "shape": "rect",
                "checked": True,
            })

        if "porpoises" in _loaded_data_layers:
            entries.append({
                "id": "porpoises",
                "label": "Porpoises",
                "color": "rgb(0,150,255)",
                "shape": "circle",
                "checked": True,
            })

        if "porpoise-trails" in _loaded_data_layers:
            entries.append({
                "id": "porpoise-trails",
                "label": "Porpoise traces",
                "color": "rgb(0,150,255)",
                "shape": "rect",
                "checked": True,
            })

        await session.send_custom_message("cenop_legend_update", {"entries": entries})

    _porpoise_legend_sent = False
    _noise_legend_sent = False

    @reactive.effect
    async def _init_map():
        """Send initial empty layers, legend widget, and map controls on startup."""
        await _push_all_layers()
        await _sync_legend()

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
    ticks_per_update_value = [48]  # Default 48 ticks (1 day) per UI update
    ticks_lock = threading.Lock()  # Protects ticks_per_update_value access
    # Skip visualization flag for fast headless runs
    skip_viz_value = [False]
    skip_viz_lock = threading.Lock()
    # Shared trace settings for thread-safe updates
    trace_enabled_value = [False]
    trace_length_value = [2]  # days
    trace_lock = threading.Lock()

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
        from ..ui.sidebar import LANDSCAPE_TURBINE_COMPATIBILITY

        # Get current landscape with error handling
        landscape = _safe_input(input, "landscape", "Homogeneous")

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
        _ = _safe_input(input, "refresh_landscapes")

        try:
            from cenop.landscape.loader import LandscapeLoader
            lands = LandscapeLoader.list_landscapes()
            logger.info(f"Landscape selector: found {len(lands)} landscapes")
        except Exception as e:
            lands = []
            logger.error(f"Could not list landscapes: {e}", exc_info=True)

        choices = ["Homogeneous"] + sorted(lands)
        
        # Keep currently selected landscape if still available
        current = _safe_input(input, "landscape")
            
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
        except AttributeError:
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
        _ = _safe_input(input, "refresh_data_available")
        refresh_clicked = _ is not None

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
                except AttributeError:
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
        name = _safe_input(input, "detail_landscape")
        if not name:
            return

        try:
            from cenop.landscape.loader import LandscapeLoader
            loader = LandscapeLoader(name)
            info = loader.list_files()
            # load_all may return loader warnings (heavy operation but single-landscape)
            loaded = loader.load_all()
            warnings = loaded.get('warnings', [])
        except Exception as e:
            logger.error("Error loading landscape details for %s: %s", name, e, exc_info=True)
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
        logger.info("start_simulation() TRIGGERED")
        if state.running():
            logger.info("Already running, skipping")
            return

        try:
            logger.info("Creating simulation from inputs...")
            # Import simulation controller (always import fresh to avoid stale references)
            from .simulation_controller import create_simulation_from_inputs as create_sim, SimulationRunner as Runner
            sim = create_sim(input)
            logger.info("Simulation created, is_initialized=%s", sim._is_initialized)
            sim.initialize()
            logger.info("Simulation initialized, pop_size=%s, max_ticks=%s", sim.population_size, sim.max_ticks)

            runner = Runner(sim)
            logger.info("SimulationRunner created, max_ticks=%s", runner.max_ticks)
        except Exception as exc:
            logger.exception("Failed to create/initialize simulation")
            state.progress_message.set(f"Error: {exc}")
            ui.notification_show(f"Simulation failed: {exc}", type="error", duration=10)
            return

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

        # Update ticks per update from slider value (may not exist in UI)
        ticks_val = _safe_input(input, "ticks_per_update", ticks_per_update_value[0])
        with ticks_lock:
            ticks_per_update_value[0] = ticks_val

        # Start background thread
        sim_thread = threading.Thread(
            target=run_simulation_loop,
            args=(
                runner, result_queue, stop_event,
                throttle_value, throttle_lock,
                ticks_per_update_value, ticks_lock,
                trace_enabled_value, trace_length_value, trace_lock,
                skip_viz_value, skip_viz_lock,
            ),
            daemon=True,
        )
        sim_thread.start()
        logger.info("Simulation thread started")

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
        logger.debug("Speed slider changed: %s%% -> throttle=%.3f", speed_percent, new_throttle)
    
    @reactive.effect
    @reactive.event(input.ticks_per_update)
    def update_ticks_per_update():
        """Update ticks per update when slider changes during simulation."""
        ticks_val = input.ticks_per_update()
        with ticks_lock:
            ticks_per_update_value[0] = ticks_val
        logger.debug("Ticks per update changed: %s", ticks_val)

    @reactive.effect
    def _sync_skip_viz():
        """Sync skip-visualization toggle to thread-safe flag."""
        enabled = (
            input.skip_viz()
            if hasattr(input, "skip_viz")
            else False
        )
        with skip_viz_lock:
            skip_viz_value[0] = bool(enabled)

    @reactive.effect
    def _sync_trace_settings():
        """Sync trace toggle and slider to thread-safe flags."""
        enabled = (
            input.show_traces()
            if hasattr(input, "show_traces")
            else False
        )
        days = (
            input.trace_length_days()
            if hasattr(input, "trace_length_days")
            else 2
        )
        with trace_lock:
            trace_enabled_value[0] = bool(enabled)
            trace_length_value[0] = int(days) if days else 2

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
                logger.debug("Poll: Simulation COMPLETE, final history len=%d", len(state.population_history()))
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
                        except AttributeError:
                            pass
                    if msg.get("porpoise_trails") is not None:
                        try:
                            state.porpoise_trails.set(
                                msg.get("porpoise_trails")
                            )
                        except AttributeError:
                            pass

        # Flush batched entries to reactive state so dashboard updates
        if entries_batch:
            current_hist = state.population_history()
            state.population_history.set(current_hist + entries_batch)
        if energy_entries_batch:
            current_energy = state.energy_history()
            state.energy_history.set(current_energy + energy_entries_batch)
        if dispersal_entries_batch:
            current_dispersal = state.dispersal_history()
            state.dispersal_history.set(current_dispersal + dispersal_entries_batch)

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
        """Porpoise Population Size chart — lightweight SVG."""
        history = state.population_history()
        if not history:
            return no_data_placeholder()

        df = pd.DataFrame(history)
        if 'tick' not in df.columns or 'population' not in df.columns:
            return no_data_placeholder("Missing required data columns")

        return create_svg_chart(
            df=df, x_col='tick',
            y_cols=['population', 'lact_calf'],
            colors=['#2563eb', '#dc2626'],
            names=['Total Count', 'Lactating + Calf'],
            title='Porpoise Population Size',
        )

    @render.ui
    def life_death_plot():
        """Life and Death chart — lightweight SVG."""
        history = state.population_history()
        if not history:
            return no_data_placeholder()

        df = pd.DataFrame(history)
        df['daily_births'] = df['births'].diff().fillna(0)
        df['daily_deaths'] = df['deaths'].diff().fillna(0)

        return create_svg_chart(
            df=df, x_col='tick',
            y_cols=['daily_births', 'daily_deaths'],
            colors=['#2563eb', '#dc2626'],
            names=['Births', 'Deaths'],
            title='Life and Death',
        )

    @render.ui
    def energy_balance_plot():
        """Food consumption and expenditure chart — lightweight SVG."""
        history = state.energy_history()
        if not history:
            return no_data_placeholder("No energy data yet.")

        df = pd.DataFrame(history)
        return create_svg_chart(
            df=df, x_col='day',
            y_cols=['avg_food_eaten', 'avg_energy_expended'],
            colors=['#2563eb', '#dc2626'],
            names=['Avg Food Eaten', 'Avg Energy Expended'],
            title='Food Consumption and Expenditure',
        )
    
    @reactive.effect
    @reactive.event(state.landscape_load_counter)
    async def _update_depth_layer():
        """Rebuild depth bitmap when landscape is loaded."""
        loaded_name = state.landscape_loaded_name()
        if not loaded_name:
            _layer_cache["depth-bitmap"] = bitmap_layer("depth-bitmap", "", [], visible=False)
            _layer_cache["depth-tooltip"] = scatterplot_layer("depth-tooltip", [], visible=False)
            _loaded_data_layers.discard("depth-bitmap")
            return

        try:
            if loaded_name == "Homogeneous":
                from cenop.landscape import create_homogeneous_landscape
                landscape = create_homogeneous_landscape()
            else:
                from cenop.landscape import CellData
                landscape = CellData(loaded_name)
                landscape.load()

            depth = landscape._depth
            if depth is None:
                _layer_cache["depth-bitmap"] = bitmap_layer("depth-bitmap", "", [], visible=False)
                _layer_cache["depth-tooltip"] = scatterplot_layer("depth-tooltip", [], visible=False)
                _loaded_data_layers.discard("depth-bitmap")
                return

            from cenop.ui.sidebar import LANDSCAPE_CRS, LANDSCAPE_BOUNDS
            source_crs = LANDSCAPE_CRS.get(loaded_name, "EPSG:3035")

            layers = build_grid_bitmap_layer(
                "depth", depth, landscape.metadata, source_crs, "viridis",
            )
            _layer_cache["depth-bitmap"] = layers[0]
            _layer_cache["depth-tooltip"] = layers[1]
            _loaded_data_layers.add("depth-bitmap")

            bounds = LANDSCAPE_BOUNDS.get(loaded_name, (54.5, 56.5, 19.5, 22.5))
            lat_min, lat_max, lon_min, lon_max = bounds
            center_lat = (lat_min + lat_max) / 2
            center_lon = (lon_min + lon_max) / 2
            await sim_map.fly_to(session, longitude=center_lon, latitude=center_lat, zoom=6)
            await _push_all_layers()
            await _sync_legend()
            logger.info("Depth bitmap rendered for '%s'", loaded_name)
        except Exception as e:
            logger.error(f"Error building depth layer: {e}", exc_info=True)
    
    @reactive.effect
    @reactive.event(state.landscape_load_counter)
    async def _update_foraging_layer():
        """Rebuild foraging bitmap when landscape is loaded."""
        loaded_name = state.landscape_loaded_name()
        if not loaded_name:
            _layer_cache["foraging-bitmap"] = bitmap_layer("foraging-bitmap", "", [], visible=False)
            _layer_cache["foraging-tooltip"] = scatterplot_layer("foraging-tooltip", [], visible=False)
            _loaded_data_layers.discard("foraging-bitmap")
            return

        try:
            if loaded_name == "Homogeneous":
                from cenop.landscape import create_homogeneous_landscape
                landscape = create_homogeneous_landscape()
            else:
                from cenop.landscape import CellData
                landscape = CellData(loaded_name)
                landscape.load()

            food = landscape._food_prob
            if food is None:
                _layer_cache["foraging-bitmap"] = bitmap_layer("foraging-bitmap", "", [], visible=False)
                _layer_cache["foraging-tooltip"] = scatterplot_layer("foraging-tooltip", [], visible=False)
                _loaded_data_layers.discard("foraging-bitmap")
                return

            from cenop.ui.sidebar import LANDSCAPE_CRS
            source_crs = LANDSCAPE_CRS.get(loaded_name, "EPSG:3035")

            layers = build_grid_bitmap_layer(
                "foraging", food, landscape.metadata, source_crs, "green",
            )
            # Foraging starts hidden — user can toggle via legend checkbox
            layers[0]["visible"] = False
            layers[1]["visible"] = False
            _layer_cache["foraging-bitmap"] = layers[0]
            _layer_cache["foraging-tooltip"] = layers[1]
            _loaded_data_layers.add("foraging-bitmap")

            await _push_all_layers()
            await _sync_legend()
            logger.info("Foraging bitmap rendered for '%s'", loaded_name)
        except Exception as e:
            logger.error(f"Error building foraging layer: {e}", exc_info=True)
    
    # (Ship data layer removed — out of scope for shiny-deckgl migration)
    
    @reactive.effect
    @reactive.event(state.turbine_load_counter)
    async def _update_turbine_layers():
        """Rebuild turbine layers when turbines are loaded."""
        loaded_name = state.turbine_loaded_name()
        if not loaded_name or loaded_name == "off":
            _layer_cache["turbine-poles"] = build_turbine_pole_layer([])
            _layer_cache["turbine-blades"] = build_turbine_blade_layer([])
            _layer_cache["_turbine_data_raw"] = []
            _loaded_data_layers.discard("turbine-poles")
            await _push_all_layers()
            await _sync_legend()
            return

        try:
            import os
            landscape_name = _safe_input(input, "landscape", "Homogeneous")

            base_paths = [
                os.path.join("data", "wind-farms"),
                os.path.join("data", "landscapes", landscape_name, "wind-farms"),
                os.path.join("landscapes", landscape_name, "wind-farms"),
            ]
            wf_dir = None
            for p in base_paths:
                if os.path.isdir(p):
                    wf_dir = p
                    break

            if not wf_dir:
                logger.warning(f"No wind-farms directory found for {landscape_name}")
                return

            turbine_file = os.path.join(wf_dir, f"{loaded_name}.txt")
            if not os.path.isfile(turbine_file):
                logger.warning(f"Turbine file not found: {turbine_file}")
                return

            import pandas as pd_local
            df = pd_local.read_csv(turbine_file, sep=r'\s+')

            from cenop.ui.sidebar import LANDSCAPE_CRS
            source_crs = LANDSCAPE_CRS.get(landscape_name, "EPSG:3035")
            try:
                transformer = _get_transformer(source_crs)
            except (ImportError, RuntimeError) as e:
                logger.error("pyproj not available for coordinate transform: %s", e)
                return

            turbine_data = []
            for _, row in df.iterrows():
                x = float(row.get("x", row.iloc[0]))
                y = float(row.get("y", row.iloc[1]))
                lon, lat = transformer.transform(x, y)
                impact = float(row.get("impact", 234))
                start = int(row.get("start", 0))
                end = int(row.get("end", 0))
                turbine_data.append({
                    "position": [lon, lat],
                    "impact": impact,
                    "start": start,
                    "end": end,
                    "radius": 600,
                    "phase": "planned",
                    "color": [255, 100, 50, 220],
                    "layerType": "Turbine",
                    "info": f"Impact: {impact} dB",
                })

            state.turbine_count.set(len(turbine_data))
            _layer_cache["_turbine_data_raw"] = turbine_data
            _layer_cache["turbine-poles"] = build_turbine_pole_layer(turbine_data)
            _loaded_data_layers.add("turbine-poles")
            animate = _safe_input(input, "blade_animation", True)
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                turbine_data, client_animated=animate
            )
            await _push_all_layers()
            await _sync_legend()
            logger.info(f"Turbine layers: {len(turbine_data)} turbines loaded")
        except Exception as e:
            logger.error(f"Error building turbine layers: {e}", exc_info=True)

    @reactive.effect
    @reactive.event(state.map_update_counter)
    async def _update_turbine_phases():
        """Update turbine phases based on current simulation tick."""
        raw = _layer_cache.get("_turbine_data_raw", [])
        if not raw:
            return

        sim = state.simulation()
        if sim is None:
            return

        try:
            current_tick = sim.state.tick
            updated = []
            for t in raw:
                phase = "planned"
                if t["start"] <= current_tick <= t["end"]:
                    phase = "construction"
                elif current_tick > t["end"] and t["end"] > 0:
                    phase = "operational"
                updated.append({**t, "phase": phase})

            _layer_cache["_turbine_data_raw"] = updated
            _layer_cache["turbine-poles"] = build_turbine_pole_layer(updated)
            animate = _safe_input(input, "blade_animation", True)
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                updated, client_animated=animate
            )
            await _push_dynamic_layers("turbine-poles", "turbine-blades")
        except Exception as e:
            logger.error(f"Error updating turbine phases: {e}", exc_info=True)

    @reactive.effect
    @reactive.event(state.turbine_load_counter, state.map_update_counter)
    async def _update_noise_layers():
        """Rebuild noise layers based on turbine phases."""
        raw = _layer_cache.get("_turbine_data_raw", [])
        if not raw:
            _layer_cache["noise-construction"] = build_noise_construction_layer([])
            _layer_cache["noise-operational"] = build_noise_operational_layer([])
            _loaded_data_layers.discard("noise-construction")
            _loaded_data_layers.discard("noise-operational")
            await _push_dynamic_layers("noise-construction", "noise-operational")
            return

        sim = state.simulation()
        current_tick = sim.state.tick if sim else 0

        construction_noise = []
        operational_noise = []

        for t in raw:
            start = t.get("start", 0)
            end = t.get("end", 0)
            impact = t.get("impact", 234)
            pos = t["position"]

            if start <= current_tick <= end:
                threshold = 158
                radius = 10 ** ((impact - threshold) / 20)
                construction_noise.append({
                    "position": pos,
                    "radius": radius,
                    "layerType": "Noise",
                    "info": f"Construction: {impact} dB, radius: {radius:.0f}m",
                })
            elif current_tick > end and end > 0:
                operational_noise.append({
                    "position": pos,
                    "radius": 500,
                    "layerType": "Noise",
                    "info": "Operational: 145 dB, radius: 500m",
                })

        _layer_cache["noise-construction"] = build_noise_construction_layer(construction_noise)
        _layer_cache["noise-operational"] = build_noise_operational_layer(operational_noise)
        if construction_noise:
            _loaded_data_layers.add("noise-construction")
        if operational_noise:
            _loaded_data_layers.add("noise-operational")
        await _push_dynamic_layers("noise-construction", "noise-operational")
        nonlocal _noise_legend_sent
        if not _noise_legend_sent and (construction_noise or operational_noise):
            _noise_legend_sent = True
            await _sync_legend()

    @reactive.effect
    @reactive.event(state.map_update_counter)
    async def _update_porpoise_layer():
        """Rebuild porpoise layer and push via partial_update on simulation tick."""
        positions_raw = state.porpoise_positions()
        if not positions_raw:
            _layer_cache["porpoises"] = build_porpoise_layer([])
            _loaded_data_layers.discard("porpoises")
            _layer_cache["porpoise-trails"] = (
                build_porpoise_trails_layer([])
            )
            _loaded_data_layers.discard("porpoise-trails")
            await _push_dynamic_layers("porpoises", "porpoise-trails")
            return

        try:
            points = []
            for p in positions_raw[:1000]:
                lon, lat = p[1], p[2]
                heading = p[4] if len(p) > 4 else 0
                age = p[5] if len(p) > 5 else 5
                is_disturbed = p[6] if len(p) > 6 else False

                if is_disturbed:
                    color = [255, 40, 40, 240]
                elif age < 2:
                    color = [60, 180, 75, 240]
                elif age < 12:
                    color = [0, 150, 255, 240]
                else:
                    color = [160, 160, 160, 240]

                points.append({
                    "position": [lon, lat],
                    "heading": heading,
                    "age": age,
                    "is_disturbed": is_disturbed,
                    "radius": 200,
                    "color": color,
                    "layerType": "Porpoise",
                    "info": f"Age: {age:.1f}y" if isinstance(age, float) else f"Age: {age}",
                })

            _layer_cache["porpoises"] = build_porpoise_layer(points)
            _loaded_data_layers.add("porpoises")

            # Build trail layer if traces enabled
            trails_raw = state.porpoise_trails()
            show_traces = False
            try:
                show_traces = input.show_traces()
            except Exception:
                pass
            if show_traces and trails_raw:
                pid_to_color = {}
                for p in positions_raw[:1000]:
                    pid = int(p[0])
                    age = p[5] if len(p) > 5 else 5
                    is_dispersing = p[6] if len(p) > 6 else False
                    if is_dispersing:
                        color = [255, 40, 40, 240]
                    elif age < 2:
                        color = [60, 180, 75, 240]
                    elif age < 12:
                        color = [0, 150, 255, 240]
                    else:
                        color = [160, 160, 160, 240]
                    pid_to_color[pid] = color
                colored_trails = []
                for trail in trails_raw[:1000]:
                    pid = trail.get("pid", -1)
                    colored_trails.append({
                        "path": trail["path"],
                        "color": pid_to_color.get(
                            pid, [0, 150, 255, 240]
                        ),
                    })
                _layer_cache["porpoise-trails"] = (
                    build_porpoise_trails_layer(colored_trails)
                )
                _loaded_data_layers.add("porpoise-trails")
            else:
                _layer_cache["porpoise-trails"] = (
                    build_porpoise_trails_layer([])
                )
                _loaded_data_layers.discard("porpoise-trails")
            await _push_dynamic_layers("porpoises", "porpoise-trails")
            nonlocal _porpoise_legend_sent
            if not _porpoise_legend_sent:
                _porpoise_legend_sent = True
                await _sync_legend()
        except Exception as e:
            logger.error(f"Error updating porpoise layer: {e}", exc_info=True)

    @reactive.effect
    @reactive.event(input.blade_animation, state.turbine_load_counter)
    async def _manage_blade_animation():
        """Start or stop client-side blade animation based on toggle."""
        from cenop.server.map_layers import BLADE_ANIMATION_JS, BLADE_ANIMATION_STOP_JS

        animate = input.blade_animation()
        raw = _layer_cache.get("_turbine_data_raw", [])
        has_operational = any(t.get("phase") == "operational" for t in raw)

        if animate and has_operational:
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(
                raw, client_animated=True
            )
            await _push_dynamic_layers("turbine-blades")
            await session.send_custom_message("eval_js", BLADE_ANIMATION_JS)
        else:
            _layer_cache["turbine-blades"] = build_turbine_blade_layer(raw, rotation=0)
            await _push_dynamic_layers("turbine-blades")
            await session.send_custom_message("eval_js", BLADE_ANIMATION_STOP_JS)
    
    # =========================================================================
    # Population Tab Renderers
    # =========================================================================
    
    @render.ui
    def age_histogram():
        """Age distribution histogram."""
        try:
            _ = state.population_history()
            sim = state.simulation()
            if sim is None:
                return no_data_placeholder("Run simulation to see age distribution.")

            ages = []
            if hasattr(sim, 'population_manager') and sim.population_manager is not None:
                pm = sim.population_manager
                if hasattr(pm, 'age') and hasattr(pm, 'active_mask'):
                    active = pm.active_mask
                    if np.any(active):
                        ages = pm.age[active].tolist()
            elif hasattr(sim, 'agents_df'):
                df = sim.agents_df
                if not df.empty and 'age' in df.columns:
                    ages = df['age'].tolist()

            if not ages:
                return no_data_placeholder("No age data available.")

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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("age_histogram error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering age histogram.")
    
    @render.ui
    def energy_histogram():
        """Energy level histogram."""
        try:
            _ = state.population_history()
            sim = state.simulation()
            if sim is None:
                return no_data_placeholder("Run simulation to see energy distribution.")

            energies = []
            if hasattr(sim, 'population_manager') and sim.population_manager is not None:
                pm = sim.population_manager
                if hasattr(pm, 'energy') and hasattr(pm, 'active_mask'):
                    active = pm.active_mask
                    if np.any(active):
                        energies = pm.energy[active].tolist()
            elif hasattr(sim, 'agents_df'):
                df = sim.agents_df
                if not df.empty and 'energy' in df.columns:
                    energies = df['energy'].tolist()

            if not energies:
                return no_data_placeholder("No energy data available.")

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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("energy_histogram error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering energy histogram.")
    
    @render.ui
    def landscape_energy_plot():
        """Landscape energy level over time - uses avg porpoise energy as proxy."""
        try:
            history = state.energy_history()
            if not history:
                return no_data_placeholder("Run simulation to see energy trends.")

            df = pd.DataFrame(history)
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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("landscape_energy_plot error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering energy data.")
    
    @render.ui
    def movement_plot():
        """Average porpoise movement chart - uses dispersal data."""
        try:
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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("movement_plot error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering movement data.")
    
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
        except (ValueError, TypeError, KeyError) as e:
            logger.warning("Vital stats table rendering failed: %s", e)
            return pd.DataFrame()
    
    # =========================================================================
    # Disturbance Tab Renderers
    # =========================================================================
    
    @render.ui
    def dispersal_plot():
        """Porpoise Dispersal chart."""
        try:
            history = state.dispersal_history()
            if not history:
                return no_data_placeholder("No dispersal data yet. Run simulation to see results.")

            df = pd.DataFrame(history)
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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("dispersal_plot error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering dispersal data.")
    
    @render.ui
    def deterrence_plot():
        """Deterrence events display."""
        try:
            history = state.population_history()
            if not history:
                return no_data_placeholder("Deterrence data will appear when simulation runs with turbines/ships.")

            deterred_data = [{'day': h['day'], 'deterred': h.get('deterred_count', 0)} for h in history]
            df = pd.DataFrame(deterred_data)

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
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("deterrence_plot error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering deterrence data.")
    
    @render.ui
    def noise_map():
        """Noise model info and exposure chart."""
        try:
            sim = state.simulation()
            loaded = state.turbine_loaded_name()
            history = state.population_history()

            # --- Noise Model Parameters ---
            if sim is not None:
                p = sim.params
            else:
                p = None

            beta = getattr(p, 'beta_hat', 20.0) if p else 20.0
            alpha = getattr(p, 'alpha_hat', 0.0) if p else 0.0
            threshold = getattr(p, 'deter_threshold', 158.0) if p else 158.0
            coeff = getattr(p, 'deter_coeff', 0.07) if p else 0.07
            max_dist = getattr(p, 'deter_max_distance', 50.0) if p else 50.0

            # --- Turbine Info ---
            turbine_info = ""
            if sim is not None and hasattr(sim, '_turbine_manager'):
                tm = sim._turbine_manager
                n_total = len(tm.turbines)
                if n_total > 0:
                    impacts = [t.impact for t in tm.turbines]
                    sl_min, sl_max = min(impacts), max(impacts)
                    # Deterrence radius at threshold
                    radius_m = 10 ** ((sl_max - threshold) / beta) if beta > 0 else 0
                    sl_str = f"{sl_min:.0f}" if sl_min == sl_max else f"{sl_min:.0f}-{sl_max:.0f}"
                    turbine_info = (
                        f'<tr><td style="padding:2px 8px;"><b>Turbines</b></td>'
                        f'<td style="padding:2px 8px;">{n_total} ({loaded})</td></tr>'
                        f'<tr><td style="padding:2px 8px;"><b>Source Level (SL)</b></td>'
                        f'<td style="padding:2px 8px;">{sl_str} dB re 1\u00b5Pa @ 1m</td></tr>'
                        f'<tr><td style="padding:2px 8px;"><b>Deterrence Radius</b></td>'
                        f'<td style="padding:2px 8px;">{radius_m/1000:.1f} km (where RL = {threshold:.0f} dB)</td></tr>'
                    )
            elif loaded and loaded != "off":
                turbine_info = (
                    f'<tr><td style="padding:2px 8px;"><b>Scenario</b></td>'
                    f'<td style="padding:2px 8px;">{loaded} (not yet simulated)</td></tr>'
                )

            model_html = f'''
            <div style="font-size:0.8rem; margin-bottom:8px;">
            <b>DEPONS Noise Propagation Model</b><br>
            <span style="font-family:monospace;">TL = \u03b2\u00b7log\u2081\u2080(r) + \u03b1\u00b7r &nbsp;&nbsp; RL = SL \u2212 TL</span>
            <table style="margin-top:4px; font-size:0.78rem;">
            <tr><td style="padding:2px 8px;"><b>Spreading (\u03b2)</b></td><td style="padding:2px 8px;">{beta}</td></tr>
            <tr><td style="padding:2px 8px;"><b>Absorption (\u03b1)</b></td><td style="padding:2px 8px;">{alpha}</td></tr>
            <tr><td style="padding:2px 8px;"><b>Deterrence Threshold</b></td><td style="padding:2px 8px;">{threshold:.0f} dB</td></tr>
            <tr><td style="padding:2px 8px;"><b>Deterrence Coeff</b></td><td style="padding:2px 8px;">{coeff}</td></tr>
            <tr><td style="padding:2px 8px;"><b>Max Distance</b></td><td style="padding:2px 8px;">{max_dist:.0f} km</td></tr>
            {turbine_info}
            </table>
            </div>
            '''

            # --- Exposure Chart ---
            if history:
                data = [{'day': h['day'], 'deterred': h.get('deterred_count', 0)} for h in history]
                df = pd.DataFrame(data)
                if df['deterred'].sum() > 0:
                    chart = create_time_series_chart(
                        df=df,
                        x_col='day',
                        y_cols=['deterred'],
                        colors=['#e74c3c'],
                        names=['Porpoises Exposed to Noise'],
                        title='Noise Exposure Over Time',
                        x_title='Day',
                        y_title='# Exposed',
                        height=220
                    )
                    return ui.div(ui.HTML(model_html), chart)

            return ui.HTML(model_html + '<p style="color:#888; font-size:0.8rem;">No noise exposure events recorded yet.</p>')
        except (ValueError, TypeError, IndexError, KeyError) as e:
            logger.error("noise_map error: %s", e, exc_info=True)
            return no_data_placeholder("Error rendering noise data.")
    
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

    # =========================================================================
    # GIS Landscape Editor
    # =========================================================================

    from .renderers.gis_editor import register_gis_editor_renderers
    register_gis_editor_renderers(input, output, session, state)
