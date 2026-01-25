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
            update = {
                "type": "update",
                "progress": runner.progress_percent,
                "entry": entry,
                "total_births": runner.total_births,
                "total_deaths": runner.total_deaths,
                "should_update_map": runner.should_update_map,
                "sim": runner.sim if runner.should_update_map else None
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