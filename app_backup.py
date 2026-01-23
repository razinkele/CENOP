"""
DEPYTHON - Shiny Application

Main entry point for the Python Shiny web interface.
Visualization design based on DEPONS (Disturbance Effects on the Harbour Porpoise Population in the North Sea).
"""

from shiny import App, render, ui, reactive
from shiny.types import ImgData
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import asyncio
from typing import Optional, List, Dict
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("DEPYTHON")
logger.setLevel(logging.DEBUG)

# Shinyswatch for theming
import shinyswatch

# Shinywidgets disabled - using Plotly for map instead (more reliable)
# from shinywidgets import output_widget, render_widget
# from ipyleaflet import Map, Marker, CircleMarker, LayerGroup, basemaps, TileLayer

# Import CENOP modules
from cenop import Simulation, SimulationParameters
from cenop.landscape import CellData, create_homogeneous_landscape
from cenop.parameters import SimulationConstants

logger.info("All imports successful")


# =============================================================================
# UI Definition  
# =============================================================================

# Custom CSS for better styling
CUSTOM_CSS = """
/* Progress bar styling */
.progress { height: 20px; }
.progress-bar { transition: width 0.3s ease-in-out; }

/* Card improvements */
.card { box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
.card-header { font-weight: 600; background-color: #f8f9fa; }

/* Value box styling */
.value-box { min-height: 100px; }

/* Sidebar styling */
.sidebar { background-color: #f8f9fa; }

/* Error display */
.shiny-output-error { color: #dc3545; }
.shiny-output-error:before { content: '⚠ '; }
"""

app_ui = ui.page_navbar(
    # Dashboard Tab
    ui.nav_panel(
        "Dashboard",
        ui.layout_column_wrap(
            ui.value_box(
                "Population",
                ui.output_text("current_population"),
                showcase=ui.span("🐬", style="font-size: 2rem;"),
                theme="primary"
            ),
            ui.value_box(
                "Year",
                ui.output_text("current_year"),
                showcase=ui.span("📅", style="font-size: 2rem;"),
                theme="info"
            ),
            ui.value_box(
                "Births",
                ui.output_text("total_births"),
                showcase=ui.span("🎂", style="font-size: 2rem;"),
                theme="success"
            ),
            ui.value_box(
                "Deaths",
                ui.output_text("total_deaths"),
                showcase=ui.span("💀", style="font-size: 2rem;"),
                theme="warning"
            ),
            width=1/4
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Spatial Distribution"),
                ui.output_ui("porpoise_map"),
                height="650px"
            ),
            ui.div(
                ui.card(
                    ui.card_header("Porpoise Population Size"),
                    ui.output_ui("population_plot"),
                    height="200px"
                ),
                ui.card(
                    ui.card_header("Life and Death"),
                    ui.output_ui("life_death_plot"),
                    height="200px"
                ),
                ui.card(
                    ui.card_header("Food Consumption and Expenditure"),
                    ui.output_ui("energy_balance_plot"),
                    height="200px"
                )
            ),
            col_widths=[7, 5]
        )
    ),
    
    # Model Settings Tab
    ui.nav_panel(
        "Model Settings",
        ui.navset_card_tab(
            ui.nav_panel(
                "Basic",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("🎯 Simulation Setup"),
                        ui.input_numeric("porpoise_count", "Initial Population", value=1000, min=1, max=50000, step=100),
                        ui.input_numeric("sim_years", "Simulation Years", value=5, min=1, max=100),
                        ui.input_select("landscape", "Landscape", 
                            choices=["Homogeneous", "NorthSea", "InnerDanishWaters", "Kattegat", "DanTysk", "Gemini", "UserDefined"], 
                            selected="Homogeneous"),
                        ui.input_numeric("random_seed", "Random Seed (0 = auto)", value=0, min=0),
                        ui.input_numeric("tracked_porpoise_count", "Tracked Porpoises", value=1, min=0, max=100),
                    ),
                    ui.card(
                        ui.card_header("⚠️ Disturbance & Threats"),
                        ui.input_select("turbines", "Wind Turbines", 
                            choices=["off", "NorthSea_scenario1", "NorthSea_scenario2", "NorthSea_scenario3", 
                                     "DanTysk-construction", "Gemini-construction", "User-def"], 
                            selected="off"),
                        ui.input_switch("ships_enabled", "Ship Traffic Enabled", value=False),
                        ui.input_numeric("bycatch_prob", "Annual Bycatch Probability", value=0.0, step=0.001, min=0.0, max=1.0),
                    ),
                    col_widths=[6, 6]
                )
            ),
            ui.nav_panel(
                "Movement",
                ui.card(
                    ui.card_header("🧭 Correlated Random Walk (CRW) Parameters"),
                    ui.p("These parameters control porpoise movement behavior.", class_="text-muted mb-3"),
                    ui.layout_column_wrap(
                        ui.input_numeric("param_k", "k - Inertia constant", value=0.001, step=0.001),
                        ui.input_numeric("param_a0", "a0 - AutoReg for log₁₀(d/100)", value=0.35, step=0.01),
                        ui.input_numeric("param_a1", "a1 - Water depth effect", value=0.0005, step=0.0001),
                        ui.input_numeric("param_a2", "a2 - Salinity effect", value=-0.02, step=0.01),
                        ui.input_numeric("param_b0", "b0 - AutoReg for turning", value=-0.024, step=0.001),
                        ui.input_numeric("param_b1", "b1 - Depth on turning", value=-0.008, step=0.001),
                        ui.input_numeric("param_b2", "b2 - Salinity on turning", value=0.93, step=0.01),
                        ui.input_numeric("param_b3", "b3 - Intercept", value=-14.0, step=1.0),
                        width=1/4
                    )
                )
            ),
            ui.nav_panel(
                "Dispersal",
                ui.layout_columns(
                    ui.card(
                        ui.card_header("🌊 Dispersal Settings"),
                        ui.input_select("dispersal", "Dispersal Type", 
                            choices=["off", "PSM-Type2", "Undirected", "InnerDanishWaters"], 
                            selected="PSM-Type2"),
                        ui.input_numeric("tdisp", "Days to Dispersal (tDisp)", value=3, min=1),
                    ),
                    ui.card(
                        ui.card_header("📢 PSM Parameters (Porpoise Scare Model)"),
                        ui.input_numeric("psm_log", "PSM_log - Logistic increase", value=0.6, step=0.1),
                        ui.input_text("psm_dist", "PSM_dist - Preferred distance", value="N(300;100)"),
                        ui.input_numeric("psm_tol", "PSM_tol - Tolerance (km)", value=5.0, step=0.5),
                        ui.input_numeric("psm_angle", "PSM_angle - Max turn (deg)", value=20.0, step=1.0),
                    ),
                    col_widths=[6, 6]
                )
            ),
            ui.nav_panel(
                "Energy",
                ui.card(
                    ui.card_header("⚡ Energy & Memory Parameters"),
                    ui.p("Memory decay rates and food replenishment.", class_="text-muted mb-3"),
                    ui.layout_column_wrap(
                        ui.input_numeric("param_rS", "rS - Satiation memory decay", value=0.04, step=0.01),
                        ui.input_numeric("param_rR", "rR - Reference memory decay", value=0.04, step=0.01),
                        ui.input_numeric("param_rU", "rU - Food replenishment rate", value=0.1, step=0.01),
                        width=1/3
                    )
                )
            )
        )
    ),
    
    # Population Tab
    ui.nav_panel(
        "Population",
        ui.layout_columns(
            ui.card(
                ui.card_header("Porpoise Age Distribution (0-30 years)"),
                ui.output_ui("age_histogram"),
                height="320px"
            ),
            ui.card(
                ui.card_header("Energy Level Distribution (0-20)"),
                ui.output_ui("energy_histogram"),
                height="320px"
            ),
            col_widths=[6, 6]
        ),
        ui.layout_columns(
            ui.card(
                ui.card_header("Landscape Energy Level"),
                ui.output_ui("landscape_energy_plot"),
                height="320px"
            ),
            ui.card(
                ui.card_header("Average Porpoise Movement"),
                ui.output_ui("movement_plot"),
                height="320px"
            ),
            col_widths=[6, 6]
        ),
        ui.card(
            ui.card_header("Vital Statistics"),
            ui.output_data_frame("vital_stats_table"),
            height="250px"
        )
    ),
    
    # Disturbance Tab
    ui.nav_panel(
        "Disturbance",
        ui.layout_columns(
            ui.card(
                ui.card_header("Porpoise Dispersal"),
                ui.output_ui("dispersal_plot"),
                height="350px"
            ),
            ui.card(
                ui.card_header("Deterrence Events"),
                ui.output_ui("deterrence_plot"),
                height="350px"
            ),
            col_widths=[6, 6]
        ),
        ui.card(
            ui.card_header("Noise Exposure Map"),
            ui.output_ui("noise_map"),
            height="350px"
        )
    ),
    
    # Export Tab
    ui.nav_panel(
        "Export",
        ui.layout_columns(
            ui.card(
                ui.card_header("📊 Export Simulation Data"),
                ui.p("Download simulation results in CSV format."),
                ui.download_button("download_data", "Download Results CSV", class_="btn-success btn-lg"),
            ),
            ui.card(
                ui.card_header("ℹ️ About"),
                ui.p("DEPYTHON - Python implementation of DEPONS"),
                ui.p("Disturbance Effects on the Harbour Porpoise Population in the North Sea", class_="text-muted"),
            ),
            col_widths=[6, 6]
        )
    ),
    
    # Sidebar with controls
    sidebar=ui.sidebar(
        ui.h5("🎮 Simulation Control", class_="mb-3"),
        
        # Progress section
        ui.div(
            ui.output_ui("progress_bar"),
            ui.output_text("progress_text", inline=True),
            class_="mb-3"
        ),
        
        # Run controls
        ui.div(
            ui.input_action_button("run_sim", "▶ Run Simulation", class_="btn-primary w-100 mb-2"),
            ui.input_action_button("stop_sim", "⏹ Stop", class_="btn-danger w-100 mb-2"),
            ui.input_action_button("reset_sim", "🔄 Reset", class_="btn-secondary w-100"),
            class_="mb-3"
        ),
        
        ui.tags.hr(),
        
        ui.p("Configure parameters in 'Model Settings' tab before running.", class_="text-muted small"),
        
        width=260,
        bg="#f8f9fa"
    ),
    
    # Page settings
    title="DEPYTHON",
    theme=shinyswatch.theme.flatly,
    header=ui.tags.style(CUSTOM_CSS),
    fillable=True
)


# =============================================================================
# Server Logic
# =============================================================================

def server(input, output, session):
    """Server function for Shiny app."""
    print("=== SERVER FUNCTION CALLED ===", flush=True)
    logger.info("=== SERVER FUNCTION CALLED ===")
    
    # Reactive values
    simulation = reactive.Value(None)
    running = reactive.Value(False)
    population_history = reactive.Value([])
    progress = reactive.Value(0.0)
    progress_message = reactive.Value("Ready to run")
    
    # Extended history for detailed charts
    energy_history = reactive.Value([])
    movement_history = reactive.Value([])
    dispersal_history = reactive.Value([])
    
    # Track totals
    birth_count = reactive.Value(0)
    death_count = reactive.Value(0)
    
    # Counter to force map updates (increment to trigger re-render)
    map_update_counter = reactive.Value(0)
    
    @render.text
    def progress_text():
        msg = progress_message()
        print(f"RENDER progress_text: {msg}", flush=True)
        return msg
    
    def create_simulation() -> Simulation:
        """Create a new simulation with current parameters."""
        seed_value = input.random_seed()
        
        # Parse PSM Dist string "N(300;100)"
        psm_dist_str = input.psm_dist()
        psm_dist_mean = 300.0
        psm_dist_sd = 100.0
        try:
            # Simple parsing for "N(mean;sd)"
            if psm_dist_str and psm_dist_str.startswith("N(") and psm_dist_str.endswith(")"):
                inner = psm_dist_str[2:-1]
                parts = inner.split(";")
                if len(parts) == 2:
                    psm_dist_mean = float(parts[0])
                    psm_dist_sd = float(parts[1])
        except Exception:
            pass # Use defaults
            
        params = SimulationParameters(
            porpoise_count=input.porpoise_count(),
            sim_years=input.sim_years(),
            landscape=input.landscape(),
            turbines=input.turbines(),
            ships_enabled=input.ships_enabled(),
            dispersal=input.dispersal(),
            random_seed=seed_value if seed_value > 0 else None,
            
            # Advanced Parameters
            tracked_porpoise_count=input.tracked_porpoise_count(),
            t_disp=input.tdisp(),
            psm_log=input.psm_log(),
            psm_dist_mean=psm_dist_mean,
            psm_dist_sd=psm_dist_sd,
            psm_tol=input.psm_tol(),
            psm_angle=input.psm_angle(),
            
            # Memory & Energy
            r_s=input.param_rS(),
            r_r=input.param_rR(),
            r_u=input.param_rU(),
            
            # Survival
            bycatch_prob=input.bycatch_prob(),
            
            # Movement Coefficients (CRW)
            inertia_const=input.param_k(),
            corr_logmov_length=input.param_a0(),
            corr_logmov_bathy=input.param_a1(),
            corr_logmov_salinity=input.param_a2(),
            corr_angle_base=input.param_b0(),
            corr_angle_bathy=input.param_b1(),
            corr_angle_salinity=input.param_b2(),
            corr_angle_base_sd=input.param_b3(),
        )
        
        # Create landscape
        if params.is_homogeneous:
            landscape = create_homogeneous_landscape()
        else:
            landscape = CellData(params.landscape)
        
        sim = Simulation(params, cell_data=landscape)
        logger.info(f"Simulation created: {params.porpoise_count} porpoises, {params.sim_years} years")
        return sim
    
    # Simulation state (using plain Python to avoid reactive cycles)
    _sim = [None]  # Active simulation instance
    _tick = [0]    # Current tick
    _max_ticks = [0]  # Total ticks
    _history = [[]]  # Population history
    _total_births = [0]
    _total_deaths = [0]
    _last_pop = [0]  # Track population for death counting
    _update_count = [0]  # Counter for map update throttling
    
    @reactive.effect
    @reactive.event(input.run_sim)
    def start_simulation():
        """Start the simulation by initializing and resetting."""
        if running():
            return
        
        # Create and initialize simulation
        sim = create_simulation()
        sim.initialize()
        
        _sim[0] = sim
        _tick[0] = 0
        _max_ticks[0] = sim.max_ticks
        _history[0] = []
        _total_births[0] = 0
        _total_deaths[0] = 0
        _last_pop[0] = sim.state.population
        
        # Store simulation in reactive value for rendering
        simulation.set(sim)
        running.set(True)
        population_history.set([])
        birth_count.set(0)
        death_count.set(0)
        progress.set(0.0)
        progress_message.set("Running simulation...")
    
    @reactive.effect
    def advance_simulation():
        """Advance simulation one day at a time using reactive polling."""
        if not running():
            return
        
        sim = _sim[0]
        if sim is None:
            running.set(False)
            return
        
        tick = _tick[0]
        max_ticks = _max_ticks[0]
        
        if tick >= max_ticks:
            running.set(False)
            years = sim.state.year
            progress_message.set(f"Complete! {years} years simulated")
            progress.set(100.0)
            return
        
        # Run 48 ticks (one day) per UI update for performance
        ticks_per_update = 48
        for _ in range(ticks_per_update):
            if _tick[0] >= max_ticks:
                break
            sim.step()
            _tick[0] += 1
        
        # Update reactive values from simulation state
        pct = (_tick[0] / max_ticks) * 100
        progress.set(pct)
        
        # Track births and deaths based on population changes
        current_pop = sim.state.population
        last_pop = _last_pop[0]
        
        # Count lactating porpoises with calves (these represent births)
        lact_calf_count = 0
        if sim._porpoises:
            lact_calf_count = sum(1 for p in sim._porpoises if hasattr(p, 'with_lact_calf') and p.with_lact_calf)
        
        # Deaths = population decrease (when pop decreases, porpoises died)
        if current_pop < last_pop:
            _total_deaths[0] += (last_pop - current_pop)
        # Births = population increase OR new lactating mothers
        if current_pop > last_pop:
            _total_births[0] += (current_pop - last_pop)
        
        _last_pop[0] = current_pop
        
        birth_count.set(_total_births[0])
        death_count.set(_total_deaths[0])
        
        progress_message.set(f"Year {sim.state.year}, Day {sim.state.day % 360}")
        
        _history[0].append({
            'day': sim.state.day,
            'tick': sim.state.tick,
            'year': sim.state.year,
            'population': current_pop,
            'lact_calf': lact_calf_count,
            'births': _total_births[0],
            'deaths': _total_deaths[0]
        })
        population_history.set(_history[0].copy())
        
        # Update simulation reactive value for other renderers
        simulation.set(sim)
        
        # Only update map every 10 steps to prevent spinner flashing
        _update_count[0] += 1
        if _update_count[0] % 10 == 0:
            map_update_counter.set(map_update_counter() + 1)
        
        # Schedule next update
        reactive.invalidate_later(0.05)
    
    @reactive.effect
    @reactive.event(input.stop_sim)
    def stop_simulation():
        """Stop the running simulation."""
        running.set(False)
    
    @reactive.effect
    @reactive.event(input.reset_sim)
    def reset_simulation():
        """Reset the simulation."""
        running.set(False)
        simulation.set(None)
        population_history.set([])
        energy_history.set([])
        movement_history.set([])
        dispersal_history.set([])
        progress.set(0.0)
        progress_message.set("Ready to run")
        birth_count.set(0)
        death_count.set(0)
    
    # === Output Renderers ===
    

    @render.ui
    def progress_bar():
        print("RENDER progress_bar called", flush=True)
        try:
            pct = progress()
            is_running = running()
            print(f"RENDER progress_bar: pct={pct}, running={is_running}", flush=True)
            # Color changes based on progress
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
        except Exception as e:
            logger.error(f"ERROR in progress_bar: {e}")
            return ui.p(f"Error: {e}", class_="text-danger")
    
    # progress_text is now defined earlier in server function

    # Value box renders - using @render.text which is correct for value_box
    @render.text
    def current_population():
        print("RENDER current_population called", flush=True)
        sim = simulation()
        if sim:
            count = sim.state.population
            print(f"RENDER current_population: {count}", flush=True)
            return str(count)
        print("RENDER current_population: 0 (no sim)", flush=True)
        return "0"
    
    @render.text
    def current_year():
        sim = simulation()
        year = sim.state.year if sim else 0
        print(f"RENDER current_year: {year}", flush=True)
        return str(year)
    
    @render.text
    def total_births():
        b = birth_count()
        print(f"RENDER total_births: {b}", flush=True)
        return str(b)
    
    @render.text
    def total_deaths():
        d = death_count()
        print(f"RENDER total_deaths: {d}", flush=True)
        return str(d)
    

    @render.ui
    def population_plot():
        """Porpoise Population Size - matching DEPONS chart with Count and LactCalf."""
        print("RENDER population_plot called", flush=True)
        try:
            history = population_history()
            if not history:
                print("RENDER population_plot: no data yet", flush=True)
                return ui.p("No data yet. Run simulation to see results.", class_="text-muted text-center mt-3")
        except Exception as e:
            logger.error(f"ERROR in population_plot: {e}")
            return ui.p(f"Error: {e}", class_="text-danger")
            
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Total population (blue)
        fig.add_trace(go.Scatter(
            x=df['tick'],
            y=df['population'],
            mode='lines',
            name='Total Count',
            line=dict(color='blue', width=2)
        ))
        
        # Lactating with calf (red) - only if column exists
        if 'lact_calf' in df.columns:
            fig.add_trace(go.Scatter(
                x=df['tick'],
                y=df['lact_calf'],
                mode='lines',
                name='Lactating + Calf',
                line=dict(color='red', width=2)
            ))
        
        fig.update_layout(
            title='Porpoise Population Size',
            xaxis_title='Tick Count',
            yaxis_title='Population Size',
            height=180,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=20, t=30, b=30),
            plot_bgcolor='rgb(192, 192, 192)',
            paper_bgcolor='white'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    
    @render.ui
    def porpoise_map():
        """Spatial distribution map using Plotly (more reliable than ipyleaflet)."""
        # Depend on map_update_counter to force re-renders
        _ = map_update_counter()
        
        sim = simulation()
        
        # North Sea bounding box
        lat_min, lat_max = 53.0, 58.0
        lon_min, lon_max = 3.0, 12.0
        center_lat, center_lon = 55.5, 7.5
        
        if sim is not None and hasattr(sim, '_porpoises') and sim._porpoises:
            # Get world dimensions for coordinate scaling
            world_width = getattr(sim.params, 'world_width', 1000)
            world_height = getattr(sim.params, 'world_height', 1000)
            
            # Collect porpoise positions
            lats = []
            lons = []
            for p in sim._porpoises[:1000]:  # Limit for performance
                if hasattr(p, 'alive') and p.alive:
                    lat = lat_min + (p.y / world_height) * (lat_max - lat_min)
                    lon = lon_min + (p.x / world_width) * (lon_max - lon_min)
                    lats.append(lat)
                    lons.append(lon)
            
            if lats:
                # Create scatter plot on map
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
        else:
            fig = go.Figure()
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=center_lat, lon=center_lon),
                zoom=5
            ),
            margin=dict(l=0, r=0, t=0, b=0),
            height=650,
            showlegend=False
        )
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))

    @render.ui
    def spatial_plot():
        """Fallback spatial plot using plotly (kept for backup)."""
        return ui.p("Using map above.", class_="text-muted text-center mt-5")
    

    @render.ui
    def life_death_plot():
        """Life and Death chart matching DEPONS."""
        history = population_history()
        if not history:
            return ui.p("No data yet.", class_="text-muted text-center mt-5")
            
        df = pd.DataFrame(history)
        
        # Calculate daily births/deaths from cumulative
        df['daily_births'] = df['births'].diff().fillna(0)
        df['daily_deaths'] = df['deaths'].diff().fillna(0)
        
        fig = go.Figure()
        
        # Births (blue)
        fig.add_trace(go.Scatter(
            x=df['tick'],
            y=df['daily_births'],
            mode='lines',
            name='Births',
            line=dict(color='blue', width=2)
        ))
        
        # Deaths (red)
        fig.add_trace(go.Scatter(
            x=df['tick'],
            y=df['daily_deaths'],
            mode='lines',
            name='Deaths',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Life and Death',
            xaxis_title='Tick Count',
            yaxis_title='Count',
            height=180,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=20, t=30, b=30),
            plot_bgcolor='rgb(192, 192, 192)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def energy_balance_plot():
        """Food consumption and expenditure matching DEPONS."""
        history = energy_history()
        if not history:
            return ui.p("No energy data yet.", class_="text-muted text-center mt-5")
            
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Average food eaten (blue)
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['avg_food_eaten'],
            mode='lines',
            name='Avg Food Eaten',
            line=dict(color='blue', width=2)
        ))
        
        # Average energy expended (red)
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['avg_energy_expended'],
            mode='lines',
            name='Avg Energy Expended',
            line=dict(color='red', width=2)
        ))
        
        fig.update_layout(
            title='Food Consumption and Expenditure',
            xaxis_title='Day',
            yaxis_title='Energy',
            height=180,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=20, t=30, b=30),
            plot_bgcolor='rgb(192, 192, 192)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def age_histogram():
        """Age distribution histogram matching DEPONS (0-30 years, 30 bins)."""
        sim = simulation()
        if sim is None:
            return ui.p("No data available.", class_="text-muted text-center mt-5")
        
        agents_list = list(sim.agents) if sim.agents else []
        ages = [a.age for a in agents_list if hasattr(a, 'age')]
        if not ages:
            return ui.p("No age data.", class_="text-muted text-center mt-5")
            
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=ages,
            nbinsx=30,
            xbins=dict(start=0, end=30, size=1),
            marker_color='red',
            name='Age'
        ))
        
        fig.update_layout(
            title='Porpoise Age Distribution',
            xaxis_title='Age (years)',
            yaxis_title='Count',
            height=300,
            xaxis=dict(range=[0, 30]),
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgb(192, 192, 192)',
            bargap=0.1
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def energy_histogram():
        """Energy level histogram matching DEPONS (0-20, 20 bins)."""
        sim = simulation()
        if sim is None:
            return ui.p("No data available.", class_="text-muted text-center mt-5")
        
        agents_list = list(sim.agents) if sim.agents else []
        energies = [getattr(a, 'energy_level', 0) for a in agents_list]
        if not energies:
            return ui.p("No energy data.", class_="text-muted text-center mt-5")
            
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=energies,
            nbinsx=20,
            xbins=dict(start=0, end=20, size=1),
            marker_color='red',
            name='Energy'
        ))
        
        fig.update_layout(
            title='Energy Level Distribution',
            xaxis_title='Energy',
            yaxis_title='Porpoise Count',
            height=300,
            xaxis=dict(range=[0, 20]),
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgb(192, 192, 192)',
            bargap=0.1
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def landscape_energy_plot():
        """Landscape energy level over time matching DEPONS."""
        history = energy_history()
        if not history:
            return ui.p("No landscape data yet.", class_="text-muted text-center mt-5")
            
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['landscape_energy'],
            mode='lines',
            name='Landscape Energy',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Landscape Energy Level',
            xaxis_title='Day',
            yaxis_title='Energy',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgb(192, 192, 192)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def movement_plot():
        """Average Porpoise Movement matching DEPONS."""
        history = movement_history()
        if not history:
            return ui.p("No movement data yet.", class_="text-muted text-center mt-5")
            
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['day'],
            y=df['avg_daily_movement'],
            mode='lines',
            name='Average Daily Movement',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Average Porpoise Movement',
            xaxis_title='Day',
            yaxis_title='Moved Cells Daily',
            height=300,
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgb(192, 192, 192)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.ui
    def dispersal_plot():
        """Porpoise Dispersal chart matching DEPONS."""
        history = dispersal_history()
        if not history:
            return ui.p("No dispersal data yet.", class_="text-muted text-center mt-5")
            
        df = pd.DataFrame(history)
        
        fig = go.Figure()
        
        # Dispersing levels with different colors
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['dispersing_1'],
            mode='lines', name='Dispersing 1',
            line=dict(color='blue', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['dispersing_2'],
            mode='lines', name='Dispersing 2',
            line=dict(color='green', width=2)
        ))
        fig.add_trace(go.Scatter(
            x=df['day'], y=df['dispersing_3'],
            mode='lines', name='Dispersing 3',
            line=dict(color='orange', width=2)
        ))
        
        fig.update_layout(
            title='Porpoise Dispersal',
            xaxis_title='Day',
            yaxis_title='# Porpoises',
            height=350,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
            margin=dict(l=50, r=20, t=40, b=40),
            plot_bgcolor='rgb(192, 192, 192)'
        )
        fig.update_xaxes(showgrid=True, gridcolor='white')
        fig.update_yaxes(showgrid=True, gridcolor='white')
        
        return ui.HTML(fig.to_html(include_plotlyjs="cdn", full_html=False))
    

    @render.data_frame
    def vital_stats_table():
        sim = simulation()
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
    

    @render.ui
    def deterrence_plot():
        """Deterrence events over time."""
        sim = simulation()
        if sim is None:
            return ui.p("Deterrence data will appear when turbines/ships are active.", 
                       class_="text-muted text-center mt-5")
        
        agents_list = list(sim.agents) if sim.agents else []
        
        # Check if any deterrence is happening
        deterred = sum(1 for a in agents_list if getattr(a, 'deter_strength', 0) > 0)
        if deterred == 0:
            return ui.p("No deterrence events detected. Enable turbines or ships.", 
                       class_="text-muted text-center mt-5")
        
        return ui.p(f"Currently deterred: {deterred} porpoises", class_="text-center mt-5")
    

    @render.ui
    def noise_map():
        """Noise exposure map placeholder."""
        return ui.p("Noise exposure map will appear when disturbance sources are active.",
                   class_="text-muted text-center mt-5")
    
    @render.download(filename="depython_results.csv")
    def download_data():
        history = population_history()
        if history:
            df = pd.DataFrame(history)
            return df.to_csv(index=False)
        return ""


# =============================================================================
# Create App
# =============================================================================

app = App(app_ui, server)


if __name__ == "__main__":
    from shiny import run_app
    run_app(app)
