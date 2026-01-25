# DEPONS to Python Shiny Translation Plan

## Executive Summary

This document outlines a comprehensive plan to translate the DEPONS (Disturbance Effects of POrpoises in the North Sea) agent-based model from Java/Repast Simphony to Python Shiny. The translation will maintain scientific fidelity while leveraging Python's data science ecosystem for enhanced visualization and accessibility.

---

## 1. Model Overview

### 1.1 What is DEPONS?

DEPONS is an agent-based model (ABM) that simulates how harbour porpoise population dynamics are affected by disturbances, specifically:
- **Pile-driving noise** from offshore wind farm construction
- **Ship noise** (Version 3.0+)

The model links individual porpoise behavior to population-level effects through:
- Energy-based survival mechanics
- Realistic movement patterns calibrated from satellite-tagged animals
- Noise-induced deterrence behavior
- Reproductive dynamics

### 1.2 Current Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Java |
| ABM Framework | Repast Simphony |
| Spatial Model | Continuous space + Grid overlay |
| Time Step | 30 minutes |
| Typical Simulation | 50 years (864,000 steps) |

---

## 2. Architecture Analysis

### 2.1 Core Components Identified

```
┌─────────────────────────────────────────────────────────────────┐
│                      DEPONS ARCHITECTURE                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                    │
│  │  Simulation     │    │   Landscape     │                    │
│  │  Engine         │    │   Data          │                    │
│  │  (Scheduler)    │    │   (CellData)    │                    │
│  └────────┬────────┘    └────────┬────────┘                    │
│           │                      │                              │
│           ▼                      ▼                              │
│  ┌─────────────────────────────────────────────┐               │
│  │              AGENT SYSTEM                    │               │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐        │               │
│  │  │Porpoise │ │ Turbine │ │  Ship   │        │               │
│  │  │ (1686   │ │ (258    │ │ (150+   │        │               │
│  │  │ lines)  │ │ lines)  │ │ lines)  │        │               │
│  │  └─────────┘ └─────────┘ └─────────┘        │               │
│  └─────────────────────────────────────────────┘               │
│           │                                                     │
│           ▼                                                     │
│  ┌─────────────────────────────────────────────┐               │
│  │           BEHAVIOR MODULES                   │               │
│  │  • Movement (CRW + Reference Memory)         │               │
│  │  • Dispersal (PSM Types 1-3)                 │               │
│  │  • Deterrence (Noise response)               │               │
│  │  • Energetics (Consumption/Usage)            │               │
│  │  • Reproduction (Mating/Birth/Nursing)       │               │
│  └─────────────────────────────────────────────┘               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 2.2 Key Classes and Their Responsibilities

| Java Class | Lines | Responsibility |
|------------|-------|----------------|
| `Porpoise.java` | 1,686 | Main agent: movement, energy, reproduction, deterrence |
| `SimulationParameters.java` | 779 | All configurable model parameters |
| `PorpoiseSimBuilder.java` | 405 | Model initialization and scheduling |
| `CellData.java` | 276 | Landscape data management |
| `Globals.java` | 211 | Global state and utilities |
| `RefMem.java` | 150 | Reference memory mechanics |
| `Dispersal*.java` | ~400 | Various dispersal behaviors (PSM Types) |
| `Turbine.java` | 258 | Wind turbine noise sources |
| `Ship.java` + related | ~500 | Ship movement and noise |

### 2.3 Key Parameters (from parameters.xml)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `porpoiseCount` | 10,000 | Initial population |
| `simYears` | 50 | Simulation duration |
| `landscape` | NorthSea | Study area |
| `turbines` | off | Wind farm scenario |
| `ships` | false | Enable ship noise |
| `Euse` | 4.5 | Energy use per 30-min step |
| `h` | 0.68 | Pregnancy probability |
| `RT` | 152.9 | Deterrence threshold (dB) |

---

## 3. Translation Strategy

### 3.1 Technology Stack for Python Implementation

```
┌─────────────────────────────────────────────────────────────────┐
│                 DEPYTHON TECHNOLOGY STACK                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  UI Layer:          Shiny for Python (shiny)                   │
│                     + shinyswatch (themes)                      │
│                     + shinywidgets (interactive plots)          │
│                                                                 │
│  Visualization:     Plotly (interactive maps/charts)           │
│                     Matplotlib (static plots)                   │
│                     Folium/Leaflet (GIS maps)                   │
│                                                                 │
│  Simulation Core:   Mesa (ABM framework) OR                    │
│                     Custom NumPy-based engine                   │
│                                                                 │
│  Spatial:           NumPy (arrays/grids)                       │
│                     SciPy (spatial operations)                  │
│                     GeoPandas (GIS data)                        │
│                     Rasterio (raster data I/O)                  │
│                                                                 │
│  Data:              Pandas (tabular data)                      │
│                     Xarray (multi-dimensional)                  │
│                                                                 │
│  Performance:       Numba (JIT compilation)                    │
│                     Multiprocessing (parallel runs)             │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 Recommended Approach: Custom Engine with Mesa Inspiration

Rather than using Mesa directly (which has overhead), I recommend a **custom NumPy-based simulation engine** for performance, with Shiny for the UI. This approach:

1. **Maximizes performance** - NumPy vectorization can handle 10,000+ agents efficiently
2. **Maintains scientific accuracy** - Direct translation of equations
3. **Enables modern UI** - Shiny provides reactive, web-based interface
4. **Supports batch processing** - Headless execution for scenario analysis

---

## 4. Module Translation Plan

### 4.1 Project Structure

```
depython/
├── app.py                      # Shiny application entry point
├── pyproject.toml              # Project configuration
├── requirements.txt            # Dependencies
│
├── depython/                   # Core simulation package
│   ├── __init__.py
│   │
│   ├── core/                   # Simulation engine
│   │   ├── __init__.py
│   │   ├── simulation.py       # Main simulation controller
│   │   ├── scheduler.py        # Time step management
│   │   └── random_source.py    # Random number generation
│   │
│   ├── agents/                 # Agent definitions
│   │   ├── __init__.py
│   │   ├── base.py             # Base agent class
│   │   ├── porpoise.py         # Porpoise agent (~500 lines)
│   │   ├── turbine.py          # Turbine agent
│   │   └── ship.py             # Ship agent
│   │
│   ├── behavior/               # Behavioral modules
│   │   ├── __init__.py
│   │   ├── movement.py         # CRW movement
│   │   ├── dispersal.py        # PSM dispersal types
│   │   ├── deterrence.py       # Noise response
│   │   ├── energetics.py       # Energy dynamics
│   │   ├── reproduction.py     # Mating/birth/nursing
│   │   └── memory.py           # Reference/working memory
│   │
│   ├── landscape/              # Environmental data
│   │   ├── __init__.py
│   │   ├── cell_data.py        # Grid data management
│   │   ├── loader.py           # Data file loading
│   │   └── spatial.py          # Spatial utilities
│   │
│   ├── parameters/             # Configuration
│   │   ├── __init__.py
│   │   ├── simulation_params.py
│   │   └── constants.py
│   │
│   └── outputs/                # Data collection
│       ├── __init__.py
│       ├── statistics.py       # Population statistics
│       └── exporters.py        # Data export utilities
│
├── ui/                         # Shiny UI components
│   ├── __init__.py
│   ├── layouts/                # Page layouts
│   │   ├── main_layout.py
│   │   ├── parameters_panel.py
│   │   └── results_panel.py
│   │
│   ├── components/             # Reusable UI components
│   │   ├── map_display.py
│   │   ├── charts.py
│   │   └── controls.py
│   │
│   └── server/                 # Server-side logic
│       ├── simulation_server.py
│       └── visualization_server.py
│
├── data/                       # Landscape data (same format as Java)
│   ├── NorthSea/
│   ├── DanTysk/
│   ├── Kattegat/
│   └── wind-farms/
│
└── tests/                      # Test suite
    ├── test_porpoise.py
    ├── test_movement.py
    └── test_validation.py
```

### 4.2 Translation Mapping

#### Phase 1: Core Infrastructure (Weeks 1-3)

| Java Component | Python Module | Priority | Complexity |
|----------------|---------------|----------|------------|
| `SimulationParameters.java` | `parameters/simulation_params.py` | HIGH | Medium |
| `SimulationConstants.java` | `parameters/constants.py` | HIGH | Low |
| `Globals.java` | `core/simulation.py` | HIGH | Medium |
| `CellData.java` | `landscape/cell_data.py` | HIGH | Medium |
| `LandscapeLoader.java` | `landscape/loader.py` | HIGH | Medium |
| `Agent.java` | `agents/base.py` | HIGH | Low |

#### Phase 2: Agent Implementation (Weeks 4-8)

| Java Component | Python Module | Priority | Complexity |
|----------------|---------------|----------|------------|
| `Porpoise.java` (movement) | `behavior/movement.py` | HIGH | High |
| `Porpoise.java` (energy) | `behavior/energetics.py` | HIGH | Medium |
| `RefMem.java` | `behavior/memory.py` | HIGH | Medium |
| `Dispersal*.java` | `behavior/dispersal.py` | HIGH | High |
| `Porpoise.java` (reproduction) | `behavior/reproduction.py` | MEDIUM | Medium |
| `Porpoise.java` (deterrence) | `behavior/deterrence.py` | MEDIUM | Medium |
| `Turbine.java` | `agents/turbine.py` | MEDIUM | Low |
| `Ship.java` + related | `agents/ship.py` | MEDIUM | Medium |

#### Phase 3: Shiny UI (Weeks 9-12)

| Feature | Python Module | Priority | Complexity |
|---------|---------------|----------|------------|
| Parameter input panel | `ui/layouts/parameters_panel.py` | HIGH | Medium |
| Map visualization | `ui/components/map_display.py` | HIGH | High |
| Population charts | `ui/components/charts.py` | HIGH | Medium |
| Simulation controls | `ui/components/controls.py` | HIGH | Low |
| Results export | `outputs/exporters.py` | MEDIUM | Low |
| Real-time animation | `ui/server/visualization_server.py` | LOW | High |

---

## 5. Detailed Component Specifications

### 5.1 Porpoise Agent Translation

```python
# depython/agents/porpoise.py - Skeleton

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple
from enum import Enum, auto

class PregnancyStatus(Enum):
    UNABLE_YOUNG = 0
    PREGNANT = 1  
    READY_TO_MATE = 2

@dataclass
class Porpoise:
    """Main porpoise agent - translates from Porpoise.java"""
    
    # Identity
    id: int
    
    # Position and movement
    x: float
    y: float
    heading: float  # degrees
    prev_log_mov: float = 0.0
    pres_log_mov: float = 0.0
    prev_angle: float = 0.0
    pres_angle: float = 0.0
    
    # Energy
    energy_level: float = 10.0
    energy_consumed_daily: float = 0.0
    food_eaten_daily: float = 0.0
    
    # Age and reproduction
    age: float = 0.0  # years
    age_of_maturity: float = 3.44
    pregnancy_status: PregnancyStatus = PregnancyStatus.UNABLE_YOUNG
    mating_day: int = 225
    days_since_mating: int = -99
    days_since_giving_birth: int = -99
    with_lact_calf: bool = False
    calves_born: int = 0
    calves_weaned: int = 0
    
    # Deterrence
    deter_vt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    deter_strength: float = 0.0
    deter_time_left: int = 0
    
    # Memory
    stored_util_list: list = field(default_factory=list)
    vt: np.ndarray = field(default_factory=lambda: np.zeros(2))
    ve_total: float = 0.0
    
    # Dispersal
    is_dispersing: bool = False
    disp_num_ticks: int = 0
    
    # State
    alive: bool = True
    
    def step(self) -> None:
        """Main simulation step - called every 30 minutes"""
        if not self.alive:
            return
            
        if self.is_dispersing:
            self._dispersal_step()
        else:
            self._standard_move()
            
        self._update_energy()
        
    def _standard_move(self) -> None:
        """Correlated random walk with memory attraction"""
        # Translate from Porpoise.stdMove()
        pass
    
    def _dispersal_step(self) -> None:
        """PSM-based dispersal movement"""
        # Translate from Dispersal classes
        pass
    
    def _update_energy(self) -> None:
        """Energy consumption and food intake"""
        # Translate from Porpoise.updEnergeticStatus()
        pass
    
    def daily_step(self) -> None:
        """Daily updates: mortality, dispersal triggers, pregnancy"""
        # Translate from Porpoise.performDailyStep()
        pass
```

### 5.2 Movement System Translation

The movement system is the most complex component. Key equations to translate:

```python
# depython/behavior/movement.py

import numpy as np
from scipy.stats import norm

def calculate_turning_angle(
    prev_angle: float,
    depth: float,
    salinity: float,
    params: dict
) -> float:
    """
    Calculate turning angle for CRW movement.
    
    Translates from Porpoise.stdMove() lines 320-380
    
    Formula:
    pres_angle = (b0 * prev_angle + random) * (b1 * depth + b2 * salinity + b3)
    """
    b0 = params['corr_angle_base']      # 0.26
    b1 = params['corr_angle_bathy']     # 0.26  
    b2 = params['corr_angle_salinity']  # 0.26
    b3 = params['corr_angle_base_sd']   # 38.0
    
    random_component = np.random.normal(0, 38)
    
    angle_base = b0 * prev_angle
    angle_bathy = b1 * depth
    angle_salinity = b2 * salinity
    
    angle = (angle_base + random_component) * (angle_bathy + angle_salinity + b3)
    
    # Constrain to [-180, 180]
    while abs(angle) > 180:
        angle = np.random.normal(0, 20) + 90 * np.sign(angle)
    
    return angle

def calculate_step_length(
    prev_log_mov: float,
    depth: float,
    salinity: float,
    params: dict
) -> float:
    """
    Calculate log10 step length.
    
    Translates from Porpoise.stdMove() lines 400-450
    
    Formula:
    pres_log_mov = a0 * prev_log_mov + a1 * depth + a2 * salinity + random
    """
    a0 = params['corr_logmov_length']    # 0.94
    a1 = params['corr_logmov_bathy']     # 0.94
    a2 = params['corr_logmov_salinity']  # 0.94
    
    # R1 parameter: N(1.25, 0.15)
    random_component = np.random.normal(1.25, 0.15)
    
    log_mov = a0 * prev_log_mov + random_component
    
    return log_mov
```

### 5.3 Landscape Data Management

```python
# depython/landscape/cell_data.py

import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import rasterio

@dataclass
class LandscapeMetadata:
    """Metadata from ASC file headers"""
    ncols: int
    nrows: int
    xllcorner: float
    yllcorner: float
    cellsize: float = 400.0
    nodata_value: float = -9999.0

class CellData:
    """
    Manages all spatial data layers.
    Translates from CellData.java
    """
    
    def __init__(self, landscape_name: str):
        self.landscape_name = landscape_name
        self._load_all_layers()
    
    def _load_all_layers(self):
        """Load all required data files"""
        base_path = Path(f"data/{self.landscape_name}")
        
        # Core layers
        self.depth = self._load_asc(base_path / "bathy.asc")
        self.dist_to_coast = self._load_asc(base_path / "disttocoast.asc")
        self.sediment = self._load_asc(base_path / "sediment.asc")
        self.food_prob = self._load_asc(base_path / "patches.asc")
        self.blocks = self._load_asc(base_path / "blocks.asc").astype(int)
        
        # Monthly layers (12 months)
        self.entropy = self._load_monthly(base_path, "prey")
        self.salinity = self._load_monthly(base_path, "salinity")
        
        # Initialize food values
        self.food_value = np.zeros_like(self.food_prob)
        
    def _load_asc(self, filepath: Path) -> np.ndarray:
        """Load ASCII grid file"""
        with rasterio.open(filepath) as src:
            data = src.read(1)
            self.metadata = LandscapeMetadata(
                ncols=src.width,
                nrows=src.height,
                xllcorner=src.bounds.left,
                yllcorner=src.bounds.bottom
            )
        return data
    
    def _load_monthly(self, base_path: Path, prefix: str) -> np.ndarray:
        """Load 12 monthly data files"""
        monthly_data = []
        for month in range(1, 13):
            filepath = base_path / f"{prefix}{month:02d}.asc"
            monthly_data.append(self._load_asc(filepath))
        return np.stack(monthly_data)
    
    def get_depth(self, x: float, y: float) -> float:
        """Get water depth at position"""
        i, j = self._pos_to_grid(x, y)
        return self.depth[i, j]
    
    def get_salinity(self, x: float, y: float, month: int) -> float:
        """Get salinity at position for given month"""
        i, j = self._pos_to_grid(x, y)
        return self.salinity[month - 1, i, j]
    
    def get_food_level(self, x: float, y: float) -> float:
        """Get current food level at position"""
        i, j = self._pos_to_grid(x, y)
        return self.food_value[i, j]
    
    def _pos_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Convert continuous position to grid indices"""
        i = int(y)
        j = int(x)
        return (i, j)
```

### 5.4 Shiny UI Design

```python
# app.py - Main Shiny application

from shiny import App, Inputs, Outputs, Session, reactive, render, ui
from shinywidgets import output_widget, render_widget
import plotly.express as px
import plotly.graph_objects as go

from depython.core.simulation import Simulation
from depython.parameters.simulation_params import SimulationParameters

# Define UI
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("DEPYTHON"),
        ui.p("Porpoise Population Simulator"),
        ui.hr(),
        
        # Simulation Settings
        ui.accordion(
            ui.accordion_panel(
                "Simulation Settings",
                ui.input_numeric("porpoise_count", "Initial Population", 
                                value=10000, min=100, max=50000),
                ui.input_numeric("sim_years", "Simulation Years", 
                                value=50, min=1, max=100),
                ui.input_select("landscape", "Landscape",
                               choices=["NorthSea", "DanTysk", "Gemini", 
                                       "Kattegat", "Homogeneous"]),
            ),
            ui.accordion_panel(
                "Disturbance Sources",
                ui.input_select("turbines", "Wind Farm Scenario",
                               choices=["off", "NorthSea_scenario1", 
                                       "NorthSea_scenario2", "NorthSea_scenario3"]),
                ui.input_checkbox("ships_enabled", "Enable Ship Noise", False),
            ),
            ui.accordion_panel(
                "Behavioral Parameters",
                ui.input_slider("deterrence_threshold", "Deterrence Threshold (dB)",
                               min=100, max=200, value=152.9),
                ui.input_slider("energy_use", "Energy Use per Step",
                               min=1.0, max=10.0, value=4.5, step=0.1),
            ),
            open="Simulation Settings"
        ),
        
        ui.hr(),
        ui.input_action_button("run_sim", "Run Simulation", class_="btn-primary"),
        ui.input_action_button("stop_sim", "Stop", class_="btn-danger"),
        
        width=300
    ),
    
    # Main panel
    ui.navset_card_tab(
        ui.nav_panel(
            "Map View",
            output_widget("map_display"),
            ui.output_text("current_tick")
        ),
        ui.nav_panel(
            "Population",
            ui.row(
                ui.column(6, output_widget("population_chart")),
                ui.column(6, output_widget("births_deaths_chart"))
            )
        ),
        ui.nav_panel(
            "Energy",
            output_widget("energy_distribution")
        ),
        ui.nav_panel(
            "Dispersal",
            output_widget("dispersal_map")
        ),
        ui.nav_panel(
            "Data Export",
            ui.download_button("download_results", "Download Results (CSV)")
        )
    ),
    
    title="DEPYTHON - Porpoise Population Model"
)

def server(input: Inputs, output: Outputs, session: Session):
    # Reactive values
    simulation = reactive.Value(None)
    is_running = reactive.Value(False)
    
    @reactive.Effect
    @reactive.event(input.run_sim)
    async def run_simulation():
        params = SimulationParameters(
            porpoise_count=input.porpoise_count(),
            sim_years=input.sim_years(),
            landscape=input.landscape(),
            turbines=input.turbines(),
            ships_enabled=input.ships_enabled(),
            deterrence_threshold=input.deterrence_threshold(),
            energy_use=input.energy_use()
        )
        
        sim = Simulation(params)
        simulation.set(sim)
        is_running.set(True)
        
        # Run simulation with periodic UI updates
        while is_running.get() and sim.tick < sim.max_ticks:
            sim.step()
            if sim.tick % 48 == 0:  # Update UI daily
                await reactive.flush()
    
    @output
    @render_widget
    def map_display():
        sim = simulation.get()
        if sim is None:
            return go.Figure()
        
        # Create scatter plot of porpoise positions
        positions = sim.get_porpoise_positions()
        fig = px.scatter(
            x=positions[:, 0],
            y=positions[:, 1],
            color=positions[:, 2],  # energy level
            title=f"Porpoise Distribution (Day {sim.tick // 48})"
        )
        return fig
    
    @output
    @render_widget
    def population_chart():
        sim = simulation.get()
        if sim is None:
            return go.Figure()
        
        history = sim.get_population_history()
        fig = px.line(
            x=history['day'],
            y=history['population'],
            title="Population Over Time"
        )
        return fig

app = App(app_ui, server)
```

---

## 6. Validation Strategy

### 6.1 Unit Tests

Each translated component must pass unit tests comparing outputs with the Java version:

```python
# tests/test_movement.py

import pytest
import numpy as np
from depython.behavior.movement import calculate_turning_angle

class TestMovement:
    """Validate movement calculations against Java reference"""
    
    def test_turning_angle_basic(self):
        """Test basic turning angle calculation"""
        params = {
            'corr_angle_base': 0.26,
            'corr_angle_bathy': 0.26,
            'corr_angle_salinity': 0.26,
            'corr_angle_base_sd': 38.0
        }
        
        # Use fixed seed for reproducibility
        np.random.seed(42)
        
        angle = calculate_turning_angle(
            prev_angle=10.0,
            depth=20.0,
            salinity=30.0,
            params=params
        )
        
        # Compare with Java reference output
        assert -180 <= angle <= 180
        
    def test_movement_bounds(self):
        """Ensure movement stays within valid ranges"""
        # Test that angles are always in [-180, 180]
        for _ in range(1000):
            angle = calculate_turning_angle(...)
            assert -180 <= angle <= 180
```

### 6.2 Integration Tests

Compare full simulation trajectories:

```python
# tests/test_validation.py

def test_population_trajectory():
    """Compare population dynamics with Java baseline"""
    # Load Java reference data
    java_results = load_java_baseline("baseline_50years.csv")
    
    # Run Python simulation with same parameters and seed
    sim = Simulation(params, seed=12345)
    sim.run()
    python_results = sim.get_results()
    
    # Compare population at yearly intervals
    for year in range(50):
        java_pop = java_results[java_results['year'] == year]['population']
        python_pop = python_results[python_results['year'] == year]['population']
        
        # Allow 5% tolerance for stochastic variation
        assert abs(java_pop - python_pop) / java_pop < 0.05
```

---

## 7. Performance Optimization

### 7.1 Vectorization Strategy

The main performance bottleneck is the per-agent step() calculations. We'll use NumPy vectorization:

```python
# Vectorized movement for all porpoises at once

def vectorized_step(positions: np.ndarray, 
                    headings: np.ndarray,
                    prev_log_movs: np.ndarray,
                    cell_data: CellData,
                    params: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate movement for all porpoises simultaneously.
    
    Args:
        positions: (N, 2) array of [x, y] positions
        headings: (N,) array of headings in degrees
        prev_log_movs: (N,) array of previous log movement
        cell_data: Landscape data
        params: Model parameters
    
    Returns:
        new_positions: (N, 2) updated positions
        new_headings: (N,) updated headings
    """
    n = len(positions)
    
    # Get environmental data for all positions at once
    depths = cell_data.get_depths_vectorized(positions)
    salinities = cell_data.get_salinities_vectorized(positions)
    
    # Calculate turning angles (vectorized)
    random_angles = np.random.normal(0, 38, n)
    angle_base = params['b0'] * prev_angles
    pres_angles = (angle_base + random_angles) * (
        params['b1'] * depths + 
        params['b2'] * salinities + 
        params['b3']
    )
    
    # Update headings
    new_headings = headings + pres_angles
    new_headings = np.mod(new_headings + 180, 360) - 180
    
    # Calculate step lengths
    random_lengths = np.random.normal(1.25, 0.15, n)
    log_movs = params['a0'] * prev_log_movs + random_lengths
    step_lengths = 10 ** log_movs / 4.0  # Convert to grid units
    
    # Calculate new positions
    heading_rad = np.radians(new_headings)
    dx = step_lengths * np.sin(heading_rad)
    dy = step_lengths * np.cos(heading_rad)
    new_positions = positions + np.column_stack([dx, dy])
    
    return new_positions, new_headings
```

### 7.2 Numba JIT Compilation

For hot loops that can't be vectorized:

```python
from numba import jit, prange

@jit(nopython=True, parallel=True)
def calculate_deterrence_all(
    porpoise_positions: np.ndarray,
    turbine_positions: np.ndarray,
    turbine_impacts: np.ndarray,
    params: tuple
) -> np.ndarray:
    """
    Calculate deterrence vectors for all porpoises.
    Uses Numba for parallel execution.
    """
    n_porpoises = len(porpoise_positions)
    n_turbines = len(turbine_positions)
    deterrence = np.zeros((n_porpoises, 2))
    
    for i in prange(n_porpoises):
        for j in range(n_turbines):
            dx = porpoise_positions[i, 0] - turbine_positions[j, 0]
            dy = porpoise_positions[i, 1] - turbine_positions[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            
            if dist < params[0]:  # max_deter_distance
                impact = turbine_impacts[j]
                strength = impact * np.exp(-dist / params[1])
                deterrence[i, 0] += strength * dx / dist
                deterrence[i, 1] += strength * dy / dist
    
    return deterrence
```

---

## 8. Implementation Timeline

### Phase 1: Foundation (Weeks 1-3)
- [ ] Set up project structure
- [ ] Implement `SimulationParameters` and `Constants`
- [ ] Implement `CellData` and landscape loading
- [ ] Basic `Porpoise` class structure
- [ ] Unit test framework

### Phase 2: Core Simulation (Weeks 4-8)
- [ ] Implement CRW movement system
- [ ] Implement reference memory
- [ ] Implement PSM dispersal (Type 2 priority)
- [ ] Implement energetics system
- [ ] Implement reproduction system
- [ ] Implement mortality
- [ ] Integration tests

### Phase 3: Disturbance Agents (Weeks 9-10)
- [ ] Implement `Turbine` agent
- [ ] Implement deterrence behavior
- [ ] Implement `Ship` agent (if needed)
- [ ] Implement ship noise calculations

### Phase 4: Shiny UI (Weeks 11-14)
- [ ] Parameter input interface
- [ ] Map visualization with Plotly
- [ ] Population time series charts
- [ ] Energy distribution histograms
- [ ] Real-time simulation updates
- [ ] Data export functionality

### Phase 5: Validation & Polish (Weeks 15-16)
- [ ] Full validation against Java baseline
- [ ] Performance optimization
- [ ] Documentation
- [ ] Deployment preparation

---

## 9. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Numerical precision differences | High | Medium | Use same random seed, validate step-by-step |
| Performance issues with 10k+ agents | Medium | High | Vectorization, Numba, batch updates |
| Complex dispersal behavior translation | Medium | High | Focus on PSM-Type2 first, extensive testing |
| Shiny reactive performance | Medium | Medium | Throttle updates, background processing |
| Missing edge cases | Medium | Medium | Comprehensive unit testing |

---

## 10. Dependencies

```
# requirements.txt

# Core simulation
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
numba>=0.57.0

# Spatial data
rasterio>=1.3.0
geopandas>=0.13.0
shapely>=2.0.0

# Visualization
plotly>=5.15.0
matplotlib>=3.7.0
folium>=0.14.0

# Shiny framework
shiny>=0.10.0
shinyswatch>=0.4.0
shinywidgets>=0.2.0
htmltools>=0.5.0

# Utilities
pydantic>=2.0.0
python-dotenv>=1.0.0
tqdm>=4.65.0

# Testing
pytest>=7.4.0
pytest-asyncio>=0.21.0
```

---

## 11. Conclusion

This translation plan provides a comprehensive roadmap for converting DEPONS from Java/Repast Simphony to Python/Shiny. The key success factors are:

1. **Faithful translation** of the scientific algorithms
2. **Thorough validation** against the Java reference implementation
3. **Modern Python idioms** for maintainability
4. **Performance optimization** through vectorization and JIT compilation
5. **User-friendly Shiny interface** for accessibility

The estimated timeline is 16 weeks for a fully validated, production-ready implementation.

---

## Appendix A: Key Algorithm Reference

### A.1 Energy Dynamics

```
E(t+1) = E(t) + food_eaten - energy_used

energy_used = Euse * movement_multiplier * lactation_multiplier * temperature_multiplier

survival_probability = 1 / (1 + exp(-beta * E))
```

### A.2 CRW Movement

```
turning_angle = (b0 * prev_angle + N(0,38)) * (b1*depth + b2*salinity + b3)

log_step = a0 * prev_log_step + N(1.25, 0.15)

step_length = 10^log_step / 4  (400m cells)
```

### A.3 Deterrence Response

```
received_level = source_level - transmission_loss(distance)

if received_level > threshold:
    deterrence_vector = c * (porpoise_pos - source_pos) / distance
```

---

*Document Version: 1.0*  
*Created: January 2026*  
*Author: AI4WIND Project Team*
