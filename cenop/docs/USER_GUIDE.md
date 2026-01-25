# CENOP User Guide

## Cetacean Noise Operations Planner

A web-based simulation tool for assessing the impact of offshore wind farm construction on harbor porpoise populations.

**Version 1.0** | Python Shiny Implementation of DEPONS 3.0

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Interface Overview](#interface-overview)
4. [Running a Simulation](#running-a-simulation)
5. [Understanding the Results](#understanding-the-results)
6. [Parameter Reference](#parameter-reference)
7. [Data Export](#data-export)
8. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is CENOP?

CENOP (Cetacean Noise Operations Planner) is a Python translation of the DEPONS (Disturbance Effects on the Harbour Porpoise Population in the North Sea) model. It simulates how harbor porpoise populations respond to noise from offshore wind farm construction.

### Key Features

- **Agent-Based Simulation**: Each porpoise is modeled individually with realistic behavior
- **Real-Time Visualization**: Watch population dynamics unfold on an interactive map
- **Energy Budget Modeling**: Tracks individual energy reserves and their effect on survival
- **Deterrence Response**: Models porpoise avoidance of noise sources (threshold: 158 dB)
- **Persistent Spatial Memory (PSM)**: Porpoises remember good foraging locations
- **DEPONS Compatibility**: Outputs match the original Java model format
- **Vectorized Performance**: NumPy-based simulation supports 1000+ porpoises in real-time

### Scientific Background

The model is based on:
- Nabe-Nielsen et al. (2018) - DEPONS model framework
- Hin et al. (2019) - Dynamic Energy Budget model
- Tougaard et al. - Deterrence response curves

---

## Getting Started

### System Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Network connection to the server (laguna.ku.lt)
- No local installation required

### Accessing CENOP

1. Open your web browser
2. Navigate to: `https://laguna.ku.lt/cenop/`
3. The application will load automatically

### First Run

When you first access CENOP:
1. The default Homogeneous landscape is selected
2. Initial population of 1000 porpoises is configured
3. Default parameters are set for a typical 5-year simulation

---

## Interface Overview

### Main Layout

The interface is divided into three main areas:

```
┌─────────────────────────────────────────────────────────┐
│  Logo    Dashboard | Settings | Population | ...  Help │
├───────────────┬─────────────────────────────────────────┤
│               │                                         │
│   Sidebar     │           Main View                     │
│   Controls    │      (Map/Charts/Tables)                │
│               │                                         │
│               │                                         │
└───────────────┴─────────────────────────────────────────┘
```

### Sidebar Controls

#### Setup Section
- **Initial Population**: Starting number of porpoises (1-50,000)
- **Simulation Years**: Duration of simulation (1-100 years)
- **Landscape**: Geographic area (Homogeneous, NorthSea, DanTysk, etc.)
- **Load Landscape**: Button to load bathymetry and display on map
- **Wind Turbines**: Turbine scenario (filtered by landscape compatibility)
- **Load Turbines**: Button to display turbines and noise overlay

#### Run Controls
- **▶ Run Simulation**: Start the simulation
- **⏹ Stop**: Pause the simulation
- **🔄 Reset**: Reset to initial state

#### Speed Control
- **Simulation Speed** (1-100%):
  - 1% = Slowest (0.3s per day) - watch individual movements
  - 50% = Medium (~0.075s per day)
  - 100% = Maximum speed (no delay) - for long runs

### Main View Tabs

1. **Dashboard**: Interactive map and population charts
2. **Model Settings**: Advanced parameter configuration
3. **Population**: Age/energy histograms and vital statistics
4. **Disturbance**: Dispersal and deterrence monitoring
5. **Export**: Download results and about information

---

## Running a Simulation

### Step 1: Configure Population

1. In the sidebar, set **Initial Population** (recommended: 500-2000)
2. Set **Simulation Years** (1-5 years for quick tests, 10+ for population dynamics)

### Step 2: Select Landscape

1. Choose a **Landscape** from the dropdown:
   - **Homogeneous**: Uniform test grid (400×400 cells)
   - **NorthSea**: North Sea with real bathymetry (400×400 @ 400m)
   - **UserDefined**: DEPONS default landscape data files
2. Click **🗺️ Load Landscape** to display:
   - Depth overlay (bathymetry)
   - Foraging overlay (food probability patches)

> **Note**: Other DEPONS landscapes (Kattegat, InnerDanishWaters, DanTysk, Gemini)
> require separate data files not included in this distribution.

### Step 3: Add Wind Turbines (Optional)

1. Select a **Wind Turbines** scenario (options filtered by landscape):
   - NorthSea → Scenarios 1-3 (80-240 turbines)
   - UserDefined → User-defined scenario
2. Click **🌬️ Load Turbines** to display:
   - Orange dots for turbine locations
   - Red shading for noise levels above 158 dB threshold

### Step 4: Configure Advanced Settings (Optional)

1. Click the **Model Settings** tab
2. Adjust parameters (all have tooltip explanations):
   - **Basic**: Random seed, ship traffic, bycatch probability
   - **Movement**: CRW parameters (k, a0-a2, b0-b3)
   - **Dispersal**: Dispersal type, PSM parameters
   - **Energy**: Memory decay rates (rS, rR, rU)

### Step 5: Run Simulation

1. Click **▶ Run Simulation**
2. Watch the progress bar and status message
3. Adjust **Simulation Speed** as needed:
   - Slow down to observe porpoise movements
   - Speed up for long-duration runs
4. Click **⏹ Stop** to pause at any time
5. Click **🔄 Reset** to start over

### Step 6: Analyze Results

1. View real-time updates on the **Dashboard**:
   - Map shows porpoise positions (blue dots)
   - Charts show population, births/deaths, energy
2. Switch to **Population** tab for:
   - Age distribution histogram
   - Energy distribution histogram
   - Vital statistics table
3. Check **Disturbance** tab for:
   - Dispersal events
   - Deterrence counts

---

## Understanding the Results

### Dashboard Map

The interactive map shows:

- **Blue dots**: Individual porpoises (up to 1000 displayed)
- **Orange dots**: Wind turbine locations
- **Red shading**: Noise levels above deterrence threshold
- **Blue gradient**: Bathymetry (water depth)
- **Green shading**: Foraging areas (food probability patches)

**Layer Controls** (top-right panel):

- Toggle Depth, Turbines, Noise, and Foraging layers on/off
- Drag panels to reposition

### Dashboard Charts

| Chart | Description |
| ----- | ----------- |
| **Population Size** | Total porpoises and lactating+calf pairs over time |
| **Life and Death** | Daily births (blue) and deaths (red) |
| **Energy Balance** | Average food eaten vs energy expended |

### Population Tab

| Visualization | Description |
|---------------|-------------|
| **Age Distribution** | Histogram of ages (0-30 years) |
| **Energy Distribution** | Histogram of energy levels (0-20 units) |
| **Landscape Energy** | Total food availability over time |
| **Average Movement** | Mean daily movement distance |
| **Vital Statistics** | Summary table of population metrics |

### Key Metrics

| Metric | Healthy Range | Concern |
|--------|---------------|---------|
| Population | Stable ±20%/year | Declining >30%/year |
| Mean Energy | >10 units | <5 units |
| Daily Births | 0.5-2 per 1000 | <0.1 per 1000 |
| Daily Deaths | 0.5-2 per 1000 | >5 per 1000 |

---

## Parameter Reference

### Sidebar Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Initial Population | 1000 | 1-50,000 | Starting porpoise count |
| Simulation Years | 5 | 1-100 | Duration in simulated years |
| Landscape | Homogeneous | - | Geographic area |
| Simulation Speed | 50% | 1-100% | Run speed control |

### Model Settings: Basic Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| Random Seed | 0 | Reproducibility seed (0=random) |
| Tracked Porpoises | 1 | Individuals tracked in detail |
| Ship Traffic | Off | Enable vessel disturbance |
| Bycatch Probability | 0.0 | Annual fishing mortality (0-1) |

### Model Settings: Movement Tab (CRW)

| Parameter | Default | Description |
|-----------|---------|-------------|
| k | 0.001 | Inertia - directional persistence |
| a0 | 0.35 | Step length autocorrelation |
| a1 | 0.0005 | Depth effect on step length |
| a2 | -0.02 | Salinity effect on step length |
| b0 | -0.024 | Turning angle autocorrelation |
| b1 | -0.008 | Depth effect on turning |
| b2 | 0.93 | Salinity effect on turning |
| b3 | -14.0 | Turning angle intercept |

### Model Settings: Dispersal Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| Dispersal Type | PSM-Type2 | Memory-based with heading dampening |
| tDisp | 3 days | Days of declining energy to trigger |
| PSM_log | 0.6 | Memory strengthening rate |
| PSM_dist | N(300;100) | Preferred dispersal distance (km) |
| PSM_tol | 5 km | Target tolerance distance |
| PSM_angle | 20° | Maximum turn per step |

### Model Settings: Energy Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| rS | 0.04 | Satiation memory decay rate |
| rR | 0.04 | Reference memory decay rate |
| rU | 0.1 | Food replenishment rate |

---

## Data Export

### CSV Export (UI)

1. Run a simulation
2. Go to the **Export** tab
3. Click **📥 Download Results CSV**

**Exported columns:**
- `tick`: Simulation tick (30-min intervals)
- `day`, `year`: Time markers
- `population`: Total living porpoises
- `births`, `deaths`: Cumulative counts
- `avg_energy`: Mean energy level
- Additional simulation metrics

### DEPONS-Compatible Outputs (Python API)

For advanced users, the full DEPONS output format is available via Python:

```python
from cenop.core.output_writer import OutputWriter, OutputConfig

config = OutputConfig(
    output_dir="output",
    run_id="simulation_001",
    population=True,
    porpoise_statistics=True,
    mortality=True,
    dispersal=True,
    energy=True
)

writer = OutputWriter(config)
# ... run simulation ...
writer.record_tick(simulation)
writer.finalize()
```

**Output files:**
- `Population.txt`: Daily population counts
- `PorpoiseStatistics.txt`: Individual porpoise data
- `Mortality.txt`: Death events with causes
- `Dispersal.txt`: Dispersal events
- `Energy.txt`: Energy statistics

---

## Troubleshooting

### Common Issues

#### Simulation Runs Slowly

**Cause**: Large population or browser limitations

**Solutions**:
- Reduce initial population to <2000
- Increase simulation speed to 100%
- Use a modern browser (Chrome recommended)
- Close other browser tabs

#### Porpoises Disappear Quickly

**Cause**: Population collapse from starvation or disturbance

**Solutions**:
- Use Homogeneous landscape (uniform food)
- Reduce initial population
- Disable turbines initially
- Check energy parameters

#### Map Doesn't Update

**Cause**: Browser or connection issues

**Solutions**:
- Refresh the page (F5)
- Wait for "Load Landscape" to complete
- Check browser console for errors

#### Speed Slider Doesn't Respond

**Cause**: Slider updates during simulation

**Solutions**:
- Move slider slowly
- Wait a moment after changing
- Speed changes take effect immediately

### Getting Help

1. Click **❓ Help** in the top navigation bar
2. Check this documentation
3. Contact the AI4WIND project team

---

## Appendix: Scientific Notes

### Model Validation

CENOP has been validated against:
- Original DEPONS 3.0 Java model outputs
- Empirical porpoise tracking data
- Published population estimates

### Time Steps

- 1 tick = 30 minutes
- 48 ticks = 1 day
- 17,280 ticks = 1 year (360 days)

### Spatial Resolution

- Cell size: 400m × 400m
- Grid varies by landscape (e.g., 400×400 for Homogeneous)

### Energy Model

- **Seasonal scaling**: Cold (Nov-Mar) = 1.0×, Warm (May-Sep) = 1.3×
- **Lactation cost**: 1.4× normal metabolism
- **Starvation threshold**: Energy < 0.1 → increased mortality

### Citation

If using CENOP in publications, please cite:
```
CENOP: Cetacean Noise Operations Planner
A Python Shiny translation of DEPONS 3.0
AI4WIND Project, 2024-2026
```

### References

1. Nabe-Nielsen J., et al. (2018). Predicting the impacts of anthropogenic disturbances on marine populations. *Conservation Letters*.
2. Hin V., et al. (2019). A bioenergetics model for harbour porpoise. *Ecological Modelling*.
3. Tougaard J., et al. (2015). Noise from operation of offshore wind farms. *Marine Ecology Progress Series*.
4. DEPONS Project: [www.depons.dk](http://www.depons.dk)
