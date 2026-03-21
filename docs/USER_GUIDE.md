# CENOP-JASMINE User Guide

## Cetacean Noise Operations Planner with JASMINE Extensions

A web-based simulation tool for assessing the impact of offshore wind farm construction on harbor porpoise populations, with advanced research-grade behavioral modeling.

**Version 2.1** | Python Shiny Implementation aligned with DEPONS 3.2 + JASMINE Extensions

---

## Table of Contents

1. [Introduction](#introduction)
2. [Simulation Modes](#simulation-modes)
3. [Getting Started](#getting-started)
4. [Interface Overview](#interface-overview)
5. [Running a Simulation](#running-a-simulation)
6. [Understanding the Results](#understanding-the-results)
7. [Parameter Reference](#parameter-reference)
8. [Data Export](#data-export)
9. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is CENOP-JASMINE?

CENOP (Cetacean Noise Operations Planner) is a Python translation of the DEPONS (Disturbance Effects on the Harbour Porpoise Population in the North Sea) model. It simulates how harbor porpoise populations respond to noise from offshore wind farm construction.

The JASMINE (Just Another Simulation Model In Nature Environments) extension adds research-grade features including physics-based movement, dynamic energy budgets, and learned avoidance behaviors.

### Key Features

#### Core Features (DEPONS Mode)
- **Full DEPONS 3.2 Parity:** 153 simulation features verified against Java source (136 exact match, 13 intentional improvements, 4 extensions)
- **High Performance:** 1.05 ms/tick in DEPONS mode — only 1.3× slower than the Java original, with 12 Numba JIT kernels
- **Agent-Based Simulation**: Each porpoise is modeled individually with realistic behavior
- **Real-Time Visualization**: Watch population dynamics unfold on an interactive map
- **Pregnancy FSM**: Three-state reproductive cycle (immature → pregnant → ready-to-mate) with daily scheduling
- **Energy Budget Modeling**: Tracks individual energy reserves with starvation mortality formula
- **Logistic Food Regrowth**: MaxEnt-based carrying capacity with 48-iteration daily compounding
- **Reference Memory**: 120-entry circular buffer with vectorized attraction (vt) and expected value (veTotal)
- **CRW Rejection Sampling**: Retry-based angle and step length sampling (max 200 retries)
- **PSM-Type2 Dispersal**: SSLogis heading dampening, energy-based stop, deterrence deactivation
- **Deterrence Response**: Models porpoise avoidance of noise sources (threshold: 152 dB)
- **Ship Noise**: JOMOPANS 13-class source levels with Weston flux physics-based transmission loss
- **DEPONS 3.2 Alignment**: Full algorithmic sync across all 5 subsystems (516+ automated tests)
- **Numba JIT Performance**: 7 hot-path kernels compiled to machine code with prange parallelism
- **Vectorized Architecture**: NumPy SoA + Numba kernels support 1000+ porpoises in real-time
- **Bitmap Grid Rendering**: Full-resolution server-side PNG rendering of landscape layers

#### JASMINE Extensions (Research Mode)
- **Behavioral State Machine**: Five behavioral states (FORAGING, TRAVELING, RESTING, DISPERSING, DISTURBED) with configurable transitions
- **Dynamic Energy Budget (DEB)**: Body mass-dependent metabolism, activity costs, thermoregulation, and disturbance energy impacts
- **Disturbance Memory**: Spatial memory with learned avoidance of disturbance zones
- **Habituation**: Reduced response to repeated disturbance exposure
- **Physics-Based Movement**: Hydrodynamic drag, thrust-based propulsion, and ocean current advection

---

## Simulation Modes

CENOP-JASMINE supports two simulation modes:

### DEPONS Mode (Default)
- Regulatory-compatible empirical models fully aligned with DEPONS 3.2
- Pregnancy FSM (3-state: immature → pregnant → ready-to-mate) with daily scheduling
- CRW with rejection sampling (up to 200 retries for angle and step length)
- Reference memory (120-entry circular buffer) with vectorized attraction computation
- Logistic food regrowth with 48-iteration daily compounding and MaxEnt carrying capacity
- PSM-Type2 dispersal with SSLogis heading, energy-based stop, deterrence deactivation
- Ship deterrence with JOMOPANS source levels and Weston flux transmission loss
- Energy-based starvation mortality (m_mort_prob_const=1.0, x_survival_const=0.4)
- Suitable for environmental impact assessments

### JASMINE Mode (Research)
- Physics-based movement and bioenergetics
- Dynamic Energy Budget with body mass scaling
- Learned avoidance with spatial memory
- Habituation to repeated disturbance
- Enhanced behavioral state machine
- Suitable for research and hypothesis testing

**Selecting a Mode**: Use the "Simulation Mode" dropdown in the sidebar to switch between DEPONS and JASMINE modes

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

### Accessing CENOP-JASMINE

1. Open your web browser
2. Navigate to: `https://laguna.ku.lt/cenjas/`
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
- **Simulation Mode**: Select DEPONS (Regulatory) or JASMINE (Research) mode
- **Initial Population**: Starting number of porpoises (1-50,000)
- **Simulation Years**: Duration of simulation (1-100 years)
- **Landscape**: Geographic area (Homogeneous, NorthSea, Lithuania, etc.)
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

1. **Dashboard**: Interactive DeckGL map and population charts
2. **Model Settings**: Advanced parameter configuration
3. **Population**: Age/energy histograms and vital statistics
4. **Disturbance**: Dispersal and deterrence monitoring
5. **Landscape**: Spatial viewer for all environmental data layers
6. **Export**: Download results as CSV

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
   - Red shading for noise levels above 152 dB threshold

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
| Dispersal Type | PSM-Type2 | Memory-based with SSLogis heading dampening |
| tDisp | 3 days | Days of declining energy to trigger |
| mean_disp_dist | 2.0 km | Mean distance per dispersal step (DEPONS 3.2) |
| PSM_log | 0.6 | Memory strengthening rate |
| PSM_dist | N(300;100) | Preferred dispersal distance (km) |
| PSM_tol | 5 km | Target tolerance distance |
| PSM_angle | 20° | Maximum turn per step |
| q1 | 0.02 | PSM-Type3 distance-cost coefficient |

### Model Settings: Energy Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| rS | 0.03 | Satiation memory decay rate (DEPONS 3.2) |
| rR | 0.03 | Reference memory decay rate (DEPONS 3.2) |
| rU | 0.1 | Food replenishment rate |

### JASMINE Mode Parameters

These parameters are only active when JASMINE mode is selected.

#### Behavioral State Machine

| Parameter | Default | Description |
|-----------|---------|-------------|
| FSM Mode | JASMINE | State machine type (DEPONS or JASMINE) |
| Recovery Ticks | 48 | Ticks to recover from disturbance |
| Energy Threshold Low | 0.3 | Low energy threshold for state transitions |
| Energy Threshold High | 0.7 | High energy threshold for state transitions |
| Speed Threshold | 2.0 | Speed threshold for TRAVELING state (m/s) |

#### Dynamic Energy Budget (DEB)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Body Mass | 50.0 kg | Adult porpoise body mass |
| BMR Scale | 1.0 | Basal metabolic rate multiplier |
| Activity Cost | 2.0 | Activity cost multiplier |
| Thermal Model | On | Enable temperature-dependent metabolism |
| Disturbance Cost | 1.5 | Energy cost multiplier during disturbance |

#### Disturbance Memory

| Parameter | Default | Description |
|-----------|---------|-------------|
| Memory Mode | JASMINE | Memory type (None or JASMINE) |
| Memory Decay Rate | 0.001 | Per-tick memory decay |
| Avoidance Radius | 20 | Influence radius in grid cells |
| Habituation | On | Enable habituation to repeated disturbance |
| Habituation Rate | 0.05 | Rate of habituation per exposure |

#### Physics-Based Movement

| Parameter | Default | Description |
|-----------|---------|-------------|
| Drag Coefficient | 0.01 | Hydrodynamic drag coefficient |
| Max Thrust | 100.0 N | Maximum propulsive thrust |
| Current Weight | 0.5 | Ocean current influence (0-1) |

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
3. Contact the arturas.razinkovas-baziukas@ku.lt

---

## Appendix: Scientific Notes

### Model Validation

**DEPONS Mode** is algorithmically aligned with:
- DEPONS 3.2 Java implementation across all 5 subsystems:
  - Reproduction (pregnancy FSM, weaning-based calf creation)
  - Food & energy (logistic regrowth, MaxEnt carrying capacity, starvation formula)
  - Movement & memory (reference memory circular buffers, CRW rejection sampling, heading composition)
  - Dispersal (PSM-Type2 with SSLogis heading, energy-based stop)
  - Deterrence (raw displacement vectors, WestonFlux TL, JOMOPANS SPL)
- 516+ automated tests verifying parameter defaults, formula outputs, and population stability

**JASMINE Mode** is research-grade and designed for:
- Exploring advanced behavioral hypotheses
- Testing learned avoidance scenarios
- Evaluating bioenergetics models
- Not yet validated for regulatory use

### Time Steps

- 1 tick = 30 minutes
- 48 ticks = 1 day
- 17,280 ticks = 1 year (360 days, DEPONS convention)

### Spatial Resolution

- Cell size: 400m × 400m
- Grid varies by landscape (e.g., 400×400 for Homogeneous)

### Energy Model

- **Seasonal scaling**: Cold (Nov-Mar) = 1.0×, Warm (May-Sep) = 1.3×
- **Lactation cost**: 1.4× normal metabolism
- **Starvation threshold**: Energy < 0.1 → increased mortality

### Behavioral States (JASMINE Mode)

| State | Description | Movement |
|-------|-------------|----------|
| FORAGING | Searching for/consuming food | DEPONS CRW |
| TRAVELING | Directed movement between areas | Physics-based |
| RESTING | Low activity energy recovery | Physics-based |
| DISPERSING | Memory-driven dispersal to new areas | PSM-based |
| DISTURBED | Response to disturbance events | Avoidance |

### Performance Benchmarks

CENOP achieves near-Java performance through Numba JIT compilation:

| Population Size | DEPONS Mode | JASMINE Mode | Java Reference |
|----------------|-------------|--------------|----------------|
| 100 agents | 0.82 ms/tick | 0.88 ms/tick | 0.15 ms/tick |
| 500 agents | 1.05 ms/tick | 1.88 ms/tick | 0.80 ms/tick |
| 1,000 agents | 1.54 ms/tick | 4.19 ms/tick | 1.64 ms/tick |
| 2,000 agents | 2.40 ms/tick | 11.2 ms/tick | 3.34 ms/tick |

A 30-year simulation with 500 agents completes in approximately 9 minutes (DEPONS) or 16 minutes (JASMINE).

### Citation

If using CENOP-JASMINE in publications, please cite:
```
CENOP-JASMINE: Cetacean Noise Operations Planner with JASMINE Extensions
A Python Shiny implementation aligned with DEPONS 3.2 with research-grade behavioral modeling
Arturas Razinkovas-Baziukas, Klaipeda University, 2024-2026
```

### References

1. Nabe-Nielsen J., Sibly R.M., Tougaard J., Teilmann J., Sveegaard S. (2014). Effects of noise and by-catch on a Danish harbour porpoise population. *Ecological Modelling*, 272, 242–251. [doi:10.1016/j.ecolmodel.2013.09.025](https://doi.org/10.1016/j.ecolmodel.2013.09.025)
2. Nabe-Nielsen J., van Beest F.M., Grimm V., Sibly R.M., Teilmann J., Thompson P.M. (2018). Predicting the impacts of anthropogenic disturbances on marine populations. *Conservation Letters*, 11(5), e12563. [doi:10.1111/conl.12563](https://doi.org/10.1111/conl.12563)
3. Nabe-Nielsen J., Harwood J. (2016). Comparison of the iPCoD and DEPONS models for modelling population consequences of noise on harbour porpoises. *Scientific Report from DCE*, No. 186.
4. van Beest F.M., Nabe-Nielsen J., Carstensen J., Teilmann J., Sveegaard S. (2015). Disturbance Effects on the Harbour Porpoise Population in the North Sea (DEPONS): Status report on model development. *Scientific Report from DCE*, No. 140.
5. Hin V., Harwood J., de Roos A.M. (2019). Bio-energetic modeling of medium-sized cetaceans shows that physiological structure is key to determining the cumulative effects of disturbance. *Ecological Modelling*, 394, 82–93. [doi:10.1016/j.ecolmodel.2018.12.019](https://doi.org/10.1016/j.ecolmodel.2018.12.019)
6. Tougaard J., Wright A.J., Madsen P.T. (2015). Cetacean noise criteria revisited in the light of proposed exposure limits for harbour porpoises. *Marine Pollution Bulletin*, 90(1–2), 196–208. [doi:10.1016/j.marpolbul.2014.10.051](https://doi.org/10.1016/j.marpolbul.2014.10.051)
7. Kooijman S.A.L.M. (2010). *Dynamic Energy Budget theory for metabolic organisation*. 3rd ed. Cambridge University Press.
8. Grimm V., Railsback S.F. (2005). *Individual-based Modeling and Ecology*. Princeton University Press.
9. DEPONS Project: [www.depons.dk](http://www.depons.dk)
