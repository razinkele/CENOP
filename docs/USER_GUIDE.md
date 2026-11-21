# CENOP-JASMINE User Guide

## CETacean Noise-Population Model with JASMINE Extensions

A web-based simulation tool for assessing the impact of offshore wind farm construction and ship noise on harbour porpoise populations, with experimental research extensions.

**Version 2.3** | Python Shiny implementation aligned with DEPONS 3.2 + JASMINE extensions

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

CENOP (CETacean Noise-Population Model) is a Python translation of the DEPONS (Disturbance Effects on the Harbour Porpoise Population in the North Sea) model. It simulates how harbour porpoise populations respond to noise from offshore wind farm construction and from ships.

The JASMINE (JASCO's Animal Simulation Model Including Noise Exposure) extension adds experimental, uncalibrated features: learned avoidance of remembered disturbances, behaviour states, and optional JASMINE movement and energy budgets. See the known issues in `CHANGELOG.md` before using its results.

### Key Features

#### Core Features (DEPONS Mode)
- **DEPONS 3.2 algorithms**: ported feature by feature (see `docs/DEPONS-CENOP-PARITY-ANALYSIS.md`) and checked against runs of DEPONS 3.2 on the Kattegat landscape (`docs/validation/A2-parity-report.md`)
- **Agent-based simulation**: each porpoise is modelled individually
- **Live visualization**: porpoise positions on an interactive map, with charts updated during the run
- **Pregnancy FSM**: three-state reproductive cycle (immature → pregnant → ready-to-mate) with daily scheduling
- **Energy budget**: individual energy reserves with an energy-dependent starvation mortality formula
- **Logistic food regrowth**: MaxEnt-based carrying capacity with up to 48 iterations per day
- **Reference memory**: 120-entry circular buffer with attraction vector (vt) and expected food value (veTotal)
- **CRW rejection sampling**: retry-based angle and step length sampling (max 200 retries)
- **Dispersal**: PSM-Type2 (default), Undirected, InnerDanishWaters (Kattegat only) or off
- **Deterrence response**: porpoises avoid noise sources above a received level of 152 dB
- **Ship noise**: JOMOPANS 13-class source levels with Weston flux transmission loss
- **Performance**: NumPy struct-of-arrays population with Numba-compiled kernels for the hot paths
- **Bitmap grid rendering**: landscape layers drawn at full grid resolution on the map

#### JASMINE Extensions (Research Mode, experimental)
- **Learned avoidance**: porpoises remember where they were disturbed and steer away from those places
- **Behaviour states**: five states (FORAGING, TRAVELING, RESTING, DISPERSING, DISTURBED) that label each animal and set its activity cost in the JASMINE energy budget; they do not change movement and are not shown in the app
- **JASMINE energy budget** (optional, uncalibrated): metabolism, activity and thermal costs in physical units
- **JASMINE movement** (optional): a simplified walk; the physics integrator (drag, thrust, currents) exists but is not wired in

---

## Simulation Modes

CENOP-JASMINE supports two simulation modes:

### DEPONS Mode (Default)
- Regulatory-compatible empirical models aligned with DEPONS 3.2
- Pregnancy FSM (3-state: immature → pregnant → ready-to-mate) with daily scheduling
- CRW with rejection sampling (up to 200 retries for angle and step length)
- Reference memory (120-entry circular buffer) with vectorized attraction computation
- Logistic food regrowth with up to 48 iterations per day and MaxEnt carrying capacity
- PSM-Type2 dispersal with SSLogis heading, energy-based stop, deterrence deactivation
- Ship deterrence with JOMOPANS source levels and Weston flux transmission loss
- Energy-based starvation mortality (m_mort_prob_const=1.0, x_survival_const=0.4)
- Suitable for environmental impact assessments

### JASMINE Mode (Research)
- Runs DEPONS movement and the DEPONS energy budget by default
- Adds learned avoidance of remembered disturbances and behaviour states
- The JASMINE movement and JASMINE energy budget run only when chosen under **Model Settings → JASMINE → Subsystem Mode Overrides** (Movement Mode, Energy Mode; the default is "Default (DEPONS)")
- Experimental and not calibrated; suitable for research and hypothesis testing

**Selecting a Mode**: Use the "Simulation Mode" dropdown in the sidebar to switch between DEPONS and JASMINE modes.

### Scientific Background

The model is based on the DEPONS model (Nabe-Nielsen et al. 2014, 2018); see [References](#references).

---

## Getting Started

### System Requirements

- Modern web browser (Chrome, Firefox, Edge, Safari)
- Network connection to the server (laguna.ku.lt)
- No local installation required

### Accessing CENOP-JASMINE

1. Open your web browser
2. Navigate to `https://laguna.ku.lt/cenjas/` (or `https://laguna.ku.lt/cenop/`, the same app)
3. The application will load automatically

### First Run

When you first open CENOP:
1. The Lithuania landscape is selected
2. The initial population is 100 porpoises
3. The simulation runs for 5 years in DEPONS mode

---

## Interface Overview

### Main Layout

The interface is divided into three main areas:

```
┌──────────────────────────────────────────────────────────────────┐
│ Logo  Dashboard | Model Settings | Population | ...  Share ◐ Help │
├───────────────┬──────────────────────────────────────────────────┤
│               │                                                  │
│   Sidebar     │           Main View                              │
│   Controls    │      (Map/Charts/Tables)                         │
│               │                                                  │
└───────────────┴──────────────────────────────────────────────────┘
```

### Navbar Controls

- **Share**: opens a link that reopens CENOP with the current settings and map view (not the results)
- **Dark mode switch**: switches the app and the map basemaps between light and dark
- **❓ Help**: opens the in-app user manual

### Sidebar Controls

#### Setup Section
- **Initial Population**: Starting number of porpoises (1-50,000; default 100)
- **Simulation Years**: Duration of simulation (1-100 years; default 5)
- **Simulation Mode**: DEPONS (Regulatory) or JASMINE (Research)
- **Landscape**: Geographic area (Homogeneous, Lithuania, CentralBaltic, Kattegat, NorthSea)
- **Load Landscape**: Shows the landscape's bathymetry and food patches on the map (the simulation loads the landscape itself when it starts)
- **Wind Turbines**: Turbine scenario (filtered by landscape)
- **Load Turbines**: Shows the turbines on the map

#### Run Controls
- **Run Simulation**: Start a new run (progress bar and status above the buttons)
- **Stop**: End the current run; it cannot be resumed, and Run Simulation starts again from the beginning
- **Reset**: Clear the results, charts and porpoise positions

#### Speed Control
- **Simulation Speed** (1-100%, default 50%):
  - 1% = Slowest (0.3 s per simulated day) - watch individual movements
  - 50% = Medium (~0.075 s per day)
  - 100% = Maximum speed (no delay) - for long runs

#### Display Options
- **Skip visualization (fast run)**: stops map updates and the Population-tab histograms and table during the run; the charts and the CSV still fill
- **Show porpoise traces**: draws a short trail behind each porpoise; **Trace length** (1-5) sets how long

#### Social Communication (collapsed panel)
- **Enable Social Calls** (off by default) adds a pull towards calling neighbours; detection range, source level, detection threshold and response slope set who hears whom
- **Social Weight** (0-1, default 0.3): the pull as a fraction of the random walk's own pull (not fitted to observed group sizes)

### Main View Tabs

1. **Dashboard**: Interactive deck.gl map and population charts
2. **Model Settings**: Advanced parameter configuration
3. **Population**: Age/energy histograms and vital statistics
4. **Disturbance**: Dispersal and deterrence monitoring
5. **Landscape**: Viewer for the environmental data layers
6. **Export**: Download results as CSV

---

## Running a Simulation

### Step 1: Configure Population

1. In the sidebar, set **Initial Population** (default 100)
2. Set **Simulation Years** (1-5 years for quick tests, 10+ for population dynamics)

### Step 2: Select Landscape

1. Choose a **Landscape** from the dropdown:
   - **Homogeneous**: Uniform test grid (400×400 cells, 30 m deep, land along the edges)
   - **Lithuania** (default): South-eastern Baltic, Lithuanian waters; 1 km public data (EMODnet depth and seabed, Copernicus salinity), food patches with uniform prey for now, ship traffic from 2024 AIS
   - **CentralBaltic**: Baltic Proper, Bornholm to north of Gotland; 615×677 cells of 1 km from the same public sources, food patches with uniform prey, ship traffic from 2024 AIS
   - **Kattegat**: Kattegat / Inner Danish Waters (600×1000 @ 400 m), the DEPONS landscape
   - **NorthSea**: North Sea with real bathymetry (2088×2175 @ 400 m)
2. Click **Load Landscape** to display:
   - Depth overlay (bathymetry)
   - Foraging overlay (food patches; hidden at first, switch it on in the map's Layers panel)

> **Note**: Sources for the Baltic landscapes are described in `docs/2026-11-20-baltic-public-data.md`.

### Step 3: Add Wind Turbines (Optional)

1. Select a **Wind Turbines** scenario (options filtered by landscape):
   - Lithuania → Curonian Nord 35×15 MW (35 turbines) or Curonian Nord 60×10 MW (60 turbines)
   - NorthSea → Scenarios 1-3 (3,900 turbines each)
   - Homogeneous, CentralBaltic and Kattegat → No turbines
2. Click **Load Turbines** to display the turbine locations; during a run the map also draws noise circles around turbines (see [Dashboard Map](#dashboard-map))

### Step 4: Configure Advanced Settings (Optional)

1. Click the **Model Settings** tab
2. Adjust parameters (all have tooltip explanations):
   - **Basic**: Random seed, ship traffic (with Weston Flux TL), bycatch probability
   - **Movement**: CRW parameters (k, a0-a2, b0-b3) and Movement salinity
   - **Dispersal**: Dispersal type, tDisp, PSM parameters
   - **Energy**: Memory decay rates (rS, rR) and food replenishment rate (rU)
   - **JASMINE**: Subsystem mode overrides and JASMINE parameters

### Step 5: Run Simulation

1. Click **Run Simulation**
2. Watch the progress bar and status message
3. Adjust **Simulation Speed** as needed:
   - Slow down to observe porpoise movements
   - Speed up for long-duration runs
4. Click **Stop** to end the run (a new click on Run Simulation starts over)
5. Click **Reset** to clear the results

### Step 6: Analyze Results

1. View live updates on the **Dashboard**:
   - Map shows porpoise positions
   - Charts show population, births/deaths, food eaten and energy expended
2. Switch to **Population** tab for:
   - Age distribution histogram
   - Energy distribution histogram
   - Vital statistics table
3. Check **Disturbance** tab for:
   - Dispersing porpoises
   - Deterred porpoises
   - Noise model parameters and exposure over time

---

## Understanding the Results

### Dashboard Map

The interactive map shows:

- **Porpoises** (up to 1000 displayed), coloured by age: green under 2 years, blue 2-12 years, grey 12 years and older; red while dispersing
- **Wind turbines**: grey before a run (planned); during a run red while being piled and blue once operating
- **Red circles**: approximate noise zones around turbines being piled (drawn for display only; the model computes received levels itself)
- **Yellow circles**: operating turbines
- **Bathymetry**: water depth, yellow (shallow) to purple (deep)
- **Foraging**: food patches, green shading

**Map controls**:

- **Layers** panel (bottom-right): tick a row to show or hide a layer (background map, bathymetry, foraging, noise, turbines, porpoises, traces); a row appears once its layer is loaded
- Zoom and compass (top-right), fullscreen (top-left), scale bar (bottom-left)
- The **Population, Year, Births, Deaths** chips above the map show the current values
- The **Charts** bar below the map shows or hides the chart panel

### Dashboard Charts

| Chart | Description |
| ----- | ----------- |
| **Porpoise Population Size** | Total porpoises and lactating females with calves over time |
| **Life and Death** | Births (blue) and deaths (red) per update (one simulated day) |
| **Food Consumption and Expenditure** | Average food eaten vs energy expended per porpoise per day |

### Population Tab

| Visualization | Description |
|---------------|-------------|
| **Age Distribution** | Histogram of ages (0-30 years) |
| **Energy Distribution** | Histogram of energy levels (0-20 units) |
| **Landscape Energy Level** | Total food on the landscape over time |
| **Average Porpoise Movement** | Mean straight-line distance moved per day (km), from the second day |
| **Vital Statistics** | Current tick, day, year, population, total births and deaths, mean age, mean energy and mothers with calves |

### Disturbance Tab

| Visualization | Description |
|---------------|-------------|
| **Porpoise Dispersal** | "Dispersing Porpoises" and "Energy falling N+ days" (animals whose energy has fallen for at least tDisp days, the dispersal trigger) |
| **Deterrence Events** | Porpoises deterred by turbines or ships during each update |
| **Noise Exposure Map** | The noise propagation parameters (spreading, absorption, threshold, deterrence coefficient), turbine count, source level and deterrence radius, and porpoises exposed to noise over time |

### Reading the Numbers

Births and deaths are counted from the net change in population between updates, so a birth and a death on the same day cancel out. For how CENOP's population and energy compare with DEPONS 3.2 on Kattegat, see `docs/validation/A2-parity-report.md`.

---

## Parameter Reference

### Sidebar Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| Initial Population | 100 | 1-50,000 | Starting porpoise count |
| Simulation Years | 5 | 1-100 | Duration in simulated years |
| Simulation Mode | DEPONS | DEPONS, JASMINE | Model mode |
| Landscape | Lithuania | - | Geographic area |
| Simulation Speed | 50% | 1-100% | Run speed control |
| Social Weight | 0.3 | 0-1 | Pull towards calling neighbours (only with Enable Social Calls) |

### Model Settings: Basic Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| Random Seed | 0 | Reproducibility seed (0 = random each run) |
| Tracked Porpoises | 1 | Not used by the model or the app's outputs yet |
| Ship Traffic Enabled | Off | Enable vessel noise disturbance; Lithuania, CentralBaltic, Kattegat and NorthSea have their own ship files, Homogeneous uses a sample |
| Weston Flux TL | On | Shown when ship traffic is on: physics-based transmission loss using depth, sediment and salinity at the ship's position; off = simple spreading-loss formula |
| Annual Bycatch Probability | 0.0 | Annual probability of fishing net mortality (0-1) |

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
| Movement salinity | Auto | Salinity used in the movement model. Auto: fixed 34.07 on Kattegat, Lithuania and CentralBaltic, the landscape map elsewhere. Landscape map: map salinity everywhere. Fixed 34.07 everywhere. The DEPONS movement model was fitted to North Sea porpoises and is not valid at Baltic salinity (about 7); sound propagation always uses the map |

### Model Settings: Dispersal Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| Dispersal Type | PSM-Type2 | `PSM-Type2`: memory-based with SSLogis heading dampening. `Undirected`: random target, undamped turns. `InnerDanishWaters`: DEPONS Kattegat mode (block targets, then coast following), Kattegat landscape only. `off`: no dispersal |
| tDisp | 3 days | Consecutive days of declining energy to trigger dispersal |
| PSM_log | 0.6 | Memory strengthening rate |
| PSM_dist | N(350;100) | Preferred dispersal distance (km): mean 350, standard deviation 100 |
| PSM_tol | 5 km | Target tolerance distance |
| PSM_angle | 40° | Maximum turn per step |

### Model Settings: Energy Tab

| Parameter | Default | Description |
|-----------|---------|-------------|
| rS | 0.03 | Satiation memory decay rate (DEPONS 3.2) |
| rR | 0.03 | Reference memory decay rate (DEPONS 3.2) |
| rU | 0.1 | Food replenishment rate |

### JASMINE Mode Parameters

These parameters apply when JASMINE mode (or the matching subsystem override) is selected. Of the JASMINE parameters, only Memory Decay Rate and Avoidance Radius are read by the model; the others are labelled "not used yet" in the app.

#### Subsystem Mode Overrides

| Setting | Default | Description |
|---------|---------|-------------|
| Movement Mode | Default (DEPONS) | JASMINE = experimental simplified walk (DEPONS step lengths, small random turns, no food memory or dispersal steering) |
| Energy Mode | Default (DEPONS) | JASMINE = experimental energy budget in physical units, not calibrated |
| Time Mode, Behavior FSM Mode, Memory Mode | Follow main mode | Memory Mode JASMINE = learned avoidance, DEPONS = no disturbance memory |

#### Learned Avoidance

| Parameter | Default | Description |
|-----------|---------|-------------|
| Memory Decay Rate | 0.016 | Per-tick memory decay (a full memory fades in about 3 days) |
| Avoidance Radius | 2.5 | Influence radius in memory cells of 2 km (5 km) |
| Avoidance Strength | 0.8 | Not used yet |

The disturbance is remembered at the animal's own cell. Habituation is not implemented.

#### Physics and DEB Energy Parameters (not used yet)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Body Mass | 50.0 kg | Adult porpoise body mass |
| Drag Coefficient | 0.01 | Hydrodynamic drag coefficient |
| Max Thrust | 100.0 N | Maximum propulsive thrust |
| Current Weight | 0.5 | Ocean current influence (0-1) |
| BMR Scale Factor | 1.0 | Basal metabolic rate multiplier |
| Activity Cost Multiplier | 2.0 | Activity cost multiplier |
| Disturbance Cost Multiplier | 1.5 | Energy cost multiplier during disturbance |

---

## Data Export

### CSV Export (UI)

1. Run a simulation
2. Go to the **Export** tab
3. Click **Download Results CSV** (file `cenop_results.csv`)

**Exported columns** (one row per update, one simulated day by default):
- `day`, `tick`, `year`: Time markers (1 tick = 30 minutes)
- `population`: Total living porpoises
- `lact_calf`: Lactating females with calves
- `births`, `deaths`: Cumulative counts (from net population change, see [Reading the Numbers](#reading-the-numbers))
- `energy_entry`: Average food eaten and energy expended per porpoise per day, total food on the landscape (`landscape_food`) and the mean distance moved per day (`movement_km_per_day`, km; none at the first update)
- `dispersal_entry`: Dispersing and energy-declining counts (stored as a dictionary)
- `deterred_count`: Porpoises deterred during the update

### DEPONS-Compatible Outputs (Python API)

For advanced users, DEPONS-style output files are available via Python:

```python
from cenop import SimulationParameters
from cenop.core.output_writer import OutputConfig, OutputWriter
from cenop.server.simulation_controller import build_simulation

params = SimulationParameters(porpoise_count=1000, sim_years=1, landscape="Kattegat")
sim = build_simulation(params)  # the same model the app builds

config = OutputConfig(output_dir="output", run_id="simulation_001")
with OutputWriter(config) as writer:
    for _ in range(sim.max_ticks):
        sim.step()
        writer.record_tick(sim)
```

**Output files** (tab-separated; `_<run_id>` is added to each name when `run_id` is set):
- `Population.txt`: Daily population counts, births and deaths by cause
- `PorpoiseStatistics.txt`: Individual porpoise data
- `Mortality.txt`: Death events with causes
- `Dispersal.txt`: Dispersal events
- `Energy.txt`: Energy statistics

---

## Troubleshooting

### Common Issues

#### Simulation Runs Slowly

**Cause**: Large population, ship traffic, or map updates

**Solutions**:
- Reduce initial population
- Increase simulation speed to 100%
- Tick **Skip visualization (fast run)**
- Runs with ship traffic take much longer than runs without
- Close other browser tabs

#### Porpoises Disappear Quickly

**Cause**: Population collapse from starvation or disturbance

**Solutions**:
- Use Homogeneous landscape (uniform food)
- Disable turbines and ships initially
- Check energy parameters

#### Map Doesn't Update

**Cause**: Browser or connection issues, or Skip visualization is ticked

**Solutions**:
- Untick **Skip visualization (fast run)**
- Wait for "Load Landscape" to complete
- Refresh the page (F5)
- After a short network drop, wait: the page reconnects to the running session and redraws the maps
- Check browser console for errors

#### Speed Slider Doesn't Respond

**Cause**: The new speed is applied between updates

**Solutions**:
- Wait a moment after changing; the new speed takes effect from the next simulated day

### Getting Help

1. Click **❓ Help** in the top navigation bar
2. Check this documentation
3. Contact arturas.razinkovas-baziukas@ku.lt

---

## Appendix: Scientific Notes

### Model Validation

**DEPONS Mode** follows the DEPONS 3.2 Java implementation across five subsystems:
- Reproduction (pregnancy FSM, weaning-based calf creation)
- Food & energy (logistic regrowth, MaxEnt carrying capacity, starvation formula)
- Movement & memory (reference memory circular buffers, CRW rejection sampling, heading composition)
- Dispersal (PSM-Type2 with SSLogis heading and energy-based stop, Undirected, InnerDanishWaters)
- Deterrence (raw displacement vectors, Weston flux TL, JOMOPANS source levels)

The feature-by-feature comparison is in `docs/DEPONS-CENOP-PARITY-ANALYSIS.md`. Runs on the Kattegat landscape were compared with DEPONS 3.2 runs (eight seeds per model, four scenarios including ships) in `docs/validation/A2-parity-report.md`. The test suite has about 1,270 fast tests and 11 slow validation tests.

**JASMINE Mode** is experimental and designed for:
- Exploring behavioural hypotheses
- Testing learned avoidance scenarios
- Evaluating bioenergetics models
- Not calibrated or validated for regulatory use

### Time Steps

- 1 tick = 30 minutes
- 48 ticks = 1 day
- 17,280 ticks = 1 year (360 days, DEPONS convention)

### Spatial Resolution

- Cell size: 400 m × 400 m
- Landscapes on other grids (Lithuania and CentralBaltic at 1 km) are resampled to 400 m cells when loaded
- Grid varies by landscape (e.g., 400×400 for Homogeneous)

### Energy Model

- **Seasonal scaling**: Nov-Mar = 1.0×, Apr and Oct = 1.15×, May-Sep = 1.3×
- **Lactation cost**: 1.4× normal metabolism
- **Starvation mortality**: yearly survival = 1 − 1.0 × exp(−0.4 × energy), applied as a per-tick probability, so low energy raises the risk of death smoothly; a starving mother first abandons her calf

### Behavioral States (JASMINE Mode)

The state labels each animal and sets its activity cost in the JASMINE energy budget; it does not change how the animal moves, and the app does not display it.

| State | Assigned when | Activity cost |
|-------|---------------|---------------|
| FORAGING | Default | ×1.0 |
| TRAVELING | Step longer than 2 cells per tick | ×1.5 |
| RESTING | Energy below 30% of the maximum | ×0.6 |
| DISPERSING | The model is dispersing the animal | ×1.2 |
| DISTURBED | Deterred; until a day without deterrence | ×2.0 |

### Performance Benchmarks

Measured on Kattegat without ships or turbines (`docs/performance/2026-11-20-tick-profile.md`):

| Population Size | Time per tick |
|----------------|---------------|
| 500 agents | 1.9-2.3 ms (InnerDanishWaters, PSM-Type2) |
| 2,000 agents | 4.9-7.4 ms (InnerDanishWaters, PSM-Type2) |

With ship traffic a tick at 2,000 agents takes about 34 ms (`docs/performance/2026-11-20-ship-loop.md`). The current speed ratio to DEPONS-Java has not been re-measured.

### Citation

If using CENOP-JASMINE in publications, please cite:
```
CENOP-JASMINE: CETacean Noise-Population Model with JASMINE Extensions
A Python Shiny implementation aligned with DEPONS 3.2 with experimental behavioural extensions
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
