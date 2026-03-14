# CENOP-JASMINE

<img src="static/CENOP_logo.png" alt="CENOP Logo" height="80">

**CETacean Noise-Population Model with JASMINE Extensions**

CENOP is a Python translation of the DEPONS (Disturbance Effects of POrpoises in the North Sea) agent-based model, algorithmically aligned with DEPONS 3.2. It simulates how harbour porpoise population dynamics are affected by disturbances from offshore wind farm construction and ship noise.

The JASMINE (Just Another Simulation Model In Nature Environments) extension adds research-grade physics-based movement, dynamic energy budgets, and learned avoidance behaviors.

## Simulation Modes

- **DEPONS Mode (Default):** Regulatory-compatible empirical models aligned with DEPONS 3.2 algorithms (pregnancy FSM, dispersal, deterrence, reference memory, CRW rejection sampling)
- **JASMINE Mode (Research):** Physics-based movement, Dynamic Energy Budget (DEB), and learned animal behaviors

## Features

### Core Simulation
- Agent-based simulation of harbour porpoise populations (Structure-of-Arrays vectorized architecture)
- Numba JIT-compiled kernels for hot-path operations with prange parallelism
- Correlated Random Walk (CRW) with rejection sampling (DEPONS 3.2 algorithm)
- Pregnancy finite state machine with daily mortality schedule
- Persistent spatial memory (PSM) and reference memory for food-area attraction
- PSM-Type2 dispersal with SSLogis heading dampening
- Energy-based dispersal triggering and stopping
- Logistic food regrowth with proportional-sharing consumption
- Noise disturbance modeling (pile-driving and ship noise)
- Ship deterrence with JOMOPANS 13-class source levels and Weston flux transmission loss

### Landscapes
- North Sea, Kattegat, Central Baltic, and Lithuanian waters with real bathymetry
- Monthly salinity fields driving CRW movement modulation
- Food probability grids with logistic regrowth dynamics
- Monthly MaxEnt prey distribution layers (sets carrying capacity for food patches)
- All grids at mandatory 400m cell resolution

### Web Interface
- Interactive Shiny web interface with real-time visualization
- 6 tabs: Dashboard, Settings, Population, Disturbance, Landscape Editor, Export
- DeckGL map with porpoise positions, depth heatmap, turbine locations, noise contours, food patches
- Collapsible chart panel with population, births/deaths, and energy time series
- Comprehensive in-app help documentation

### JASMINE Extensions
- **Behavioral State Machine:** FORAGING, TRAVELING, RESTING, DISPERSING, and DISTURBED states
- **Dynamic Energy Budget:** Body mass-dependent metabolism, activity costs, thermoregulation
- **Disturbance Memory:** Spatial memory with learned avoidance and habituation
- **Physics-Based Movement:** Hydrodynamic drag, thrust-based propulsion, and ocean current advection

### Performance
- 7 Numba JIT kernels: `reflect_boundaries`, `seed_numba_rng`, `crw_angle_step`, `turn_position`, `eat_food`, `depons_bmr_cost`, `social_accumulate`
- Three kernels parallelized with `prange` (reflect, turn_position, BMR cost)
- Pre-allocated float64 buffers, RefMemWorkspace (~1.5 MB/tick saved), vectorized land avoidance
- Sub-millisecond per-kernel performance for 500 agents
- 502+ tests passing across 24 test files (unit, integration, equivalence, and parallel determinism tests)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd CENOP

# Create virtual environment (or use micromamba)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Or install as package
pip install -e .
```

### Micromamba (recommended for Numba compatibility)

```bash
eval "$(micromamba shell hook --shell bash)"
micromamba activate shiny
```

## Quick Start

```bash
# Run the Shiny application
shiny run app.py
```

Then open your browser to http://localhost:8000

## Running Tests

```bash
# Full test suite (502+ tests)
python3 -m pytest tests/ -x -q

# Numba kernel tests only
python3 -m pytest tests/test_numba_kernels.py -v

# Kernel benchmark
python3 scripts/benchmark_kernels.py
```

## Deployment

### Production Server: laguna.ku.lt

The application is deployed on the Shiny Server at `laguna.ku.lt`.

**Server Configuration:**
- Shiny Server path: `/srv/shiny-server/cenjas` (symlink)
- Application directory: `/home/razinka/cenjas`
- User: `razinka`
- URL: https://laguna.ku.lt/cenjas/

### Windows Deployment (Recommended)

Use the provided `deploy.cmd` script from Windows. Double-click or run from command prompt:

```cmd
deploy.cmd
```

The interactive menu provides the following options:

```
[1] Full deployment (pull, install, permissions, restart)
[2] Pull latest changes only
[3] Update dependencies only
[4] Fix permissions for shiny user only
[5] Restart Shiny Server only
[6] View server logs
[7] Check application status
[0] Exit
```

### Manual Deployment (Linux/SSH)

```bash
# 1. SSH into the server
ssh razinka@laguna.ku.lt

# 2. Navigate to the application directory
cd ~/cenjas

# 3. Pull the latest changes
git fetch origin && git reset --hard origin/main

# 4. Update dependencies
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# 5. Set permissions for shiny user
find ~/cenjas -type d -exec chmod 755 {} \;
find ~/cenjas -type f -exec chmod 644 {} \;
chmod 755 ~/cenjas/venv/bin/*
chmod 755 /home/razinka

# 6. Restart Shiny Server
sudo systemctl restart shiny-server
```

### Shiny User Permissions

The shiny server runs as the `shiny` user, which needs read access to the application files. Required permissions:
- Home directory `/home/razinka`: 755 (allows shiny to traverse)
- All directories: 755 (read + execute for shiny)
- All files: 644 (read for shiny)
- venv binaries: 755 (executable)

The `deploy.cmd --permissions-only` command sets these automatically

## Project Structure

```
CENOP/
├── app.py                      # Shiny entry point (imports UI + server, warms up Numba)
├── pyproject.toml              # Package metadata and dependencies
├── requirements.txt            # Pip dependencies
├── deploy.cmd                  # Windows deployment script
├── static/                     # Logo, icons, SVG assets
├── scripts/
│   ├── benchmark_kernels.py    # Numba kernel performance benchmark
│   ├── generate_central_baltic.py
│   └── download_emodnet_*.py   # Bathymetry data acquisition
├── data/
│   ├── NorthSea/               # North Sea landscape (2088×2175 @ 400m)
│   ├── Kattegat/               # Kattegat landscape (600×1000 @ 400m)
│   ├── CentralBaltic/          # Central Baltic landscape
│   ├── Lithuania/              # Lithuanian waters (Curonian Nord)
│   ├── Gemini/                 # Gemini wind farm area
│   ├── DanTysk/                # DanTysk wind farm area
│   └── wind-farms/             # Turbine scenario definitions
├── src/cenop/                  # Core simulation package
│   ├── agents/                 # Agent definitions
│   │   ├── population.py       # SoA vectorized population (main simulation engine)
│   │   ├── porpoise.py         # Individual porpoise data structures
│   │   ├── ship.py             # Shipping agents
│   │   └── turbine.py          # Wind turbine agents
│   ├── behavior/               # Behavioral modules
│   │   ├── hybrid_fsm.py       # 5-state behavioral state machine
│   │   ├── psm.py              # Persistent spatial memory (food)
│   │   ├── ref_mem.py          # Reference memory (vectorized attraction)
│   │   ├── dispersal.py        # PSM-Type2 dispersal with SSLogis heading
│   │   ├── disturbance_memory.py  # Learned avoidance with habituation
│   │   ├── sound.py            # Sound/disturbance event handling
│   │   ├── jomopans_spl.py     # 13-class JOMOPANS ship source levels
│   │   └── weston_flux.py      # Physics-based transmission loss
│   ├── core/                   # Simulation engine
│   │   ├── simulation.py       # Main simulation controller
│   │   ├── scheduler.py        # Event scheduler
│   │   ├── time_manager.py     # Time tracking (48 ticks/day)
│   │   ├── batch_runner.py     # Batch simulation runner
│   │   ├── random_source.py    # Random number management
│   │   ├── output_writer.py    # Results export
│   │   └── profiler.py         # Performance profiling
│   ├── landscape/              # Environmental data
│   │   ├── loader.py           # ASC grid file parser (validates 400m cells)
│   │   └── cell_data.py        # Cell data (bathymetry, food, salinity)
│   ├── movement/               # Movement systems
│   │   ├── depons_crw.py       # Correlated random walk (rejection sampling)
│   │   ├── hybrid.py           # Mode selector (DEPONS vs JASMINE)
│   │   └── jasmine_physics.py  # Physics-based movement
│   ├── optimizations/          # Performance optimizations
│   │   ├── kernels.py          # Numba JIT kernels (7 kernels + warmup)
│   │   └── numba_helpers.py    # Numba utility functions
│   ├── parameters/             # Configuration
│   │   ├── constants.py        # Fixed constants (REQUIRED_CELL_SIZE = 400)
│   │   ├── demography.py       # Demographic parameters
│   │   └── simulation_params.py  # Runtime simulation parameters
│   ├── physiology/             # Energy budget
│   │   └── energy_budget.py    # DEPONS & JASMINE energy systems
│   ├── server/                 # Shiny server logic
│   │   ├── main.py             # Server function and render callbacks
│   │   ├── reactive_state.py   # Reactive simulation state
│   │   ├── simulation_controller.py  # Simulation start/stop/pause
│   │   ├── map_layers.py       # DeckGL map layer construction
│   │   └── renderers/
│   │       ├── chart_helpers.py  # Plotly chart rendering
│   │       └── gis_editor.py    # Landscape editor interactions
│   └── ui/                     # Shiny UI components
│       ├── layout.py           # Main layout, CSS theme, help modal
│       ├── sidebar.py          # Sidebar controls
│       └── tabs/
│           ├── dashboard.py    # Map + live charts
│           ├── settings.py     # Parameter configuration
│           ├── population.py   # Population statistics
│           ├── disturbance.py  # Disturbance impacts
│           ├── landscape_editor.py  # Spatial data viewer
│           └── export.py       # Results download
└── tests/                      # Test suite (502+ tests)
    ├── conftest.py             # Fixtures, Numba/coverage compatibility
    ├── test_numba_kernels.py   # Numba kernel tests (25 tests)
    ├── test_integration.py     # Full simulation integration tests
    ├── test_depons_*.py        # DEPONS algorithm validation
    ├── test_energy_budget.py   # Energy system tests
    ├── test_ref_memory.py      # Reference memory tests
    ├── test_dispersal.py       # Dispersal behavior tests
    ├── test_phase5.py          # Phase 5 integration tests
    └── ...                     # 24 test files total
```

## Configuration

### Selecting Simulation Mode

**Via UI:** Select "DEPONS (Regulatory)" or "JASMINE (Research)" from the sidebar dropdown.

**Via Code:**
```python
from cenop import Simulation, SimulationParameters

params = SimulationParameters(
    porpoise_count=1000,
    sim_years=5,
    simulation_mode="JASMINE",  # or "DEPONS"
    # Optional subsystem overrides:
    energy_mode="JASMINE",      # Use DEB energy budget
    memory_mode="JASMINE",      # Use learned avoidance
    fsm_mode="JASMINE",         # Use enhanced behavioral FSM
)

sim = Simulation(params)
```

### JASMINE-Specific Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `jasmine_mass_kg` | 50.0 | Body mass (kg) |
| `jasmine_drag_coeff` | 0.01 | Hydrodynamic drag coefficient |
| `jasmine_bmr_scale` | 1.0 | Basal metabolic rate scale factor |
| `memory_decay_rate` | 0.001 | Memory decay per tick |
| `habituation_enabled` | True | Enable habituation to disturbance |

## Validation

- **DEPONS mode:** Algorithmically aligned with DEPONS 3.2 — pregnancy FSM, daily mortality, dispersal (PSM-Type2), deterrence, reference memory, CRW with rejection sampling
- **JASMINE mode:** Research-grade, designed for exploring advanced behavioral hypotheses

## License

This project is licensed under the GNU General Public License v2.0, following the original DEPONS model.

## Acknowledgments

- Original DEPONS model by Jacob Nabe-Nielsen, Aarhus University
- JASMINE behavioral extensions developed at Klaipeda University
- EU Horizon 2020 SATURN project (GA 101006443)
- arturas.razinkovas-baziukas@ku.lt
