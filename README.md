# CENOP

<img src="static/CENOP_logo.png" alt="CENOP Logo" height="80">

**CETacean Noise-Population Model**

CENOP is a Python agent-based model for simulating harbour porpoise population
dynamics under anthropogenic noise disturbance (offshore wind-farm construction
and vessel traffic). It implements two complementary modelling frameworks:

- **DEPONS** (Disturbance Effects on the harbour Porpoise population in the
  North Sea) — fixed-timestep, empirically-calibrated Correlated Random Walk
  (CRW) movement with regulatory-grade reproducibility.
- **JASMINE** (Joint Agent Simulations of Marine Interactions with Noise and
  the Environment) — flexible-timestep, physics-based movement with symplectic
  integration, event scheduling, and advection fields.

Both modes share a unified **Structure-of-Arrays (SoA)** vectorised population
manager that operates entirely on NumPy arrays, enabling efficient simulation of
large populations without per-agent Python objects.

## Key Features

- Vectorised SoA population manager (positions, headings, energy, ages, PSM
  buffers — all contiguous NumPy arrays)
- Dual movement engine — DEPONS CRW and JASMINE physics, selectable per run or
  via a hybrid context-switching strategy
- Persistent Spatial Memory (PSM) with configurable dispersal types
- Energy budget, reproduction, mortality (starvation, natural, bycatch)
- Noise propagation from pile-driving (construction) and operational turbines
- Ship deterrence with probabilistic dose–response
- Landscape loading from DEPONS-format ASC bathymetry files (North Sea, Central
  Baltic, or homogeneous)
- Per-simulation `np.random.Generator` for thread-safe, reproducible runs
- Batch runner with parameter sweeps and Monte Carlo replication
- Interactive [Shiny for Python](https://shiny.posit.co/py/) web UI with
  real-time maps, charts, and controls
- Comprehensive test suite (320 tests)

## Installation

```bash
# Clone the repository
git clone https://github.com/razinkele/CENOP.git
cd CENOP
git checkout CENOP-JASMINE

# Option A: pip (venv)
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"

# Option B: micromamba / conda
micromamba create -n cenop python=3.13 -y
micromamba activate cenop
pip install -e ".[dev]"
```

## Quick Start

```bash
# Run the interactive Shiny application
shiny run app.py
```

Open your browser at http://localhost:8000.

```python
# Or run a simulation programmatically
from cenop import Simulation, SimulationParameters

params = SimulationParameters(
    porpoise_count=200,
    sim_years=5,
    landscape_name="Homogeneous",
)
sim = Simulation(params=params, seed=42)
sim.initialize()

for _ in range(1000):
    sim.step()

print(f"Population after 1 000 ticks: {sim.population_size}")
```

## Project Structure

```
CENOP/
├── app.py                      # Shiny application entry point
├── pyproject.toml              # Build metadata & dependencies (hatchling)
├── src/cenop/                  # Main simulation package
│   ├── core/                   # Simulation engine, time manager, batch runner
│   │   ├── simulation.py       # Simulation orchestrator
│   │   ├── time_manager.py     # DEPONS/JASMINE unified time & seeding
│   │   ├── batch_runner.py     # Parameter sweeps & Monte Carlo runs
│   │   └── output_writer.py    # DEPONS-compatible file output
│   ├── agents/                 # Agent definitions
│   │   ├── population.py       # Vectorised SoA population manager
│   │   ├── ship.py             # Vessel traffic & noise
│   │   └── turbine.py          # Wind-farm turbines & piling schedules
│   ├── movement/               # Pluggable movement modules
│   │   ├── depons_crw.py       # Empirical CRW (DEPONS 3.0)
│   │   ├── jasmine_physics.py  # Physics-based (JASMINE)
│   │   └── hybrid.py           # Context-switching selector
│   ├── behavior/               # Behavioral sub-models
│   │   ├── psm.py              # Persistent Spatial Memory
│   │   ├── dispersal.py        # Dispersal strategies
│   │   ├── sound.py            # Acoustic propagation & dose–response
│   │   └── memory.py           # Disturbance memory
│   ├── landscape/              # Environmental data loaders
│   ├── parameters/             # Simulation parameters & demography
│   ├── physiology/             # Energy budget model
│   ├── server/                 # Shiny server-side logic
│   └── ui/                     # Shiny UI layout & components
├── data/                       # Landscape data (bathymetry, food, wind farms)
├── tests/                      # Test suite (pytest, 320 tests)
├── docs/                       # Documentation & proposed fixes
└── static/                     # Logo and web assets
```

## Running Tests

```bash
pytest tests/ -q
```

## Simulation Modes

### DEPONS Mode (default)

Fixed 30-minute timesteps (48 ticks/day), deterministic seeding per tick,
empirical CRW movement calibrated to North Sea GPS telemetry data. Designed for
regulatory impact assessment of offshore wind farms.

### JASMINE Mode

Flexible sub-stepping with event-driven scheduling, physics-based movement using
symplectic integration, ocean current advection fields, and adaptive update
frequencies. Designed for research on movement ecology and multi-stressor
interactions.

### Hybrid Mode

Context-dependent switching between DEPONS CRW and JASMINE physics based on
behavioural state (e.g. CRW during foraging, physics during deterrence
response). Configurable via `HybridStrategy`.

## License

This project is licensed under the GNU General Public License v2.0, following
the original DEPONS model. See [LICENSE](LICENSE) for details.

## Acknowledgments

- Original DEPONS model by Jacob Nabe-Nielsen, Aarhus University
- EU Horizon 2020 SATURN project (GA 101006443)
