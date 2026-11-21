# CENOP-JASMINE API Documentation

## Overview

CENOP-JASMINE (CETacean Noise-Population Model with JASMINE Extensions) is a Python Shiny application for simulating harbour porpoise population dynamics in response to offshore wind farm construction and ship noise. This document describes the core API for developers and researchers.

The API supports two simulation modes:
- **DEPONS Mode**: Regulatory-compatible empirical models aligned with DEPONS 3.2 (default)
- **JASMINE Mode** (experimental): DEPONS movement and the DEPONS energy budget, plus learned avoidance of remembered disturbances and behaviour states. The JASMINE movement (a simplified walk: DEPONS step lengths, random turns; the physics integrator in `src/cenop/movement/jasmine_physics.py` is not wired in) and the uncalibrated JASMINE energy budget run only with `movement_mode="JASMINE"` / `energy_mode="JASMINE"`.

`Simulation(params)` on its own runs the DEPONS model without any subsystem modules. To run JASMINE mode (or any subsystem override) from code, build the simulation with `cenop.server.simulation_controller.build_simulation(params, landscape=None)`, as the app does (see [Usage Examples](#usage-examples)).

---

## DEPONS 3.2 Algorithmic Parity

- **Code reading (March 2026):** [DEPONS-CENOP Parity Analysis](DEPONS-CENOP-PARITY-ANALYSIS.md) lists 136 features as MATCH and 13 intentional divergences (vectorization, proportional food sharing, explicit FSM), plus the CENOP extensions. Its status section (2026-11-20) notes that runs of DEPONS 3.2 later showed several MATCH rows did not hold; those rows were corrected and the defects fixed, and the rest of the table means "the formulas were ported", not "the behaviour was verified".
- **Runs against DEPONS 3.2 (November 2026):** [A2 report](validation/A2-parity-report.md), Kattegat, eight seeds per model, four scenarios (undisturbed with three dispersal modes, and ships). Every metric passes or is inconclusive except mean energy in the ship scenario at day 150 (4.29 against 4.14), which a sixteen-seed check fixed in advance did not confirm (+0.05, 1.0 standard error).
- **Dispersal:** the four modes DEPONS 3.2 offers (`off`, `PSM-Type2`, `Undirected`, `InnerDanishWaters`). PSM-Type1 and the PSM-Type3 variants exist in the DEPONS code but are not offered by DEPONS either, and are not ported.

### Performance (Kattegat, 2026-11-20)

Default NumPy/Numba backend, seed 42. Sources: [tick profile](performance/2026-11-20-tick-profile.md) (days 10-20, no ships or turbines) and [ship loop](performance/2026-11-20-ship-loop.md) (days 1-2, 5,148 ships). Measured on a shared machine.

| Run | ms/tick |
|-----|---------|
| 500 animals, InnerDanishWaters | 1.88 |
| 500 animals, PSM-Type2 | 2.26 |
| 2000 animals, InnerDanishWaters | 4.94 |
| 2000 animals, PSM-Type2 | 7.41 |
| 2000 animals, with ships | about 34 |

---

## Core Modules

### 1. Landscape Module

**Location:** `src/cenop/landscape/cell_data.py`

The landscape module manages the spatial environment: bathymetry, food patches and food levels, salinity, sediment, blocks and the monthly MaxEnt prey layers. Positions are continuous grid coordinates `(x, y)` in cells; landscapes not on 400 m cells are resampled to 400 m when they are loaded (`cenop.landscape.resample`).

#### Classes

##### `CellData`

```python
class CellData:
    """
    Manages the landscape layers loaded from the ASC files in data/<landscape_name>/.

    Attributes:
        metadata (LandscapeMetadata): Grid geometry (ncols, nrows, xllcorner, yllcorner, cellsize, nodata_value)
        _depth (np.ndarray): Bathymetry (m)
        _food_prob (np.ndarray): Food patch layer (only cells with food_prob > 0 hold food)
        _food_value (np.ndarray): Current food level per cell (dynamic, depleted by foraging)
        _salinity (np.ndarray): Monthly salinity, shape (12, height, width)
        _entropy (np.ndarray): Monthly MaxEnt prey layers, shape (12, height, width)
    """

    def __init__(self, landscape_name: str, data_dir: str = None):
        """Initialize CellData for a named landscape (e.g., 'Kattegat', 'Lithuania').
        data_dir defaults to the repository's data/ directory."""

    def load(self) -> None:
        """Load all layers (also done lazily on first access)."""

    def get_depth(self, x: float, y: float) -> float:
        """Get water depth at a position."""

    def get_food_level(self, x: float, y: float) -> float:
        """Get current food level at a position."""

    def eat_food(self, x: float, y: float, fraction: float) -> float:
        """Eat a fraction (0-1) of the cell's food, returning the amount eaten.
        A cell that had food keeps at least 0.01."""

    def replenish_food(
        self,
        rate: float,
        max_u: float = 1.0,
        regrowth_qualifier: float = 0.001,
        max_ent: np.ndarray | None = None,
        mean_max_ent_in_quarter: float = 1.0,
    ) -> None:
        """DEPONS 3.2 logistic regrowth F += rU * F * (1 - F/K), repeated 47 more times
        (48 in all) in cells where the first step changed food by more than
        regrowth_qualifier. Simulation calls it once per simulated day."""

    def get_salinity(self, x: float, y: float, month: int | None = None) -> float:
        """Get salinity at a position for a month (default: the current month)."""
```

Vectorized variants used by the population: `get_food_levels_vectorized`, `eat_food_vectorized`, `get_depths_vectorized`, `get_salinities_vectorized`, `get_sediments_vectorized`. `create_homogeneous_landscape(width=400, height=400, depth=30.0, food_prob=0.5)` builds the test landscape used for `landscape="Homogeneous"`.

---

### 2. Population Module

**Location:** `src/cenop/agents/population.py`

Structure-of-Arrays (SoA) vectorized implementation of porpoise population dynamics. All animals are female, as in DEPONS.

#### Classes

##### `PorpoisePopulation`

```python
class PorpoisePopulation:
    """
    SoA vectorized porpoise population management.

    Core arrays (shape (n,); the arrays grow when weaned calves need slots):
        x, y (np.float32): Grid coordinates (cells)
        heading (np.float32): Heading in degrees [0, 360), 0 = north
        energy (np.float32): Energy reserves (0-20 scale)
        age (np.float32): Age in years
        pregnancy_status (np.int8): 0=immature, 1=pregnant, 2=ready-to-mate
        with_calf (np.bool_): Nursing a calf
        is_dispersing (np.bool_): Currently dispersing (any dispersal mode)
        active_mask (np.bool_): True for living porpoises
        deter_strength (np.float32): Deterrence magnitude this tick

    Reference memory arrays (circular buffers, shape (n, ref_mem_size), default 120):
        _pos_history_x, _pos_history_y (np.float32): Remembered positions
        _stored_util (np.float32): Food stored with each remembered position
        _mem_ptr, _mem_count (np.int32, shape (n,)): Write index and number of entries
    """

    def __init__(
        self,
        count: int,
        params: SimulationParameters,
        landscape: CellData | None = None,
        movement_module=None,
        behavior_fsm=None,
        energy_module=None,
        memory_module=None,
    ):
        """Initialize population with SoA arrays. Simulation.initialize() creates it."""

    def step(
        self,
        deterrence_vectors: tuple[np.ndarray, np.ndarray] | None = None,
        ambient_rl: np.ndarray | None = None,
        turbine_deterrence_vectors: tuple[np.ndarray, np.ndarray] | None = None,
    ):
        """Execute one simulation tick for all active porpoises: movement, land avoidance,
        positions, food intake, starvation check, energy costs, disturbance memory,
        reproduction (daily)."""

    @property
    def population_size(self) -> int:
        """Current number of living porpoises."""

    def to_dataframe(self) -> pd.DataFrame:
        """Active agents: id, x, y, age, energy, heading, is_disturbed, behavioral_state, depth, alive."""

    def get_energy_stats(self) -> dict:
        """mean, std, min, max, hungry (energy < 10), starving (energy < 5)."""

    def get_dispersal_stats(self) -> dict:
        """dispersing_count, total_active, avg_psm_cells, max_declining_days, declining_count."""
```

Internal steps (private): `_apply_dispersal_heading(mask)` (dispersal turn, SSLogis-damped for PSM-Type2), `_check_mortality(mask, active_before)` (starvation every tick; bycatch and max-age daily, at tick 1 and when `tick % 48 == 0`), `_update_pregnancy_status(mask)` (daily pregnancy FSM), `_eat_food_vectorized(mask, fract_to_eat)` (food is eaten at the cell the animal occupied at the start of the tick). `step()` caches the active indices once per tick and reuses pre-allocated per-tick buffers.

---

### 3. Energy Budget Module

**Location:** `src/cenop/physiology/energy_budget.py`

Dual-mode energy system supporting DEPONS and JASMINE energy models. Without an energy module (plain `Simulation(params)`), the population applies the DEPONS energy budget itself.

#### Classes

##### `EnergyModule` (ABC)

```python
class EnergyModule(ABC):
    """Abstract base class for energy budget modules."""

    def __init__(self, params: SimulationParameters): ...

    @abstractmethod
    def compute_energy_update(
        self, state: EnergyState, context: EnergyContext, mask: np.ndarray, dt_seconds: int = 1800
    ) -> EnergyResult:
        """Compute energy changes for one tick."""

    @abstractmethod
    def apply_result(self, state: EnergyState, result: EnergyResult, mask: np.ndarray) -> None:
        """Apply an EnergyResult to the state."""

    @abstractmethod
    def compute_survival_probability(self, state: EnergyState, mask: np.ndarray) -> np.ndarray:
        """Compute per-tick survival probability for each agent."""

    @abstractmethod
    def get_mode(self) -> EnergyMode:
        """EnergyMode.DEPONS or EnergyMode.JASMINE."""
```

Both implementations also provide `compute_food_intake(state, context, mask)` and `compute_bmr_cost(state, context, mask)`. These two phases are what the population calls (food, then the starvation check, then costs); it does not call `apply_result` or `compute_survival_probability`, and starvation always uses the DEPONS formula on the population's energy.

##### `DEPONSEnergyModule`

```python
class DEPONSEnergyModule(EnergyModule):
    """
    DEPONS 3.2 energy model.

    - Energy scale: 0-20 (dimensionless)
    - BMR per tick: 0.001 * seasonal scaling * e_use_per_30_min (4.5)
    - Seasonal scaling: May-Sep = e_warm (1.3), Apr and Oct = 1.15, Nov-Mar = 1.0
    - Lactation cost: e_lact (1.4) x BMR
    - Starvation formula: yearlySurv = 1 - m_mort_prob_const * exp(-energy * x_survival_const)
      with m_mort_prob_const=1.0, x_survival_const=0.4 (DEPONS 3.2 calibration)
    - BMR computed via depons_bmr_cost_kernel (Numba)
    - Swimming (e_use_per_km) and disturbance (jasmine_disturbance_energy) terms are off by default
    """
```

##### `JASMINEEnergyModule`

See [section 6](#6-dynamic-energy-budget-jasmine).

##### `EnergyState`

```python
@dataclass
class EnergyState:
    """Per-agent energy state arrays. Create with EnergyState.create(count, initial_energy=10.0)."""
    energy: np.ndarray                     # Current energy level (0-20); shares the population's array
    body_mass: np.ndarray                  # Body mass (kg)
    body_condition: np.ndarray             # Body condition index (0-1)
    fat_reserve: np.ndarray                # Fat reserve (kg)
    activity_level: np.ndarray
    distance_traveled: np.ndarray
    disturbance_energy_cost: np.ndarray    # Cumulative disturbance cost
    disturbance_events: np.ndarray
    cumulative_energy_deficit: np.ndarray
```

##### Factory

```python
def create_energy_module(
    params: SimulationParameters, mode: EnergyMode = EnergyMode.DEPONS
) -> EnergyModule:
    """Create the energy module for a mode (EnergyMode.HYBRID also gives the JASMINE module).
    Always returns a module (never None). build_simulation passes the mode from
    params.get_effective_energy_mode()."""
```

---

### 4. Reference Memory Module

**Location:** `src/cenop/behavior/ref_mem.py`

Implements the DEPONS 3.2 reference memory system with precomputed decay tables.

#### Classes

##### `RefMemWorkspace`

```python
@dataclass
class RefMemWorkspace:
    """
    Reusable workspace for compute_ve_total and compute_attraction_vector.

    Pre-allocates intermediate arrays to avoid per-tick allocations.
    Pass as the optional workspace parameter of compute_ve_total() and compute_attraction_vector().
    """

    @classmethod
    def create(cls, n_agents: int, mem_size: int) -> "RefMemWorkspace":
        """Allocate workspace arrays of shape (n_agents, mem_size)."""
```

#### Functions

```python
def get_ref_mem_strength_table(r_r: float = 0.03, size: int = 120) -> np.ndarray:
    """refMemStrength table (rR decay)."""

def get_work_mem_strength_table(r_s: float = 0.03, size: int = 120) -> np.ndarray:
    """workMemStrength table (rS decay)."""
# Both: logistic decay s[i+1] = s[i] - r * s[i] * (1 - s[i]) from s[0] = 0.999,
# rounded to 4 decimals (as DEPONS RefMem.calcArray); cached and read-only.

def compute_ve_total(
    stored_util: np.ndarray,
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    work_mem_table: np.ndarray,
    mask: np.ndarray,
    workspace: RefMemWorkspace | None = None,
) -> np.ndarray:
    """Compute expected food value (veTotal) for all agents (Numba kernel, NumPy fallback)."""

def compute_attraction_vector(
    stored_util: np.ndarray,
    pos_history_x: np.ndarray,
    pos_history_y: np.ndarray,
    mem_ptr: np.ndarray,
    mem_count: np.ndarray,
    current_x: np.ndarray,
    current_y: np.ndarray,
    ref_mem_table: np.ndarray,
    mask: np.ndarray,
    world_width: int = 0,
    world_height: int = 0,
    workspace: RefMemWorkspace | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attraction vector (vt_x, vt_y) towards remembered food-rich areas."""
```

---

### 5. Hybrid Behavioral FSM (JASMINE)

**Location:** `src/cenop/behavior/hybrid_fsm.py` (states and contexts in `src/cenop/behavior/states.py`)

Finite state machine that labels each animal with a behaviour state. The states do not steer movement: they reach `Simulation.agents_df["behavioral_state"]` and, with `energy_mode="JASMINE"`, set the activity cost (`JASMINEEnergyModule.ACTIVITY_MULTIPLIERS`). `get_movement_mode()`, `get_speed_multiplier()` and `get_energy_cost_multiplier()` exist but are not called by the model. The FSM runs only when a simulation is built with `build_simulation` (the app always does; scripts and the batch runner that use `Simulation(params)` do not).

#### Classes

##### `HybridBehaviorFSM`

```python
class HybridBehaviorFSM:
    """
    Hybrid Finite State Machine for porpoise behavior.

    DEPONS mode:
        any -> DISTURBED:        deterrence > 0.01
        DISTURBED -> FORAGING:   no deterrence for more than disturbance_recovery_ticks (48)
        any -> DISPERSING:       the model is dispersing the animal (DISTURBED keeps priority)
        DISPERSING -> FORAGING:  the model ended the dispersal
    JASMINE mode adds:
        FORAGING <-> TRAVELING:  speed above / below speed_threshold
        FORAGING/TRAVELING -> RESTING, RESTING -> FORAGING: energy thresholds
    """

    def __init__(
        self,
        params: SimulationParameters,
        time_mode: TimeMode = TimeMode.DEPONS,
        fsm_mode: FSMMode | None = None,
    ):
        """FSM mode: fsm_mode if given, else DEPONS for TimeMode.DEPONS and JASMINE otherwise."""

    def update_states(
        self,
        state_vector: BehaviorStateVector,
        context: BehaviorContext,
        mask: np.ndarray,
    ) -> np.ndarray:
        """Update states in place; returns a boolean array of agents that changed state."""


def create_behavior_fsm(params: SimulationParameters, time_mode: TimeMode = TimeMode.DEPONS) -> HybridBehaviorFSM:
    """Factory used by build_simulation, which passes the effective FSM mode (fsm_mode override, else simulation_mode)."""
```

##### `BehaviorState`

```python
class BehaviorState(Enum):  # src/cenop/behavior/states.py
    FORAGING = 1
    TRAVELING = 2
    RESTING = 3
    DISPERSING = 4
    DISTURBED = 5
```

##### `BehaviorContext`

```python
@dataclass
class BehaviorContext:
    """Context for behavioral state transitions (arrays of shape (n,))."""

    deterrence_magnitude: np.ndarray   # Current deterrence strength
    time_since_disturbance: np.ndarray # Ticks since last disturbance
    current_energy: np.ndarray         # Energy / 20 (0-1)
    current_speed: np.ndarray          # Step length this tick
    is_dispersing: np.ndarray          # Dispersing in the model (the state follows it)
    energy_threshold_low: float = 0.3
    energy_threshold_high: float = 0.7
    speed_threshold: float = 2.0
    recovery_ticks: int = 48
```

`BehaviorStateVector.create(count)` holds the per-agent `state`, `state_duration` and `previous_state` arrays.

---

### 6. Dynamic Energy Budget (JASMINE)

**Location:** `src/cenop/physiology/energy_budget.py`

The JASMINE energy budget is uncalibrated and runs only with `energy_mode="JASMINE"` (see the known issues in [CHANGELOG.md](../CHANGELOG.md)). It is created by `create_energy_module(params, EnergyMode.JASMINE)` (section 3).

#### Classes

##### `JASMINEEnergyModule`

```python
class JASMINEEnergyModule(EnergyModule):
    """
    JASMINE Dynamic Energy Budget model.

    Features:
        - Food intake scaled by body mass, body condition and food quality
        - Body mass-dependent BMR (Kleiber scaling; params.jasmine_body_mass_scaling)
        - Activity cost: cost of transport times the behaviour-state multiplier
        - Thermoregulation cost outside the thermoneutral zone (params.jasmine_thermal_model)
        - Lactation and pregnancy costs (0.4 and 0.2 x BMR)
        - Disturbance energy cost (0.1 x deterrence + speed penalty while disturbed),
          scaled by params.jasmine_disturbance_cost_mult
        - apply_result() sets body condition = energy / 20 (clipped to 0.1-1) and
          compute_survival_probability() maps it to the DEPONS starvation formula; the
          population calls neither, so in a simulation body condition stays at 0.5
    """

    # Bioenergetics constants
    BODY_MASS_ADULT = 50.0       # kg
    BODY_MASS_CALF = 15.0        # kg
    BMR_COEFFICIENT = 3.4        # W/kg^0.75
    BMR_EXPONENT = 0.75
    COT_COEFFICIENT = 0.0001
    ACTIVITY_MULTIPLIERS = {1: 1.0, 2: 1.5, 3: 0.6, 4: 1.2, 5: 2.0}  # by BehaviorState value
    THERMONEUTRAL_LOWER = 5.0    # °C
    THERMONEUTRAL_UPPER = 20.0   # °C
    THERMAL_CONDUCTANCE = 0.02   # W/kg/°C
    DISTURBANCE_BASE_COST = 0.1
    DISTURBANCE_SPEED_MULT = 2.0

    def get_fitness_metrics(self, state: EnergyState, mask: np.ndarray) -> dict:
        """mean_body_condition, mean_fat_reserve, total_disturbance_cost,
        mean_disturbance_events, agents_in_deficit, mean_energy_deficit."""
```

In a simulation the water temperature is 10 °C (no landscape supplies a temperature layer), inside the thermoneutral zone, so the thermoregulation cost is zero.

---

### 7. Disturbance Memory (JASMINE)

**Location:** `src/cenop/behavior/disturbance_memory.py`

Spatial memory system for learned avoidance of disturbance zones. Created by `create_memory_module(params, mode=MemoryMode.DEPONS)`: `MemoryMode.DEPONS` gives `DEPONSMemoryModule` (no memory, zero avoidance), `MemoryMode.JASMINE` gives `JASMINEMemoryModule`. The memory mode follows `simulation_mode` unless `memory_mode` is set. There is no habituation.

#### Classes

##### `JASMINEMemoryModule`

```python
class JASMINEMemoryModule(DisturbanceMemoryModule):
    """
    Spatial memory for disturbance zones.

    - Each animal keeps a sparse grid of memory cells of 5 x 5 grid cells (2 km on 400 m cells)
    - A disturbed animal (deterrence > 0.01) stores min(deterrence strength, 1) at its own
      cell, keeping the larger of the old and new value
    - Every tick memories decay by params.memory_decay_rate (default 0.016: a full memory
      falls below the 0.1 avoidance threshold in about 3 days); values below 0.01 are dropped
    - Avoidance: memories of at least 0.1 within params.avoidance_radius memory cells
      (default 2.5, i.e. 5 km) push the animal away, with strength up to 0.8
    """

    def __init__(self, params: SimulationParameters): ...

    def record_disturbance(
        self, state: DisturbanceMemoryState, context: DisturbanceMemoryContext, mask: np.ndarray
    ) -> None:
        """Record disturbances in memory."""

    def decay_memory(
        self, state: DisturbanceMemoryState, mask: np.ndarray, ticks_elapsed: int = 1
    ) -> None:
        """Apply exponential memory decay."""

    def compute_avoidance(
        self,
        state: DisturbanceMemoryState,
        agent_x: np.ndarray,
        agent_y: np.ndarray,
        mask: np.ndarray,
    ) -> AvoidanceResult:
        """Avoidance vectors (avoidance_dx, avoidance_dy, avoidance_strength, cells_avoided)."""

    def get_avoidance_map(self, state: DisturbanceMemoryState, agent_idx: int) -> dict:
        """{(x, y): strength} of one agent's remembered cells, for visualization."""
```

The avoidance computed at the end of a tick acts on the next tick's movement. On the DEPONS walk it is added to the heading as `avoidance_strength x crwContrib` (the random walk's own weight, as for the social pull); on the JASMINE walk it is added to the deterrence vector.

`DisturbanceMemoryState.create(count)` holds the per-agent memory grids and statistics.

---

### Dispersal Module

**Location:** `src/cenop/agents/population.py` (`off`, `PSM-Type2`, `Undirected`) and `src/cenop/agents/dispersal_idw.py` (`InnerDanishWaters`)

Dispersal moves porpoises away from areas where their energy has declined. The mode is `SimulationParameters.dispersal`, one of `DISPERSAL_MODES`:
- `off` — No dispersal
- `PSM-Type2` — SSLogis-damped dispersal turn, target from spatial memory or else random, stops on path length (default)
- `Undirected` — random target, full +/- `psm_angle` turn, stops on straight-line distance
- `InnerDanishWaters` — block-based navigation (60 blocks of 40 km, 2 phases); Kattegat landscape only

Any other value raises a `ValueError`. `src/cenop/behavior/dispersal.py` holds per-animal translations of eight DEPONS dispersal classes (`DispersalType`, including PSM-Type1 and the PSM-Type3 variants); the simulation does not use them.

---

### 8. Sound & Deterrence Module

**Location:** `src/cenop/behavior/sound.py`

Noise propagation helpers (`calculate_transmission_loss`, `calculate_received_level`, `combine_rls`) and the ship deterrence model. Turbine and ship deterrence in the simulation are computed by `TurbineManager` (`src/cenop/agents/turbine.py`) and `ShipManager` (`src/cenop/agents/ship.py`) from the `SimulationParameters` values.

#### Classes

##### `ShipDeterrenceModel`

```python
class ShipDeterrenceModel:
    """
    Ship deterrence using standardised logistic regression (DEPONS 3.2).

    Inputs are standardised (x - mean) / sd before the regression.
    Constants in STD_PROB_DAY, STD_PROB_NIGHT (probability) and STD_MAG_DAY, STD_MAG_NIGHT (magnitude).
    Each Ship holds one as ship.deterrence_model.
    """

    def calculate_deterrence_probability(self, spl: float, distance_km: float, is_day: bool = True):
        """Probability of a deterrence response to ship noise."""

    def calculate_deterrence_magnitude(self, spl: float, distance_km: float, is_day: bool = True):
        """Deterrence magnitude."""

    def deterrence_components(self, rl, dist_m, grid_dx, grid_dy, is_day, u_draw, tships):
        """Vectorized per-porpoise deterrence for one ship: returns (vx, vy, prob, mag, react)."""
```

##### `Hydrophone`

Passive acoustic monitoring station (translates `Hydrophone.java`). The simulation does not create hydrophones.

```python
from cenop.behavior.sound import Hydrophone

h = Hydrophone(name="H1", x=100.0, y=200.0)
h.receive_sound_level("Ship-A", ship_utm_x=500000, ship_utm_y=6000000,
                      source_level=170.0, received_level=145.0)
print(h.received_level)  # 145.0
h.reset()  # Call at end of each tick
```

##### `SoundPropagationParams`

```python
@dataclass
class SoundPropagationParams:
    """Standalone defaults; the simulation does not read this class. It uses
    SimulationParameters (alpha_hat=0.00027, beta_hat=14.72, deter_threshold=152.0,
    deter_max_distance=1000.0)."""
    alpha_hat: float = 0.0                 # Absorption coefficient
    beta_hat: float = 20.0                 # Spreading loss factor
    response_threshold: float = 152.9      # Minimum received level (dB re 1 µPa)
    max_deter_distance: float = 50.0       # km
    min_deter_distance_ships: float = 0.1  # km
```

---

### 9. Weston Flux Transmission Loss

**Location:** `src/cenop/behavior/weston_flux.py`

Physics-based transmission loss model ported from DEPONS `WestonFlux.java` (Numba-compiled).

```python
def weston_flux_tl(
    distance: float,
    depth: float,
    grain_size: float,
    temperature: float,
    salinity: float,
    frequency: float = FREQUENCY,  # 10**1.2 * 1000, about 15.85 kHz (DEPONS 3.2)
    ph: float = 8.0,
    c_s: float = 1700.0,
) -> float:
    """
    Calculate transmission loss using Weston flux theory.

    Uses sediment grain size to derive sound speed ratio, density ratio and
    attenuation coefficient. Returns 0.0 when distance or depth is not positive.

    Args:
        distance: Distance from source (m)
        depth: Water depth at source (m)
        grain_size: Sediment grain size on phi scale
        temperature: Water temperature (°C)
        salinity: Salinity (PSU)

    Returns:
        Transmission loss in dB
    """
```

Ships use it by default (`weston_flux_percell=True`) with depth, sediment and salinity at the ship's position and `weston_flux_default_temperature` (10 °C). Where depth or sediment is missing, `src/cenop/agents/ship.py` falls back to `beta_hat * log10(r) + alpha_hat * r`.

---

### 10. JOMOPANS Ship Source Levels

**Location:** `src/cenop/behavior/jomopans_spl.py` (`VesselClass` in `src/cenop/agents/ship.py`)

Calibrated ship source levels from the JOMOPANS project.

```python
class VesselClass(Enum):
    """15 vessel classes (13 JOMOPANS + 2 aliases: CARGO and CHEMICAL_TANKER)."""
    BULKER = "bulker"
    CARGO = "cargo"
    # ... CONTAINER, CRUISE, DREDGER, FISHING, GOVERNMENT, NAVAL, PASSENGER,
    #     RECREATIONAL, TANKER, TUG, VEHICLE_CARRIER, OTHER

def jomopans_spl(
    vessel_class: VesselClass, speed_knots: float, length_m: float, band: int = 12
) -> float:
    """Source level in dB re 1 µPa @ 1 m for a vessel class, speed and length
    (decidecade band 12, about 15.85 kHz). Returns 0.0 at speed 0."""
```

---

### 11. Simulation Module

**Location:** `src/cenop/core/simulation.py`

Main simulation orchestration.

#### Classes

##### `Simulation`

```python
class Simulation:
    """
    Main simulation controller.

    Manages landscape, population, turbines, ships, and the tick loop.
    """

    def __init__(
        self,
        params: SimulationParameters,
        cell_data: CellData | None = None,
        seed: int | None = None,
        time_manager: TimeManager | None = None,
        time_mode: TimeMode = TimeMode.DEPONS,
        movement_module: MovementModule | None = None,
        behavior_fsm: HybridBehaviorFSM | None = None,
        energy_module: EnergyModule | None = None,
        memory_module: DisturbanceMemoryModule | None = None,
    ):
        """Initialize simulation. Initializes at once when cell_data is given or
        landscape == "Homogeneous"; otherwise on initialize() or the first step().
        seed (default: params.random_seed, else 42) seeds NumPy's global generator, the
        time manager and the ship reaction draws; the population's own generator is
        seeded by params.random_seed, so set random_seed for a reproducible run."""

    def initialize(self) -> None:
        """Load landscape (and its suntimes.csv), create population, set up turbines and ships."""

    def step(self) -> None:
        """Execute one 30-minute tick: deterrence, population step, time, daily/monthly/yearly tasks."""

    def run(self, progress: bool = True) -> None:
        """Run to the end (sim.max_ticks = sim_years * 360 * 48)."""

    def get_porpoise_positions(self) -> np.ndarray:
        """
        Get current positions and state of all porpoises.

        Returns:
            Array (N, 7): [original_index, x, y, energy, heading, age, is_dispersing]
        """
```

Also: `state` (`SimulationState`: tick, day, month, year, population, births, deaths), `population_manager` (the `PorpoisePopulation`), `population_size`, `agents_df`, `get_population_history()`, `get_statistics()`, `total_births`, `total_deaths`.

##### `build_simulation`

**Location:** `src/cenop/server/simulation_controller.py` (importing it needs `shiny`)

```python
def build_simulation(params: SimulationParameters, landscape: CellData | None = None) -> Simulation:
    """A Simulation with the subsystem modules that params selects, as the app builds it.
    The landscape is loaded from params.landscape unless one is given. Returns an
    initialized simulation."""
```

##### `SimulationRunner`

**Location:** `src/cenop/server/simulation_controller.py`

```python
class SimulationRunner:
    """
    Wraps Simulation for the Shiny server with progress tracking.

    Attributes:
        sim (Simulation): The underlying simulation
        progress_percent (float): Current progress (0-100)
        should_update_map (bool): Whether map needs refresh (once per simulated day)
        total_births, total_deaths (int): Cumulative counts
    """

    def __init__(self, simulation: Simulation): ...

    def step_ticks(self) -> dict:
        """Advance ticks_per_update ticks (default 48) and return a history entry."""
```

---

### 12. Batch Runner

**Location:** `src/cenop/core/batch_runner.py`

Runs multiple simulations for sensitivity analysis. Each run is a plain `Simulation(params, seed=seed)`, so batch runs use no JASMINE modules (`simulation_mode="JASMINE"` has no effect there).

#### Classes

##### `BatchConfiguration` and `BatchRunner`

```python
@dataclass
class BatchConfiguration:
    base_params: dict[str, Any] = field(default_factory=dict)        # SimulationParameters fields
    variations: dict[str, list[Any]] = field(default_factory=dict)   # every combination is run
    replicates: int = 1
    seeds: list[int] | None = None          # one per replicate (random if None)
    output_dir: str = "output/batch"
    parallel: bool = False
    max_workers: int = 4
    progress_callback: Callable[[int, int, str], None] | None = None


class BatchRunner:
    """
    Runs batch simulations with parameter variations.
    """

    def __init__(self, config: BatchConfiguration):
        """Initialize batch runner (creates config.output_dir)."""

    def run(self, progress: bool = True) -> list[BatchResult]:
        """Run every combination x replicates. BatchResult holds run_id, parameters,
        final_population, initial_population, total_births, total_deaths, total_ticks,
        execution_time_seconds, random_seed, population_change_pct, annual_growth_rate."""

    def export_results(self, results=None, filename: str = "batch_results.csv") -> Path: ...
    def export_summary(self, results=None, filename: str = "batch_summary.json") -> Path: ...
    def calculate_sensitivity(self, param_name: str, metric: str = "final_population") -> dict: ...
```

The replicate seed is passed as `Simulation(seed=...)` (see section 11): it reaches NumPy's global generator, the time manager and the ship reaction draws, but not the population's own generator, which is seeded by `random_seed`. Without ships, replicates with a fixed `random_seed` in `base_params` are identical, and replicates with `random_seed` unset differ but are not reproducible. Convenience functions: `run_sensitivity_analysis(base_params, param_name, param_values, replicates=5, output_dir="output/sensitivity")` and `run_scenario_comparison(scenarios, replicates=5, output_dir="output/scenarios")`.

---

### 13. Output Writer

**Location:** `src/cenop/core/output_writer.py`

DEPONS-compatible file output generation (tab-separated).

#### Classes

##### `OutputWriter`

```python
class OutputWriter:
    """
    Writes DEPONS-compatible output files.

    Output files:
        - Population.txt: Population counts over time
        - PorpoiseStatistics.txt: Detailed individual data
        - Mortality.txt: Mortality events
        - Dispersal.txt: Dispersal events
        - Energy.txt: Energy statistics
    """

    def __init__(self, config: OutputConfig | None = None):
        """Initialize output writer (OutputConfig: output_dir, which files, intervals in ticks, run_id)."""

    def open(self) -> None:
        """Open all output files and write headers (also via `with OutputWriter(config) as writer:`)."""

    def record_tick(self, sim: Simulation) -> None:
        """Record the current tick: detects deaths and new dispersals, writes the interval outputs."""

    def close(self) -> None:
        """Close all output files."""
```

`run_with_output(sim, output_dir="output", progress=True)` runs a simulation to the end with all outputs.

---

## Configuration

### SimulationParameters

**Location:** `src/cenop/parameters/simulation_params.py`

```python
@dataclass
class SimulationParameters:
    """Main simulation configuration (selected fields)."""

    # Population & Time
    porpoise_count: int = 10000
    max_population: int = 0          # 0 = no limit
    sim_years: int = 50
    landscape: str = "Lithuania"     # e.g. "Homogeneous", "Kattegat", "NorthSea"
    random_seed: int | None = None   # None = unseeded, runs differ

    # Simulation Mode
    simulation_mode: str = "DEPONS"  # "DEPONS" or "JASMINE"
    movement_mode: str | None = None # None = DEPONS in both modes; "JASMINE" = simplified walk
    energy_mode: str | None = None   # None = DEPONS in both modes; "JASMINE" = JASMINE budget
    time_mode: str | None = None     # None = follow simulation_mode
    memory_mode: str | None = None   # None = follow simulation_mode
    fsm_mode: str | None = None      # Behaviour-state rules (None = follow simulation_mode)

    # Disturbance
    turbines: str = "off"            # or a wind-farm scenario in data/wind-farms
    ships_enabled: bool = False

    # Dispersal
    dispersal: str = "PSM-Type2"     # "off", "PSM-Type2", "Undirected", "InnerDanishWaters"
    t_disp: int = 3                  # Days of declining energy before dispersal
    psm_dist_mean: float = 350.0     # Preferred dispersal distance (km)
    psm_angle: float = 40.0          # Max dispersal turn (degrees)
    mean_disp_dist: float = 2.0      # Dispersal step distance (km)

    # CRW Parameters (DEPONS 3.2)
    inertia_const: float = 0.001     # k: directional persistence
    corr_logmov_length: float = 0.35 # a0: step length autocorrelation
    corr_angle_base: float = -0.024  # b0: turning angle autocorrelation
    movement_salinity: str = "auto"  # "auto" (fixed 34.07 on Kattegat, Lithuania,
                                     # CentralBaltic), "map" or "fixed"

    # Energy & Memory (DEPONS 3.2)
    r_s: float = 0.03                # Working memory (veTotal) decay rate
    r_r: float = 0.03                # Reference memory decay rate
    food_growth_rate: float = 0.1    # rU: food replenishment rate
    ref_mem_size: int = 120          # Reference memory circular buffer size

    # Mortality (DEPONS 3.2 calibration)
    m_mort_prob_const: float = 1.0   # Starvation formula constant
    x_survival_const: float = 0.4    # Starvation formula exponent
    bycatch_prob: float = 0.0        # Annual bycatch probability

    # Deterrence & sound propagation (DEPONS 3.2)
    deter_coeff: float = 0.012       # Deterrence coefficient
    deter_threshold: float = 152.0   # Minimum received level (dB)
    deter_max_distance: float = 1000.0  # Max deterrence distance (km); ships capped at 10 km
    alpha_hat: float = 0.00027       # Absorption coefficient
    beta_hat: float = 14.72          # Spreading loss factor
    weston_flux_percell: bool = True # WestonFlux TL for ships

    # Social communication (JASMINE, opt-in)
    communication_enabled: bool = False
    social_weight: float = 0.3       # Fraction of the random walk's pull (uncalibrated)

    # JASMINE Memory (memory_mode JASMINE)
    memory_decay_rate: float = 0.016  # a full memory fades in about 3 days
    avoidance_radius: float = 2.5     # memory cells of 2 km: 5 km

    # JASMINE energy (energy_mode JASMINE)
    jasmine_body_mass_scaling: bool = True
    jasmine_thermal_model: bool = True
    jasmine_disturbance_cost_mult: float = 1.0

    use_jax: bool = False            # True raises ValueError (JAX backend removed)
```

`get_effective_movement_mode()` and `get_effective_energy_mode()` return the override or `"DEPONS"`; `get_effective_time_mode()`, `get_effective_fsm_mode()` and `get_effective_memory_mode()` return the override or `simulation_mode`. Invalid `dispersal` or `movement_salinity` values, and `InnerDanishWaters` on a landscape other than Kattegat, raise a `ValueError`.

Accepted but not read by the model: `jasmine_mass_kg`, `jasmine_drag_coeff`, `jasmine_max_thrust`, `jasmine_current_weight`, `jasmine_bmr_scale`, `jasmine_activity_cost`, `jasmine_disturbance_cost`, `jasmine_avoidance_strength`, the copies `jasmine_memory_decay_rate` and `jasmine_avoidance_radius` (the app copies their inputs to `memory_decay_rate` and `avoidance_radius`), and `mortality_juvenile`, `mortality_adult`, `mortality_elderly`.

---

## Usage Examples

### Basic Simulation (DEPONS Mode)

```python
from cenop import Simulation, SimulationParameters

# Create configuration
params = SimulationParameters(
    porpoise_count=500,
    sim_years=5,
    landscape="Kattegat",
    random_seed=42,
)

# Initialize and run
sim = Simulation(params)
sim.initialize()

for tick in range(sim.max_ticks):
    sim.step()
    if sim.state.tick % 48 == 0:  # Daily output
        print(f"Day {sim.state.day}: Pop={sim.population_size}")
```

### JASMINE Mode Simulation

```python
from cenop import SimulationParameters
from cenop.behavior.states import BehaviorState
from cenop.server.simulation_controller import build_simulation

# Create JASMINE configuration
params = SimulationParameters(
    porpoise_count=1000,
    sim_years=5,
    landscape="Kattegat",
    ships_enabled=True,
    simulation_mode="JASMINE",  # DEPONS movement and energy + learned avoidance + behaviour states
    memory_decay_rate=0.008,    # Slower memory decay than the default 0.016
    # energy_mode="JASMINE",    # optional: uncalibrated JASMINE energy budget
    # movement_mode="JASMINE",  # optional: simplified walk instead of the DEPONS CRW
)

# Create the simulation with the JASMINE modules (already initialized)
sim = build_simulation(params)

for tick in range(sim.max_ticks):
    sim.step()

    if sim.state.tick % 48 == 0:  # Daily
        agents = sim.agents_df
        states = agents["behavioral_state"].value_counts()
        print(f"Day {sim.state.day}:")
        print(f"  Population: {sim.population_size}")
        print(f"  Mean energy: {sim.population_manager.get_energy_stats()['mean']:.2f}")
        print(f"  Deterred now: {int(agents['is_disturbed'].sum())}")
        print(f"  DISTURBED state: {states.get(BehaviorState.DISTURBED.value, 0)}")
```

### Batch Analysis

```python
from cenop.core.batch_runner import BatchConfiguration, BatchRunner

config = BatchConfiguration(
    base_params={"sim_years": 1, "landscape": "Homogeneous", "porpoise_count": 200},
    variations={"bycatch_prob": [0.0, 0.05, 0.1]},  # Every combination is run
    replicates=3,
    output_dir="output/batch",
)

runner = BatchRunner(config)
results = runner.run(progress=False)
runner.export_results(results)  # output/batch/batch_results.csv

# Analyze results
for result in results:
    print(f"Params: {result.parameters['bycatch_prob']}")
    print(f"Final pop: {result.final_population} ({result.population_change_pct:+.1f}%)")
print(runner.calculate_sensitivity("bycatch_prob"))
```

---

## Time Manager

**Location:** `src/cenop/core/time_manager.py`

### Suntimes

Seasonal sunrise/sunset from CSV (translates `Suntimes.java`). Falls back to fixed 6am-6pm when no CSV provided. `Simulation.initialize()` loads `data/<landscape>/suntimes.csv` when the landscape has one (only Kattegat does).

```python
from cenop.core.time_manager import TimeManager, Suntimes

tm = TimeManager(suntimes_path="data/Kattegat/suntimes.csv")
print(tm.is_daytime)  # Uses per-DOY sunrise/sunset from CSV
```

---

## Version History

Summarised from [CHANGELOG.md](../CHANGELOG.md), which is authoritative.

### Unreleased (after 2.3.0)
- `build_simulation(params, landscape=None)` builds the simulation with the modules the parameters select; `Simulation(params)` alone runs DEPONS without modules
- JASMINE mode runs DEPONS movement and the DEPONS energy budget by default; the JASMINE movement and energy budget only with `movement_mode="JASMINE"` / `energy_mode="JASMINE"`
- Learned avoidance calibrated to observed recovery after pile driving (stores a response capped at 1 at the animal's own cell, `memory_decay_rate` 0.016, `avoidance_radius` 2.5 memory cells = 5 km; on the DEPONS walk the pull is scaled by crwContrib)
- Removed: `habituation_enabled` (habituation was never implemented) and the JAX backend (`use_jax=True` raises `ValueError`)
- New parameter `movement_salinity` (fixed movement salinity on the Baltic landscapes, as DEPONS does on Kattegat)
- Behaviour states follow the model's dispersal; social pull scaled to the random walk; land-avoidance tie and grid-edge fixes; ship deterrence about 2.6 times faster with identical results
- CentralBaltic rebuilt from public data; Lithuania real salinity, sediment, food patches and ship traffic
- App: dark mode, shareable settings link, map redraw after a reconnect

### v2.3.0 (2026-11-20)
- First comparison against runs of DEPONS 3.2: reference harness (`validation/depons_java/`) and A2 cross-validation (`docs/validation/A2-parity-report.md`)
- Dispersal modes `off`, `Undirected` and `InnerDanishWaters` (Kattegat only); the dispersal setting had been ignored
- Starting population: females only, pregnancy at setup, daily step at tick 1, arrays grow for weaned calves (`max_population`)
- PSM-Type2 dispersal and land avoidance follow DEPONS 3.2; phantom food in the eat kernels fixed
- `data/Kattegat` is the DEPONS v3.2 landscape; landscapes not on 400 m cells are resampled at load
- Ship movement along routes and ship sound (WestonFlux at the ship's position, on by default) as in DEPONS
- `communication_enabled` defaults to `False`; JAX backend deprecated; UI aligned with Shiny 1.9 and shiny_deckgl 1.16; faster tick with identical results

### v2.2.0 (2026-07-10)
- Deep-review fix cycle: turbine deterrence vector from grid displacement (was about 400 times too large), web-app movement routed through the validated `crw_core`, double daily food regrowth fixed, recycled slots fully reset, seeded PSM RNG
- DEPONS parity: BMR-only energy and deterministic turbine deterrence by default; true per-tick birth/death and per-cause mortality counters
- Cython backend repaired; CI (lint + fast suite, nightly slow tier)
- 810 fast tests + 11 slow validation tests

### v2.1.0 (2026-03-14)
- DEPONS 3.2 full sync + performance optimization + new features
  - DEPONS 3.2 sync in five phases (parameters and energy, reproduction and mortality, food, movement and memory, deterrence and dispersal)
  - Pregnancy FSM (immature → pregnant → ready-to-mate) with daily mortality schedule
  - Logistic food regrowth with 48-iteration compounding and MaxEnt carrying capacity
  - Reference memory (120-entry circular buffer) with vectorized vt and veTotal
  - CRW rejection sampling (up to 200 retries for angle and step length)
  - PSM-Type2 dispersal with SSLogis heading, energy-based stop, deterrence deactivation
  - Ship deterrence with JOMOPANS 13-class source levels and Weston flux TL
  - Deterrence vectors use raw displacement (not unit vectors), matching DEPONS Java
  - 10 Numba JIT kernels, 6 with prange parallelism (3.27 ms/tick, 1.68x speedup)
  - WestonFlux @njit compiled, fused ref_mem kernels, food regrowth prange kernel
  - Pre-allocated buffers, RefMemWorkspace, land avoidance for blocked agents only
  - Per-cell WestonFlux TL option (reads sediment/depth/salinity from landscape grids)
  - Porpoise trace trails using TripsLayer, coloured by porpoise state
  - Skip visualization toggle (25.7% end-to-end speedup)
  - Unified bitmap grid rendering (server-side PNG, one pixel per cell)
  - 515+ automated tests across 24 test files

### v2.0.0 (2025)
- CENOP-JASMINE merge
  - Added JASMINE simulation mode
  - Hybrid behavioral FSM (5 states)
  - Dynamic Energy Budget with body mass scaling
  - Disturbance memory with learned avoidance (habituation was listed but never implemented;
    the setting was removed after 2.3.0)
  - Physics-based movement module (its integrator was never wired into the population; JASMINE movement is a simplified walk)
  - Dual-mode configuration (DEPONS/JASMINE)
  - 6-tab Shiny web interface with DeckGL map

### v0.1.0 (2024)
- Initial release with core simulation functionality
  - Core modules: Landscape, Population, Energetics, PSM
  - DEPONS-compatible output
  - Shiny web interface
