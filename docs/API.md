# CENOP-JASMINE API Documentation

## Overview

CENOP-JASMINE (Cetacean Noise Operations Planner with JASMINE Extensions) is a Python Shiny application for simulating harbor porpoise population dynamics in response to wind farm construction noise. This document describes the core API for developers and researchers.

The API supports two simulation modes:
- **DEPONS Mode**: Regulatory-compatible empirical models aligned with DEPONS 3.2 (default)
- **JASMINE Mode**: Research-grade physics-based and bioenergetics models

---

## DEPONS 3.2 Algorithmic Parity

CENOP's DEPONS mode achieves **100% functional equivalence** with DEPONS 3.2 Java:

- **136 features** verified MATCH (identical formulas, parameters, output)
- **13 intentional divergences** (vectorization, proportional food sharing, explicit FSM)
- **4 CENOP extensions** (JASMINE states, DEB energy, disturbance memory, age-stratified mortality)
- **0 gaps remaining** (all 12 original gaps closed)

Full analysis: [DEPONS-CENOP Parity Analysis](DEPONS-CENOP-PARITY-ANALYSIS.md)

### Performance vs Java (N=500, 200×200 grid)

| System | ms/tick | Ratio |
|--------|---------|-------|
| Java DEPONS 3.2 | 0.80 | 1.0× |
| Python DEPONS mode | 1.05 | 1.3× |
| Python JASMINE mode | 1.88 | 2.4× |

---

## Core Modules

### 1. Landscape Module

**Location:** `src/cenop/landscape/cell_data.py`

The landscape module manages the spatial environment including bathymetry, food availability, salinity, and MaxEnt prey distribution.

#### Classes

##### `CellData`

```python
class CellData:
    """
    Manages landscape grid data loaded from ASC files.

    Attributes:
        metadata (LandscapeMetadata): Grid geometry (ncols, nrows, xllcorner, yllcorner, cellsize)
        _depth (np.ndarray): Bathymetry values (metres below sea level)
        _food_prob (np.ndarray): Food probability / carrying capacity (0-1)
        _food_level (np.ndarray): Current food level per cell (dynamic, depleted by foraging)
        _salinity (list[np.ndarray]): 12 monthly salinity fields
        _maxent (list[np.ndarray]): 12 monthly MaxEnt prey distribution layers
    """

    def __init__(self, landscape_name: str):
        """Initialize CellData for a named landscape (e.g., 'Kattegat', 'NorthSea')."""

    def load(self):
        """Load all ASC grid files from the landscape data directory."""

    def get_depth(self, row: int, col: int) -> float:
        """Get water depth at grid cell."""

    def get_food(self, row: int, col: int) -> float:
        """Get current food level at grid cell."""

    def eat_food(self, row: int, col: int, amount: float) -> float:
        """Consume food from cell, returning amount actually eaten."""

    def regrow_food(self, tick: int):
        """Apply logistic food regrowth (48 iterations per day, DEPONS 3.2)."""

    def get_salinity(self, row: int, col: int, month: int) -> float:
        """Get salinity value for a cell in a given month."""
```

---

### 2. Population Module

**Location:** `src/cenop/agents/population.py`

Structure-of-Arrays (SoA) vectorized implementation of porpoise population dynamics.

#### Classes

##### `PorpoisePopulation`

```python
class PorpoisePopulation:
    """
    SoA vectorized porpoise population management.

    Core arrays (all shape (max_agents,)):
        x, y (np.float64): Grid coordinates (column, row)
        heading (np.float64): Current heading in degrees [0, 360)
        energy (np.float64): Energy reserves (0-20 scale)
        age (np.float64): Age in years
        sex (np.int8): 0=male, 1=female
        pregnancy_status (np.int8): 0=immature, 1=pregnant, 2=ready-to-mate
        is_dispersing (np.bool_): Currently in PSM-Type2 dispersal
        active_mask (np.bool_): True for living porpoises

    Reference memory arrays:
        ref_mem_x, ref_mem_y (np.float64): Circular buffer of remembered positions (max_agents × 120)
        ref_mem_food (np.float64): Food utility at remembered positions
        ref_mem_count (np.int32): Number of entries in circular buffer
    """

    def __init__(self, params: SimulationParameters, cell_data: CellData):
        """Initialize population with SoA arrays."""

    def step(self, tick: int):
        """Execute one simulation tick for all active porpoises."""

**New in v2.2:** `step()` now caches `_active_idx` per tick, uses fused Numba kernels for heading+position+reflect, and pre-allocates all per-tick buffers for zero-allocation steady state.

    def _compute_crw_heading(self, tick: int):
        """Compute CRW heading with rejection sampling and reference memory attraction."""

    def _apply_dispersal_heading(self, mask: np.ndarray):
        """Apply SSLogis dispersal heading for dispersing porpoises."""

    def _check_mortality(self, tick: int):
        """Daily mortality check: starvation, bycatch, max-age (tick % 48 == 0)."""

    def _update_pregnancy(self, tick: int):
        """Daily pregnancy FSM transitions (immature → pregnant → ready-to-mate)."""

    def _eat_food_vectorized(self):
        """Vectorized food consumption using eat_food_kernel."""

    def _apply_deterrence(self, deterrence_vectors: tuple[np.ndarray, np.ndarray]):
        """Apply deterrence displacement vectors (raw displacement × strength × coeff)."""
```

---

### 3. Energy Budget Module

**Location:** `src/cenop/physiology/energy_budget.py`

Dual-mode energy system supporting DEPONS and JASMINE energy models.

#### Classes

##### `EnergyModule` (ABC)

```python
class EnergyModule(ABC):
    """Abstract base class for energy budget modules."""

    @abstractmethod
    def compute_energy_update(self, state: EnergyState, context: EnergyContext, mask: np.ndarray) -> EnergyResult:
        """Compute energy changes for one tick."""

    @abstractmethod
    def compute_survival_probability(self, state: EnergyState, mask: np.ndarray) -> np.ndarray:
        """Compute per-tick survival probability for each agent."""
```

##### `DEPONSEnergyModule`

```python
class DEPONSEnergyModule(EnergyModule):
    """
    DEPONS 3.2 energy model.

    - Energy scale: 0-20 (dimensionless)
    - Seasonal scaling: cold (Nov-Mar) = 1.0×, warm (May-Sep) = 1.3×
    - Lactation cost: 1.4× normal metabolism
    - Starvation formula: yearlySurv = 1 - m_mort_prob_const * exp(-energy * x_survival_const)
      with m_mort_prob_const=1.0, x_survival_const=0.4 (DEPONS 3.2 calibration)
    - BMR computed via depons_bmr_cost_kernel (Numba, prange-parallel)
    """
```

##### `JASMINEEnergyModule`

```python
class JASMINEEnergyModule(EnergyModule):
    """
    JASMINE Dynamic Energy Budget model.

    - Body mass-dependent BMR (Kleiber scaling)
    - Activity cost based on movement speed
    - Thermoregulation cost outside thermoneutral zone
    - Disturbance energy cost with cumulative tracking
    - Body condition index (0-1)
    - Survival uses DEPONS starvation formula (body_condition mapped to effective energy)
    """
```

##### `EnergyState`

```python
@dataclass
class EnergyState:
    """Per-agent energy state arrays (all shape (max_agents,))."""
    energy: np.ndarray          # Current energy level
    body_condition: np.ndarray  # Body condition index (JASMINE)
    disturbance_energy_cost: np.ndarray  # Cumulative disturbance cost
```

##### Factory

```python
def create_energy_module(params: SimulationParameters) -> EnergyModule:
    """Create energy module based on simulation mode. Always returns a module (never None)."""
```

---

### 4. Reference Memory Module

**Location:** `src/cenop/behavior/ref_mem.py`

Implements the DEPONS 3.2 reference memory system with precomputed decay tables.

#### Classes

##### `RefMemWorkspace`

```python
class RefMemWorkspace:
    """
    Reusable workspace for compute_ve_total and compute_attraction_vector.

    Pre-allocates intermediate arrays to avoid ~1.5 MB/tick of allocations.
    Pass as optional parameter to compute_ve_total() and compute_attraction_vector().
    """

    def __init__(self, max_agents: int, ref_mem_size: int = 120):
        """Allocate workspace arrays."""
```

#### Functions

```python
def build_ref_mem_strength(r_r: float, size: int = 120) -> np.ndarray:
    """Precompute refMemStrength decay table: (1 - r_r)^i for i in [0, size)."""

def build_work_mem_strength(r_s: float, size: int = 120) -> np.ndarray:
    """Precompute workMemStrength decay table: (1 - r_s)^i for i in [0, size)."""

def compute_ve_total(
    ref_mem_food: np.ndarray,
    ref_mem_count: np.ndarray,
    work_mem_strength: np.ndarray,
    active_mask: np.ndarray,
    workspace: RefMemWorkspace = None,
) -> np.ndarray:
    """Compute expected food value (veTotal) for all agents using vectorized NumPy."""

def compute_attraction_vector(
    x: np.ndarray, y: np.ndarray,
    ref_mem_x: np.ndarray, ref_mem_y: np.ndarray,
    ref_mem_food: np.ndarray, ref_mem_count: np.ndarray,
    ref_mem_strength: np.ndarray,
    active_mask: np.ndarray,
    workspace: RefMemWorkspace = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute attraction vector (vt_x, vt_y) towards remembered food-rich areas."""
```

---

### 5. Hybrid Behavioral FSM (JASMINE)

**Location:** `src/cenop/behavior/hybrid_fsm.py`

Unified finite state machine supporting both DEPONS and JASMINE behavioral models.

#### Classes

##### `HybridFSM`

```python
class HybridFSM:
    """
    Hybrid Finite State Machine for porpoise behavior.

    Supports both DEPONS (simple) and JASMINE (enhanced) modes.

    States:
        - FORAGING: Default state, searching for food (DEPONS CRW)
        - TRAVELING: Directed movement between areas (JASMINE physics)
        - RESTING: Low activity energy recovery (JASMINE physics)
        - DISPERSING: Memory-driven dispersal (PSM-based)
        - DISTURBED: Response to disturbance events
    """

    class State(Enum):
        FORAGING = 0
        TRAVELING = 1
        RESTING = 2
        DISPERSING = 3
        DISTURBED = 4

    def __init__(self, mode: str = "DEPONS"):
        """
        Initialize FSM.

        Args:
            mode: "DEPONS" or "JASMINE"
        """

    def update(
        self,
        context: BehaviorContext
    ) -> State:
        """
        Update behavioral state based on context.

        Args:
            context: BehaviorContext with energy, disturbance, speed, etc.

        Returns:
            New behavioral state
        """

    def get_movement_mode(self, state: State) -> str:
        """
        Get movement mode for given state.

        Returns:
            "DEPONS_CRW" or "JASMINE_PHYSICS"
        """
```

##### `BehaviorContext`

```python
@dataclass
class BehaviorContext:
    """Context for behavioral state transitions."""

    energy: float              # Current energy (0-1)
    speed: float               # Current speed (m/s)
    disturbance_level: float   # Local disturbance (0-1)
    memory_intensity: float    # Disturbance memory at location
    food_available: float      # Local food availability
    days_declining: int        # Days of energy decline
    is_lactating: bool         # Lactation status
    tick: int                  # Current simulation tick
```

---

### 6. Dynamic Energy Budget (JASMINE)

**Location:** `src/cenop/physiology/energy_budget.py`

Modular energy budget system supporting DEPONS and JASMINE models.

#### Classes

##### `EnergyBudget`

```python
class EnergyBudget:
    """
    Factory for energy budget modules.
    """

    @staticmethod
    def create(mode: str = "DEPONS") -> EnergyModule:
        """
        Create energy module for specified mode.

        Args:
            mode: "DEPONS" or "JASMINE"

        Returns:
            DEPONSEnergy or JASMINEEnergy instance
        """
```

##### `JASMINEEnergy`

```python
class JASMINEEnergy(EnergyModule):
    """
    JASMINE Dynamic Energy Budget model.

    Features:
        - Body mass-dependent metabolism (Kleiber scaling)
        - Activity cost based on movement speed
        - Thermoregulation cost based on water temperature
        - Disturbance energy cost with cumulative tracking
        - Body condition index and fat reserves
    """

    # Bioenergetics constants
    ADULT_MASS_KG: float = 50.0
    BMR_COEFFICIENT: float = 3.4  # W/kg^0.75
    THERMONEUTRAL_LOW: float = 5.0   # °C
    THERMONEUTRAL_HIGH: float = 20.0  # °C
    DISTURBANCE_BASE_COST: float = 0.1  # MJ

    def calculate_bmr(self, body_mass: float) -> float:
        """
        Calculate basal metabolic rate using Kleiber scaling.

        BMR = coefficient * mass^0.75

        Args:
            body_mass: Body mass in kg

        Returns:
            BMR in MJ/day
        """

    def calculate_activity_cost(
        self,
        speed: float,
        duration: float
    ) -> float:
        """
        Calculate activity energy cost.

        Args:
            speed: Movement speed in m/s
            duration: Duration in hours

        Returns:
            Activity cost in MJ
        """

    def calculate_thermoregulation_cost(
        self,
        water_temp: float,
        body_mass: float
    ) -> float:
        """
        Calculate thermoregulation cost outside thermoneutral zone.

        Args:
            water_temp: Water temperature in °C
            body_mass: Body mass in kg

        Returns:
            Thermoregulation cost in MJ/day
        """

    def calculate_disturbance_cost(
        self,
        disturbance_level: float,
        duration: float,
        cumulative_exposure: float
    ) -> float:
        """
        Calculate energy cost of disturbance response.

        Args:
            disturbance_level: Disturbance intensity (0-1)
            duration: Exposure duration in hours
            cumulative_exposure: Previous cumulative exposure

        Returns:
            Disturbance energy cost in MJ
        """

    def get_body_condition_index(self) -> float:
        """
        Calculate body condition index (0-1 scale).

        Returns:
            Body condition index
        """
```

---

### 7. Disturbance Memory (JASMINE)

**Location:** `src/cenop/behavior/disturbance_memory.py`

Spatial memory system for learned avoidance of disturbance zones.

#### Classes

##### `DisturbanceMemory`

```python
class DisturbanceMemory:
    """
    Spatial memory for disturbance zones.

    Features:
        - Grid-based memory storage
        - Exponential memory decay
        - Learned avoidance behavior
        - Habituation to repeated exposure
    """

    def __init__(
        self,
        grid_shape: Tuple[int, int],
        decay_rate: float = 0.001,
        habituation_enabled: bool = True,
        habituation_rate: float = 0.05
    ):
        """
        Initialize disturbance memory.

        Args:
            grid_shape: (rows, cols) of memory grid
            decay_rate: Per-tick memory decay rate
            habituation_enabled: Enable habituation
            habituation_rate: Rate of habituation per exposure
        """

    def record_disturbance(
        self,
        x: int,
        y: int,
        intensity: float
    ):
        """
        Record disturbance at location.

        Args:
            x, y: Grid coordinates
            intensity: Disturbance intensity (0-1)
        """

    def get_memory_intensity(self, x: int, y: int) -> float:
        """
        Get memory intensity at location.

        Args:
            x, y: Grid coordinates

        Returns:
            Memory intensity (0-1)
        """

    def calculate_avoidance_vector(
        self,
        x: int,
        y: int,
        radius: int = 20
    ) -> Tuple[float, float]:
        """
        Calculate avoidance direction based on remembered disturbances.

        Args:
            x, y: Current grid coordinates
            radius: Search radius in cells

        Returns:
            (dx, dy) avoidance vector
        """

    def decay(self):
        """Apply per-tick memory decay."""

    def get_habituation_factor(self, x: int, y: int) -> float:
        """
        Get habituation factor at location.

        Args:
            x, y: Grid coordinates

        Returns:
            Habituation factor (0-1, lower = more habituated)
        """
```

---

### Dispersal Module

**Location:** `src/cenop/behavior/dispersal.py`

Implements all DEPONS 3.2 dispersal types for porpoise movement away from energy-depleted areas.

Supported dispersal types (all 8 DEPONS 3.2 types):
- `OFF` — No dispersal
- `PSM_TYPE1` — Straight-line to target
- `PSM_TYPE2` — SSLogis heading dampening (default)
- `PSM_TYPE3` — Logistic increase in turning + distance-cost target selection (q1=0.02)
- `PSM_TYPE3_RANDDIR` — Type3 with random target (always bypasses PSM)
- `PSM_TYPE3_RANDDIST` — Type3 that never stops on distance
- `UNDIRECTED` — Type2 heading + random target + no calf PSM inheritance
- `INNER_DANISH_WATERS` — Block-based navigation for Danish waters (60 blocks, 2-phase)

---

### 8. Sound & Deterrence Module

**Location:** `src/cenop/behavior/sound.py`

Handles noise propagation, deterrence thresholds, and ship deterrence probability.

#### Classes

##### `ShipDeterrenceModel`

```python
class ShipDeterrenceModel:
    """
    Ship deterrence using standardised logistic regression (DEPONS 3.2).

    Inputs are standardised (x - mean) / sd before logistic regression.
    Constants in STD_PROB_DAY, STD_PROB_NIGHT for day/night probabilities.
    """

    def compute_deterrence_probability(self, received_level: float, is_day: bool) -> float:
        """Compute probability of deterrence response to ship noise."""
```

##### `Hydrophone`

Passive acoustic monitoring station (translates `Hydrophone.java`).

```python
from cenop.behavior.sound import Hydrophone

h = Hydrophone(name="H1", x=100.0, y=200.0)
h.receive_sound_level("Ship-A", utm_x=500000, utm_y=6000000,
                       source_level=170.0, received_level=145.0)
print(h.received_level)  # 145.0
h.reset()  # Call at end of each tick
```

##### `SoundPropagationParams`

```python
@dataclass
class SoundPropagationParams:
    """Sound propagation parameters (DEPONS 3.2 defaults)."""
    alpha_hat: float = 0.00027   # Absorption coefficient
    beta_hat: float = 14.72      # Spreading loss factor
    deter_threshold: float = 152.0  # Minimum received level (dB)
    deter_coeff: float = 0.012    # Deterrence coefficient
    deter_max_distance: float = 1000.0  # Max deterrence distance (km)
```

---

### 9. Weston Flux Transmission Loss

**Location:** `src/cenop/behavior/weston_flux.py`

Physics-based transmission loss model ported from DEPONS `WestonFlux.java`.

```python
def weston_flux_tl(distance: float, depth: float, grain_size: float, frequency: float = 125000) -> float:
    """
    Calculate transmission loss using Weston flux theory.

    Uses sediment grain size to derive sound speed ratio, density ratio,
    and attenuation coefficient. Falls back to simple beta*log10(r)+alpha*r
    when physics-based calculation is not applicable.

    Args:
        distance: Distance from source (m)
        depth: Water depth (m)
        grain_size: Sediment grain size on phi scale
        frequency: Sound frequency (Hz, default 125 kHz for porpoise hearing)

    Returns:
        Transmission loss in dB
    """
```

---

### 10. JOMOPANS Ship Source Levels

**Location:** `src/cenop/behavior/jomopans_spl.py`

Calibrated ship source levels from the JOMOPANS project.

```python
class VesselClass(Enum):
    """15 vessel classes (13 JOMOPANS + 2 aliases)."""
    FISHING = 0
    DREDGING = 1
    # ... 13 JOMOPANS classes total

def get_source_level(vessel_class: VesselClass, speed_knots: float) -> float:
    """Get source level in dB for a vessel class at given speed."""
```

---

### 7. Simulation Module

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

    def __init__(self, params: SimulationParameters):
        """Initialize simulation with parameters."""

    def setup(self):
        """Load landscape, create population, initialize subsystems."""

    def step(self) -> dict:
        """Execute one tick: move, forage, energy, mortality, reproduction."""

    def get_porpoise_positions(self) -> np.ndarray:
        """
        Get current positions and state of all porpoises.

        Returns:
            Array (N, 7): [original_index, x, y, energy, heading, age, is_dispersing]
        """
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
        should_update_map (bool): Whether map needs refresh this tick
        total_births, total_deaths (int): Cumulative counts
    """

    def step(self) -> dict:
        """Execute one tick and update progress."""
```

---

### 8. Batch Runner

**Location:** `src/cenop/core/batch_runner.py`

Runs multiple simulations for sensitivity analysis.

#### Classes

##### `BatchRunner`

```python
class BatchRunner:
    """
    Runs batch simulations with parameter variations.
    """
    
    def __init__(
        self,
        base_config: SimulationConfig,
        output_dir: str = "./output/batch"
    ):
        """Initialize batch runner."""
        
    def run_batch(
        self,
        parameter_sets: List[Dict[str, Any]],
        num_replicates: int = 5,
        parallel: bool = True
    ) -> List[BatchResult]:
        """
        Run batch of simulations.
        
        Args:
            parameter_sets: List of parameter dictionaries
            num_replicates: Replicates per parameter set
            parallel: Use parallel execution
            
        Returns:
            List of BatchResult objects
        """
        
    @staticmethod
    def generate_sensitivity_matrix(
        base_params: Dict[str, Any],
        vary_params: Dict[str, List[Any]]
    ) -> List[Dict[str, Any]]:
        """Generate parameter combinations for sensitivity analysis."""
```

---

### 9. Output Writer

**Location:** `src/cenop/core/output_writer.py`

DEPONS-compatible file output generation.

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
    
    def __init__(self, output_dir: str, config: OutputConfig = None):
        """Initialize output writer."""
        
    def write_population(self, tick: int, population: Population):
        """Write population statistics for current tick."""
        
    def write_mortality(self, event: MortalityEvent):
        """Record mortality event."""
        
    def write_dispersal(self, event: DispersalEvent):
        """Record dispersal event."""
        
    def finalize(self):
        """Close all output files."""
```

---

## Configuration

### SimulationParameters

**Location:** `src/cenop/parameters/simulation_params.py`

```python
@dataclass
class SimulationParameters:
    """Main simulation configuration."""

    # Population & Time
    porpoise_count: int = 1000
    sim_years: int = 5
    landscape: str = "Homogeneous"
    random_seed: int = 0          # 0 = random each run

    # Simulation Mode
    simulation_mode: str = "DEPONS"  # "DEPONS" or "JASMINE"
    energy_mode: str = None          # Override energy subsystem
    memory_mode: str = None          # Override memory subsystem
    fsm_mode: str = None             # Override FSM subsystem
    movement_mode: str = None        # Override movement subsystem

    # CRW Parameters (DEPONS 3.2 Kattegat calibration)
    inertia_const: float = 0.001     # k: directional persistence
    a0: float = 0.35                 # Step length autocorrelation
    b0: float = -0.024               # Turning angle autocorrelation

    # Energy & Memory (DEPONS 3.2)
    r_s: float = 0.03               # Satiation memory decay rate
    r_r: float = 0.03               # Reference memory decay rate
    r_u: float = 0.1                # Food replenishment rate
    ref_mem_size: int = 120          # Reference memory circular buffer size

    # Mortality (DEPONS 3.2 calibration)
    m_mort_prob_const: float = 1.0   # Starvation formula constant
    x_survival_const: float = 0.4    # Starvation formula exponent

    # Dispersal
    mean_disp_dist: float = 2.0      # Dispersal step distance (km)

    # Deterrence (DEPONS 3.2)
    deter_coeff: float = 0.012       # Deterrence coefficient
    deter_threshold: float = 152.0   # Minimum received level (dB)
    deter_max_distance: float = 1000.0  # Max deterrence distance (km)

    # Sound propagation (DEPONS 3.2)
    alpha_hat: float = 0.00027       # Absorption coefficient
    beta_hat: float = 14.72          # Spreading loss factor

    # Ship traffic
    ships_enabled: bool = False
    bycatch_prob: float = 0.0

    # JASMINE Physics
    jasmine_mass_kg: float = 50.0
    jasmine_drag_coeff: float = 0.01
    jasmine_max_thrust: float = 100.0
    jasmine_current_weight: float = 0.5

    # JASMINE DEB
    jasmine_bmr_scale: float = 1.0
    jasmine_activity_cost: float = 2.0
    jasmine_disturbance_cost: float = 1.5

    # JASMINE Memory
    memory_decay_rate: float = 0.001
    avoidance_radius: float = 20.0
    habituation_enabled: bool = True
    habituation_rate: float = 0.05
```

---

## Usage Examples

### Basic Simulation (DEPONS Mode)

```python
from cenop.core.simulation import Simulation
from cenop.parameters.simulation_params import SimulationParameters

# Create configuration
params = SimulationParameters(
    porpoise_count=500,
    sim_years=5,
    landscape="Kattegat",
)

# Initialize and run
sim = Simulation(params)
sim.setup()

for tick in range(params.sim_years * 360 * 48):
    state = sim.step()
    if tick % 48 == 0:  # Daily output
        positions = sim.get_porpoise_positions()
        print(f"Day {tick//48}: Pop={len(positions)}")
```

### JASMINE Mode Simulation

```python
from cenop.core.simulation import Simulation
from cenop.parameters.simulation_params import SimulationParameters

# Create JASMINE configuration
params = SimulationParameters(
    porpoise_count=1000,
    sim_years=5,
    simulation_mode="JASMINE",

    # Enable all JASMINE subsystems
    energy_mode="JASMINE",
    memory_mode="JASMINE",
    fsm_mode="JASMINE",

    # Custom JASMINE parameters
    jasmine_bmr_scale=1.2,      # 20% higher metabolism
    memory_decay_rate=0.001,    # Slow memory decay
    habituation_enabled=True,   # Enable habituation
    jasmine_thermal_model=True  # Temperature-dependent costs
)

# Create and run simulation
sim = Simulation(params)
sim.initialize()

# Run with state tracking
for tick in range(params.total_ticks):
    state = sim.step()

    # Access JASMINE-specific metrics
    if tick % 48 == 0:  # Daily
        print(f"Day {tick//48}:")
        print(f"  Population: {state['population']}")
        print(f"  Avg Body Condition: {state['avg_body_condition']:.2f}")
        print(f"  Disturbed Count: {state['disturbed_count']}")
        print(f"  Memory Intensity: {state['avg_memory_intensity']:.3f}")
```

### Batch Analysis

```python
from cenop.core.batch_runner import BatchRunner

# Define parameter variations
vary_params = {
    'initial_population': [100, 200, 300],
    'starvation_threshold': [0.05, 0.1, 0.15]
}

# Generate parameter sets
param_sets = BatchRunner.generate_sensitivity_matrix(
    base_config.to_dict(),
    vary_params
)

# Run batch
runner = BatchRunner(base_config)
results = runner.run_batch(param_sets, num_replicates=5)

# Analyze results
for result in results:
    print(f"Params: {result.parameters}")
    print(f"Final pop: {result.final_population} ± {result.std_population}")
```

---

## Time Manager

**Location:** `src/cenop/core/time_manager.py`

### Suntimes

Seasonal sunrise/sunset from CSV (translates `Suntimes.java`). Falls back to fixed 6am-6pm when no CSV provided.

```python
from cenop.core.time_manager import TimeManager, Suntimes

tm = TimeManager(suntimes_path="data/suntimes.csv")
print(tm.is_daytime)  # Uses per-DOY sunrise/sunset from CSV
```

---

## Version History

### v2.3.0 (2026-03-21)
- Full DEPONS 3.2 parity: all 153 features verified (136 MATCH, 13 divergences, 4 extensions)
- 8 dispersal types (added PSM-Type3-randdir, randdist, Undirected, InnerDanishWaters)
- Hydrophone monitoring, Suntimes CSV support, death age tracking
- Performance: 1.05 ms/tick DEPONS mode (was 2.10 ms), 12 Numba kernels (fused heading+position+reflect, social sound)
- 516+ tests (up from 498)

- **v2.1.0**: DEPONS 3.2 full sync + performance optimization + new features
  - Algorithmically aligned with DEPONS 3.2 across all 5 subsystems
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
  - Porpoise trace trails using TripsLayer with age-based coloring
  - Skip visualization toggle (25.7% end-to-end speedup)
  - Unified bitmap grid rendering (server-side PNG, one pixel per cell)
  - 515+ automated tests across 24 test files

- **v2.0.0**: CENOP-JASMINE merge
  - Added JASMINE simulation mode
  - Hybrid behavioral FSM (5 states)
  - Dynamic Energy Budget with body mass scaling
  - Disturbance memory with learned avoidance
  - Habituation to repeated disturbance
  - Physics-based movement option
  - Dual-mode configuration (DEPONS/JASMINE)

- **v0.1.0**: Initial release with core simulation functionality
  - Core modules: Landscape, Population, Energetics, PSM
  - DEPONS-compatible output
  - Shiny web interface
