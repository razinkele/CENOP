# CENOP-JASMINE API Documentation

## Overview

CENOP-JASMINE (Cetacean Noise Operations Planner with JASMINE Extensions) is a Python Shiny application for simulating harbor porpoise population dynamics in response to wind farm construction noise. This document describes the core API for developers and researchers.

The API supports two simulation modes:
- **DEPONS Mode**: Regulatory-compatible empirical models (default)
- **JASMINE Mode**: Research-grade physics-based and bioenergetics models

---

## Core Modules

### 1. Landscape Module

**Location:** `src/cenop/core/landscape.py`

The landscape module manages the spatial environment including bathymetry, food availability, and environmental data.

#### Classes

##### `Landscape`

```python
class Landscape:
    """
    Manages the spatial grid environment for porpoise simulation.
    
    Attributes:
        width (int): Grid width in cells
        height (int): Grid height in cells
        cell_size (float): Size of each cell in meters (default: 400m)
        bathymetry (np.ndarray): Depth values for each cell
        food_level (np.ndarray): Food availability (0-1) for each cell
        food_prob (np.ndarray): Probability of food presence
        salinity (np.ndarray): Salinity values (optional)
        disturbance (np.ndarray): Disturbance levels from noise sources
    """
    
    def __init__(self, width: int, height: int, cell_size: float = 400.0):
        """Initialize landscape with given dimensions."""
        
    def load_from_raster(self, bathymetry_path: str, food_path: str = None):
        """Load landscape data from GeoTIFF raster files."""
        
    def get_cell_value(self, x: int, y: int, layer: str) -> float:
        """Get value at specific cell for given layer."""
        
    def update_disturbance(self, ships: List[Ship], turbines: List[Turbine]):
        """Update disturbance layer based on noise sources."""
        
    def get_attractiveness(self, x: int, y: int) -> float:
        """Calculate cell attractiveness for porpoise movement."""
```

---

### 2. Population Module

**Location:** `src/cenop/agents/population.py`

Vectorized implementation of porpoise population dynamics using NumPy arrays.

#### Classes

##### `Population`

```python
class Population:
    """
    Vectorized porpoise population management.
    
    Attributes:
        size (int): Current population size
        max_size (int): Maximum population capacity
        positions (np.ndarray): Shape (N, 2) - x, y coordinates
        energy (np.ndarray): Shape (N,) - energy reserves (0-1)
        age (np.ndarray): Shape (N,) - age in days
        sex (np.ndarray): Shape (N,) - 0=male, 1=female
        pregnant (np.ndarray): Shape (N,) - pregnancy status (boolean)
        lactating (np.ndarray): Shape (N,) - lactation status (boolean)
        dispersing (np.ndarray): Shape (N,) - dispersal status (boolean)
        alive (np.ndarray): Shape (N,) - alive status (boolean)
    """
    
    def __init__(self, initial_size: int = 200, max_size: int = 500):
        """Initialize population with random individuals."""
        
    def step(self, landscape: Landscape, tick: int):
        """Execute one simulation step for all porpoises."""
        
    def move(self, landscape: Landscape):
        """Update positions based on landscape attractiveness."""
        
    def forage(self, landscape: Landscape):
        """Update energy based on foraging success."""
        
    def update_energy(self, tick: int):
        """Apply metabolic costs and update energy reserves."""
        
    def reproduce(self, tick: int):
        """Handle reproduction for pregnant females."""
        
    def mortality(self):
        """Apply mortality based on age and energy."""
        
    def respond_to_disturbance(self, landscape: Landscape):
        """Apply deterrence response to noise disturbance."""
        
    def get_statistics(self) -> Dict[str, Any]:
        """Return population statistics for current tick."""
```

---

### 3. Energetics Module

**Location:** `src/cenop/core/energetics.py`

Energy budget calculations based on the Dynamic Energy Budget (DEB) theory.

#### Classes

##### `EnergeticsModel`

```python
class EnergeticsModel:
    """
    Dynamic Energy Budget model for porpoise energetics.
    
    Parameters follow Hin et al. (2019) and DEPONS parameterization.
    """
    
    # Constants (kJ/day at standard metabolic rate)
    BMR_ADULT: float = 15000.0  # Basal metabolic rate
    ACTIVITY_MULTIPLIER: float = 2.5  # Active metabolism
    GESTATION_COST: float = 1.2  # Multiplier during pregnancy
    LACTATION_COST: float = 2.0  # Multiplier during lactation
    
    def calculate_metabolic_cost(
        self, 
        age: float, 
        weight: float,
        is_pregnant: bool = False,
        is_lactating: bool = False,
        activity_level: float = 1.0
    ) -> float:
        """
        Calculate daily metabolic energy cost in kJ.
        
        Args:
            age: Age in days
            weight: Body weight in kg
            is_pregnant: Pregnancy status
            is_lactating: Lactation status
            activity_level: Activity multiplier (0.5-2.0)
            
        Returns:
            Daily energy expenditure in kJ
        """
        
    def calculate_foraging_gain(
        self,
        food_availability: float,
        foraging_time: float,
        body_size: float
    ) -> float:
        """
        Calculate energy gain from foraging.
        
        Args:
            food_availability: Local food level (0-1)
            foraging_time: Time spent foraging in hours
            body_size: Body size factor
            
        Returns:
            Energy gained in kJ
        """
        
    def calculate_starvation_risk(self, energy_reserve: float) -> float:
        """
        Calculate probability of starvation-related mortality.
        
        Args:
            energy_reserve: Current energy reserve (0-1)
            
        Returns:
            Daily mortality probability (0-1)
        """
```

---

### 4. Porpoise State Machine (PSM)

**Location:** `src/cenop/agents/psm.py`

Implements the Porpoise State Machine for behavioral state transitions.

#### Classes

##### `PSM`

```python
class PSM:
    """
    Porpoise State Machine managing behavioral states.
    
    States:
        - RESTING: Low activity, minimal movement
        - FORAGING: Actively searching for food
        - TRAVELLING: Moving between areas
        - DISPERSING: Long-distance movement (juveniles/adults)
    """
    
    class State(Enum):
        RESTING = 0
        FORAGING = 1
        TRAVELLING = 2
        DISPERSING = 3
    
    def __init__(self):
        """Initialize PSM with default transition probabilities."""
        
    def get_next_state(
        self, 
        current_state: State,
        energy: float,
        food_available: float,
        disturbance: float
    ) -> State:
        """
        Determine next behavioral state.
        
        Args:
            current_state: Current behavioral state
            energy: Energy reserve (0-1)
            food_available: Local food availability (0-1)
            disturbance: Local disturbance level (0-1)
            
        Returns:
            Next behavioral state
        """
        
    def get_movement_parameters(self, state: State) -> Tuple[float, float]:
        """
        Get movement parameters for given state.

        Returns:
            (step_length, turning_angle_std) in meters and radians
        """
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

### 8. Deterrence Module

**Location:** `src/cenop/agents/deterrence.py`

Handles porpoise response to anthropogenic disturbance.

#### Classes

##### `DeterrenceModel`

```python
class DeterrenceModel:
    """
    Models porpoise deterrence response to noise.
    
    Based on empirical dose-response relationships from
    Tougaard et al. and other studies.
    """
    
    # Threshold levels (dB re 1 µPa)
    THRESHOLD_MILD: float = 120.0  # Mild behavioral response
    THRESHOLD_STRONG: float = 140.0  # Strong avoidance
    THRESHOLD_TTS: float = 160.0  # Temporary threshold shift
    
    def calculate_response_probability(
        self,
        received_level: float,
        exposure_duration: float
    ) -> float:
        """
        Calculate probability of deterrence response.
        
        Args:
            received_level: Sound pressure level in dB
            exposure_duration: Cumulative exposure in hours
            
        Returns:
            Response probability (0-1)
        """
        
    def calculate_flight_distance(
        self,
        received_level: float,
        current_distance: float
    ) -> float:
        """
        Calculate distance porpoise will flee.
        
        Args:
            received_level: Sound pressure level in dB
            current_distance: Current distance from source in meters
            
        Returns:
            Flight distance in meters
        """
```

---

### 6. Noise Source Module

**Location:** `src/cenop/core/noise.py`

Manages noise sources including pile driving and vessel traffic.

#### Classes

##### `NoiseSource`

```python
class NoiseSource:
    """Base class for noise sources."""
    
    def __init__(self, x: float, y: float, source_level: float):
        """
        Initialize noise source.
        
        Args:
            x, y: Position in meters
            source_level: Source level in dB re 1 µPa @ 1m
        """
        
    def get_received_level(self, distance: float) -> float:
        """
        Calculate received level at given distance.
        
        Uses spherical spreading with absorption.
        """

##### `PileDriver`

```python
class PileDriver(NoiseSource):
    """Pile driving noise source."""
    
    def __init__(
        self,
        x: float, 
        y: float,
        hammer_energy: float = 2000.0,  # kJ
        strike_rate: float = 30.0,  # strikes/minute
        source_level: float = 220.0  # dB
    ):
        """Initialize pile driver."""
        
    def get_sel(self, distance: float, duration: float) -> float:
        """
        Calculate Sound Exposure Level.
        
        Args:
            distance: Distance in meters
            duration: Exposure duration in seconds
            
        Returns:
            SEL in dB re 1 µPa²·s
        """
```

##### `Vessel`

```python
class Vessel(NoiseSource):
    """Vessel noise source."""
    
    def __init__(
        self,
        x: float,
        y: float,
        speed: float = 10.0,  # m/s
        source_level: float = 170.0  # dB
    ):
        """Initialize vessel."""
        
    def move(self, dx: float, dy: float):
        """Update vessel position."""
```

---

### 7. Simulation Controller

**Location:** `server/simulation_controller.py`

Main simulation orchestration class.

#### Classes

##### `SimulationController`

```python
class SimulationController:
    """
    Controls the simulation lifecycle.
    
    Manages initialization, stepping, and data collection.
    """
    
    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation with configuration.
        
        Args:
            config: SimulationConfig with all parameters
        """
        
    def initialize(self):
        """Set up initial simulation state."""
        
    def step(self) -> Dict[str, Any]:
        """
        Execute one simulation tick.
        
        Returns:
            Dictionary with current state and statistics
        """
        
    def run(self, num_ticks: int, callback: Callable = None):
        """
        Run simulation for specified ticks.
        
        Args:
            num_ticks: Number of ticks to run
            callback: Optional callback after each tick
        """
        
    def get_state(self) -> Dict[str, Any]:
        """Return current simulation state."""
        
    def export_results(self, output_dir: str):
        """Export simulation results to files."""
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

### SimulationConfig

```python
@dataclass
class SimulationConfig:
    """Main simulation configuration."""
    
    # Landscape
    landscape_file: str = None
    grid_width: int = 100
    grid_height: int = 100
    cell_size: float = 400.0
    
    # Population
    initial_population: int = 200
    max_population: int = 500
    
    # Time
    start_tick: int = 0
    end_tick: int = 8760  # 1 year in hours
    ticks_per_day: int = 24
    
    # Energetics
    use_deb_model: bool = True
    starvation_threshold: float = 0.1
    
    # Reproduction
    mating_season_start: int = 182  # July
    mating_season_end: int = 243  # September
    gestation_period: int = 300  # days
    
    # Disturbance
    deterrence_enabled: bool = True
    noise_sources: List[NoiseSource] = field(default_factory=list)
    
    # Output
    output_dir: str = "./output"
    save_interval: int = 24

    # JASMINE Mode Settings
    simulation_mode: str = "DEPONS"  # "DEPONS" or "JASMINE"

    # Subsystem overrides (optional)
    fsm_mode: str = None       # Override FSM mode
    energy_mode: str = None    # Override energy mode
    memory_mode: str = None    # Override memory mode
    movement_mode: str = None  # Override movement mode

    # JASMINE Physics Parameters
    jasmine_mass_kg: float = 50.0
    jasmine_drag_coeff: float = 0.01
    jasmine_max_thrust: float = 100.0
    jasmine_current_weight: float = 0.5

    # JASMINE DEB Parameters
    jasmine_bmr_scale: float = 1.0
    jasmine_activity_cost: float = 2.0
    jasmine_thermal_model: bool = True
    jasmine_disturbance_cost: float = 1.5

    # JASMINE Memory Parameters
    memory_decay_rate: float = 0.001
    avoidance_radius: float = 20.0
    habituation_enabled: bool = True
    habituation_rate: float = 0.05
```

---

## Usage Examples

### Basic Simulation (DEPONS Mode)

```python
from cenop.core.landscape import Landscape
from cenop.agents.population import Population
from server.simulation_controller import SimulationController

# Create configuration
config = SimulationConfig(
    initial_population=200,
    end_tick=8760,
    deterrence_enabled=True
)

# Initialize controller
controller = SimulationController(config)
controller.initialize()

# Run simulation
for tick in range(config.end_tick):
    state = controller.step()
    if tick % 24 == 0:  # Daily output
        print(f"Day {tick//24}: Pop={state['population_size']}")
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

## Version History

- **v2.1.0**: DEPONS 3.2 full sync + Numba optimization
  - Algorithmically aligned with DEPONS 3.2 Java implementation
  - Pregnancy finite state machine with daily mortality schedule
  - PSM-Type2 dispersal with SSLogis heading dampening
  - Reference memory with vectorized attraction computation
  - CRW rejection sampling (3 loops: angle, distance-modulated angle, step length)
  - Ship deterrence with JOMOPANS 13-class source levels and Weston flux TL
  - Numba JIT kernels for 7 hot-path operations with prange parallelism
  - Logistic food regrowth with proportional-sharing consumption
  - 502 automated tests

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
