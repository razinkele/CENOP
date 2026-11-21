"""
Simulation parameters configuration.

All configurable model parameters with their defaults and validation.
Translates from: SimulationParameters.java (779 lines)
"""

from __future__ import annotations

from dataclasses import dataclass

DISPERSAL_MODES: tuple[str, ...] = ("off", "PSM-Type2", "Undirected", "InnerDanishWaters")

# Salinity in the CRW movement (step length and turning angle). The DEPONS movement model was
# fitted to porpoises tagged in the eastern North Sea and is not valid at much lower salinity,
# so DEPONS moves with a fixed salinity on Kattegat (Porpoise.java KATTEGAT_SALINITY; TRACE 3.2
# section 3.2.2). CENOP does the same on its Baltic landscapes (surface salinity about 7).
MOVEMENT_SALINITY_MODES: tuple[str, ...] = ("auto", "map", "fixed")
FIXED_MOVEMENT_SALINITY = 34.069105813295
FIXED_SALINITY_LANDSCAPES: tuple[str, ...] = ("Kattegat", "Lithuania", "CentralBaltic")


def fixed_movement_salinity(mode: str, landscape: str) -> float | None:
    """Salinity the CRW movement uses instead of the landscape map, or None for the map.

    "auto": fixed on Kattegat (as DEPONS) and the Baltic landscapes, the map elsewhere;
    "map": the map everywhere; "fixed": fixed everywhere.
    """
    if mode == "fixed" or (mode == "auto" and landscape in FIXED_SALINITY_LANDSCAPES):
        return FIXED_MOVEMENT_SALINITY
    return None


@dataclass
class SimulationParameters:
    """
    All simulation parameters with defaults from DEPONS.

    Translates from: SimulationParameters.java and parameters.xml
    """

    # === Simulation Setup ===
    random_seed: int | None = None
    porpoise_count: int = 10000
    # Largest number of animals the population may grow to; 0 = no limit (DEPONS).
    # Each animal costs about 320 KB of memory. Calves weaned at the limit are dropped.
    max_population: int = 0
    tracked_porpoise_count: int = 1
    sim_years: int = 50
    landscape: str = "Lithuania"
    debug: int = 0

    # === JASMINE Mode Selection ===
    # Controls which subsystem implementations to use
    # "DEPONS" = regulatory-compatible empirical models (default)
    # "JASMINE" = research-grade physics/DEB models
    simulation_mode: str = "DEPONS"  # Overall mode: "DEPONS" or "JASMINE"

    # Individual subsystem modes (override simulation_mode if set)
    # None = follow simulation_mode, "DEPONS"/"JASMINE" = override
    time_mode: str | None = None  # TimeManager mode
    movement_mode: str | None = None  # Movement module mode
    fsm_mode: str | None = None  # Behavioral FSM mode
    energy_mode: str | None = None  # Energy budget mode
    memory_mode: str | None = None  # Disturbance memory mode

    # === JASMINE Physics Parameters (when movement_mode=JASMINE) ===
    jasmine_mass_kg: float = 50.0  # Body mass (kg) for physics calculations
    jasmine_drag_coeff: float = 0.01  # Drag coefficient
    jasmine_max_thrust: float = 100.0  # Maximum thrust force (N)
    jasmine_current_weight: float = 0.5  # Weight of ocean current advection (0-1)
    dt_seconds: int = 1800  # Time step in seconds (30 min default)
    prey_attraction_weight: float = 1.0  # Weight of prey attraction force
    avoidance_weight: float = 2.0  # Weight of disturbance avoidance force

    # === JASMINE DEB Parameters (when energy_mode=JASMINE) ===
    jasmine_bmr_scale: float = 1.0  # Basal metabolic rate scale factor
    jasmine_activity_cost: float = 2.0  # Activity cost multiplier
    jasmine_disturbance_cost: float = 1.5  # Extra energy cost during disturbance
    jasmine_body_mass_scaling: bool = True  # Use body-mass-based metabolic scaling
    jasmine_thermal_model: bool = True  # Use thermal conductance model
    jasmine_disturbance_cost_mult: float = 1.0  # Disturbance energy cost multiplier

    # === JASMINE Memory Parameters (when memory_mode=JASMINE) ===
    jasmine_memory_decay_rate: float = 0.016  # Memory decay per tick (copied to memory_decay_rate)
    jasmine_avoidance_strength: float = 0.8  # Max learned avoidance strength
    jasmine_avoidance_radius: float = 2.5  # Avoidance radius, 2 km memory cells (copied)
    # Learned avoidance (JASMINE memory), calibrated 2026-11-21 so that numbers near a pile
    # driving site recover close to DEPONS's calibrated recovery: a full-strength memory (1)
    # falls below the 0.1 avoidance threshold in about 3 days (Brandt et al. 2011: activity below
    # normal for 24-72 h at 2.6 km), within 5 km (Graham et al. 2019: 50% response 1.3-7.4 km)
    memory_decay_rate: float = 0.016  # Per-tick memory decay rate
    avoidance_radius: float = 2.5  # Memory cells (5 grid cells, 2 km) for avoidance: 5 km
    habituation_enabled: bool = True  # Not used yet: habituation is not implemented

    # === Disturbance Sources ===
    turbines: str = "off"
    ships_enabled: bool = False

    # === Behavioral FSM ===
    disturbance_recovery_ticks: int = 48  # Ticks to recover from DISTURBED state
    min_memory_cells: int = 50  # Minimum visited cells before dispersal

    # Salinity in the CRW movement: "auto", "map" or "fixed" (see MOVEMENT_SALINITY_MODES)
    movement_salinity: str = "auto"

    # === Dispersal ===
    dispersal: str = "PSM-Type2"
    t_disp: int = 3  # Days of declining energy before dispersal
    psm_log: float = 0.6  # Logistic increase in random turning
    psm_dist_mean: float = 350.0  # Preferred dispersal distance (km) — Java parameters.xml:46
    psm_dist_sd: float = 100.0
    psm_tol: float = 5.0  # Tolerance band (km)
    psm_angle: float = 40.0  # Max turning angle after PSM step — Java parameters.xml:52
    q1: float = 0.02  # PSM-Type3 distance-cost coefficient (DEPONS 3.2)

    # === Memory ===
    r_s: float = 0.03  # Satiation memory decay rate (DEPONS 3.2)
    r_r: float = 0.03  # Reference memory decay rate (DEPONS 3.2)
    ref_mem_size: int = 120  # Reference memory buffer size (entries per agent)

    # === Movement ===
    inertia_const: float = 0.001  # k: tendency to keep moving with CRW
    corr_logmov_length: float = 0.35  # a0: autoregressive for log10(d/100)
    corr_logmov_bathy: float = 0.0005  # a1: depth effect on log10(d/100)
    corr_logmov_salinity: float = -0.02  # a2: salinity effect on log10(d/100)
    corr_angle_base: float = -0.024  # b0: autoregressive for turning angle
    corr_angle_bathy: float = -0.008  # b1: depth effect on turning angle
    corr_angle_salinity: float = 0.93  # b2: salinity effect on turning angle
    corr_angle_base_sd: float = -14.0  # b3: intercept for turning angle
    mean_disp_dist: float = 2.0  # Dispersal distance per step (km) — Java parameters.xml:117
    max_mov: float = 1.73  # Max movement distance (km)

    # === Random Components (TRACE Table 2) ===
    r1_mean: float = 1.25  # R1 mean for step length N(μ, σ)
    r1_sd: float = 0.15  # R1 SD for step length
    r2_mean: float = 0.0  # R2 mean for turning angle N(μ, σ)
    r2_sd: float = 4.0  # R2 SD for turning angle (degrees)
    m: float = (
        0.00001  # Java SimulationParameters.initialize():262 hardcodes this (field default was 10^0.74)
    )

    # === Energetics ===
    e_use_per_30_min: float = 4.5  # Energy use per half-hour step
    e_lact: float = 1.4  # Lactation energy multiplier
    e_warm: float = 1.3  # Warm water energy multiplier
    # Finding #10 — swimming + disturbance drains are JASMINE opt-ins.
    # DEPONS has E_USE_PER_KM=0.0 (no swimming term) and no disturbance energy term.
    e_use_per_km: float = 0.0  # Swimming activity coefficient (JASMINE opt-in, e.g. 0.0001)
    jasmine_disturbance_energy: bool = False  # If True, drain disturbance energy during deterrence
    energy_init_mean: float = 10.0  # Initial energy N(mean, sd)
    energy_init_sd: float = 1.0

    # === Deterrence ===
    deter_coeff: float = 0.012  # c: deterrence coefficient (DEPONS 3.2)
    deter_threshold: float = 152.0  # RT: minimum received level (dB) - Java default (DEPONS 3.2)
    deter_decay: float = 50.0  # Psi_deter: decay rate (%)
    deter_time: int = 0  # tdeter: deterrence duration (steps) - Java default (DEPONS 3.2)
    deter_max_distance: float = (
        1000.0  # dmax_deter: max deterrence distance (km) — parameters.xml (DEPONS 3.2)
    )
    deter_min_distance_ships: float = 0.1  # Min deterrence distance for ships (km)
    deter_ships_min_db: float = (
        80.0  # Tships: minimum RL to trigger ship deterrence (dB) — parameters.xml (DEPONS 3.2)
    )

    # WestonFlux per-cell transmission loss (optional, off by default)
    # Ship transmission loss by WestonFlux from depth, sediment and salinity at the ship's
    # position, as DEPONS always does (Ship.calculateReceivedLevelFor). False = the simple
    # spreading formula (a CENOP option; used automatically when a landscape has no sediment).
    weston_flux_percell: bool = True
    weston_flux_default_temperature: float = 10.0  # Water temperature (deg C); DEPONS WATER_TEMP

    # Probabilistic turbine deterrence response (JASMINE extension).
    # DEPONS applies FULL turbine deterrence strength deterministically once RL exceeds
    # the threshold (only ships draw a Bernoulli reaction), so the default is DEPONS-pure.
    # Set True to opt into logistic response-probability scaling of turbine strength (JASMINE).
    deter_probabilistic: bool = False
    deter_response_slope: float = 0.2  # Steepness (per dB) of logistic turbine response

    # === Social communication (new feature) ===
    # Opt-in (JASMINE): with realistic food the DEPONS CRW weight is ~0.001, so the social vector
    # would dictate headings. Off matches the UI checkbox default and DEPONS behaviour.
    communication_enabled: bool = False  # Enable social calling and cohesion
    communication_range_km: float = 10.0  # Communication detection range (km)
    communication_source_level: float = 160.0  # Source level (dB re 1 µPa) of porpoise calls
    communication_threshold: float = 120.0  # RL for 50% detection probability (dB)
    communication_response_slope: float = 0.2  # Steepness of detection logistic
    # Social attraction as a fraction of the random walk's pull (uncalibrated; see
    # docs/2026-11-20-social-weight.md)
    social_weight: float = 0.3

    # How often (in ticks) to recompute neighbor topology for social calls. Reusing
    # the neighbor pairs for a few ticks can reduce per-tick overhead when agents
    # move slowly. Set to 1 to recompute every tick (default behavior).
    communication_recompute_interval: int = 4

    # Adaptive recompute options
    communication_recompute_adaptive: bool = True
    communication_recompute_min_interval: int = 1
    communication_recompute_max_interval: int = 16
    # Threshold (meters per tick) below which we consider agents "stationary" enough to
    # safely increase the recompute interval. Default: 50 meters per tick.
    communication_recompute_disp_threshold_m: float = 50.0
    # EMA smoothing factor for displacement (0-1). Larger values track recent motion closely.
    communication_recompute_ema_alpha: float = 0.3

    # === Ship Deterrence Coefficients ===
    pship_int_day: float = -3.0569351
    pship_int_night: float = -3.233771
    pship_noise_day: float = 0.2172813
    pship_dist_day: float = -0.1303880
    pship_dist_x_noise_day: float = 0.0293443
    pship_noise_night: float = 0.0
    pship_dist_night: float = 0.085242
    pship_dist_x_noise_night: float = 0.0

    cship_int_day: float = 2.9647996
    cship_int_night: float = 2.7543376
    cship_noise_day: float = 0.0472709
    cship_dist_day: float = -0.0355541
    cship_dist_x_noise_day: float = 0.0
    cship_noise_night: float = 0.0
    cship_dist_night: float = 0.0284629
    cship_dist_x_noise_night: float = 0.0

    # === Sound Propagation ===
    alpha_hat: float = 0.00027  # Absorption coefficient (DEPONS 3.2)
    beta_hat: float = 14.72  # Spreading loss factor (DEPONS 3.2)

    # === Reproduction ===
    conceive_prob: float = 0.68  # h: probability of becoming pregnant
    gestation_time: int = 300  # tgest: gestation time (days)
    nursing_time: int = 240  # tnurs: nursing time (days)
    mating_day_mean: float = 225.0  # Mean mating day (day of year)
    mating_day_sd: float = 20.0

    # === Life History ===
    max_age: float = 30.0  # Maximum age (years)
    maturity_age: float = 3.44  # Age of maturity (years) - DEPONS default
    max_breeding_age: float = 30.0  # Set to max_age (effectively removed per spec 2.5)

    # === Environment ===
    min_depth: float = 1.0  # wmin: minimum water depth (m)
    min_depth_dispersal: float = 4.0  # wdisp: minimum depth when dispersing (m)

    # === Survival/Mortality ===
    # Starvation mortality formula: yearlySurvProb = 1 - (m_mort_prob_const * exp(-energy * x_survival_const))
    m_mort_prob_const: float = 1.0  # M_MORT_PROB_CONST in DEPONS SimulationConstants (DEPONS 3.2)
    x_survival_const: float = 0.4  # xSurvivalProbConst / beta in DEPONS parameters.xml (DEPONS 3.2)
    # Age-dependent annual mortality rates
    mortality_juvenile: float = 0.15  # Annual mortality for age < 1 year
    mortality_adult: float = 0.05  # Annual mortality for 1 <= age <= 20
    mortality_elderly: float = 0.15  # Annual mortality for age > 20
    # Bycatch
    bycatch_prob: float = 0.0  # Annual bycatch probability

    # === Food ===
    max_u: float = 1.0  # maxU: Maximum utility of a food patch (hardcoded in Java)
    food_growth_rate: float = 0.1  # rU: Logistic food growth rate per day (DEPONS 3.2)
    regrowth_food_qualifier: float = 0.001  # Umin: threshold for 48x compounding
    u_min: float = 0.001  # Minimum food level in patch

    # === Landscape ===
    wrap_border_homo: bool = True  # Wrap border for homogeneous landscape
    world_width: int = 1000  # Grid width (for homogeneous)
    world_height: int = 1000  # Grid height (for homogeneous)

    # The JAX backend was removed after 2.3.0; True raises (docs/decisions/2026-11-20-backends.md)
    use_jax: bool = False

    # === Model Version ===
    model: int = 4  # Model version (affects behavior) - DEPONS hardcodes to 4

    def __post_init__(self):
        """Validate parameters."""
        self._validate()

    def _validate(self) -> None:
        """Validate parameter ranges."""
        if self.dispersal not in DISPERSAL_MODES:
            raise ValueError(
                f"unknown dispersal {self.dispersal!r}; use one of: {', '.join(DISPERSAL_MODES)}"
            )
        if self.movement_salinity not in MOVEMENT_SALINITY_MODES:
            raise ValueError(
                f"unknown movement_salinity {self.movement_salinity!r}; use one of: "
                + ", ".join(MOVEMENT_SALINITY_MODES)
            )
        if self.dispersal == "InnerDanishWaters" and self.landscape != "Kattegat":
            # DEPONS hard-codes Kattegat geography into this mode (cell coordinates, the
            # 600 x 1000 navigation grid). Java also allows Homogeneous; CENOP does not yet.
            raise ValueError(
                "dispersal 'InnerDanishWaters' only supports the Kattegat landscape, "
                f"not {self.landscape!r}"
            )
        if self.porpoise_count < 0:
            raise ValueError("porpoise_count must be non-negative")
        if self.porpoise_count > 50000:
            raise ValueError("porpoise_count must not exceed 50,000")
        if self.max_population < 0:
            raise ValueError("max_population must be non-negative (0 = no limit)")
        if 0 < self.max_population < self.porpoise_count:
            raise ValueError("max_population must not be below porpoise_count")
        if self.sim_years < 1:
            raise ValueError("sim_years must be at least 1")
        if self.sim_years > 200:
            raise ValueError("sim_years must not exceed 200")
        if self.max_age <= 0:
            raise ValueError("max_age must be positive")
        if not 0 <= self.conceive_prob <= 1:
            raise ValueError("conceive_prob must be between 0 and 1")
        if self.r1_sd < 0:
            raise ValueError("r1_sd (step length std dev) must be non-negative")
        if self.r2_sd < 0:
            raise ValueError("r2_sd (turning angle std dev) must be non-negative")
        if self.e_use_per_30_min < 0:
            raise ValueError("e_use_per_30_min must be non-negative")
        if self.deter_threshold < 0:
            raise ValueError("deter_threshold must be non-negative")
        if self.deter_max_distance <= 0:
            raise ValueError("deter_max_distance must be positive")
        if not 0 < self.m_mort_prob_const <= 1:
            raise ValueError("m_mort_prob_const must be between 0 (exclusive) and 1 (inclusive)")
        if self.x_survival_const < 0:
            raise ValueError("x_survival_const must be non-negative")
        if not 0 <= self.bycatch_prob <= 1:
            raise ValueError("bycatch_prob must be between 0 and 1")
        if self.maturity_age < 0:
            raise ValueError("maturity_age must be non-negative")
        if self.maturity_age >= self.max_age:
            raise ValueError("maturity_age must be less than max_age")

    @classmethod
    def from_dict(cls, params: dict) -> SimulationParameters:
        """Create parameters from dictionary."""
        return cls(**{k: v for k, v in params.items() if hasattr(cls, k)})

    def to_dict(self) -> dict:
        """Convert parameters to dictionary."""
        from dataclasses import asdict

        return asdict(self)

    @property
    def is_homogeneous(self) -> bool:
        """Check if using homogeneous landscape."""
        return self.landscape.lower() == "homogeneous"

    @property
    def is_jasmine_mode(self) -> bool:
        """Check if overall simulation mode is JASMINE."""
        return self.simulation_mode.upper() == "JASMINE"

    def get_effective_time_mode(self) -> str:
        """Get effective time mode (subsystem override or global)."""
        return self.time_mode or self.simulation_mode

    def get_effective_movement_mode(self) -> str:
        """Movement mode: the override, else DEPONS in both modes. The JASMINE movement (a
        simplified walk; the physics integrator is not wired in) is experimental and only
        runs when asked for (movement_mode="JASMINE")."""
        return self.movement_mode or "DEPONS"

    def get_effective_fsm_mode(self) -> str:
        """Get effective FSM mode (subsystem override or global)."""
        return self.fsm_mode or self.simulation_mode

    def get_effective_energy_mode(self) -> str:
        """Energy mode: the override, else DEPONS in both modes. The JASMINE energy budget is
        uncalibrated (almost no mortality without disturbance, more deaths with ships; see the
        changelog's known issues) and only runs when asked for (energy_mode="JASMINE")."""
        return self.energy_mode or "DEPONS"

    def get_effective_memory_mode(self) -> str:
        """Get effective memory mode (subsystem override or global)."""
        return self.memory_mode or self.simulation_mode
