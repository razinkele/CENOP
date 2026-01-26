"""
Simulation parameters configuration.

All configurable model parameters with their defaults and validation.
Translates from: SimulationParameters.java (779 lines)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SimulationParameters:
    """
    All simulation parameters with defaults from DEPONS.
    
    Translates from: SimulationParameters.java and parameters.xml
    """
    
    # === Simulation Setup ===
    random_seed: Optional[int] = None
    porpoise_count: int = 10000
    tracked_porpoise_count: int = 1
    sim_years: int = 50
    landscape: str = "NorthSea"
    debug: int = 0
    
    # === Disturbance Sources ===
    turbines: str = "off"
    ships_enabled: bool = False
    
    # === Dispersal ===
    dispersal: str = "PSM-Type2"
    t_disp: int = 3                    # Days of declining energy before dispersal
    psm_log: float = 0.6               # Logistic increase in random turning
    psm_dist_mean: float = 300.0       # Preferred dispersal distance (km)
    psm_dist_sd: float = 100.0
    psm_tol: float = 5.0               # Tolerance band (km)
    psm_angle: float = 20.0            # Max turning angle after PSM step
    
    # === Memory ===
    r_s: float = 0.04                  # Satiation memory decay rate
    r_r: float = 0.04                  # Reference memory decay rate
    r_u: float = 0.1                   # Food replenishment rate
    
    # === Movement ===
    inertia_const: float = 0.001       # k: tendency to keep moving with CRW
    corr_logmov_length: float = 0.35   # a0: autoregressive for log10(d/100)
    corr_logmov_bathy: float = 0.0005  # a1: depth effect on log10(d/100)
    corr_logmov_salinity: float = -0.02  # a2: salinity effect on log10(d/100)
    corr_angle_base: float = -0.024    # b0: autoregressive for turning angle
    corr_angle_bathy: float = -0.008   # b1: depth effect on turning angle
    corr_angle_salinity: float = 0.93  # b2: salinity effect on turning angle
    corr_angle_base_sd: float = -14.0  # b3: intercept for turning angle
    mean_disp_dist: float = 1.05       # Dispersal distance per step (km)
    max_mov: float = 1.73              # Max movement distance (km)
    
    # === Random Components (TRACE Table 2) ===
    r1_mean: float = 1.25              # R1 mean for step length N(μ, σ)
    r1_sd: float = 0.15                # R1 SD for step length
    r2_mean: float = 0.0               # R2 mean for turning angle N(μ, σ)
    r2_sd: float = 4.0                 # R2 SD for turning angle (degrees)
    m: float = 5.495409  # 10^0.74 - limit for when turning angles stop decreasing with speed
    
    # === Energetics ===
    e_use_per_30_min: float = 4.5      # Energy use per half-hour step
    e_lact: float = 1.4                # Lactation energy multiplier
    e_warm: float = 1.3                # Warm water energy multiplier
    energy_init_mean: float = 10.0     # Initial energy N(mean, sd)
    energy_init_sd: float = 1.0
    
    # === Deterrence ===
    deter_coeff: float = 0.07          # c: deterrence coefficient
    deter_threshold: float = 158.0     # RT: minimum received level (dB) - Java default
    deter_decay: float = 50.0          # Psi_deter: decay rate (%)
    deter_time: int = 5                # tdeter: deterrence duration (steps) - Java default
    deter_max_distance: float = 50.0   # Max deterrence distance (km) - Java default 50*1000m
    deter_min_distance_ships: float = 0.1  # Min deterrence distance for ships (km)

    # Probabilistic deterrence response
    deter_probabilistic: bool = True  # Use sigmoid-based probability instead of binary threshold
    deter_response_slope: float = 0.2  # Steepness (per dB) of logistic response function

    # === Social communication (new feature) ===
    communication_enabled: bool = True          # Enable social calling and cohesion
    communication_range_km: float = 10.0        # Communication detection range (km)
    communication_source_level: float = 160.0   # Source level (dB re 1 µPa) of porpoise calls
    communication_threshold: float = 120.0      # RL for 50% detection probability (dB)
    communication_response_slope: float = 0.2   # Steepness of detection logistic
    social_weight: float = 0.3                  # Weight [0-1] of social attraction influence

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
    alpha_hat: float = 0.0             # Absorption coefficient
    beta_hat: float = 20.0             # Spreading loss factor
    
    # === Reproduction ===
    conceive_prob: float = 0.68        # h: probability of becoming pregnant
    gestation_time: int = 300          # tgest: gestation time (days)
    nursing_time: int = 240            # tnurs: nursing time (days)
    mating_day_mean: float = 225.0     # Mean mating day (day of year)
    mating_day_sd: float = 20.0
    
    # === Life History ===
    max_age: float = 30.0              # Maximum age (years)
    maturity_age: float = 3.44         # Age of maturity (years) - DEPONS default
    max_breeding_age: float = 20.0     # Maximum breeding age (years)

    # === Environment ===
    min_depth: float = 1.0             # wmin: minimum water depth (m)
    min_depth_dispersal: float = 4.0   # wdisp: minimum depth when dispersing (m)

    # === Survival/Mortality ===
    # Starvation mortality formula: yearlySurvProb = 1 - (m_mort_prob_const * exp(-energy * x_survival_const))
    m_mort_prob_const: float = 0.5     # M_MORT_PROB_CONST in DEPONS
    x_survival_const: float = 0.15     # xSurvivalProbConst in DEPONS
    # Age-dependent annual mortality rates
    mortality_juvenile: float = 0.15   # Annual mortality for age < 1 year
    mortality_adult: float = 0.05      # Annual mortality for 1 <= age <= 20
    mortality_elderly: float = 0.15    # Annual mortality for age > 20
    # Bycatch
    bycatch_prob: float = 0.0          # Annual bycatch probability
    
    # === Food ===
    u_min: float = 0.001               # Minimum food level in patch
    
    # === Landscape ===
    wrap_border_homo: bool = True      # Wrap border for homogeneous landscape
    world_width: int = 1000            # Grid width (for homogeneous)
    world_height: int = 1000           # Grid height (for homogeneous)
    
    # === Model Version ===
    model: int = 4                     # Model version (affects behavior) - DEPONS hardcodes to 4
    
    def __post_init__(self):
        """Validate parameters."""
        self._validate()
        
    def _validate(self) -> None:
        """Validate parameter ranges."""
        if self.porpoise_count < 0:
            raise ValueError("porpoise_count must be non-negative")
        if self.sim_years < 1:
            raise ValueError("sim_years must be at least 1")
        if self.max_age <= 0:
            raise ValueError("max_age must be positive")
        if not 0 <= self.conceive_prob <= 1:
            raise ValueError("conceive_prob must be between 0 and 1")
            
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
