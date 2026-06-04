"""
Sound propagation module.

Implements acoustic propagation models for noise from turbines and ships.
Translates from: SoundSource.java, Ship.java sound calculations
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from enum import Enum


class NoiseSourceType(Enum):
    """Type of noise source."""
    TURBINE_CONSTRUCTION = "turbine_construction"
    TURBINE_OPERATION = "turbine_operation"
    SHIP = "ship"


@dataclass
class SoundPropagationParams:
    """Parameters for sound propagation calculations."""
    
    # Sound propagation
    alpha_hat: float = 0.0      # Absorption coefficient (dB/km)
    beta_hat: float = 20.0      # Spreading loss factor (spherical = 20)
    
    # Deterrence thresholds
    response_threshold: float = 152.9  # RT: minimum level to cause response (dB re 1 µPa)
    
    # Maximum deterrence distance
    max_deter_distance: float = 50.0  # km
    
    # Ship-specific
    min_deter_distance_ships: float = 0.1  # km (100m minimum)


def calculate_transmission_loss(
    distance_m: float | np.ndarray,
    alpha_hat: float = 0.0,
    beta_hat: float = 20.0
) -> float | np.ndarray:
    """
    Calculate transmission loss (TL) for sound propagation.

    Uses the practical spreading model from DEPONS:
    TL = β * log10(r) + α * r

    where:
    - β (beta_hat) is the spreading loss factor (20 for spherical, 10 for cylindrical)
    - α (alpha_hat) is the absorption coefficient (dB/m)
    - r is the distance in meters

    DEPONS Java (Turbine.java line 225):
    betaHat * Math.log10(distToTurb) + alphaHat * distToTurb

    Args:
        distance_m: Distance from source in meters (scalar or array)
        alpha_hat: Absorption coefficient (dB/m)
        beta_hat: Spreading loss factor

    Returns:
        Transmission loss in dB (same shape as input)
    """
    # Convert to numpy array for uniform handling
    dist = np.asarray(distance_m)
    is_scalar = dist.ndim == 0

    # Avoid log(0) - use 1m as minimum, handle zeros
    dist_safe = np.maximum(dist, 1.0)

    # TL = β * log10(r) + α * r
    # DEPONS uses distance in meters directly with alpha in dB/m
    spreading_loss = beta_hat * np.log10(dist_safe)
    absorption_loss = alpha_hat * dist_safe

    result = spreading_loss + absorption_loss

    # For zero/negative distances, return 0
    result = np.where(dist <= 0, 0.0, result)

    # Return scalar if input was scalar
    if is_scalar:
        return float(result)
    return result


def calculate_received_level(
    source_level: float,
    distance_m: float | np.ndarray,
    alpha_hat: float = 0.0,
    beta_hat: float = 20.0
) -> float | np.ndarray:
    """
    Calculate received sound level at a given distance.

    RL = SL - TL

    Args:
        source_level: Source level in dB re 1 µPa @ 1m
        distance_m: Distance from source in meters (scalar or array)
        alpha_hat: Absorption coefficient
        beta_hat: Spreading loss factor

    Returns:
        Received level in dB re 1 µPa (same shape as distance_m)
    """
    tl = calculate_transmission_loss(distance_m, alpha_hat, beta_hat)
    return source_level - tl


def calculate_deterrence_distance(
    source_level: float,
    response_threshold: float,
    alpha_hat: float = 0.0,
    beta_hat: float = 20.0,
    max_distance: float = 50000.0  # 50 km in meters
) -> float:
    """
    Calculate the distance at which received level equals response threshold.
    
    Solves: RT = SL - TL for distance
    
    For spherical spreading (β=20) without absorption:
    r = 10^((SL - RT) / 20)
    
    Args:
        source_level: Source level in dB
        response_threshold: Response threshold in dB
        alpha_hat: Absorption coefficient
        beta_hat: Spreading loss factor
        max_distance: Maximum distance to consider (meters)
        
    Returns:
        Distance in meters where RL = RT
    """
    if source_level <= response_threshold:
        return 0.0
    
    # For simple case without absorption
    if alpha_hat == 0:
        distance = 10 ** ((source_level - response_threshold) / beta_hat)
        return min(distance, max_distance)
    
    # With absorption, use iterative approach
    # Binary search for distance
    low, high = 1.0, max_distance
    
    for _ in range(50):  # Max iterations
        mid = (low + high) / 2
        rl = calculate_received_level(source_level, mid, alpha_hat, beta_hat)
        
        if abs(rl - response_threshold) < 0.1:
            return mid
        elif rl > response_threshold:
            low = mid
        else:
            high = mid
            
    return (low + high) / 2


@dataclass
class TurbineNoise:
    """
    Turbine noise characteristics.
    
    Based on DEPONS turbine deterrence model.
    """
    
    # Source level for pile driving (construction)
    # Typical values: 180-220 dB re 1 µPa @ 1m
    source_level_construction: float = 200.0
    
    # Source level for operational turbine (much lower)
    source_level_operation: float = 145.0
    
    # Impact factor (relative to reference Roedsand turbine)
    impact: float = 1.0
    
    def get_source_level(self, is_construction: bool = True) -> float:
        """Get effective source level including impact factor."""
        base_level = (
            self.source_level_construction if is_construction 
            else self.source_level_operation
        )
        # Impact modifies the effective source level
        # impact > 1 means louder, impact < 1 means quieter
        return base_level + 10 * np.log10(self.impact) if self.impact > 0 else base_level


@dataclass  
class ShipNoise:
    """
    Ship noise characteristics.
    
    Based on JOMOPANS model used in DEPONS.
    Ship noise depends on vessel type, length, and speed.
    """
    
    # Explicit source-level override (dB re 1 µPa @ 1m). When None, SL is computed
    # from the calibrated JOMOPANS model. Set by ships.json `impact` or by tests.
    base_source_level: float = None

    # Vessel class — drives the JOMOPANS source-level model (set from Ship.vessel_type).
    vessel_class: object = None

    # Vessel length (m) and speed (knots) — JOMOPANS inputs.
    length: float = 100.0
    speed: float = 12.0

    def get_source_level(self) -> float:
        """Source level (dB re 1 µPa @ 1m).

        Returns the explicit override if set, else the calibrated JOMOPANS
        decidecade band-12 SL (DEPONS Ship.java:286 / JOMOPANS_BAND=12).
        """
        if self.base_source_level is not None:
            return self.base_source_level
        # Lazy import breaks the sound -> jomopans -> ship -> sound module cycle.
        from cenop.behavior.jomopans_spl import jomopans_spl
        return jomopans_spl(self.vessel_class, self.speed, self.length, band=12)


class ShipDeterrenceModel:
    """
    Ship deterrence probability and magnitude model.

    Based on DEPONS 3.2 ship deterrence equations with day/night variation.
    Inputs are standardized before applying model coefficients (Java Ship.java:349-398).
    Translates from: Ship.java deterrence calculations
    """

    # Standardization constants — full precision (Java Ship.java:349-398)
    STD_PROB_DAY = {'dist_mean': 5.801812, 'dist_sd': 2.602801,
                    'noise_mean': 65.95304, 'noise_sd': 18.25469}
    STD_PROB_NIGHT = {'dist_mean': 6.243703, 'dist_sd': 2.548173,
                      'noise_mean': 68.9993, 'noise_sd': 14.81663}
    STD_MAG_DAY = {'dist_mean': 5.311561, 'dist_sd': 2.698996,
                   'noise_mean': 69.28605, 'noise_sd': 17.09946}
    STD_MAG_NIGHT = {'dist_mean': 6.442084, 'dist_sd': 2.48903,
                     'noise_mean': 68.86555, 'noise_sd': 15.09977}

    def __init__(
        self,
        # Day coefficients - probability
        pship_int_day: float = -3.0569351,
        pship_noise_day: float = 0.2172813,
        pship_dist_day: float = -0.1303880,
        pship_dist_x_noise_day: float = 0.0293443,
        # Night coefficients - probability  
        pship_int_night: float = -3.233771,
        pship_noise_night: float = 0.0,
        pship_dist_night: float = 0.085242,
        pship_dist_x_noise_night: float = 0.0,
        # Day coefficients - magnitude
        cship_int_day: float = 2.9647996,
        cship_noise_day: float = 0.0472709,
        cship_dist_day: float = -0.0355541,
        cship_dist_x_noise_day: float = 0.0,
        # Night coefficients - magnitude
        cship_int_night: float = 2.7543376,
        cship_noise_night: float = 0.0,
        cship_dist_night: float = 0.0284629,
        cship_dist_x_noise_night: float = 0.0
    ):
        # Probability coefficients
        self.pship_int_day = pship_int_day
        self.pship_noise_day = pship_noise_day
        self.pship_dist_day = pship_dist_day
        self.pship_dist_x_noise_day = pship_dist_x_noise_day
        
        self.pship_int_night = pship_int_night
        self.pship_noise_night = pship_noise_night
        self.pship_dist_night = pship_dist_night
        self.pship_dist_x_noise_night = pship_dist_x_noise_night
        
        # Magnitude coefficients
        self.cship_int_day = cship_int_day
        self.cship_noise_day = cship_noise_day
        self.cship_dist_day = cship_dist_day
        self.cship_dist_x_noise_day = cship_dist_x_noise_day
        
        self.cship_int_night = cship_int_night
        self.cship_noise_night = cship_noise_night
        self.cship_dist_night = cship_dist_night
        self.cship_dist_x_noise_night = cship_dist_x_noise_night
        
    def calculate_deterrence_probability(
        self,
        spl: float,
        distance_km: float,
        is_day: bool = True
    ):
        """Calculate probability of deterrence response.

        Inputs are standardized using dataset means/SDs (Java Ship.java:349-398).
        """
        if is_day:
            std = self.STD_PROB_DAY
            linear = (
                self.pship_int_day +
                self.pship_noise_day * ((spl - std['noise_mean']) / std['noise_sd']) +
                self.pship_dist_day * ((distance_km - std['dist_mean']) / std['dist_sd']) +
                self.pship_dist_x_noise_day * ((spl - std['noise_mean']) / std['noise_sd']) *
                    ((distance_km - std['dist_mean']) / std['dist_sd'])
            )
        else:
            std = self.STD_PROB_NIGHT
            linear = (
                self.pship_int_night +
                self.pship_noise_night * ((spl - std['noise_mean']) / std['noise_sd']) +
                self.pship_dist_night * ((distance_km - std['dist_mean']) / std['dist_sd']) +
                self.pship_dist_x_noise_night * ((spl - std['noise_mean']) / std['noise_sd']) *
                    ((distance_km - std['dist_mean']) / std['dist_sd'])
            )

        linear_clipped = np.clip(linear, -500, 500)
        prob = 1.0 / (1.0 + np.exp(-linear_clipped))
        return np.clip(prob, 0.0, 1.0)
        
    def calculate_deterrence_magnitude(
        self,
        spl: float,
        distance_km: float,
        is_day: bool = True
    ):
        """Calculate deterrence magnitude with standardized inputs."""
        if is_day:
            std = self.STD_MAG_DAY
            magnitude = (
                self.cship_int_day +
                self.cship_noise_day * ((spl - std['noise_mean']) / std['noise_sd']) +
                self.cship_dist_day * ((distance_km - std['dist_mean']) / std['dist_sd']) +
                self.cship_dist_x_noise_day * ((spl - std['noise_mean']) / std['noise_sd']) *
                    ((distance_km - std['dist_mean']) / std['dist_sd'])
            )
        else:
            std = self.STD_MAG_NIGHT
            magnitude = (
                self.cship_int_night +
                self.cship_noise_night * ((spl - std['noise_mean']) / std['noise_sd']) +
                self.cship_dist_night * ((distance_km - std['dist_mean']) / std['dist_sd']) +
                self.cship_dist_x_noise_night * ((spl - std['noise_mean']) / std['noise_sd']) *
                    ((distance_km - std['dist_mean']) / std['dist_sd'])
            )
        return np.exp(np.clip(magnitude, -50.0, 50.0))

    def deterrence_components(
        self,
        rl: np.ndarray,
        dist_m: np.ndarray,
        grid_dx: np.ndarray,
        grid_dy: np.ndarray,
        is_day: bool,
        u_draw: np.ndarray,
        tships: float,
    ):
        """DEPONS ship deterrence per porpoise for ONE ship (vectorized).

        Args (all arrays shape (N,) except is_day/tships):
            rl       received level (dB), already clamped >= 0
            dist_m   porpoise<->ship distance (m), already clamped >= 1
            grid_dx  (porpoise_x - ship_x) in GRID/cell units
            grid_dy  (porpoise_y - ship_y) in GRID/cell units
            u_draw   uniform(0,1) draws for the Bernoulli reaction
            tships   minimum RL (dB) to react (deter_ships_min_db)

        Returns (vx, vy, prob, mag, react) arrays. Vector is
        DEPONS unit-vector (grid displacement / metre distance) x magnitude,
        zeroed where the porpoise does not react. No deter_coeff (turbine-only).
        """
        rl = np.asarray(rl, dtype=np.float64)
        dist_m = np.asarray(dist_m, dtype=np.float64)
        dist_km = dist_m / 1000.0
        prob = np.asarray(
            self.calculate_deterrence_probability(rl, dist_km, is_day), dtype=np.float64
        )
        mag = np.asarray(
            self.calculate_deterrence_magnitude(rl, dist_km, is_day), dtype=np.float64
        )
        gate = rl > tships
        react = gate & (np.asarray(u_draw, dtype=np.float64) < prob)
        eff_mag = np.where(react, mag, 0.0)
        vx = (np.asarray(grid_dx, dtype=np.float64) / dist_m) * eff_mag
        vy = (np.asarray(grid_dy, dtype=np.float64) / dist_m) * eff_mag
        return vx, vy, prob, mag, react


def response_probability_from_rl(
    received_level: float | np.ndarray,
    threshold: float = 152.9,
    slope: float = 0.5
) -> float | np.ndarray:
    """
    Calculate response probability from received level using logistic function.

    Uses a sigmoid/logistic curve centered at the threshold:
    P = 1 / (1 + exp(-slope * (RL - threshold)))

    This models the probability that a porpoise will respond to a sound
    at a given received level.

    Args:
        received_level: Received sound level in dB (scalar or array)
        threshold: Threshold level at which P=0.5 (dB)
        slope: Steepness of the sigmoid curve (dB^-1)

    Returns:
        Response probability (0-1), same shape as input
    """
    # Calculate the linear predictor
    linear = slope * (received_level - threshold)

    # Apply logistic function with overflow protection
    linear_clipped = np.clip(linear, -500, 500)
    prob = 1.0 / (1.0 + np.exp(-linear_clipped))

    return np.clip(prob, 0.0, 1.0)


def calculate_deterrence_vector(
    porpoise_x: float,
    porpoise_y: float,
    source_x: float,
    source_y: float,
    strength: float,
    deter_coeff: float = 0.07
) -> Tuple[float, float]:
    """
    Calculate deterrence vector pointing away from noise source.
    
    Args:
        porpoise_x, porpoise_y: Porpoise position
        source_x, source_y: Noise source position
        strength: Deterrence strength/magnitude
        deter_coeff: Deterrence coefficient (c parameter)
        
    Returns:
        (dx, dy) deterrence vector components
    """
    # Vector from source to porpoise (away from source)
    # DEPONS 3.2 (Porpoise.java:1290-1292): raw displacement, NO normalization.
    # The magnitude encodes distance — farther porpoises get larger displacement.
    dx = porpoise_x - source_x
    dy = porpoise_y - source_y

    # Scale by strength and coefficient (no division by distance)
    return (
        strength * dx * deter_coeff,
        strength * dy * deter_coeff
    )


# Backwards compatibility helper ------------------------------------------------

def combine_rls(rl_arrays: list | np.ndarray) -> np.ndarray:
    """Combine multiple received-level arrays into a single received level per
    porpoise. Uses the maximum (dominant) received level in dB across sources.

    Args:
        rl_arrays: Iterable of 1-D arrays (shape: (N,)) or a 2-D array (n_sources, N)

    Returns:
        1-D array (N,) with combined received levels (dB)
    """
    arr = np.asarray(rl_arrays)
    if arr.ndim == 1:
        return arr.copy()
    # If shape (n_sources, N) take max across sources
    return np.max(arr, axis=0)


@dataclass
class HydrophoneRecord:
    """Record of loudest sound received in a tick."""
    ship_name: str = ""
    ship_utm_x: float = -1.0
    ship_utm_y: float = -1.0
    received_level: float = 0.0
    source_level: float = 0.0


class Hydrophone:
    """
    Passive acoustic monitoring station.

    Tracks the loudest sound received per tick from ships/turbines.
    Translates from: Hydrophone.java

    Each tick, ships report their sound level to each hydrophone.
    The hydrophone keeps only the loudest source. At the end of the tick
    the record can be queried and then reset for the next tick.

    Args:
        name: Hydrophone identifier
        x: Grid x position
        y: Grid y position
        utm_x: UTM x coordinate (for reporting)
        utm_y: UTM y coordinate (for reporting)
    """

    def __init__(
        self, name: str, x: float, y: float,
        utm_x: float = 0.0, utm_y: float = 0.0,
    ):
        self.name = name
        self.x = x
        self.y = y
        self.utm_x = utm_x
        self.utm_y = utm_y
        self._record = HydrophoneRecord()

    def receive_sound_level(
        self,
        ship_name: str,
        ship_utm_x: float,
        ship_utm_y: float,
        source_level: float,
        received_level: float,
    ) -> None:
        """
        Report a sound level to this hydrophone.

        Only keeps the record if it is louder than the current max.
        """
        if received_level > self._record.received_level:
            self._record = HydrophoneRecord(
                ship_name=ship_name,
                ship_utm_x=ship_utm_x,
                ship_utm_y=ship_utm_y,
                received_level=received_level,
                source_level=source_level,
            )

    def reset(self) -> None:
        """Reset sound level record for next tick."""
        self._record = HydrophoneRecord()

    @property
    def record(self) -> HydrophoneRecord:
        """Current tick's loudest sound record."""
        return self._record

    @property
    def received_level(self) -> float:
        return self._record.received_level

    @property
    def source_level(self) -> float:
        return self._record.source_level
