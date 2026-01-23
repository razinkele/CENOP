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
    distance_m: float,
    alpha_hat: float = 0.0,
    beta_hat: float = 20.0
) -> float:
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
        distance_m: Distance from source in meters
        alpha_hat: Absorption coefficient (dB/m)
        beta_hat: Spreading loss factor
        
    Returns:
        Transmission loss in dB
    """
    if distance_m <= 0:
        return 0.0
    
    # Avoid log(0) - use 1m as minimum
    distance_m = max(1.0, distance_m)
    
    # TL = β * log10(r) + α * r
    # DEPONS uses distance in meters directly with alpha in dB/m
    spreading_loss = beta_hat * np.log10(distance_m)
    absorption_loss = alpha_hat * distance_m  # No conversion, alpha is dB/m
    
    return spreading_loss + absorption_loss


def calculate_received_level(
    source_level: float,
    distance_m: float,
    alpha_hat: float = 0.0,
    beta_hat: float = 20.0
) -> float:
    """
    Calculate received sound level at a given distance.
    
    RL = SL - TL
    
    Args:
        source_level: Source level in dB re 1 µPa @ 1m
        distance_m: Distance from source in meters
        alpha_hat: Absorption coefficient
        beta_hat: Spreading loss factor
        
    Returns:
        Received level in dB re 1 µPa
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
    
    # Base source level (dB re 1 µPa @ 1m)
    base_source_level: float = 175.0
    
    # Vessel length (meters) - affects source level
    length: float = 100.0
    
    # Vessel speed (knots) - affects source level
    speed: float = 12.0
    
    # VHF frequency weighting for porpoise hearing
    vhf_weighting: float = -10.0  # Adjustment for high frequency
    
    def get_source_level(self) -> float:
        """
        Calculate source level based on vessel characteristics.
        
        Simplified JOMOPANS model:
        SL = SL_base + 60*log10(L/100) + 20*log10(v/12)
        """
        # Length correction (reference 100m)
        length_correction = 60 * np.log10(self.length / 100.0) if self.length > 0 else 0
        
        # Speed correction (reference 12 knots)
        speed_correction = 20 * np.log10(self.speed / 12.0) if self.speed > 0 else 0
        
        return self.base_source_level + length_correction + speed_correction + self.vhf_weighting


class ShipDeterrenceModel:
    """
    Ship deterrence probability and magnitude model.
    
    Based on DEPONS ship deterrence equations with day/night variation.
    Translates from: Ship.java deterrence calculations
    """
    
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
    ) -> float:
        """
        Calculate probability of deterrence response.
        
        Uses logistic model:
        P = exp(linear) / (1 + exp(linear))
        
        where linear = intercept + noise*SPL + dist*D + noise_x_dist*SPL*D
        
        Args:
            spl: Sound pressure level at porpoise location (dB)
            distance_km: Distance from ship (km)
            is_day: True for daytime, False for nighttime
            
        Returns:
            Probability of deterrence response (0-1)
        """
        if is_day:
            linear = (
                self.pship_int_day +
                self.pship_noise_day * spl +
                self.pship_dist_day * distance_km +
                self.pship_dist_x_noise_day * spl * distance_km
            )
        else:
            linear = (
                self.pship_int_night +
                self.pship_noise_night * spl +
                self.pship_dist_night * distance_km +
                self.pship_dist_x_noise_night * spl * distance_km
            )
            
        # Logistic function with overflow protection
        # Use scipy.special.expit or manual clipping
        linear_clipped = np.clip(linear, -500, 500)  # Prevent overflow
        prob = 1.0 / (1.0 + np.exp(-linear_clipped))  # Equivalent to exp(x)/(1+exp(x))
        return float(np.clip(prob, 0.0, 1.0))
        
    def calculate_deterrence_magnitude(
        self,
        spl: float,
        distance_km: float,
        is_day: bool = True
    ) -> float:
        """
        Calculate magnitude/strength of deterrence response.
        
        Uses linear model with coefficients.
        
        Args:
            spl: Sound pressure level at porpoise location (dB)
            distance_km: Distance from ship (km)
            is_day: True for daytime, False for nighttime
            
        Returns:
            Deterrence magnitude (arbitrary units)
        """
        if is_day:
            magnitude = (
                self.cship_int_day +
                self.cship_noise_day * spl +
                self.cship_dist_day * distance_km +
                self.cship_dist_x_noise_day * spl * distance_km
            )
        else:
            magnitude = (
                self.cship_int_night +
                self.cship_noise_night * spl +
                self.cship_dist_night * distance_km +
                self.cship_dist_x_noise_night * spl * distance_km
            )
            
        return max(0.0, magnitude)


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
    dx = porpoise_x - source_x
    dy = porpoise_y - source_y
    
    # Normalize
    distance = np.sqrt(dx**2 + dy**2)
    if distance > 0:
        dx /= distance
        dy /= distance
    
    # Scale by strength and coefficient
    return (
        strength * dx * deter_coeff,
        strength * dy * deter_coeff
    )
