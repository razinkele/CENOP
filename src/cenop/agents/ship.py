"""
Ship agent representing vessel traffic.

Ships move along predefined routes and create noise that can deter porpoises.
Translates from: Ship.java (417 lines) and related classes
"""

from __future__ import annotations

import json
import logging
import numpy as np
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List, Optional, Tuple
from enum import Enum
from pathlib import Path

logger = logging.getLogger("CENOP")

from cenop.agents.base import Agent
from cenop.behavior.sound import (
    ShipNoise,
    ShipDeterrenceModel,
    calculate_received_level,
    calculate_deterrence_vector,
    response_probability_from_rl,
)
from cenop.behavior.weston_flux import weston_flux_tl

if TYPE_CHECKING:
    from cenop.parameters.simulation_params import SimulationParameters
    from cenop.landscape.cell_data import CellData


class VesselClass(Enum):
    """Types of vessels with different noise characteristics.

    Extended to 13 JOMOPANS classes (DEPONS 3.2).
    """
    BULKER = "bulker"
    CARGO = "cargo"                    # Maps to CONTAINERSHIP in JOMOPANS
    CHEMICAL_TANKER = "chemical_tanker"
    CONTAINER = "container"            # = Java CONTAINERSHIP
    CRUISE = "cruise"
    DREDGER = "dredger"
    FISHING = "fishing"
    GOVERNMENT = "government"          # = Java GOVERNMENT_RESEARCH
    NAVAL = "naval"
    PASSENGER = "passenger"
    RECREATIONAL = "recreational"
    TANKER = "tanker"
    TUG = "tug"
    VEHICLE_CARRIER = "vehicle_carrier"
    OTHER = "other"


# Base source levels by vessel class (dB re 1 µPa @ 1m) — fallback when JOMOPANS not used
VESSEL_BASE_LEVELS = {
    VesselClass.BULKER: 177.0,
    VesselClass.CARGO: 175.0,
    VesselClass.CHEMICAL_TANKER: 177.0,
    VesselClass.CONTAINER: 175.0,
    VesselClass.CRUISE: 172.0,
    VesselClass.DREDGER: 170.0,
    VesselClass.FISHING: 165.0,
    VesselClass.GOVERNMENT: 168.0,
    VesselClass.NAVAL: 170.0,
    VesselClass.PASSENGER: 172.0,
    VesselClass.RECREATIONAL: 168.0,
    VesselClass.TANKER: 177.0,
    VesselClass.TUG: 170.0,
    VesselClass.VEHICLE_CARRIER: 175.0,
    VesselClass.OTHER: 168.0,
}


@dataclass
class Buoy:
    """A waypoint along a ship's route."""
    
    x: float
    y: float
    speed: float = 10.0    # knots
    pause_ticks: int = 0   # ticks to pause at this buoy


@dataclass
class Route:
    """A ship route consisting of buoys (waypoints)."""
    
    name: str = ""
    buoys: List[Buoy] = field(default_factory=list)
    
    def get_buoy(self, index: int) -> Optional[Buoy]:
        """Get buoy at index."""
        if 0 <= index < len(self.buoys):
            return self.buoys[index]
        return None
        
    @property
    def length(self) -> int:
        """Number of buoys in route."""
        return len(self.buoys)


@dataclass
class Ship(Agent):
    """
    Ship agent representing a vessel producing noise.
    
    Ships move along routes between buoys and produce noise
    that can deter porpoises using day/night probability models.
    
    Translates from: Ship.java
    """
    
    # Identification
    name: str = ""
    
    # Vessel characteristics
    vessel_type: VesselClass = VesselClass.OTHER
    vessel_length: float = 100.0  # meters
    
    # Timing
    tick_start: int = 0
    tick_end: int = 2147483647
    
    # Route
    route: Route = field(default_factory=Route)
    current_buoy_idx: int = 0
    ticks_paused: int = 0
    
    # Current state
    current_speed: float = 10.0  # knots
    _is_active: bool = False
    
    # Noise model
    noise: ShipNoise = field(default_factory=ShipNoise)
    
    # Deterrence model
    deterrence_model: ShipDeterrenceModel = field(default_factory=ShipDeterrenceModel)
    
    def __post_init__(self):
        """Initialize noise model with vessel characteristics."""
        base_level = VESSEL_BASE_LEVELS.get(self.vessel_type, 168.0)
        self.noise = ShipNoise(
            base_source_level=base_level,
            length=self.vessel_length,
            speed=self.current_speed
        )
    
    def is_active(self, tick: Optional[int] = None) -> bool:
        """Check if ship is present at given tick."""
        if tick is not None:
            return self.tick_start <= tick < self.tick_end
        return self._is_active
        
    def update(self, current_tick: int) -> None:
        """
        Update ship position and status for current tick.
        
        Args:
            current_tick: Current simulation tick
        """
        # Check if active
        self._is_active = self.tick_start <= current_tick < self.tick_end
        
        if not self._is_active or not self.route.buoys:
            return
            
        # Handle pausing at buoy
        if self.ticks_paused > 0:
            self.ticks_paused -= 1
            return
            
        # Get current and next buoy
        current_buoy = self.route.get_buoy(self.current_buoy_idx)
        if self.route.length == 0:
            return
        next_idx = (self.current_buoy_idx + 1) % self.route.length
        next_buoy = self.route.get_buoy(next_idx)
        
        if current_buoy is None or next_buoy is None:
            return
            
        # Calculate movement towards next buoy
        dx = next_buoy.x - self.x
        dy = next_buoy.y - self.y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Speed in grid cells per tick (knots -> cells/30min)
        # 1 knot = 1.852 km/h = 0.926 km/30min
        # cell_size = 400m = 0.4km
        speed_cells = current_buoy.speed * 1.852 * 0.5 / 0.4
        
        if distance <= speed_cells:
            # Arrived at next buoy
            self.x = next_buoy.x
            self.y = next_buoy.y
            self.current_buoy_idx = next_idx
            self.ticks_paused = next_buoy.pause_ticks
            self.current_speed = next_buoy.speed
        else:
            # Move towards next buoy
            ratio = speed_cells / distance
            self.x += dx * ratio
            self.y += dy * ratio
            
        # Update heading
        if distance > 0:
            self.heading = np.degrees(np.arctan2(dx, dy))
            
        # Update noise model with current speed
        self.noise.speed = self.current_speed
        
    def get_source_level(self) -> float:
        """Get current source level."""
        return self.noise.get_source_level()
        
    def get_received_level(
        self,
        porpoise_x: float,
        porpoise_y: float,
        alpha: float = 0.0,
        beta: float = 20.0,
        cell_size: float = 400.0
    ) -> float:
        """
        Calculate received sound level at porpoise position.
        
        Args:
            porpoise_x, porpoise_y: Porpoise position
            alpha: Absorption coefficient
            beta: Spreading loss factor
            cell_size: Cell size in meters
            
        Returns:
            Received level in dB
        """
        dx = (porpoise_x - self.x) * cell_size
        dy = (porpoise_y - self.y) * cell_size
        distance_m = np.sqrt(dx**2 + dy**2)
        
        if distance_m < 1.0:
            distance_m = 1.0
            
        return calculate_received_level(
            self.get_source_level(),
            distance_m,
            alpha,
            beta
        )
        
    def calculate_deterrence(
        self,
        porpoise_x: float,
        porpoise_y: float,
        params: SimulationParameters,
        is_day: bool = True,
        cell_size: float = 400.0
    ) -> Tuple[bool, float, float, float]:
        """
        Calculate deterrence effect on a porpoise.
        
        Uses probabilistic day/night deterrence model from DEPONS.
        
        Args:
            porpoise_x, porpoise_y: Porpoise position
            params: Simulation parameters
            is_day: True for daytime, False for nighttime
            cell_size: Cell size in meters
            
        Returns:
            (should_deter, probability, magnitude, distance_km)
        """
        if not self._is_active:
            return (False, 0.0, 0.0, 0.0)
            
        # Calculate distance
        dx = (porpoise_x - self.x) * cell_size
        dy = (porpoise_y - self.y) * cell_size
        distance_m = np.sqrt(dx**2 + dy**2)
        distance_km = distance_m / 1000.0
        
        # Check min/max distances
        max_dist_km = min(10.0, params.deter_max_distance)  # Ship max is 10km
        min_dist_km = params.deter_min_distance_ships
        
        if distance_km > max_dist_km or distance_km < min_dist_km:
            return (False, 0.0, 0.0, distance_km)
            
        # Calculate received level
        wf_depth = getattr(params, 'weston_flux_depth', None)
        if wf_depth is not None:
            sl = self.get_source_level()
            tl = weston_flux_tl(distance_m, wf_depth,
                                params.weston_flux_grain_size,
                                params.weston_flux_temperature,
                                params.weston_flux_salinity)
            spl = sl - tl
        else:
            spl = self.get_received_level(
                porpoise_x, porpoise_y,
                params.alpha_hat, params.beta_hat, cell_size
            )

        # Tships gate: skip deterrence below minimum RL (Java Ship.java:228)
        tships = getattr(params, 'deter_ships_min_db', 70.0)
        if spl <= tships:
            return (False, 0.0, 0.0, distance_km)

        # Calculate deterrence probability
        prob = self.deterrence_model.calculate_deterrence_probability(
            spl, distance_km, is_day
        )
        
        # Probabilistic response
        should_deter = np.random.random() < prob
        
        if not should_deter:
            return (False, prob, 0.0, distance_km)
            
        # Calculate deterrence magnitude
        magnitude = self.deterrence_model.calculate_deterrence_magnitude(
            spl, distance_km, is_day
        )
        
        return (True, prob, magnitude, distance_km)
        
    def get_deterrence_vector(
        self,
        porpoise_x: float,
        porpoise_y: float,
        strength: float,
        deter_coeff: float = 0.07
    ) -> Tuple[float, float]:
        """Calculate deterrence vector for a porpoise."""
        return calculate_deterrence_vector(
            porpoise_x, porpoise_y,
            self.x, self.y,
            strength, deter_coeff
        )


class ShipManager:
    """
    Manages all ships in the simulation.
    
    Handles ship movement, activation, and deterrence calculations.
    """
    
    def __init__(self, ships: Optional[List[Ship]] = None):
        self.ships: List[Ship] = ships or []
        self.enabled: bool = False
        
    def set_enabled(self, enabled: bool) -> None:
        """Enable or disable ship traffic."""
        self.enabled = enabled
        
    def update(self, current_tick: int) -> None:
        """Update all ships for the current tick."""
        if not self.enabled:
            return
        for ship in self.ships:
            ship.update(current_tick)
            
    def get_active_ships(self) -> List[Ship]:
        """Get list of currently active ships."""
        if not self.enabled:
            return []
        return [s for s in self.ships if s._is_active]
        
    def calculate_aggregate_deterrence(
        self,
        porpoise_x: float,
        porpoise_y: float,
        params: SimulationParameters,
        is_day: bool = True,
        cell_size: float = 400.0
    ) -> Tuple[float, float, float]:
        """
        Calculate aggregate deterrence from all ships.
        
        Args:
            porpoise_x, porpoise_y: Porpoise position
            params: Simulation parameters
            is_day: True for daytime
            cell_size: Cell size in meters
            
        Returns:
            (max_magnitude, total_dx, total_dy)
        """
        if not self.enabled:
            return (0.0, 0.0, 0.0)
            
        max_magnitude = 0.0
        total_dx = 0.0
        total_dy = 0.0
        
        for ship in self.get_active_ships():
            should_deter, _, magnitude, _ = ship.calculate_deterrence(
                porpoise_x, porpoise_y, params, is_day, cell_size
            )
            
            if should_deter and magnitude > 0:
                dx, dy = ship.get_deterrence_vector(
                    porpoise_x, porpoise_y,
                    magnitude, params.deter_coeff
                )
                
                if magnitude > max_magnitude:
                    max_magnitude = magnitude
                    
                total_dx += dx
                total_dy += dy
                
        return (max_magnitude, total_dx, total_dy)

    def calculate_aggregate_deterrence_vectorized(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: SimulationParameters,
        is_day: bool = True,
        cell_size: float = 400.0,
        cell_data=None,
        month: int = 1,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate aggregate deterrence vector from all ships for a population.
        """
        if not self.enabled:
            zeros = np.zeros_like(porpoise_x)
            return (zeros, zeros)
            
        total_dx = np.zeros_like(porpoise_x)
        total_dy = np.zeros_like(porpoise_y)
        
        active_ships = self.get_active_ships()
        if not active_ships:
            return (total_dx, total_dy)
            
        max_dist_m = params.deter_max_distance * 1000.0
        
        for ship in active_ships:
            # Check ship probability (if day/night dependent)
            # Implemented in Ship.calculate_deterrence, but usually simpler here:
            # Ships are generally always active if active, 
            # BUT ShipDeterrenceModel uses probability to decide if a ping happens?
            # Looking at original method: ship.calculate_deterrence
            # It checks prob_response.
            
            # If the ship noise model is probabilistic, we might need to roll dice per porpoise
            # or per ship-step. Usually per ship-step.
            # Assuming deterministic propagation here for performance or 
            # simplifying to mean impact.
            
            # 1. Distances
            # ship.x is current position (updated by Ship.update)
            dx_m = (porpoise_x - ship.x) * cell_size
            dy_m = (porpoise_y - ship.y) * cell_size
            dist_sq = dx_m**2 + dy_m**2
            dist_m = np.sqrt(dist_sq)
            
            np.maximum(dist_m, 1.0, out=dist_m)
            
            # 2. Range Mask
            in_range_mask = dist_m < max_dist_m
            if not np.any(in_range_mask):
                continue
                
            # 3. Source Level
            source_level = ship.noise.get_source_level()
            
            # 4. Transmission Loss & Strength
            d_masked = dist_m[in_range_mask]
            if (
                params.weston_flux_percell
                and cell_data is not None
                and getattr(cell_data, "_sediment", None) is not None
            ):
                # Per-cell WestonFlux: look up depth/sediment/salinity
                masked_positions = np.column_stack((
                    porpoise_x[in_range_mask],
                    porpoise_y[in_range_mask],
                ))
                depths = cell_data.get_depths_vectorized(
                    masked_positions
                )
                grain_sizes = cell_data.get_sediments_vectorized(
                    masked_positions
                )
                salinities = cell_data.get_salinities_vectorized(
                    masked_positions, month
                )
                temperature = params.weston_flux_default_temperature

                # Per-porpoise TL with NODATA fallback
                tl = np.empty_like(d_masked)
                for i in range(len(d_masked)):
                    if depths[i] > 0 and grain_sizes[i] != -9999.0:
                        tl[i] = weston_flux_tl(
                            d_masked[i],
                            depths[i],
                            grain_sizes[i],
                            temperature,
                            salinities[i],
                        )
                    else:
                        # NODATA fallback: simple formula
                        tl[i] = (
                            params.beta_hat
                            * np.log10(max(d_masked[i], 1.0))
                            + params.alpha_hat * d_masked[i]
                        )
            else:
                # Simple formula (default, fully vectorized)
                tl = (
                    params.beta_hat * np.log10(d_masked)
                    + params.alpha_hat * d_masked
                )
            rl = source_level - tl
            str_val = rl - params.deter_threshold
            
            # Probabilistic deterrence: match scalar path behavior
            # Roll per-ship to decide if this ship's ping deters this tick
            prob_response = getattr(ship, 'prob_response', 1.0)
            if prob_response < 1.0 and np.random.random() > prob_response:
                continue
            
            deter_mask_local = str_val > 0
            if not np.any(deter_mask_local):
                continue
            
            # Probabilistic scaling
            if params.deter_probabilistic:
                p = response_probability_from_rl(
                    rl, params.deter_threshold, params.deter_response_slope
                )
            else:
                p = None
            
            # 5. Vectors
            full_mask = np.zeros_like(in_range_mask)
            full_mask[in_range_mask] = deter_mask_local
            
            # Apply p where appropriate
            s_final = np.zeros_like(d_masked)
            s_final[deter_mask_local] = str_val[deter_mask_local]
            if p is not None:
                p_full = np.zeros_like(d_masked)
                p_full[:] = p
                s_final = s_final * p_full
            
            # Map back to full mask arrays
            # s_final corresponds to positions in d_masked (in_range_mask true positions)
            # Build array aligned with full_mask
            strength_full = np.zeros_like(dist_m)
            strength_full[in_range_mask] = s_final
            
            s = strength_full[full_mask]
            # DEPONS 3.2: raw displacement, NOT unit vector (Porpoise.java:1290-1292)
            vec_x = dx_m[full_mask] * s * params.deter_coeff
            vec_y = dy_m[full_mask] * s * params.deter_coeff
            
            total_dx[full_mask] += vec_x
            total_dy[full_mask] += vec_y
            
        return (total_dx, total_dy)

    def ambient_received_level_at_positions(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: SimulationParameters,
        is_day: bool = True,
        cell_size: float = 400.0
    ) -> np.ndarray:
        """
        Compute ambient RL at porpoise positions from active ships.
        Returns array of RL in dB (same length as porpoise_x) or -999 if none.
        """
        if not self.enabled:
            return np.full(len(porpoise_x), -999.0, dtype=np.float32)
        active_ships = self.get_active_ships()
        if not active_ships:
            return np.full(len(porpoise_x), -999.0, dtype=np.float32)

        lin_power = np.zeros(len(porpoise_x), dtype=np.float64)
        max_dist_m = params.deter_max_distance * 1000.0
        for ship in active_ships:
            dx_m = (porpoise_x - ship.x) * cell_size
            dy_m = (porpoise_y - ship.y) * cell_size
            dist_m = np.sqrt(dx_m**2 + dy_m**2)
            dist_m = np.maximum(dist_m, 1.0)
            mask = dist_m < max_dist_m
            if not np.any(mask):
                continue
            source_level = ship.noise.get_source_level()
            dmask = dist_m[mask]
            tl = params.beta_hat * np.log10(dmask) + params.alpha_hat * dmask
            rl_mask = source_level - tl
            lin_power[mask] += 10.0 ** (rl_mask / 10.0)

        rl_combined = np.full(len(porpoise_x), -999.0, dtype=np.float32)
        nonzero = lin_power > 0
        rl_combined[nonzero] = 10.0 * np.log10(lin_power[nonzero])
        return rl_combined
        
    def load_from_file(
        self,
        routes_file: str,
        ships_file: str,
        utm_origin_x: float = 0.0,
        utm_origin_y: float = 0.0,
        cell_size: float = 400.0
    ) -> None:
        """
        Load ships and routes from files.
        
        Args:
            routes_file: Path to routes definition file
            ships_file: Path to ships definition file
            utm_origin_x, utm_origin_y: UTM origin
            cell_size: Cell size in meters
        """
        # Load routes first
        routes = self._load_routes(routes_file, utm_origin_x, utm_origin_y, cell_size)
        
        # Load ships and assign routes
        self.ships = self._load_ships(ships_file, routes)
        
    def _load_routes(
        self,
        filepath: str,
        utm_origin_x: float,
        utm_origin_y: float,
        cell_size: float
    ) -> dict:
        """Load routes from file."""
        routes = {}
        path = Path(filepath)
        
        if not path.exists():
            return routes
            
        # Parse route file format
        # (simplified - actual format may vary)
        current_route = None
        
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                if line.startswith('ROUTE'):
                    parts = line.split()
                    route_name = parts[1] if len(parts) > 1 else f"route_{len(routes)}"
                    current_route = Route(name=route_name)
                    routes[route_name] = current_route
                elif current_route is not None:
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            utm_x = float(parts[0])
                            utm_y = float(parts[1])
                            speed = float(parts[2]) if len(parts) > 2 else 10.0
                            pause = int(parts[3]) if len(parts) > 3 else 0
                        except ValueError as e:
                            logger.warning("Route file: invalid value in route '%s' (%s) — skipping waypoint",
                                         current_route.name, e)
                            continue
                        
                        grid_x = (utm_x - utm_origin_x) / cell_size
                        grid_y = (utm_y - utm_origin_y) / cell_size
                        
                        buoy = Buoy(x=grid_x, y=grid_y, speed=speed, pause_ticks=pause)
                        current_route.buoys.append(buoy)
                        
        return routes
        
    def _load_ships(self, filepath: str, routes: dict) -> List[Ship]:
        """Load ships from file."""
        ships = []
        path = Path(filepath)
        
        if not path.exists():
            return ships
            
        with open(path, 'r') as f:
            # Skip header
            next(f, None)
            
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                    
                parts = line.split()
                if len(parts) < 4:
                    continue
                    
                try:
                    name = parts[0]
                    vessel_type_str = parts[1].lower()
                    length = float(parts[2])
                    route_name = parts[3]
                except (ValueError, IndexError) as e:
                    logger.warning("Ships file line %d: invalid data (%s) — skipping", i + 2, e)
                    continue
                
                # Parse vessel type
                vessel_type = VesselClass.OTHER
                for vt in VesselClass:
                    if vt.value == vessel_type_str:
                        vessel_type = vt
                        break
                        
                # Get route
                route = routes.get(route_name, Route())
                
                # Optional timing
                try:
                    tick_start = int(parts[4]) if len(parts) > 4 else 0
                    tick_end = int(parts[5]) if len(parts) > 5 else 2147483647
                except ValueError:
                    tick_start = 0
                    tick_end = 2147483647
                
                # Initial position from first buoy
                x, y = 0.0, 0.0
                if route.buoys:
                    x = route.buoys[0].x
                    y = route.buoys[0].y
                
                ship = Ship(
                    id=i,
                    x=x,
                    y=y,
                    heading=0.0,
                    name=name,
                    vessel_type=vessel_type,
                    vessel_length=length,
                    route=route,
                    tick_start=tick_start,
                    tick_end=tick_end
                )
                ships.append(ship)
                
        return ships
        
    @property
    def count(self) -> int:
        """Number of ships."""
        return len(self.ships)
        
    @property
    def active_count(self) -> int:
        """Number of active ships."""
        return len(self.get_active_ships())
    
    def load_from_json(
        self,
        json_file: str,
        utm_origin_x: float = 3976618.0,  # Fallback UTM origin X
        utm_origin_y: float = 3363923.0,  # Fallback UTM origin Y
        cell_size: float = 400.0
    ) -> None:
        """
        Load ships and routes from DEPONS-format JSON file.
        
        The JSON format matches DEPONS ships.json:
        {
            "routes": [
                {"name": "route1", "route": [{"x": utm_x, "y": utm_y}, ...]},
                ...
            ],
            "ships": [
                {"name": "ship1", "speed": 2.5, "impact": 33.25, "start": 0, "route": "route1", ...},
                ...
            ]
        }
        
        Args:
            json_file: Path to ships.json file
            utm_origin_x: UTM X origin (XLLCORNER from bathy.asc)
            utm_origin_y: UTM Y origin (YLLCORNER from bathy.asc)
            cell_size: Cell size in meters (default 400m)
        """
        path = Path(json_file)
        if not path.exists():
            logger.warning("Ships JSON file not found: %s", json_file)
            return

        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error("Failed to parse ships JSON: %s", e)
            return
            
        # Parse routes
        routes_dict = {}
        for route_data in data.get("routes", []):
            route_name = route_data.get("name", f"route_{len(routes_dict)}")
            buoys = []
            
            for waypoint in route_data.get("route", []):
                # Convert UTM to grid coordinates
                utm_x = waypoint.get("x", 0.0)
                utm_y = waypoint.get("y", 0.0)
                
                grid_x = (utm_x - utm_origin_x) / cell_size
                grid_y = (utm_y - utm_origin_y) / cell_size
                
                # Default speed from route or ship (will be overridden per ship)
                buoy = Buoy(x=grid_x, y=grid_y, speed=10.0, pause_ticks=0)
                buoys.append(buoy)
                
            routes_dict[route_name] = Route(name=route_name, buoys=buoys)
            
        # Parse ships
        self.ships = []
        for i, ship_data in enumerate(data.get("ships", [])):
            name = ship_data.get("name", f"ship_{i}")
            speed = ship_data.get("speed", 10.0)  # knots
            impact = ship_data.get("impact", 170.0)  # dB source level
            start_tick = ship_data.get("start", 0)
            route_name = ship_data.get("route", "")
            
            # Get route
            route = routes_dict.get(route_name, Route())
            
            # Update buoy speeds from ship speed
            for buoy in route.buoys:
                buoy.speed = speed
            
            # Initial position from first buoy
            x, y = 0.0, 0.0
            if route.buoys:
                x = route.buoys[0].x
                y = route.buoys[0].y
            
            # Check for survey configuration (special ship type)
            survey = ship_data.get("survey", {})
            is_survey = bool(survey.get("point", {}).get("x"))
            
            # Determine vessel type based on name/impact
            vessel_type = VesselClass.OTHER
            if "cargo" in name.lower():
                vessel_type = VesselClass.CARGO
            elif "tanker" in name.lower():
                vessel_type = VesselClass.TANKER
            elif "survey" in name.lower() or is_survey:
                vessel_type = VesselClass.OTHER  # Survey vessels
            elif "fishing" in name.lower():
                vessel_type = VesselClass.FISHING
            
            ship = Ship(
                id=i,
                x=x,
                y=y,
                heading=0.0,
                name=name,
                vessel_type=vessel_type,
                vessel_length=100.0,  # Default length
                route=route,
                tick_start=start_tick,
                tick_end=2147483647
            )
            
            # Override source level from impact if provided
            if impact > 0:
                ship.noise.base_source_level = impact
            
            self.ships.append(ship)
            
        logger.info("Loaded %d ships with %d routes from %s", len(self.ships), len(routes_dict), json_file)
