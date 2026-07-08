"""
Ship agent representing vessel traffic.

Ships move along predefined routes and create noise that can deter porpoises.
Translates from: Ship.java (417 lines) and related classes
"""

from __future__ import annotations

import json
import logging
import math
import re
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
)
from cenop.behavior.weston_flux import weston_flux_tl

try:
    from numba import njit

    _NUMBA = True
except ImportError:
    _NUMBA = False


# DEPONS Ship.java:51 — ship deterrence is hard-capped at 10 km regardless of dmax_deter.
MAX_DETER_DIST_M = 10_000.0


def _compute_tl_percell(d_masked, depths, grain_sizes, salinities,
                        temperature, beta_hat, alpha_hat):
    """Compute per-porpoise TL using WestonFlux with NODATA fallback."""
    n = len(d_masked)
    tl = np.empty(n, dtype=np.float64)
    for i in range(n):
        if depths[i] > 0.0 and grain_sizes[i] != -9999.0:
            tl[i] = weston_flux_tl(
                d_masked[i], depths[i], grain_sizes[i],
                temperature, salinities[i],
            )
        else:
            d = d_masked[i]
            if d < 1.0:
                d = 1.0
            tl[i] = beta_hat * math.log10(d) + alpha_hat * d
    return tl


if _NUMBA:
    _compute_tl_percell = njit(cache=True)(_compute_tl_percell)


def _ship_received_level_from_env(source_level, dist_m, depths, grains, sal, params, weston):
    """Received level (dB, clamped >= 0) from a ship, given the porpoise-cell environment
    already fetched.

    WestonFlux per-cell when `weston` (using the supplied `depths`/`grains`/`sal`), else
    simple alpha/beta TL (env args ignored). NODATA on depth/grain/salinity OR TL <= 0 ->
    RL 0 (DEPONS Ship.java:296-307 + valueIsNoData). `dist_m` (and, when weston, the env
    arrays) are the in-range subset; `source_level` is a scalar.

    Splitting the per-cell lookups out of the RL formula lets callers that evaluate one
    porpoise at many distances (sub-tick interpolation) fetch the fixed-per-tick
    environment once instead of once per sub-step.
    """
    if weston:
        tl = _compute_tl_percell(
            dist_m, depths, grains, sal,
            params.weston_flux_default_temperature,
            params.beta_hat, params.alpha_hat,
        )
        rl = source_level - tl
        nodata = (depths <= -9999.0) | (grains <= -9999.0) | (sal <= -9999.0)
        rl = np.where(nodata | (tl <= 0.0), 0.0, rl)
    else:
        tl = params.beta_hat * np.log10(dist_m) + params.alpha_hat * dist_m
        rl = source_level - tl
    return np.maximum(rl, 0.0)


def _ship_received_level(source_level, dist_m, px, py, params, cell_data, month, weston):
    """Received level (dB, clamped >= 0) at the given porpoise positions for one ship.

    Fetches the per-cell WestonFlux environment at the porpoise positions (when `weston`)
    then delegates to `_ship_received_level_from_env`. All array args are the in-range
    subset; `source_level` is a scalar. NODATA/depth are evaluated at the PORPOISE position
    (an accepted SoA divergence; DEPONS uses the ship cell).
    """
    if weston:
        pos = np.column_stack((px, py))
        depths = cell_data.get_depths_vectorized(pos)
        grains = cell_data.get_sediments_vectorized(pos)
        sal = cell_data.get_salinities_vectorized(pos, month)
    else:
        depths = grains = sal = None
    return _ship_received_level_from_env(
        source_level, dist_m, depths, grains, sal, params, weston)


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


def _vessel_class_from_type(type_str: str) -> VesselClass:
    """Map a ships.json `type` string to a VesselClass (DEPONS VesselClass.forValue
    normalization: strip [-/ _], uppercase, match enum name). Raises on unknown type
    (fail-fast, matching DEPONS JomopansEchoSPL)."""
    norm = re.sub(r"[-/ _]", "", (type_str or "")).upper()
    aliases = {
        "CONTAINERSHIP": VesselClass.CONTAINER,
        "GOVERNMENTRESEARCH": VesselClass.GOVERNMENT,
    }
    if norm in aliases:
        return aliases[norm]
    for vc in VesselClass:
        if vc.name.replace("_", "") == norm:
            return vc
    raise ValueError(f"Unknown ship type: {type_str!r}")





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
        """Initialize the noise model (JOMOPANS source level by default)."""
        self.noise = ShipNoise(
            vessel_class=self.vessel_type,
            length=self.vessel_length,
            speed=self.current_speed,
        )
        self._prev_x = self.x
        self._prev_y = self.y
    
    def is_active(self, tick: Optional[int] = None) -> bool:
        """Check if ship is present at given tick."""
        if tick is not None:
            return self.tick_start <= tick < self.tick_end
        return self._is_active

    @property
    def is_deterring(self) -> bool:
        """DEPONS Ship.deterPorpoise gate (Ship.java:197-203): an active ship deters a
        porpoise only when it has a valid current buoy (current_buoy_idx >= 0) and is not
        paused (ticks_paused == 0). A paused ship stays _is_active with _prev == current,
        so without this gate its 30 interpolated sub-positions collapse to a stationary
        point and it would deter for the whole pause."""
        return self._is_active and self.current_buoy_idx >= 0 and self.ticks_paused <= 0

    def update(self, current_tick: int) -> None:
        """
        Update ship position and status for current tick.

        Args:
            current_tick: Current simulation tick
        """
        # Start-of-tick position for sub-tick swept-path deterrence (set before any
        # early return so paused/inactive ships keep prev == current).
        self._prev_x, self._prev_y = self.x, self.y
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
        cell_size: float = 400.0,
        cell_data=None,
        month: int = 1,
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
        
        # Distance gates — DEPONS Ship.java:220-222 (strict > at floor, <= at cap)
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        min_dist_m = params.deter_min_distance_ships * 1000.0
        if not (distance_m > min_dist_m and distance_m <= max_dist_m):
            return (False, 0.0, 0.0, distance_km)
            
        # Calculate received level
        if (params.weston_flux_percell
                and cell_data is not None
                and getattr(cell_data, '_sediment', None) is not None):
            depth = cell_data.get_depth(porpoise_x, porpoise_y)
            grain_size = cell_data.get_sediment(porpoise_x, porpoise_y)
            if depth > 0 and grain_size != -9999.0:
                salinity = cell_data.get_salinity(
                    porpoise_x, porpoise_y, month
                )
                sl = self.get_source_level()
                tl = weston_flux_tl(
                    distance_m, depth, grain_size,
                    params.weston_flux_default_temperature, salinity,
                )
                spl = sl - tl
            else:
                spl = self.get_received_level(
                    porpoise_x, porpoise_y,
                    params.alpha_hat, params.beta_hat, cell_size,
                )
        else:
            spl = self.get_received_level(
                porpoise_x, porpoise_y,
                params.alpha_hat, params.beta_hat, cell_size,
            )

        # Tships gate: skip deterrence below minimum RL (Java Ship.java:228)
        tships = getattr(params, 'deter_ships_min_db', 80.0)
        # Fast-path: skip array allocation + kernel call when RL is below threshold
        # (the kernel also gates on tships, so this only short-circuits the no-reaction case).
        if spl <= tships:
            return (False, 0.0, 0.0, distance_km)

        rl = np.array([max(0.0, float(spl))], dtype=np.float64)
        gdx = np.array([porpoise_x - self.x], dtype=np.float64)
        gdy = np.array([porpoise_y - self.y], dtype=np.float64)
        dm = np.array([max(distance_m, 1.0)], dtype=np.float64)
        u = np.array([np.random.random()], dtype=np.float64)
        _, _, prob, mag, react = self.deterrence_model.deterrence_components(
            rl, dm, gdx, gdy, is_day, u, getattr(params, "deter_ships_min_db", 80.0))
        return (bool(react[0]), float(prob[0]), float(mag[0]) if react[0] else 0.0, distance_km)


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

    def get_deterring_ships(self) -> List[Ship]:
        """Active ships eligible to deter this tick: excludes paused ships and ships with
        no current buoy, mirroring DEPONS Ship.deterPorpoise (Ship.java:197-203)."""
        if not self.enabled:
            return []
        return [s for s in self.ships if s.is_deterring]

    def calculate_aggregate_deterrence(
        self,
        porpoise_x: float,
        porpoise_y: float,
        params: SimulationParameters,
        is_day: bool = True,
        cell_size: float = 400.0,
        cell_data=None,
        month: int = 1,
    ) -> Tuple[float, float, float]:
        """
        Calculate aggregate deterrence from all ships (scalar oracle path).

        NOTE: NOT on the production tick path (Simulation.step uses
        calculate_aggregate_deterrence_vectorized). This per-porpoise oracle is a
        SINGLE-POSITION + loudest-ship oracle (no sub-tick interpolation). Its RL now
        flows through the shared _ship_received_level helper, so it honors
        weston_flux_percell exactly like the vectorized path. It is deliberately NOT a
        sub-tick oracle; do not use it to validate sub-tick aggregation.

        Args:
            porpoise_x, porpoise_y: Porpoise position
            params: Simulation parameters
            is_day: True for daytime
            cell_size: Cell size in meters
            cell_data: Landscape cell data (enables per-cell WestonFlux TL when
                weston_flux_percell is set); None falls back to alpha/beta TL
            month: Month index (1-12) for salinity lookup in the WestonFlux path

        Returns:
            (max_magnitude, total_dx, total_dy)
        """
        if not self.enabled:
            return (0.0, 0.0, 0.0)
        best_rl = -np.inf
        best_dx = best_dy = best_mag = 0.0
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        min_dist_m = params.deter_min_distance_ships * 1000.0
        tships = getattr(params, "deter_ships_min_db", 80.0)
        weston = (params.weston_flux_percell and cell_data is not None
                  and getattr(cell_data, "_sediment", None) is not None)
        for ship in self.get_deterring_ships():
            gdx = porpoise_x - ship.x
            gdy = porpoise_y - ship.y
            dist_m = max(float(np.hypot(gdx * cell_size, gdy * cell_size)), 1.0)
            if not (dist_m > min_dist_m and dist_m <= max_dist_m):
                continue
            # Compute RL ONCE; use the same value for selection and the kernel (no double-compute).
            source_level = ship.noise.get_source_level()
            rl = float(_ship_received_level(
                source_level, np.array([dist_m]), np.array([porpoise_x]),
                np.array([porpoise_y]), params, cell_data, month, weston)[0])
            if rl <= best_rl:
                continue
            best_rl = rl
            vx, vy, _, mag, react = ship.deterrence_model.deterrence_components(
                np.array([rl]), np.array([dist_m]), np.array([gdx]), np.array([gdy]),
                is_day, np.array([np.random.random()]), tships)
            best_dx, best_dy = float(vx[0]), float(vy[0])
            best_mag = float(mag[0]) if bool(react[0]) else 0.0
        return (best_mag, best_dx, best_dy)

    def calculate_aggregate_deterrence_vectorized(
        self,
        porpoise_x: np.ndarray,
        porpoise_y: np.ndarray,
        params: "SimulationParameters",
        is_day: bool = True,
        cell_size: float = 400.0,
        cell_data=None,
        month: int = 1,
        base_seed: int = 0,
        tick: int = 0,
        _force_u: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Aggregate DEPONS ship deterrence over active ships with 30-substep
        within-tick interpolation (Ship.java interpolateStep).

        For each ship, 30 sub-positions are interpolated along the ship's within-tick
        swept path (start-of-tick `_prev_x/_prev_y` -> end-of-tick `x/y`), positions
        start + (end-start)*i/30 for i=1..30. Per porpoise, per sub-step slot, the
        ship with the maximum received level wins (ShipDeterrence.recordStep); the 30
        slots are summed into the returned vector (deterrenceVtX/Y). A gated ship that
        does not react occupies its slot with a zero vector.

        NOTE: per-ship SeedSequence draws preserve the marginal Bernoulli
        probability but DELIBERATELY do not reproduce DEPONS' global-RNG draw
        order (impossible under SoA). Only the reaction *rate* matches DEPONS.

        _force_u (test-only): if set, every porpoise's reaction draw for every
        ship is this constant instead of the seeded draw. The DEPONS ship-response
        probability never approaches 1 (it caps ~0.2), so reaction-dependent
        semantics (loudest-ship-wins, order invariance, no-deter_coeff) can only be
        tested deterministically by forcing u (e.g. 0.0 = always react).
        """
        n = porpoise_x.shape[0]
        total_dx = np.zeros(n, dtype=np.float64)
        total_dy = np.zeros(n, dtype=np.float64)
        if not self.enabled:
            return (total_dx, total_dy)
        active_ships = self.get_deterring_ships()
        if not active_ships:
            return (total_dx, total_dy)

        STEPS = 30
        # DEPONS interpolateStep: positions start + (end-start)*i/30 for i=1..30
        # (excludes start, includes end).
        t_frac = np.arange(1, STEPS + 1, dtype=np.float64) / STEPS

        # Per (porpoise, sub-step slot): keep the max-RL ship's vector, then sum slots
        # (DEPONS ShipDeterrence.recordStep + deterrenceVtX/Y).
        best_rl = np.full((n, STEPS), -np.inf, dtype=np.float64)
        accum_dx = np.zeros((n, STEPS), dtype=np.float64)
        accum_dy = np.zeros((n, STEPS), dtype=np.float64)

        min_dist_m = params.deter_min_distance_ships * 1000.0
        max_dist_m = min(MAX_DETER_DIST_M, params.deter_max_distance * 1000.0)
        tships = getattr(params, "deter_ships_min_db", 80.0)
        weston = (params.weston_flux_percell and cell_data is not None
                  and getattr(cell_data, "_sediment", None) is not None)

        # Process ships in ascending id order so the result is invariant to the input
        # ship-list order, including on an exact RL tie (the strict `>` winner test below
        # keeps the FIRST-processed ship, i.e. the lowest id, on a tie). This preserves
        # this method's documented order/count invariance -- without sorting, the
        # per-ship (n, STEPS) reaction streams would make an exact-RL-tie winner (and hence
        # the stream used) depend on list order.
        for ship in sorted(active_ships, key=lambda s: int(s.id)):
            prev_x = getattr(ship, "_prev_x", ship.x)
            prev_y = getattr(ship, "_prev_y", ship.y)
            sub_x = prev_x + (ship.x - prev_x) * t_frac   # (STEPS,)
            sub_y = prev_y + (ship.y - prev_y) * t_frac

            # Pre-cull: any porpoise in range at some slot lies within max_dist of the
            # swept segment, hence within (max_dist + half segment length) of its midpoint.
            mid_x = 0.5 * (prev_x + ship.x)
            mid_y = 0.5 * (prev_y + ship.y)
            seg_len_m = float(np.hypot((ship.x - prev_x) * cell_size,
                                       (ship.y - prev_y) * cell_size))
            cand_r = max_dist_m + 0.5 * seg_len_m
            mid_d = np.hypot((porpoise_x - mid_x) * cell_size,
                             (porpoise_y - mid_y) * cell_size)
            cand = np.flatnonzero(mid_d <= cand_r)
            if cand.size == 0:
                continue

            source_level = ship.noise.get_source_level()
            # Reaction draws: PORPOISE-MAJOR (n, STEPS) stream seeded per
            # (base_seed, tick, ship.id). Porpoise i's 30 slot-draws are the contiguous
            # block u_all[i, :], so they depend ONLY on the global porpoise index i, not on
            # the total count n -> invariant to ship order AND porpoise count/membership.
            # (A (STEPS, n) layout would NOT be count-invariant: C-order interleaves
            # porpoise i's draws at flat positions i, n+i, 2n+i, ... which shift with n.)
            # Only the marginal Bernoulli RATE matches DEPONS (global draw order is
            # unreproducible under SoA).
            if _force_u is None:
                rng = np.random.default_rng(
                    np.random.SeedSequence([base_seed, tick, int(ship.id)]))
                u_all = rng.random((n, STEPS))
            else:
                u_all = None

            px_c = porpoise_x[cand]
            py_c = porpoise_y[cand]
            m = cand.size
            # Distances for every (candidate porpoise, sub-step) pair (m, STEPS).
            gdx = px_c[:, None] - sub_x[None, :]
            gdy = py_c[:, None] - sub_y[None, :]
            dist_m = np.hypot(gdx * cell_size, gdy * cell_size)
            np.maximum(dist_m, 1.0, out=dist_m)

            # Only in-range pairs can deter; the rest are gated out and their RL/vector
            # would be discarded. Compute the WestonFlux TL and the deterrence kernel for
            # the in-range pairs ONLY (a candidate near a long swept path is out of range
            # at the far sub-steps). `ir` indexes the flattened (m, STEPS) grid in C-order,
            # so the porpoise row of each pair is `ir // STEPS`.
            in_range = (dist_m > min_dist_m) & (dist_m <= max_dist_m)
            ir = np.flatnonzero(in_range.ravel())
            if ir.size == 0:
                continue
            rows = ir // STEPS
            d_ir = dist_m.ravel()[ir]
            gdx_ir = gdx.ravel()[ir]
            gdy_ir = gdy.ravel()[ir]

            # Fixed-per-tick porpoise-cell environment for WestonFlux: fetch ONCE per
            # candidate (m lookups) -- depth/grain/salinity are at the porpoise cell and
            # don't vary across the ship's sub-positions, only the distance does -- then
            # index by the in-range pairs' porpoise rows. The per-pair TL formula still
            # recomputes because distance varies.
            if weston:
                pos_c = np.column_stack((px_c, py_c))
                depths_ir = cell_data.get_depths_vectorized(pos_c)[rows]
                grains_ir = cell_data.get_sediments_vectorized(pos_c)[rows]
                sal_ir = cell_data.get_salinities_vectorized(pos_c, month)[rows]
            else:
                depths_ir = grains_ir = sal_ir = None
            rl_ir = _ship_received_level_from_env(
                source_level, d_ir, depths_ir, grains_ir, sal_ir, params, weston)

            if _force_u is None:
                u_ir = u_all[cand, :].ravel()[ir]         # porpoise-major rows
            else:
                u_ir = np.full(ir.size, float(_force_u), dtype=np.float64)
            vx_ir, vy_ir, _, _, _ = ship.deterrence_model.deterrence_components(
                rl_ir, d_ir, gdx_ir, gdy_ir, is_day, u_ir, tships)

            # Scatter back onto the (m, STEPS) grid: out-of-range slots keep RL = -inf
            # (so the `rl > tships` winner test excludes them, exactly as the explicit
            # in_range mask did) and a zero vector.
            rl = np.full((m, STEPS), -np.inf, dtype=np.float64)
            vx = np.zeros((m, STEPS), dtype=np.float64)
            vy = np.zeros((m, STEPS), dtype=np.float64)
            rl.ravel()[ir] = rl_ir
            vx.ravel()[ir] = vx_ir
            vy.ravel()[ir] = vy_ir

            # Loudest gated ship wins each (porpoise, slot); its vector is 0 if it did not
            # react. Slots a ship does not win keep the incumbent value.
            cur_best = best_rl[cand, :]
            wins = (rl > tships) & (rl > cur_best)
            best_rl[cand, :] = np.where(wins, rl, cur_best)
            accum_dx[cand, :] = np.where(wins, vx, accum_dx[cand, :])
            accum_dy[cand, :] = np.where(wins, vy, accum_dy[cand, :])

        total_dx = accum_dx.sum(axis=1)
        total_dy = accum_dy.sum(axis=1)
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
            logger.warning("Ship route file not found: %s. Ships will have no routes.", filepath)
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
                route = routes.get(route_name)
                if route is None:
                    logger.warning(
                        "Ship '%s' references unknown route '%s' — ship will be stationary.",
                        name, route_name,
                    )
                    route = Route()
                
                # Optional timing
                try:
                    tick_start = int(parts[4]) if len(parts) > 4 else 0
                    tick_end = int(parts[5]) if len(parts) > 5 else 2147483647
                except ValueError as e:
                    logger.warning(
                        "Ship '%s': invalid tick timing values (%s) — "
                        "ship will be active for entire simulation.",
                        name, e,
                    )
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
            self.enabled = False
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
                
                # Speed from waypoint JSON; may be overridden per-ship below if ship record supplies an explicit speed.
                buoy = Buoy(x=grid_x, y=grid_y,
                            speed=waypoint.get("speed", 10.0),
                            pause_ticks=waypoint.get("pause", 0))
                buoys.append(buoy)
                
            routes_dict[route_name] = Route(name=route_name, buoys=buoys)
            
        # Parse ships
        self.ships = []
        for i, ship_data in enumerate(data.get("ships", [])):
            name = ship_data.get("name", f"ship_{i}")
            if ship_data.get("survey"):
                logger.debug("Ship %s: 'survey' field present but not modeled (ignored)", name)
            speed = ship_data.get("speed")          # ship-level override; None -> keep buoy speeds
            impact = ship_data.get("impact")        # explicit SL override; None -> JOMOPANS
            start_tick = ship_data.get("start", 0)
            route_name = ship_data.get("route", "")
            length_m = ship_data.get("length", 100.0)

            route = routes_dict.get(route_name, Route())

            # Only overwrite buoy speeds when the ship record gives an explicit speed;
            # otherwise preserve the per-waypoint speeds the JSON route provides (JOMOPANS
            # is speed-dependent, so clobbering them with a default would corrupt SL).
            if speed is not None:
                for buoy in route.buoys:
                    buoy.speed = speed

            x, y = 0.0, 0.0
            if route.buoys:
                x = route.buoys[0].x
                y = route.buoys[0].y

            vessel_type = _vessel_class_from_type(ship_data.get("type") or "Other")

            ship = Ship(
                id=i, x=x, y=y, heading=0.0, name=name,
                vessel_type=vessel_type, vessel_length=length_m,
                route=route, tick_start=start_tick, tick_end=2147483647,
            )

            # Explicit dB override only when impact is present and positive (CENOP extension;
            # DEPONS always uses JOMOPANS). Absent impact -> base_source_level stays None -> JOMOPANS.
            if impact is not None and impact > 0:
                ship.noise.base_source_level = impact

            self.ships.append(ship)
            
        logger.info("Loaded %d ships with %d routes from %s", len(self.ships), len(routes_dict), json_file)
