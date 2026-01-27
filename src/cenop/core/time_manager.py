"""
Unified Time Manager supporting DEPONS (fixed) and JASMINE (flexible) modes.

The TimeManager is the central coordinator for:
- Temporal discretization
- Deterministic random seeding
- Event scheduling
- Update frequency management
- Reproducibility guarantees

This module enables the CENOP-JASMINE hybrid simulation framework while
preserving DEPONS 3.0 reproducibility for regulatory use.

Translates from: JASMINE-MB TimeManager concept + DEPONS tick system
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Any
from enum import Enum, auto


class TimeMode(Enum):
    """
    Time management modes.

    DEPONS: Fixed timestep, deterministic, regulatory-compliant
    JASMINE: Flexible timestep, event-driven, research mode
    """
    DEPONS = auto()
    JASMINE = auto()


@dataclass(frozen=True)
class TimeState:
    """
    Immutable snapshot of current simulation time.

    This replaces the time-related fields from SimulationState,
    keeping them separate from population statistics.

    Attributes:
        tick: Current tick number (0-indexed)
        day: Day counter (0-indexed, resets conceptually but tracks total)
        month: Month (1-12)
        year: Year counter (starts at 1)
    """
    tick: int = 0
    day: int = 0
    month: int = 1
    year: int = 1

    @property
    def hour(self) -> int:
        """Current hour of day (0-23). Each tick = 30 min."""
        return (self.tick % 48) // 2

    @property
    def minute(self) -> int:
        """Current minute within hour (0 or 30)."""
        return (self.tick % 2) * 30

    @property
    def is_daytime(self) -> bool:
        """True if between 6:00-18:00."""
        return 6 <= self.hour < 18

    @property
    def quarter(self) -> int:
        """Calendar quarter (0-3)."""
        return (self.month - 1) // 3

    @property
    def day_of_year(self) -> int:
        """Day of year (1-360)."""
        return ((self.month - 1) * 30) + (self.day % 30) + 1

    @property
    def total_days(self) -> float:
        """Total simulation days elapsed."""
        return self.tick / 48.0


class TimeManager:
    """
    Unified time manager for CENOP-JASMINE hybrid simulation.

    Supports two modes:
    - DEPONS: Fixed 30-minute timestep, deterministic seeding, regulatory-compliant
    - JASMINE: Flexible timestep, event scheduling, multi-frequency updates

    Key Responsibilities:
        1. Temporal discretization (dt management)
        2. Deterministic random seeding (per-tick, per-agent)
        3. Event scheduling and dispatch
        4. Update frequency coordination
        5. Reproducibility guarantees

    DEPONS Mode Guarantees:
        - Fixed 30-minute timestep (cannot be changed)
        - Deterministic seed = base_seed + tick
        - No event scheduling (raises RuntimeError)
        - Identical to original CENOP time behavior

    JASMINE Mode Features:
        - Variable timestep via set_dt()
        - Event scheduling at specific ticks or datetimes
        - Subsystem update frequency configuration
        - Multi-frequency environment updates

    Example:
        # DEPONS mode (regulatory compliance)
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        while not tm.is_finished():
            np.random.seed(tm.get_seed())
            # ... simulation step ...
            tm.advance()

        # JASMINE mode (research)
        tm = TimeManager(mode=TimeMode.JASMINE, dt_seconds=60, base_seed=42)
        tm.schedule_event(1000, lambda: print("Event at tick 1000"))

    Translates from:
        - DEPONS: SimulationState tick management
        - JASMINE-MB: TimeManager with event scheduling
    """

    # DEPONS constants (must match original CENOP)
    DEPONS_DT_SECONDS: int = 1800  # 30 minutes
    TICKS_PER_DAY: int = 48
    DAYS_PER_MONTH: int = 30
    DAYS_PER_YEAR: int = 360
    TICKS_PER_YEAR: int = TICKS_PER_DAY * DAYS_PER_YEAR  # 17,280

    def __init__(
        self,
        mode: TimeMode = TimeMode.DEPONS,
        dt_seconds: int = 1800,
        base_seed: int = 42,
        sim_years: int = 1,
        start_datetime: Optional[datetime] = None
    ):
        """
        Initialize time manager.

        Args:
            mode: DEPONS (fixed) or JASMINE (flexible) mode
            dt_seconds: Timestep in seconds (ignored in DEPONS mode, forced to 1800)
            base_seed: Base random seed for reproducibility
            sim_years: Total simulation duration in years
            start_datetime: Optional real-world start time for datetime calculations
        """
        self._mode = mode
        self._base_seed = base_seed
        self._sim_years = sim_years
        self._start_datetime = start_datetime or datetime(2020, 1, 1)

        # In DEPONS mode, force fixed timestep for regulatory compliance
        if mode == TimeMode.DEPONS:
            self._dt_seconds = self.DEPONS_DT_SECONDS
        else:
            self._dt_seconds = dt_seconds

        # Current time state (immutable dataclass)
        self._state = TimeState()

        # Event scheduling (JASMINE mode only)
        self._scheduled_events: Dict[int, List[Callable[[], None]]] = {}

        # Update frequency registry for subsystems
        # Key: subsystem name, Value: tick interval
        self._update_frequencies: Dict[str, int] = {
            'movement': 1,       # Every tick
            'physiology': 1,     # Every tick
            'deterrence': 1,     # Every tick
            'food': 48,          # Daily
            'memory': 48,        # Daily
            'landscape': 48,     # Daily (monthly variation via set_month)
            'reproduction': 48,  # Daily
            'mortality': 48,     # Daily
        }

        # Calculate total ticks
        self._max_ticks = sim_years * self.TICKS_PER_YEAR

    # =========================================================================
    # Properties
    # =========================================================================

    @property
    def mode(self) -> TimeMode:
        """Current time mode (DEPONS or JASMINE)."""
        return self._mode

    @property
    def state(self) -> TimeState:
        """Current time state (immutable snapshot)."""
        return self._state

    @property
    def tick(self) -> int:
        """Current tick number."""
        return self._state.tick

    @property
    def day(self) -> int:
        """Current day counter."""
        return self._state.day

    @property
    def month(self) -> int:
        """Current month (1-12)."""
        return self._state.month

    @property
    def year(self) -> int:
        """Current year."""
        return self._state.year

    @property
    def hour(self) -> int:
        """Current hour of day (0-23)."""
        return self._state.hour

    @property
    def is_daytime(self) -> bool:
        """True if between 6:00-18:00."""
        return self._state.is_daytime

    @property
    def quarter(self) -> int:
        """Current quarter (0-3)."""
        return self._state.quarter

    @property
    def dt_seconds(self) -> int:
        """Current timestep in seconds."""
        return self._dt_seconds

    @property
    def max_ticks(self) -> int:
        """Total ticks for simulation."""
        return self._max_ticks

    @property
    def base_seed(self) -> int:
        """Base random seed."""
        return self._base_seed

    @property
    def current_datetime(self) -> datetime:
        """Current simulation datetime (based on start_datetime + elapsed time)."""
        delta = timedelta(seconds=self._state.tick * self._dt_seconds)
        return self._start_datetime + delta

    @property
    def progress(self) -> float:
        """Simulation progress as fraction (0.0 to 1.0)."""
        if self._max_ticks == 0:
            return 1.0
        return self._state.tick / self._max_ticks

    # =========================================================================
    # Core Time Operations
    # =========================================================================

    def is_finished(self) -> bool:
        """
        Check if simulation should end.

        Returns:
            True if current tick >= max_ticks
        """
        return self._state.tick >= self._max_ticks

    def advance(self) -> None:
        """
        Advance simulation by one timestep.

        Updates tick, day, month, year counters following DEPONS conventions:
        - 48 ticks per day
        - 30 days per month
        - 12 months per year (360 days total)

        Creates a new immutable TimeState.
        """
        new_tick = self._state.tick + 1
        new_day = self._state.day
        new_month = self._state.month
        new_year = self._state.year

        # Day boundary (48 ticks per day)
        if new_tick % self.TICKS_PER_DAY == 0:
            new_day += 1

            # Month boundary (30 days per month)
            if new_day % self.DAYS_PER_MONTH == 0:
                new_month = (new_month % 12) + 1

                # Year boundary
                if new_month == 1:
                    new_year += 1

        self._state = TimeState(
            tick=new_tick,
            day=new_day,
            month=new_month,
            year=new_year
        )

    def reset(self) -> None:
        """Reset time manager to initial state."""
        self._state = TimeState()
        self._scheduled_events.clear()

    # =========================================================================
    # Deterministic Seeding
    # =========================================================================

    def get_seed(self) -> int:
        """
        Get deterministic seed for current tick.

        DEPONS Mode: seed = base_seed + tick
            Simple linear progression ensures reproducibility.

        JASMINE Mode: seed = base_seed + tick * 1000
            Larger multiplier provides more seed space for sub-operations.

        Returns:
            Deterministic seed for this timestep
        """
        if self._mode == TimeMode.DEPONS:
            return self._base_seed + self._state.tick
        else:
            return self._base_seed + self._state.tick * 1000

    def get_agent_seed(self, agent_id: int) -> int:
        """
        Get deterministic seed for specific agent at current tick.

        Used for per-agent stochastic operations (movement, decisions)
        while maintaining reproducibility.

        Formula: tick_seed * 10000 + agent_id

        Args:
            agent_id: Unique agent identifier (0-indexed)

        Returns:
            Deterministic seed for this agent at this timestep
        """
        return self.get_seed() * 10000 + agent_id

    # =========================================================================
    # Boundary Detection
    # =========================================================================

    def is_day_boundary(self) -> bool:
        """
        Check if at day boundary (tick 48, 96, 144, ...).

        Returns:
            True if tick is multiple of 48 (and tick > 0)
        """
        return self._state.tick > 0 and self._state.tick % self.TICKS_PER_DAY == 0

    def is_month_boundary(self) -> bool:
        """
        Check if at month boundary (day 30, 60, 90, ...).

        Returns:
            True if at day boundary AND day is multiple of 30
        """
        return (self.is_day_boundary() and
                self._state.day > 0 and
                self._state.day % self.DAYS_PER_MONTH == 0)

    def is_year_boundary(self) -> bool:
        """
        Check if at year boundary (day 360, 720, ...).

        Returns:
            True if at day boundary AND day is multiple of 360
        """
        return (self.is_day_boundary() and
                self._state.day > 0 and
                self._state.day % self.DAYS_PER_YEAR == 0)

    # =========================================================================
    # Subsystem Update Scheduling
    # =========================================================================

    def should_update(self, subsystem: str) -> bool:
        """
        Check if a subsystem should update at current tick.

        Args:
            subsystem: Name of subsystem (e.g., 'movement', 'food', 'memory')

        Returns:
            True if subsystem should update this tick
        """
        frequency = self._update_frequencies.get(subsystem, 1)
        if frequency <= 0:
            return False
        return self._state.tick % frequency == 0

    def set_update_frequency(self, subsystem: str, frequency: int) -> None:
        """
        Set update frequency for a subsystem (JASMINE mode only).

        Args:
            subsystem: Name of subsystem
            frequency: Tick interval (1 = every tick, 48 = daily, etc.)

        Raises:
            RuntimeError: If called in DEPONS mode
        """
        if self._mode == TimeMode.DEPONS:
            raise RuntimeError("Cannot change update frequencies in DEPONS mode")
        self._update_frequencies[subsystem] = frequency

    def get_update_frequency(self, subsystem: str) -> int:
        """
        Get update frequency for a subsystem.

        Args:
            subsystem: Name of subsystem

        Returns:
            Tick interval for this subsystem
        """
        return self._update_frequencies.get(subsystem, 1)

    # =========================================================================
    # Event Scheduling (JASMINE Mode Only)
    # =========================================================================

    def schedule_event(self, tick: int, callback: Callable[[], None]) -> None:
        """
        Schedule an event for a specific tick (JASMINE mode only).

        Args:
            tick: Tick number when event should fire
            callback: Function to call (no arguments)

        Raises:
            RuntimeError: If called in DEPONS mode
        """
        if self._mode == TimeMode.DEPONS:
            raise RuntimeError("Event scheduling not available in DEPONS mode")

        if tick not in self._scheduled_events:
            self._scheduled_events[tick] = []
        self._scheduled_events[tick].append(callback)

    def schedule_event_at_datetime(
        self,
        dt: datetime,
        callback: Callable[[], None]
    ) -> None:
        """
        Schedule an event at a specific datetime (JASMINE mode only).

        Args:
            dt: Datetime when event should fire
            callback: Function to call

        Raises:
            RuntimeError: If called in DEPONS mode
        """
        delta = dt - self._start_datetime
        tick = int(delta.total_seconds() / self._dt_seconds)
        self.schedule_event(tick, callback)

    def get_scheduled_events(self) -> List[Callable[[], None]]:
        """
        Get events scheduled for current tick.

        Returns:
            List of callback functions to execute (empty if none)
        """
        return self._scheduled_events.get(self._state.tick, [])

    def clear_past_events(self) -> None:
        """Remove events for ticks that have passed."""
        past_ticks = [t for t in self._scheduled_events if t < self._state.tick]
        for t in past_ticks:
            del self._scheduled_events[t]

    def has_pending_events(self) -> bool:
        """Check if there are any future scheduled events."""
        return any(t > self._state.tick for t in self._scheduled_events)

    # =========================================================================
    # Variable Timestep (JASMINE Mode Only)
    # =========================================================================

    def set_dt(self, dt_seconds: int) -> None:
        """
        Change timestep (JASMINE mode only).

        Args:
            dt_seconds: New timestep in seconds

        Raises:
            RuntimeError: If called in DEPONS mode
            ValueError: If dt_seconds <= 0
        """
        if self._mode == TimeMode.DEPONS:
            raise RuntimeError("Cannot change timestep in DEPONS mode")
        if dt_seconds <= 0:
            raise ValueError("Timestep must be positive")
        self._dt_seconds = dt_seconds

    # =========================================================================
    # Serialization / Checkpointing
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialize time manager state for checkpointing.

        Returns:
            Dictionary with all state needed to restore TimeManager
        """
        return {
            'mode': self._mode.name,
            'tick': self._state.tick,
            'day': self._state.day,
            'month': self._state.month,
            'year': self._state.year,
            'base_seed': self._base_seed,
            'dt_seconds': self._dt_seconds,
            'sim_years': self._sim_years,
            'start_datetime': self._start_datetime.isoformat(),
            'update_frequencies': self._update_frequencies.copy(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeManager':
        """
        Restore time manager from checkpoint.

        Args:
            data: Dictionary from to_dict()

        Returns:
            Restored TimeManager instance
        """
        start_dt = datetime.fromisoformat(data['start_datetime'])

        tm = cls(
            mode=TimeMode[data['mode']],
            base_seed=data['base_seed'],
            dt_seconds=data['dt_seconds'],
            sim_years=data['sim_years'],
            start_datetime=start_dt
        )

        tm._state = TimeState(
            tick=data['tick'],
            day=data['day'],
            month=data['month'],
            year=data['year']
        )

        if 'update_frequencies' in data:
            tm._update_frequencies = data['update_frequencies'].copy()

        return tm

    # =========================================================================
    # String Representation
    # =========================================================================

    def __repr__(self) -> str:
        return (
            f"TimeManager(mode={self._mode.name}, tick={self._state.tick}, "
            f"day={self._state.day}, month={self._state.month}, year={self._state.year})"
        )

    def __str__(self) -> str:
        return (
            f"TimeManager [{self._mode.name}]: "
            f"Year {self._state.year}, Month {self._state.month}, "
            f"Day {self._state.day % 30 + 1}, {self._state.hour:02d}:{self._state.minute:02d}"
        )
