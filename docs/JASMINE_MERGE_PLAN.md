# JASMINE-CENOP Merge Plan

## Executive Summary

This document outlines the implementation plan for merging JASMINE-MB capabilities into CENOP while **preserving DEPONS 3.0 reproducibility for regulatory use**. The merge starts with the Time Manager as the foundational component.

---

## Phase 1: JASMINE Time Manager Migration

### 1.1 Current State Analysis

**CENOP Current Time Management:**
- Located in: `src/cenop/core/simulation.py` (`SimulationState` class)
- Timestep: 30 minutes (fixed)
- Time units: tick (30min) -> day (48 ticks) -> month (30 days) -> year (360 days)
- Seeding: Single seed at simulation start
- Update order: Hard-coded in `Simulation.step()`

**Key Constraint:** DEPONS uses a **fixed 30-minute timestep** for:
- Movement model calibration (step length distributions)
- Deterrence response timing
- Energy consumption rates
- Reproduction timing

### 1.2 Target Architecture

Create a `TimeManager` class that supports:
1. **DEPONS Mode**: Fixed timestep, deterministic, bit-reproducible
2. **JASMINE Mode**: Flexible timestep, event scheduling, multi-frequency updates

```
src/cenop/
├── core/
│   ├── time_manager.py      # NEW: Unified time management
│   ├── simulation.py        # MODIFIED: Use TimeManager
│   └── scheduler.py         # ENHANCED: Event scheduling
```

---

## Phase 1 Implementation Steps

### Step 1.1: Create TimeManager Base Class

**File:** `src/cenop/core/time_manager.py`

```python
"""
Unified Time Manager supporting DEPONS (fixed) and JASMINE (flexible) modes.

The TimeManager is the central coordinator for:
- Temporal discretization
- Deterministic random seeding
- Event scheduling
- Update frequency management
- Reproducibility guarantees
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Callable, Optional, Literal, Any
from enum import Enum, auto
import numpy as np


class TimeMode(Enum):
    """Time management modes."""
    DEPONS = auto()   # Fixed timestep, deterministic, regulatory-compliant
    JASMINE = auto()  # Flexible timestep, event-driven, research mode


@dataclass
class TimeState:
    """
    Immutable snapshot of current simulation time.

    Replaces: SimulationState time-related fields
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
        """True if 6:00-18:00."""
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

    Usage:
        # DEPONS mode (regulatory)
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # JASMINE mode (research)
        tm = TimeManager(mode=TimeMode.JASMINE, dt_seconds=60, base_seed=42)
    """

    # DEPONS constants
    DEPONS_DT_SECONDS: int = 1800  # 30 minutes
    TICKS_PER_DAY: int = 48
    DAYS_PER_MONTH: int = 30
    DAYS_PER_YEAR: int = 360

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
            mode: DEPONS (fixed) or JASMINE (flexible)
            dt_seconds: Timestep in seconds (ignored in DEPONS mode)
            base_seed: Base random seed for reproducibility
            sim_years: Total simulation years
            start_datetime: Optional real-world start time
        """
        self.mode = mode
        self._base_seed = base_seed
        self._sim_years = sim_years
        self._start_datetime = start_datetime or datetime(2020, 1, 1)

        # In DEPONS mode, force fixed timestep
        if mode == TimeMode.DEPONS:
            self._dt_seconds = self.DEPONS_DT_SECONDS
        else:
            self._dt_seconds = dt_seconds

        # State
        self._state = TimeState()
        self._is_running = False

        # Event scheduling (JASMINE mode only)
        self._scheduled_events: Dict[int, List[Callable]] = {}

        # Update frequency registry
        self._update_frequencies: Dict[str, int] = {
            'movement': 1,      # Every tick
            'physiology': 1,    # Every tick
            'deterrence': 1,    # Every tick
            'food': 48,         # Daily
            'memory': 48,       # Daily
            'landscape': 48,    # Daily (monthly variation via set_month)
            'reproduction': 48, # Daily
            'mortality': 48,    # Daily
        }

        # Calculate total ticks
        self._max_ticks = sim_years * self.DAYS_PER_YEAR * self.TICKS_PER_DAY

    @property
    def state(self) -> TimeState:
        """Current time state (immutable snapshot)."""
        return self._state

    @property
    def tick(self) -> int:
        """Current tick number."""
        return self._state.tick

    @property
    def dt_seconds(self) -> int:
        """Current timestep in seconds."""
        return self._dt_seconds

    @property
    def max_ticks(self) -> int:
        """Total ticks for simulation."""
        return self._max_ticks

    @property
    def current_datetime(self) -> datetime:
        """Current simulation datetime."""
        delta = timedelta(seconds=self._state.tick * self._dt_seconds)
        return self._start_datetime + delta

    def is_finished(self) -> bool:
        """Check if simulation should end."""
        return self._state.tick >= self._max_ticks

    def advance(self) -> None:
        """
        Advance simulation by one timestep.

        Updates tick, day, month, year counters.
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

    def get_seed(self) -> int:
        """
        Get deterministic seed for current tick.

        DEPONS Mode: seed = base_seed + tick
        JASMINE Mode: seed = base_seed + tick * multiplier

        Returns:
            Deterministic seed for this timestep
        """
        if self.mode == TimeMode.DEPONS:
            return self._base_seed + self._state.tick
        else:
            return self._base_seed + self._state.tick * 1000

    def get_agent_seed(self, agent_id: int) -> int:
        """
        Get deterministic seed for specific agent at current tick.

        Used for per-agent stochastic operations (movement, decisions).

        Args:
            agent_id: Unique agent identifier

        Returns:
            Deterministic seed for this agent at this timestep
        """
        return self.get_seed() * 10000 + agent_id

    def should_update(self, subsystem: str) -> bool:
        """
        Check if a subsystem should update at current tick.

        Args:
            subsystem: Name of subsystem (e.g., 'movement', 'food', 'memory')

        Returns:
            True if subsystem should update this tick
        """
        frequency = self._update_frequencies.get(subsystem, 1)
        return self._state.tick % frequency == 0

    def is_day_boundary(self) -> bool:
        """Check if at day boundary (tick 0, 48, 96, ...)."""
        return self._state.tick > 0 and self._state.tick % self.TICKS_PER_DAY == 0

    def is_month_boundary(self) -> bool:
        """Check if at month boundary."""
        return (self.is_day_boundary() and
                self._state.day > 0 and
                self._state.day % self.DAYS_PER_MONTH == 0)

    def is_year_boundary(self) -> bool:
        """Check if at year boundary."""
        return (self.is_day_boundary() and
                self._state.day > 0 and
                self._state.day % self.DAYS_PER_YEAR == 0)

    # =========================================================================
    # Event Scheduling (JASMINE Mode)
    # =========================================================================

    def schedule_event(self, tick: int, callback: Callable[[], None]) -> None:
        """
        Schedule an event for a specific tick (JASMINE mode only).

        Args:
            tick: Tick number when event should fire
            callback: Function to call
        """
        if self.mode == TimeMode.DEPONS:
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
        """
        delta = dt - self._start_datetime
        tick = int(delta.total_seconds() / self._dt_seconds)
        self.schedule_event(tick, callback)

    def get_scheduled_events(self) -> List[Callable[[], None]]:
        """
        Get events scheduled for current tick.

        Returns:
            List of callback functions to execute
        """
        return self._scheduled_events.get(self._state.tick, [])

    def clear_past_events(self) -> None:
        """Remove events for ticks that have passed."""
        past_ticks = [t for t in self._scheduled_events if t < self._state.tick]
        for t in past_ticks:
            del self._scheduled_events[t]

    # =========================================================================
    # Variable Timestep (JASMINE Mode)
    # =========================================================================

    def set_dt(self, dt_seconds: int) -> None:
        """
        Change timestep (JASMINE mode only).

        Args:
            dt_seconds: New timestep in seconds
        """
        if self.mode == TimeMode.DEPONS:
            raise RuntimeError("Cannot change timestep in DEPONS mode")
        self._dt_seconds = dt_seconds

    # =========================================================================
    # Serialization / Checkpointing
    # =========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Serialize time manager state for checkpointing."""
        return {
            'mode': self.mode.name,
            'tick': self._state.tick,
            'day': self._state.day,
            'month': self._state.month,
            'year': self._state.year,
            'base_seed': self._base_seed,
            'dt_seconds': self._dt_seconds,
            'sim_years': self._sim_years,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TimeManager':
        """Restore time manager from checkpoint."""
        tm = cls(
            mode=TimeMode[data['mode']],
            base_seed=data['base_seed'],
            dt_seconds=data['dt_seconds'],
            sim_years=data['sim_years']
        )
        tm._state = TimeState(
            tick=data['tick'],
            day=data['day'],
            month=data['month'],
            year=data['year']
        )
        return tm
```

### Step 1.2: Integration with Simulation Class

**Modify:** `src/cenop/core/simulation.py`

1. Add `TimeManager` as constructor parameter
2. Replace `SimulationState` time fields with `TimeManager.state`
3. Replace manual seed management with `TimeManager.get_seed()`
4. Keep `SimulationState` for non-time state (population stats)

```python
# In Simulation.__init__:
from cenop.core.time_manager import TimeManager, TimeMode

def __init__(
    self,
    params: SimulationParameters,
    cell_data: Optional[CellData] = None,
    seed: Optional[int] = None,
    time_manager: Optional[TimeManager] = None  # NEW
):
    # Time management
    actual_seed = seed if seed is not None else params.random_seed

    if time_manager is not None:
        self.time_manager = time_manager
    else:
        # Default: DEPONS mode for backward compatibility
        self.time_manager = TimeManager(
            mode=TimeMode.DEPONS,
            base_seed=actual_seed or 42,
            sim_years=params.sim_years
        )

    # Legacy state (population stats only)
    self.state = SimulationState()

    # ... rest of init
```

### Step 1.3: Update Simulation Loop

**Modify:** `src/cenop/core/simulation.py` `step()` method

```python
def step(self) -> None:
    """Execute one simulation step."""
    if not self._is_initialized:
        self.initialize()

    # 1. Set deterministic seed for this tick
    np.random.seed(self.time_manager.get_seed())

    # 2. Process scheduled events (JASMINE mode only)
    for event in self.time_manager.get_scheduled_events():
        event()

    # 3. Update environment subsystems (if due)
    if self.time_manager.should_update('deterrence'):
        self._turbine_manager.update(self.time_manager.tick)
        self._ship_manager.update(self.time_manager.tick)

    # 4. Calculate deterrence (every tick)
    # ... existing deterrence code ...

    # 5. Step population
    self.population_manager.step(deterrence_vectors=(total_dx, total_dy))

    # 6. Daily tasks
    if self.time_manager.is_day_boundary():
        self._daily_tasks()

    # 7. Monthly tasks
    if self.time_manager.is_month_boundary():
        self._monthly_tasks()

    # 8. Yearly tasks
    if self.time_manager.is_year_boundary():
        self._yearly_tasks()

    # 9. Record history (daily)
    if self.time_manager.is_day_boundary():
        self._record_history()

    # 10. Advance time
    self.time_manager.advance()

    # 11. Sync legacy state
    self.state.tick = self.time_manager.tick
    self.state.day = self.time_manager.state.day
    self.state.month = self.time_manager.state.month
    self.state.year = self.time_manager.state.year
```

---

## Phase 1 Testing Strategy

### Step 1.4: Regression Tests for DEPONS Reproducibility

**Create:** `tests/test_time_manager.py`

```python
"""
Tests for TimeManager ensuring DEPONS reproducibility.

Critical: These tests MUST pass before any production use.
They verify bit-exact reproducibility with the old CENOP implementation.
"""

import pytest
import numpy as np
from cenop.core.time_manager import TimeManager, TimeMode, TimeState
from cenop.core.simulation import Simulation
from cenop.parameters import SimulationParameters


class TestTimeManagerDEPONSMode:
    """Verify DEPONS mode matches original CENOP behavior."""

    def test_fixed_timestep_enforced(self):
        """DEPONS mode should enforce 30-minute timestep."""
        tm = TimeManager(mode=TimeMode.DEPONS, dt_seconds=60)  # Try to set 60s
        assert tm.dt_seconds == 1800, "DEPONS mode must use 30-min timestep"

    def test_timestep_change_blocked(self):
        """Cannot change timestep in DEPONS mode."""
        tm = TimeManager(mode=TimeMode.DEPONS)
        with pytest.raises(RuntimeError):
            tm.set_dt(60)

    def test_event_scheduling_blocked(self):
        """Cannot schedule events in DEPONS mode."""
        tm = TimeManager(mode=TimeMode.DEPONS)
        with pytest.raises(RuntimeError):
            tm.schedule_event(100, lambda: None)

    def test_deterministic_seeding(self):
        """Same tick should produce same seed."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        seed_at_0 = tm.get_seed()
        tm.advance()
        seed_at_1 = tm.get_seed()

        # Reset and verify
        tm2 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        assert tm2.get_seed() == seed_at_0
        tm2.advance()
        assert tm2.get_seed() == seed_at_1

    def test_agent_seed_deterministic(self):
        """Agent seeds should be deterministic."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        seed_agent_0 = tm.get_agent_seed(0)
        seed_agent_1 = tm.get_agent_seed(1)

        assert seed_agent_0 != seed_agent_1, "Different agents need different seeds"

        # Same agent same tick = same seed
        tm2 = TimeManager(mode=TimeMode.DEPONS, base_seed=42)
        assert tm2.get_agent_seed(0) == seed_agent_0

    def test_time_boundaries(self):
        """Verify day/month/year boundary detection."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # Advance 47 ticks (not yet day boundary)
        for _ in range(47):
            tm.advance()
        assert not tm.is_day_boundary()

        # Tick 48 is day boundary
        tm.advance()
        assert tm.is_day_boundary()
        assert tm.state.day == 1

    def test_month_boundary(self):
        """Verify month boundary at day 30."""
        tm = TimeManager(mode=TimeMode.DEPONS, base_seed=42)

        # Advance 30 days (30 * 48 = 1440 ticks)
        for _ in range(1440):
            tm.advance()

        assert tm.is_month_boundary()
        assert tm.state.month == 2


class TestDEPONSReproducibility:
    """
    Critical regression tests comparing TimeManager-based simulation
    to original CENOP implementation.
    """

    def test_same_seed_same_trajectories(self):
        """
        Identical seeds must produce identical trajectories.

        This is the MOST CRITICAL test for regulatory compliance.
        """
        params = SimulationParameters(
            porpoise_count=10,
            sim_years=1,
            random_seed=42
        )

        # Run simulation twice with same seed
        sim1 = Simulation(params=params)
        sim1.initialize()

        positions_1 = []
        for _ in range(100):
            sim1.step()
            positions_1.append((
                sim1.population_manager.x.copy(),
                sim1.population_manager.y.copy()
            ))

        # Reset and run again
        sim2 = Simulation(params=params)
        sim2.initialize()

        positions_2 = []
        for _ in range(100):
            sim2.step()
            positions_2.append((
                sim2.population_manager.x.copy(),
                sim2.population_manager.y.copy()
            ))

        # Compare trajectories
        for i, ((x1, y1), (x2, y2)) in enumerate(zip(positions_1, positions_2)):
            np.testing.assert_array_almost_equal(
                x1, x2, decimal=10,
                err_msg=f"X positions differ at step {i}"
            )
            np.testing.assert_array_almost_equal(
                y1, y2, decimal=10,
                err_msg=f"Y positions differ at step {i}"
            )

    def test_population_dynamics_match(self):
        """Population stats should match between implementations."""
        params = SimulationParameters(
            porpoise_count=50,
            sim_years=1,
            random_seed=12345
        )

        sim = Simulation(params=params)

        # Run for 1 year
        for _ in range(17280):
            sim.step()

        stats = sim.get_statistics()

        # These values should be deterministic
        assert stats['tick'] == 17280
        assert stats['year'] == 2  # Started at year 1, now year 2

        # Population should be within expected range (validated elsewhere)
        assert 20 <= stats['population'] <= 100


class TestTimeManagerJASMINEMode:
    """Verify JASMINE mode provides expected flexibility."""

    def test_variable_timestep(self):
        """JASMINE mode should allow timestep changes."""
        tm = TimeManager(mode=TimeMode.JASMINE, dt_seconds=60)
        assert tm.dt_seconds == 60

        tm.set_dt(30)
        assert tm.dt_seconds == 30

    def test_event_scheduling(self):
        """JASMINE mode should allow event scheduling."""
        tm = TimeManager(mode=TimeMode.JASMINE)

        events_fired = []
        tm.schedule_event(10, lambda: events_fired.append(10))
        tm.schedule_event(20, lambda: events_fired.append(20))

        for _ in range(25):
            for event in tm.get_scheduled_events():
                event()
            tm.advance()

        assert events_fired == [10, 20]

    def test_subsystem_update_frequencies(self):
        """Subsystems should update at configured frequencies."""
        tm = TimeManager(mode=TimeMode.JASMINE)

        # Default: movement every tick, food every 48 ticks
        assert tm.should_update('movement')  # tick 0
        assert not tm.should_update('food')   # food updates at tick 48

        # Advance to tick 48
        for _ in range(48):
            tm.advance()

        assert tm.should_update('movement')
        assert tm.should_update('food')
```

---

## Phase 2: Movement Model Integration

After Time Manager is stable, integrate DEPONS/JASMINE movement modules.

### 2.1 Movement Module Architecture

```
src/cenop/
├── movement/
│   ├── __init__.py
│   ├── base.py              # Abstract movement interface
│   ├── depons_crw.py        # DEPONS Correlated Random Walk
│   ├── jasmine_physics.py   # JASMINE physics-based movement
│   └── hybrid.py            # Hybrid selector
```

### 2.2 Movement Interface

```python
# src/cenop/movement/base.py
from abc import ABC, abstractmethod
import numpy as np

class MovementModule(ABC):
    """Abstract interface for movement modules."""

    @abstractmethod
    def compute_step(
        self,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        prev_log_mov: np.ndarray,
        depth: np.ndarray,
        salinity: np.ndarray,
        deterrence_dx: np.ndarray,
        deterrence_dy: np.ndarray,
        params: 'SimulationParameters'
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute movement step.

        Returns:
            Tuple of (new_x, new_y, new_heading)
        """
        pass
```

---

## Phase 3: Behavioral State Machine Integration

### 3.1 State Mapping

| DEPONS State | JASMINE State | Hybrid Behavior |
|--------------|---------------|-----------------|
| Foraging     | Foraging      | DEPONS movement |
| Traveling    | Transit       | JASMINE if >threshold |
| Resting      | Resting       | JASMINE |
| Disturbed    | Avoiding      | DEPONS deterrence |

### 3.2 Hybrid State Machine

```python
# src/cenop/behavior/hybrid_fsm.py
class HybridBehaviorFSM:
    """
    Hybrid finite state machine combining DEPONS and JASMINE behaviors.

    - DEPONS: Governs disturbance-driven transitions
    - JASMINE: Governs foraging, memory, sociality
    """

    def __init__(self, mode: TimeMode):
        self.mode = mode

    def update_state(self, agent_state, perception, params):
        """
        Update behavioral state.

        In DEPONS mode: Use DEPONS state transitions
        In JASMINE mode: Use JASMINE FSM with memory/sociality
        """
        if self.mode == TimeMode.DEPONS:
            return self._depons_transition(agent_state, perception)
        else:
            return self._jasmine_transition(agent_state, perception, params)
```

---

## Phase 4: Energy Budget Enhancement

### 4.1 Add JASMINE Energy Model

DEPONS has no explicit energy tracking. JASMINE adds:
- Dynamic energy budget (intake, cost of transport, metabolism)
- Fitness consequences of disturbance
- Cumulative impact modeling

```python
# src/cenop/physiology/energy_budget.py
class EnergyBudget:
    """
    Full dynamic energy budget (JASMINE extension).

    DEPONS mode: Simplified energy tracking (existing)
    JASMINE mode: Full DEB model
    """
    pass
```

---

## Phase 5: Memory & Cognition Integration

### 5.1 Extend PSM for JASMINE

Current PSM (Persistent Spatial Memory) supports:
- Memory of prey patches
- Dispersal targeting

JASMINE adds:
- Memory of disturbance zones
- Learned avoidance
- Memory decay with configurable rates

---

## Implementation Timeline

| Phase | Component | Estimated Effort | Dependencies |
|-------|-----------|------------------|--------------|
| 1.1   | TimeManager class | 1 session | None |
| 1.2   | Simulation integration | 1 session | 1.1 |
| 1.3   | Loop refactoring | 1 session | 1.2 |
| 1.4   | Regression tests | 1 session | 1.3 |
| 2.1-2.2 | Movement modules | 2 sessions | Phase 1 |
| 3.1-3.2 | State machine | 2 sessions | Phase 2 |
| 4.1   | Energy budget | 1 session | Phase 2 |
| 5.1   | Memory extension | 1 session | Phase 3 |

---

## Regression Testing Protocol

### Before Any Merge

1. Run all existing DEPONS validation tests:
   ```bash
   pytest tests/test_depons_*.py -v
   ```

2. Generate reference trajectories with current code:
   ```python
   # scripts/generate_reference_trajectories.py
   params = SimulationParameters(random_seed=42, porpoise_count=100)
   sim = Simulation(params)
   trajectories = run_and_record(sim, ticks=1000)
   save_reference(trajectories, 'reference_v0.1.0.npz')
   ```

3. After each change, compare:
   ```python
   # scripts/compare_trajectories.py
   ref = load_reference('reference_v0.1.0.npz')
   new = run_and_record(Simulation(params), ticks=1000)
   assert_trajectories_match(ref, new, tolerance=1e-10)
   ```

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Breaking DEPONS reproducibility | Regression tests before every commit |
| Performance degradation | Benchmark tests, vectorized operations |
| Complex merge conflicts | Incremental integration, feature flags |
| Regulatory rejection | Document all changes, maintain dual-mode |

---

## Next Steps

1. **Immediate**: Create `TimeManager` class (Step 1.1)
2. **This Session**: Integrate with Simulation (Steps 1.2-1.3)
3. **Validation**: Run regression tests (Step 1.4)
4. **Document**: Update CHANGELOG with merge progress

---

## Appendix: File Change Summary

### New Files
- `src/cenop/core/time_manager.py`
- `tests/test_time_manager.py`
- `docs/JASMINE_MERGE_PLAN.md` (this file)

### Modified Files
- `src/cenop/core/simulation.py` (TimeManager integration)
- `src/cenop/core/__init__.py` (exports)

### Unchanged Files
- All existing behavior modules (Phase 1)
- All existing tests (should still pass)
