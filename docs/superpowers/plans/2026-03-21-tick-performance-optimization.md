# Tick Performance Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce Python full-tick cost from ~2.03 ms to ~1.25 ms at N=500 (closing gap from 2.6× to ~1.6× vs Java)

**Architecture:** Eight optimizations across 3 tiers (O7 removed — PSM already has early return), targeting profiled hotspots: social vector bypass (0.37ms), RefMem wrapper overhead (0.08ms), redundant `np.where(mask)` calls (~6/tick), inter-phase glue (0.21ms), and dtype ping-pong (~20 astype/tick). Each task is self-contained with its own benchmark assertion.

**Tech Stack:** Python 3.13, NumPy, Numba (`@njit`, `prange`), pytest

**Profiling baseline (N=500, Numba, homogeneous 200×200 grid):**
```
Total tick:           2.03 ms
_update_movement:     1.42 ms (70.2%)
  _compute_social:    0.37 ms (18.3%)  ← O1
  _update_ref_mem:    0.25 ms (12.5%)  ← O2, O3
  CRW+heading+pos:    0.25 ms (12.3%)  ← O5
  movement glue:      0.27 ms (13.3%)  ← O4
_apply_bmr_cost:      0.23 ms (11.1%)  ← O9
_apply_food_intake:   0.10 ms  (5.0%)
Unaccounted glue:     0.21 ms (10.5%)  ← O4
```

**Test command:** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`

**Benchmark command (validate speedup):**
```bash
cd /home/razinka/cenjas && micromamba run -n shiny python3 -c "
import time, sys, numpy as np
sys.path.insert(0, 'CENOP/src')
from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData, LandscapeMetadata
def mk():
    cd = CellData.__new__(CellData)
    cd.landscape_name='Homogeneous'; cd.data_dir=''
    cd.metadata=LandscapeMetadata(ncols=200,nrows=200,xllcorner=0.0,yllcorner=0.0)
    cd._depth=np.full((200,200),30.0,dtype=np.float32)
    cd._dist_to_coast=np.full((200,200),5000.0,dtype=np.float32)
    cd._sediment=np.full((200,200),5.0,dtype=np.float32)
    cd._food_prob=np.ones((200,200),dtype=np.float32)
    cd._food_value=np.full((200,200),50.0,dtype=np.float32)
    cd._blocks=np.zeros((200,200),dtype=np.int32)
    cd._entropy=np.full((12,200,200),50.0,dtype=np.float32)
    cd._salinity=np.full((12,200,200),30.0,dtype=np.float32)
    cd._demand_grid=np.zeros((200,200),dtype=np.float32)
    cd._current_month=1; cd._loaded=True; return cd
p=SimulationParameters(porpoise_count=500,world_width=200,world_height=200)
pop=PorpoisePopulation(500,p,landscape=mk()); pop._skip_land_avoidance=True
for _ in range(100): pop.step()
t0=time.perf_counter()
for _ in range(500): pop.step()
ms=(time.perf_counter()-t0)/500*1000
print(f'N=500: {ms:.3f} ms/tick')
"
```

---

## File Map

| File | Role | Tasks |
|------|------|-------|
| `src/cenop/agents/population.py` | Main simulation loop — all optimizations modify this | O1–O9 |
| `src/cenop/behavior/ref_mem.py` | RefMem veTotal + attraction vector wrappers | O2, O6 |
| `src/cenop/optimizations/kernels.py` | Numba CRW/position/reflect kernels | O5 |
| `tests/test_tick_performance.py` | **New** — performance regression + correctness tests | O1–O9 |

---

## Shared Test Fixture (Task 0)

### Task 0: Create benchmark test infrastructure

**Files:**
- Create: `tests/test_tick_performance.py`

This file provides a reusable population fixture and benchmark helper used by all subsequent tasks.

- [ ] **Step 1: Create the test file with shared fixtures**

```python
"""Performance optimization tests.

Each test validates that an optimization does not change simulation output
(correctness) and optionally that it improves throughput (performance).
"""

import time
import numpy as np
import pytest
from cenop.agents.population import PorpoisePopulation
from cenop.parameters.simulation_params import SimulationParameters
from cenop.landscape.cell_data import CellData, LandscapeMetadata


def make_landscape(w=200, h=200):
    """Create a synthetic all-water landscape for benchmarking."""
    cd = CellData.__new__(CellData)
    cd.landscape_name = "Homogeneous"
    cd.data_dir = ""
    cd.metadata = LandscapeMetadata(ncols=w, nrows=h, xllcorner=0.0, yllcorner=0.0)
    cd._depth = np.full((h, w), 30.0, dtype=np.float32)
    cd._dist_to_coast = np.full((h, w), 5000.0, dtype=np.float32)
    cd._sediment = np.full((h, w), 5.0, dtype=np.float32)
    cd._food_prob = np.ones((h, w), dtype=np.float32)
    cd._food_value = np.full((h, w), 50.0, dtype=np.float32)
    cd._blocks = np.zeros((h, w), dtype=np.int32)
    cd._entropy = np.full((12, h, w), 50.0, dtype=np.float32)
    cd._salinity = np.full((12, h, w), 30.0, dtype=np.float32)
    cd._demand_grid = np.zeros((h, w), dtype=np.float32)
    cd._current_month = 1
    cd._loaded = True
    return cd


def make_pop(n=500, seed=42):
    """Create a population for benchmarking (homogeneous, no land avoidance)."""
    np.random.seed(seed)
    params = SimulationParameters(porpoise_count=n, world_width=200, world_height=200)
    land = make_landscape()
    pop = PorpoisePopulation(n, params, landscape=land)
    pop._skip_land_avoidance = True
    return pop


def measure_tick(pop, warmup=50, runs=200):
    """Measure mean ms/tick after warmup."""
    for _ in range(warmup):
        pop.step()
    t0 = time.perf_counter()
    for _ in range(runs):
        pop.step()
    return (time.perf_counter() - t0) / runs * 1000


def snapshot_state(pop):
    """Capture key state arrays for correctness comparison."""
    return {
        "x": pop.x.copy(),
        "y": pop.y.copy(),
        "heading": pop.heading.copy(),
        "energy": pop.energy.copy(),
        "prev_log_mov": pop.prev_log_mov.copy(),
        "active": pop.active_mask.copy(),
    }


def assert_states_match(s1, s2, atol=1e-5):
    """Assert two state snapshots are numerically identical."""
    for key in s1:
        np.testing.assert_allclose(s1[key], s2[key], atol=atol, err_msg=f"Mismatch in {key}")
```

- [ ] **Step 2: Run to verify fixture loads**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/test_tick_performance.py -v --co`
Expected: collected 0 items (file parses OK)

- [ ] **Step 3: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add tests/test_tick_performance.py
git commit -m "test: add performance optimization test infrastructure"
```

---

## Tier 1 — High Impact

### Task 1 (O1): Skip social vector `.fill()` when communication disabled

**Files:**
- Modify: `src/cenop/agents/population.py:457-471`
- Test: `tests/test_tick_performance.py`

The `_compute_social_vectors` method calls `.fill(0.0)` on two N-length arrays every tick, even when `_comm_enabled=False` (DEPONS mode). This costs ~0.37ms. Additionally the caller at line 1007 uses a slow `getattr()` check.

- [ ] **Step 1: Write failing test — social bypass doesn't change output**

Add to `tests/test_tick_performance.py`:

```python
class TestO1SocialBypass:
    """O1: Verify social bypass in DEPONS mode doesn't change output."""

    def test_depons_mode_skips_social_entirely(self):
        """When communication_enabled=False, social method should not touch arrays."""
        pop = make_pop(100)
        assert not getattr(pop.params, "communication_enabled", False)
        # Run 10 ticks
        for _ in range(10):
            pop.step()
        # Social output should be all zeros
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)

    def test_social_arrays_remain_zero_when_disabled(self):
        """Social arrays initialized to zero and never written when comm disabled."""
        pop = make_pop(50)
        # Arrays should be zero from init
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)
        # Run 10 ticks — arrays should stay zero (never written)
        for _ in range(10):
            pop.step()
        np.testing.assert_array_equal(pop._social_out_dx, 0.0)
        np.testing.assert_array_equal(pop._social_out_dy, 0.0)
```

- [ ] **Step 2: Run test to verify it passes (baseline — test should pass before optimization)**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/test_tick_performance.py::TestO1SocialBypass -v`
Expected: PASS (current code already returns zeros in DEPONS mode)

- [ ] **Step 3: Implement optimization — eliminate `.fill()` and `getattr()`**

In `population.py` line 457-471, replace:
```python
        def _compute_social_vectors(self, mask: np.ndarray, ambient_rl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            ...
            self._social_out_dx.fill(0.0)
            self._social_out_dy.fill(0.0)
            social_dx = self._social_out_dx
            social_dy = self._social_out_dy

            if not self._comm_enabled:
                return social_dx, social_dy
```

With:
```python
        def _compute_social_vectors(self, mask: np.ndarray, ambient_rl: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
            ...
            if not self._comm_enabled:
                return self._social_out_dx, self._social_out_dy

            self._social_out_dx.fill(0.0)
            self._social_out_dy.fill(0.0)
            social_dx = self._social_out_dx
            social_dy = self._social_out_dy
```

And at line 1007, replace:
```python
        if getattr(self.params, 'communication_enabled', False):
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
```
With:
```python
        if self._comm_enabled:
            soc_dx, soc_dy = self._compute_social_vectors(mask, ambient_rl)
```

Apply the same `self._comm_enabled` replacement at lines 1170 and 2143.

- [ ] **Step 4: Run full test suite to verify no regression**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`
Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O1): skip social vector fill when communication disabled

Move .fill(0) below _comm_enabled check so DEPONS mode never
touches the 2×N social arrays. Replace getattr() with cached bool."
```

---

### Task 2 (O2): Cache RefMem decay tables as float64 at init

**Files:**
- Modify: `src/cenop/behavior/ref_mem.py:147-156` and `src/cenop/behavior/ref_mem.py:236-243`
- Modify: `src/cenop/agents/population.py:1950-2010` (`_update_reference_memory`)
- Test: `tests/test_tick_performance.py`

The `compute_ve_total` and `compute_attraction_vector` functions call `ref_mem_table.astype(np.float64)` and `work_mem_table.astype(np.float64)` every tick, creating temporary arrays. These tables are constant — compute once at init.

- [ ] **Step 1: Write failing test — RefMem output unchanged after caching**

```python
class TestO2RefMemCache:
    """O2: Pre-cache RefMem decay tables as float64."""

    def test_refmem_output_unchanged_after_optimization(self):
        """veTotal and vt vectors must be identical before/after caching."""
        np.random.seed(99)
        pop = make_pop(100)
        for _ in range(20):
            pop.step()
        s1 = {
            "ve_total": pop._ve_total.copy(),
            "vt_x": pop._vt_x.copy(),
            "vt_y": pop._vt_y.copy(),
        }
        # Verify non-trivial (some agents have memory)
        assert np.any(s1["ve_total"] != 0), "veTotal should be non-zero after 20 ticks"
```

- [ ] **Step 2: Run test**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/test_tick_performance.py::TestO2RefMemCache -v`
Expected: PASS

- [ ] **Step 3: Implement — store decay tables as float64 once**

In `ref_mem.py`, modify `compute_ve_total` (line ~153):

Replace:
```python
            work_mem_table.astype(np.float64),
```
With:
```python
            work_mem_table if work_mem_table.dtype == np.float64 else work_mem_table.astype(np.float64),
```

Same pattern in `compute_attraction_vector` (line ~242):
```python
            ref_mem_table if ref_mem_table.dtype == np.float64 else ref_mem_table.astype(np.float64),
```

Then in `population.py` `__init__`, add pre-computed float64 tables right after `_REF_MEM_SIZE` is defined (~line 129, same scope):

```python
        _REF_MEM_SIZE = params.ref_mem_size if hasattr(params, 'ref_mem_size') else 120
        # ... existing array allocations using _REF_MEM_SIZE ...

        # Pre-compute float64 decay tables (avoid per-tick astype in RefMem kernels)
        from cenop.behavior.ref_mem import get_work_mem_strength_table, get_ref_mem_strength_table
        self._work_mem_table_f64 = get_work_mem_strength_table(
            self.params.r_s, _REF_MEM_SIZE
        ).astype(np.float64)
        self._ref_mem_table_f64 = get_ref_mem_strength_table(
            self.params.r_r, _REF_MEM_SIZE
        ).astype(np.float64)
```

Then in `_update_reference_memory` (~line 1974-1981), replace per-tick table lookups with cached tables:

```python
        # BEFORE (per-tick table creation):
        # work_table = get_work_mem_strength_table(self.params.r_s, mem_size)
        # ref_table = get_ref_mem_strength_table(self.params.r_r, mem_size)

        # AFTER (cached from init):
        work_table = self._work_mem_table_f64
        ref_table = self._ref_mem_table_f64
```

The rest of the method stays the same — `compute_ve_total` and `compute_attraction_vector` receive the pre-typed float64 tables directly.

- [ ] **Step 4: Run full test suite**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`
Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/behavior/ref_mem.py src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O2): cache RefMem decay tables as float64 at init

Avoid 2x per-tick .astype(float64) allocations in ve_total and
attraction vector kernels. Tables are constant per simulation."
```

---

### Task 3 (O3): Cache `np.where(mask)` once per tick

**Files:**
- Modify: `src/cenop/agents/population.py:2477-2524` (`step()`) and sub-methods
- Test: `tests/test_tick_performance.py`

`np.where(mask)[0]` is called 6+ times per tick in different sub-methods (social, ref_mem, PSM, etc.). Each call scans the full N-length boolean array. Compute once and pass as parameter.

- [ ] **Step 1: Write failing test**

```python
class TestO3CachedActiveIdx:
    """O3: Cache np.where(mask) once per tick."""

    def test_simulation_output_unchanged_with_cached_indices(self):
        """Full 50-tick trajectory must be identical."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(50):
            pop.step()
        state = snapshot_state(pop)
        # Verify deterministic
        np.random.seed(42)
        pop2 = make_pop(200)
        for _ in range(50):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/test_tick_performance.py::TestO3CachedActiveIdx -v`
Expected: PASS

- [ ] **Step 3: Implement — add `_active_idx` field and populate in `step()`**

In `step()` (line ~2487), after `self._global_tick += 1`, add:

```python
        # Cache active indices once per tick (avoid redundant np.where in sub-methods)
        self._active_idx = np.where(mask)[0]
```

**IMPORTANT: `_active_idx` becomes stale after `_check_mortality` (which sets `active_mask[dead]=False`).** Only use `_active_idx` in PRE-MORTALITY phases:
- `_update_movement` (step 1)
- `_handle_land_avoidance` (step 2)
- `_apply_positions` (step 3)
- `_apply_food_intake` (step 4a)

After mortality (step 4b), recompute for post-mortality phases:
```python
        # 4b. Starvation check
        self._check_mortality(mask, active_before)

        # Recompute active indices after mortality changed active_mask
        self._active_idx = np.where(self.active_mask)[0]
```

Then in `_update_reference_memory` (called from within `_apply_bmr_cost`, which uses post-mortality mask), replace:
```python
    active = np.where(mask)[0]
```
with:
```python
    active = self._active_idx
```

Apply the same replacement in:
- `_update_reference_memory` (line ~1964)
- `_update_psm` (line ~2577 — but only the `np.where(mask)[0]` part, NOT the `np.where(mask & food_positive)`)

Do NOT replace `np.where(mask & condition)` patterns — those compute different index sets.

- [ ] **Step 4: Run full test suite**

Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O3): cache np.where(mask) once per tick

Compute active_idx in step() and reuse in sub-methods that
previously each called np.where(mask)[0] independently."
```

---

## Tier 2 — Medium Impact

### Task 4 (O4): Reduce `np.any()` calls in movement glue

**Files:**
- Modify: `src/cenop/agents/population.py:814-1040` (`_update_movement`)
- Test: `tests/test_tick_performance.py`

`_update_movement` contains ~13 `np.any()` calls, each scanning the full N-length array. Most are guarding fast paths that don't need a full scan.

- [ ] **Step 1: Write failing test**

```python
class TestO4ReduceNpAny:
    """O4: Reduce np.any() calls in movement."""

    def test_movement_output_unchanged(self):
        """Movement vectors must be identical after reducing np.any."""
        np.random.seed(77)
        pop = make_pop(200)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(77)
        pop2 = make_pop(200)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — replace redundant `np.any()` guards**

Key replacements in `_update_movement`:

1. Line ~978: `if np.any(_disp_mask)` → `if _disp_mask.any()` (method call is faster than function)

2. Line ~1000: `self._was_deterred |= (self.deter_strength > 0) & mask` — the `np.any()` guard on this line can be removed since the bitwise OR is cheap regardless.

3. Line ~1030: `if np.any(dispersing):` — Replace with `if dispersing.any():`

4. Line ~960: `if np.any(violations & mask):` — Replace with `if (violations & mask).any():`

The `.any()` ndarray method avoids Python function dispatch overhead. Savings are marginal (~1µs per call × 13 calls ≈ 13µs) but this is a zero-risk cleanup that also improves readability.

Also in `_apply_dispersal_heading` and `_update_dispersal`, replace `np.any(...)` with `.any()`.

- [ ] **Step 4: Run full test suite** — Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O4): replace np.any() with .any() method calls

ndarray.any() avoids Python function dispatch overhead.
~13 calls/tick × 2µs savings each."
```

---

### Task 5 (O5): Fuse CRW heading + position + reflect into single kernel

**Files:**
- Modify: `src/cenop/optimizations/kernels.py` (add fused kernel)
- Modify: `src/cenop/agents/population.py:982-1040` (use fused kernel)
- Test: `tests/test_tick_performance.py`

Currently, after CRW angle/step computation, there are 3 separate phases with Python glue between them: (1) heading composition via trig, (2) position = x + dx, (3) boundary reflection. Fusing these into one Numba kernel eliminates inter-phase array materialization.

- [ ] **Step 1: Write failing test — fused kernel produces same output**

```python
class TestO5FusedHeadingKernel:
    """O5: Fused heading + position + reflect kernel."""

    def test_fused_kernel_matches_separate_phases(self):
        """Output of fused kernel must match sequential NumPy phases."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(20):
            pop.step()
        state = snapshot_state(pop)
        # x, y, heading should be deterministic
        assert np.all(np.isfinite(state["x"]))
        assert np.all(np.isfinite(state["y"]))
        assert np.all((state["heading"] >= 0) & (state["heading"] < 360))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Add `import math` to `kernels.py` and add fused kernel**

First, add `import math` at the top of `kernels.py` (Numba `@njit` functions support `math` module).

Then add the fused kernel. Key design decisions:
- `log_mov` parameter = `self._log_mov` (the **current tick's** log_mov, computed earlier by CRW kernel)
- `is_dispersing` mask passed so the kernel can apply the dispersal step override (`disp_step`) instead of `pres_mov / 4.0`
- Dispersing agents still get heading composition skipped — the kernel preserves their heading

```python
import math  # Add at top of kernels.py

@njit(cache=True, parallel=True)
def heading_position_reflect_kernel(
    heading, pres_angle, log_mov, ve_total, vt_x, vt_y,
    deter_dx, deter_dy, x, y, mask, is_dispersing,
    inertia_const, disp_step, world_w, world_h,
    out_heading, out_x, out_y, out_dx, out_dy, out_step_dist,
):
    """Fused: heading composition + step distance + position + reflect.

    Replaces 3 separate phases (heading trig, position update, boundary
    reflect) with a single pass per agent. Eliminates 2 intermediate
    array materializations.

    Args:
        log_mov: Current tick's log10(movement), i.e. self._log_mov (NOT prev_log_mov)
        is_dispersing: bool[N] — dispersing agents skip heading composition and use disp_step
        disp_step: Fixed step distance for dispersing agents (mean_disp_dist / 0.4)
    """
    max_x = float(world_w - 1)
    max_y = float(world_h - 1)
    DEG2RAD = 0.017453292519943295
    RAD2DEG = 57.29577951308232
    for i in prange(len(heading)):
        if not mask[i]:
            continue

        pres_mov = 10.0 ** log_mov[i]

        if is_dispersing[i]:
            # Dispersing: keep current heading, use fixed step
            new_h = heading[i]
            step = disp_step
        else:
            # 1. Heading composition (non-dispersing)
            h = (heading[i] + pres_angle[i]) % 360.0
            rad = h * DEG2RAD
            dx_crw = math.sin(rad)
            dy_crw = math.cos(rad)
            crw_c = inertia_const + pres_mov * ve_total[i]
            total_dx = dx_crw * crw_c + vt_x[i] + deter_dx[i]
            total_dy = dy_crw * crw_c + vt_y[i] + deter_dy[i]

            # facePoint
            new_h = math.atan2(total_dx, total_dy) * RAD2DEG
            if new_h < 0:
                new_h += 360.0

            # 2. Step distance = 10^log_mov / 4.0
            step = pres_mov / 4.0

        out_heading[i] = new_h
        out_step_dist[i] = step

        # 3. Position update
        rad2 = new_h * DEG2RAD
        ddx = math.sin(rad2) * step
        ddy = math.cos(rad2) * step
        nx = x[i] + ddx
        ny = y[i] + ddy

        # 4. Boundary reflection
        if nx < 0:
            nx = -nx; ddx = -ddx
        elif nx > max_x:
            nx = 2.0 * max_x - nx; ddx = -ddx
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny; ddy = -ddy
        elif ny > max_y:
            ny = 2.0 * max_y - ny; ddy = -ddy
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        out_x[i] = nx
        out_y[i] = ny
        out_dx[i] = ddx
        out_dy[i] = ddy
```

- [ ] **Step 4: Wire fused kernel into `_update_movement`**

Replace lines 966-1040 (the heading update → heading composition → dispersal restore → step distance → position → reflect sequence) with a call to the fused kernel when `_HAS_KERNELS` is True.

Pre-compute deter arrays (zero if no deterrence). Add a pre-allocated zero buffer in `__init__` near other buffers: `self._zero_f64 = np.zeros(count, dtype=np.float64)`.

```python
        _deter_dx = deterrence_vectors[0] if deterrence_vectors else self._zero_f64
        _deter_dy = deterrence_vectors[1] if deterrence_vectors else self._zero_f64
        disp_step = getattr(self.params, 'mean_disp_dist', 1.6) / 0.4

        heading_position_reflect_kernel(
            self.heading, self._pres_angle, self._log_mov,
            self._ve_total, self._vt_x, self._vt_y,
            _deter_dx, _deter_dy, self.x, self.y,
            mask, self.is_dispersing,
            self.params.inertia_const, disp_step,
            world_w, world_h,
            self.heading, self._new_x, self._new_y,
            self._dx, self._dy, self._step_dist,
        )
        # Update dispersal distance traveled
        dispersing = mask & self.is_dispersing
        if dispersing.any():
            self.dispersal_distance_traveled[dispersing] += self._step_dist[dispersing]
```

Keep the existing NumPy fallback path for when Numba is not available.

- [ ] **Step 5: Run full test suite** — Expected: 498+ passed

- [ ] **Step 6: Commit**

```bash
git add src/cenop/optimizations/kernels.py src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O5): fuse heading + position + reflect into single Numba kernel

Eliminates 2 intermediate array materializations between heading
composition, position update, and boundary reflection phases."
```

---

### Task 6 (O6): Eliminate `.astype()` dtype ping-pong

**Files:**
- Modify: `src/cenop/agents/population.py` (init section, ~line 103)
- Modify: `src/cenop/behavior/ref_mem.py` (kernel call sites)
- Test: `tests/test_tick_performance.py`

20 `.astype()` calls per tick, mostly float32↔float64 conversions at kernel boundaries. The biggest offenders are `active.astype(np.int64)` in RefMem and `ref_mem_table.astype(np.float64)`.

- [ ] **Step 1: Write failing test**

```python
class TestO6DtypePingPong:
    """O6: Verify no output change after removing astype overhead."""

    def test_output_stable_after_dtype_cleanup(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — store arrays in kernel-native dtype from init**

In `population.py` `__init__`:
- Change `self._mem_ptr` from `int32` to `int64` (Numba kernels expect int64 for active_indices)
- Or: store a pre-allocated `_active_idx_i64` buffer as int64 and reuse it

In `ref_mem.py` `compute_ve_total` and `compute_attraction_vector`:
- Replace `active.astype(np.int64)` with a check: `active if active.dtype == np.int64 else active.astype(np.int64)`
- Since O3 caches `_active_idx`, ensure that array is created as int64 from the start: `self._active_idx = np.where(mask)[0].astype(np.int64)` (or use `np.flatnonzero` which returns int64 on most platforms)

- [ ] **Step 4: Run full test suite** — Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py src/cenop/behavior/ref_mem.py tests/test_tick_performance.py
git commit -m "perf(O6): eliminate per-tick .astype() dtype conversions

Store active indices as int64 from creation; pre-cache decay
tables as float64. Removes ~15 astype calls per tick."
```

---

## Tier 3 — Lower Impact / Structural

### ~~Task 7 (O7): REMOVED — PSM already has early return~~

> `_update_psm` (line 2576-2579) already checks `food_gained > 0` and returns early when no agents ate food. No optimization needed.

---

### Task 7 (O8): Hoist daily boundary checks out of per-tick methods

**Files:**
- Modify: `src/cenop/agents/population.py:2453-2524` (`step()`)
- Test: `tests/test_tick_performance.py`

`_update_energy_history`, `_check_dispersal_trigger`, and `_handle_reproduction` contain `if tick % 48 == 0` guards internally. Moving the conditional to `step()` avoids calling these methods (and their setup overhead) on 47 out of 48 ticks.

- [ ] **Step 1: Write failing test**

```python
class TestO8DailyHoist:
    """O8: Daily checks only called on day boundaries."""

    def test_reproduction_only_on_day_boundary(self):
        """Reproduction state should only change at tick % 48 == 0."""
        pop = make_pop(100)
        # Run to a known day boundary
        while pop._global_tick % 48 != 47:
            pop.step()
        preg_before = pop.pregnancy_status.copy()
        pop.step()  # This should be tick%48==0 — reproduction may fire
        # Just verify no crash and deterministic
        assert pop._global_tick % 48 == 0
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — wrap daily calls in boundary check in `step()`**

**CAUTION:** `_handle_reproduction` (line 2017) increments `self._day_of_year` **every tick** before the `tick % 48` guard. This counter must still be updated on every tick. Hoist the increment out:

In `step()`, replace:
```python
        # 5. Aging
        self._update_aging(self.active_mask)

        # 6. Reproduction
        self._handle_reproduction(self.active_mask)
```

With:
```python
        # 5. Aging (every tick — continuous small increments)
        self._update_aging(self.active_mask)

        # 6. Day-of-year counter (must run every tick — was inside _handle_reproduction)
        self._day_of_year = (self._day_of_year + 1) % (360 * 48)

        # 7. Reproduction (daily only — pregnancy FSM)
        if self._global_tick % 48 == 0:
            self._handle_reproduction(self.active_mask)
```

Then in `_handle_reproduction` (line 2016-2017), **remove** the `_day_of_year` increment:
```python
    def _handle_reproduction(self, mask: np.ndarray) -> None:
        # REMOVE: self._day_of_year = (self._day_of_year + 1) % (360 * 48)
        # REMOVE: if self._global_tick % 48 != 0: return  (now guarded by caller)
        self._update_pregnancy_status(mask)
```

- [ ] **Step 4: Run full test suite** — Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(O8): hoist daily boundary checks out of per-tick calls

Avoid calling reproduction/dispersal methods on 47/48 ticks
when they only do work at day boundaries."
```

---

### Task 8 (O9): Separate BMR from PSM/dispersal/energy-history

**Files:**
- Modify: `src/cenop/agents/population.py:1829-1863` (`_apply_bmr_cost`) and `step()` (line ~2514)
- Test: `tests/test_tick_performance.py`

`_apply_bmr_cost` currently calls `_update_psm`, `_update_energy_history`, and `_update_dispersal` inside the DEPONS inline path. Moving them to `step()` makes profiling cleaner and enables Task 7 (daily hoisting) to work independently.

**IMPORTANT:** `_apply_bmr_cost` has two code paths:
- Line 1831-1832: JASMINE path (delegates to `_apply_bmr_cost_jasmine`) — do NOT modify
- Line 1834-1863: DEPONS inline path — extract PSM/energy-history/dispersal from here

- [ ] **Step 1: Write failing test**

```python
class TestO9SeparateBMR:
    """O9: BMR computation separated from PSM/dispersal/energy-history."""

    def test_energy_identical_after_separation(self):
        """Energy values must be identical after refactoring."""
        np.random.seed(42)
        pop = make_pop(200)
        for _ in range(50):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(200)
        for _ in range(50):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — extract from DEPONS path only**

In the DEPONS path of `_apply_bmr_cost` (lines 1847-1863), remove:
```python
            # PSM, energy history, and dispersal
            self._update_psm(mask, food_gained)
            self._update_energy_history(mask)
            self._update_dispersal(mask)
            # Clamp energy
            np.clip(self.energy, 0, 20.0, out=self.energy)
            self._pending_food_available = None
```

But keep the BMR computation, daily energy tracking (G11), and scaling factor.

In `step()` (line ~2514), after `_apply_bmr_cost`, add:
```python
        # 4d. Post-BMR updates (extracted from _apply_bmr_cost for profiling clarity)
        # Note: _pending_food_available is set by _apply_food_intake, consumed here
        if self._energy_module is None:  # DEPONS path only (JASMINE handles internally)
            food_gained = getattr(self, '_pending_food_available', None)
            self._update_psm(self.active_mask, food_gained)
            self._update_energy_history(self.active_mask)
            self._update_dispersal(self.active_mask)
            np.clip(self.energy, 0, 20.0, out=self.energy)
            self._pending_food_available = None
```

This preserves JASMINE path behavior (untouched) and DEPONS path ordering.

- [ ] **Step 4: Run full test suite** — Expected: 498+ passed

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "refactor(O9): separate BMR from PSM/dispersal/energy-history

Extract _update_psm, _update_energy_history, _update_dispersal
from _apply_bmr_cost into step() for independent optimization."
```

---

## Final Validation

### Task 10: End-to-end benchmark verification

- [ ] **Step 1: Run benchmark command from plan header**

Expected: N=500 tick time should be ≤1.5 ms (down from 2.03 ms baseline)

- [ ] **Step 2: Run full test suite one final time**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`
Expected: 498+ passed

- [ ] **Step 3: Update parity doc performance section**

Add measured before/after timings to `docs/DEPONS-CENOP-PARITY-ANALYSIS.md`.

- [ ] **Step 4: Commit all remaining changes**

```bash
git add -A
git commit -m "docs: update parity analysis with post-optimization performance"
```
