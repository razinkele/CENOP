# Ship-Deterrence Sub-Tick Interpolation + Scalar-TL Consistency — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the production vectorized ship-deterrence path evaluate 30 interpolated ship positions per tick along the ship's within-tick swept path (DEPONS `interpolateStep` parity), and make the test-only scalar aggregator's transmission loss consistent with the production path via a shared received-level helper.

**Architecture:** Five tasks. (1) Extract a shared `_ship_received_level` helper (pure refactor of the vectorized RL block). (2) Route the scalar oracle through it (fixes the alpha/beta-vs-WestonFlux TL inconsistency). (3) Track each ship's start-of-tick position (`_prev_x/_prev_y`). (4) Replace the single-position vectorized aggregator with a 30-slot `recordStep` accumulator (max received-level ship per slot, summed over slots = DEPONS `deterrenceVtX/Y`), pre-culling candidates by the swept-segment midpoint. (5) Regenerate the Kattegat ship baseline.

**Tech Stack:** Python 3, NumPy, Numba (`@njit` helpers), pytest. Source: `src/cenop/agents/ship.py`. Tests: `tests/test_ship_deterrence_port.py`. Spec: `docs/superpowers/specs/2026-06-08-ship-deterrence-subtick-design.md`.

**Environment:** Run all commands from `/home/razinka/cenjas/CENOP/`. Prefix Python/pytest with `micromamba run -n shiny`. CENOP is a nested git repo — commit from inside `CENOP/`. Branch off `CENOP-JASMINE`. Commit messages end with `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

**Key DEPONS facts (from the spec):**
- `interpolateStep` builds 30 positions `start + (end-start)·i/30` for **i = 1..30** (excludes start, includes end).
- `recordStep(k, …)`: per porpoise, per slot `k∈0..29`, keep the ship with the **maximum received level** at that slot. Recording happens only for **gated** steps (`RL > Tships`); the stored vector is `unit×mag×Bernoulli` (so a gated-but-not-reacting loud ship occupies the slot with a **zero** vector, blocking a quieter ship).
- `deterrenceVtX/Y` = **sum** of the 30 slots → the tick's ship-deterrence vector. Downstream persistence/decay/strength in `population.py` is unchanged.

**Test-impact note (do not skip):** existing aggregator tests assert *invariants* (sign / zero / equality / ratio), and the test ships are constructed without calling `update()`, so `_prev == cur` → a degenerate segment → 30 identical slots. With `_force_u=0.0` that yields exactly 30× the single-position vector, preserving every existing invariant. `test_kernel_snapshot_day` tests the kernel directly and is unaffected. **Therefore no existing test should need magnitude edits.** If any existing test in `tests/test_ship_deterrence_port.py` fails after Task 4 for a reason other than an intended semantic change, treat it as a real regression and investigate — do not silently rewrite its expected values.

**File Structure:**
- `src/cenop/agents/ship.py` — the only source file modified:
  - new module-level `_ship_received_level(...)` helper (after `_compute_tl_percell`).
  - `Ship.__post_init__` / `Ship.update` — `_prev_x/_prev_y`.
  - `ShipManager.calculate_aggregate_deterrence` — shared-helper TL.
  - `ShipManager.calculate_aggregate_deterrence_vectorized` — 30-slot sub-tick aggregation.
- `tests/test_ship_deterrence_port.py` — new test classes appended.
- `output/kattegat_ref_ships/` — regenerated baseline (Task 5).

---

### Task 1: Shared received-level helper (pure refactor)

Extract the vectorized path's RL computation into a module-level helper, then make the vectorized path call it. No behavior change — existing tests stay green.

**Files:**
- Modify: `src/cenop/agents/ship.py` (add helper after `_compute_tl_percell` at line ~58; refactor `calculate_aggregate_deterrence_vectorized` RL block at lines ~512-534)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ship_deterrence_port.py`:

```python
class TestSharedReceivedLevel:
    def test_non_weston_uses_alpha_beta(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        dist_m = np.array([2000.0])
        px = np.array([50.0]); py = np.array([50.0])
        rl = _ship_received_level(180.0, dist_m, px, py, p,
                                  cell_data=None, month=1, weston=False)
        expected = 180.0 - (p.beta_hat * np.log10(2000.0) + p.alpha_hat * 2000.0)
        assert rl[0] == pytest.approx(max(0.0, expected))

    def test_weston_nodata_gives_zero(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        p = SimulationParameters(); p.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        land._depth[:] = -9999.0  # all NODATA depth
        dist_m = np.array([2000.0])
        px = np.array([50.0]); py = np.array([50.0])
        rl = _ship_received_level(210.0, dist_m, px, py, p,
                                  cell_data=land, month=1, weston=True)
        assert rl[0] == 0.0

    def test_clamped_non_negative(self):
        import numpy as np
        from cenop.agents.ship import _ship_received_level
        from cenop.parameters.simulation_params import SimulationParameters
        p = SimulationParameters()
        # Tiny source level, large distance -> negative before clamp
        rl = _ship_received_level(10.0, np.array([9000.0]), np.array([0.0]),
                                  np.array([0.0]), p, cell_data=None, month=1, weston=False)
        assert rl[0] == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestSharedReceivedLevel -q`
Expected: FAIL — `ImportError: cannot import name '_ship_received_level'`.

- [ ] **Step 3: Add the helper**

Insert after the `_compute_tl_percell` njit-wrap block (after line 62, before `if TYPE_CHECKING:`):

```python
def _ship_received_level(source_level, dist_m, px, py, params, cell_data, month, weston):
    """Received level (dB, clamped >= 0) at the given porpoise positions for one ship.

    WestonFlux per-cell when `weston`, else simple alpha/beta TL. NODATA on
    depth/grain/salinity OR TL <= 0 -> RL 0 (DEPONS Ship.java:296-307 + valueIsNoData).
    All array args are the in-range subset; `source_level` is a scalar.
    """
    if weston:
        pos = np.column_stack((px, py))
        depths = cell_data.get_depths_vectorized(pos)
        grains = cell_data.get_sediments_vectorized(pos)
        sal = cell_data.get_salinities_vectorized(pos, month)
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
```

- [ ] **Step 4: Refactor the vectorized path to call the helper**

In `calculate_aggregate_deterrence_vectorized`, replace the RL block (currently lines ~512-534, from `source_level = ship.noise.get_source_level()` through `rl_sub = np.maximum(rl_sub, 0.0)`) with:

```python
            source_level = ship.noise.get_source_level()
            rl_sub = _ship_received_level(
                source_level, d_sub, porpoise_x[idx], porpoise_y[idx],
                params, cell_data, month, weston)
```

(The `weston` flag is already computed earlier in the method at line ~495. Leave the rest of the method unchanged in this task.)

- [ ] **Step 5: Run tests to verify green (new + regression)**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py tests/test_weston_flux.py -q`
Expected: PASS (new helper tests pass; `test_nodata_cell_gives_zero_rl_weston` and all others still pass — behavior unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "refactor: extract shared _ship_received_level helper

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 2: Scalar aggregator TL consistency

Route `ShipManager.calculate_aggregate_deterrence` (test-only oracle) through `_ship_received_level` so it honors WestonFlux when enabled, matching production. Keep it single-position; update its docstring.

**Files:**
- Modify: `src/cenop/agents/ship.py:398-451` (`calculate_aggregate_deterrence`)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ship_deterrence_port.py`:

```python
class TestScalarAggregatorTL:
    def _mgr(self, sl=205.0):
        from cenop.agents.ship import Ship, ShipManager, VesselClass
        s = Ship(id=1, x=50.0, y=50.0, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = sl
        mgr = ShipManager([s]); mgr.enabled = True
        return mgr, s

    def test_scalar_uses_weston_when_enabled(self):
        """Scalar aggregator RL must use WestonFlux (per-cell) when enabled, not alpha/beta."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.landscape.cell_data import create_homogeneous_landscape
        from cenop.agents.ship import _ship_received_level
        mgr, s = self._mgr()
        p = SimulationParameters(); p.weston_flux_percell = True
        land = create_homogeneous_landscape(width=100, height=100, depth=20.0, food_prob=0.5)
        px, py = 50.0 + 2000.0 / 400.0, 50.0
        # Force a reaction so the path is non-vacuous.
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0
        try:
            mag_w, dx_w, dy_w = mgr.calculate_aggregate_deterrence(
                px, py, p, is_day=True, cell_data=land, month=1)
        finally:
            np.random.random = orig
        # Expected RL via the shared helper (WestonFlux) must drive a non-zero vector.
        weston = True
        dist_m = np.array([2000.0])
        rl = _ship_received_level(s.noise.get_source_level(), dist_m,
                                  np.array([px]), np.array([py]), p, land, 1, weston)
        assert rl[0] > p.deter_ships_min_db        # precondition: gated in
        assert dx_w != 0.0                         # scalar produced deterrence via WestonFlux

    def test_scalar_without_celldata_uses_alpha_beta(self):
        """No cell_data -> alpha/beta TL (unchanged legacy behavior)."""
        import numpy as np
        from cenop.parameters.simulation_params import SimulationParameters
        from cenop.agents.ship import _ship_received_level
        mgr, s = self._mgr()
        p = SimulationParameters()
        px, py = 50.0 + 2000.0 / 400.0, 50.0
        orig = np.random.random
        np.random.random = lambda *a, **k: 0.0
        try:
            mag, dx, dy = mgr.calculate_aggregate_deterrence(px, py, p, is_day=True)
        finally:
            np.random.random = orig
        rl = _ship_received_level(s.noise.get_source_level(), np.array([2000.0]),
                                  np.array([px]), np.array([py]), p, None, 1, False)
        # Same RL the scalar path should now use -> reacts (non-vacuous, alpha/beta).
        assert rl[0] > p.deter_ships_min_db
        assert dx != 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestScalarAggregatorTL -q`
Expected: `test_scalar_uses_weston_when_enabled` FAILS — the current scalar path ignores `cell_data`/WestonFlux. (`test_scalar_without_celldata_uses_alpha_beta` may already pass.)

- [ ] **Step 3: Implement the TL fix**

In `calculate_aggregate_deterrence`, (a) update the docstring NOTE, and (b) replace the per-ship RL line. The method body loops active ships; before the loop add the `weston` flag, and inside replace `rl = max(0.0, ship.get_received_level(...))` with the shared helper.

Replace the docstring NOTE paragraph (lines ~411-416) with:

```python
        NOTE: NOT on the production tick path (Simulation.step uses
        calculate_aggregate_deterrence_vectorized). This per-porpoise oracle is a
        SINGLE-POSITION + loudest-ship oracle (no sub-tick interpolation). Its RL now
        flows through the shared _ship_received_level helper, so it honors
        weston_flux_percell exactly like the vectorized path. It is deliberately NOT a
        sub-tick oracle; do not use it to validate sub-tick aggregation.
```

Add the `weston` flag immediately after `tships = getattr(params, "deter_ships_min_db", 80.0)` (line ~433):

```python
        weston = (params.weston_flux_percell and cell_data is not None
                  and getattr(cell_data, "_sediment", None) is not None)
```

Replace the RL line (currently lines ~441-442):

```python
            rl = max(0.0, ship.get_received_level(
                porpoise_x, porpoise_y, params.alpha_hat, params.beta_hat, cell_size))
```

with:

```python
            source_level = ship.noise.get_source_level()
            rl = float(_ship_received_level(
                source_level, np.array([dist_m]), np.array([porpoise_x]),
                np.array([porpoise_y]), params, cell_data, month, weston)[0])
```

- [ ] **Step 4: Run tests to verify green**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestScalarAggregatorTL tests/test_deterrence.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "fix: scalar ship aggregator uses shared RL helper (WestonFlux parity)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Track ship start-of-tick position

Add `_prev_x/_prev_y` to `Ship`, initialized in `__post_init__` and set as the first statement of `update()` (before any early return).

**Files:**
- Modify: `src/cenop/agents/ship.py` (`Ship.__post_init__` lines ~177-183; `Ship.update` line ~191)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_ship_deterrence_port.py`:

```python
class TestShipPrevPosition:
    def test_post_init_sets_prev_to_initial(self):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=0, x=7.0, y=9.0, vessel_type=VesselClass.CARGO)
        assert s._prev_x == 7.0 and s._prev_y == 9.0

    def test_update_records_pre_move_position(self):
        from cenop.agents.ship import Ship, Route, Buoy, VesselClass
        route = Route(buoys=[Buoy(x=0.0, y=0.0, speed=10.0),
                             Buoy(x=100.0, y=0.0, speed=10.0)])
        s = Ship(id=1, x=0.0, y=0.0, route=route, vessel_type=VesselClass.CARGO)
        s.tick_start = 0; s.tick_end = 100
        s.update(1)
        assert (s._prev_x, s._prev_y) == (0.0, 0.0)   # start-of-tick position
        assert s.x != 0.0                              # moved toward the next buoy

    def test_update_inactive_leaves_prev_equal_current(self):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=1, x=5.0, y=5.0, vessel_type=VesselClass.CARGO)
        s.tick_start = 10; s.tick_end = 20   # inactive at tick 1
        s.update(1)
        assert (s._prev_x, s._prev_y) == (5.0, 5.0)
        assert (s.x, s.y) == (5.0, 5.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestShipPrevPosition -q`
Expected: FAIL — `AttributeError: 'Ship' object has no attribute '_prev_x'`.

- [ ] **Step 3: Implement `_prev_x/_prev_y`**

In `__post_init__` (after the `self.noise = ShipNoise(...)` block, line ~183), add:

```python
        self._prev_x = self.x
        self._prev_y = self.y
```

In `update`, make the **first statement** (immediately after the `def update(self, current_tick: int) -> None:` docstring, before `# Check if active`):

```python
        # Start-of-tick position for sub-tick swept-path deterrence (set before any
        # early return so paused/inactive ships keep prev == current).
        self._prev_x, self._prev_y = self.x, self.y
```

- [ ] **Step 4: Run test to verify it passes**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestShipPrevPosition -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "feat: track ship start-of-tick position for sub-tick interpolation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Sub-tick aggregation in the vectorized path

Replace the single-position loop in `calculate_aggregate_deterrence_vectorized` with a 30-slot `recordStep` accumulator over the within-tick swept path, pre-culling candidates by the segment midpoint.

**Files:**
- Modify: `src/cenop/agents/ship.py:453-555` (`calculate_aggregate_deterrence_vectorized`)
- Test: `tests/test_ship_deterrence_port.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_ship_deterrence_port.py`:

```python
class TestSubTickInterpolation:
    def _params(self):
        from cenop.parameters.simulation_params import SimulationParameters
        return SimulationParameters()

    def _ship(self, sid, x, y, prev=None, sl=205.0):
        from cenop.agents.ship import Ship, VesselClass
        s = Ship(id=sid, x=x, y=y, vessel_type=VesselClass.CARGO)
        s._is_active = True; s.noise.base_source_level = sl
        if prev is not None:
            s._prev_x, s._prev_y = prev
        else:
            s._prev_x, s._prev_y = x, y
        return s

    def _kernel_vec(self, s, px, py, p, sub_x, sub_y, cell=400.0):
        """Single-slot kernel vector for a ship sub-position (force react), using the same
        non-WestonFlux RL the implementation uses: source_level - (beta*log10(d) + alpha*d)."""
        import numpy as np
        gdx = np.array([px - sub_x]); gdy = np.array([py - sub_y])
        dist_m = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
        tl = p.beta_hat * np.log10(dist_m[0]) + p.alpha_hat * dist_m[0]
        rl = np.array([max(0.0, float(s.noise.get_source_level() - tl))])
        vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
            rl, dist_m, gdx, gdy, True, np.array([0.0]), p.deter_ships_min_db)
        return float(vx[0]), float(vy[0])

    def test_stationary_ship_is_30x_single_position(self):
        """prev == cur -> 30 identical slots -> total == 30 x single-position vector (force_u=0)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        s = self._ship(1, 50.0, 50.0)   # prev == cur
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0 + 2000.0/400.0]); py = np.array([50.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        vx1, vy1 = self._kernel_vec(s, px[0], py[0], p, 50.0, 50.0)
        assert dx[0] == pytest.approx(30.0 * vx1)
        assert dy[0] == pytest.approx(30.0 * vy1)

    def test_moving_ship_sums_distinct_substep_vectors(self):
        """Total equals the slot-wise sum over i=1..30 sub-positions (force_u=0)."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        # Swept path 40->60 east along y=50; porpoise north at (50, 60).
        s = self._ship(1, 60.0, 50.0, prev=(40.0, 50.0))
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([50.0]); py = np.array([60.0])
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        exp_x = exp_y = 0.0
        for i in range(1, 31):
            sub_x = 40.0 + (60.0 - 40.0) * i / 30.0
            sub_y = 50.0
            vx, vy = self._kernel_vec(s, px[0], py[0], p, sub_x, sub_y)
            exp_x += vx; exp_y += vy
        assert exp_y > 0.0                  # non-vacuous: some slots gated + reacting
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)
        assert dy[0] > 0.0   # net push north, away from the east-west path

    def test_substep_endpoints_exclude_start_include_end(self):
        """i=1..30: first sub-position is start+delta/30, last is exactly the end position."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        # A porpoise exactly at the START point would only be hit if a slot sat on start.
        # Slot 30 sits on the END; place porpoise offset from END and confirm a contribution
        # consistent with the end being included (compare against explicit i=1..30 sum).
        s = self._ship(1, 10.0, 0.0, prev=(0.0, 0.0))
        mgr = ShipManager([s]); mgr.enabled = True
        px = np.array([10.0]); py = np.array([5.0])   # near the END (10,0), north
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)
        exp_x = exp_y = 0.0
        for i in range(1, 31):
            sub_x = 0.0 + 10.0 * i / 30.0
            vx, vy = self._kernel_vec(s, px[0], py[0], p, sub_x, 0.0)
            exp_x += vx; exp_y += vy
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)

    def test_per_slot_max_rl_ship_wins(self):
        """Different ships win different slots (DEPONS recordStep). Aggregator must match a
        brute-force per-slot max-RL+sum reference, and the winner set must include BOTH ships."""
        import numpy as np
        from cenop.agents.ship import ShipManager, VesselClass
        p = self._params()
        # Asymmetric crossing so the distance curves cross inside i=1..30:
        #   A approaches  (dist_A = 6 - i/6, from ~5.83 down to 1 cell),
        #   B recedes     (dist_B = 1 + i/6, from ~1.17 up to 6 cells).
        # B is closer for i<15 (B wins), A is closer for i>15 (A wins) -> both win slots.
        A = self._ship(1, 49.0, 50.0, prev=(44.0, 50.0), sl=195.0)   # approaching
        B = self._ship(2, 56.0, 50.0, prev=(51.0, 50.0), sl=195.0)   # receding
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([A, B]); mgr.enabled = True
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)

        # Brute-force reference: per slot keep max-RL ship's vector, then sum.
        cell = 400.0
        def slot_rl_vec(s, i):
            sub_x = s._prev_x + (s.x - s._prev_x) * i / 30.0
            sub_y = s._prev_y + (s.y - s._prev_y) * i / 30.0
            gdx = np.array([px[0] - sub_x]); gdy = np.array([py[0] - sub_y])
            dist_m = np.array([max(float(np.hypot(gdx[0]*cell, gdy[0]*cell)), 1.0)])
            rl = max(0.0, float(s.noise.get_source_level()
                                - (p.beta_hat*np.log10(dist_m[0]) + p.alpha_hat*dist_m[0])))
            vx, vy, _, _, _ = s.deterrence_model.deterrence_components(
                np.array([rl]), dist_m, gdx, gdy, True, np.array([0.0]), p.deter_ships_min_db)
            return rl, float(vx[0]), float(vy[0])
        exp_x = exp_y = 0.0; winners = set()
        for i in range(1, 31):
            ra, vax, vay = slot_rl_vec(A, i)
            rb, vbx, vby = slot_rl_vec(B, i)
            if ra > p.deter_ships_min_db or rb > p.deter_ships_min_db:
                if ra >= rb:
                    exp_x += vax; exp_y += vay; winners.add(1)
                else:
                    exp_x += vbx; exp_y += vby; winners.add(2)
        assert winners == {1, 2}                    # both ships win some slots
        assert dx[0] == pytest.approx(exp_x)
        assert dy[0] == pytest.approx(exp_y)

    def test_gated_nonreacting_winner_contributes_zero(self):
        """A gated ship that does NOT react stores a zero vector in its slots (DEPONS
        recordStep keeps the max-RL step with deterX=0 when reactingOrNot=0). With a uniform
        non-reacting draw the total is exactly zero even though every slot is gated in.
        Combined with test_per_slot_max_rl_ship_wins (winner = max RL), this composes to the
        DEPONS 'loud non-reacting ship occupies the slot, blocking a quieter one' behavior."""
        import numpy as np
        from cenop.agents.ship import ShipManager
        p = self._params()
        loud = self._ship(1, 49.0, 50.0, sl=205.0)   # gated in (RL >> Tships)
        px = np.array([50.0]); py = np.array([50.0])
        mgr = ShipManager([loud]); mgr.enabled = True
        dx, dy = mgr.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=1.0)  # never react
        assert dx[0] == 0.0 and dy[0] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestSubTickInterpolation -q`
Expected: FAIL — the current single-position aggregator returns the single-slot vector (≈ 1× not 30×; no per-slot behavior).

- [ ] **Step 3: Rewrite the vectorized aggregator**

Replace the entire body of `calculate_aggregate_deterrence_vectorized` (lines ~482-555, from `n = porpoise_x.shape[0]` through `return (total_dx, total_dy)`) with:

```python
        n = porpoise_x.shape[0]
        total_dx = np.zeros(n, dtype=np.float64)
        total_dy = np.zeros(n, dtype=np.float64)
        if not self.enabled:
            return (total_dx, total_dy)
        active_ships = self.get_active_ships()
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

        for ship in active_ships:
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
            # Reaction draws: full (STEPS, n) stream seeded per (base_seed, tick, ship.id),
            # indexed by GLOBAL porpoise index -> invariant to ship order/count. Only the
            # marginal Bernoulli RATE matches DEPONS (global draw order is unreproducible).
            if _force_u is None:
                rng = np.random.default_rng(
                    np.random.SeedSequence([base_seed, tick, int(ship.id)]))
                u_all = rng.random((STEPS, n))
            else:
                u_all = None

            px_c = porpoise_x[cand]
            py_c = porpoise_y[cand]
            for k in range(STEPS):
                gdx = px_c - sub_x[k]
                gdy = py_c - sub_y[k]
                dist_m = np.hypot(gdx * cell_size, gdy * cell_size)
                np.maximum(dist_m, 1.0, out=dist_m)
                inr = (dist_m > min_dist_m) & (dist_m <= max_dist_m)
                if not inr.any():
                    continue
                sub = cand[inr]                # global porpoise indices in range at slot k
                d_k = dist_m[inr]
                rl_k = _ship_received_level(
                    source_level, d_k, porpoise_x[sub], porpoise_y[sub],
                    params, cell_data, month, weston)
                if _force_u is None:
                    u_k = u_all[k][sub]
                else:
                    u_k = np.full(sub.size, float(_force_u), dtype=np.float64)
                vx, vy, _, _, _ = ship.deterrence_model.deterrence_components(
                    rl_k, d_k, gdx[inr], gdy[inr], is_day, u_k, tships)
                # Loudest gated ship wins this slot; its vector is 0 if it did not react.
                gated = rl_k > tships
                wins = gated & (rl_k > best_rl[sub, k])
                sel = sub[wins]
                best_rl[sel, k] = rl_k[wins]
                accum_dx[sel, k] = vx[wins]
                accum_dy[sel, k] = vy[wins]

        total_dx = accum_dx.sum(axis=1)
        total_dy = accum_dy.sum(axis=1)
        return (total_dx, total_dy)
```

Also update the method docstring (lines ~466-481) to describe the 30-substep swept-path behavior. Replace the first paragraph with:

```python
        """Aggregate DEPONS ship deterrence over active ships with 30-substep
        within-tick interpolation (Ship.java interpolateStep).

        For each ship, 30 sub-positions are interpolated along the ship's within-tick
        swept path (start-of-tick `_prev_x/_prev_y` -> end-of-tick `x/y`), positions
        start + (end-start)*i/30 for i=1..30. Per porpoise, per sub-step slot, the
        ship with the maximum received level wins (ShipDeterrence.recordStep); the 30
        slots are summed into the returned vector (deterrenceVtX/Y). A gated ship that
        does not react occupies its slot with a zero vector.
```

(Keep the existing `_force_u` and per-ship-seed NOTE paragraphs that follow.)

- [ ] **Step 4: Run new tests to verify they pass**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py::TestSubTickInterpolation -q`
Expected: PASS.

- [ ] **Step 5: Run the full ship/deterrence regression (unfiltered)**

Run: `micromamba run -n shiny python3 -m pytest tests/test_ship_deterrence_port.py tests/test_deterrence.py tests/test_weston_flux.py tests/test_integration.py -q`
Expected: PASS. If a pre-existing test fails, confirm it is an intended semantic change per the "Test-impact note" (it should NOT be — investigate before editing any expected value).

- [ ] **Step 6: Run the broad fast suite**

Run: `micromamba run -n shiny python3 -m pytest tests/ -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py`
Expected: PASS (the two ignored files hang pre-existing and are excluded).

- [ ] **Step 7: Commit**

```bash
git add src/cenop/agents/ship.py tests/test_ship_deterrence_port.py
git commit -m "feat: 30-substep sub-tick ship deterrence interpolation (DEPONS parity)

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Regenerate the Kattegat ship baseline

Sub-tick interpolation raises ship-deterrence magnitudes (integration over the swept path), so the committed ship baseline must be regenerated.

**Files:**
- Regenerate: `output/kattegat_ref_ships/Population.txt`, `Energy.txt`, `Mortality.txt`, `PROVENANCE.txt`
- Reference: `scripts/run_kattegat_reference.py`

- [ ] **Step 1: Record the current commit for provenance**

Run: `git rev-parse --short HEAD`
Note the SHA for the PROVENANCE file.

- [ ] **Step 2: Regenerate the baseline**

Run: `micromamba run -n shiny python3 scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships`
Expected: completes; population remains stable (~2000); prints a nonzero `deter_strength` event count.

- [ ] **Step 3: Update PROVENANCE.txt**

Edit `output/kattegat_ref_ships/PROVENANCE.txt`: set the date to 2026-06-08, the producing commit to the Task-4 SHA, and add a paragraph:

```
This baseline reflects the sub-tick interpolation change (30-substep within-tick
swept-path ship deterrence, DEPONS interpolateStep parity). Deterrence is now
integrated over each ship's within-tick swept path (max-received-level ship per
sub-step slot, summed over 30 slots), so per-tick ship-deterrence magnitudes are
higher than the single-end-of-tick-position prior baseline. Report the new nonzero
deter_strength event count and max below.
```

Fill in the actual event count / max / population endpoints printed by Step 2.

- [ ] **Step 4: Commit**

```bash
git add output/kattegat_ref_ships/
git commit -m "data: regenerate Kattegat ship baseline for sub-tick interpolation

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>"
```

---

## Self-Review

**Spec coverage:**
- §3.1 swept-path source → Task 3.
- §4.1 `_prev` first-statement / post_init init → Task 3 (all three tests).
- §4.2 sub-tick aggregation (i=1..30, per-slot max-RL, sum, gated-non-reacting blocks) → Task 4 (all tests).
- §4.3 RNG `(30,n)` per-ship seed → Task 4 implementation + existing `test_seed_order_invariance_still_holds` / `test_tick_varies` cover invariance.
- §4.4 shared RL helper + scalar TL fix → Tasks 1 & 2.
- §5 test impact + baseline regen → Task 4 Steps 5-6, Task 5.
- §6 files → matches File Structure.

**Placeholder scan:** no TBD/TODO; every code step shows complete code; commands are exact.

**Type/name consistency:** `_ship_received_level(source_level, dist_m, px, py, params, cell_data, month, weston)` signature identical across Tasks 1, 2, 4. `_prev_x/_prev_y` consistent across Tasks 3, 4. `STEPS=30`, `t_frac`, `best_rl/accum_dx/accum_dy` defined and used within Task 4. `deterrence_components(rl, dist_m, gdx, gdy, is_day, u, tships)` argument order matches existing usage.

**Note on the perf pre-cull:** the midpoint candidate radius `max_dist + 0.5·seg_len` is a correct *superset* (any in-range porpoise at any slot is within `max_dist` of a segment point, hence within `max_dist + half_len` of the midpoint), so it cannot drop a porpoise that the brute-force reference includes — `test_per_slot_max_rl_ship_wins` and `test_moving_ship_sums_distinct_substep_vectors` validate the implementation against unculled reference loops.
