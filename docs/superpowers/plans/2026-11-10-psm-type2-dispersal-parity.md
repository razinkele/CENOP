# PSM-Type2 Dispersal Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make CENOP's PSM-Type2 dispersal reproduce DEPONS 3.2: trigger, target selection,
starting heading, turn angle, per-step stops and path-length completion (spec differences D1–D13).

**Architecture:**
- **Activation:** one entry point, `_start_dispersal(idx) -> bool`, called by the daily trigger.
  It picks a target the DEPONS way (memory cell, or a validated random point) and refuses to
  disperse when there is none.
- **Per tick:** one new method, `_disperse_step(mask)`, runs at the start of movement, as DEPONS
  `disperse()` runs before `stdMove()`. It checks completion, turns the PSM heading and applies
  the look-ahead stops. Animals that stop take a normal CRW step in the same tick; animals still
  dispersing keep their CRW state frozen.
- **JAX:** gets the same structure in its own kernels (Task 5).

**Tech Stack:** Python 3, NumPy, Numba (`src/cenop/optimizations/kernels.py`), JAX (optional
backend), pytest. Env: `micromamba run -n shiny`.

**Spec:** `docs/superpowers/specs/2026-11-10-phantom-food-investigation.md`, sections "Line-by-line
PSM-Type2 comparison", "Like-for-like rates", "Corrections combined" and "Decision needed" item 3.
DEPONS reference sources:
- `DEPONS-3.2/src/dk/au/bios/porpoise/Porpoise.java`: `move` 209–313, `performDailyStep`
  1073–1120;
- `behavior/AbstractPSMDispersal.java`;
- `behavior/DispersalPSMType2.java`.

## Global Constraints

- **DEPONS 3.2 runtime values** (from `DEPONS.rs/parameters.xml`, not Java field defaults):
  - tdisp 3, PSM_angle 40, PSM_dist N(350;100) km, PSM_tol 5 km, PSM_log 0.6;
  - ddisp 2 km, so a dispersal step is 5 cells; wdisp 4 m.
- **Existing params carry these values; use them, and add none:**
  - `t_disp=3`, `psm_angle=40.0`, `psm_tol=5.0`, `psm_log=0.6`;
  - `mean_disp_dist=2.0`, `min_depth_dispersal=4.0`.
- Do not read `psm_type2_random_angle` anywhere in `population.py`. That attribute belongs to the
  unwired `cenop.behavior.dispersal.DispersalParams`.
- **Cell lookup:** use CENOP's convention, `int` truncation of a position (cell i spans [i, i+1)),
  as `_handle_land_avoidance` does. Do not use DEPONS's `Math.round`.
- **Memory-cell centre:** `col * psm_cell_size + psm_cell_size / 2` (+2.5). In CENOP's convention
  this is the geometric equivalent of DEPONS's integer +2.
- **Out of scope:**
  - the not-enough-water first-call filter (needs D15's look-ahead state; separate plan);
  - the non-CRW movement modules (`_update_movement_module_generic`);
  - the unwired `cenop/behavior/dispersal.py` classes and their tests;
  - regenerating the Kattegat baselines.
- **Repo conventions:**
  - line length 100 (black + ruff); no new type annotations or docstrings on unchanged code;
  - run pytest from `/home/razinka/cenjas/CENOP`;
  - `git -C /home/razinka/cenjas/CENOP …`; commit messages via `-F <file>`; no `$()`;
  - work on local branch `fix/phantom-food-2026-11-10`; do not push.
- **Fast suite:** `micromamba run -n shiny python3 -m pytest tests/ -q`.
- **Slow tier:** add `-m slow`.

## Review Focus

1. **A disperser at the grid edge** must stop, not raise `IndexError`; out-of-grid points are
   decided before any depth lookup. Task 4: `test_stops_before_leaving_grid`.
2. **Populations with `landscape=None`** (unit tests, homogeneous fallback) must work: random
   targets use `params.world_width/height`, and the depth and prey checks are skipped. Task 1:
   `test_random_target_without_landscape`; Task 4: `test_stops_before_leaving_grid`.
3. **A failed activation** must leave every dispersal field untouched, so no half-initialised
   disperser reaches the heading code. Task 1: `test_failed_activation_leaves_agent_untouched`.
4. **A target distance of 0** (or a tiny random distance) must not produce NaN headings or a
   division by zero; completion fires on the first step. Task 4:
   `test_zero_target_distance_completes_without_nan`.
5. **An inactive slot** still flagged `is_dispersing` (dead, or awaiting recycling) is untouched
   by `_disperse_step`. Task 4: `test_inactive_slots_ignored`.

---

### Task 1: DEPONS target selection and activation (D3, D4, D5, D6)

**Files:**
- Modify: `src/cenop/agents/population.py`:
  - `_check_dispersal_trigger` (3105–3151): remove the 50-cell gate and the batch init;
  - `_start_dispersal` (3225–3238);
  - `_select_dispersal_target` (3240–3320);
  - delete `_set_random_dispersal_target` (3322–3357).
- Create: `tests/test_dispersal_parity.py`
- Modify: `tests/test_dispersal.py` (`TestDispersalBatchInit`); `tests/test_integration.py`
  (`test_dispersal_heading_dampening`, `test_dispersal_target_selection`)

**Interfaces:**
- Produces:
  - `_memory_dispersal_target(self, idx: int) -> tuple[float, float] | None`
  - `_random_dispersal_target(self, idx: int) -> tuple[float, float] | None`
  - `_select_dispersal_target(self, idx: int) -> tuple[float, float] | None`: the memory target,
    else the random one. Writes no state.
  - `_start_dispersal(self, idx: int) -> bool`. On `None` it returns False and changes nothing.
    On success it sets:
    - `is_dispersing = True`, `dispersal_start_x/y = x/y`, `dispersal_target_x/y`;
    - `dispersal_target_distance` = straight-line distance to the target (not ×0.95);
    - `dispersal_distance_traveled = 0`;
    - `_prev_step_heading` = compass heading to the target, `degrees(arctan2(dx, dy)) % 360`.

    It does **not** write `heading`. DEPONS `activate()` sets only `previousStepHeading`, and
    `disperse()` restores the animal's own heading on a stop. So an activation cancelled the
    same day, or stopped on its first step, must leave the CRW heading as it was.
  - `_check_dispersal_trigger` calls `_start_dispersal(i)` for each declining, non-dispersing
    active agent.

**Selection rules (DEPONS `findMostAttractiveMemCell` / `activate`):**
- **Memory target:**
  - visited cells are `psm_buffer[idx, :, :, 0] > 0`; fewer than 50 means `None`;
  - candidate band: (pref − tol)/0.4 ≤ d ≤ (pref + tol)/0.4 cells, inclusive, where
    pref = `_psm_instances[idx].preferred_distance` (km) and tol = `params.psm_tol`;
  - fitness = food / ticks, with no `q1` weighting; the first maximum wins, and zero fitness is
    allowed;
  - no candidate means `None`.
- **Random target:**
  - draw 1000 tries in one call: heading U(0, 360)°, distance (pref + U(−tol, tol))/0.4 cells,
    target = position + distance·(sin h, cos h);
  - a try is valid iff 0 ≤ tx < W and 0 ≤ ty < H, and, when the landscape has them, both
    `_depth[int ty, int tx] > 0` and `_entropy[month − 1][int ty, int tx] > 0`, with
    month = `self._get_current_month()`;
  - the first valid try wins; none means `None`;
  - W and H are `landscape.width/height`, or `params.world_width/height` when
    `landscape is None`.

- [ ] **Step 1: Write the failing tests** in `tests/test_dispersal_parity.py`

Shared helpers at the top of the file (Tasks 2–4 reuse them):
```python
def _pop(count=4, landscape=None, **kw):
    params = SimulationParameters(random_seed=1, communication_enabled=False, **kw)
    return PorpoisePopulation(count, params, landscape)

def _disperser(pop, i, x, y, prev_heading, target_dist=500.0, traveled=0.0):
    pop.x[i], pop.y[i] = x, y
    pop.is_dispersing[i] = True
    pop.dispersal_start_x[i], pop.dispersal_start_y[i] = x, y
    pop.dispersal_target_distance[i] = target_dist
    pop.dispersal_distance_traveled[i] = traveled
    pop._prev_step_heading[i] = prev_heading
    pop.heading[i] = prev_heading
```
Tests in `class TestTargetSelection` (agent 0 at x = y = 100.5, off cell boundaries so float trig
cannot flip a cell; `preferred_distance = 20.0` km, which puts the band at [37.5, 62.5] cells):
- `test_memory_target_needs_50_cells`:
  - mark 49 cells visited, all in the band → `pop._memory_dispersal_target(0) is None`;
  - with 50 → not `None`.
- `test_memory_target_best_food_per_tick_in_band`:
  - 60 visited cells: PSM rows 0–5, cols 0–9 (out of band), fitness 10;
  - (row 20, col 30) fitness 3.0 and (row 30, col 20) fitness 2.0, both about 52 cells away;
  - assert the result `== (152.5, 102.5)`.
- `test_random_target_requires_water_and_prey`:
  - `create_homogeneous_landscape(width=200, height=200)` with `_entropy[:] = 0` →
    `_random_dispersal_target(0) is None`;
  - then `_entropy[:, :, 150:] = 0.5` → the result's x ≥ 150.
- `test_random_target_without_landscape`:
  - `landscape=None`, agent at (500, 500), pref 20 → target ~37.5–62.5 cells away;
  - pref 10000 → `None`.
- `test_failed_activation_leaves_agent_untouched`:
  - pref 10000 → `_start_dispersal(0) is False`;
  - `is_dispersing[0]` is False; `dispersal_target_distance[0]`, `_prev_step_heading[0]` and
    `heading[0]` equal their prior values.
- `test_activation_heads_for_target` (D6):
  - on success, `_prev_step_heading[0] == degrees(arctan2(tx − x, ty − y)) % 360`
    (tolerance 1e-4), and `heading[0]` is unchanged;
  - `dispersal_target_distance[0] == hypot(tx − x, ty − y)`.
- `test_trigger_not_gated_on_memory` (D3):
  - empty `psm_buffer`, `_energy_history[0, :4] = [4, 5, 6, 7]`, pref 20, agent at (500, 500);
  - `_check_dispersal_trigger(pop.active_mask)` → `is_dispersing[0]` is True.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py -q`
Expected: FAIL (`AttributeError: … _memory_dispersal_target`; trigger test fails on the 50-cell gate).

- [ ] **Step 3: Implement the four methods and the trigger change in `population.py`**

The 1000 random tries are one vectorised draw:
- `self.rng.random(1000)` for headings and for distances;
- `np.argmax(valid)` picks the first valid try.

- [ ] **Step 4: Update the three existing tests that relied on `pref ≈ 350 km` in a 1000-cell world**

Set `pop._psm_instances[i].preferred_distance = 20.0` before the trigger or `_start_dispersal`:
- `tests/test_dispersal.py::TestDispersalBatchInit::test_batch_dispersal_matches_individual`
  (agents 0–4);
- `tests/test_integration.py::test_dispersal_heading_dampening` and
  `::test_dispersal_target_selection` (agent 0).

- [ ] **Step 5: Run the new tests and the touched files**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py tests/test_dispersal.py tests/test_integration.py -q`
Expected: all pass, except `test_integration.py::…::test_dispersal_completion`, which may fail
until Task 4 (it relies on displacement completion).

- [ ] **Step 6: Commit**

Message: `fix(dispersal): DEPONS PSM-Type2 target selection; head for the target (D3-D6)`.

---

### Task 2: Trigger needs `t_disp` strict drops; daily means rounded to 3 d.p. (D1, D2)

**Files:**
- Modify: `src/cenop/agents/population.py`: `_check_dispersal_trigger` (declining loop) and
  `_update_energy_history` (3188–3223)
- Test: `tests/test_dispersal_parity.py`

**Interfaces:**
- Consumes: `_start_dispersal` (Task 1).
- Produces:
  - declining means `history[:, i] < history[:, i + 1]` for all `i in range(t_disp)`, with
    `t_disp` clamped to `[1, history_len − 1]`;
  - `_energy_history[:, 0]` holds `np.round(daily_sum / 48.0, 3)`.

- [ ] **Step 1: Write the failing tests** (`class TestTrigger`; agent 0 at (500, 500), pref 20, `landscape=None`)
- `test_trigger_requires_three_strict_drops`, parametrised over `history[0, :4]`:
  - `[4, 5, 6, 7]` → dispersing;
  - `[4, 5, 6, 6]` → not;
  - `[4, 5, 5, 6]` → not.
- `test_daily_mean_rounded_to_3dp`:
  - call `_update_energy_history(mask)` 48 times, incrementing `pop._global_tick` before each
    call (the method skips repeat calls within one tick);
  - `energy[0] = 5.0004` gives `_energy_history[0, 0] == np.float32(5.0)`;
  - `5.0006` gives `np.float32(5.001)`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py::TestTrigger -q`
Expected: FAIL; `[4, 5, 6, 6]` disperses under the current two-drop loop, and the mean is unrounded.

- [ ] **Step 3: Change the loop bound to `range(t_disp)` and round `daily_avg` before the shift**

- [ ] **Step 4: Run the new tests and `tests/test_dispersal.py`**

Expected: PASS.

- [ ] **Step 5: Commit**

Message: `fix(dispersal): trigger needs t_disp strict drops; round daily mean to 3 dp (D1, D2)`.

---

### Task 3: PSM turn uses `psm_angle` and path length (D7, D8)

**Files:**
- Modify: `src/cenop/agents/population.py`:
  - `_apply_dispersal_heading` (1088–1134);
  - the JAX argument at 2580 becomes `float(self.params.psm_angle)`.
- Modify: `src/cenop/ui/tabs/settings.py`:
  - line 428 default `20.0` → `40.0`;
  - line 36 tooltip "DEPONS default: 20°" → "DEPONS default: 40°".
- Test: `tests/test_dispersal_parity.py`; `tests/test_settings_defaults.py`

**Interfaces:**
- Produces: `_apply_dispersal_heading(mask)`, unchanged signature. Task 4 calls it from
  `_disperse_step`. The new turn:
  - `delta = U(−psm_angle, psm_angle) · logistic(3·p − 1.5)`, with
    `p = dispersal_distance_traveled / (0.95 · dispersal_target_distance)`, and p = 0 where the
    target distance is ≤ 0;
  - `logistic(x) = 1 / (1 + exp(−x / psm_log))`.

- [ ] **Step 1: Write the failing tests**

`class TestPsmTurn` uses 2000 dispersing agents built with `_disperser`; delta is the change in
heading, wrapped to (−180, 180]:
- `test_turn_uses_psm_angle` (D7):
  - `traveled = displacement = 0.95·T` (both formulas give p ≈ 1, so logistic ≈ 0.924);
  - assert `max|delta| > 25` (impossible with 20°) and `≤ 40·0.925`.
- `test_logistic_uses_path_length` (D8):
  - displacement 0 (x equal to start), traveled 0.95·T;
  - assert `max|delta| > 25`; displacement-based input would cap it near 3°.
- In `tests/test_settings_defaults.py`, `test_ui_psm_angle_default_is_40`: mirror
  `TestPSMDistDefaults.test_ui_and_controller_defaults_are_350` (`tests/test_dispersal.py:474`)
  for `psm_angle == 40`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py::TestPsmTurn tests/test_settings_defaults.py -q`
Expected: FAIL (`max|delta|` ≈ 18.5 and ≈ 3; UI default is 20).

- [ ] **Step 3: Implement the formula change and the UI default**

- [ ] **Step 4: Run the tests again and confirm they pass**

- [ ] **Step 5: Commit**

Message: `fix(dispersal): PSM turn uses psm_angle (40) and path length (D7, D8)`.

---

### Task 4: `_disperse_step`: completion, look-ahead stops, same-tick CRW, frozen CRW state (D9–D13)

**Files:**
- Modify: `src/cenop/agents/population.py`:
  - add `_disperse_step` and `_end_dispersal`;
  - `_update_movement` (910–1086): call `_disperse_step` first; give CRW `crw_mask`;
  - `_update_movement_jasmine` (CRW-module branch, 1180–1261): the same;
  - `_update_dispersal` (3359–3403): remove distance completion; use `_end_dispersal`.
- Test: `tests/test_dispersal_parity.py`
- Modify:
  - `tests/test_depons_movement.py::…::test_dispersal_completion` (238–260);
  - `tests/test_integration.py::…::test_dispersal_completion` (650–671).
  Both become path-length tests via `_disperse_step`.

**Interfaces:**
- Consumes: `_apply_dispersal_heading` (Task 3).
- Produces:
  - `_end_dispersal(self, sel: np.ndarray) -> None` sets `is_dispersing = False`,
    `dispersal_distance_traveled = 0` and `days_declining_energy = 0` for `sel`.
  - `_disperse_step(self, mask: np.ndarray) -> None`, applied in DEPONS order to
    `d = mask & is_dispersing`:
    1. **Completion:** `dispersal_distance_traveled ≥ 0.95 · dispersal_target_distance` →
       `_end_dispersal`.
    2. **Turn:** save `heading` for the rest of `d`, then `_apply_dispersal_heading`; this also
       updates `_prev_step_heading`, as DEPONS does even when a stop follows.
    3. **Look-ahead point:** p = position + (mean_disp_dist / 0.4)·(sin h, cos h).
    4. **Off-grid stop:** `px < 0 | py < 0 | px ≥ W − 0.5 | py ≥ H − 0.5`.
    5. **Depth stop:** only when the landscape has `_depth`; `depth[int py, int px] ≤
       min_depth_dispersal`, or NaN, stops. Indices are clipped, and off-grid agents are
       already decided by step 4.
    6. **Stopped agents:** `_end_dispersal`, and `heading` is restored to the saved value.
- **Changes in `_update_movement`:**
  - call `_disperse_step(mask)` *after* the `_movement_module` dispatch at the top (line 925–927),
    so it runs exactly once on each path and never on the out-of-scope generic module path;
  - `crw_mask = mask & ~self.is_dispersing`, evaluated after `_disperse_step`;
  - the CRW kernel and `generate_crw_angle_step` get `crw_mask`;
  - `self.heading[crw_mask] += self._pres_angle[crw_mask]`;
  - delete the `_apply_dispersal_heading(mask)` call;
  - the `prev_angle` and `prev_log_mov` writes use `crw_mask` (D13).
  The heading kernel's `is_dispersing` branch (fixed 5-cell step) is unchanged. A disperser's
  destination now always has depth > 4 m and lies inside the grid, so land avoidance and
  reflection never act on it (D11).
- **`_update_movement_jasmine` (CRW-module branch only):** `_disperse_step(mask)` before
  `compute_step`, then the same edits, with `compute_step(…, crw_mask)`.

- [ ] **Step 1: Write the failing tests**

`class TestDisperseStep`:
- Fixtures:
  - `psm_angle=0.0`, so the heading equals `_prev_step_heading`;
  - `create_homogeneous_landscape(200, 200)`; disperser at (100.5, 100.5) with heading 90 (east),
    so the look-ahead point is (105.5, 100.5), cell `[100, 105]`.
- `test_stops_before_shallow_water`:
  - `_depth[100, 105] = 3.0` → after `_disperse_step(mask)`, not dispersing, heading 90,
    traveled 0;
  - `_depth = 30` → still dispersing.
- `test_stops_before_leaving_grid`: with `landscape=None`, an agent at (998.5, 500.5) heading 90
  stops.
- `test_completes_on_path_length`:
  - traveled = 0.95·T with displacement 0 → stops in `_disperse_step`;
  - displacement ≥ 0.95·T with traveled 0 → `_update_dispersal` leaves it dispersing.
- `test_zero_target_distance_completes_without_nan`: T = 0 → stopped; `np.isfinite(heading).all()`.
- `test_inactive_slots_ignored`: with `active_mask[1] = False` and `is_dispersing[1] = True`,
  after the step `is_dispersing[1]` is still True and the heading is unchanged.
- `test_stopped_disperser_takes_crw_step_same_tick` (D12):
  - after `pop._recompute_cell_indices()` and `pop._update_movement(pop.active_mask, None, None)`,
    the stopped agent has `_step_dist != approx(5.0)`;
  - a disperser in deep water still has `_step_dist == approx(5.0)`.
- `test_crw_state_frozen_while_dispersing` (D13): for the continuing disperser, `prev_angle[i]`
  and `prev_log_mov[i]` are unchanged by `_update_movement`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py::TestDisperseStep -q`
Expected: FAIL (`AttributeError: … _disperse_step`).

- [ ] **Step 3: Implement `_end_dispersal`, `_disperse_step` and the edits to `_update_movement`, `_update_movement_jasmine` and `_update_dispersal`**

- [ ] **Step 4: Rewrite the two old completion tests to the path-length rule; run every test file that touches dispersal**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py tests/test_dispersal.py tests/test_depons_movement.py tests/test_integration.py tests/test_movement_parity.py tests/test_movement_modules.py tests/test_ship_deterrence_port.py tests/test_reproduction.py -q`
Expected: all pass. `test_movement_parity` checks that the inline and CRW-module paths agree.

- [ ] **Step 5: Run the fast suite and the slow tier**

Run:
- `micromamba run -n shiny python3 -m pytest tests/ -q`
- `micromamba run -n shiny python3 -m pytest tests/ -m slow -q`

Expected:
- **Fast:** passes, apart from `tests/test_jax_tick.py` failures caused only by JAX-vs-NumPy
  dispersal reference comparisons (fixed in Task 5). Record the list in the commit message.
- **Slow:** 11 pass. The slow population tests run on small homogeneous grids, where targets
  ~875 cells away rarely exist, so dispersal there will mostly stop. If a slow test fails,
  **stop and report** the test, its assertion and the values; do not loosen tolerances.

- [ ] **Step 6: Commit**

Message: `fix(dispersal): per-step DEPONS stops, path-length completion, same-tick CRW (D9-D13)`.

---

### Task 5: Mirror the per-tick changes in the JAX backend (D2, D7–D13)

**Files:**
- Modify: `src/cenop/optimizations/jax_kernels.py`:
  - add `jax_disperse_step`;
  - `jax_heading_composition` (273–437): remove the dispersal override (lines 342–375); keep
    the dispersers' fixed step and their skipped composition; mask the `prev_angle` update with
    non-dispersers;
  - `jax_energy_history_update` (806–857): `jnp.round(daily_avg, 3)`;
  - `jax_dispersal_update` (860+): remove distance completion.
- Modify: `src/cenop/optimizations/tick_jax.py`: `jax_tick_movement` (47–260) calls
  `jax_disperse_step` before `jax_crw_kernel`; the CRW mask is `mask & ~is_dispersing`.
- Modify: `src/cenop/agents/population.py` `_step_jax` (≈2434–2720): plumb the new inputs and
  outputs.
- Test: `tests/test_jax_tick.py`

**Interfaces:**
- `jax_disperse_step(x, y, heading, prev_step_heading, is_dispersing, dispersal_distance_traveled,
  dispersal_target_distance, mask, psm_angle, psm_log, mean_disp_dist, depth_grid,
  min_depth_dispersal, world_w, world_h, key) -> (heading, prev_step_heading, is_dispersing,
  dispersal_distance_traveled)`
- Semantics are identical to Task 4's `_disperse_step`, steps 1–6. With `psm_angle = 0` the
  result is deterministic and must equal NumPy's.

- [ ] **Step 1: Write the failing tests** (in `tests/test_jax_tick.py`; skip if JAX is unavailable, as the existing tests do)
- `test_jax_disperse_step_matches_numpy`:
  - `psm_angle=0`; homogeneous 200×200 landscape; agents placed to cover each case (deep water,
    shallow ahead, off-grid ahead, path complete, inactive);
  - `is_dispersing`, `heading` and `dispersal_distance_traveled` from `jax_disperse_step` equal
    `_disperse_step` on a copy.
- `test_jax_energy_history_rounded_to_3dp`: the same values as Task 2.
- Update the existing dispersal-heading reference tests (`test_dispersal_heading_matches_reference`
  ≈600, and the one at ≈1542) to call `jax_disperse_step` and use the path-length `p`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py -q`

- [ ] **Step 3: Implement the JAX changes and the plumbing in `_step_jax`**

- [ ] **Step 4: Run `tests/test_jax_tick.py`, then the fast suite**

Expected: all pass.

- [ ] **Step 5: Commit**

Message: `fix(jax): mirror DEPONS PSM-Type2 per-tick dispersal in the JAX backend`.

---

### Task 6: Acceptance against DEPONS, docs

**Files:**
- Modify: `scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py`
- Modify: `CHANGELOG.md` (`[Unreleased]`); `docs/superpowers/specs/2026-11-10-phantom-food-investigation.md`

- [ ] **Step 1: Adapt the probe to the fixed code**
- **Count moves, not flags.** Starts, moves and tick share must count only animals still
  dispersing *after* `_disperse_step` returns, i.e. those that actually make a dispersal move
  that tick. That matches DEPONS rows. The current main loop reads `is_dispersing` before
  `sim.step()`, and in the fixed code the stops run inside the step, before the move. Left as it
  is, it would count a first-step stop as a one-move episode and read every episode one move too
  long. On the seed-42 probe numbers it would report about 0.175 activation for a correct port.
  Fix: have the `_disperse_step` wrapper store `mask & is_dispersing` after the call, and use that
  as the per-tick `d0`.
- **Attribute ends** in the same wrapper: reason 1 (off-grid or shallow ahead) or 3 (path
  complete), by comparing `is_dispersing` before and after.
- State in the docstring that only the `stock` and `nodisp` modes are valid on the fixed code; the
  other modes re-apply fixes that are now in the source.

- [ ] **Step 2: Run the acceptance set** (v3.2 landscape copy under v3.0 names, as in the spec's Reproduce section)

Run, in parallel:
```bash
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> stock 180 120 42
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> stock 180 120 43
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> nodisp 180 120 42
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> nodisp 180 120 43
```
Acceptance, on both seeds. DEPONS: activation 0.133, median 5 / mean 12.1, share 1.34%, on/off
ratio at day 150 0.80 / 0.81.

| Metric | Accepted range |
|---|---|
| activation | 0.10–0.16 |
| episode moves | median 4–6, mean 8–15 |
| tick share | 0.8–1.8% |
| on/off N ratio at day 150 | 0.75–0.87 |

If a metric falls outside its range, stop and report; do not tune.

- [ ] **Step 3: Run the slow tier and lint**

Run:
- `micromamba run -n shiny python3 -m pytest tests/ -m slow -q`
- `~/.local/bin/black --check -l 100 src tests validation`
- `~/.local/bin/ruff check src tests validation`

Expected: 11 slow pass (as after Task 4; stop and report on any failure); lint clean.

- [ ] **Step 4: Record the outcome in the docs**
- **CHANGELOG** `[Unreleased]` → Changed: PSM-Type2 dispersal now follows DEPONS 3.2:
  - trigger after `t_disp` strict drops of the 3-d.p. daily mean;
  - validated targets, with no dispersal when none is found;
  - starts heading for the target;
  - ±`psm_angle` turn on path length;
  - stops when the point 5 cells ahead is off the grid or ≤ `wdisp` deep;
  - completion on path length.
  Add that this is a behaviour change, and that the Kattegat baselines are stale until
  regenerated.
- **Spec:** under "Decision needed" item 3, record the commits and the acceptance table.

- [ ] **Step 5: Commit**

Message: `docs(dispersal): acceptance vs DEPONS; CHANGELOG behaviour change`.
