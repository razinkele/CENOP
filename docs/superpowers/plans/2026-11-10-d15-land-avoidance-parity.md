# D15 Land-Avoidance Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace CENOP's post-composition, destination-only land check with DEPONS 3.2's
land-avoidance sequence (spec difference D15), and add the D9 first-call water filter that
sequence makes possible.

**Architecture:**
- **One shared land check:** a single implementation, `depons_land_check` in a new module
  `src/cenop/movement/land_check.py`, written as plain loops so it runs under Numba `@njit` when
  available and as Python otherwise.
- **Where it runs:** `PorpoisePopulation._depons_land_check(mask)` calls it right after the CRW
  turn and before heading composition, in both CRW movement paths. It also records each animal's
  not-enough-water-ahead state, which `_disperse_step` then reads.
- **After composition:** on the CRW paths, only border reflection remains. The existing post-move
  rollback gains DEPONS's heading flip.
- **JAX:** `jax_kernels.py` gets the same sequence.

**Tech Stack:** Python 3, NumPy, Numba, JAX (optional backend), pytest. Env: `micromamba run -n shiny`.

**Spec:** `docs/superpowers/specs/2026-11-10-phantom-food-investigation.md`, "Finding 4" (Steps 1–4,
remainders) and the D9/D15 rows of the line-by-line table. The in-memory prototype that the
acceptance numbers come from is `scripts/investigations/2026-11-10-phantom-food/d15_land_port.py`
(commit 46e2a39). DEPONS reference, `DEPONS-3.2/src/dk/au/bios/porpoise/Porpoise.java`:
- `stdMove` 320–666: CRW turn `incHeading` at 400, land loop 467–533, composition 535–577, beh 6
  639–661;
- `checkDepth` 795–833;
- `avoidLandTurn` 885–904;
- `isPointGood` 906–916;
- `avoidLand` 918–985;
- `AbstractPSMDispersal.disperse` 166–176 (not-enough-water stop).

## Global Constraints

- **DEPONS land sequence, exact values:**
  - Look-ahead distance = `presMov` cells (`presMov = 10**log_mov`), i.e. 4× the step `presMov/4`.
  - **Path check (`checkDepth`):**
    - the point `presMov` cells ahead must have depth ≥ 0;
    - every point at 0.1-cell spacing, k = 1 … ceil(presMov / 0.1), must have depth > 0.
  - **Escape points (`avoidLand`):**
    - one U(0, 10) jitter `r` per call;
    - try ±(40 + r), then ±(70 + r), then ±(120 + r) at `presMov` cells ahead;
    - a point is good iff depth ≥ `params.min_depth` (wmin, 1.0);
    - if both sides are good, turn toward the deeper one (right on ties); if one is good, turn
      toward it (right = +, left = −).
  - **beh 4:** no tier good → walk back through the newest ≤ 20 remembered positions, facing and
    stepping onto each, until the path check passes.
  - **Rounds:** at most 100, then one final path check.
  - **beh 5:** if the final check fails, face the centre of the deepest of the 8 neighbouring
    cells, and set `log_mov = 0` (presMov = 1).
- **Cell lookup:** `floor` of the position. Out of grid or NaN counts as −9999 (DEPONS `getDepth`).
- **Remembered positions for beh 4:** CENOP's reference-memory ring buffer
  (`_pos_history_x/_y`, `_mem_ptr`, `_mem_count`). The land check runs after
  `_update_reference_memory`, which stores the current position (population.py ~2295), so the
  newest entry is DEPONS's `posList.get(0)`, and walking back from it (j = 0, 1, …, ≤ 20) follows
  DEPONS exactly.
- **Order (DEPONS `move`: `refMemTurn` before `stdMove`):** reference memory, then the CRW turn
  (in either order: reference memory is RNG-free and heading-independent), then the land check,
  then composition. Both CRW paths must use this order.
- **After composition, on CRW paths:**
  - no land check, only border reflection;
  - post-move: depth ≤ 0 → return to the pre-move position (existing) and face back toward it
    (beh 6, new).
- **D9 first-call filter:** in `_disperse_step`, directly after the PSM turn, a disperser whose
  stored not-enough-water state is set stops: heading restored, `_end_dispersal`. The state is
  written only for animals that are not dispersing, so it is frozen while dispersing, as in DEPONS.
- **Unchanged, out of scope:**
  - homogeneous landscapes and `landscape=None` (`_skip_land_avoidance` is True: no land check,
    the state stays True);
  - the Cython fast path (it runs only when `_skip_land_avoidance`);
  - non-CRW movement modules (`_update_movement_module_generic`), which keep today's
    post-composition land avoidance and rollback without the flip;
  - the energy-undershoot and late-death remainders;
  - grid units.
- **Randomness:** draw the land check's jitter from Numba's RNG, seeded once per call from
  `self.rng` with the existing `_seed_numba_rng`.
  - The kernel loops over agents **sequentially, with no `prange`**: per-thread RNG streams would
    break the seeding.
  - The Python fallback seeds NumPy's global RNG (`np.random.seed`) from `self.rng` the same way.
  - Both CRW paths make the seeding call at the same point, so `tests/test_movement_parity.py`
    keeps holding.
- **JAX on homogeneous grids:** `_step_jax` today runs its post-composition `jax_land_avoidance`
  against the Homogeneous grid's land border, unlike NumPy, which skips it. Keep that unchanged:
  when `_skip_land_avoidance` is True, JAX keeps today's path and skips the new check.
- **Repo conventions:**
  - line length 100 (black + ruff); no new annotations or docstrings on unchanged code;
  - pytest from `/home/razinka/cenjas/CENOP`;
  - `git -C`; commit messages via `-F <file>`; no `$()`;
  - local branch `fix/phantom-food-2026-11-10`; do not push.
- **Fast suite:** `micromamba run -n shiny python3 -m pytest tests/ -q`.
- **Slow tier:** add `-m slow`. JAX tests: prefix `JAX_PLATFORMS=cpu`.

## Review Focus

1. **Look-ahead points off the grid** (animal near the edge, long `presMov`) count as land and
   trigger avoidance, never an `IndexError`. Task 1: `test_offgrid_lookahead_is_land`.
2. **NaN depth cells** count as land on both the path and the escape points. Task 1:
   `test_nan_depth_is_land`.
3. **The longest look-ahead** (`log_mov = max_mov = 1.73`, about 537 samples) must be bounded and
   correct. Task 1: `test_long_lookahead_detects_far_land`.
4. **Homogeneous landscapes and `landscape=None`** see no behaviour change, and the slow tier
   stays green. Task 2: `test_homogeneous_unchanged`, and the slow tier run.
5. **A recycled slot** starts with the water state True, so a newborn's first activation is not
   wrongly filtered. Task 3: `test_recycled_slot_resets_water_state`.

---

### Task 1: `depons_land_check` core (NumPy/Numba), tested in isolation

**Files:**
- Create: `src/cenop/movement/land_check.py`
- Create: `tests/test_land_check.py`

**Interfaces:**
- Produces:
  - `depons_land_check(x, y, heading, log_mov, mask, depth, wmin, pos_hist_x, pos_hist_y,
    mem_ptr, mem_count, out_enough) -> None`;
  - `HAS_NUMBA: bool`.
- Behaviour of `depons_land_check`:
  - edits `x`, `y`, `heading` and `log_mov` in place, for `mask` entries only;
  - writes `out_enough[i]` (the final path-check result) for masked agents;
  - array dtypes are as in `PorpoisePopulation`: float32 positions and heading, float64
    `log_mov`, bool `mask`/`out_enough`, int32 `mem_ptr`/`mem_count`, float32 `pos_hist_*`;
  - `depth` is the 2-D float array indexed `[y, x]`;
  - jitter comes from `np.random.uniform(0.0, 10.0)` inside the function (Numba's RNG under njit).
- Implementation shape:
  - plain loops over agents and rounds, compiled with `njit(cache=True)` when Numba imports,
    else run as Python;
  - inner helpers `_depth_at(depth, x, y)` and `_path_ok(depth, x, y, h, pm)`, njit-compatible.

- [ ] **Step 1: Write the failing tests** in `tests/test_land_check.py`

Helper `_grid(h=60, w=60, depth=30.0)` returns a float64 depth grid. Tests call
`depons_land_check` on 1–3 agents with explicit arrays. Seed with `np.random.seed(0)` before the
call (the Python path) and assert only on jitter-independent outcomes or ranges:
- `test_open_water_passes_unchanged`: deep grid, agent at (30.5, 30.5) heading 0, `log_mov =
  log10(6)` → heading, position and `log_mov` unchanged; `out_enough[0]` True.
Shared geometry: agent at (30.5, 30.5) heading 0 (north), `presMov = 6`, so the path cells are
(30, 31…36).
- First-tier escape points (±40–50°, 6 cells) fall in cells x ∈ {25, 26} or {34, 35}, y ∈ {34, 35}.
- Second-tier points (±70–80°) fall in x ∈ {24, 25} or {36}, y ∈ {31, 32}.
- These cell sets do not depend on the jitter.

Tests:
- `test_lookahead_sees_land_four_steps_ahead`:
  - land only at path cell (30, 33), so the destination one step ahead (30.5, 32.0) is water and
    the old destination-only check would not turn;
  - expect |Δheading| ≥ 40 and `out_enough[0]` True.
- `test_path_threshold_is_depth_above_zero`:
  - path cell (30, 33) at depth exactly 0.0 → turn;
  - the same cell at 0.5 m → no turn (the path threshold is > 0, not wmin).
- `test_turns_toward_deeper_side`:
  - land at (30, 33); cells x ≥ 33 at 20 m and x ≤ 27 at 5 m → Δheading in [40, 50);
  - swap the depths → Δheading in (−50, −40].
- `test_second_tier`:
  - land at (30, 33); first-tier cells (x ∈ {25, 26, 34, 35}, y ∈ {34, 35}) at 0.5 m (below wmin);
  - everything else 30 m → |Δheading| in [70, 80).
- `test_beh5_deepest_neighbour_sets_presmov_one`:
  - agent boxed in by depth ≤ 0 within `presMov` in every direction, empty memory
    (`mem_count = 0`);
  - one neighbour cell at 2 m, the others shallower;
  - expect `log_mov[0] == 0.0`, heading pointing at that neighbour's centre (±1e-3°), and
    `out_enough[0]` False.
- `test_beh4_walks_back_to_remembered_position`:
  - all tiers bad at the current position;
  - the newest history entry is a deep-water point with open water ahead in the faced direction;
  - expect the position to equal that entry.
- `test_offgrid_lookahead_is_land`: agent at (58.5, 30.5) heading 90 on a 60-wide grid → turns, no
  exception.
- `test_nan_depth_is_land`: a NaN path cell → turn.
- `test_long_lookahead_detects_far_land`: `log_mov = 1.73` (presMov ≈ 53.7) with land 50 cells
  ahead → turn; with land 55 cells ahead → no turn.
- `test_mask_respected`: an unmasked agent with land ahead is untouched.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_land_check.py -q`
Expected: FAIL (`ModuleNotFoundError: cenop.movement.land_check`).

- [ ] **Step 3: Implement `depons_land_check`** following Global Constraints line by line.

- [ ] **Step 4: Run the tests again; then run them with Numba disabled**

Monkeypatch the module to use its Python implementation, via a module-level
`_impl = njit(...)(_py)` / `_py` pair, so the tests can call `land_check._py` directly. Both must
pass.

- [ ] **Step 5: Commit**

Message: `feat(movement): DEPONS land-avoidance sequence (checkDepth/avoidLand/beh 4-5) as a kernel`.

---

### Task 2: Wire the land check into both CRW paths; reflection-only after composition; beh 6 flip

**Files:**
- Modify: `src/cenop/agents/population.py`:
  - `__init__`: add the `_enough_water_ahead` bool array, initialised True;
  - `_update_movement` (909+);
  - `_update_movement_jasmine` (CRW-module branch, 1204+);
  - `_handle_land_avoidance` (1419+);
  - `_apply_positions` (1692+).
- Test: `tests/test_land_check.py` (`class TestPopulationWiring`)
- Modify, where they assert today's destination-only behaviour on real landscapes:
  - `tests/test_land_avoidance.py` (`TestLandAvoidanceKernelIntegration`,
    `TestLandAvoidanceBufferPreallocation::test_land_avoidance_still_works_with_buffers`);
  - `tests/test_depons_movement.py::…::test_land_avoidance_turn_angles`.
  Make each assert the DEPONS sequence instead, or confine it to the non-CRW module path; explain
  each change in the report.

**Interfaces:**
- Consumes: `depons_land_check` (Task 1).
- Produces:
  - `PorpoisePopulation._depons_land_check(self, mask: np.ndarray) -> None`:
    - a no-op when `self._skip_land_avoidance` or the landscape has no `_depth`;
    - otherwise seeds Numba's RNG once from `self.rng` (as the CRW kernel does), calls
      `depons_land_check`, and writes `_enough_water_ahead[mask]`.
  - `self._enough_water_ahead: np.ndarray` (bool, length `count`), read by Task 3 and Task 4.
- **Placement:**
  - In `_update_movement`, call `self._depons_land_check(crw_mask)` right after
    `self._update_reference_memory(mask)` (which follows the CRW turn) and before the deterrence
    and composition block.
  - In `_update_movement_jasmine`'s CRW-module branch (reference memory runs first there), call
    it after the heading turn and before composition.
  - Both paths then store the same remembered position before any beh-4 move.
- **`_handle_land_avoidance`:** keep the border reflection for everyone. Run the depth/turn/backtrack
  logic only when `self._movement_module` is a non-CRW module (not a `DEPONSCRWMovement`).
- **`_apply_positions`:** keep the depth ≤ 0 rollback for everyone. On the CRW paths only, rolled-back
  agents also face back toward the pre-move position:
  `heading = degrees(arctan2(pre_x − tried_x, pre_y − tried_y)) % 360`.

- [ ] **Step 1: Write the failing tests** (`class TestPopulationWiring`, on a homogeneous 200×200 landscape with `_depth` edited, unless stated)
- `test_land_check_runs_before_composition`:
  - setup:
    - `_skip_land_avoidance` must be False: rename the landscape (e.g. `landscape_name =
      "TestCoast"`) and set `pop._skip_land_avoidance = False`;
    - a straight coastline: all cells with y ≥ 100 at depth −10;
    - 60 non-dispersing agents at y = 98.5, spread along x, with headings spread over 0–360°;
    - 10 agents in open water at y = 40.5;
  - record:
    - wrap `PorpoisePopulation._depons_land_check` to copy `heading` on entry and assert the mask
      it receives excludes dispersers;
    - wrap `_update_reference_memory` to copy `heading` on entry;
  - then run `pop._recompute_cell_indices(); pop._update_movement(pop.active_mask, None, None)`;
  - expect:
    - the land check ran before `_update_reference_memory`;
    - at least one coastal agent's heading changed by ≥ 40° between the two copies (the turn
      happened before composition);
    - no open-water agent's heading changed between them;
    - `pop._enough_water_ahead` is True for every open-water agent.
- `test_no_post_composition_land_check_on_crw_path`: a destination cell at 0.5 m (between 0 and
  wmin) is accepted, so the agent ends on it (DEPONS has no post-composition check; 0.16%
  occupancy below 1 m).
- `test_rollback_faces_back`: a destination on depth ≤ 0 → the agent stays at its pre-move
  position and its heading points back along the attempted move (beh 6).
- `test_generic_module_path_keeps_old_land_avoidance`: with a `JASMINEPhysicsMovement` module, a
  destination at 0.5 m is still turned away from (old destination check).
- `test_homogeneous_unchanged`: on `create_homogeneous_landscape(200, 200)`, the
  `_depons_land_check` patch is never called and `_enough_water_ahead` stays True.
- Parity: `tests/test_movement_parity.py` must keep passing; it compares the inline and CRW-module
  paths. Add `test_module_path_matches_inline_on_coast` there:
  - same comparison, on a renamed non-homogeneous landscape (`_skip_land_avoidance` False) with a
    coastline;
  - include one agent boxed in so that beh 4 fires (non-empty position history), plus agents that
    take tier turns;
  - assert identical positions, headings and `_enough_water_ahead` after one step.
  A homogeneous grid would prove nothing here.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_land_check.py::TestPopulationWiring -q`

- [ ] **Step 3: Implement the wiring, `_handle_land_avoidance` gating and the beh-6 flip**

- [ ] **Step 4: Update the existing land-avoidance tests listed above; run the touched files, the fast suite and the slow tier**

Run:
- `micromamba run -n shiny python3 -m pytest tests/test_land_check.py tests/test_land_avoidance.py tests/test_depons_movement.py tests/test_movement_parity.py tests/test_movement_modules.py tests/test_dispersal_parity.py -q`
- `micromamba run -n shiny python3 -m pytest tests/ -q`
- `micromamba run -n shiny python3 -m pytest tests/ -m slow -q`

Expected:
- **Fast suite:** passes, except `tests/test_jax_tick.py` / `tests/test_backend_equivalence.py`
  failures that come only from comparing JAX land avoidance with NumPy (fixed in Task 4). List them
  in the commit message.
- **Slow tier:** 11 pass (homogeneous landscapes, unchanged). If a slow test fails, **stop and
  report**.

- [ ] **Step 5: Commit**

Message: `fix(movement): DEPONS land avoidance before composition on CRW paths (D15)`.

---

### Task 3: D9 first-call water filter in `_disperse_step`

Faithful but small in effect. The stored state is False only when the final path check fails,
i.e. when beh 5 fires (about 0.03% of animal-ticks), so this filter does **not** materially lower
the activation rate. It is ported for completeness only.

**Files:**
- Modify: `src/cenop/agents/population.py`:
  - `_disperse_step` (1100+);
  - the slot-recycling block that resets dispersal state (≈2420–2470,
    `self._prev_step_heading[slots] = 0.0`).
- Test: `tests/test_dispersal_parity.py` (`class TestWaterFilter`)

**Interfaces:**
- Consumes: `self._enough_water_ahead` (Task 2).
- Produces: in `_disperse_step`, a new stop applied after the PSM turn and before the off-grid
  check: `~self._enough_water_ahead` → restore heading, `_end_dispersal`. Recycled slots get
  `_enough_water_ahead = True`.

- [ ] **Step 1: Write the failing tests**
- `test_new_disperser_with_no_water_ahead_stops_first_step`: dispersing agent with
  `_enough_water_ahead[i] = False` and `psm_angle=0`, in open water → after `_disperse_step`, not
  dispersing, heading restored, traveled 0.
- `test_water_state_true_does_not_stop`: the same with True → still dispersing.
- `test_water_state_frozen_while_dispersing`: a disperser's `_enough_water_ahead` is not rewritten
  by `_update_movement`, because the land check runs on `crw_mask` only.
- `test_recycled_slot_resets_water_state`: after the recycle path runs on slot k, whose state was
  False, `_enough_water_ahead[k]` is True. Follow the existing recycle test in
  `tests/test_reproduction.py:~471-523`.

- [ ] **Step 2: Run them and confirm they fail**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_parity.py::TestWaterFilter -q`

- [ ] **Step 3: Implement the filter and the recycle reset**

- [ ] **Step 4: Run `tests/test_dispersal_parity.py` and `tests/test_reproduction.py`, then the fast suite**

- [ ] **Step 5: Commit**

Message: `fix(dispersal): DEPONS first-call not-enough-water stop (D9)`.

---

### Task 4: Mirror Tasks 1–3 in the JAX backend

**Files:**
- Modify: `src/cenop/optimizations/jax_kernels.py`:
  - add `jax_depons_land_check`;
  - add the water stop to `jax_disperse_step`;
  - make the post-composition land handling (`jax_land_avoidance` at 478) reflection-only on the
    CRW path;
  - add the beh-6 flip to `jax_update_positions` (567).
- Modify: `src/cenop/optimizations/tick_jax.py`: in `jax_tick_movement`, call the land check after
  the CRW turn and before composition, on `crw_mask`.
- Modify: `src/cenop/agents/population.py` `_step_jax`: plumb `_enough_water_ahead` and
  `min_depth`, and return the state.
- Test: `tests/test_jax_tick.py`

**Interfaces:**
- `jax_depons_land_check(x, y, heading, log_mov, mask, depth_grid, wmin, pos_hist_x, pos_hist_y,
  mem_ptr, mem_count, key, n_samples_max) -> (x, y, heading, log_mov, enough)`
  - `n_samples_max = ceil(10**max_mov / 0.1)` is a static argument;
  - rounds use `lax.while_loop`, ending when no agent needs avoidance or after 100 rounds;
  - one jitter draw per needing agent per round, from a key split per round;
  - beh 4 uses `pos_hist_*` / `mem_ptr` / `mem_count` (already JAX inputs);
  - semantics identical to Task 1.
- `jax_disperse_step(...)` gains an `enough_water_ahead` argument and the Task 3 stop.
- When `_skip_land_avoidance` is True (homogeneous grid or no landscape), JAX keeps today's
  post-composition `jax_land_avoidance` path unchanged and skips `jax_depons_land_check`. Pass an
  explicit static flag; do not infer it from the depth grid.

- [ ] **Step 1: Write the failing tests** (skip if JAX is unavailable, as the existing tests do)
- `test_jax_land_check_matches_numpy_geometry`:
  - cases where the jitter cannot change the outcome: open water, a wall with only the right side
    good, only the left side good, the second tier, beh 5 boxed in, off-grid look-ahead, NaN cell,
    masked agent;
  - `heading` within the jitter band, position, `log_mov` and `enough` equal NumPy's
    `depons_land_check` on the same arrays.
- `test_jax_disperse_step_water_filter`: mirrors Task 3's first two tests.
- `test_jax_rollback_faces_back`: mirrors Task 2's beh-6 test.
- `test_jax_step_land_parity` (full `pop.step()`):
  - `use_jax` True vs False on the same pre-state, with one agent positioned so the jitter cannot
    matter;
  - the agent's `heading` sign and |turn| tier, and `_enough_water_ahead`, agree.

- [ ] **Step 2: Run and confirm they fail**

Run: `JAX_PLATFORMS=cpu micromamba run -n shiny python3 -m pytest tests/test_jax_tick.py -q`

- [ ] **Step 3: Implement the JAX kernels and plumbing**

- [ ] **Step 4: Run `tests/test_jax_tick.py` and `tests/test_backend_equivalence.py`, then the fast suite**

Expected: all pass.

- [ ] **Step 5: Commit**

Message: `fix(jax): mirror DEPONS land avoidance and the D9 water filter`.

---

### Task 5: Acceptance against DEPONS, probes, docs

**Files:**
- Modify:
  - `scripts/investigations/2026-11-10-phantom-food/cenop_budget.py`: remove the `port` option;
  - `scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py`: remove `--d15-port`;
  - `scripts/investigations/2026-11-10-phantom-food/spatial_fingerprint.py`: remove `port`.
  The prototype is now in the source, so re-applying it would double-patch. Keep
  `d15_land_port.py` as the historical prototype, and say in its docstring that it applies only
  to code before this plan (commit 46e2a39).
- Modify: `CHANGELOG.md` (`[Unreleased]`); the spec, Finding 4 and "Decision needed".

- [ ] **Step 1: Run the acceptance set** (v3.2 Kattegat copy at the path recorded in the spec's Reproduce section; `JAX_PLATFORMS=cpu`; in parallel)

```bash
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_budget.py <dir> 42 180
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_budget.py <dir> 43 180
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> stock 180 120 42
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> stock 180 120 43
micromamba run -n shiny python3 scripts/investigations/2026-11-10-phantom-food/spatial_fingerprint.py cenop <dir> 42
```

Acceptance, on both seeds, against the DEPONS values in spec Finding 4:

| Metric | Accepted range | DEPONS |
|---|---|---|
| dispersal-off N at days 90, 120, 150, 165 | within ±4% of DEPONS | 1737/1722, 1442/1417, 989/994, 780/784 |
| dispersal-off intake per animal-tick, days 0–90 bins | within ±3% | — |
| time within 1 km of coast, days 0–30 | 0.5–2.5% | — |
| activation rate | 0.11–0.19 | 0.133 (s42), 0.173 (s43) |
| episode moves | median 4–6, mean 8–15 | — |
| tick share dispersing | 0.8–1.8% | — |
| on/off N ratio at day 150 (`stock` vs dispersal-off N of the same seed) | 0.75–0.87 | — |

Day 180 is informational. Outside a range: stop, report, do not tune.

Activation note:
- DEPONS's tracked-200 reference spans 0.133 (seed 42) to 0.173 (seed 43); the in-memory prototype
  gave 0.151 / 0.162, inside that span.
- DEPONS's other dispersal rates: episode moves median 5 in both seeds, mean 12.1 / 10.1; tick
  share 1.34% / 1.36%.

- [ ] **Step 1b: Default landscape check.** Run one short (60-day) Lithuania simulation with the
  default parameters. Record N, mean energy, and the share of time within 1 km of the coast, against
  the same run on commit 46e2a39. Add one CHANGELOG line on what changed for the default landscape.
  This is informational; there is no DEPONS reference for Lithuania.

- [ ] **Step 2: Run the fast suite, the slow tier and lint** (black/ruff on `src tests validation` and the edited scripts). Record the wall time of one 180-day Kattegat run before and after (informational).

- [ ] **Step 3: Record the outcome in the docs**
- **CHANGELOG** `[Unreleased]` → Changed: land avoidance follows DEPONS 3.2 on CRW movement:
  - look-ahead of `presMov` cells before heading composition;
  - escape turns 40/70/120°;
  - no post-composition check;
  - beh-6 flip;
  - the D9 first-call water stop.

  Add that this is a behaviour change, that the Kattegat baselines are stale, and that seeded runs
  are not comparable with earlier ones. Under Known issues, list the two remainders (energy
  undershoot, late deaths) and remove D15 as a candidate.
- **Spec:** mark D15 and the D9 first-call filter fixed with commit hashes, and add the acceptance
  table.

- [ ] **Step 4: Commit**

Message: `docs(movement): D15 acceptance vs DEPONS; CHANGELOG behaviour change`.
