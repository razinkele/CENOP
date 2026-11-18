# InnerDanishWaters Dispersal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `dispersal="InnerDanishWaters"` runs on Kattegat and reproduces DEPONS 3.2 `InnerDanishWatersDispersal.java` step for step, opt-in, with the other three modes bit-identical.

**Architecture:** A new scalar module holds a line-for-line transliteration of the Java (target selection, phase 1, phase 2), written in the Numba subset and compiled when Numba is present. The population keeps two new pieces of state (phase, ten daily positions), loops over dispersing animals each tick, and applies the module's result at the end of the tick's position update.

**Tech Stack:** Python 3, NumPy, Numba (optional at runtime), pytest. Env: `micromamba run -n shiny`, run from `/home/razinka/cenop/CENOP`.

**Spec:** `docs/superpowers/specs/2026-11-18-dispersal-idw-design.md`

## Global Constraints

- Java source of truth: `../DEPONS-3.2/src/dk/au/bios/porpoise/behavior/InnerDanishWatersDispersal.java` (tag `8f533f8`). Where this plan and the Java disagree, the Java wins; record a ruling.
- Quirks listed in the spec are copied, never fixed.
- Opt-in only: no default changes, no baseline edits.
- No silent fallback: wrong landscape, JAX, or a non-CRW movement module raises `ValueError`.
- The other modes draw no extra random numbers. Gate after every task that touches `src/`:
  `micromamba run -n shiny python3 scripts/investigations/2026-11-18-dispersal-selector/type2_state_hash.py <mode>` must print
  `PSM-Type2` -> `aec15387d1425dd9`, `off` -> `7f2de66bb7ef401b`, `Undirected` -> `ba0c5558af305b87`.
- Never run the fast suite and the slow tier at the same time (JAX GPU out-of-memory).
- Shell rules in `CLAUDE.md` (no `$()`, `git -C`, commit messages via `-F`). Line length 100; black + ruff clean.

## Decisions fixed here (the spec left them to the builder)

- **Coordinates.** Java's continuous space has origin 0.5 (`PorpoiseSimBuilder.java:224-226`) and rounds to cells; CENOP truncates. For the same raster cell, `x_cenop = x_java + 0.5`. The module works in **Java coordinates and Java rounding throughout**, with the Java's own numbers (438, 478, block centres 50..550 / 950..50). The population converts at the boundary: `xj = x - 0.5` on the way in, `x = xj + 0.5` on the way out. The daily position buffer is stored in CENOP coordinates and initialised to 0.5 (Java's (0, 0)).
- **Rounding.** Java `Math.round` is half-up: use `math.floor(v + 0.5)` everywhere, never `round` or `np.round`.
- **Grid reads.** Grids are indexed `[row = y, col = x]` with row 0 south (as CENOP loads them). Land reads -9999 for depth, coast distance and blocks; verified on Kattegat. The navigation grid is `landscape._blocks`; its values are block index + 1.
- **How the move is applied.** The module returns the animal's position after the call. For an animal still dispersing afterwards, the population stores that position and writes it in `_apply_positions` in place of the kernel's result, and exempts the animal from the post-move land rollback (Java dispersers skip `stdMove`, so nothing checks their landing). For an animal that stopped, the population writes the position and heading to `x`, `y`, `heading` immediately, refreshes its cell indices, and lets it take its normal CRW move in the same tick.
  - Known deviation, accepted: in the rare case of a stop after a non-zero displacement (the 1-cell move north of Funen followed by a phase-2 stop, or a border-reflected step back), the animal eats at the cell it was moved to rather than its start-of-tick cell.
- **PSM-only state for this mode.** `dispersal_start_x/y` = start position; `dispersal_target_x/y` = block centre in CENOP coordinates; `dispersal_target_distance` = straight-line distance to it (for `Dispersal.txt`); `dispersal_distance_traveled` and `_prev_step_heading` are ignored. The FSM's `dispersal_complete` is forced False in this mode.
- **Energy.** Cost per paid 5-cell move is `0.001 * e_use_per_km * mean_disp_dist / 0.4`, using the same `e_use_per_km` value the population passes to its energy kernel (0.0).
- **Random draw.** `u = self.rng.random()` is drawn in the population and passed to `select_target`; the module contains no random calls.

## Review Focus

1. A look-ahead point or a move that crosses the grid edge: the point must bounce as Java's does, not index out of range or wrap.
2. A slot reused by a newborn: phase 0 and a daily buffer of 0.5s, not the dead animal's.
3. The first eight days of a run, when the daily buffer still holds the origin: no animal is wrongly kept in or thrown out of phase 1 compared with the Java rule.
4. A deterred or energy-recovered animal in phase 1 or 2: it stops, its phase is 0, and it never resumes without a new trigger.
5. Depth exactly 0 at a phase-1 landing: Java accepts it (`>= 0`); CENOP's rollback (`<= 0`) must not undo it.

---

### Task 1: Module skeleton, helpers and target selection

**Files:**
- Create: `src/cenop/agents/dispersal_idw.py`
- Test: `tests/test_dispersal_idw.py`

**Interfaces:**
- Produces (all in Java coordinates, degrees, heading 0 = north clockwise; every function Numba-compatible, decorated with an `njit` that falls back to identity when Numba is missing, and also importable undecorated as `<name>_py` for the equality test):
  - `BLOCK_CENTRES_X`, `BLOCK_CENTRES_Y`, `BLOCK_VAL_KAT_1`: `np.ndarray` (60,), `float64`, copied from the Java.
  - `N_DISP_TARGET = 12`, `MIN_DIST_TO_TARGET_CELLS = 250.0`.
  - `java_round(v: float) -> float` = `floor(v + 0.5)`.
  - `cell_of(xj: float, yj: float, w: int, h: int) -> tuple[int, int]`: Java `ndPointToGridPoint` (round; `x == w -> w - 1`, same for y).
  - `grid_at(grid: np.ndarray, xj: float, yj: float) -> float`: value at `cell_of`, `grid[yi, xi]`.
  - `point_at(xj, yj, heading, dist, w, h) -> tuple[float, float]`: Java `getPointAtHeadingAndDist`, including the border bounce.
  - `select_target(xj, yj, u, blocks) -> tuple[float, float]`: Java `dispTargetSelect`; returns the chosen block centre.

- [ ] **Step 1: Find the border bounce rule.** Locate `repast.simphony.space.continuous.BouncyBorders` in `.depons-cache/install/repast.simphony/` (search jars with `unzip -l`; decompile with `javap -c -p` if no source). Write its `translate` rule as a comment block at the top of `point_at`. If it cannot be found, use mirror reflection about the Java bounds (-0.5 and `w - 0.5`), ledger a ruling, and say so in the final report.

- [ ] **Step 2: Write the failing tests**

```python
import math
import numpy as np
import pytest
from cenop.agents import dispersal_idw as idw


def test_java_round_is_half_up():
    assert idw.java_round(2.5) == 3.0
    assert idw.java_round(3.5) == 4.0
    assert idw.java_round(-0.5) == 0.0


def test_tables_match_java():
    assert idw.BLOCK_CENTRES_X[:7].tolist() == [50, 150, 250, 350, 450, 550, 50]
    assert idw.BLOCK_CENTRES_Y[[0, 6, 59]].tolist() == [950, 850, 50]
    assert idw.BLOCK_VAL_KAT_1[[0, 17, 38, 59]].tolist() == [0, 0.4446, 1, 0]
    assert len(idw.BLOCK_VAL_KAT_1) == 60


def test_cell_of_rounds_and_clamps():
    assert idw.cell_of(438.4, 477.5, 600, 1000) == (438, 478)
    assert idw.cell_of(599.6, 999.7, 600, 1000) == (599, 999)


def test_kattegat_nav_block_numbering(kattegat):  # fixture: loaded CellData
    # Java coords (150, 450) is the centre of block index 31; the grid holds 32
    assert idw.grid_at(kattegat._blocks, 150.0, 450.0) == 32


def _blocks(value):
    return np.full((1000, 600), value, dtype=np.int64)


def test_select_target_excludes_blocks_within_100km():
    # At the centre of block 38 (quality 1.0, the best) that block is excluded
    for u in np.linspace(0, 0.999, 24):
        tx, ty = idw.select_target(250.0, 350.0, u, _blocks(39))
        assert (tx, ty) != (250.0, 350.0)


def test_select_target_draw_index_and_last_match():
    # From the far south-west corner nothing is excluded except nearby blocks; the
    # expected list is built by the reference loop below, which mirrors the Java.
    expected = _reference_top12(0.0, 0.0)
    for k in range(12):
        tx, ty = idw.select_target(0.0, 0.0, (k + 0.5) / 12, _blocks(55))
        assert (tx, ty) == (idw.BLOCK_CENTRES_X[expected[k]], idw.BLOCK_CENTRES_Y[expected[k]])


def test_select_target_east_shift():
    # nav value < 20 and pick index 30 -> 32. Find a u that picks 30 from a northern position.
    xj, yj = 250.0, 900.0
    top = _reference_top12(xj, yj)
    assert 30 in top
    u = (top.index(30) + 0.5) / 12
    assert idw.select_target(xj, yj, u, _blocks(3)) == (250.0, 450.0)   # index 32
    assert idw.select_target(xj, yj, u, _blocks(20)) == (50.0, 450.0)   # index 30, no shift


def _reference_top12(xj, yj):
    q = idw.BLOCK_VAL_KAT_1.copy()
    for n in range(60):
        d = math.floor(math.hypot(xj - idw.BLOCK_CENTRES_X[n], yj - idw.BLOCK_CENTRES_Y[n]) + 0.5)
        if d < 250:
            q[n] = 0
    s = np.sort(q)
    out = []
    for i in range(12):
        hit = 0
        for j in range(60):
            if q[j] == s[59 - i]:
                hit = j
        out.append(hit)
    return out
```

Add a session-scoped `kattegat` fixture to this file loading `CellData("Kattegat", "data")`. Add one `point_at` test per border (a point 3 cells beyond each edge comes back inside by the rule found in Step 1) and one interior case (`point_at(10, 10, 90, 5, 600, 1000) == (15, 10)` within 1e-9).

- [ ] **Step 3: Run to verify failure**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw.py -q`
Expected: FAIL, `ModuleNotFoundError: cenop.agents.dispersal_idw` (or ImportError).

- [ ] **Step 4: Implement the module pieces listed under Produces**

`select_target` follows `dispTargetSelect` (Java 495-564) loop for loop: quality copy, rounded distance, zeroing below 250, ascending sort, twelve passes each keeping the last matching index, `int(u * 12)`, the east shift on the navigation grid value at the animal's cell.

- [ ] **Step 5: Run to verify pass**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw.py -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/cenop/agents/dispersal_idw.py tests/test_dispersal_idw.py
git commit -F <message file>   # feat(dispersal): InnerDanishWaters tables, helpers, target selection
```

---

### Task 2: Phase 1 and phase 2

**Files:**
- Modify: `src/cenop/agents/dispersal_idw.py`
- Test: `tests/test_dispersal_idw.py`

**Interfaces:**
- Consumes: Task 1 helpers.
- Produces:
  - `forward(xj, yj, heading, dist, w, h) -> tuple[float, float, float]`: Java `Porpoise.forward` (1491-1534) for the non-homogeneous case, returning new position and heading (the off-edge heading formula `atan(dy/dx) - pi/2` copied as is; the move itself uses the original direction and the border bounce).
  - `disperse(phase, xj, yj, heading, tx, ty, d1x, d1y, d8x, d8y, enough_water, depth, dtc, blocks, mean_disp_dist, min_disp_depth, min_depth) -> tuple[int, float, float, float, int]`
    returning `(phase, xj, yj, heading, paid_moves)`. `d1*` / `d8*` are the daily positions at buffer index 1 and 8 in Java coordinates; `depth`, `dtc`, `blocks` are the full grids; `w, h` come from `depth.shape`. Implements `disperse()`: phase 1 if `phase == 1`, then phase 2 if `phase == 2` by then.

- [ ] **Step 1: Write the failing tests.** Synthetic grids are `(1000, 600)` arrays built by a helper `grids(depth=30.0, dtc=10000.0, block=40)` returning uniform arrays that individual tests then edit. Daily positions default to far away (`d1 = d8 = (300, 100)` for an animal near `(300, 500)`). One test per item; each asserts the returned tuple.

  Phase 1:
  1. `x > 438 and y < 478` -> phase becomes 2 and a phase-2 move follows in the same call (position changed by 5 cells, `paid_moves == 1`).
  2. Navigation value 14, and 31 -> same.
  3. Navigation value 32 -> heading 0 after phase 1, position moved 1 cell north then a phase-2 step; `paid_moves == 1`.
  4. Open water, target due north 300 cells away, uniform depth and coast distance: heading ends at `0 + 20 + 30` (last tie wins in both loops), position moves 5 cells along heading 50, phase stays 1.
  5. Deepest of five: make the cell 2 cells ahead at -10 degrees deepest -> first turn is -10.
  6. Stale filter: set the depth 16 cells ahead at +30 degrees (relative to the target-facing heading) to -9999 -> the +30 coast candidate is discarded, turn is +20.
  7. All far-ahead depths -9999 -> coast turn is +40.
  8. Each switch to phase 2, one test each: `d1` within 2 cells; `d8` within 6 cells; coast distance at the animal < 2000; target within 50 cells; depth <= 0 at exactly one of 4, 6, 8 cells ahead. Each asserts the same-call phase-2 move.
  9. `enough_water=False` -> phase 0, no move, heading keeps both turns, `paid_moves == 0`; also when a switch condition from item 8 holds (stop overrides switch).
  10. Landing on a cell with depth -1 -> phase 0, position back at the start (interior case), `paid_moves == 1`. Landing on depth exactly 0 -> stays, phase 1.
  11. `d1 = d8 = (0, 0)` for an animal at `(300, 500)`: no switch from the buffer tests.

  Phase 2 (call with `phase=2`):
  12. Heading faces away from `d1`: animal at `(300, 500)`, `d1 = (300, 490)` -> base heading 0.
  13. Coast distance 5000 everywhere -> smallest difference, last tie -> turn +80.
  14. Coast distance at the animal 3000, point at -40 degrees 3000, others 9000 -> turn -40.
  15. Coast distance at the animal 500, point at +10 degrees 9000 and at +60 degrees 9000, others 600 -> turn +10 (first tie).
  16. Coast distance at the animal 1500 -> heading changes by -9999 (i.e. base + 81 mod 360).
  17. `enough_water=False` -> phase 0, no move. Depth 5 cells ahead <= `min_depth` -> phase 0, no move. Both keep the turned heading.
  18. Otherwise moves 5 cells, `paid_moves == 1`.

  Cross-checks:
  19. `forward` off each edge: heading and position by hand from the Java formula.
  20. Numba = Python: 3000 random states (random phase 1/2, positions over water of the real Kattegat grids, random headings, targets, daily positions, `enough_water`); `disperse` and `disperse_py` return identical tuples. Skip if Numba is missing.

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw.py -q`
Expected: new tests FAIL with `AttributeError: ... has no attribute 'disperse'`.

- [ ] **Step 3: Implement `forward` and `disperse`** as transliterations of Java 148-161, 192-379, 384-493. Keep the Java's list layouts (9-entry padded lists, 17-entry list), its loop comparisons (`==` on the max for last match, `<=`, `<=`, `>`), its sentinels (-999, 9999999, 9999, -9999, index 9) and its order of statements. Put the Java line number in a short comment at the head of each block.

- [ ] **Step 4: Run to verify pass**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw.py -q`
Expected: all pass. If a hand-derived expectation and the transliteration disagree, re-read the Java before changing either; ledger which one was wrong.

- [ ] **Step 5: Commit** (`feat(dispersal): InnerDanishWaters phase 1 and phase 2`)

---

### Task 3: Parameters, guards and population state

**Files:**
- Modify: `src/cenop/parameters/simulation_params.py` (`DISPERSAL_MODES`, `_validate`)
- Modify: `src/cenop/agents/population.py` (constructor, `_update_energy_history`, newborn slot reset, `_end_dispersal`)
- Test: `tests/test_dispersal_modes.py`

**Interfaces:**
- Produces:
  - `DISPERSAL_MODES == ("off", "PSM-Type2", "Undirected", "InnerDanishWaters")`.
  - `SimulationParameters(dispersal="InnerDanishWaters", landscape=X)` raises `ValueError` matching `"InnerDanishWaters.*Kattegat"` unless `X == "Kattegat"`.
  - `PorpoisePopulation(...)` raises `ValueError` matching `"InnerDanishWaters.*JAX"` when JAX is in use, and `"InnerDanishWaters.*movement"` when a movement module other than `DEPONSCRWMovement` is passed.
  - `pop.dispersal_phase`: `int8[count]`, 0 / 1 / 2.
  - `pop._daily_pos_x`, `pop._daily_pos_y`: `float32[count, 10]`, newest at index 0, initial 0.5.

- [ ] **Step 1: Write the failing tests**
  - Update `test_modes_constant`, `test_valid_modes_accepted` (InnerDanishWaters with `landscape="Kattegat"`), replace `test_inner_danish_waters_not_implemented` with a test that the default landscape raises with the Kattegat message. Update the unknown-mode message expectations to list four modes.
  - Guards: JAX (`importorskip`), `JASMINEPhysicsMovement`.
  - State: after construction both buffers are all 0.5 and phase all 0. After one day boundary through `_update_energy_history` with `_tick_counter` set as the existing day-boundary tests do, index 0 equals the current `x, y` and index 1 equals 0.5; after a second, index 1 equals the first day's position. The push happens before `_check_dispersal_trigger` in the same day step.
  - Newborn reset: set phase 2 and a non-default buffer on a slot, run the newborn slot reset on it, expect phase 0 and 0.5s.
  - `_end_dispersal` sets phase 0.
  - The buffer is pushed in every mode (it is cheap and keeps one code path), and pushing draws nothing: covered by the hash gate.

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_modes.py -q`
Expected: the new and changed tests FAIL (mode rejected as not implemented; missing attributes).

- [ ] **Step 3: Implement.** Validation in `_validate`; guards next to the existing `Undirected` JAX guard; arrays allocated with the other dispersal arrays; the push in the day-boundary branch of `_update_energy_history` where the energy history shifts (all three step paths call it); reset in the newborn slot reset next to `is_dispersing[slots] = False`; phase cleared in `_end_dispersal`. The settings-panel test from stage 1 that asserts `InnerDanishWaters` is absent will now fail by design: leave it failing until Task 5 only if it is in a different file; otherwise fix it there now and note it.

- [ ] **Step 4: Run to verify pass, then the gates**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_modes.py tests/test_dispersal_idw.py -q` -> all pass.
Run the three hash gates from Global Constraints -> the three recorded hashes.

- [ ] **Step 5: Commit** (`feat(dispersal): accept InnerDanishWaters on Kattegat; phase and daily-position state`)

---

### Task 4: Population wiring

**Files:**
- Modify: `src/cenop/agents/population.py` (`_start_dispersal`, `_disperse_step`, `_apply_positions`, `_update_behavior_fsm`)
- Test: `tests/test_dispersal_idw_population.py` (new)

**Interfaces:**
- Consumes: `dispersal_idw.select_target`, `dispersal_idw.disperse`; state from Task 3.
- Produces: no new public names. Private: `_idw_pending` (`bool[count]`), `_idw_x`, `_idw_y` (`float32[count]`) holding this tick's end position for animals still dispersing.

- [ ] **Step 1: Write the failing tests** (Kattegat landscape, `dispersal="InnerDanishWaters"`, `communication_enabled=False`, seed fixed; place animals by hand at open-water positions):
  1. `_start_dispersal(i)` returns True, sets phase 1, `is_dispersing`, target = a block centre + 0.5, `dispersal_target_distance` = distance to it, and consumes exactly one `rng.random()` (compare generator state against a twin generator).
  2. One full tick (`pop.step` or the same entry the existing dispersal tests use) for a phase-1 animal in open water: end position equals `disperse_py` applied to its start state, converted back by +0.5, to float32 precision; heading likewise.
  3. Same for a phase-2 animal.
  4. Rollback exemption: phase-1 animal whose landing cell has depth exactly 0 (edit a copy of the depth grid) ends on that cell.
  5. Stop: `_enough_water_ahead[i] = False` -> after the tick the animal is not dispersing, phase 0, and has moved by a CRW step (position differs from start by less than 5 cells and is not the module's position).
  6. Deterrence stop and energy-recovery stop clear the phase (drive `_update_dispersal` as the stage-1 tests do).
  7. FSM: with the FSM enabled, a phase-1 animal is never reported `dispersal_complete`.
  8. `dispersal_distance_traveled` plays no part: set it above the target distance, the animal keeps dispersing.
  9. End to end, marked slow if it exceeds 20 s: N=500, 30 days on Kattegat: at least one animal reaches each of phase 1 and phase 2, no NaN in `x, y, heading, energy`, every active animal's cell has depth >= 0.

- [ ] **Step 2: Run to verify failure**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw_population.py -q`
Expected: FAIL (start uses the PSM target; no phase set).

- [ ] **Step 3: Implement.**
  - `_start_dispersal`: for this mode draw `u`, call `select_target`, fill the state listed under "PSM-only state", set phase 1. Never returns False.
  - `_disperse_step`: for this mode, clear `_idw_pending`, then for each index in `flatnonzero(mask & is_dispersing)` call `disperse` with Java-coordinate inputs. Still dispersing -> store position in `_idw_x/_idw_y`, set `_idw_pending`, write heading and phase. Stopped -> write `x, y, heading`, call `_end_dispersal`, refresh cell indices once after the loop if any stopped animal moved. Apply `paid_moves` energy cost. Return before the PSM code.
  - `_apply_positions`: after `self.x[mask] = self._new_x[mask]`, overwrite `x, y` for `_idw_pending` animals; exclude them from `on_land`; clear `_idw_pending` at the end. Set `_step_dist` for them to the displacement length.
  - `_update_behavior_fsm`: `dispersal_complete` all False in this mode.
  - Check with a grep that nothing else between `_disperse_step` and `_apply_positions` reads `x, y` of a pending animal in a way that matters (the heading kernel's fixed step and `dispersal_distance_traveled` update are overwritten or ignored); ledger what you found.

- [ ] **Step 4: Run to verify pass, then the gates and the fast suite**

Run: `micromamba run -n shiny python3 -m pytest tests/test_dispersal_idw_population.py tests/test_dispersal_modes.py tests/test_dispersal_idw.py -q` -> all pass.
Run the three hash gates -> recorded hashes.
Run: `micromamba run -n shiny python3 -m pytest tests/ -q` -> all pass except, at most, the stage-1 dropdown test that Task 5 updates.

- [ ] **Step 5: Measure.** Kattegat N=2000, 30 days, `InnerDanishWaters` vs `PSM-Type2` wall time (use `type2_state_hash.py <mode>` under `time`). Ledger both numbers. More than 25 % slower -> ledger it as a finding for the final report; do not optimise in this plan.

- [ ] **Step 6: Commit** (`feat(dispersal): InnerDanishWaters in the population tick`)

---

### Task 5: UI, docs, changelog

**Files:**
- Modify: `src/cenop/ui/tabs/settings.py` (tooltip), `src/cenop/ui/layout.py` (help text), `CHANGELOG.md`, `docs/USER_GUIDE.md` (dispersal modes line)
- Test: `tests/test_settings_defaults.py`

- [ ] **Step 1: Change the dropdown test** to expect all four modes present, and the tooltip to contain `"Kattegat only"` and no longer `"not available yet"`.
- [ ] **Step 2: Run** `micromamba run -n shiny python3 -m pytest tests/test_settings_defaults.py -q` -> the dropdown test FAILS on the tooltip text.
- [ ] **Step 3: Implement.** Tooltip: `'InnerDanishWaters' = DEPONS Kattegat mode (block targets, coast following); Kattegat only.` Help text: one sentence naming the four modes and that PSM-Type2 is the default on every landscape. CHANGELOG [Unreleased]: Added (the mode, opt-in, Kattegat only, refused on JAX and non-CRW movement modules, DEPONS quirks copied), Known issues (the eating-cell deviation from "Decisions fixed here"). USER_GUIDE: replace the Type2-only statement with the four modes.
- [ ] **Step 4: Run** the same test -> PASS; then the full fast suite -> all pass; ruff + black clean on changed files.
- [ ] **Step 5: Commit** (`feat(ui): offer InnerDanishWaters; docs and changelog`)

---

### Task 6: Comparison with DEPONS-Java

**Files:**
- Create: `scripts/investigations/2026-11-18-dispersal-idw/compare_with_java.py`
- Create: `docs/superpowers/specs/2026-11-18-dispersal-idw-java-comparison.md`

**Interfaces:**
- Consumes: Java output under `output/depons_java/kattegat-idw-undisturbed-v1/N2000-Y5/seed-{42,43}/` (`Statistics.csv`: tick, PorpoiseCount, energy; `PorpoisePerBlock.csv`: tick, Block, PorpoiseCount; `RandomPorpoise.csv`: per-tick `DispersalMode` of sampled animals), and the PSM-Type2 Java runs under `kattegat-psm2-undisturbed-v1` for scale.

- [ ] **Step 1: Check the Java inputs.** In each run's `batch_params.xml` confirm `dispersal` is `InnerDanishWaters` and `landscape` is `Kattegat`; record `landscape_sha256` from `manifest.json` and compare it with the hash of CENOP's `data/Kattegat` as the harness computes it (see `validation/depons_java/landscape.py`). A different landscape goes at the top of the write-up.
- [ ] **Step 2: Write the script.** It runs CENOP with the scenario's settings (Kattegat, 2000 animals, 5 years, ships off, turbines off, `InnerDanishWaters`) for seeds 42 and 43 plus three more (44-46), and records per simulated day: population count, mean energy, animals per block (same block numbering as the Java file; establish it from the Java output and say how), and the share of animals in phase 1 and in phase 2. It writes one CSV per seed under `output/` (gitignored) and prints a table: yearly population (Java per seed, CENOP per seed), yearly per-block share for the ten most populated blocks, mean share of time in phase 1 / phase 2 (Java from `RandomPorpoise.csv`, CENOP from all animals). Run it in the background; do not run test suites while it runs if it uses the GPU.
- [ ] **Step 3: Write up.** State the numbers, the between-seed spread on each side, and for each quantity whether CENOP falls inside the Java spread, with the limits: two Java seeds, different random streams, and the known PSM-Type2-era differences between CENOP and DEPONS on Kattegat (cite the existing Java-vs-CENOP PSM-Type2 numbers for scale). A mismatch is reported with the most likely cause and what would test it; nothing in `src/` is tuned in this task.
- [ ] **Step 4: Run the slow tier** (alone): `micromamba run -n shiny python3 -m pytest tests/ -m slow -q` -> 11 passed (plus the Task 4 end-to-end test if it was marked slow), no baseline edits.
- [ ] **Step 5: Commit** (`docs: InnerDanishWaters comparison with DEPONS-Java`)
