# Ship-Deterrence Sub-Tick Interpolation + Scalar-TL Consistency — Design

**Date:** 2026-06-08
**Status:** Design approved; ready for implementation plan.
**Branch target:** new branch off `CENOP-JASMINE` (HEAD `3363366`).
**Scope:** the two remaining deferred non-goals from the ship-deterrence parity work
(see `2026-06-04-ship-deterrence-parity-round2-design.md`):

1. **Sub-tick interpolation** of ship deterrence in the **production vectorized** path.
2. **Scalar-aggregator TL consistency** (an off-production oracle fix).

The two items are independent (one is a production restructure that changes the
committed baseline; the other is a small DRY fix to test-only code) but share one new
helper, so they live in one plan as two task groups.

---

## 1. Background — the authoritative DEPONS model

`DEPONS-3.2/.../Ship.java::deterPorpoise` evaluates ship deterrence at **30
interpolated ship positions per tick**, not one:

- **Movement model (`Ship.java::move`, line 144-170):** DEPONS teleports the ship to
  the next buoy **every tick** (`currentBuoyIdx++; setPosition(buoy)`). One buoy = one
  tick. `getSpeed()` returns the buoy's speed and is used **only** for the JOMOPANS
  source level, not for movement distance.
- **`interpolateStep(start, end)` (line 268-283):** builds 30 positions
  `start + (end-start)·i/30` for **i = 1..30** — excludes `start`, includes `end`.
  `start = getPosition()` (current buoy), `end = findNextBuoyPoint()` (next buoy). So
  the 30 positions span the genuine within-tick swept path.
- **Per sub-step (line 217-247):** for each porpoise, for each sub-step position:
  distance gate (`deterMinDistanceShips < dist ≤ min(MAX_DETER_DIST=10 km,
  deterMaxDistance)`), `RL = calculateReceivedLevelFor(...)` clamped ≥ 0, `Tships`
  gate (`RL > deterShipsMinDB`), probability `predictProbResponse`, magnitude
  `predictMag` (with `exp` link), an **independent** Bernoulli `reactingOrNot`, and a
  unit-vector-away × magnitude × Bernoulli step → `p.deterShipStep(step, ...)`.
- **`ShipDeterrence.recordStep(step, ...)`:** per porpoise, per sub-step **slot k ∈
  0..29**, keeps the contribution of the ship with the **maximum received level** at
  that slot (`deterSteps[k].receivedLevel < receivedLevel` replaces).
- **`deterrenceVtX/Y()`:** **sums the 30 slots** → the tick's ship-deterrence vector.
  `deterrenceStrength() = sqrt(sumX² + sumY²)`.
- **`Porpoise.applyShipDeterrence` (line 1243):** persistence — only overwrites the
  porpoise's `deterShipVt`/`deterShipStrength` when the new strength exceeds the
  persisted one; `deterShipVt` then enters the heading composition
  (`totalDX = dx·crwContrib + vt + deterTurbineVt + deterShipVt`, line 565-566) and
  decays each tick (vector halved, strength × `(100-deterDecay)·0.01`).

**Key consequence:** because each sub-step draws its *own* Bernoulli, even a stationary
ship (start == end → 30 identical positions) integrates ~`prob·30` reactions of the same
vector. Sub-tick interpolation is therefore a **per-tick magnitude amplifier for every
ship**, not only a fast-ship effect. The ship baseline must be **regenerated**.

## 2. Current CENOP state (what changes)

- **`Ship.update` (`agents/ship.py:191`)** moves by speed (`speed_cells`) and clamps to
  the next buoy; it does **not** teleport buoy-per-tick and does **not** track the
  previous position. Deterrence is computed **after** the move, so `ship.x/ship.y` is
  the end-of-tick position.
- **`ShipManager.calculate_aggregate_deterrence_vectorized` (`ship.py:453`)** —
  PRODUCTION (called by `core/simulation.py:489`). Evaluates a **single** end-of-tick
  ship position, keeps the **single loudest ship overall** per porpoise. Uses inline RL
  (WestonFlux per-cell when `weston_flux_percell` and `cell_data` present, else α/β).
- **`ShipManager.calculate_aggregate_deterrence` (`ship.py:398`)** — scalar oracle,
  **test-only** (no `src/` caller). Computes RL via `ship.get_received_level(...)`
  (α/β **only** — never WestonFlux), single position, loudest ship overall.
- The per-position kernel `ShipDeterrenceModel.deterrence_components` (`behavior/sound.py`)
  is correct and **unchanged**.

## 3. Design decisions (locked)

| # | Decision | Choice |
|---|----------|--------|
| 1 | Sub-step count | **30 fixed** (full DEPONS parity) |
| 2 | Interpolation source | **Within-tick swept path**: ship start-of-tick → end-of-tick |
| 3 | Aggregation | **Per-slot `recordStep`**: `(n,30)` state, max-RL ship per slot, sum slots |
| 4 | Scalar oracle | **Fix TL only** (shared RL helper); keep single-position; no sub-tick |

### 3.1 Why swept-path (not literal `findNextBuoyPoint`)

DEPONS' "interpolate to next buoy" is correct only because it teleports one-buoy-per-tick
(next buoy *is* one tick away). CENOP moves by speed and clamps, so a buoy can be many
ticks away; interpolating current → next-buoy would smear a slow ship's noise across
cells it won't reach for many ticks (over-deterring). The within-tick swept path
(`prev → cur`) is the ship's *actual* per-tick trajectory, faithful to DEPONS' intent,
cheap (one cached position), and correct under CENOP's movement model. It carries a
one-tick phase offset vs DEPONS (the segment is the same, labelled one tick later) —
within the SoA divergences already documented for this subsystem (RNG order,
NODATA-at-porpoise-cell).

## 4. Architecture

### 4.1 Swept-path source — `Ship`

- `__post_init__`: initialize `self._prev_x = self.x`, `self._prev_y = self.y`.
- `update(current_tick)`: as the **first statement**, before any early return,
  set `self._prev_x, self._prev_y = self.x, self.y`. Existing move logic then sets
  `self.x/self.y` to the end-of-tick position. Paused/inactive/zero-delta ticks leave
  `prev == cur` (degenerate segment → 30 identical positions, DEPONS-consistent).

### 4.2 Sub-tick aggregation — `calculate_aggregate_deterrence_vectorized`

Replace the single-position loop with per-slot accumulation (Approach 1, faithful
`recordStep`):

State (allocated once per call, `n = porpoise count`):
```python
best_rl  = np.full((n, 30), -np.inf)   # max RL seen per (porpoise, slot)
accum_dx = np.zeros((n, 30))           # winning ship's deterX per slot
accum_dy = np.zeros((n, 30))
```
Per active ship:
1. Build 30 sub-positions (i = 1..30):
   ```python
   t = (np.arange(1, 31) / 30.0)                  # (30,)
   sub_x = ship._prev_x + (ship.x - ship._prev_x) * t   # (30,)
   sub_y = ship._prev_y + (ship.y - ship._prev_y) * t
   ```
2. Draw reaction noise once: `rng = default_rng(SeedSequence([base_seed, tick, ship.id]))`,
   `u = rng.random((30, n))` (or `_force_u` fills all). `source_level = ship.noise.get_source_level()`.
3. For each slot `k` in 0..29:
   - `grid_dx = porpoise_x - sub_x[k]`, `grid_dy = porpoise_y - sub_y[k]`,
     `dist_m = hypot(grid_dx·cell, grid_dy·cell)`, floored at 1.0.
   - `idx = flatnonzero((dist_m > min_dist_m) & (dist_m <= max_dist_m))`; skip if empty.
   - `rl_sub = _ship_received_level(source_level, dist_m[idx], porpoise_x[idx],
     porpoise_y[idx], params, cell_data, month, weston)` (§4.4); clamp ≥ 0.
   - `vx, vy, _, _, react = ship.deterrence_model.deterrence_components(rl_sub, dist_m[idx],
     grid_dx[idx], grid_dy[idx], is_day, u[k][idx], tships)`.
   - `gated = rl_sub > tships`; `wins = gated & (rl_sub > best_rl[idx, k])`;
     `sel = idx[wins]`; `best_rl[sel, k] = rl_sub[wins]`;
     `accum_dx[sel, k] = vx[wins]`; `accum_dy[sel, k] = vy[wins]`.
4. After all ships: `total_dx = accum_dx.sum(axis=1)`, `total_dy = accum_dy.sum(axis=1)`.

`max_dist_m = min(MAX_DETER_DIST_M, deter_max_distance·1000)`,
`min_dist_m = deter_min_distance_ships·1000`, `tships = getattr(params,
"deter_ships_min_db", 80.0)`, `weston = params.weston_flux_percell and cell_data is not
None and cell_data._sediment is not None`.

Downstream persistence/decay/strength in `population.py` is **unchanged**: it receives
`(total_dx, total_dy)` exactly as before, now the sum over slots (DEPONS
`deterrenceVtX/Y`); strength is `hypot(total_dx, total_dy)` (DEPONS `deterrenceStrength`).

### 4.3 RNG determinism

Per ship, `u = rng.random((30, n))` from `SeedSequence([base_seed, tick, int(ship.id)])`.
Slot k uses `u[k]`. Invariant to ship order/count; reproduces the marginal Bernoulli
**rate** (not DEPONS' global draw order — impossible under SoA, already documented).
`_force_u` (test seam): when set, every slot's draw is the constant (e.g. 0.0 = always
react), enabling deterministic assertions of the slot/sum semantics.

### 4.4 Shared RL helper (resolves the scalar-TL inconsistency)

Module-level helper in `agents/ship.py`:
```python
def _ship_received_level(source_level, dist_m, px, py, params, cell_data, month, weston):
    """Received level (dB, clamped >= 0) for given porpoise positions.

    WestonFlux per-cell when `weston`, else simple alpha/beta TL. NODATA on
    depth/grain/salinity OR TL <= 0 -> RL 0 (DEPONS Ship.java:296-307 + valueIsNoData).
    """
```
It encapsulates the WestonFlux block currently inline in the vectorized path (NODATA and
`tl <= 0` → RL 0), and the α/β fallback. Both the vectorized path and the scalar
aggregator call it. The scalar `calculate_aggregate_deterrence` gains `cell_data=None,
month=1` parameters and replaces `ship.get_received_level(...)` with this helper —
so it uses WestonFlux when enabled, matching production. Non-WestonFlux callers (no
`cell_data`) get α/β exactly as before. The scalar aggregator remains **single-position**
and is documented in its docstring as a per-position + loudest-ship oracle, **not** a
sub-tick oracle.

## 5. Testing (TDD)

All new tests in `tests/test_ship_deterrence_port.py`.

**New — sub-tick:**
- Sub-positions are `prev + (cur-prev)·i/30`, i = 1..30 (endpoint included, start excluded).
- Per-slot max-RL recordStep: with two ships whose swept paths put **different ships
  loudest at different slots**, each slot's winner is the louder ship at that slot.
- Sum-over-slots: with `_force_u=0.0` (always react) and a moving ship, `total` equals
  the slot-wise sum of the 30 sub-step vectors (hand-computed for a small geometry).
- Stationary ship (`prev == cur`) with `_force_u=0.0` → `total ≈ 30 ×` the single-position
  vector (all slots identical); compared against a direct 30× single-position computation.
- Order/count invariance still holds across ships (extend existing test to moving ships).
- `tick` varies the draws; reproducible across sequential runs (extend existing).

**New — scalar TL:**
- Scalar `calculate_aggregate_deterrence` with `weston_flux_percell` + `cell_data` uses
  WestonFlux RL (differs from α/β; matches a direct WestonFlux computation).
- Without `cell_data`, scalar still returns α/β RL (unchanged).
- Vectorized↔scalar agree at a **single position** (zero-length swept segment via
  `prev == cur`, `_force_u` fixed) — confirms the shared helper + per-position kernel
  match across paths.

**Revise (new sum-over-slots semantics):**
- `test_kernel_snapshot_day` and any magnitude/snapshot assertions that assumed a single
  draw → expected values become the slot-wise sum (recompute with `_force_u` or the
  stationary 30× identity).
- `test_loudest_ship_wins_not_sum` — re-verify: per-slot loudest still dominates, so
  `dx_both ≈ dx_loud ≠ dx_loud + dx_quiet`; adjust magnitudes if needed.

**Regression:** full `tests/test_ship_deterrence_port.py`, `test_deterrence.py`,
`test_weston_flux.py`, `test_integration.py` (unfiltered — do not let a `-k` filter
deselect regression suites), plus the standard fast suite.

**Baseline:** regenerate `output/kattegat_ref_ships/` via
`python3 scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships` and
update `PROVENANCE.txt` (note: deterrence magnitudes rise due to swept-path integration;
event counts and stability reported).

## 6. Files

- **Modify:** `src/cenop/agents/ship.py`
  - `Ship.__post_init__` / `Ship.update` — `_prev_x/_prev_y`.
  - `_ship_received_level` — new module-level helper.
  - `calculate_aggregate_deterrence_vectorized` — per-slot sub-tick aggregation.
  - `calculate_aggregate_deterrence` — shared-helper TL + `cell_data/month` params + docstring.
- **Test:** `tests/test_ship_deterrence_port.py` — new sub-tick + scalar-TL cases; revise snapshot/magnitude tests.
- **Regenerate:** `output/kattegat_ref_ships/` (+ `PROVENANCE.txt`).

## 7. Non-goals (unchanged from prior rounds)

- Changing CENOP's ship **movement** model to teleport buoy-per-tick (out of scope:
  would change every trajectory and the baseline beyond the deterrence change).
- Sub-tick interpolation in the scalar oracle (decision 4).
- Per-buoy source-level recompute beyond what `Ship.update` already syncs
  (`noise.speed = current_speed`).
- Reproducing DEPONS' global-RNG draw order (impossible under SoA).
