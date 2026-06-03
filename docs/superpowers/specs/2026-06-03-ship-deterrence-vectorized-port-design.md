# Design: Port the DEPONS ship-deterrence model into the vectorized path

**Date:** 2026-06-03
**Status:** Approved design, **revised v2** after multi-angle spec review (see "Review corrections")
**Companion investigation:** `2026-06-03-ship-deterrence-parity-investigation.md`

## Problem

`Simulation.step()` uses `ShipManager.calculate_aggregate_deterrence_vectorized`,
which applies the **turbine** deterrence model to ships: it gates on
`deter_threshold` (152 dB) instead of `deter_ships_min_db` / Tships (80 dB),
omits the DEPONS ship probabilistic response model (`pship_*`/`cship_*`), uses
`deter_max_distance` (1000 km) instead of the 10 km ship cap, and applies a raw
displacement × `(rl−152)` × `deter_coeff` vector. Production ship deterrence
therefore fires only within ~25 m of a 170 dB ship (`deter_strength=0` across a
2-yr, 637-ship Kattegat baseline).

## Goal

Make the production vectorized path implement the **DEPONS ship _response_ model**
(probabilistic reaction + magnitude), gated by Tships and the 10 km cap, with the
DEPONS unit-vector × magnitude displacement. Source level remains CENOP's flat
class-based value (JOMOPANS rewiring is out of scope) — so this achieves the DEPONS
*response* model, not absolute SL parity.

## Review corrections (v2)

A four-angle review (parity / architecture / testing / scope) found that v1 rested
on two false premises. All corrected here and verified against source:

1. **The scalar path is NOT a clean oracle.** `calculate_deterrence_magnitude`
   omits the `exp` link (`sound.py:354` vs DEPONS `Ship.java:395,405`), and the scalar
   aggregator builds vectors with the turbine formula (`get_deterrence_vector(...,
   deter_coeff)`). **The oracle must be fixed first** (Tasks O1–O2), then shared.
2. **`is_day` is already wired** (`simulation.py:490` passes
   `is_day=self.time_manager.is_daytime`). The only real gap is that `suntimes.csv`
   is not loaded into `TimeManager` (fixed 06–18 fallback). De-scoped accordingly.
3. **Vector units:** DEPONS unit vector is **grid-displacement / metre-distance**
   (`Ship.java:231-235`), i.e. `(px−ship_x)/dist_m`, NOT `dx_m/dist_m` (which is
   ~`cell_size`× too large). Corrected in the kernel formula.
4. **Aggregation across ships:** DEPONS keeps the **max-receivedLevel ship** per
   (sub)step (`ShipDeterrence.recordStep`); with sub-tick interpolation out of scope
   this collapses to "loudest ship per porpoise", NOT a sum over ships.
5. **`deter_strength` is L1** (`population.py:1236`); DEPONS is **L2**. Addressed as
   an explicit decision (Task S1).
6. **Existing tests pin the old formula** (`test_no_normalization_vectorized_ship`,
   `test_deterrence.py:77`). These are **superseded**, not preserved.

## Decisions (from brainstorming)

- **Correctness target:** match the DEPONS ship response model, realised as a
  **corrected** shared `ShipDeterrenceModel`. Equivalence is asserted both
  scalar↔vectorized AND against hand-computed DEPONS values for known inputs (so a
  latent oracle bug like the missing `exp` cannot hide).
- **Architecture:** shared kernel — one set of array-shaped functions both paths call.

## Prerequisite oracle fixes (must land first)

- **O1 — add the `exp` link:** `ShipDeterrenceModel.calculate_deterrence_magnitude`
  returns `np.exp(magnitude)` (matches `predictMag`; non-negative, so the `max(0,·)`
  guard is redundant). Verify `STD_MAG_NIGHT` constants match DEPONS
  (`dist 6.442084/2.48903`, `noise 68.86555/15.09977`, `Ship.java:397-399`).
- **O2 — correct the ship vector formula:** ship displacement is
  `(grid_disp / dist_m) · magnitude · react` with `grid_disp = (px−ship_x)` in **cell
  units** and `dist_m` in **metres** — **no `deter_coeff`** (turbine-only). Remove the
  scalar aggregator's `get_deterrence_vector(..., deter_coeff)` usage for ships.

## Architecture (shared kernel)

1. **`behavior/sound.py::ShipDeterrenceModel`** owns the model: `*_probability` (logit,
   already array-friendly) and `*_magnitude` (with O1's `exp`). Add a method
   `deterrence_for(px, py, ship_x, ship_y, source_level, dist_m, rl, is_day, u_draw)
   -> record` that applies Tships gate → prob → Bernoulli(`u_draw`) → magnitude →
   unit-vector(grid/m) × magnitude, returning a small record
   `(dx, dy, prob, mag, dist_km, react)` (so both wrappers can project what they need —
   the scalar contract `(should_deter, prob, magnitude, dist_km)` is reconstructable,
   which a bare `(dx,dy)` is not).
2. **`agents/ship.py`** supplies geometry + RL: distances, distance gates
   (`deter_min_distance_ships·1000 < dist_m ≤ min(MAX_DETER_DIST_M, deter_max_distance·1000)`,
   with `MAX_DETER_DIST_M = 10_000` as a single named constant), RL via simple `α/β`
   TL or per-cell WestonFlux with **NODATA→RL=0** and `TL≤0→RL=0` (matches
   `Ship.java:296-307`), clamp `RL = max(RL, 0)`.
3. **`calculate_aggregate_deterrence_vectorized`** loops active ships and, per porpoise,
   keeps the contribution of the **max-RL ship** (DEPONS `recordStep`), not a sum.
   Accumulate into float64 `total_dx/total_dy`. The ≤10 km gate culls downstream work.
4. **`Ship.calculate_deterrence`** (scalar) and `ShipManager.calculate_aggregate_deterrence`
   refactor onto the same kernel; scalar becomes the thin length-1 oracle wrapper and
   keeps its public return contract.
5. **Delete the old turbine-model ship code**: the `str_val = rl − deter_threshold`,
   `deter_probabilistic` / `response_probability_from_rl`, and `prob_response` blocks
   (`ship.py:558-602`) — not just `deter_coeff`. Leave `ambient_received_level_at_positions`
   and the `simulation.py:519-531` communication-SNR path untouched.

## Determinism / RNG

Use identity-based streams, not draw order: per active ship, derive
`rng = np.random.default_rng(np.random.SeedSequence([base_seed, tick, ship_id]))` and
draw `u = rng.random(N)` indexed by **global porpoise index** (then mask). This makes
each (ship, porpoise) draw independent of ship count/order. `compute_*` accepts
`u_draw` so tests inject fixed draws. (This replaces v1's per-tick global
`np.random.random(n)`, which was order/count-fragile.)

## Day/night

Already wired (`is_day=self.time_manager.is_daytime`). Optional follow-up: thread
`suntimes_path` into the `TimeManager` constructor (`simulation.py:134`) for seasonal
day/night instead of the fixed 06–18 fallback. Not required for this port.

## Testing

- **Equivalence:** vectorized == scalar (record fields) over randomized inputs, AND
  both == hand-computed DEPONS values for ≥1 known `(RL, dist, day, u)` point —
  including a magnitude value pinned to `exp(Mag)` (guards O1).
- **Gates:** RL = 80 (excluded, strict `>`) / 80.0001 (included); dist 99 m (excluded)
  / 101 m / 9.9 km (included) / 10.1 km (excluded); `deter_max_distance=5 km` tightens
  the cap to 5 km (the `min()` works both ways — guards the original bug).
- **Direction & sign:** vector dot `(porpoise−ship) ≥ 0` (away); a negative raw
  magnitude (large dist / low noise within gate) → **zero** vector, never toward the
  ship (direction-inversion guard).
- **Bernoulli:** reacts iff `u < prob`; `u == prob` → no react (strict `<`).
- **Multi-ship:** two ships near one porpoise → **loudest-ship** contribution only
  (max-RL), confirming no summation.
- **Degenerate:** porpoise on ship (`dist_m→0`, clamp 1.0, zero direction → defined
  finite output); NODATA cell → RL=0/skip; empty active set / `ships_enabled=False` /
  all-out-of-range → zero vectors of correct shape and **no RNG consumed**.
- **Determinism:** ship order `[A,B]` vs `[B,A]` → identical per-ship vectors; adding a
  far ship C doesn't change A/B; two full `Simulation` runs at the same seed → identical
  trajectories.
- **Integration:** `Simulation.step()` with a ship near porpoises → `deter_strength>0`,
  porpoises pushed away (dot ≥ 0), and zero when source level lowered so RL<80; day vs
  night differ. Strengthen the near-empty `test_integration.py:300-350`.
- **Superseded:** rewrite `test_no_normalization_vectorized_ship` (it pins the wrong
  model); keep the turbine equivalent (`test_no_normalization_vectorized_turbine`) — turbines
  correctly use raw displacement. Add a guard that `deter_coeff` changes do **not** alter
  ship vectors (but do alter turbine vectors).
- **Characterization snapshot:** small fixed-seed run (few ships, ~50 ticks) with a
  checked-in expected `deter_strength`/position array, locking the new live behavior.

## Scope / non-goals

- **No** sub-tick interpolation. This is a **real DEPONS parity gap** (it defines the
  per-substep max-RL + sum), not a no-op — documented, not hidden. Acceptable first cut.
- **No** JOMOPANS source-level rewiring — flat class-based SL. The result is the DEPONS
  ship **response** model with an approximate SL; absolute RL is offset from DEPONS.
- **Numba** deferred; validate the 10 km mask actually culls in Kattegat ship-route
  geometry before declaring perf adequate.

## Decisions to confirm during implementation

- **S1 — `deter_strength` L1→L2:** `population.py:1236` uses `|dx|+|dy|`; DEPONS uses
  `sqrt`. Switching affects the **combined** turbine+ship vector (turbines too).
  Recommendation: align to L2 and document that CENOP tracks a single combined
  deterrence (no separate ship/turbine strengths, no `max()` stuck-detection, no
  per-source `deterTime`/`deterDecay`) as an accepted divergence — unless that
  persistence is required, in which case it becomes its own task.

## Consequences / risks

- Ship deterrence becomes **live** → ship-scenario outputs change (desired).
  **Regenerate `output/kattegat_ref_ships/`** and record (in the baseline provenance and
  the runner) that it supersedes the near-zero baseline, with the producing commit.
- Update the parity investigation doc status once implemented (the v2 corrections are
  already in it).
- Performance: per-ship `O(N)` distance pass over 637 ships at N=2000 ≈ 1.3 M ops/tick;
  the 10 km cap culls only **downstream** work, not the distance pass. Profile; add a
  ship-bbox pre-filter only if needed.

## Omitted-task checklist (now folded into the plan)

O1 exp link · O2 vector formula + remove `deter_coeff` for ships · delete turbine-model
ship block (`str_val`/`deter_probabilistic`/`prob_response`) · max-RL-ship aggregation ·
SeedSequence RNG + order/membership tests · kernel record return · S1 deter_strength
L1→L2 decision · rewrite superseded tests + characterization snapshot · regenerate &
re-provenance ship baseline · perf validation of the 10 km cull · NODATA→RL=0 handling.
