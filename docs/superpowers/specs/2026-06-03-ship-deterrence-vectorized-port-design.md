# Design: Port the DEPONS ship-deterrence model into the vectorized path

**Date:** 2026-06-03
**Status:** Approved design (pre-implementation)
**Companion investigation:** `2026-06-03-ship-deterrence-parity-investigation.md`

## Problem

`Simulation.step()` uses `ShipManager.calculate_aggregate_deterrence_vectorized`,
which applies the **turbine** deterrence model to ships: it gates on
`deter_threshold` (152 dB) instead of `deter_ships_min_db` / Tships (80 dB),
omits the DEPONS ship probabilistic response model (`pship_*`/`cship_*`), uses
`deter_max_distance` (1000 km) instead of the 10 km ship cap, and applies a raw
displacement × `(rl−152)` × `deter_coeff` vector. As a result production ship
deterrence fires only within ~25 m of a 170 dB ship (observed `deter_strength=0`
across a 2-yr, 637-ship Kattegat baseline).

CENOP's **scalar** path (`Ship.calculate_deterrence` + `ShipDeterrenceModel`)
already implements the DEPONS model faithfully, but `Simulation.step()` never
calls it.

## Goal

Make the production vectorized path implement the DEPONS ship model, using the
existing scalar `ShipDeterrenceModel` as the correctness oracle.

## Decisions (from brainstorming)

- **Correctness target:** match the scalar `ShipDeterrenceModel` (already faithful
  to DEPONS). Statistically equivalent to DEPONS; not bit-identical to the Java
  RNG sequence.
- **Architecture:** shared kernel — one set of array-shaped functions both paths
  call, so equivalence holds by construction.

## Architecture (shared kernel)

1. **`behavior/sound.py::ShipDeterrenceModel`** — `calculate_deterrence_probability`
   and `calculate_deterrence_magnitude` already operate elementwise via `np.exp`;
   confirm/lock that they accept arrays unchanged. These are the shared prob/mag
   kernel.
2. **New helper** in `agents/ship.py`:
   `compute_ship_deterrence(px, py, ship_x, ship_y, source_level, params, is_day,
   u_draw, cell_size, cell_data, month) -> (dx, dy)`. Operates over porpoise
   arrays. Pipeline per ship:
   - `dx_m = (px − ship_x)·cell_size`, `dy_m = (py − ship_y)·cell_size`,
     `dist_m = hypot(dx_m, dy_m)`, `dist_m = max(dist_m, 1.0)`.
   - **Distance gate:** `deter_min_distance_ships·1000 < dist_m ≤
     min(MAX_DETER_DIST_M=10_000, deter_max_distance·1000)`.
   - **RL:** `source_level − TL`, TL via simple `β·log10(d)+α·d` or per-cell
     WestonFlux when `weston_flux_percell` (mirrors scalar `get_received_level`);
     clamp RL ≥ 0.
   - **Tships gate:** keep only `RL > deter_ships_min_db (80)`.
   - **Probability:** `prob = ShipDeterrenceModel.calculate_deterrence_probability(
     RL, dist_m/1000, is_day)`.
   - **Bernoulli:** `react = u_draw < prob`.
   - **Magnitude:** `mag = ShipDeterrenceModel.calculate_deterrence_magnitude(
     RL, dist_m/1000, is_day)`.
   - **Vector:** `dx = (dx_m/dist_m)·mag·react`, `dy = (dy_m/dist_m)·mag·react`
     (unit vector away from ship × magnitude; **no `deter_coeff`** — ships use
     `predictMag` directly, unlike turbines).
   - Return zeros for porpoises failing any gate.
3. **`calculate_aggregate_deterrence_vectorized`** loops active ships and sums
   `compute_ship_deterrence` into `(total_dx, total_dy)`. The early distance mask
   (now ≤10 km) culls most porpoises per ship cheaply.
4. **`Ship.calculate_deterrence`** (scalar) refactored to call the same helper with
   length-1 arrays, becoming the thin oracle wrapper. Its public return contract
   `(should_deter, prob, magnitude, dist_km)` is preserved for existing tests.

## Data flow

`Simulation.step()` → `ShipManager.calculate_aggregate_deterrence_vectorized(px,
py, params, is_day, cell_size, cell_data, month)` → per active ship:
`compute_ship_deterrence(...)` → accumulate → `(total_dx, total_dy)` →
`population.step(deterrence_vectors=(total_dx+turb_dx, total_dy+turb_dy))`.
`population` already derives `deter_strength` from the combined vector magnitude,
so ship deterrence becomes visible in stats automatically.

## Determinism / RNG

`compute_ship_deterrence` accepts `u_draw` (uniform array, one value per candidate
porpoise for that ship). **Production:** draw from the per-tick-seeded RNG
(`Simulation.step()` already reseeds each tick for reproducibility); structure as a
per-ship `rng.random(n)` call. **Tests:** inject fixed `u_draw` so the equivalence
test feeds identical draws to scalar and vectorized paths and asserts identical
decisions and vectors. Not coupled to the Java RNG draw order.

## Day/night wiring

DEPONS chooses day vs night coefficients via `SimulationTime.isDaytime()`.
`Simulation.step()` currently passes a hardcoded `is_day`. Wire the real day/night
flag from the simulation clock (`data/Kattegat/suntimes.csv` is present). If the
clock plumbing proves non-trivial, it is split into its own plan task; the model
port does not block on it (defaults remain explicit).

## Testing

- **Equivalence:** vectorized == scalar (prob, magnitude, vector) over randomized
  (ship pos, porpoise pos, day/night, `u_draw`). By construction, but asserted.
- **Gates:** RL just above/below 80 dB; distance just inside/outside 100 m and
  10 km (including the `deter_max_distance` cap no longer overriding 10 km).
- **Direction:** deterrence vector points away from the ship.
- **Bernoulli:** porpoise reacts iff `u_draw < prob`; zero vector when `u_draw ≥ prob`.
- **Integration (regression the baseline exposed):** `Simulation.step()` with a
  ship placed near porpoises produces non-zero `deter_strength`.
- **No regression:** existing scalar `test_deterrence.py` ship tests still pass.

## Scope / non-goals

- **No** sub-tick interpolation (neither path does it today; matches the oracle).
- **No** JOMOPANS source-level rewiring (`jomopans_spl.py` stays unwired; source
  level unchanged).
- Scalar path **retained** as the oracle, refactored onto the shared kernel.
- **Numba** kernel deferred to a later optimization pass; the 10 km cap culls.

## Consequences / risks

- Ship deterrence becomes **live** → ship-scenario population outputs change
  (desired). **Regenerate `output/kattegat_ref_ships/` after the fix** and note the
  baseline supersession.
- `deter_strength` becomes non-zero in ship runs (combined-vector magnitude).
- `deter_coeff` is intentionally **not** applied to ships; audit shared code for
  any assumption that it is.
- Performance: per-ship `O(N)` distance compute over 637 ships at N=2000 ≈ 1.3 M
  ops/tick — acceptable in NumPy; the 10 km mask culls downstream work. Revisit
  with Numba only if profiling demands it.
