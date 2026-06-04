# Design: Ship-Deterrence Parity Round 2

**Date:** 2026-06-04
**Status:** Approved design, **revised v2** after a four-angle review (DEPONS parity /
architecture / test-impact / scope), all findings verified against code. Key corrections:
#3 needs JSON-loader fixes (the `impact=170` default would make JOMOPANS inert; `type`/`length`
were ignored) + vessel-class mapping; #1 is a single site (`main.py:2102` doesn't apply); #2's
JAX threading goes through `tick_jax.py` (not just `jax_kernels.py`); three existing tests
need updating. DEPONS parity of the approach confirmed faithful (band 12, speed-0→0,
turbine-only deactivation). **v3:** a second review (DEPONS tables verified to match
`JomopansEchoSPL.java` exactly) added a real per-buoy-speed loader bug (the loader overwrites
JSON route speeds with 10.0 → wrong JOMOPANS speed), a DEPONS-`forValue`-style type
normalization, a second JAX test caller + the rename decision, pinning the integration test's
SL, and two stale line-ref fixes (`3062`, `ship.py:874`).
**Follows:** the merged DEPONS ship-deterrence port (`2026-06-03-ship-deterrence-parity-investigation.md`,
`2026-06-04-ship-deterrence-vectorized-port-design.md`). Picks up the three documented
follow-ups left by that work.

## Goal

Close three remaining DEPONS-parity / consistency gaps in ship deterrence, surfaced by
the port and its reviews:

1. **`is_disturbed` reporting threshold** is mis-scaled for ships.
2. **Dispersal deactivation** fires for ship deterrence, which DEPONS does only for
   turbine / sound-source deterrence.
3. **Ship source level** uses a simplified formula instead of the calibrated JOMOPANS
   model that already exists (unwired) in the codebase.

Sub-tick interpolation and the off-production scalar aggregator's simple-TL remain
**non-goals** (documented, deferred).

## Scope & structure

One cohesive effort, one spec → one implementation plan. Four work areas, ordered by
isolation: **#1** (trivial, reporting-only) → **#3** (JOMOPANS, self-contained) →
**#2** (dispersal plumbing, NumPy + JAX) → **baseline regeneration**.

#2 and #3 both change ship-scenario dynamics, so the committed Kattegat ship baseline
(`output/kattegat_ref_ships/`) is regenerated at the end.

---

## #1 — `is_disturbed` reporting threshold (trivial, no dynamics change)

**Problem:** the export `is_disturbed` flag uses `deter_strength > 0.1` at the **single**
site `population.py:2718` (inside `to_dataframe()`), but ship deterrence magnitudes are
~0.04, so ship-deterred porpoises never report as disturbed. The disturbance-memory
threshold (movement-affecting) is `> 0.01`.

**Fix:** change the one `> 0.1` site (`population.py:2718`) to `> 0.01`, aligning the
reporting flag to the disturbance-memory definition. Reporting-only — no effect on
movement, energy, mortality, or dispersal. (Review correction: there is **no** second site
— `main.py:2102` is a `deter_max_distance` getattr, not an `is_disturbed` threshold; do not
touch it. Separately, the live-map `p[6]` field is actually `is_dispersing`, not
`is_disturbed` — a pre-existing mislabel, out of scope here.)

**Test:** a porpoise with `deter_strength ≈ 0.04` reports `is_disturbed = True`.

---

## #3 — JOMOPANS ship source level

**Problem:** `ShipNoise.get_source_level()` (`sound.py`) uses a simplified
`base + 60·log10(L/100) + 20·log10(v/12) + vhf_weighting` formula. The calibrated
`jomopans_spl(vessel_class, speed_knots, length_m, band=12) -> float` in
`behavior/jomopans_spl.py` is complete but **unwired**. DEPONS uses the JOMOPANS
decidecade band-12 SL (the VHF-relevant band for porpoises).

**Design (JOMOPANS as default, with an explicit-override escape hatch):**
- Add a `vessel_class: VesselClass` field to `ShipNoise`. **Change `Ship.__post_init__`**:
  it currently constructs `ShipNoise(base_source_level=VESSEL_BASE_LEVELS[type], ...)` —
  which would shadow JOMOPANS. Instead construct `ShipNoise(vessel_class=self.vessel_type,
  length=self.vessel_length, speed=self.current_speed)` with `base_source_level` left at its
  new `None` default, so the JOMOPANS branch is the default. `VESSEL_BASE_LEVELS` becomes
  unused for SL — audit/remove or keep only as reference data.
- Make `base_source_level` an **optional explicit override**: change its default to
  `None`. `get_source_level()` becomes:
  - if `self.base_source_level is not None`: return it directly (explicit override) —
    preserves the existing data path (`ships.json` `impact` already sets
    `base_source_level`, `ship.py:874`) and lets tests force a specific SL;
  - else: `return jomopans_spl(self.vessel_class, self.speed, self.length, band=12)`.
- This is **not** the old simplified formula (which is removed entirely, along with the
  ad-hoc `vhf_weighting`); it is the calibrated JOMOPANS band-12 SL as the default, with a
  raw-dB override for data/tests. Production Kattegat ships (no `impact` in `ships.json`)
  get JOMOPANS; the simplified `60·log10(L/100)+20·log10(v/12)+vhf` formula is gone.
- **Lazy import** `jomopans_spl` *inside* `get_source_level` (`from
  cenop.behavior.jomopans_spl import jomopans_spl`) to break the module cycle:
  `sound.py` is imported by `ship.py`, and `jomopans_spl.py` imports `VesselClass` from
  `ship.py`; a module-level import in `sound.py` would form `sound → jomopans → ship →
  sound`. A function-local import resolves at call time, after all modules are loaded.
- **`speed = 0` → `jomopans_spl` returns 0.0** (stationary ship is silent; only the
  JOMOPANS branch — an explicit override is returned as-is). Downstream: source level 0 →
  RL clamps to 0 → gate fails → no deterrence. Correct.

**REQUIRED loader fixes (review found these — without them #3 is inert / fed garbage):**
The JSON ship loader (`ShipManager.load_from_json`, `ship.py`) currently (a) defaults
`impact = ship_data.get("impact", 170.0)` and sets `base_source_level = impact` whenever
`impact > 0`, so **every** Kattegat ship (none have an `impact` field) is forced to a 170 dB
override → **JOMOPANS would never fire in production**; and (b) ignores the JSON `type` and
`length` fields — it derives `vessel_type` from name-substring matching (→ mostly `OTHER`)
and hardcodes `vessel_length = 100.0`. The Kattegat `ships.json` actually provides real
`type` (12 classes: Bulker 228, Tanker 97, Passenger 60, Other 57, Containership 53, …) and
`length` (5–315 m) for all 637 ships. Fixes:
- Change the loader to only set `base_source_level` when `impact` is **explicitly present**
  (`ship_data.get("impact")` with no 170 default; set override only if not None and > 0).
  Absent `impact` → leave `None` → JOMOPANS.
- Read `length` from the JSON record (fallback 100.0 if absent) and map the JSON `type`
  string → `VesselClass` via DEPONS-style normalization (`VesselClass.forValue`: strip
  `[-/ _]`, uppercase, match the enum name) with two aliases for CENOP's renamed constants:
  `CONTAINERSHIP → CONTAINER`, `GOVERNMENTRESEARCH → GOVERNMENT`. This resolves all 12
  Kattegat strings (Bulker, Containership, Tanker, Government/Research, Cruise, Dredger,
  Passenger, Tug, Recreational, Fishing, Naval, Other).
- **Preserve per-buoy speed (review-found bug):** the loader currently does
  `for buoy in route.buoys: buoy.speed = ship_data.get("speed", 10.0)`, which **overwrites
  the real per-waypoint speeds** the Kattegat `ships.json` provides (e.g. 34.3, 26.3, …) with
  a constant 10.0 — feeding JOMOPANS (speed-dependent) a wrong speed for every ship. Fix:
  only overwrite buoy speed when the ship record has an explicit ship-level `speed`; otherwise
  keep the JSON per-buoy speeds (which `Buoy` already loads). `Ship.update` syncs
  `noise.speed` from the current buoy, so JOMOPANS then sees the real per-buoy speed.
  (Verify the speed unit is knots, as JOMOPANS expects; flag if the JSON speeds look like a
  different unit.)
- **Vessel-class validation:** `jomopans_spl` uses `_VC_SPEED.get(vc, 7.4)` — an unmapped
  class silently gets a tug-like 7.4 kn reference (Java throws instead). Assert/validate that
  every ship's `vessel_type` resolves to an explicit `_VC_SPEED` entry; confirm all 12
  Kattegat `type` strings map.

**Override is a non-DEPONS extension:** DEPONS always computes SL from JOMOPANS (no dB
override). The `base_source_level` override path is a CENOP convenience (data `impact` +
tests); document that DEPONS-faithful runs leave it `None`.

**Test impact (minimized by the override):** existing tests that set
`noise.base_source_level = X` keep working unchanged (override path). Only ships whose SL
is left to the default now use JOMOPANS. Add new tests for the JOMOPANS default branch
(`base_source_level=None`) and for the loader mapping JSON `type`/`length` correctly.

**Consumers (unchanged call site, new value):** both the vectorized production path and
the scalar `Ship.calculate_deterrence` call `ship.noise.get_source_level()`, so both pick
up JOMOPANS automatically. The communication-SNR path (`simulation.py` ambient RL) also
uses `get_source_level()` — it correctly gets JOMOPANS too.

**Tests:** `ShipNoise.get_source_level()` equals `jomopans_spl(vessel_class, speed, length, 12)`
for a known vessel/speed/length; `speed=0 → 0.0`; a CARGO vs FISHING ship produce
different SL (vessel-class dependence); a vectorized ship run still produces live
deterrence with the new SL.

**Risk:** the lazy import must be verified to actually break the cycle (import `Ship` and
construct one at runtime). The absolute RL shifts vs the current baseline → deterrence
magnitudes differ (expected; documented in the regenerated PROVENANCE).

---

## #2 — Turbine-only dispersal deactivation

**Problem:** DEPONS deactivates dispersal only when a porpoise is deterred by a turbine or
sound source (`Porpoise.java:1277`; `applyShipDeterrence` does NOT). CENOP's dispersal-
deactivation gates on the **combined** `deter_strength > 0` (`population.py:3062` NumPy;
`jax_kernels.py:843` JAX), so now that ship deterrence is live, ships wrongly stop
dispersal.

**Design:**
- `simulation.py` already computes `turb_dx/turb_dy` and `ship_dx/ship_dy` separately
  before summing into `total_dx/total_dy`. Pass the turbine-only pair to the population:
  add `turbine_deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None` to
  `PorpoisePopulation.step` (and `_step_jax`).
- In `population.step`, compute `_turbine_deter_strength = hypot(turb_dx, turb_dy)` into a
  **pre-allocated `(count,)` float32 buffer allocated in `__init__`** (refilled per tick,
  zeros when `turbine_deterrence_vectors is None` — mirroring the existing `deter_strength`
  buffer; no per-tick allocation per the project convention).
- Change the dispersal-deactivation gate from `deter_strength > 0` to
  `_turbine_deter_strength > 0` at **both** backends, in lockstep (the L1→L2 lesson):
  - **NumPy:** `population.py:3062` (`deterred = dispersing & (self.deter_strength > 0)`;
    note the coupled resets of `dispersal_distance_traveled`/`days_declining_energy` at
    `:3065-3066` move with the gate).
  - **JAX:** the gate is `jax_kernels.py:843` inside `jax_dispersal_update`, which is called
    from **`tick_jax.py:360` inside `jax_tick_energy`** (NOT directly from `_step_jax`). So
    threading turbine-only requires FOUR edits: in `jax_dispersal_update`
    (`jax_kernels.py:812`) **rename** its `deter_strength` param → `turbine_deter_strength`
    (the gate at :843 is its only use), and add the param to `jax_tick_energy`
    (`tick_jax.py:255`, a ~30-arg JIT fn — signature change → re-trace); pass it at the
    `jax_dispersal_update` call (`tick_jax.py:360`); and supply
    `jnp.asarray(self._turbine_deter_strength)` at the `jax_tick_energy` call in `_step_jax`
    (`population.py:~2408`).
  - **Critical:** `jax_tick_energy` must take BOTH the combined `deter_strength`/`is_disturbed`
    (drives BMR disturbance cost + reporting — stays combined) AND the new turbine-only array
    (drives dispersal only). Two deterrence inputs into one function — do not conflate them.
- The **combined** deterrence vector is unchanged for everything else: it still drives
  movement (DEPONS sums `deterTurbineVt + deterShipVt` into `totalD`) and still sets
  `deter_strength` (which feeds `is_disturbed` and disturbance memory — correctly reflecting
  total deterrence).
- **Gate signal nuance (DEPONS):** DEPONS deactivates whenever a turbine/sound-source has
  `currentDeterence > 0` in range (`Turbine.java:230`), regardless of strength update. The
  turbine vector is nonzero exactly when turbine strength > 0 and dist ≥ 1 m, so
  `_turbine_deter_strength > 0` is a faithful proxy. CENOP models **no `SoundSource` agent**
  (DEPONS' SoundSource is a testing-only agent), so turbine deterrence is the complete set of
  dispersal-deactivating sources — "turbine-only" drops nothing.
- **Out of scope (noted):** the legacy scalar `Porpoise.deter()` (`porpoise.py:808`)
  unconditionally deactivates dispersal on any deterrence — it is NOT on the production path
  (`Simulation.step` uses the vectorized population), so it stays as a known deferred
  inconsistency. The Cython tick path (`tick_cython`) reads `is_dispersing` read-only and does
  no deterrence/deactivation, so it needs no change.

**Tests:** with a ship-only scenario, a dispersing porpoise deterred by the ship does NOT
deactivate dispersal (`is_dispersing` stays True); with a turbine, it does; combined
turbine+ship still deactivates (turbine component present); determinism/order invariance
preserved; both NumPy and JAX paths exercised.

---

## Cross-cutting — baseline regeneration & verification

- **Regenerate `output/kattegat_ref_ships/`** after #2+#3 (both change dynamics) via
  `scripts/run_kattegat_reference.py --count 2000 --years 2 --seed 42 --ships`
  (≈1 hr). Confirm ship deterrence still live (`deter_strength` nonzero, new magnitudes
  reflect JOMOPANS SL). Update `PROVENANCE.txt` (new producing commit; note JOMOPANS SL +
  turbine-only dispersal changes vs the prior baseline).
- Whole suite green (excluding the two pre-existing hanging files; JAX run with GPU free).
- **Baseline loader path:** the regen uses the JSON loader, so it only reflects JOMOPANS
  after the #3 loader fixes (impact default + type/length mapping) land. Confirm the
  regenerated ship SLs vary by vessel class/length (not a constant 170/134).

**Existing tests the plan MUST update (review found these — spec previously omitted):**
- `tests/test_jax_tick.py::test_deterrence_cancels_dispersal` (~line 1265) AND
  `test_distance_completion` (~line 1295) — BOTH call `jax_dispersal_update(...,
  deter_strength=...)`. Since `jax_dispersal_update`'s param is renamed to
  `turbine_deter_strength`, both calls must be updated (else `TypeError`). Update
  `test_deterrence_cancels_dispersal` to assert turbine deterrence cancels dispersal and ship
  deterrence does not.
- `tests/test_dispersal.py::test_deterrence_deactivates_dispersal` (~line 337) — re-implements
  the gate inline on combined `deter_strength`; update to turbine-only so it doesn't enshrine
  the now-incorrect contract.
- `tests/test_integration.py::test_ship_manager_creates_deterrence_vectors` (~line 163) —
  constructs a CARGO ship WITHOUT setting `base_source_level`, so after #3 it gets JOMOPANS
  (~128–134 dB vs the old 175), narrowing the in-range margin. **Pin an explicit
  `ship.noise.base_source_level`** (override path) so the assertion is guaranteed, and fix
  the stale "~175 dB / 7 m / threshold 158" comment (actual Tships=80).
- **New loader test:** assert all **12** Kattegat `type` strings map to a valid `VesselClass`
  (and that `length` is read), so an unmapped/normalization miss fails loudly.
- **No change needed (note in plan):** the text loader `_load_ships` and the sample-ship
  fallback (`simulation.py`) construct ships via `__post_init__` and so inherit JOMOPANS
  automatically; the Cython tick path does no deterrence/dispersal-deactivation.

## Non-goals (carried forward, documented)

- **Sub-tick interpolation** (DEPONS evaluates ship deterrence at up to 30 interpolated
  positions/tick and sums; CENOP uses one end-of-tick position).
- **Scalar aggregator simple-TL** — off the production path; already documented in its
  docstring.

## Risks summary

- **Circular import (#3):** mitigated by the function-local import; must be runtime-verified.
- **NumPy/JAX divergence (#2):** both dispersal gates changed together; tested on both paths.
- **Baseline shift (#2, #3):** deterrence magnitudes and ship-dispersal dynamics change vs
  the current baseline — expected; the regenerated PROVENANCE records the producing commit
  and the behavioral deltas.
- **`base_source_level` semantics change (#3):** it goes from "always-set class default" to
  "optional explicit override (default None → JOMOPANS)". `Ship.__post_init__` must stop
  seeding it. Existing tests set it post-construction (override) and are unaffected; audit
  any reader of `base_source_level`/`vhf_weighting`/`VESSEL_BASE_LEVELS` before deleting.
