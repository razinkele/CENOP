# Design: Ship-Deterrence Parity Round 2

**Date:** 2026-06-04
**Status:** Approved design (pre-implementation)
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

**Problem:** the UI/export `is_disturbed` flag uses `deter_strength > 0.1`
(`population.py:2718`, `main.py:2102`), but ship deterrence magnitudes are ~0.04, so
ship-deterred porpoises never report as disturbed. The disturbance-memory threshold
(movement-affecting) is `> 0.01`.

**Fix:** change both `> 0.1` sites to `> 0.01`, aligning the reporting flag to the
disturbance-memory definition (one "disturbed" threshold). Reporting-only — no effect on
movement, energy, mortality, or dispersal.

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
    `base_source_level`, `ship.py:923`) and lets tests force a specific SL;
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

**Test impact (minimized by the override):** existing tests that set
`noise.base_source_level = X` keep working unchanged (override path). Only ships whose SL
is left to the default now use JOMOPANS. Add new tests for the JOMOPANS default branch
(set `base_source_level=None`).

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
deactivation gates on the **combined** `deter_strength > 0` (`population.py:3060` NumPy;
`jax_kernels.py:843` JAX), so now that ship deterrence is live, ships wrongly stop
dispersal.

**Design:**
- `simulation.py` already computes `turb_dx/turb_dy` and `ship_dx/ship_dy` separately
  before summing into `total_dx/total_dy`. Pass the turbine-only pair to the population:
  add `turbine_deterrence_vectors: Optional[Tuple[np.ndarray, np.ndarray]] = None` to
  `PorpoisePopulation.step` (and `_step_jax`).
- In `population.step`, compute `_turbine_deter_strength = hypot(turb_dx, turb_dy)` (a new
  pre-allocated `(count,)` float array, zeros when not provided).
- Change the dispersal-deactivation gate from `deter_strength > 0` to
  `_turbine_deter_strength > 0` at **both** `population.py:3060` (NumPy) and, threaded
  through `_step_jax`, `jax_kernels.py:843` (JAX) — fixed in lockstep so the two backends
  stay consistent (the lesson from the L1→L2 change).
- The **combined** deterrence vector is unchanged for everything else: it still drives
  movement (DEPONS sums `deterTurbineVt + deterShipVt` into `totalD`) and still sets
  `deter_strength` (which feeds `is_disturbed` and disturbance memory — those reflect total
  deterrence, which is correct).

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
