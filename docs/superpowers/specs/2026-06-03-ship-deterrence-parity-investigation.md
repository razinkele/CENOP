# Ship-Deterrence Parity Investigation

**Date:** 2026-06-03
**Status:** Investigation / findings (no code changes)
**Trigger:** Kattegat ships reference baseline (`output/kattegat_ref_ships/`) showed
`deter_strength = 0` for the entire run despite 637 active 170 dB ships overlapping
2000 porpoises. Root-cause analysis revealed a structural parity gap.

---

## TL;DR

CENOP has **two ship-deterrence implementations that disagree**:

| | Scalar path | Vectorized path |
|---|---|---|
| Code | `Ship.calculate_deterrence` + `ShipDeterrenceModel` (`sound.py`) → `ShipManager.calculate_aggregate_deterrence` | `ShipManager.calculate_aggregate_deterrence_vectorized` |
| DEPONS parity | ✅ faithful | ❌ wrong model |
| Called by `Simulation.step()`? | ❌ **never** | ✅ **yes (production)** |

The **correct** implementation (scalar) is dead code in production. The path that
actually runs (vectorized) applies the **turbine** deterrence formula to ships:
it gates on `deter_threshold` (152 dB) instead of `Tships` (80 dB), ignores the
entire DEPONS ship probabilistic response model (`pship_*`/`cship_*`), and uses
`deter_max_distance` (now 1000 km) instead of the 10 km ship cap.

**Consequence:** ship deterrence in production fires essentially never (requires a
porpoise within ~25 m of a 170 dB ship), and the round-3 `deter_ships_min_db`
parameter is inert. Ship traffic currently has almost no behavioral effect.

---

## The authoritative DEPONS model

`DEPONS-3.2/.../Ship.java` (`deterPorpoise`, lines ~205–248):

1. **Distance gates** (`Ship.java:221-223`):
   `deterMinDistanceShips (100 m) < distToShip ≤ MAX_DETER_DIST (10 km)` **and**
   `≤ deterMaxDistance`. `MAX_DETER_DIST = 10·1000` is hardcoded "regardless of
   dmax_deter value" (`Ship.java:51`) — so ships are effectively capped at **10 km**.
2. **Received level** (`calculateReceivedLevelFor`, `Ship.java:290`):
   `RL = sourceLevel − WestonFlux.calc(dist, depth, grain, temp, salinity)`, clamped ≥ 0;
   returns 0 on NODATA. `sourceLevel` = JOMOPANS decidecade band SPL from vessel
   type/speed/length (`calculateSourceLevel`, `Ship.java:285`).
3. **Tships gate** (`Ship.java:228`): proceed only if `RL > deterShipsMinDB (80 dB)`.
4. **Probability** (`predictProbResponse`, `Ship.java:334`): logit model on
   **standardized** (distance_km, RL) with day/night means/SDs and `pship_*` coeffs:
   `prob = exp(P)/(exp(P)+1)`, `P = p_int + noise·noiseScale + dist·distScale + nd·noiseScale·distScale`.
5. **Magnitude** (`predictMag`, `Ship.java:374`): analogous logit on (dist, RL) with `cship_*` coeffs.
6. **Stochastic response**: `reactingOrNot = rand < prob ? 1 : 0`.
7. **Vector**: **unit vector away from ship** × `deterMagnitude` × `reactingOrNot`
   (`Ship.java:236-243`).

## CENOP scalar path — *mostly* faithful ⚠️ (but unused)

**Correction (2026-06-03, post multi-angle review):** the scalar path is faithful
for the **probability/gate** logic but has two real DEPONS divergences of its own —
it is NOT a clean oracle as originally claimed.

- `behavior/sound.py::ShipDeterrenceModel.calculate_deterrence_probability`
  reproduces the standardization constants exactly (`5.801812/2.602801`,
  `6.243703/2.548173`, …) and the logit `1/(1+exp(−x))`. ✅
- **BUG — `calculate_deterrence_magnitude` omits the `exp` link.** DEPONS `predictMag`
  returns `Math.exp(Mag)` (`Ship.java:395,405`); CENOP returns `max(0.0, magnitude)`
  (`sound.py:354`) — the raw linear predictor. ~6.5× magnitude error at the mean and
  wrong shape vs RL/distance. ❌
- `agents/ship.py::Ship.calculate_deterrence` (~line 320): RL via simple `α/β` TL or
  per-cell WestonFlux; **Tships=80 gate** (`ship.py:353`); probabilistic response
  (`np.random.random() < prob`); returns `(should_deter, prob, magnitude, dist_km)`. ✅ gate/prob
- **BUG — the scalar aggregator `ShipManager.calculate_aggregate_deterrence`
  (`ship.py:418`) builds the vector with `get_deterrence_vector(..., deter_coeff)` →
  `calculate_deterrence_vector` = raw-displacement × magnitude × `deter_coeff`** — the
  **turbine** formula (`Porpoise.java:1290`), not the DEPONS ship unit-vector ×
  `predictMag` (no `deter_coeff`, `Ship.java:231-242`). ❌
- **Only caller in `src/` is its own aggregate method; `Simulation.step()` does not
  invoke it.** It survives only in tests (`test_deterrence.py`).

**Implication:** the fix must first *correct the oracle* (add `exp`, replace the
vector formula), then share it with the vectorized path — not adopt the scalar
output as-is. See the design spec `2026-06-03-ship-deterrence-vectorized-port-design.md`.

## CENOP vectorized path — wrong ❌ (production)

`agents/ship.py::calculate_aggregate_deterrence_vectorized` (~line 470), called by
`core/simulation.py::Simulation.step()` (~line 484):

| Aspect | DEPONS / scalar | Vectorized (production) | Verdict |
|---|---|---|---|
| RL gate | `RL > Tships (80)` | `str_val = RL − deter_threshold (152) > 0` | ❌ ~70 dB too strict → fires only < ~25 m |
| Distance cap | `min(10 km, deterMaxDistance)` | `RL` mask `dist < deterMaxDistance·1000` (1000 km) | ❌ no 10 km cap; min-dist floor missing |
| Response prob | `predictProbResponse` (pship_*, std) | `response_probability_from_rl(RL, 152, slope)` or none | ❌ turbine model, not ship model |
| Magnitude | `predictMag` (cship_*, std) | `str_val = RL − 152` | ❌ wrong units/model |
| Stochastic | Bernoulli `rand < prob` | none (deterministic, optional turbine-prob) | ❌ |
| Displacement | unit × magnitude | `dx_m · str_val · deter_coeff` (raw, scaled) | ❌ comment cites `Porpoise.java:1290` which is **turbine** code |
| `deter_ships_min_db` | used | **not referenced** | ❌ param inert |

The vectorized path is, in effect, the **turbine** deterrence model (`rl − 152`,
`deter_coeff`, raw displacement) copied onto ships.

---

## Impact

- **Production ship traffic is near-inert.** A 170 dB ship reaches 152 dB only within
  ~25 m (Kattegat TL), so `str_val > 0` essentially never holds → `deter_strength = 0`
  observed in the 2-yr, 637-ship baseline.
- **Round-3 `deter_ships_min_db` 70→80 dB change is behaviorally inert** in production
  (the production path never reads it). Correct per `parameters.xml`; only matters to
  the scalar path + tests.
- **`deter_max_distance` 50→1000 km change** is mis-wired for ships (should be capped at
  10 km); currently harmless only because the 152 dB gate fires first.

## Recommended fix (for a follow-up plan, not done here)

Make the **vectorized** path implement the DEPONS ship model:

1. Vectorize `ShipDeterrenceModel.calculate_deterrence_probability/_magnitude`
   (already array-friendly) and call them per active ship.
2. Replace the `RL − 152` gate with `RL > deter_ships_min_db (80)`.
3. Apply distance gates `deter_min_distance_ships < dist ≤ min(10 km, deter_max_distance)`.
4. Use **unit-vector × magnitude × Bernoulli(prob)**, not `dx_m · str_val · deter_coeff`.
5. Reconcile source level (JOMOPANS band vs `ship.noise.base_source_level=170`) and RL
   (prefer per-cell WestonFlux when `weston_flux_percell`).
6. Decide the fate of the scalar path: delete (dedupe) or keep as the reference oracle
   that the vectorized path is property-tested against.

### Test gaps to close
- No test asserts ship deterrence is **non-zero** through `Simulation.step()` (the
  production path). Existing tests exercise only the scalar path.
- Add an equivalence test: vectorized vs scalar produce matching deterrence for a
  fixed (ship, porpoise, seed) configuration.

## Determinism note
Reproducing the DEPONS Bernoulli draw under SoA/vectorization needs a per-(ship,porpoise)
RNG that stays deterministic under reordering — design this alongside the fix.
