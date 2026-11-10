# Phantom food in CENOP's vectorised eating — investigation (2026-11-10)

**Status (updated the same day):** three CENOP-vs-DEPONS differences found on undisturbed
Kattegat. Each is confirmed by a controlled single-variable experiment.

1. **Phantom food.** Fixed (TDD) in `eba75f9` on branch `fix/phantom-food-2026-11-10`; not merged.
2. **JASMINE social-communication vector overrides the DEPONS random walk once food is
   realistic.** Needs a design decision. The fix in (1) must not merge without it.
3. **Dispersal over-triggering.** Confirmed: with dispersal off in both models, the trajectories
   agree to day 120. Three code differences have been found: a trigger off-by-one, missing
   per-step stop rules, and a straight-line rather than path-length completion criterion.
   Together they leave about 25% of CENOP animals dispersing, against about 1% in DEPONS, so the
   main termination mechanism is still open. A smaller late residual also remains.

The fix and the decisions change released (v2.2.0) model behaviour and invalidate the committed
Kattegat baselines. See "Findings 2 and 3" and "Decision needed".

## Question

Track A1 (`2026-11-10-track-a1-spike-report.md` §9) found that DEPONS 3.2's undisturbed
Kattegat population (PSM-Type2, N=2000, v3.2 landscape) falls 2000 → 51/57 in 5 years, while
CENOP's committed baseline (`output/kattegat_ref`, v3.0 landscape) ends at 1999. The candidate
confounders were landscape version, scale, dispersal mode and initial conditions. Is the gap
caused by the landscape or by the model?

## Answer

**By the model.** CENOP's default (Numba) eating kernel creates food from nothing: it applies
the 0.01 "artificial food" floor to every cell an animal eats from, including empty cells
outside the food patches. DEPONS applies the floor only to cells that had food. Within a month
almost the whole sea holds 0.01 phantom food, which animals eat at every step. That is more
than double their basal energy use. On its own this explains why CENOP's Kattegat population
is stable. With the floor corrected, CENOP collapses as DEPONS does, and faster. That leaves a
second gap (below).

## Evidence

### 1. Trajectories differ in energy, not food supply

Seed 42, N=2000. Analysis script: `traj_compare.py`.

| | Y1 mean energy | pop at end of Y1 | pop at end of Y5 |
|---|---|---|---|
| DEPONS (v3.2 landscape) | 6.75 (min 4.42) | 161 | 51 |
| CENOP baseline (v3.0) | 15.28 | 1995 | 1999 |

DEPONS mean energy falls from 10 to about 4.5 by day 180 (mass starvation). CENOP mean energy
rises from 10 to about 15.3 and stays there.

### 2. The v3.2 landscape has more food than v3.0, not less (`landscape_food_stats.py`)

- `patches.asc`: same 4,572 food cells in both versions (0 cells differ). `bathy.asc` is identical.
- Monthly prey summed over the patch cells: v3.0 ranges from 898 (Dec–Feb) to 1,747 (Sep–Nov).
  v3.2 is about 1,756 in every month. That is a ratio of 1.0–1.96 in favour of v3.2.
- DEPONS starves on the richer landscape, so the landscape cannot explain CENOP's stability.

### 3. Parameters match (`param_diff.py`)

Everything that drives energetics, food and mortality matches between the effective DEPONS
parameters (A1 manifest) and CENOP's defaults:
- Euse 4.5, Ewarm 1.3, Elact 1.4;
- rU 0.1, Umin 0.001, maxU 1;
- beta 0.4, rR and rS 0.03;
- Einit N(10, 1);
- `M_MORT_PROB_CONST` 1.0, `E_USE_PER_KM` 0.

DEPONS `meanMaxEntInQuarters` is hard-coded to {1,1,1,1} and never set from
`mean_maxent_values.txt`, so CENOP's divisor of 1.0 also matches.

### 4. CENOP gains energy without removing the food (`depons_intake.py`, `cenop_intake.py`, `cenop_ledger.py`)

Intake is inferred identically for both models: total patch food drops within a day, divided by N.
Regrowth runs once per day (every 48 ticks) in both models.

| first 3 days | patch food removed / animal / tick | basal use / tick | mean energy, day 0 → 60 |
|---|---|---|---|
| DEPONS (v3.2) | 0.0039 | 0.0045 | 10.0 → 6.6 |
| CENOP (v3.0) | 0.0016 | 0.0045 | 9.9 → **15.9** |
| CENOP (v3.2 copy) | 0.0033 | 0.0045 | 9.9 → **16.8** |

The ledger on day 0 (no deaths, no regrowth) shows the mismatch directly:
- Food summed over **all** cells rises by 4–8 units per tick.
- Total animal energy rises by 19 units per tick more than food removed minus basal use.

### 5. Root cause in code

- DEPONS `CellData.eatFood` (`DEPONS-3.2/src/dk/au/bios/porpoise/landscape/CellData.java:179`)
  does nothing when `food > 0.0` is false. It applies the 0.01 floor only after eating from a
  cell that had food.
- CENOP `eat_food_kernel_v2` (`src/cenop/optimizations/kernels.py`, pass 3, ~line 400) runs
  `food_grid -= food_eaten[i]` and then `if food_grid < min_food: food_grid = min_food` for
  **every** animal, including those that ate 0 from an empty cell. So every visited empty cell
  becomes 0.01.

### 6. Hypothesis test (`hypothesis_test.py`, `occupancy_probe.py`)

The kernel was replaced in memory by one that floors only cells that had food. No source was
edited.

| v3.2 landscape | day 1 | day 30 | day 60 |
|---|---|---|---|
| stock: non-patch cells holding food | 51,550 | 279,401 | 279,726 (2,797 units) |
| stock: mean energy / N | 10.08 / 2000 | 15.92 / 1998 | 16.79 / 1998 |
| corrected: non-patch cells holding food | 0 | 0 | 0 |
| corrected: mean energy / N | 9.83 / 2000 | 4.61 / 1988 | 2.65 / **450** |
| DEPONS: mean energy / N | 10.02 / 2000 | 8.49 / 1998 | 6.63 / 1953 |

On v3.0 (`data/`, the baseline configuration), the corrected kernel gives N=236 by day 60.

## Scope

| path | phantom food? | since | used by |
|---|---|---|---|
| Numba `eat_food_kernel_v2` | **yes**: floors every visited cell | `024f870` (2026-03-19) | default real-landscape path (`population.py` → `eat_food_vectorized(energy=…)`) |
| Numba `eat_food_kernel` (v1) | **yes**: same pass 3 | `7ee50f9` (2026-03-12) | `eat_food_vectorized` when `energy` is not passed |
| pre-Numba NumPy path | **yes**: floored the **whole grid** (`np.maximum(new_food, 0.01)`) | `7c634f4` (2026-01-26) | until 2026-03-12 |
| NumPy fallback (current) | no: floors only cells with demand | fixed in `dc4f95d` (2026-04-05) | only without Numba |
| JAX `jax_eat_food` | **yes**: `jnp.maximum(food_grid - eaten, min_food)` over the whole grid (runtime-verified: 15/15 empty cells floored after one visit) | `22eadee` (2026-03-20) | opt-in JAX backend |
| Cython post-CRW | yes (unconditional floor) | — | gated to homogeneous landscapes, where every cell has food, so it has no effect there |
| scalar `CellData.eat_food` | no (returns early for empty cells) | — | legacy scalar path |

Consequences:
- **Affected:** every CENOP run on a real (patchy) landscape with the default backend, including
  the committed Kattegat baselines (`output/kattegat_ref*`) and v2.2.0.
- **Not affected:** the 11 slow-tier "demographic validation" tests. They all use
  `create_homogeneous_landscape`, which gives every cell, land included, a positive food value,
  so the floor never creates food. They therefore validate demography only under food in every
  cell, not foraging on a patchy landscape.

## Findings 2 and 3

With phantom food removed, CENOP still collapsed much faster than DEPONS: N=450 against 1,953
at day 60. A DEPONS probe run gave population-level movement data: 1 year, N=2000, seed 42,
`trackedPorpoiseCount=200`, run through the harness.

> **Correction.** An earlier version of this note said the patches CENOP animals occupy hold
> only 0.011–0.015 food. That number was measured *after* the tick's eating at the same cell,
> so it was an artefact. Measured before eating (`onpatch_rate.py`), CENOP finds 0.16 food per
> on-patch step on day 1, falling to 0.04 by day 30.

### Finding 2: the JASMINE social vector overrides the DEPONS random walk (confirmed)

| days 1–30 (`turning_angles.py`, `revisit_compare.py`) | DEPONS (200 tracked) | CENOP after the food fix, communication on (default) | CENOP after the food fix, communication off |
|---|---|---|---|
| step length (mean) | 852 m | 829 m | ≈ (unchanged) |
| median / p90 \|turn\| | 50° / 123° | **162° / 178°** | 51° / 122° |
| turns > 120° | 10.9% | **68.6%** | 10.7% |
| mean cos(turn) | +0.43 | **−0.48** | +0.42 |
| self-revisit of a cell within 48 ticks | 5.1% | **67%** | — |
| net displacement, 1 day | 10.5 km | **1.9 km** | 9.1 km |

What rules things out:
- **On-patch rate is not the cause.** It is at chance level in DEPONS: 1.59–1.69% across all 200
  tracked animals for the whole year, against a patch share of 1.59%.
- **Coarse spatial spread is not the cause.** It matches across the 51 blocks
  (`block_distribution.py`).
- **The CRW generator is not the cause.** The turning angles CENOP generates are DEPONS-like
  (median 61°), but the applied heading changes by a median of 155°
  (`crw_generated_vs_realized.py`).
- **The memory attraction is not the cause.** With `vt` zeroed, the reversals remain.
- **Land is not the cause.** No animal ends a step on land.

The mechanism:
- `population.py:1024-1027` adds the social communication / cohesion vector into `d_dx`/`d_dy`
  when `communication_enabled` is on, which is the default.
- `heading_position_reflect_kernel` then computes
  `total = (sin h, cos h)·crwContrib + vt + d`.
- With DEPONS's `crwContrib = k + presMov·veTotal ≈ 0.001` (k = 0.001; veTotal is 0 for most
  animals), any non-zero social vector decides the heading completely. Animals reverse and
  stay local.
- **The mechanism is inferred from code.** Only the effect of switching communication on or off
  was measured; the social vector's magnitude was not printed.
- **The food fix exposes the problem.** With the **pre-fix (v2.2.0) kernel and communication on**,
  CENOP's turning angles match DEPONS: median 51°, 10.7% reversals, mean cos +0.42
  (`turning_angles.py … prefix`). Phantom food kept `veTotal` high, so `crwContrib` outweighed
  the social vector. Under realistic food `veTotal` is about 0, `crwContrib` is about 0.001, and
  the social vector takes over.
- **Consequence:** the phantom-food fix must not be merged on its own. With communication on,
  it turns released movement into reversing movement.
- **Effect after the food fix (communication on):** CENOP animals forage a small area, empty the
  patches in it, and starve.

### Finding 3: dispersal over-triggering (located, cause open)

With phantom food fixed and communication off, CENOP tracks DEPONS closely to about day 105,
then dies off faster (`population_check.py`):

| day | 30 | 60 | 105 | 135 | 180 |
|---|---|---|---|---|---|
| CENOP N / mean E | 1996 / 8.65 | 1968 / 7.33 | 1414 / 4.78 | 445 / 4.24 | 65 / 4.19 |
| DEPONS N / mean E | 1998 / 8.49 | 1953 / 6.63 | 1418 / 5.39 | 981 / 4.83 | 528 / 4.54 |
| CENOP share dispersing | 0 | 3.5% | **75%** | **77%** | — |
| DEPONS share dispersing (tracked, PSM active) | 1.1% | 1.1% | 1.2% | 0.7% | 0.4% |

DEPONS seed 43 gives 992 at day 135, so the gap is not seed noise.

In CENOP, most animals enter PSM-Type2 dispersal and stay there; dispersing animals take fixed
directed steps instead of foraging.

**Controlled test (confirmed).** Dispersal was switched off in both models: DEPONS
`dispersal=off` (`scenarios/dispoff.json`); CENOP `_check_dispersal_trigger` disabled in memory
(`population_check.py … nodisp`). The two then agree to day 120 (N 1,452 against 1,472).

| day | 60 | 105 | 120 | 150 | 180 |
|---|---|---|---|---|---|
| DEPONS, dispersal off: N / E | 1965 / 7.16 | 1624 / 5.63 | 1452 / 5.11 | 992 / 3.97 | 568 / 3.56 |
| CENOP, dispersal off: N / E | 1974 / 7.32 | 1626 / 6.15 | 1472 / 5.70 | 1107 / 4.28 | 694 / 3.87 |

A **smaller residual remains**: from day 75, CENOP's mean energy is 0.3–0.6 higher, and N is 22%
higher at day 180. DEPONS seeds differ by about 3%. Open.

**Trigger against termination** (`dispersal_episodes.py`, days 0–120, communication off):

| | episode starts / animal-day | episode length (median / mean) |
|---|---|---|
| DEPONS (200 tracked) | 0.0040 | 1.0 / 1.1 days |
| CENOP (200 sampled) | **0.0179** (4.5×) | 1.0 / **4.3 days**, and many never end |

What this shows:
- **The trigger differs by one comparison.** DEPONS `Porpoise.performDailyStep` loops
  `i < tDisp`: three consecutive drops, `E[0] < E[1] < E[2] < E[3]`, where `CircularBuffer.get(0)`
  is the newest day. CENOP `_check_dispersal_trigger` loops `range(t_disp - 1)`: two drops. That
  predicts about 4× as many starts (1/6 against 1/24 under random noise), close to the observed
  4.5×.
- **Correcting it alone is not enough.** With CENOP `t_disp=4`, which reproduces DEPONS's three
  drops, 68% of CENOP animals are still dispersing at day 105, and N=117 at day 180.
- **Termination is the larger driver.** The daily energy-stop rule is the same in both, and the
  branch is reached daily. But CENOP episodes persist.

**What ends DEPONS episodes.** Measured at tick resolution from the 200 tracked animals, days
0–120:
- 0.0535 starts per animal-day, all at tick 0 of the day (the daily trigger);
- episode length median **5 ticks**, mean 12, p90 36;
- 1.3% of animal-ticks dispersing.

So DEPONS triggers dispersal often, and episodes end within hours through the per-step stop
rules in `AbstractPSMDispersal.disperse()`.

**Three further code differences, each measured in memory with `dispersal_waterstop.py`:**

| CENOP change, cumulative (communication off) | share dispersing, day 105 / 180 | N at day 180 (DEPONS 528) |
|---|---|---|
| none | 75% / — | 65 |
| trigger needs 3 drops (`t_disp=4`) | 68% / 62% | 117 |
| per-step stops only (no trigger fix) | 40% / 34% | 520 |
| + trigger fix | 33% / 29% | 562 |
| + stops checked on the dispersal (PSM) heading | 31% / 29% | 610 |
| + completion on path length | 33% / 26% | 588 |

1. **Missing per-step stop rules.** DEPONS ends dispersal when the point one dispersal step ahead
   (`ddisp/0.4` = 5 cells) is outside the landscape or at depth ≤ `wdisp` (4 m), or when
   `isEnoughWaterAhead()` is false. CENOP has none of these. Only the first two were modelled
   in the probe.
2. **Completion criterion.** DEPONS compares *path length travelled* with 0.95 × the target
   distance; CENOP compares *straight-line displacement* from the start. Targets are about 872
   cells (350 km / 0.4), which cannot be reached in straight-line terms inside Kattegat.
   CENOP dispersers at day 90 had a median path of 2,400 cells but only 115 cells of
   displacement.
3. **The trigger off-by-one**, as above.

**Still open.** Even with all three corrections, about a quarter of CENOP animals are dispersing,
against about 1% in DEPONS. CENOP episodes still last days where DEPONS's last a median of 5
ticks, so the mechanism that ends DEPONS episodes so fast has not yet been reproduced.
Remaining candidates:
- the `isEnoughWaterAhead()` stop (not modelled);
- the ordering of the dispersal step relative to the CRW and land avoidance;
- differences in the PSM-Type2 heading update;
- target selection (`findMostAttractiveMemCell` against the random fallback).

A DEPONS run with `debug=7` (`scenarios/debug7.json`) logged no stop reasons for porpoises 0 and
1. Either all their stops were the unlogged distance rule, or they did not disperse; this is
inconclusive.

Recommended next step: a line-by-line comparison of CENOP's PSM-Type2 path against DEPONS
`AbstractPSMDispersal.activate`/`disperse`, `DispersalPSMType2` and `Porpoise.performDailyStep`.
Then fix each difference test-first.

Two other things checked:
- **Calendar ruled out.** DEPONS `OFFSET_MONTH` shifts the month by only one tick.
- **Tracking perturbs DEPONS's trajectory slightly** (day 90: 1,633 untracked against 1,619 with
  200 tracked), but not its statistics. This answers an open item in the A1 report.

## Unverified

- **Now verified:**
  - JAX phantom food at runtime: before the fix, one visit floored all 15 empty cells of a 4×4
    grid.
  - DEPONS population on-patch rate (200 tracked animals).
- DEPONS run on the v3.0 landscape (not done). Not needed for the verdicts, because v3.2 has
  more food.
- Finding 3's root cause.
- Single seed (42) for CENOP comparisons. DEPONS seeds 42 and 43 agree closely.
- N=200 (TRACE scale) not re-checked after the fixes.
- Whether `communication_enabled=True` also distorts movement on homogeneous landscapes and in
  JASMINE scenarios. Likely, since the same composition applies, but not measured.

## Decision needed

1. **Phantom-food fix: done on branch `fix/phantom-food-2026-11-10` (`eba75f9`).** Floors only
   cells that had food, in Numba v1/v2, JAX (plus an active mask, so dead slots no longer eat)
   and Cython.
   - 6 tests went RED → GREEN (`tests/test_food_floor_parity.py`).
   - Fast suite 901 passed; slow tier 11 passed (homogeneous landscapes, unchanged).
   - Not merged.
2. **Communication default** (finding 2). **Decided 2026-11-10: option (b), default off.**
   - `SimulationParameters.communication_enabled` is now `False`, matching the UI checkbox, which
     was already off. Tests are in `tests/test_settings_defaults.py::TestCommunicationDefault`.
   - Rescaling the social vector relative to DEPONS's `crwContrib` (option c) remains an open
     JASMINE issue (CHANGELOG Known issues).
   - Side effect: with communication off, homogeneous landscapes now take the Cython fast path.
3. **Finish finding 3** before regenerating baselines or making any parity claim:
   - do a line-by-line port comparison of PSM-Type2 dispersal;
   - fix the trigger off-by-one, the missing per-step stops and the completion criterion
     (TDD), plus whatever the comparison finds;
   - then re-check the late residual.
4. **Then regenerate the Kattegat baselines and record the behaviour change** in the CHANGELOG.
   Any paper text or figures built on CENOP real-landscape populations need re-running.

## Reproduce

Probe scripts are in `scripts/investigations/2026-11-10-phantom-food/`. They are investigation
records: some paths are absolute to the author's workstation, so adjust them before reuse. Run
them with the `shiny` env from the repo root.

**Which code produced which numbers:**
- Evidence §1–6, including every "stock" column, ran on the **pre-fix code** (`42ebd7f`,
  `CENOP-JASMINE`). On the fix branch the stock kernel is the fixed one; `hypothesis_test.py`
  and `turning_angles.py … prefix` reinstate the pre-fix kernel in memory.
- Findings 2–3 ran on the fixed code (`eba75f9`).
- The DEPONS probe scenarios (200 tracked animals; dispersal off) are in `scenarios/`. Run them
  with `python3 -m validation.depons_java run --scenario <file> --out <dir>`. Their outputs are
  not committed.

The v3.2 copy under v3.0 names is the harness's extracted landscape
(`.depons-cache/landscape/Kattegat`), with `prey0000_MM_XX.asc` → `prey0000_MM.asc` and
`salinity0000_MM_XX.asc` → `salinity0000_MM.asc`, placed in `<dir>/Kattegat/`.

```bash
python3 scripts/investigations/2026-11-10-phantom-food/traj_compare.py 42 43
python3 scripts/investigations/2026-11-10-phantom-food/landscape_food_stats.py
python3 scripts/investigations/2026-11-10-phantom-food/depons_intake.py <seed-42>/Statistics.csv
python3 scripts/investigations/2026-11-10-phantom-food/cenop_intake.py <dir> 63
python3 scripts/investigations/2026-11-10-phantom-food/cenop_ledger.py <dir>
python3 scripts/investigations/2026-11-10-phantom-food/hypothesis_test.py <dir> stock|patched
python3 scripts/investigations/2026-11-10-phantom-food/occupancy_probe.py <dir> stock|patched 63
python3 scripts/investigations/2026-11-10-phantom-food/movement_compare.py <dir> stock|patched
# findings 2-3 (run on the fixed code; <rp> = RandomPorpoise.csv of a DEPONS run with
# trackedPorpoiseCount=200, simYears=1, seed 42, made via a scenario override)
python3 scripts/investigations/2026-11-10-phantom-food/depons_onpatch_tracked.py <rp> <patches.asc> [<untracked Statistics.csv>]
python3 scripts/investigations/2026-11-10-phantom-food/onpatch_rate.py <rp> <dir> 30
python3 scripts/investigations/2026-11-10-phantom-food/block_distribution.py <PorpoisePerBlock.csv> <dir>
python3 scripts/investigations/2026-11-10-phantom-food/revisit_compare.py <rp> <dir>
python3 scripts/investigations/2026-11-10-phantom-food/turning_angles.py <rp> <dir> stock|novt|nocomm
python3 scripts/investigations/2026-11-10-phantom-food/crw_generated_vs_realized.py <dir>
python3 scripts/investigations/2026-11-10-phantom-food/population_check.py <dir> on|off 180 <Statistics.csv> [nodisp]
python3 scripts/investigations/2026-11-10-phantom-food/dispersal_episodes.py <rp> <dir>
python3 scripts/investigations/2026-11-10-phantom-food/dispersal_waterstop.py <dir> water[+tdisp4][+psmheading][+pathlen] 180 <Statistics.csv> <rp>
```
