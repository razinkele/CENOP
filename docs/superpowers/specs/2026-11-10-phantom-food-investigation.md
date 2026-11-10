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

### Finding 3: dispersal over-triggering (root cause established)

> **Superseded in part.** The CENOP "share dispersing" figures in this first half (75%, 68%,
> 33%, …) are day-boundary snapshots taken just after the daily trigger. They are not
> comparable with DEPONS's tick averages: the stock all-tick share is 20.3%. On the same
> grounds, the "4.5×" start ratio, "termination is the larger driver" and "Still open" are
> superseded. See **Line-by-line PSM-Type2 comparison** and **Like-for-like rates** below for
> the current numbers and conclusion.

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

**Line-by-line PSM-Type2 comparison (done).**

Sources compared:
- DEPONS: `Porpoise.move()` (209–313), `performDailyStep()` (1073–1120), `AbstractPSMDispersal`
  (`activate` 69–124, `disperse` 148–223, `findMostAttractiveMemCell` 296–331) and
  `DispersalPSMType2.calculateNewHeading` (68–87).
- CENOP `population.py`: `_check_dispersal_trigger`, `_select_dispersal_target`,
  `_set_random_dispersal_target`, `_apply_dispersal_heading`, `_update_dispersal`,
  `_handle_land_avoidance`; plus `heading_position_reflect_kernel` in `kernels.py`.
- Runtime values from `DEPONS.rs/parameters.xml`: tdisp 3, PSM_angle 40, PSM_dist N(350;100),
  PSM_tol 5, PSM_log 0.6, ddisp 2, wdisp 4.

The CENOP column describes the code before the dispersal port; "Fixed in" gives the commit(s)
that fixed each row on branch `fix/phantom-food-2026-11-10`.

| # | Stage | DEPONS 3.2 | CENOP before the fix | Status | Fixed in |
|---|---|---|---|---|---|
| D1 | Trigger | `i < tDisp`: three strict drops | `range(t_disp - 1)`: two drops | known | `a255bbe` |
| D2 | Trigger | daily mean rounded to 3 d.p., so ties break the chain | not rounded | minor | `a255bbe` |
| D3 | Trigger gate | no gate; with fewer than 50 memory cells, uses a random target | no trigger with fewer than 50 visited PSM cells | new | `8e17aa7` |
| D4 | Memory target | fitness = food/ticks over candidates in [pref ± tol]/0.4 (inclusive); a zero-fitness candidate still wins | multiplies by the PSM-**Type3** `exp(-q1·d)`; strict `<`; random fallback if the best expectation over *all* cells is ≤ 0 | new, minor | `8e17aa7` |
| D5 | Random target | up to 1000 tries at (pref + U(−tol, tol))/0.4; cell must be in the grid, depth > 0, prey > 0 this month; otherwise **no dispersal** | one try at pref/0.4, reflected into the box, no water or prey check, always disperses; target distance stays pref/0.4 after reflection | new | `8e17aa7` |
| D6 | Heading at activation | `previousStepHeading = targetHeading` | `_prev_step_heading` captured *before* target selection (3148/3236 vs 3150/3238); the target heading written to `heading` is overwritten next tick, so the animal disperses along its last foraging heading | **new, bug** | `8e17aa7` |
| D7 | Turn angle | ±`PSM_angle` = 40° | reads `psm_type2_random_angle`, which `SimulationParameters` lacks, so falls back to 20°; the same at the JAX call (2580); the UI `psm_angle` is ignored | **new, bug** | `d75636c` |
| D8 | Logistic input | path length / (0.95 × start-to-target) | straight-line displacement / target distance | known | `d75636c` |
| D9 | Per-step stops | stops on any of: the point 5 cells ahead is off the grid, or its depth ≤ wdisp; also on the not-enough-water state, but that is written only by `stdMove`/`avoidLand` and is frozen while dispersing, so it can fire only on the first `disperse()` call, before any move (an activation filter, not an episode end) | none | known | per-step stops `5a8ae40`; first-call water filter fixed: `2e7401b`, `4c39546` (NumPy), `480c7b7` (JAX) |
| D10 | Completion | path length ≥ 0.95 × start-to-target, checked *before* the step | displacement ≥ 0.95 × target distance, checked *after* the step | known | `5a8ae40` (+ `0b72289`: non-CRW modules keep displacement completion) |
| D11 | Land while dispersing | `forward()` has no land test; the stops keep the animal off land | land avoidance turns `heading`; the next tick rebuilds it from `_prev_step_heading`, so the turn is lost | new | `5a8ae40` |
| D12 | Same-tick fallthrough | after a stop inside `disperse()`, `stdMove` runs in the same tick | stop evaluated after the dispersal step | minor | `5a8ae40` |
| D13 | CRW state | `prevAngle` and `prevLogMov` frozen while dispersing (`stdMove` skipped) | updated every tick for dispersers | minor | `5a8ae40` |
| D14 | JAX backend | — | the JAX backend (`jax_kernels.py`, `tick_jax.py`) now mirrors the fixed NumPy behaviour (Task 5, `b9659a1`, `eae0d3a`, `bd55bda`), so D1–D13 apply there too; every later change must be mirrored | fixed with D1–D13 | `b9659a1`, `eae0d3a`, `bd55bda` (+ `bb7c829`: new activations get the same-day stops) |
| D15 | Foraging land check (outside dispersal) | `checkDepth()` samples every 0.1 cell along the heading out to `presMov` cells (4× the step) before turning; its result is also the not-enough-water state that D9 reads | land avoidance tests only the destination cell at the step distance | established as the main cause of the dispersal-off residual (Finding 4) | fixed: `75df4b9`, `b0c770d`, `14dc077`, `c6d4194` (NumPy/Numba), `480c7b7` (JAX); acceptance: see Step 5 (accepted, one borderline bin) |
| D16 | Reference-memory order | DEPONS `refMemTurn` runs after the dispersal `forward()`, so it sees the post-move position | CENOP's `_update_reference_memory` runs before the disperser moves | deferred with D15 (pre-existing, outside D1–D14) | open |
| — | Logistic shape | `LogisticDecreaseSSLogis(1, 0, 0.6)` = 1/(1+e^(−x/0.6)), *increasing* despite the name | same | match | — |
| — | Dispersal energy cost | `E_USE_PER_KM = 0` | 0 | match | — |
| — | Daily energy stop | 10-day buffer; stop if E[0] > min(E[1..7]), in the same daily step as the trigger | 10 columns; same rule, same tick | match | JAX new activations: `bb7c829` |
| — | Memory cell | 5 cells, centre +2 (integer division) | 5 cells, centre +2.5 | negligible | — |

The earlier "three code differences" are D1, D9 and D10.

**DEPONS dispersal from the tracked-200 run (days 0–120, `depons_disp_targets.py`):**
- 98 of 200 animals dispersed at least once, with 1,151 starts.
- Starts are spread through the middle of the grid: only 8% have grid-y < 150 or > 850.
- Targets: median 550 cells (220 km; p10 380, p90 737), piled up at the N/S ends.
- Episodes: median 5 ticks (p90 36); median displacement 21 cells, i.e. a short straight run.

Interpretation:
- Successful targets come from the lower tail of N(350,100) km, which fits D5: activation fails
  for many animals.
- The 51% that never dispersed still need sizing against how often the trigger condition held.

**Measurement caveat.** The CENOP "share dispersing" figures in the table above were sampled at
`tick % 720 == 0`. That is a day-boundary tick, immediately after the daily trigger, whereas the
DEPONS share averages every tick of a 15-day window. With short episodes, the CENOP figure is
inflated, so the "quarter vs 1%" residual is not yet like-for-like. N at day 180 with the stops
(520–610) already brackets DEPONS (528).

**Like-for-like rates (tick resolution, days 0–120).** Scripts: `depons_disp_rates.py` (DEPONS,
tracked-200 CSV, end reasons classified from the last step) and `cenop_disp_rates.py` (CENOP, all
2000 animals, reasons attributed in code; the CENOP rows below come from the `057d391` version of
the probe). Definitions:
- predicate: three strict drops in the 3-d.p. daily mean, regardless of dispersal state;
- activation: visible starts per eligible predicate-day.

| | DEPONS | CENOP stock | CENOP + D1 + D9 (`water+tdisp4`) |
|---|---|---|---|
| predicate rate / animal-day | 0.405 | 0.443 | 0.359 |
| activation rate | **0.133** | 0.070 | **0.242** |
| episode moves, median / mean / p90 | **5 / 12.1 / 36** | 48 / 235 / 720 | **10 / 18.2 / 48** |
| end reasons | depth ahead 86%, other 9%, energy stop 3%, complete 2% | energy stop 95% | water/off-grid ahead 96% |
| **tick share dispersing** | **1.34%** | 20.3% | **3.2%** |
| N at day 120 (DEPONS seed 42: 1,210) | 1,210 | 890 | 1,455 |

What this shows:
- **The "quarter vs 1%" residual was mostly the sampling artifact.** Over all ticks, the trigger
  fix plus the stops leaves CENOP at 2.4× DEPONS's share, not 25×.
- **The upstream energy dynamics agree:** the predicate rates are close.
- **The remaining 2.4× has two factors:**
  - activation 1.8× higher, since CENOP always finds a target (D5);
  - episodes 1.5× longer (D6, D7 or D11 candidates).
- **DEPONS's activation loss splits** as follows: 89.5% of eligible days survive the same-day
  energy stop, but only 15% of those become a visible episode. A median of 5 moves implies a
  per-step stop hazard of roughly 13%, so target failure (D5) most likely accounts for the
  larger part of the loss.
- **Population is a separate gap.** DEPONS dispersal costs about 13% of N by day 105 compared
  with dispersal off (1,418 vs 1,624). CENOP + D1 + D9 barely changes N (1,455 vs 1,472 with
  dispersal off at day 120), even though its share dispersing is higher. A candidate is D6:
  DEPONS targets cluster at the grid's N/S ends, so dispersal relocates animals, while CENOP
  disperses along the last foraging heading.

**Corrections combined, in memory** (pre-fix probe from `057d391`, see "Which code produced which
numbers": `cenop_disp_rates.py <mode> 180 120 <seed>`; rates over days
0–120; seed 42 unless marked):

| Variant | tick share | activation | moves median / mean | N at days 120 / 150 / 180 |
|---|---|---|---|---|
| **DEPONS** s42 / s43 | **1.34%** (s42) | **0.133** (s42) | **5 / 12.1** (s42) | **1,210 / 792 / 528** · 1,203 / 824 / 546 |
| CENOP stock | 20.3% | 0.070 | 48 / 235 | 890 / — / — |
| + D1 + D9 | 3.2% | 0.242 | 10 / 18.2 | 1,455 / — / — |
| + D1 + D9 + D6 | 3.2% | 0.256 | 8 / 16.5 | 1,388 / 868 / 487 |
| + D1 + D9 + D6, s43 | 3.1% | 0.255 | 7 / 15.9 | 1,384 / 875 / 485 |
| + D1 + D9 + D3–D5 | 2.45% | 0.188 | 10 / 17.5 | 1,445 / 1,035 / 641 |
| + D1 + D9 + D3–D6 | 1.13% | 0.125 | 5 / 11.3 | 1,299 / 888 / 589 |
| **+ all (D1, D3–D10)** | **1.07%** | **0.128** | **5 / 10.6** | **1,282 / 910 / 627** |
| **+ all, s43** | **1.13%** | **0.134** | **5 / 10.6** | **1,291 / 893 / 573** |
| dispersal off: DEPONS s42 / s43 | — | — | — | 1,452 / 992 / 568 · 1,458 / 1,014 / 592 |
| dispersal off: CENOP s42 / s43 | — | — | — | 1,472 / 1,107 / 694 · 1,478 / 1,088 / 728 |

- In the full variant, 97% of episodes end on the water/off-grid stop. DEPONS: depth 86% plus
  "other" 9%, which is classifier error from heading jitter and grid rounding; the
  not-enough-water state cannot end a visible episode (see D9).
- Target failures: about 66k activation attempts per run.
- CENOP seed-to-seed spread is about ±10 animals at day 120 and ±50 at day 180.
- The CENOP dispersal-off s42 run reproduces the earlier controlled-test numbers exactly (same
  landscape, communication off).

**Dispersal's population cost** (N with dispersal on ÷ N with it off, same model and seed):

| | day 120 | day 150 | day 180 |
|---|---|---|---|
| DEPONS s42 / s43 | 0.833 / 0.825 | 0.798 / 0.813 | 0.930 / 0.922 |
| CENOP, all corrections, s42 / s43 | 0.871 / 0.873 | 0.822 / 0.821 | 0.904 / 0.787 |

**Root cause of finding 3: established** (two seeds for the rates; mechanism partly inferred):
- **D1 and D9 set how often and how long.** The trigger needed one drop too few, and nothing
  ended an episode except the daily energy rule.
- **D5 is the activation filter.** In Kattegat, most targets at N(350,100) km are off the grid or
  on land, so DEPONS usually does not disperse at all.
- **D6 changes N while leaving frequency unchanged.** Adding it lowers N at day 120 by 67–146
  animals (well above the ±10 seed spread). The proposed mechanism is relocation: DEPONS heads
  for targets at the grid's ends, while CENOP walked along its last foraging heading. That
  mechanism is not measured.
- **D7, D8 and D10:** differences are within seed noise on this landscape (one seed).

With all corrections:
- CENOP's dispersal statistics match DEPONS on both seeds: activation 0.128–0.134 vs 0.133,
  median 5 moves in both.
- Dispersal's population cost matches at day 150 (about 18% vs 19–20%). It is about 4 points
  smaller at day 120 on both seeds, a possible small remaining dispersal difference (unmodelled
  in the probes: D2, D11–D13, the first-call water filter). Day 180 is noisy.
- Most of the remaining N excess is the **non-dispersal residual**, which replicates with
  dispersal off in both models: +1% / +7–12% / +22–23% at days 120 / 150 / 180. D15 is a
  candidate (untested).

Two other things checked:
- **Calendar ruled out.** DEPONS `OFFSET_MONTH` shifts the month by only one tick.
- **Tracking perturbs DEPONS's trajectory slightly** (day 90: 1,633 untracked against 1,619 with
  200 tracked), but not its statistics. This answers an open item in the A1 report.

### Finding 4: the dispersal-off residual (D15; ported, acceptance in Step 5)

Runs: dispersal off in both models, communication off.
- DEPONS: `scenarios/dispoff_tracked200.json`, 200 tracked animals.
- CENOP: the fixed code at `e016fe4`, run with the trigger disabled in memory.

**Step 1: the gap is in intake** (`cenop_budget.py`, `depons_budget.py`; intake per animal-tick
inferred from landscape food drops, the same way in both models):

| days | 0–15 | 15–30 | 30–45 | 45–60 | 60–75 | 75–90 |
|---|---|---|---|---|---|---|
| DEPONS s42 | 0.00352 | 0.00329 | 0.00333 | 0.00326 | 0.00326 | 0.00336 |
| CENOP s42 | 0.00360 | 0.00343 | 0.00346 | 0.00340 | 0.00344 | 0.00353 |
| CENOP s43 | 0.00362 | 0.00343 | 0.00345 | 0.00341 | 0.00344 | 0.00352 |

CENOP eats 2–5% more per animal-tick from the first fortnight, while N and deaths are still equal.
The other budget terms do not explain the gap:
- Energy use is deterministic and has the same formula in both models (DEPONS `updEnergeticStatus`
  715–787: `0.001·scaling·EUse`, ×`eLact` when lactating, `E_USE_PER_KM = 0`).
- The lactating share is 0 before about day 120.
- Mortality is a fixed function of energy.

So the residual is an intake residual. CENOP's exact intake is about 0.0001 above its inferred
value, an inference bias that applies to both models equally.

**Step 2: spatial fingerprint** (`spatial_fingerprint.py`; days 0–30, seed 42; cells are share of
animal-ticks / intake per animal-tick; each model uses its own cell lookup; intake is attributed
to the cell where it was eaten):

| distance to coast | <1 km | 1–2 | 2–5 | 5–10 | >10 |
|---|---|---|---|---|---|
| DEPONS | 0.009 / 0.00248 | 0.034 / 0.00380 | 0.231 / 0.00311 | 0.271 / 0.00361 | 0.455 / 0.00369 |
| CENOP | 0.079 / 0.00260 | 0.073 / 0.00288 | 0.204 / 0.00324 | 0.226 / 0.00392 | 0.418 / 0.00393 |

| depth | 1–5 m | 5–10 | 10–20 | 20–30 | >30 |
|---|---|---|---|---|---|
| DEPONS | 0.035 / 0.00246 | 0.133 / 0.00266 | 0.407 / 0.00357 | 0.297 / 0.00382 | 0.126 / 0.00393 |
| CENOP | 0.084 / 0.00229 | 0.172 / 0.00277 | 0.377 / 0.00375 | 0.243 / 0.00402 | 0.124 / 0.00441 |

DEPONS has 0.16% of positions on cells under 1 m (it never checks the composed step); CENOP has 0.

What this shows:
- **D15's fingerprint is real and large.** CENOP spends 9× as much time within 1 km of the coast and
  2.4× as much in 1–5 m water. This fits DEPONS's look-ahead of `presMov` cells (4× the step),
  sampled every 0.1 cell, which turns animals away from the coast earlier.
- **Band intakes are too noisy to carry a mechanism.** On seed 42, the coastal bands are the
  poorer places to eat in both models, and CENOP's excess appears offshore (+5% at 10–30 m, +12%
  beyond 30 m). But DEPONS's band intakes come from only 200 tracked animals:
  - seed 43's reconstructed DEPONS mean is 0.00369, above CENOP's 0.00360;
  - its 10–20 m band is 0.00388 against CENOP's 0.00375.
  The occupancy difference replicates on both seeds; the band intakes do not.
- **Proposed mechanism (not measured): competition and depletion.** With more animals on the coast,
  offshore patches are shared among fewer foragers, and CENOP also harvests coastal regrowth that
  DEPONS leaves. A distribution comparison cannot settle this.
- **The decisive test is an in-memory port of DEPONS's land sequence**, accepted on DEPONS's
  dispersal-off intake, energy and N, seeds 42 and 43. If the gap stays, the cause lies in eating
  or regrowth, not movement.

**Step 3: the in-memory port closes the gap.** The port is `d15_land_port.py`, applied with
`cenop_budget.py <dir> <seed> 180 port`. It does three things, as in DEPONS:
- **Before composition:** after the CRW turn, run `checkDepth` (look-ahead of `presMov` cells,
  sampled every 0.1 cell, depth > 0) and `avoidLand` (±40/70/120° + U(0,10), escape points at
  depth ≥ 1 m), with beh 4/5.
- **After composition:** no land check, only border reflection.
- **After the move:** the existing beh-6 rollback, plus its heading flip.

N with dispersal off (seed 42 / seed 43):

| day | 90 | 120 | 150 | 165 | 180 |
|---|---|---|---|---|---|
| DEPONS | 1737 / 1722 | 1442 / 1417 | 989 / 994 | 780 / 784 | ≈577 / ≈593 |
| CENOP before | 1765 / 1761 | 1472 / 1478 | 1107 / 1088 | 911 / 903 | ≈694 / ≈728 |
| **CENOP + port** | **1746 / 1764** | **1420 / 1449** | **987 / 1002** | **774 / 769** | **≈622 / ≈608** |

- **Intake:** with the port, intake per animal-tick is within ±2% of DEPONS, and the sign varies by
  bin. Examples:
  - days 0–15: 0.00352 / 0.00357 against 0.00352 / 0.00352;
  - days 60–75: 0.00329 (s43) against 0.00322 (s43);
  - days 90–105: 0.00347 against 0.00354;
  - days 120–135: 0.00376 against 0.00381.
  Before the port it was +2–5% in every early bin.
- **Coastal occupancy:** time within 1 km of the coast falls from 7.9% to 0.8–1.2% (DEPONS
  0.9% / 1.8%), and time at depth < 1 m is 0.1% (DEPONS 0.16% / 0.6%).
- **Energy now undershoots.** Mean energy with the port is 0.1–0.4 below DEPONS over days 90–150
  on both seeds:

  | day | 90 | 120 | 135 | 150 | 165 |
  |---|---|---|---|---|---|
  | DEPONS | 6.36 / 6.33 | 5.24 / 5.31 | 4.59 / 4.67 | 4.09 / 4.17 | 3.79 / 3.88 |
  | CENOP + port | 6.22 / 6.16 | 5.10 / 4.99 | 4.31 / 4.30 | 3.88 / 3.78 | 3.71 / 3.75 |

**Root cause of most of the dispersal-off residual: D15, established on two seeds.** Porting
DEPONS's land sequence moves CENOP from +12–17% N at days 150–165 to within about 2% of DEPONS, and
removes the early intake excess. The mechanism (crowding at the coast, then competition) is
proposed, not measured.

**Remainders (open):**
- **Energy undershoot:** mean energy 0.1–0.4 below DEPONS at days 90–150 with the port, with N
  matching. **It is not spatial.** With the port, occupancy matches DEPONS in every band (days
  0–30, seed 42; DEPONS s42 / s43 in brackets):
  - distance to coast: <1 km 0.011 (0.009/0.018), 1–2 km 0.040 (0.034/0.043), 2–5 km 0.234
    (0.231/0.253), 5–10 km 0.264 (0.271/0.268), >10 km 0.452 (0.455/0.418);
  - depth: 1–5 m 0.031 (0.035/0.032), 5–10 m 0.137 (0.133/0.139), 10–20 m 0.409 (0.407/0.407),
    20–30 m 0.283 (0.297/0.272), >30 m 0.140 (0.126/0.144).
  So the floor-vs-round cell lookup does not matter at this resolution.
- **Port branch counts** (seed 42, days 0–90, per animal-tick): `avoidLand` 6.6%, beh 4 (walk back
  along the position list) 0.036%, beh 5 (deepest neighbour) 0.032%, beh 6 (post-move rollback)
  0.002%. At that rate, CENOP's reference-memory position buffer is an adequate stand-in for
  DEPONS's position list.
- **Late deaths:** days 165–180 show fewer CENOP deaths (152 / 161 against 203 / 191) at mean
  energy 3.7 against 3.8. This coincides with CENOP's lactating share rising to 4–5%, and
  lactating females abandon the calf instead of dying. DEPONS outputs no lactating share, and the
  death hazard by energy band has not been measured. Candidates: reproduction timing and the
  mortality implementation.
- Accept the fix through day 165; treat day 180 as informational.

**Step 4: combined check, dispersal on (fixed code `e016fe4`) + port, against DEPONS PSM-Type2**
(`cenop_disp_rates.py <dir> stock 180 120 <seed> --d15-port`; seed 42 / seed 43):

| | DEPONS | CENOP fixed, no port | CENOP fixed + port |
|---|---|---|---|
| N at day 120 / 150 / 180 | 1210 / 792 / 528 · 1203 / 824 / 546 | 150: 941 / 893 | 1194 / 815 / 568 · 1212 / 845 / 598 |
| on/off ratio at days 120 / 150 / 180 | 0.839 / 0.801 / 0.915 · 0.849 / 0.829 / 0.921 | 0.872 / 0.850 / 0.867 · 0.873 / 0.821 / 0.787 | 0.841 / 0.826 / 0.913 · 0.836 / 0.843 / 0.984 |
| activation rate | 0.133 | 0.126 / 0.134 | **0.151 / 0.162** |
| episode moves, median / mean | 5 / 12.1 | 5 / 11.0 · 5 / 10.8 | 5 / 10.5 · 5 / 10.8 |
| tick share dispersing | 1.34% | 1.09% / 1.15% | 1.28% / 1.38% |

What this shows:
- **Population and dispersal's cost now track DEPONS** to day 150 on both seeds (within +3%), with
  both arms ported.
- **Activation rises** to 0.151 / 0.162; seed 43 is just above the accepted 0.10–0.16. With
  DEPONS's spatial distribution, more animals pass the activation filter.
- **The first-call water filter is not the answer.** It (D9: a new disperser is stopped if its last
  foraging step left not enough water ahead) was first proposed as the candidate. In the port, the
  stored state is False only when the final path check fails, i.e. when beh 5 fires (about 0.03%
  of animal-ticks). So the filter barely changes activation. It is still ported, for fidelity.
- **No activation gap.** DEPONS's 0.133 came from one seed's 200 tracked animals. A seed-43 DEPONS
  run (`tracked200.json --seeds 43`) gives:
  - activation 0.173;
  - episode moves median 5 / mean 10.1;
  - tick share 1.36%;
  - end reasons: depth 86%, other 9%, complete 3%, energy stop 2%.

  DEPONS's activation therefore spans 0.133–0.173 across seeds, and CENOP + port's 0.151 / 0.162
  lies inside that span.

**Step 5: acceptance on the fixed source** (D15 and the D9 first-call filter ported; commits
`75df4b9`, `b0c770d`, `14dc077`, `c6d4194`, `2e7401b`, `4c39546`, `480c7b7`, plus the docs commit
of this section). The probe prototype switches (`port`, `--d15-port`) were removed from
`cenop_budget.py`, `cenop_disp_rates.py` and `spatial_fingerprint.py`; `d15_land_port.py` stays as
the historical prototype and applies only to code before commit `46e2a39`. Runs: NumPy/Numba,
`JAX_PLATFORMS=cpu`, v3.2 Kattegat copy, communication off, 2000 animals, seeds 42 / 43.
DEPONS values are the dispersal-off tracked-200 runs (`depons_budget.py`); intake per animal-tick
is the inferred column, at 7 d.p. for the ±3% test.

**Result: accepted with one borderline bin (controller ruling).** One metric is outside its range:
the dispersal-off intake of seed 43 in the day 45–60 bin (+3.27% against ±3%). Everything else is
inside. No tuning was done. Accepted because the same bin is +1.0% against seed 42 (DEPONS's
seed-to-seed spread there is about 1%), the other 11 intake bins are within +2.3%, every
population-level metric passes on both seeds, and the ±3% band was a judgement.

| Metric (range) | DEPONS s42 / s43 | CENOP s42 | CENOP s43 | Result |
|---|---|---|---|---|
| off N at day 90 (±4%) | 1737 / 1722 | 1741 (+0.2%) | 1777 (+3.2%) | pass |
| off N at day 120 (±4%) | 1442 / 1417 | 1451 (+0.6%) | 1464 (+3.3%) | pass |
| off N at day 150 (±4%) | 989 / 994 | 1004 (+1.5%) | 1013 (+1.9%) | pass |
| off N at day 165 (±4%) | 780 / 784 | 795 (+1.9%) | 789 (+0.6%) | pass |
| off intake days 0–15 (±3%) | 0.0035233 / 0.0035183 | 0.0035416 (+0.5%) | 0.0035462 (+0.8%) | pass |
| off intake days 15–30 | 0.0032873 / 0.0033413 | 0.0033631 (+2.3%) | 0.0033698 (+0.9%) | pass |
| off intake days 30–45 | 0.0033274 / 0.0032658 | 0.0033294 (+0.1%) | 0.0033368 (+2.2%) | pass |
| off intake days 45–60 | 0.0032553 / 0.0032269 | 0.0032879 (+1.0%) | 0.0033325 (**+3.3%**) | **s43 fail** |
| off intake days 60–75 | 0.0032631 / 0.0032177 | 0.0033285 (+2.0%) | 0.0032835 (+2.0%) | pass |
| off intake days 75–90 | 0.0033553 / 0.0033767 | 0.0033993 (+1.3%) | 0.0033897 (+0.4%) | pass |
| time within 1 km of coast, days 0–15 / 15–30 (0.5–2.5%) | — | 1.16% / 0.95% | 1.11% / 1.06% | pass |
| activation rate (0.11–0.19) | 0.133 / 0.173 | 0.152 | 0.157 | pass |
| episode moves median (4–6) / mean (8–15) | 5 / 5; 12.1 / 10.1 | 5 / 10.5 | 5 / 10.7 | pass |
| tick share dispersing (0.8–1.8%) | 1.34% / 1.36% | 1.26% | 1.35% | pass |
| on/off N ratio, day 150 (0.75–0.87) | 0.801 / 0.829 | 824 / 1004 = 0.821 | 817 / 1013 = 0.807 | pass |

Informational:
- Stock (dispersal on) N at days 120 / 150 / 180: s42 1203 / 824 / 598 (DEPONS 1210 / 792 / 528);
  s43 1193 / 817 / 587 (DEPONS 1203 / 824 / 546). Day 180 is +13% / +8% above DEPONS.
- Spatial fingerprint, seed 42, days 0–30: coast <1 km 1.1%, depth <1 m 0.10%.
- Remainders unchanged: mean energy (dispersal off, days 90 / 120 / 135 / 150 / 165) s42
  6.32 / 5.13 / 4.41 / 3.92 / 3.77, s43 6.23 / 5.08 / 4.36 / 3.91 / 3.77, against DEPONS
  6.36 / 5.24 / 4.59 / 4.09 / 3.79 (s42) and 6.33 / 5.31 / 4.67 / 4.17 / 3.88 (s43): an undershoot
  of 0.0–0.3. Deaths days 165–180: 169 / 160 against DEPONS 203 / 191, lactating share 4.3% /
  6.3%. Both remainders stay open.
- Lithuania (default landscape, 60 days, 10000 animals, defaults), this code against `46e2a39`:
  N 9856 / 9839 against 9989 / 9991 (s42 / s43); mean energy 15.51 / 15.50 against 14.06 / 14.05;
  share of time within 1 km of the coast (disttocoast in km) 19.9% / 19.8% against 20.5% / 20.5%.
  No DEPONS reference exists for Lithuania.
- Known deviations introduced: beh-5 depth ties use a fixed neighbour order (DEPONS shuffles);
  look-ahead points are reflected at the grid edge as in DEPONS's BouncyBorders, but beh-5
  neighbours outside the grid read as land.
- JAX is about 5x slower per tick on real landscapes after this change (about 70 ms against 13 ms
  at 2000 animals; Numba about 7 ms). It stays opt-in.

**Next:** regenerate the Kattegat baselines. The two remainders (energy undershoot, late deaths) are the next
investigation.

## Unverified

- **Now verified:**
  - JAX phantom food at runtime: before the fix, one visit floored all 15 empty cells of a 4×4
    grid.
  - DEPONS population on-patch rate (200 tracked animals).
- DEPONS run on the v3.0 landscape (not done). Not needed for the verdicts, because v3.2 has
  more food.
- Finding 3's root cause: **now established** with in-memory probes, and the fixed code was
  verified 2026-11-10 against DEPONS rates on two seeds (Decision needed, item 3).
- The non-dispersal residual (CENOP N +12–22% from day 150, with dispersal off in both models).
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
3. **Finding 3: PSM-Type2 dispersal ported to DEPONS 3.2, on branch `fix/phantom-food-2026-11-10`
   (not merged).** Acceptance measured 2026-11-10: **met on both seeds.**
   - Commits:
     - `8e17aa7` target selection: validated targets, no dispersal when none found, start heading
       for the target (D3–D6)
     - `a255bbe` trigger needs `t_disp` strict drops of the 3-d.p. daily mean (D1, D2)
     - `d75636c` PSM turn uses `psm_angle` (40) and path length (D7, D8)
     - `5a8ae40` per-step stops (off-grid, shallow ahead), path-length completion, same-tick CRW
       (D9–D13)
     - `0b72289` non-CRW movement modules keep the straight-line displacement completion
     - `b9659a1` JAX backend mirrors the per-tick dispersal
     - `eae0d3a` JAX skips the depth stop when there is no depth grid
     - `bd55bda` JAX rounds the daily energy mean exactly as NumPy
     - `bb7c829` JAX: a new activation gets the same-day energy stop and the same-tick turbine
       deterrence cancel, as on NumPy
   - Controller ruling: the non-CRW modules (JASMINE physics, hybrid;
     `_update_movement_module_generic`) keep the pre-existing straight-line displacement
     completion. DEPONS parity applies to the CRW paths only.
   - Not ported at the time: the not-enough-water first-call filter and D15. Both are now ported
     (Finding 4, Step 5). D16 remains deferred.
   - Acceptance (`cenop_disp_rates.py <mode> 180 120 <seed>` on the fixed code, rates over days
     0–120; the probe counts animals that make a dispersal move, i.e. after `_disperse_step`):

| Metric | DEPONS | CENOP s42 | CENOP s43 | Accepted | Result |
|---|---|---|---|---|---|
| activation (starts per eligible predicate-day) | 0.133 | 0.126 | 0.134 | 0.10–0.16 | pass |
| episode moves median | 5 | 5 | 5 | 4–6 | pass |
| episode moves mean | 12.1 | 11.0 | 10.8 | 8–15 | pass |
| tick share dispersing | 1.34% | 1.09% | 1.15% | 0.8–1.8% | pass |
| on/off N ratio, day 150 | 0.80 / 0.81 (s42 / s43) | 0.850 | 0.821 | 0.75–0.87 | pass |

   - CENOP N at days 120 / 150 / 180: s42 stock 1,283 / 941 / 613, dispersal off 1,472 / 1,107 /
     694; s43 stock 1,281 / 893 / 607, dispersal off 1,478 / 1,088 / 728. On/off ratio at days
     120 / 180: s42 0.872 / 0.883, s43 0.867 / 0.834 (DEPONS 0.833 / 0.930 s42, 0.825 / 0.922
     s43).
   - 96–97% of episodes end on the off-grid/shallow look-ahead stop, 2% on the energy rule, 1–2%
     on path completion.
   - JAX population-level acceptance was not measured: the probe wraps `_disperse_step`, which
     the JAX path bypasses. JAX parity is asserted by unit tests in `tests/test_jax_tick.py`:
     `TestJaxDisperseStep::test_jax_disperse_step_matches_numpy`,
     `TestJaxDisperseStep::test_dispersal_heading_matches_reference`,
     `TestJaxHeadingAndPosition::test_dispersers_keep_psm_heading_and_crw_state`,
     `TestJaxEnergyHistory::test_jax_energy_history_rounded_to_3dp`,
     `TestJaxEnergyHistory::test_jax_energy_history_rounding_matches_numpy_at_boundaries`,
     `TestJaxDispersalUpdate::test_deterrence_cancels_dispersal`,
     `TestJaxDispersalUpdate::test_no_distance_completion`,
     `TestJaxTickComposition::test_tick_movement_threads_dispersal_params`,
     `TestJaxStepDispersal::test_step_jax_disperse_step`,
     `TestJaxStepDispersal::test_step_jax_without_depth_grid_skips_depth_stop` and
     `TestJaxStepDailyDispersalOrder::test_new_activation_gets_same_day_stops`.
   - Targets at N(350,100) km rarely fit on small grids, so dispersal is now rare on Lithuania
     (the default landscape), Homogeneous and the 400×400 landscapes: random-target success
     (empty memory) is 0.5% on Lithuania 329×381, 2.6% on Homogeneous 400×400 and 31.9% on
     Kattegat 600×1000 (final review, `target_success.py`; CHANGELOG).
   - Slow tier: 11 passed.
   - The dispersal-off N excess (non-dispersal residual) was D15; it is now ported (Finding 4,
     Step 5: accepted, one borderline intake bin of seed 43 at +3.3%). Baselines are stale.
4. **Then regenerate the Kattegat baselines and record the behaviour change** in the CHANGELOG.
   Any paper text or figures built on CENOP real-landscape populations need re-running.

## Reproduce

Probe scripts are in `scripts/investigations/2026-11-10-phantom-food/`. They are investigation
records: some paths are absolute to the author's workstation, so adjust them before reuse. Run
them with the `shiny` env from the repo root.

**Which code produced which numbers:**
- Evidence §1–6, including every "stock" column in those sections, ran on the **pre-fix code**
  (`42ebd7f`, `CENOP-JASMINE`). The "CENOP stock" entries in the like-for-like and "Corrections
  combined" tables ran at `057d391` instead (phantom-food fix in, dispersal port not; see the
  item on those rows below). On the fix branch the stock kernel is the fixed one;
  `hypothesis_test.py` and `turning_angles.py … prefix` reinstate the pre-fix kernel in memory.
- Findings 2–3 ran on the fixed code (`eba75f9`).
- The like-for-like CENOP rows and the "Corrections combined" rows (including every
  `water+tdisp4[+target][+heading][+angle40][+pathlen]` mode) were produced at commit `057d391`, the
  fix branch before the dispersal port (it has the phantom-food fix `eba75f9` and communication off
  `864d0c9`, but not the dispersal fixes). Reproduce them with that version of the probe:
  `git show 057d391:scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py`. On HEAD
  the probe supports only `stock` and `nodisp`, and both run the fixed code.
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
# line-by-line follow-up: like-for-like rates (tick resolution) and corrections combined
python3 scripts/investigations/2026-11-10-phantom-food/depons_disp_targets.py <rp> 120
python3 scripts/investigations/2026-11-10-phantom-food/depons_disp_rates.py <rp> .depons-cache/landscape/Kattegat/bathy.asc 120
python3 scripts/investigations/2026-11-10-phantom-food/cenop_disp_rates.py <dir> stock|nodisp 180 120 [seed]   # HEAD: fixed code; the old modes need the 057d391 probe, see "Which code produced which numbers"
python3 -m validation.depons_java run --scenario scripts/investigations/2026-11-10-phantom-food/scenarios/dispoff.json --seeds 43 --out <dir>
```
