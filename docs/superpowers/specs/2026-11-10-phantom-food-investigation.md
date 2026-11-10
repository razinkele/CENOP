# Phantom food in CENOP's vectorised eating — investigation (2026-11-10)

**Status:** root cause of the main discrepancy confirmed; **not fixed**. A second, smaller
discrepancy remains open. The fix changes released (v2.2.0) model behaviour and invalidates the
committed Kattegat baselines, so it needs a decision first (see "Decision needed").

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
| JAX `jax_eat_food` | **yes**: `jnp.maximum(food_grid - eaten, min_food)` over the whole grid | `22eadee` (2026-03-20) | opt-in JAX backend (**unverified** at runtime) |
| Cython post-CRW | yes (unconditional floor) | — | gated to homogeneous landscapes, where every cell has food, so it has no effect there |
| scalar `CellData.eat_food` | no (returns early for empty cells) | — | legacy scalar path |

Consequences:
- **Affected:** every CENOP run on a real (patchy) landscape with the default backend, including
  the committed Kattegat baselines (`output/kattegat_ref*`) and v2.2.0.
- **Not affected:** the 11 slow-tier "demographic validation" tests. They all use
  `create_homogeneous_landscape`, which gives every cell, land included, a positive food value,
  so the floor never creates food. They therefore validate demography only under food in every
  cell, not foraging on a patchy landscape.

## Second discrepancy (open)

With phantom food removed, CENOP still collapses much faster than DEPONS:
- N=450 against 1,953 at day 60;
- patch intake of 0.0020 against 0.0039 per animal per tick on day 0.

What `occupancy_probe.py` and `movement_compare.py` show:
- **Step length and cell changes match.** DEPONS tracked animal: mean 852 m, median 782 m,
  changes cell on 99.1% of steps. CENOP: 829 m, 774 m, 99.4%.
- **CENOP animals are on patch cells at about chance rate** (1.2–1.9% of animal-ticks; patch
  cells are 1.6% of water), with or without the fix.
- **With the fix, the patches CENOP animals occupy hold only 0.011–0.015 food**, essentially
  the floor. Total patch food stays near full (about 1,700). So animals keep returning to
  emptied patches while most patches go unvisited.
- DEPONS's intake implies about 0.24 food per on-patch step, if its on-patch rate is similar.
  This is **inferred**: DEPONS writes no occupancy output.

Suspects, in order (none tested yet):
1. food-directed movement: reference memory, stored utility and the `veTotal` attraction term;
2. regrowth recovery of cells floored at 0.01. With Umin=0.001, a cell at exactly 0.01 does not
   trigger the 47 extra iterations, so recovery depends on K;
3. the timing of eating relative to the move.

## Unverified

- JAX path runtime behaviour (inferred from code only).
- DEPONS on-patch rate (not output by DEPONS).
- DEPONS run on the v3.0 landscape (not done). Not needed for the landscape-versus-model
  verdict, because v3.2 has more food, but it would complete the 2×2 comparison.
- The second discrepancy's cause.
- Whether the effect differs at N=200 (TRACE scale). DEPONS 1-yr N=200 smoke runs also halved.

## Decision needed

1. **Fix the floor** in Numba v1/v2, JAX and Cython, to match DEPONS (floor only cells that had
   food).
   - Failing test first: after stepping on a landscape with empty cells, food on every
     `food_prob == 0` cell is exactly 0, for each backend.
   - This changes v2.2.0 behaviour. CENOP Kattegat populations will collapse, faster than DEPONS
     until the second gap is resolved.
2. **Investigate the second discrepancy** before regenerating baselines or making any parity
   claim.
3. **Regenerate the Kattegat baselines and record the change** in CHANGELOG Known issues. Any
   paper text or figures built on CENOP real-landscape populations need re-running.

## Reproduce

Probe scripts are in `scripts/investigations/2026-11-10-phantom-food/`. They are investigation
records: paths are absolute to the author's workstation, so adjust them before reuse. Run them
with the `shiny` env from the repo root.

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
```
