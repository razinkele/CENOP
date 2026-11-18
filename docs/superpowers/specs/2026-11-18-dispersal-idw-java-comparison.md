# InnerDanishWaters: CENOP compared with DEPONS-Java

Date: 2026-11-18. Branch: `feat/dispersal-idw`. Design:
`2026-11-18-dispersal-idw-design.md`.

## Summary

- **Where animals are: agrees.** The share of animals per 40 km block in CENOP follows
  Java closely (correlation 0.96), much more closely than under PSM-Type2 (0.61).
- **Time spent dispersing: agrees within what two Java animals can show.**
- **Population: does not agree.** After 5 years CENOP keeps 87-107 animals; Java keeps
  47-53. Under PSM-Type2 the two models end at the same level (55 vs 51-57). The cause is
  not established.

## Setup

- Scenario `kattegat-idw-undisturbed-v1`: Kattegat, 2000 animals, 5 years, no ships, no
  turbines, `dispersal=InnerDanishWaters`.
- Java: DEPONS 3.2 release through the reference harness, seeds 42 and 43 (run
  2026-11-18). `batch_params.xml` confirms `InnerDanishWaters` and `Kattegat`.
- CENOP: `feat/dispersal-idw`, seeds 42-46, plus PSM-Type2 seeds 42-43 for scale.
  Existing Java PSM-Type2 runs (seeds 42, 43) are used for the same purpose.
- Scripts: `scripts/investigations/2026-11-18-dispersal-idw/` (`run_cenop.py`,
  `compare_with_java.py`, `episodes.py`).

## Limits

- **Different landscapes.** Java ran on the DEPONS v3.2 Kattegat landscape; CENOP's
  `data/Kattegat` is the v3.0 landscape (prey, salinity, coast distance and patches differ;
  the block grid is identical). This confounds every number below.
- **Two Java seeds.** "Inside the Java spread" means inside the range of two runs.
- **Different random streams.** Only distributions can be compared.
- **Java tracks one animal per run** for dispersal phase, and in both seeds that animal
  died around day 120.

## Population at the end of each year

| | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| Java IDW, seed 42 | 97 | 85 | 75 | 70 | 53 |
| Java IDW, seed 43 | 99 | 87 | 74 | 69 | 47 |
| CENOP IDW, seeds 42-46 (range) | 121-140 | 108-134 | 101-124 | 97-115 | 87-107 |
| Java PSM-Type2, seeds 42, 43 | 161, 173 | 113, 124 | 89, 97 | 75, 83 | 51, 57 |
| CENOP PSM-Type2, seeds 42, 43 | 184, 159 | 126, 104 | 91, 81 | 67, 67 | 55, 55 |

- Both models collapse from 2000 to about 5 % within the first year in both modes.
- CENOP IDW is above the Java IDW range in every year, by a quarter in year 1 and by about
  a factor of two in year 5.
- Both models agree on the direction of the mode's effect in year 1: fewer survivors under
  InnerDanishWaters than under PSM-Type2 (Java 98 vs 167, CENOP 127 vs 172, means).
- After year 1 CENOP IDW declines more slowly than Java IDW (about 5 % a year against
  about 14 %).

## Where the animals are

Mean share of animals per block over days 30-360, compared across all 60 blocks:

| Pair | Total variation distance | Correlation |
|---|---|---|
| Java IDW vs CENOP IDW | 0.130 | 0.959 |
| Java PSM-Type2 vs CENOP PSM-Type2 | 0.377 | 0.607 |
| Java IDW vs Java PSM-Type2 | 0.416 | 0.404 |
| CENOP IDW vs CENOP PSM-Type2 | 0.247 | 0.838 |
| Two Java IDW seeds | 0.045 | |
| Two CENOP IDW seeds | 0.034 | |

- The two models put animals in the same blocks under InnerDanishWaters. The three most
  populated blocks are the same, with similar shares (block index 38: Java 0.108, CENOP
  0.123; index 32: 0.094 and 0.104; index 44: 0.063 and 0.053).
- The remaining distance (0.130) is about three times the distance between two seeds of the
  same model, so the distributions are close but not the same.
- The change the mode makes relative to PSM-Type2 points the same way in both models
  (correlation of the per-block shift 0.75). It is smaller in CENOP because CENOP's
  PSM-Type2 distribution already sits nearer the InnerDanishWaters one.

This is the most direct evidence that target selection, the geographic exceptions and the
coordinate conversion are right: an error of one block or half a cell in any of them would
move animals to different blocks.

## Time spent dispersing

Per animal-tick over the first 120 days (CENOP: all animals; Java: the one tracked animal
while alive):

| | Phase 1 share | Phase 2 share | Starts per animal-day | Mean episode (ticks) |
|---|---|---|---|---|
| Java seed 42 | 0.098 | 0.094 | 0.21 | 45 |
| Java seed 43 | 0.065 | 0.172 | 0.31 | 37 |
| CENOP seed 42 | 0.101 | 0.186 | 0.39 | 36 |
| CENOP seed 43 | 0.102 | 0.183 | 0.39 | 35 |

- Episode length and phase shares are of the same size. CENOP starts dispersal somewhat
  more often than either Java animal.
- With one Java animal per seed this cannot show more than "no gross difference".
- An earlier version of the comparison used CENOP's once-a-day snapshots and showed far
  more dispersal in CENOP. Those snapshots are taken right after the daily trigger and
  overstate it; they are not used here.

## The population difference

What is known:
- It is specific to this mode: under PSM-Type2 the two models end within each other's range.
- It is present by the end of year 1 and grows afterwards.
- Spatial distribution and dispersal behaviour, the things the port controls directly,
  agree.

Candidate causes, none tested:
1. **Landscape.** InnerDanishWaters concentrates animals in a few blocks (a tenth of the
   population in one). Survival then depends on the food in those blocks, which differs
   between the v3.0 and v3.2 prey maps. PSM-Type2 spreads animals more widely and would be
   less sensitive to this.
2. **A difference in the port that changes energy but not location**, for example what a
   dispersing animal eats along the way, or the accepted deviation in the eating cell
   after a stop with a move (see the changelog's known issues).
3. **Existing CENOP-DEPONS differences in energy and mortality** that matter more when
   animals disperse more often.

What would separate them: run Java on the v3.0 landscape, or CENOP on v3.2 (its loader
cannot read the v3.2 files as shipped). If the models then agree, it is cause 1. If not,
compare the energy of dispersing animals tick by tick.

## Conclusion

The port reproduces where DEPONS sends animals and roughly how long they disperse.
Population size under this mode is about twice Java's after five years and is not explained.
Until it is, CENOP InnerDanishWaters runs should not be presented as matching canonical
DEPONS Kattegat population results.
