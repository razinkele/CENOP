# InnerDanishWaters: CENOP compared with DEPONS-Java

Date: 2026-11-18. Branch: `feat/dispersal-idw`. Design:
`2026-11-18-dispersal-idw-design.md`.

## Summary

- **Where animals are: agrees.** On the same landscape as Java (v3.2), CENOP's share of
  animals per 40 km block matches Java's as closely as two Java seeds match each other
  (correlation 0.997, distance 0.034 against 0.045 between Java seeds).
- **The mode's effect on population: agrees.** InnerDanishWaters leaves 0.59 times as many
  animals as PSM-Type2 after year 1 in Java and 0.60 times in CENOP; after year 5 the
  ratios are 0.93 and 0.92.
- **Time spent dispersing: agrees within what two Java animals can show.**
- **Absolute population: does not agree, in either mode.** On the same landscape CENOP
  keeps about 1.5 times Java's animals after year 1 and 1.7-1.9 times after year 5, under
  InnerDanishWaters and under PSM-Type2 alike. This is an existing CENOP-DEPONS
  difference, not something the port introduced.
- **The landscape hypothesis is rejected.** Running CENOP on Java's landscape did not
  bring its InnerDanishWaters population down (88-98 against 87-107 on v3.0).

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

- **Different landscapes in the first comparison.** Java ran on the DEPONS v3.2 Kattegat
  landscape; CENOP's `data/Kattegat` is the v3.0 landscape (prey, salinity, coast distance
  and patches differ; the block grid is identical). The sections up to "Time spent
  dispersing" carry that confound; "Same landscape on both sides" removes it.
- **Two Java seeds.** "Inside the Java spread" means inside the range of two runs.
- **Different random streams.** Only distributions can be compared.
- **Java tracks one animal per run** for dispersal phase, and in both seeds that animal
  died around day 120.

## Population at the end of each year (CENOP on v3.0)

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

## Where the animals are (CENOP on v3.0)

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

## Same landscape on both sides (added after the first comparison)

The first comparison above ran CENOP on its own `data/Kattegat` (v3.0). To test whether the
landscape explained the population gap, CENOP was rerun on the v3.2 landscape Java used:
the files from the harness cache linked under the names CENOP's loader reads
(`prey0000_MM_XX.asc` -> `prey0000_MM.asc`, same for salinity). Seeds as before;
`compare_with_java.py v32`.

Population at the end of each year, CENOP on v3.2:

| | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| Java IDW, seeds 42, 43 | 97, 99 | 85, 87 | 75, 74 | 70, 69 | 53, 47 |
| CENOP IDW, seeds 42-46 (range) | 131-147 | 123-137 | 110-127 | 98-112 | 88-98 |
| Java PSM-Type2, seeds 42, 43 | 161, 173 | 113, 124 | 89, 97 | 75, 83 | 51, 57 |
| CENOP PSM-Type2, seeds 42, 43 | 240, 237 | 186, 169 | 152, 149 | 132, 127 | 95, 108 |

| Ratio of means | Year 1 | Year 5 |
|---|---|---|
| CENOP / Java, InnerDanishWaters | 1.45 | 1.86 |
| CENOP / Java, PSM-Type2 | 1.43 | 1.88 |
| IDW / PSM-Type2, Java | 0.59 | 0.93 |
| IDW / PSM-Type2, CENOP | 0.60 | 0.92 |

Share of animals per block, days 30-360, CENOP on v3.2:

| Pair | Total variation distance | Correlation |
|---|---|---|
| Java IDW vs CENOP IDW | 0.034 | 0.997 |
| Java PSM-Type2 vs CENOP PSM-Type2 | 0.033 | 0.995 |
| Java IDW vs Java PSM-Type2 | 0.416 | 0.404 |
| CENOP IDW vs CENOP PSM-Type2 | 0.389 | 0.461 |
| Per-block shift IDW minus PSM-Type2, Java vs CENOP | | 0.993 |

What this shows:
- **The landscape was not the cause of the population gap.** CENOP's InnerDanishWaters
  population is the same on either landscape.
- **The gap is not specific to this mode.** On the same landscape CENOP is above Java by
  the same factor under PSM-Type2. The PSM-Type2 "agreement" in the first table (55
  against 51-57) came from comparing across two different landscapes and was a coincidence.
- **The port moves animals as Java does.** Block shares agree to within Java's own
  seed-to-seed spread, and the change the mode makes relative to PSM-Type2 is the same in
  both models, in location and in relative population.
- **The landscape did explain the location differences** in the first comparison
  (correlation 0.96 there, 0.997 here; PSM-Type2 0.61 there, 0.995 here).

## The remaining population difference

CENOP keeps more animals than DEPONS on Kattegat in both dispersal modes: about 1.45 times
after the first year, about 1.9 times after five. The first-year collapse is slightly
shallower in CENOP and the later decline is slower (about 10 % a year against about 15 % under InnerDanishWaters).
This belongs to the existing CENOP-DEPONS energy and mortality differences, outside this
port. It needs its own investigation; the like-for-like setup used here (CENOP on the v3.2
landscape, two modes, per-block and per-year numbers) is the place to start.

Not tested: the first-120-day dispersal statistics were not rerun on the v3.2 landscape.

## Update (2026-11-18, later)

The population difference was traced to the starting sex and pregnancy state and fixed
(`668b5bf`, `596eec6`, `1a0efbc`). On the v3.2 landscape CENOP now ends year 5 at 44-60
(PSM-Type2) and 40-48 (InnerDanishWaters) against Java's 51-57 and 47-53. See
`2026-11-18-kattegat-population-gap.md`. The population statements above and in the
conclusion below describe the code before that fix.

## Update (2026-11-19): final code on the bundled v3.2 landscape

`data/Kattegat` is now the v3.2 landscape, and the calf memory, the memory-update order
when dispersing (D16) and the edge heading are fixed (`1cfa8b0`). `run_cenop.py <mode> <seed>
5 data final`, then `compare_with_java.py final`.

| | Year 1 | Year 2 | Year 3 | Year 4 | Year 5 |
|---|---|---|---|---|---|
| Java IDW, seeds 42, 43 | 97, 99 | 85, 87 | 75, 74 | 70, 69 | 53, 47 |
| CENOP IDW, seeds 42-44 | 96, 91, 93 | 80, 79, 74 | 75, 76, 61 | 65, 69, 55 | 36, 47, 36 |
| Java PSM-Type2, seeds 42, 43 | 161, 173 | 113, 124 | 89, 97 | 75, 83 | 51, 57 |
| CENOP PSM-Type2, seeds 42-44 | 176, 164, 156 | 131, 121, 107 | 109, 99, 93 | 96, 79, 83 | 60, 55, 52 |

Share of animals per block, days 30-360: Java IDW vs CENOP IDW distance 0.028, correlation
0.998; Java PSM-Type2 vs CENOP PSM-Type2 distance 0.034, correlation 0.994.

**A small InnerDanishWaters offset.** Six CENOP seeds (42-47), two years, InnerDanishWaters:

| Code | Year 1 (mean, SD) | Year 2 (mean, SD) |
|---|---|---|
| Before the edge-heading fix (`1deb759` on v3.2 data) | 92.8, 11.1 | 78.7, 7.9 |
| With it (`35a5359`) | 88.3, 7.5 | 75.7, 5.2 |
| With D16 as well (`1cfa8b0`) | 91.5, 3.3 | 77.7, 2.9 |
| Java, seeds 42 and 43 | 97, 99 | 85, 87 |

The three code states do not differ beyond seed spread, so the fixes did not move the mode
relative to Java. All three sit about 7% below Java after year 1 and 9% after year 2. The
three seeds used in the tables above (42-44) happened to be the high ones before the fixes,
which hid this. Cause not investigated; PSM-Type2 shows no such offset.

## Conclusion

On the same landscape the port reproduces where DEPONS sends animals, to within Java's own
seed-to-seed variation, and reproduces the mode's effect on population relative to
PSM-Type2. CENOP's absolute Kattegat population is higher than DEPONS's by the same factor
in both modes, which is an existing difference in the shared energy and mortality code.
CENOP Kattegat population numbers, in any dispersal mode, should not yet be presented as
matching DEPONS.
