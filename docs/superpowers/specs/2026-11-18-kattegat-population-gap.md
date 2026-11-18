# Kattegat population gap: CENOP vs DEPONS-Java

Date: 2026-11-18. Branch: `investigate/kattegat-population-gap`. Status: cause found, not
fixed. Follows `2026-11-18-dispersal-idw-java-comparison.md`.

## Finding

CENOP starts a run with the wrong sex and pregnancy state, and that explains the gap.

| | DEPONS 3.2 | CENOP |
|---|---|---|
| Sex of the starting animals | all female (the model simulates females only) | half male (`population.py:599`, "Sex ratio 50%") |
| Who starts pregnant | every animal with age > 0, with probability h = 0.68, whatever its age (`Porpoise.java:165-178`) | mature females only (age >= 3.44), with probability 0.68 (`population.py:951-969`) |

In DEPONS about half of the 2000 starting animals are therefore pregnant. They give birth
around day 165 (gestation 300 days, mating day N(225, 20) counted from the previous year)
and then nurse for 240 days, using 1.4 times the energy. In CENOP roughly a tenth do. From
about day 150, DEPONS animals die faster at the same mean energy, and the survivors, all
female, keep paying for a calf in later years while half of CENOP's never do.

`docs/DEPONS-CENOP-PARITY-ANALYSIS.md` (row 9.6) records "Sex ratio: 50 % female, MATCH".
That misreads DEPONS: its 50 % is the chance that a weaned calf is a female and becomes an
agent (`Porpoise.java:1198-1205`), which CENOP does copy.

## Evidence

All runs: Kattegat, the DEPONS v3.2 landscape on both sides, 2000 animals, no ships, no
turbines. Java seeds 42 and 43; CENOP seeds 42 and 43.

**1. Where the gap opens (PSM-Type2).** Population and mean energy agree to day 150, then
Java loses animals faster. Births are negligible on both sides in year 1.

| Day | Java N | CENOP N | Java mean energy | CENOP mean energy |
|---|---|---|---|---|
| 120 | 1210, 1203 | 1203, 1193 | 5.11 | 5.22 |
| 150 | 792, 824 | 824, 817 | 4.70 | 4.71 |
| 180 | 528, 546 | 598, 587 | 4.54 | 4.60 |
| 240 | 234, 260 | 337, 335 | 5.94 | 5.25 |
| 360 | 161, 173 | 240, 237 | 10.97 | 10.54 |

Deaths per animal in days 150-180: Java 0.33 and 0.34, CENOP 0.27. Days 180-210: Java 0.37
and 0.37, CENOP 0.28. Deaths are spread over all ages alike on both sides. Per-year death
rates in years 2-5 average about 0.27 in Java and 0.20 in CENOP.

**2. The mortality formula and its parameters are the same** (read side by side:
`Porpoise.java:755-777`, `population.py:_check_mortality`; the Java run's
`batch_params.xml` against CENOP's defaults).

**3. Test.** CENOP rerun with only the starting state changed to DEPONS's rule (all female;
every animal with age > 0 pregnant with probability 0.68; days since mating
360 - round(N(225, 20))). Nothing in `src/` was changed; the script sets the state after
initialisation with its own random generator.

PSM-Type2, population:

| Day | Java | CENOP as shipped | CENOP with the DEPONS start |
|---|---|---|---|
| 150 | 792, 824 | 824, 817 | 816, 798 |
| 180 | 528, 546 | 598, 587 | 546, 540 |
| 240 | 234, 260 | 337, 335 | 241, 253 |
| 360 | 161, 173 | 240, 237 | 151, 172 |
| 720 | 113, 124 | 186, 169 | 119, 127 |
| 1080 | 89, 97 | 152, 149 | 103, 98 |
| 1440 | 75, 83 | 132, 127 | 85, 83 |
| 1800 | 51, 57 | 95, 108 | 49, 53 |

Year-1 deaths: Java 1839 and 1827; CENOP as shipped 1760 and 1763; with the DEPONS start
1849 and 1828.

InnerDanishWaters, population at the end of each year:

| Year | Java | CENOP as shipped (5 seeds) | CENOP with the DEPONS start |
|---|---|---|---|
| 1 | 97, 99 | 131-147 | 92, 82 |
| 2 | 85, 87 | 123-137 | 79, 62 |
| 3 | 75, 74 | 110-127 | 73, 57 |
| 4 | 70, 69 | 98-112 | 65, 53 |
| 5 | 53, 47 | 88-98 | 43, 36 |

- Under PSM-Type2 the gap closes at every checkpoint.
- Under InnerDanishWaters the gap closes and CENOP ends somewhat below Java (seed 43 more
  than seed 42). With two seeds a side this may be noise; it is not established either way.

## What this means

- Every CENOP run on every landscape starts with half its animals male and too few
  pregnant, in every release so far. Males never reproduce or lactate, so populations
  decline more slowly than DEPONS's and recruit less.
- The committed baselines (Kattegat and the slow validation tier) were produced with this
  start.

## Not done

- No fix. Changing the start changes default results on every landscape and needs the
  baselines regenerated, so it needs a decision.
- Open design question for the fix: CENOP records `is_female` and exports it. DEPONS has no
  males at all. Either drop males entirely (DEPONS parity) or keep a sex ratio as a
  JASMINE-only option with all-female as the DEPONS default.
- Other landscapes were not tested.
- The InnerDanishWaters residual needs more seeds on both sides.

## Reproduce

- CENOP on the v3.2 landscape: link `.depons-cache/landscape/Kattegat/*` into a scratch
  `<dir>/Kattegat/`, dropping `_XX` from the monthly prey and salinity file names.
- `python3 scripts/investigations/2026-11-18-population-gap/cenop_deaths.py <mode> <seed>
  1800 <dir> [java-init]` writes deaths and daily numbers to `output/population_gap/`.
- Java reference: `python3 -m validation.depons_java run --scenario
  validation/depons_java/scenarios/kattegat-{psm2,idw}-undisturbed-v1.json`.
