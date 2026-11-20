# The JASMINE social weight (2026-11-20)

Script: `scripts/investigations/2026-11-20-social-vector/curve.py`. Applies only when
communication is switched on; it is off by default.

## What changed

An animal's heading is the direction of the sum of a random-walk term, the pull of
remembered food and any deterrence. The random-walk term has length
`inertia_const + presMov * veTotal` (DEPONS's `crwContrib`); it is small when little food has
been found. The social pull was added as `social_weight * step length`, a quantity of a
different kind, so where food was scarce it set the heading on its own: on Kattegat it was
longer than the random-walk term on 57% of animal-ticks, and headings reversed five times as
often as without communication.

The social pull in the heading is now `social_weight * (inertia_const + presMov * veTotal)`:
the weight is the social attraction as a fraction of the random walk's own pull, the same
everywhere. A weight below 1 never outweighs the random walk. The JASMINE physics and hybrid
movement modules add the pull to the displacement, not the heading, and keep the step-length
scale.

## Effect of the weight

Kattegat, 2000 animals, seed 42, `InnerDanishWaters`, no ships, days 10-20.

| Social weight | Median net movement per day | Heading reversals (> 150 degrees) | Mean distance to nearest animal | Animals with another within 1 km |
|---|---|---|---|---|
| communication off | 11.4 km | 4.4% | 2.22 km | 21% |
| 0.1 | 11.2 km | 4.3% | 2.05 km | 27% |
| **0.3 (default)** | 11.0 km | 4.6% | 1.68 km | 41% |
| 1 | 5.8 km | 11.8% | 0.89 km | 79% |
| 2 | 4.3 km | 28.1% | 0.80 km | 83% |
| 5 | 2.3 km | 45.3% | 0.78 km | 84% |
| 0.3, before this change | 8.7 km | 20.2% | 1.45 km | 58% |

At the default weight animals end up closer together than without communication, while
movement and turning stay close to DEPONS's. Above about 1 the social pull dominates:
animals bunch, movement falls and reversals rise. The UI slider stops at 1.

## Checked against observed group sizes (2026-11-20)

**Target.** Aerial surveys in the Belt Sea and the German North Sea count about 1.16 adults per
group, and 82-88% of groups are single animals (summer; both figures derived from published
counts with calves removed: Unger et al. 2021, MiniSCANS-II, same counts in Owen et al. 2024,
Suppl. Table 1; Gilles et al. 2009). Survey protocols do not state a distance rule; the ferry
and shore studies that do use "within 100 m" (Bouveroux et al. 2020; Uluduz et al. 2020). The
model's agents exclude calves, so the adult figures are the right comparison. Aerial counts
miss submerged animals (g(0) about 0.3-0.4; Gilles et al. 2023), so true grouping is somewhat
higher.

**Model.** Kattegat, 2000 animals, days 10-30; groups are animals linked within 100 m (0.25
cells) at the end of each day.

| Social weight | Mean group size | Single-animal groups | Median net movement per day | Heading reversals |
|---|---|---|---|---|
| communication off | 1.002 | 99.8% | 11.6 km | 4.4% |
| 0.1 | 1.002 | 99.8% | 11.4 km | 4.4% |
| **0.3 (default)** | 1.005 | 99.5% | 11.2 km | 4.7% |
| 0.5 | 1.010 | 99.1% | 9.8 km | 5.4% |
| 1 | 1.069 | 94.4% | 5.9 km | 11.7% |
| 2 | 1.225 | 88.6% | 3.9 km | 29.0% |
| 5 | 1.283 | 87.9% | 2.1 km | 45.7% |

**Result.**

- The survey group size is reached only near weight 2, where animals cover a third of the
  distance per day and reverse their heading six times as often as DEPONS's movement, which
  was fitted to tracked porpoises. The two kinds of data cannot both be met: attraction to
  calls heard up to 10 km away draws animals into loose aggregations at the kilometre scale
  but makes few groups at the 100 m scale. Observed groups would need a different mechanism
  (short-range cohesion, mother-calf bonds) or a model density closer to the real one: 2000
  agents on Kattegat is about 0.04 animals per km2 against 0.25-0.66 in the SCANS-IV Belt Sea
  blocks (Gilles et al. 2023).
- The movement calibration sets the upper limit. Weights up to 0.3 keep movement within 4%
  and reversals within 0.3 percentage points of communication off; 0.5 already loses 16% of
  the daily movement. The default 0.3 is the largest weight that leaves DEPONS's movement
  intact, which is its justification, not a fit to group sizes.
- The survey data only bound the weight from above (a weight that made groups larger than
  about 1.3 or singles rarer than about 75% would be too strong); no weight up to 5 does.

Literature search and figures: a scite-verified review (all DOIs checked, no retractions).
Baltic Proper: no robust group-size estimates exist (Amundin et al. 2022).

References

- Amundin, M., et al. (2022). *Ecology and Evolution*. https://doi.org/10.1002/ece3.8554
- Bouveroux, T., et al. (2020). *J. Mar. Biol. Assoc. UK*. https://doi.org/10.1017/s0025315420000326
- Gilles, A., Scheidat, M., & Siebert, U. (2009). *Mar. Ecol. Prog. Ser.* 383, 295-307. https://doi.org/10.3354/meps08020
- Gilles, A., et al. (2023). SCANS-IV final report. https://doi.org/10.13140/rg.2.2.34873.95845
- Owen, K., et al. (2024). *Frontiers in Marine Science*. https://doi.org/10.3389/fmars.2024.1289808
- Uluduz, N., et al. (2020). *Scientia Marina*. https://doi.org/10.3989/scimar.05086.21a
- Unger, B., et al. (2021). MiniSCANS-II final report. https://dce.au.dk/fileadmin/dce.au.dk/Udgivelser/Eksterne_udgivelser/20210913_Report_MiniSCANSII_2020_revised.pdf

## Remaining uncertainty

The weight is bounded by movement data, not fitted to grouping data, and the social mechanism
does not reproduce observed group sizes at any weight that keeps the calibrated movement.
Results with communication on should be read as a sensitivity analysis, not a prediction.
