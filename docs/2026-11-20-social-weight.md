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

## Not calibrated

No weight has been fitted to observations. Harbour porpoises are mostly seen alone or in
small groups; a calibration would need a measure of grouping that can be compared with survey
data, and a cited source for it. Until then, results with communication on should be read as
a sensitivity analysis, not a prediction.
