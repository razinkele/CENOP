# Landscapes on a 400 m grid

Date: 2026-11-19. Branch: `feat/landscapes-400m`. Status: approved by the user 2026-11-19 (A1), implemented.

## Why

DEPONS is calibrated for 400 m cells, and CENOP assumes 400 m in about 40 places: turbine and
ship noise distances, the random-walk step, dispersal distances, spatial memory,
communication range, swimming speed and the per-cell food amounts. Two bundled landscapes
are on other grids:

| Landscape | Cell size | Effect today |
|---|---|---|
| Lithuania (the default) | 1000 m | every distance is treated as 0.4 of its real length: noise sources 2.5 times closer, animals covering 2.5 times the intended distance per tick |
| CentralBaltic | 250 m | the opposite: distances treated as 1.6 times their real length |

Kattegat, DanTysk, Gemini and NorthSea are 400 m and are not touched.

## Goal

Every landscape the model runs on has 400 m cells, so the model code keeps one unit.

Not goals: changing any model code that uses the 400 m assumption; replacing the
placeholder food, salinity or sediment layers of Lithuania and CentralBaltic with real data.

Must not happen: a change to any 400 m landscape (Kattegat baselines and state hashes stay
bit-identical); a landscape with another cell size running without being resampled.

## Approaches considered

- **A. Resample the landscapes to 400 m** (chosen by the user, 2026-11-19). Model code
  unchanged, food and patch meaning stay DEPONS's.
- B. Make cell size a model quantity in all 40 places and four backends. Rejected: DEPONS
  defines no scaling of food per cell with cell area, so this needs a new modelling
  assumption.
- C. Fix only the noise distances. Rejected: movement stays 2.5 times off on the default
  landscape.

Within A, two ways to do it:

- **A1. Resample when the landscape is loaded (proposed).** `LandscapeLoader.load_all()`
  returns 400 m arrays and metadata whenever the files have another cell size. The files in
  `data/` stay as they are. No growth of the repository, the original data stays the source,
  and any landscape added later is handled the same way.
- A2. Regenerate the files in `data/` with a script. Lithuania grows from 20 MB to about
  125 MB in git, and the resampling rule lives in a one-off script instead of the loader.

## Design (A1)

One function, `resample_to_400m(data: dict) -> dict`, in a new module
`src/cenop/landscape/resample.py`, called at the end of `LandscapeLoader.load_all()` when
`metadata.cellsize != 400`.

- **Grid.** `ncols`, `nrows` = old size times `cellsize / 400`, rounded; the lower-left
  corner is unchanged; `cellsize` becomes 400. Lithuania 329 x 381 becomes 822 x 952;
  CentralBaltic 1800 x 1840 becomes 1125 x 1150.
- **Every layer except distance to coast: nearest neighbour.** Each new cell takes the value
  of the old cell its centre falls in. This keeps the coastline, the land/water mask, block
  and patch identities and NODATA exactly consistent between layers, and adds no invented
  values. Depth is not interpolated: a 1 km depth grid holds no information at 400 m.
- **Distance to coast: recomputed** from the resampled water mask, in metres (Euclidean
  distance to the nearest non-water cell, times 400). Lithuania's current layer is a
  placeholder plane, not a distance; CentralBaltic's is recomputed too, so both follow one
  rule.
- **Logged** once per load: old and new cell size and grid.
- Turbine files, ship routes and the map use projected coordinates and the landscape
  metadata, so they follow without change.

### Food on the resampled grids (measured)

The Lithuania and CentralBaltic food layers are placeholders with food on every water cell
(Kattegat has food patches on 1.6 % of cells). Trial, Lithuania, 2000 animals, seed 42, 120
days:

| Variant | Mean energy, day 120 | N |
|---|---|---|
| as today (1 km) | 19.92 | 2000 |
| 400 m, nearest neighbour | 19.93 | 2000 |
| 400 m, food scaled by 0.16 to keep food per km2 | 19.93 | 2000 |

Food is not limiting in any variant, so the choice does not matter here; nearest neighbour is
used. Lithuania and CentralBaltic remain landscapes without food limitation until their food
layers are replaced with real data. That is outside this change.

## What changes for users

- Lithuania and CentralBaltic results change: noise reaches 2.5 times less far on Lithuania
  (1.6 times farther on CentralBaltic), animals move the intended distance per tick, and
  dispersal targets become reachable.
- The Lithuania grid is 6.25 times larger; loading is slower and the map layers bigger.
- 400 m landscapes: no change.

## Tests

- Unit tests of `resample_to_400m` on small synthetic grids: sizes, corner, nearest-value
  rule, NODATA and land mask consistency across layers, monthly stacks, recomputed distance
  to coast, 400 m input returned untouched.
- Loading Lithuania gives `cellsize == 400` and the expected grid; water area in km2 within
  1 % of the source.
- Kattegat arrays identical to before; the four state hashes unchanged.
- A 10-day Lithuania run with a turbine: finite, in bounds, and the distance at which
  deterrence is felt matches the 400 m calculation.

## Left to the implementer

Naming inside the module, log wording, and which existing Lithuania tests need their grid
sizes updated.

## Result (2026-11-19)

Lithuania with the Curonian Nord wind farm (35 turbines), 2000 animals, 10 days, seed 42:

| | 1 km grid (before) | 400 m grid |
|---|---|---|
| Grid | 329 x 381 | 822 x 952 |
| Farthest deterred animal from a turbine | 133 cells = 133 km | 132 cells = 53 km |
| Deterred animal-ticks | 144,357 | 28,286 |
| Load time / 10 days of simulation | 0.2 s / 5.1 s | 0.6 s / 5.5 s |

No existing test needed a change. The four Kattegat state hashes are unchanged.

### Checks after the first implementation (2026-11-19)

**Food at the default population size.** The trial in the design used 2000 animals, where
food never binds. At 10,000 animals (seed 42, 60 days) the variants differ:

| Variant | Mean energy, days 15 / 30 / 45 / 60 | N at day 60 | Dispersal attempts, found |
|---|---|---|---|
| 1 km grid (before) | 14.66 / 15.81 / 16.57 / 16.99 | 9996 | 112,426, 1,075 (1%) |
| 400 m, nearest neighbour (implemented) | 19.89 / 19.89 / 19.91 / 19.91 | 10000 | 7,573, 2,153 (28%) |
| 400 m, food scaled by 0.16 (food per km2 kept) | 15.61 / 17.89 / 19.10 / 19.32 | 9997 | 1,130, 360 (32%) |

Nearest neighbour multiplies food per km2 by 6.25, which removes the mild competition the
1 km grid showed. Scaling the food keeps it early on. Nobody starves in any variant. Which
rule to use is a modelling choice on placeholder data. **Decided by the user, 2026-11-19: keep
the value per cell (nearest neighbour).** DEPONS food is a per-cell quantity (an animal eats
from its cell and each cell regrows to its own maximum), the 1 km energy curve came from runs
with wrong step lengths and is no reference, scaling still leaves nobody starving, and a 0.16
factor would read as calibration. The real fix is a patchy or measured food layer.

**Memory.** The per-animal spatial memory is sized from the grid: 76 x 65 cells before,
190 x 164 after. Peak memory at 10,000 animals goes from about 1.0 GB to 2.9 GB; time per
tick stays at about 52 ms.

**CentralBaltic.** Coarsening 250 m to 400 m by nearest neighbour keeps the water area
(114,842 -> 114,832 km2) and the sea (112,128 -> 112,200 km2) but joins the lagoon bodies of
616 and 349 km2 into one of 1,146 km2. A land-wins rule loses 2% of the water and splits
lagoons (110,074 / 643 / 318 / 195 km2), so nearest neighbour is kept and the join is a
known issue.

**Map.** The depth bitmap layer for the 822 x 952 grid builds in 2.3 s (about 1 MB). The
app was not opened in a browser.
