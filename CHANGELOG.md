# Changelog

All notable changes to CENOP-JASMINE are documented in this file.

## [Unreleased]

### Added
- **`build_simulation(params, landscape=None)`** in `cenop.server.simulation_controller`: the
  Simulation with the subsystem modules the parameters select, as the app builds it.
  `Simulation(params)` alone runs DEPONS without modules, so `simulation_mode="JASMINE"` set
  in code did nothing; the README example now uses `build_simulation`.
- **CentralBaltic rebuilt from public data (it was misplaced).** The old landscape, a
  placeholder since the first commit, carried a grid origin about 400 km south of its data:
  its download script stored latitude-longitude depths under an EPSG:3035 header without
  reprojecting, so over a million "water" cells lay over inland Poland. The new landscape
  covers the Baltic Proper from Bornholm to north of Gotland (13-22 E, 53.9-59.5 N) on
  615 x 677 cells of 1 km: EMODnet depths (checked at known places and against Lithuania:
  median difference 0.2 m), distance to coast, 40 km blocks, Copernicus monthly salinity
  for 2021 (the product DEPONS used),
  EMODnet Geology grain size, random food patches (prey uniform for now, see below) and
  10,245 ship passages from
  2024 AIS. 117 MB instead of 842 MB. Results on CentralBaltic change entirely. The scripts
  that built the old grid are removed; `scripts/build_baltic_bathymetry.py` and
  `scripts/build_baltic_environment.py` replace them.
- **Lithuania: sea gaps filled.** 12.4% of the sea in the Lithuania box (10,409 km2: a strip
  along the northern edge, parts of the Gulf of Gdansk and of the Curonian Lagoon, scattered
  holes) had no depth and acted as land, an artificial shore across open water. Filled from
  the same EMODnet DTM the landscape's depths came from, keeping every earlier water cell; the
  lagoon stays separate. Blocks are now 40 km squares (previously one block; only the map
  editor shows them, InnerDanishWaters dispersal is Kattegat-only). Lithuania results change.
- **CentralBaltic: 13 near-zero depth cells set to land.** Depths under 5 mm were written as
  0.00; the model read them as land, but distance to coast counted them as water. Distance to
  coast recomputed (76 cells change, by up to 592 m); the water mask is unchanged.
- **Lithuania: real salinity and sediment.** Its salinity was random around 7.5 and its
  sediment one value everywhere. Now Copernicus monthly surface salinity for 2021 (the
  product DEPONS used; up to 7.9, near-fresh in the lagoon) and EMODnet Geology grain size,
  as for CentralBaltic. Salinity enters the random walk, so Lithuania
  results change.
- **Lithuania: food patches and ship traffic from public data.** Food: DEPONS-style random
  patches (1.6% of the water, single 400 m cells) in place of food on every water cell; the
  prey level is uniform (the DEPONS mean, 0.38592) until monthly maps from the SAMBAH porpoise
  distribution can be published (see Known issues). Ships: 3,469 passages over 15 days of
  August 2024 from public raw AIS (aisstream.io via figshare, CC BY 4.0), encoded like DEPONS's
  Kattegat file. Tools:
  `scripts/build_food_layers.py`, `scripts/fetch_ais_huetten.py`,
  `scripts/build_ships_from_ais.py`; the loader accepts patch and prey files on the 400 m grid
  of a landscape stored on another grid. Details: `docs/2026-11-20-baltic-public-data.md`.
- **Dark mode.** A switch in the navbar turns the light shell (navbar, sidebar) dark; the
  maps switch to a dark basemap. A basemap the user hid in the legend stays hidden.
- **Share settings.** A navbar button gives a link that reopens CENOP with the current
  settings (`App(bookmark_store="url")`). Results are not included, buttons are left out so
  opening a link never starts a run, and the colour mode stays the viewer's own.
- **Map redraw after a reconnect.** If the hosting server keeps the session alive while the
  browser reconnects (Shiny >= 1.8, `session.allow_reconnect`), both maps are redrawn from
  what the server last sent. Whether the production Shiny Server keeps sessions alive has
  not been checked; a plain `shiny run` does not.

### Removed
- **Settings the model never read:** Tracked Porpoises (`tracked_porpoise_count`; DEPONS
  uses it for a per-animal report that CENOP never ported, now recorded in the parity doc)
  and the age-mortality rates `mortality_juvenile`, `mortality_adult`, `mortality_elderly`
  (not in DEPONS 3.2; mortality is starvation, bycatch and maximum age). A population test
  that added the unused 5% adult rate now uses `bycatch_prob`.
- **`habituation_enabled` and the habituation constants.** Habituation was listed as a JASMINE
  feature but never implemented: the setting and `HABITUATION_RATE` / `MIN_RESPONSE` were
  never read. Code that passes `habituation_enabled` to `SimulationParameters` now fails. If
  habituation is wanted, a published target is the decline of porpoise responses over one
  construction, from a 50% response distance of 7.4 km at the first pile location to 1.3 km
  at the last (Graham et al. 2019, doi:10.1098/rsos.190335).
- **The JAX backend.** Deprecated in 2.3.0, now removed: its kernels, the JAX branch of the
  tick, its tests and `jax[cpu]` in the CI environment. `use_jax=True` raises a `ValueError`
  with an explanation. The default NumPy/Numba backend was faster on real landscapes and the
  only one compared with DEPONS. The code is in git history up to `v2.3.0`.

### Changed
- **Learned avoidance calibrated to observed recovery after pile driving.** The JASMINE memory
  stored the raw deterrence strength (pile driving reaches about 60) and faded by 0.1% per
  tick, so a turbine memory stayed above the 0.1 avoidance threshold for about two months,
  within 40 km. It now stores a response capped at 1, fades by 1.6% per tick (a full memory
  drops below the threshold in about 3 days) and acts within 5 km (`avoidance_radius` 2.5
  memory cells). Targets: porpoise acoustic activity stayed below normal for 24-72 h at 2.6 km
  after piling (Brandt et al. 2011, doi:10.3354/meps08888); 50% response distances of 1.3-7.4
  km (Graham et al. 2019, doi:10.1098/rsos.190335); and DEPONS's own deterrence, calibrated to
  post-piling recovery at the Gemini wind farm (TRACE 3.2), which the memory should not
  outweigh. Test farm on Kattegat (11 piles of 3 h every 2 days, 2000 animals, seeds 1-3),
  animals within 5 km as a ratio to the same run without turbines: 12-24 h after a pile 0.42
  (DEPONS mode 0.56, old memory 0.03); 1-3 days after the last pile 0.48 (0.53, 0.03); 7-30
  days after 0.95 (0.91, 0.30). Radii of 10 and 20 km kept numbers 30-50% lower for 1-3 days.
  Reproduce with `scripts/investigations/2026-11-21-learned-avoidance/recovery_study.py`.
- **Learned avoidance is a fraction of the random walk's pull on the DEPONS walk.** The JASMINE
  memory's avoidance pull was added to the heading at a fixed size (up to 0.8). Where the
  random walk's weight (DEPONS crwContrib) was small, after an animal had found little food,
  the pull set the heading and drew animals off their food. It is now the avoidance strength
  times crwContrib, as the social pull. Kattegat, 500 animals, ships, 60 days, JASMINE mode:
  498 alive and mean energy 10.68 (was 438 and 7.47; 499 and 10.50 without the memory);
  deterred animal-ticks 14.7% (was 11.6%; 15.2% without). Of the levers tried, this was the
  one that mattered: remembering the animal's own cell instead gave 448 and 7.79, a 5-cell
  radius 465 and 9.36, a 2-cell radius 482 and 10.06. DEPONS mode is unchanged.
- **JASMINE mode now runs DEPONS movement and the DEPONS energy budget by default.** The JASMINE
  movement (a simplified walk; the physics integrator is not wired in) and the JASMINE energy
  budget are uncalibrated: with them, Kattegat (500 animals, seed 11, no ships) keeps 499
  animals at day 360 where DEPONS movement and energy keep 195. They now run only when chosen (`movement_mode="JASMINE"`, `energy_mode="JASMINE"`, or the Subsystem
  Mode Overrides in the Settings tab). JASMINE mode is DEPONS plus learned avoidance and
  behaviour states. Without disturbance JASMINE mode now gives the DEPONS population (195 at
  day 360); learned avoidance acts on the DEPONS walk (Kattegat with ships, 20 days: 68% of
  animals avoiding a remembered disturbance at any time on average). Habituation, listed as a JASMINE
  feature, is not implemented (its constants and `habituation_enabled` are not read); the
  README and help now say so. DEPONS mode is unchanged (app-style run byte-identical).
- **Movement on the Baltic landscapes uses a fixed salinity, as DEPONS does on Kattegat.** The
  DEPONS movement model (step length and turning angle depend on salinity) was fitted to
  porpoises tagged in the eastern North Sea, and DEPONS moves with a fixed 34.07 on Kattegat
  because the model is not valid at much lower salinity (TRACE 3.2 section 3.2.2). On
  Lithuania and CentralBaltic the map salinity (about 7) put most step-length draws near the
  maximum: steps of about 4 km per 30 minutes, five times Kattegat's. New parameter
  `movement_salinity` (Settings, Movement): "auto" (default) fixes it on Kattegat, Lithuania
  and CentralBaltic and uses the map elsewhere; "map" uses the map everywhere (DEPONS outside
  Kattegat); "fixed" uses 34.07 everywhere. Sound propagation always uses the map. Kattegat,
  North Sea and app-style DEPONS runs are byte-identical. 2000 animals, ships, 60 days:
  Lithuania mean step 10.1 -> 2.3 cells, deterred animal-ticks 3.5% -> 3.3%, mean energy at
  day 60 9.89 -> 9.30; CentralBaltic 10.1 -> 2.3 cells, 5.5% -> 5.1%, 11.41 -> 10.82.
  Lithuania and CentralBaltic results change.
- **The app runs 2.7-4.6 times faster with ships on the Baltic landscapes.** The app always
  runs the behaviour state machine (scripts and the batch runner do not), and every tick it
  counted each animal's spatial-memory cells by copying the animal's whole memory grid:
  100,000 cells per animal on CentralBaltic, for a dispersal rule that could never fire
  (its declining-days counter is never incremented). The count is gone: the state machine
  now follows the model's own dispersal (see Fixed). State fingerprints of app-style runs
  were unchanged by the speed-up (Kattegat 20 days and CentralBaltic 3 days, 300-500
  animals, ships).
  With 300 animals and ships the app went from 23 to 105 simulated days in 90 s on
  CentralBaltic (4.6x) and from 71 to 192 on Lithuania (2.7x). Scripts and the batch runner
  run without the state machine and were not affected.
- **JASMINE social attraction is scaled to the random walk (communication on only).** The
  social pull was added to the heading as weight x step length, which outweighed the random
  walk wherever little food had been found: on Kattegat headings reversed on 20% of ticks
  against 4.4% without communication. It is now weight x the random walk's own pull, so the
  weight means the same everywhere and a weight below 1 never outweighs the random walk.
  At the default weight 0.3, reversals are 4.6% and animals still group (41% have another
  within 1 km, against 21% without communication). The weight is not calibrated against
  observations. Runs with communication off (the default) are unchanged. Details:
  `docs/2026-11-20-social-weight.md`.
- **JASMINE physics and hybrid movement: dispersal ends by the same rule as the default
  movement.** These modules ended `PSM-Type2` dispersal once the straight-line distance from
  the start reached 95% of the target, because they did not track the path length. They now
  track it and end on path length, as the default movement and DEPONS do (`Undirected` still
  ends on straight-line distance, as in DEPONS). Only runs with these movement modules
  change; the default movement is untouched.
- **Runs with ships are about 2.6 times faster, with identical results.** A Kattegat tick
  with ship traffic takes about 34 ms where it took about 88 ms (2000 animals). The ship
  deterrence calculation handles all ships in one pass, finds the in-range pairs in
  compiled code, and no longer fully updates ships that are waiting at a buoy. The
  two-year ship baseline, rerun on the new code, is byte-identical and took 20 minutes
  (it took 45 to 55 before). Details:
  `docs/performance/2026-11-20-ship-loop.md`.

### Fixed
- **Map noise circles show the model's deterrence range, at the right times.** The circles
  used a 158 dB threshold with 20 log10(r) spreading instead of the model's 152 dB,
  `beta_hat`/`alpha_hat` transmission loss and maximum distance (234 dB pile: 48 km, was
  6.3 km), and operating turbines got 500 m circles although 145 dB never reaches the
  threshold. The map also compared the turbine file's `start`/`end` (days) with the tick, so
  piling showed only for the first few ticks, and files with `tick.start`/`tick.end` read as
  0. The map now reads turbines with the model's loader and uses its phase rule; the radius
  comes from `deterrence_radius_m` (new, in `cenop.agents.turbine`; on Kattegat the farthest
  deterred animal was 49.6 km and the nearest undeterred 47.7 km for a 48.3 km radius). The
  Turbine info panel uses the same radius and the model's defaults. The bathymetry legend now
  shows the map's viridis colours (was a blue ramp).
- **The Behavior FSM Mode override works.** `build_simulation` took the state machine's rules
  from the time mode and only logged `fsm_mode`, so the Settings select had no effect. It now
  passes the effective FSM mode (the override, else `simulation_mode`). Default runs are
  byte-identical in both modes; the tooltip now says what the two rule sets do.
- **Births and deaths over a run are counted once each.** `Simulation.total_births` and
  `total_deaths` (used by the batch runner) summed the daily history records, which hold
  month-to-date counts, so an event was counted once for every remaining day of its month (a
  run of 200 animals reported 118 births and 968 deaths while the population fell by 49). The
  app's runner counted the net daily population change, so births and deaths on the same day
  cancelled (11 births in the same run). Both now use run totals that the simulation keeps
  per step: births are weaned calves added to the population, deaths all removals, and births
  minus deaths equals the population change (slow-tier test). The month-to-date history and
  the output files are unchanged.
- **Batch replicates now get their own random seed.** `Simulation(params, seed=...)` used the
  seed only for ship reactions; the population drew from `params.random_seed`. `BatchRunner`
  seeds each replicate this way, so replicates with a fixed `random_seed` repeated the same
  run (without ships) and replicates without one could not be reproduced from their recorded
  seed. A seed argument now replaces `params.random_seed` on a copy of the parameters. Runs
  that set both to the same value (the Kattegat reference runs) and the app (which sets only
  `random_seed`) are unchanged.
- **Population tab: two charts showed something other than their titles.** "Landscape Energy
  Level" plotted food eaten per porpoise (titled "Average Porpoise Energy Level") and "Average
  Porpoise Movement" plotted the number of dispersing porpoises (already on the Disturbance
  tab). They now show the total food on the landscape (`CellData.total_food()`) and the mean
  straight-line distance moved per day by animals alive at both updates (from the second day;
  Kattegat, 300 animals: about 9-11.5 km per day). The runner's daily entry carries both
  (`landscape_food`, `movement_km_per_day`).
- **Documentation brought in line with the code.** `docs/API.md` documented classes, methods
  and parameters that do not exist (e.g. `EnergyBudget`, `JASMINEEnergy`, `DisturbanceMemory`,
  `SimulationConfig`, `run_batch`, `OutputWriter.finalize`) and wrong signatures and defaults;
  it now matches the code, and its examples run (JASMINE through `build_simulation`).
  `docs/USER_GUIDE.md` now matches the app (defaults, landscapes, controls such as Share, dark
  mode and Movement salinity; JASMINE as it works). README, in-app help and the parity analysis
  no longer claim an age-stratified mortality extension (its parameters are not read), a
  JAX backend, a 1.3x-Java speed, rejection of non-400 m grids, Weston flux off by default,
  six parallel kernels or old test counts; performance figures now come from
  `docs/performance/`. Old planning documents are marked historical.
- **Learned avoidance remembers where the animal was disturbed.** The remembered cell was the
  animal's position minus five times its own last step, which put it on the animal's track
  (on Kattegat with ships a median 5.5 km from the nearest deterring ship, while the animal
  itself was 5.1 km away). It is now the animal's own cell: the source is not well defined
  for moving ships or several sources summed. With all three memory changes of 2026-11-21
  (calibration, no habituation setting, own cell), animals within 5 km of the Kattegat test
  farm as a ratio to the run without turbines: 12-24 h after a pile 0.48 (DEPONS mode 0.56),
  1-3 days after the last pile 0.45 (0.53), 7-30 days after 0.97 (0.91); with ships (6 seeds)
  deterred animal-ticks 15.6% with and without the memory, mean energy 10.52 against 10.64.
- **Full JASMINE mode crashed when the population first outgrew its arrays.** With the
  JASMINE memory module on, the avoidance result cached for the next movement step kept the
  old array length when the arrays grew for newborn calves, and the next step failed with a
  broadcast error (Kattegat, 500 animals, seed 11: day 376, 625 against 500). The cached
  result now grows with the arrays (no avoidance for the new slots, unchanged for the rest).
  The same run now completes three years (676 animals at day 1,080). New slow-tier test: a
  full-JASMINE Kattegat run past the first growth. Found by the CENOP improvement review of
  2026-11-21.
- **JASMINE settings: two now work, eight are labelled as unused.** None of the ten JASMINE
  inputs in the Settings tab reached the model, which reads other parameter names. Memory
  decay rate and avoidance radius now set `memory_decay_rate` and `avoidance_radius` (same
  defaults, so default runs are unchanged). The physics inputs (body mass, drag, thrust,
  current weight), BMR scale, activity cost, disturbance cost and avoidance strength have no
  counterpart in the model yet; their tooltips and the help say so. The README, help and
  Settings tab no longer describe physics-based movement as working (JASMINE movement is a
  simplified walk; the physics integrator is not wired in), describe behavioural states as
  labels that set activity costs, and expand JASMINE as JASCO's Animal Simulation Model
  Including Noise Exposure. The avoidance radius is labelled in memory cells of 2 km (the
  default 20 reaches 40 km; it said grid cells). DEPONS-mode runs do not read the two wired
  settings (byte-identical with them changed). From the CENOP improvement review of
  2026-11-21, which counted four unused settings.
- **JASMINE movement on Kattegat: steps were three times too long.** JASMINE movement draws
  its step lengths with the DEPONS step model, but read salinity from the map (about 16 on
  Kattegat) where DEPONS moves with a fixed 34.07 on Kattegat (Porpoise.java:423-427), and
  the salinity term is strong. It now uses the same cell environment as the CRW paths
  (current month, Kattegat override). Kattegat, 500 animals, seed 11: mean step 2.07 cells
  (was 6.74; DEPONS mode 2.11); net displacement in 30 days 112 km (was 147; DEPONS mode 49,
  the rest of the gap is JASMINE's small random turns and missing food memory). JASMINE
  results change; DEPONS mode is unchanged (app-style DEPONS run byte-identical). On other
  landscapes both paths already used the map salinity.
- **Chart legends no longer cover the lines.** The legend sat inside the top-right corner of
  the plot area, over any line running high (for example "Energy falling 3+ days" on the
  dispersal chart). It is now a horizontal row above the plot area, under the title, on
  every Plotly chart with more than one line; charts with one line keep their compact layout.
- **"Plotly is not defined" in the browser console (Population and Disturbance tabs).** Every
  Plotly chart carried its own `<script src=...plotly...>`; Shiny inserted the chart and ran
  its drawing code before that download had finished, and Plotly hashed its 4.6 MB bundle on
  every render to build the tag. plotly.js is now loaded once in the page head
  (`cenop.ui.plotly_head`, same CDN file and integrity hash as Plotly's own tag) and the
  charts are built without it. Checked in a headless browser: no console errors on any tab,
  all nine Plotly charts drawn.
- **Dispersal chart: the second line was always 0.** It plotted "Max Declining Days", a value
  hard-coded to 0. The line now shows how many live animals meet the dispersal trigger's
  energy condition: daily mean energy dropping for at least t_disp days in a row ("Energy
  falling 3+ days"), next to the number dispersing. On Kattegat (500 animals) it runs at
  100-150 against 20-40 dispersing: DEPONS also ends a dispersal on the day it starts when
  today's energy is above the minimum of the previous seven days (Porpoise.java:1089-1119).
  `get_dispersal_stats()` reports `declining_count`, and `max_declining_days` is now computed
  (the longest current run; at most 9, as the energy history holds 10 days).
- **Behaviour states follow the model's dispersal (JASMINE energy; per-animal table).** The
  behaviour state machine started DISPERSING on a declining-energy counter that nothing
  increments, so no animal was ever in DISPERSING. It now mirrors the model: an animal the
  population is dispersing is DISPERSING (DISTURBED keeps priority), and returns to FORAGING
  when the population ends the dispersal. The app shows no behaviour states (its map and
  dispersal charts use the model's own dispersing flag); the states reach only
  `Simulation.agents_df["behavioral_state"]` and, in JASMINE mode, the energy costs.
  DEPONS mode: model unchanged (Kattegat 30 days, 500 animals, ships: model state identical).
  Of 7,544 dispersing animal-ticks, 826 are now DISPERSING (none before), 6,595 DISTURBED
  (priority) and 123 FORAGING: the step on which dispersal starts, because the state is
  updated before the end-of-step dispersal check. JASMINE mode: a dispersing animal now
  carries the DISPERSING activity multiplier (x1.2) instead of the one its speed or energy
  gave it. On Kattegat (60 days, 500 animals, no ships) 97% of dispersing animal-ticks were
  TRAVELING (x1.5) and 3% RESTING (x0.6) before, so for most the cost is lower. One seeded
  run: 497 alive, mean energy 11.26, against 499 and 11.42; not an effect size. The unused
  context fields (declining days, memory cell count, dispersal complete) are removed.
- **Land avoidance: a tie for the deepest neighbour is broken at random.** When an animal
  finds no water ahead and faces its deepest neighbouring cell, DEPONS shuffles the
  neighbours first, so equally deep cells are chosen at random; CENOP always took the same
  one. A random number is drawn only when there is a tie. On Kattegat the fallback fired
  3,650 times in 60 days (2000 animals) and never met a tie, so seeded results are unchanged
  there (all four 30-day state hashes identical). Landscapes with flat areas next to land
  can differ.
- **Land avoidance at the grid edge: neighbours outside the grid are left out.** In the
  same fallback, CENOP read cells beyond the grid edge as land, so for an animal on the edge
  whose neighbours inside the grid were all land, a cell outside the grid could tie with
  them and be faced. DEPONS's neighbourhood stops at the edge (Repast's `GridCellNgh` on a
  grid that is not periodic, checked in the release's compiled classes). Seeded Kattegat
  results are unchanged (four state hashes and the ship fingerprint identical).

## [2.3.0] - 2026-11-20

CENOP checked against runs of DEPONS 3.2 for the first time. The comparison found and fixed
defects that kept populations, dispersal, ship traffic and ship noise from matching DEPONS;
results of every earlier version differ. See `docs/validation/A2-parity-report.md`.

### Added
- DEPONS 3.2 reference harness (`validation/depons_java/`, roadmap Track A1). It runs the
  official, unmodified DEPONS 3.2 release headless on Linux from one command: it fetches the
  release jar and Kattegat landscape (size + SHA-256 pinned), installs them unattended, runs
  each scenario seed, and writes a provenance `manifest.json` per run. Stdlib only, 85 fast
  tests; needs Java 21 for the model and Java 11 for the release's installer. See
  `validation/depons_java/README.md` and
  `docs/superpowers/specs/2026-11-10-track-a1-spike-report.md`

- Dispersal modes `off` and `Undirected` (DEPONS `DispersalOff`, `UndirectedDispersal`).
  `Undirected` always picks a random target, turns by the full +/- `PSM_angle` draw with no
  logistic damping, and stops when the straight-line distance from its start reaches the
  target distance. It runs on the default NumPy/Numba path; the JAX backend refuses it. With
  JASMINE-physics or hybrid movement, which have no per-step dispersal turn in any mode, only
  the random target and the full-distance stop apply. See
  `docs/superpowers/specs/2026-11-18-dispersal-selector-design.md`

- Dispersal mode `InnerDanishWaters` (DEPONS `InnerDanishWatersDispersal`), the mode the
  DEPONS 3.2 TRACE document specifies for Kattegat. It is opt-in: PSM-Type2 stays the
  default on every landscape and no baseline changed.
  - Phase 1 heads for one of the twelve best of sixty 40 km blocks, steering towards deep
    water and away from the coast; phase 2 follows the coast away from the previous day's
    position. Ported one animal at a time from the Java (`src/cenop/agents/dispersal_idw.py`),
    with the DEPONS quirks kept as they are (listed in
    `docs/superpowers/specs/2026-11-18-dispersal-idw-design.md`).
  - Kattegat landscape only; any other landscape is rejected when parameters are built
    (DEPONS also allows Homogeneous, CENOP does not yet). Refused on the JAX backend and
    with JASMINE-physics or hybrid movement.
  - Comparison with DEPONS-Java:
    `docs/superpowers/specs/2026-11-18-dispersal-idw-java-comparison.md`

### Fixed — the starting population (changes every run's results)
- **Females only, as in DEPONS.** CENOP made half of the starting animals male. DEPONS
  simulates females only, so every animal can become pregnant and pay the lactation cost.
  The sex field is gone: `is_female` is removed from the population, from the
  per-individual statistics output and the agent data frame, and the dashboard's `females`
  statistic is removed.
- **Pregnancy at setup.** Every starting animal with age > 0 is now pregnant with
  probability `h` (0.68), whatever its age, as in DEPONS (`Porpoise.java:165-178`). CENOP
  started only mature females pregnant. Non-pregnant animals start at status 0 and the
  mature ones become ready to mate at the first daily update.
- **Mating day at setup.** Drawn as in DEPONS, `round(N(tmating))` from the mating-day
  parameters. It was truncated, clipped to 0-359 and hard-coded to N(225, 20), so
  first-year mating was half a day early and a changed mating-day parameter was ignored in
  year 1.
- **Daily pregnancy update at tick 1.** DEPONS runs its daily task once at tick 1 as well
  as every 48 ticks; CENOP now runs the pregnancy update then too, so initially pregnant
  animals calve on the same day as in DEPONS (they were one day late).
- **Ageing once a day.** Age now advances by 1/360 in the daily step, which also runs at
  tick 1, as in DEPONS (`Porpoise.java:1082`). It advanced by 1/17280 every tick, which left
  animals one day younger than in DEPONS. The daily bycatch / max-age check also runs at
  tick 1 (not on the JAX path).
- **Lactation cost on the tick a calf is abandoned.** A starving mother that abandons her
  calf is still charged the lactation factor for that tick, as in DEPONS
  (`Porpoise.java:751-753`, `:773`). The NumPy and Cython paths charged 1.0x; JAX already
  matched. The Cython extension must be rebuilt
  (`src/cenop/optimizations/setup_cython.py build_ext --inplace`).
- **Weaned calves are never dropped.** When every slot was occupied a weaned calf was
  silently discarded, so a population could not grow beyond its starting size. The
  per-animal arrays now grow when needed, as DEPONS adds the calf unconditionally. Runs that
  never fill up (all Kattegat baselines) are unchanged by this.
- The unused per-animal `Porpoise` class and the unused per-animal branch in `Simulation`
  no longer carry a sex field.
- **Daily bycatch and max-age deaths on every backend.** The fused Cython kernel (used on
  homogeneous landscapes) only did starvation deaths, so animals there never died of old
  age or bycatch; the daily check now runs after it. The JAX path now also does the
  tick-1 max-age check.
- **`EnergyMode.JASMINE` no longer crashes.** The JASMINE energy module lacked the two
  split phases the population calls (food intake, then costs, with the starvation check
  between them), so any run with it failed on the first step. Runs with an energy module
  also charge lactation on the tick a calf is abandoned.
- **Population growth.** The arrays grow by a quarter at a time and log the memory in
  use (about 320 KB per animal). New parameter `max_population` (default 0 = no limit, as
  in DEPONS): at the limit weaned calves are dropped, counted in
  `PorpoisePopulation.calves_dropped` and logged once.
- **Effect.** About half the starting animals now calve around day 165 and use 1.4 times
  the energy while nursing, instead of about a tenth. Populations decline faster and match
  DEPONS-Java on the same landscape: Kattegat (v3.2 landscape), 2000 animals, 5 years,
  PSM-Type2: CENOP 44-60 against Java 51-57 (was 95-108); InnerDanishWaters: 40-48 against
  47-53 (was 88-98). Investigation:
  `docs/superpowers/specs/2026-11-18-kattegat-population-gap.md`
- Present in every release so far. All seeded results change (one fewer random draw at
  setup and a different starting state); the Kattegat reference baselines are regenerated.

### Deprecated — the JAX backend (2026-11-20)
- `use_jax=True` now raises a `FutureWarning`. The backend still runs where it ran before,
  but model changes are no longer ported to it and it will be removed in a later release.
- Reasons and the options considered: `docs/decisions/2026-11-20-backends.md` (roadmap B1).
  The Cython kernel is kept.

### Changed — UI aligned with Shiny 1.9 and shiny_deckgl 1.16 (2026-11-20)
- **Map legend.** The dashboard legend is now shiny_deckgl's layer legend widget. The custom
  JavaScript legend it replaces reached into the map's internals, which changed in
  shiny_deckgl 1.12: ticking a layer raised an error in the browser and did nothing.
  Behaviour kept: the background-map toggle, bathymetry and foraging with their tooltip
  layers, turbines with their blades, and a hidden layer staying hidden while the simulation
  updates the map (`src/cenop/server/map_legend.py`).
- **Empty map layers.** Placeholder bitmap layers were sent with an empty image address,
  which deck.gl 9.4 tried to load: two errors in the browser console on every page load.
- **CSV download.** Uses `render.download_button` (`render.download` is deprecated in
  Shiny 1.9).
- **Dependencies.** `shiny>=1.9.0`, `uvicorn>=0.50`. shiny_deckgl 1.16 is needed for the map
  but is not declared: PyPI carries 1.12.0 only; it comes from the `razinka` conda channel.

### Changed — faster tick, same results (roadmap B2, 2026-11-20)
- Kattegat tick at 2000 animals: 8.50 -> 4.94 ms with `InnerDanishWaters`, 9.59 -> 7.41 ms
  with `PSM-Type2`. Seeded results are bit-identical.
- The `InnerDanishWaters` dispersal step runs in one compiled call, and the two
  reference-memory kernels no longer take a modulo per remembered position.
- Profile and measurements: `docs/performance/2026-11-20-tick-profile.md`.

### Fixed — backlog pass 3 (2026-11-20)
- **`InnerDanishWaters`: eating cell after a stop with a move.** When the mode moved an
  animal and stopped it in the same tick, the animal ate at the cell it was moved to, and a
  land rollback returned it there. DEPONS uses the position at the start of the tick for
  both. The case is rare (222 of 8.1 million animal-ticks in 90 days, 2000 animals), but
  seeded `InnerDanishWaters` results change; the Kattegat baselines are regenerated
  (undisturbed 2000 -> 97 at day 360 -> 33 at day 1800; turbines -> 80 and ships -> 62 at
  day 720; before: 96, 36, 75 and 71). These are single-seed runs and the changes are within
  the spread between seeds; the A2 comparison was not repeated.
- **JASMINE physics movement: step length (deep-review finding #17).** The simplified step
  reused the previous step length, so every animal moved about 1.58 cells per tick for the
  whole run. The length is now redrawn each tick with the DEPONS step model.
- **JASMINE physics and hybrid movement ignored `random_seed`.** Both modules drew from an
  unseeded generator. They now use the population's generator, so runs in these modes are
  reproducible. The default movement (DEPONS random walk) was not affected.

### Validation — A2 repeated on the released code (run 4, 2026-11-20)
- All 32 CENOP runs were repeated after the last model change. With eight seeds per model,
  every metric passes or is inconclusive except mean energy in the ship scenario at day 150
  (4.29 against 4.14, 2.9 standard errors; inside the 0.25 size band).
- A check with sixteen seeds per model, fixed before the extra DEPONS runs were seen, does
  not confirm that difference: +0.05 at day 150 (1.0 standard error) and +0.05 over days
  110-150 (1.4). The first eight DEPONS seeds were low by chance.
- A search for a cause found none: exposure, deterrence strength, dispersal and step length
  of exposed animals agree with DEPONS period by period.
- Report: `docs/validation/A2-parity-report.md`. Earlier runs are kept beside it.

### Added — A2 cross-validation against DEPONS 3.2
- `validation/a2/`: four scenarios (undisturbed InnerDanishWaters, PSM-Type2 and dispersal
  off; ships), eight seeds per model, seven metrics with pass bands fixed in advance, a
  CENOP runner and a report generator. Report: `docs/validation/A2-parity-report.md`.
- **Undisturbed: no metric fails.** Population within 4.2% and 1.6 standard errors at every
  checkpoint, mean energy within 0.13, block shares correlated at 0.999, age at death
  indistinguishable.
- **Ships: no metric fails on the third run**, after the comparison found two defects in
  CENOP's ship code (next section). Exposure to ship noise 0.193 of animal-ticks against
  0.192 in DEPONS; population within 4.1% at every checkpoint. Only the first run was blind.

### Fixed — ship movement along a route (changes every run with ships)
- Found by the A2 ship scenario. CENOP skipped the wait at a route's first waypoint and,
  after the last waypoint, steered the ship back to the start at that waypoint's speed. In
  the Kattegat ship file the first makes all 5,148 ships sail at once, and the second (the
  last waypoint has speed 0) stranded every ship after one passage, so ship noise ended at
  about day 25 of a run. `Ship.update` now follows DEPONS's `Ship.move()`: one waypoint per
  tick, waits honoured, route restarted; about 250 ships sail in any tick.
- Effect (Kattegat, InnerDanishWaters, eight seeds): share of animal-ticks with ship
  deterrence 0.021 -> 0.283 (DEPONS 0.192); population with ships relative to undisturbed at
  the end of year 1, 0.994 -> 0.911 (DEPONS 0.918).
- A run with ships is also much faster: the ship baseline takes about 45 minutes.
- **Ship sound as in DEPONS by default.** After the movement fix animals still reacted to
  ships 1.45 times as often as in DEPONS. Transmission loss used a spreading formula unless
  `weston_flux_percell` was switched on, where DEPONS always uses WestonFlux for ships; depth,
  sediment and salinity were read at the animal's cell instead of at the ship's position;
  the water temperature was 12 degrees instead of 10; and the simulation used a fixed
  6-18 h day instead of the landscape's `suntimes.csv`. All four now follow DEPONS;
  `weston_flux_percell` defaults to on (the switch remains, to fall back to the spreading
  formula). Exposure 0.283 -> 0.193 (DEPONS 0.192).

### Validation — population against DEPONS-Java with eight Java seeds per mode (2026-11-19)
- Same landscape (v3.2 Kattegat), 2000 animals, undisturbed, five years; DEPONS-Java seeds
  42-49, CENOP seeds 42-47. Mean population at the end of years 1 / 2 / 3 / 4 / 5:
  - `InnerDanishWaters`: Java 92.9 / 79.5 / 69.8 / 62.6 / 44.6, CENOP 91.5 / 77.7 / 69.8 /
    62.8 / 42.7;
  - `PSM-Type2`: Java 167.1 / 121.6 / 97.0 / 82.0 / 55.5, CENOP 170.5 / 122.7 / 99.0 / 85.3 /
    56.3.
  No difference reaches one standard error (largest: -4.4% and +4.1%). Share of animals
  per 40 km block: correlation 0.998 and 0.994.
- **Dispersal off, one year, eight Java and six CENOP seeds.** Population and mean energy
  agree on every day checked (days 30 to 359): mean energy differs by -0.19 to +0.08, at
  most 1.1 standard errors; population by at most 11 animals (2.0 standard errors at day
  120, under 1.5 elsewhere). The "energy undershoot" carried as a known issue since
  2026-11-11 is withdrawn: its Java reference was the mean of 200 tracked animals, not of
  the population.
- Seed-to-seed spread is large on this scenario (SD 3-8 animals), and the two Java seeds
  used until now (42, 43) are at the high end for `InnerDanishWaters`. Comparisons with
  two or three seeds per side are not reliable; use six or more.

### Changed — the Kattegat baselines use InnerDanishWaters
- `scripts/run_kattegat_reference.py` takes `--dispersal` and defaults to
  `InnerDanishWaters`, the mode DEPONS prescribes for Kattegat (TRACE 3.2 section 3.2.4).
  The three committed baselines (`output/kattegat_ref*`) are regenerated with it, on the
  v3.2 landscape. The application default stays `PSM-Type2`.

### Fixed — calf memory and the order of the memory update when dispersing
- **Calf spatial memory (changes PSM-Type2 results).** DEPONS keeps a memory for a nursing
  mother's calf: under PSM-Type2 it receives every update of her memory, and the weaned
  calf starts with it and with her preferred dispersal distance. CENOP gave every newborn
  an empty memory and a newly drawn distance. Now as in DEPONS, per mode: memory for
  PSM-Type2 only; inherited distance for PSM-Type2 and `off`; a new draw for `Undirected`
  and `InnerDanishWaters`. Five-year runs in those last two modes are byte-identical;
  PSM-Type2 runs change after the first weanings (about day 400).
- **Reference memory on a dispersal move (D16; changes all dispersing modes).** DEPONS moves
  a dispersing animal before the memory update, so the food level it stores is that of
  the cell it moved to. CENOP stored the food of the cell it left. Fixed on the
  NumPy/Numba and JAX paths; the fused Cython kernel (homogeneous landscapes) reads the
  food after eating. Runs with dispersal off are unchanged.

### Changed — `data/Kattegat` is now the DEPONS v3.2 landscape (changes every Kattegat result)
- The bundled Kattegat landscape was DEPONS v3.0. It is replaced by the v3.2 landscape
  (`https://depons.eu`, the files the DEPONS 3.2 release downloads), byte-identical to the
  reference harness's copy, under CENOP's file names (`prey0000_MM.asc`,
  `salinity0000_MM.asc`, `ships.json`). Bathymetry, sediment and blocks are unchanged;
  prey, salinity, patches and distance to coast differ, and the ship file has 5,148 ships
  (637 before). CENOP and DEPONS-Java runs on Kattegat are now on the same landscape.
- Fixed: with ships enabled, the simulation always loaded the sample `data/ships.json`,
  whose routes lie outside the Kattegat grid, so ships had no effect there. A landscape's
  own `ships.json` is now used when it has one.
- Lithuania and CentralBaltic have no ship file, and the sample file's routes (made for
  Gemini) lie outside their grids, so enabling ships there has no effect. This is now
  logged as a warning when the simulation is set up. DanTysk gets 8 of the sample's 24
  waypoints; NorthSea and Gemini use their own files.
- Ship traffic is about 44% faster with identical results (random reaction numbers are
  generated only for animals in range, and pairs below the reaction threshold are skipped
  early). With Kattegat's 5,148 ships a tick still takes about 0.4 s at 2000 animals.

### Fixed — landscapes with cells that are not 400 m (changes Lithuania and CentralBaltic results)
- DEPONS is calibrated for 400 m cells, and CENOP assumes them throughout: turbine and ship
  noise distances, the random-walk step, dispersal distances, spatial memory, communication
  range and swimming speed. Lithuania (the default landscape) has 1 km cells and
  CentralBaltic 250 m, so on Lithuania every distance was treated as 0.4 of its real length
  (noise sources 2.5 times closer, animals covering 2.5 times the intended distance per
  tick), and on CentralBaltic as 1.6 times its real length.
- Landscapes on another grid are now resampled to 400 m when they are loaded
  (`cenop.landscape.resample`). The files in `data/` are unchanged. Every layer is resampled
  by nearest neighbour; distance to coast is recomputed from the water mask, in metres
  (Lithuania's layer was a placeholder plane). Lithuania becomes 822 x 952 cells,
  CentralBaltic 1125 x 1150.
- Effect on Lithuania with the Curonian Nord wind farm (2000 animals, 10 days, seed 42):
  deterrence is felt out to about 53 km (it was about 133 km of real distance), and
  deterred animal-ticks fall from 144,357 to 28,286.
- Dispersal on Lithuania (10,000 animals, 60 days): 28% of attempts find a target (2,153
  of 7,573); it was 1% (1,075 of 112,426).
- Cost on Lithuania at 10,000 animals: the per-animal spatial memory follows the grid
  (76 x 65 -> 190 x 164 cells), so peak memory rises from about 1.0 GB to 2.9 GB. Time
  per tick is unchanged (about 52 ms).
- Landscapes with 400 m cells (Kattegat, DanTysk, Gemini, NorthSea) are not touched;
  Kattegat state hashes are unchanged.
- Design: `docs/superpowers/specs/2026-11-19-landscapes-400m-design.md`

### Fixed — smaller items
- `Dispersal.txt`: `target_distance_km` was the target distance in cells divided by 1000;
  it is now in km (cells times the landscape's cell size).
- Batch runner: fewer seeds than replicates is rejected with a clear error when the runner
  is created (it failed with an `IndexError` mid-batch); a progress callback no longer
  fails when `run(progress=False)` is used.
- Behaviour FSM: the DISTURBED -> FORAGING recovery time (`disturbance_recovery_ticks`) now
  applies without the JASMINE memory module. The time since the last disturbance was only
  recorded by that module, so otherwise animals left DISTURBED on the first quiet tick.
- Help text: `mean_disp_dist` is 2.0 km (it said 1.6 km).
- Removed the unreferenced `app_backup.py`.
- `InnerDanishWaters`: an animal whose dispersal move ended within one cell of the right or
  top grid edge had its heading rewritten by the border reflection; the mode's own heading
  is now kept. Seeded `InnerDanishWaters` runs change slightly; the other modes do not.
- `docs/DEPONS-CENOP-PARITY-ANALYSIS.md`: status section added and the rows that the
  November comparison with DEPONS runs showed to be wrong are corrected.

### Fixed — the dispersal setting was ignored
- `params.dispersal` (the Dispersal Type selector, and the same field in scripts and batch
  runs) was never read: every run used PSM-Type2 whatever was selected, in every release so
  far. `off` now starts no dispersal and `Undirected` runs as in DEPONS. PSM-Type2 results
  are unchanged (bit-identical 30-day Kattegat state; no baseline regenerated).

### Fixed — model behaviour changes (see `docs/superpowers/specs/2026-11-10-phantom-food-investigation.md`)
- **Phantom food.** The vectorised eating code raised every cell an animal visited to the
  0.01 "artificial food" floor, including empty cells outside the food patches. DEPONS floors
  only cells that had food. Within a month almost the whole sea held 0.01 food that porpoises
  ate at nearly every step (more than twice their basal energy use), so populations on
  patchy real landscapes (e.g. Kattegat) were artificially stable.
  - Fixed in the Numba (v1/v2), JAX and Cython eat paths. The JAX path also no longer lets dead
    agent slots remove food.
  - Present in every release since the vectorised eat path was introduced (2026-01-26),
    including v2.2.0.
  - Homogeneous landscapes (food in every cell), including the slow validation tier, are
    unaffected.
  - **Committed Kattegat baselines are stale.** Real-landscape population results from earlier
    versions should be re-run.
- **Lithuania data gap (27% of its water had no habitat data).** Commit `7d50cfb` (2026-03-12)
  replaced the Lithuania bathymetry with a larger 329x381 grid (+12,014 water cells: a 55 km
  open-sea strip in the west, a refined coastline and the lagoon) but only padded the other 28
  layers with NODATA. Their data still covered exactly the old water mask, so 19,767 of the
  73,464 water cells had NODATA salinity, prey, food patches, sediment, blocks and distance to
  coast.
  - Effect: salinity -9999 makes `a2 * salinity` = +200, so every CRW step on those cells was
    clamped to `max_mov` (13.4 against 6.5 cells) with a mean turn of 90 degrees, and those
    cells had no food. 25-27% of animals started there. This also produced the
    `overflow encountered in power` / `invalid value encountered in multiply` warnings at
    `crw_core.py:131-132` on the live app (an unmasked agent's unclamped `log_mov` of about 200
    overflowed the CRW module's float32 step buffer; no state was corrupted).
  - Present in every Lithuania run since 2026-03-12. Lithuania results from that period, including
    the Lithuania figures under "Changed" below, were measured on the gapped data.
  - Fix: the existing Lithuania layers are synthetic placeholders (independent per-cell noise for
    patches, prey and salinity; constants for sediment and blocks; a plane for distance to
    coast), so `scripts/fill_lithuania_data_gap.py` completes them on the gap cells only with
    the same kind of values (a fixed-seed draw from each layer's own water values, month by
    month; the fitted plane for distance to coast). Existing values are byte-identical.
    Investigation probes: `scripts/investigations/2026-11-11-lithuania-data-gap/`.
  - The landscape loader now warns when any layer has NODATA on water cells (warning only:
    DEPONS-origin Kattegat and NorthSea data keep their known gaps), and the CRW module no longer
    computes steps for agents it does not move. Kattegat starting positions are bit-identical.
  - Lithuania's food, prey and salinity layers remain placeholders with no spatial structure;
    Lithuania results reflect real bathymetry with synthetic habitat.
- **CentralBaltic data gap (3.25% of its water had generator land values).**
  `scripts/generate_central_baltic.py` built every CentralBaltic layer on its own synthetic
  bathymetry, whose only land is a border band (edge distance under 2% of the grid, about 9 km);
  `scripts/download_emodnet_tiles.py` later replaced `bathy.asc` alone with EMODnet depths, which
  have water in much of that band. Those 59,776 water cells kept NODATA salinity and prey (the
  same max-speed, 90-degree-turn movement as in Lithuania) and 0 for patches (no food),
  sediment, blocks and distance to coast; 3.2-3.4% of animals started there.
  - Fix: `scripts/fill_centralbaltic_data_gap.py` gives exactly those cells the values the
    generator gives water cells, with its own rules (fixed seed; the EMODnet depth for the
    sediment class and `disttocoast = 50 * depth`). Every other value is byte-identical.
  - The unused 1 km monthly files (`prey0000_XX.asc`, `salinity0000_XX.asc`, a 400x400 grid with
    a different origin) are removed; the loader always read the 250 m files.
  - Not changed: outside the band, sediment and distance to coast still follow the generator's
    synthetic depth rather than the EMODnet depth (sediment only affects ship noise; the
    simulation does not use distance to coast). CentralBaltic habitat layers are synthetic.

### Changed
- `SimulationParameters` rejects an unknown dispersal value with an error, and rejects
  `InnerDanishWaters` on any landscape other than Kattegat. Before, `InnerDanishWaters`
  silently ran PSM-Type2 everywhere.
- `Dispersal.txt` records the active mode (`PSM-Type2`, `Undirected`) in `dispersal_type`
  instead of the constant `PSM`.
- **Land avoidance on CRW movement follows DEPONS 3.2 (spec D15), plus the D9 first-call water
  stop.** This is a behaviour change.
  - Look-ahead of `presMov` cells (path sampled every 0.1 cell) before heading composition;
    escape turns of 40/70/120 degrees; walk-back (beh 4) and deepest-neighbour (beh 5) fallbacks;
    no post-composition land check; beh-6 turn-back after a post-move rollback.
  - D9: a new disperser whose stored not-enough-water state is set stops at its first dispersal
    call.
  - Implemented on the NumPy/Numba and JAX paths. Homogeneous landscapes, the Cython fast path
    and non-CRW movement modules are unchanged.
  - Measured against DEPONS (two seeds, Kattegat): dispersal-off N within +3.3% to day 165,
    activation 0.152 / 0.157, coast share 1.0-1.2%; one intake bin of seed 43 is +3.3%
    (accepted, spec Finding 4 Step 5).
  - **The committed Kattegat baselines are stale**, and seeded runs are not comparable with
    seeded runs from earlier versions (the land check draws from a Numba RNG seeded from the
    population RNG).
  - **Default Lithuania landscape (60 days, 10000 animals, s42 / s43), against the code before
    this change:** N 9856 / 9839 against 9989 / 9991; mean energy 15.51 / 15.50 against
    14.06 / 14.05. There is no DEPONS reference for Lithuania, and the known 1 km cell-size issue
    (km-to-cell conversions assume 400 m cells, see Known issues) applies there. Diagnostics below
    are from seed 42.
    - **The N drop is the Curonian Lagoon.** At 1 km resolution the lagoon (1,422 water cells,
      mean depth about 2.9 m, food per cell about a third of the sea's) is a separate water body,
      at least 2 land cells from the sea. The old destination-only check let animals hop the spit
      (133,118 crossings, 0.46% of animal-ticks). With the DEPONS path check, the 1.5% of animals
      placed in the lagoon at the start are trapped (1 crossing) and starve within about 60 days:
      134 of the 144 deaths are in the lagoon, which accounts for the whole N drop
      (9989 → 9856). The new behaviour is the faithful one; the trap is a data/resolution
      artefact (the real Klaipėda Strait is narrower than one cell). **Resolved (2026-11-11):**
      for Lithuania only, initial placement is restricted to the largest connected water body
      (8-connected cells at least 1 m deep). DEPONS landscapes keep DEPONS's rule (any water
      with depth > 0, as `moveAwayFromLand`), so Kattegat starting positions are bit-identical.
      With the restriction (seed 42, 10,000 animals, 60 days): no animals start in the lagoon,
      N is 9991 at day 60 (9856 without it), and the 9 deaths are all at sea.
      **Revisited after the Lithuania data-gap fix (see Fixed):** the starvation came from the
      gap, not the trap. Most lagoon cells had NODATA food, which is why lagoon food per cell
      looked like a third of the sea's. On the fixed data (same seed and size) without the
      restriction, all 151 animals that start in the lagoon survive and stay there (0
      crossings) with day-60 energy 18.89 against 17.04 at sea; N 9994, 6 deaths, all at sea.
      With the restriction: N 9996, mean energy 17.00, 4 deaths, all at sea. The restriction is
      kept (decided 2026-11-11): without it 1.5% of animals form an isolated, unrealistically
      well-fed lagoon group that never rejoins the sea population.
    - **The energy rise is in the sea population:** day-60 sea energy 15.54 against 14.06 (the
      lagoon deaths shift the mean by less than 0.03). Intake per animal-tick is +7.4% over the
      60 days (+11%, +9%, +8.6% and +6.6% in the 5 days before days 15, 30, 45 and 60), and time
      in 1-5 m water falls from 1.31% to 0.90%. It does not appear when food is not limiting
      (2000 animals, 15 days: sea energy at the cap, 19.835 in both). Ruled out: lagoon trapping,
      stuck animals (zero-displacement ticks under 0.005% of animal-ticks in both 10000-animal
      runs) and the start-of-tick eat cell. The mechanism is open; a plausible, untested
      contributor is the look-ahead, which on 1 km cells covers 2.5 times its intended distance
      (see Known issues). The sign is opposite to Kattegat, where the port brought intake down
      to DEPONS's level.
    - Time within 1 km of the coast (19.9% / 19.8% against 20.5% / 20.5%) says nothing: the
      "27% of water cells with `disttocoast` under 1 km" behind it was the data gap (NODATA,
      -9999, counts as under 1 km). Lithuania's `disttocoast` is a placeholder plane (minimum 1
      after the fix), not a distance.
  - **Slower on real landscapes (NumPy/Numba default path).** Without the fix described next,
    the land check took about 46-47% of tick time on Lithuania and 15% on Kattegat (2000
    animals); an instrumented Lithuania 60-day run (10000 animals, seed 42) took about 289 s
    against 161 s before this change, and Kattegat runs took about 30% longer (10 days, 2000
    animals). Fix: the final path check is no longer repeated for an agent that left the rounds
    on a pass (91% of checked agents on Lithuania and 95% on Kattegat pass at round 0), with
    identical results and RNG draws. Timed over 96 ticks after a one-day warm-up (2000 animals,
    three runs each): land check 9.2 → 5.5 ms per tick on Lithuania (46% → 34% of tick time;
    tick 19.9 → 16.2 ms) and 1.8 → 1.0 ms on Kattegat (15% → 9%; tick 12.2 → 11.4 ms). A 60-day
    wall time after the fix was not measured.
- **`communication_enabled` now defaults to `False` in `SimulationParameters`.** This matches
  the UI checkbox, which was already off.
  - With realistic food the DEPONS random-walk weight is about 0.001, so the JASMINE
    social/cohesion vector took over the heading: 69% of steps reversed direction, and net
    movement was 1.9 km a day against DEPONS's 10.5 km.
  - JASMINE social behaviour is now opt-in. Headless runs (scripts, baselines, tests) were the
    only ones affected.
  - With communication off, homogeneous landscapes now use the (equivalence-tested) Cython
    fast path.

- **PSM-Type2 dispersal now follows DEPONS 3.2. This is a model behaviour change.** On
  Kattegat, 20.3% of animal-ticks were dispersing in stock CENOP against 1.34% in DEPONS
  (days 0-120). Now:
  - the trigger fires after `t_disp` strict drops of the 3-d.p. daily mean energy;
  - targets are validated, and an animal with no valid target does not disperse;
  - a new disperser starts heading for its target;
  - the dispersal turn is +/-`psm_angle` (40 degrees), weighted by path length travelled;
  - dispersal stops when the point 5 cells ahead is off the grid or no deeper than `wdisp` (4 m);
  - completion is on path length travelled (95% of the target distance).
  - The JAX backend mirrors all of this.
  - Measured against DEPONS 3.2 (Kattegat, seeds 42 and 43, days 0-120): activation 0.126 /
    0.134 (DEPONS 0.133), episode median 5 moves, mean 11.0 / 10.8 (12.1), tick share dispersing
    1.09% / 1.15% (1.34%), N with/without dispersal at day 150 0.85 / 0.82 (0.80 / 0.81).
  - **Dispersal is now rare on small grids.** Targets are drawn at N(350,100) km and rarely fit
    on Lithuania (the default landscape), Homogeneous or the 400x400 landscapes (DanTysk,
    Gemini). Share of activations that find a random target (empty memory, initial positions;
    `target_success.py` from the final review): Lithuania 329x381 0.5%, Homogeneous 400x400
    2.6%, Kattegat 600x1000 31.9%. Before this change every activation found a target.
  - Seeded runs are still deterministic but are not comparable with seeded runs from earlier
    versions: the PSM-turn draw now precedes the CRW draws, and an activation with no memory
    target draws 2x1000 random numbers for its random-target tries.
  - **The committed Kattegat baselines are stale** until regenerated.

### Known issues (found 2026-11-10, not yet fixed)
- ~~JASMINE social vector scaling~~ Rescaled after 2.3.0 (see Unreleased); the weight is still uncalibrated.
- ~~Non-CRW movement modules keep the old dispersal completion~~ Fixed after 2.3.0 (see Unreleased).
- **JAX backend is about 5x slower per tick on real landscapes after the land-avoidance port**
  (about 70 ms against 13 ms before, at 2000 animals; Numba about 7 ms). JAX is deprecated since
  2026-11-20 (see Deprecated above), so this will not be fixed.
- ~~Known deviations from DEPONS in land avoidance~~ Both fixed after 2.3.0 (see Unreleased).
- ~~CentralBaltic misplaced, placeholder layers~~ Rebuilt from public data after 2.3.0 (see Unreleased).
- **Baltic prey maps are uniform for now.** Monthly prey maps for Lithuania and CentralBaltic
  have been built from the SAMBAH harbour porpoise distribution (Carlen et al. 2018) with
  `scripts/build_food_layers.py`, but are not published: the data record names no licence and
  HELCOM's copy allows non-commercial use only. Until the data owner confirms, both landscapes
  carry uniform prey on random patches, so food has no spatial or seasonal pattern there.
- ~~CentralBaltic lagoons after coarsening~~ Obsolete: CentralBaltic was rebuilt on 1 km cells after 2.3.0.
- ~~The simulation never reads the `dispersal` parameter~~ Fixed 2026-11-18: all four modes
  take effect (see Added and Fixed above). The Kattegat baselines use `InnerDanishWaters` since 2026-11-19.
- ~~`InnerDanishWaters`: eating cell after a stop with a move~~ Fixed 2026-11-20 (see Fixed, backlog pass 3).

### CI & docs
- The CI lint gate now also checks `validation/` (the harness package)
- Pinned the lint tools in `environment.yml` (`black==26.1.0`, `ruff==0.15.21`) so new
  formatter/linter releases can't fail the lint gate without a code change
- GitHub default branch switched to `CENOP-JASMINE` (2026-11-10), so the nightly
  slow-tier workflow (`slow.yml`) is now registered; it had never run while the default
  was the stale `main`
- Refreshed stale docs: `docs/backend-equivalence.md` (Cython repaired, not broken),
  `docs/validation-suite-status.md` (CI exists; default-branch and 60-day caveats),
  `CLAUDE.md` (CI and test counts)
- ~~The behaviour state machine never enters DISPERSING~~ Fixed 2026-11-21 (see Unreleased).
- **The JASMINE energy budget loses animals with ships** (found 2026-11-21; much smaller since
  the Kattegat salinity fix; JASMINE mode uses the DEPONS energy budget unless
  `energy_mode="JASMINE"`). Kattegat, 500 animals, 60 days, seed 7, JASMINE movement and
  energy: 455 alive with ships against 499 without (72 before the salinity fix, when JASMINE
  steps were 3.3 times too long); with the DEPONS energy budget and ships, 493. DISTURBED is
  sticky: any deterrence above 0.01 makes an animal DISTURBED until a day passes without
  deterrence, so 85-89% of animals are DISTURBED while 1-15% are deterred at that moment, and
  DISTURBED doubles the JASMINE activity cost; with recovery as soon as deterrence stops, 459
  survive. The JASMINE energy module also charges every deterred tick (0.1 x deterrence
  strength + 0.02 x speed). Reproduce with
  `scripts/investigations/2026-11-21-jasmine-ships/jasmine_ships_split.py`.
- ~~Learned avoidance (JASMINE disturbance memory) remembers the wrong place~~ Fixed 2026-11-21 (see Unreleased).

## [2.2.0] - 2026-07-10

### Deep-Review Fix Cycle (29 tasks, 6 phases)

Adversarial code review (2026-07-06) and TDD fixes. Two CRITICAL model-validity issues
plus default-path correctness, DEPONS parity, and backend equivalence.

#### Critical model-validity fixes
- Turbine deterrence vector now built from grid displacement, not metres — was ~400× too
  large versus DEPONS and CENOP's own scalar deterrence path
- Extracted the validated CRW generation/composition into a shared `crw_core` module and
  routed the web-app movement path through it (previously ran an unvalidated CRW)

#### Default-path correctness & reproducibility
- Fixed double daily food regrowth on the vectorized path
- Fully reset recycled dead-agent slots for weaned calves (SoA + PSM + behaviour/energy + id)
- Seeded the PSM RNG and plumbed `psm_dist_mean/sd` (N(350;100), DEPONS 3.2); fixed stale
  UI/controller defaults (300 → 350)
- Fixed stale rS/rR energy-panel UI defaults (0.04 → 0.03)
- Introduced `_WorkerHandle` (fresh stop_event + queue per run) to fix a Stop→Start worker race

#### DEPONS parity
- Gated non-DEPONS swimming/disturbance energy terms behind params (BMR-only by default)
- Made turbine deterrence deterministic by default (probabilistic scaling behind a JASMINE opt-in)
- Exposed true per-tick birth/death and per-cause mortality counters
- Excluded paused / no-buoy ships from the deterrence candidate set (DEPONS `deterPorpoise` gate)
- Restored the dropped age-1 weight in `AGE_DISTRIBUTION_FREQUENCY` (311/54 → 312/55 ones,
  now bit-identical to DEPONS `PorpoiseSimBuilder`)

#### Defense-in-depth & cleanup
- Clamp non-positive vessel length in `jomopans_spl` and validate it at load time
- Removed the effect-free blade-animation rAF loop and its dead constants/parameter

### Backend Parity (Track B)
- **JAX:** dispersal-heading parity with the NumPy reference (PSM-Type2 random turn), food
  floor 0.01 (not `u_min`), and dead-slot exclusion from reference-memory updates
- **Cython:** repaired all three documented defects — float32 `food_grid` cast, post-move
  land rollback, seeded-RNG mortality — plus heading recompute after boundary reflection;
  the single-tick backend-equivalence guard now passes (was `xfail`)

### Reference Baselines
- Regenerated the Kattegat reference baselines (undisturbed 5 yr, ships 2 yr, turbines 2 yr;
  seed 42) after the parity fixes

### Continuous Integration (new)
- GitHub Actions: lint (black + ruff) and the fast suite on every push/PR; nightly slow
  validation tier (`workflow_dispatch` on demand)
- `environment.yml` reproducing the scientific stack via conda-forge (+ pip for the Shiny
  stack and `jax[cpu]`)
- Non-cone sparse-checkout skips ~2 GB of unused landscape regions per run
- Repo-wide black + ruff format pass; ruff config migrated to `[tool.ruff.lint]` with a
  documented ignore list (high-value rules stay enforced)

### Fixes
- `BatchRunner._run_parallel`'s sequential fallback referenced an undefined `progress`
  (NameError masking the real failure) — surfaced by ruff F821
- Ship-loader tests resolve `data/Kattegat/ships.json` relative to the repo root (were CWD-dependent)

### Testing
- 810 fast tests + 11 slow multi-year validation tests passing

## [2.1.0] - 2026-03-14

### DEPONS 3.2 Full Sync (5 Phases)

#### Phase 1: Parameter Defaults & Energy Foundation
- Updated survival parameters: `m_mort_prob_const` 0.5→1.0, `x_survival_const` 0.15→0.4
- Updated deterrence parameters: `deter_coeff` 0.07→0.012, `deter_threshold` 158→152 dB,
  `deter_max_distance` 50→1000 km, `deter_time` 5→0
- Updated sound propagation: `alpha_hat` 0→0.00027, `beta_hat` 20→14.72
- Updated memory decay: `r_s` 0.04→0.03, `r_r` 0.04→0.03
- Updated dispersal: `mean_disp_dist` 1.05→2.0 km
- Fixed JASMINE survival to use DEPONS starvation formula instead of hardcoded 0.95 base

#### Phase 2: Reproduction & Mortality
- Replaced probabilistic birth model with three-state pregnancy FSM
  (immature → pregnant → ready-to-mate)
- Calves created at weaning (240 days post-birth), not at conception
- Mating days re-randomised each year
- Bycatch and max-age mortality gated to daily boundaries (tick % 48 == 0)

#### Phase 3: Food & Energy Foundation
- Replaced linear food regrowth with logistic formula: `F += rU × F × (1 − F/K)`
- Added 48-iteration daily compounding matching DEPONS 3.2
- MaxEnt-based food initialisation: `maxU × maxEnt / meanMaxEnt`
- Split energy module to expose mid-step energy for correct starvation ordering
- Fixed seasonal scaling to return scalar `np.float32`

#### Phase 4: Movement & Memory
- Added reference memory system with 120-entry circular buffer per agent
- Precomputed logistic decay tables for refMemStrength (rR=0.03) and workMemStrength (rS=0.03)
- Vectorized `compute_ve_total` and `compute_attraction_vector` using NumPy advanced indexing
- Heading composition: `totalD = (dx,dy) × crwContrib + vt + deterVt`
- CRW rejection sampling (max 200 retries) for angle and step length
- Vectorized land avoidance backtrack and deepest-neighbor fallbacks

#### Phase 5: Deterrence & Dispersal
- Fixed deterrence vector normalization: raw displacement × strength × coeff (not unit vectors)
- Added ship deterrence standardization (standardised logistic regression)
- Ported WestonFlux physics-based transmission loss from DEPONS Java
- Added JOMOPANS 13-class calibrated ship source levels (VesselClass enum)
- Wired PSM-Type2 dispersal module into population with SSLogis heading
- Added energy-based dispersal stop (checked at day boundaries)
- Added deterrence-deactivates-dispersal behavior
- Fixed dispersal step length to `mean_disp_dist / 0.4` grid cells

### Numba Optimization (10 Tasks)
- Added 7 JIT-compiled kernels in `optimizations/kernels.py`:
  `reflect_boundaries_kernel`, `seed_numba_rng`, `crw_angle_step_kernel`,
  `turn_position_kernel`, `eat_food_kernel`, `depons_bmr_cost_kernel`,
  `social_accumulate_kernel`
- `prange` parallelism on: reflect_boundaries, turn_position, depons_bmr_cost
- All kernels sub-millisecond for 500 agents

### Allocation Optimization
- Pre-allocated float64 buffers for turn_position_kernel
- Pre-allocated `_pre_move_x/y`, `_orig_dx/dy`, `_pre_heading`, `_positions` buffers
- Added `RefMemWorkspace` for reusable memory computation workspace (~1.5 MB/tick saved)
- Vectorized land avoidance backtrack/deepest-neighbor fallbacks (was Python for-loops)
- Vectorized dispersal trigger memory count check
- Removed dead `swimming_cost * 0.0` and redundant `10^log_mov`
- Removed wasted `compute_survival_probability` from energy module `compute_energy_update`

### Performance Optimization Pass 2 (3.27 ms/tick, 1.68x cumulative speedup)
- Numba-compiled all 6 WestonFlux transmission loss functions (`@njit`)
- Vectorized ship TL loop with `_compute_tl_percell` helper (also `@njit`)
- Land avoidance only processes blocked agents (typically <5% of population)
- Food regrowth `regrow_food_kernel` with `prange` parallelism (3.4x speedup)
- Fixed CRW kernel dtype: 8 SoA arrays changed from float32→float64 to match kernel signatures
- Fused reference memory into `prange` Numba kernels (`compute_ve_total_kernel`,
  `compute_attraction_kernel`), eliminating ~960KB/tick intermediate array copies
- Pre-allocated energy/context arrays (`_water_temp`, `_food_quality`,
  `_behavioral_state_buf`, `_speed_ms`) and cached mortality params
- Changed `_mem_ptr`/`_mem_count` from int16→int32, eliminating per-tick `.astype()` copies

### Per-Cell WestonFlux Transmission Loss
- Optional "Weston Flux TL" toggle in Settings > Basic tab
- Reads depth, sediment grain size, and salinity per-porpoise from landscape grids
- NODATA cells (depth ≤ 0 or grain_size == -9999) fall back to simple α/β formula
- Disabled when sediment data unavailable

### Porpoise Traces
- Fixed porpoise layer visibility (`visible=True` for partial_update)
- Expanded `get_porpoise_positions()` to 7 columns (id, x, y, energy, heading, age, is_dispersing)
- Animated trace trails using shiny-deckgl `TripsLayer`
- Sidebar controls: "Show porpoise traces" checkbox + trace length slider
- Trail history collection in background thread (per-porpoise deque, thread-safe)
- Trails colored by porpoise state (blue=adult, green=calf, red=dispersing, gray=senior)

### Skip Visualization
- "Skip visualization (fast run)" checkbox in sidebar
- Bypasses position extraction, coordinate conversion, and trail collection
- 25.7% end-to-end speedup (1.35x) measured via Playwright browser benchmark

### Benchmarking
- `scripts/benchmark_viz_overhead.py` with warm-up, multiple runs, std deviation,
  `--all-scenarios` flag for 5-configuration comparison
- `scripts/benchmark_kernels.py` for individual Numba kernel timing
- Playwright-based end-to-end browser benchmark for real-world overhead

### Unified Grid Visualization
- Replaced grid_layer/scatterplot rendering with bitmap+scatter pipeline
- Server-side PNG generation (one pixel per grid cell, colour-mapped)
- Single `build_grid_bitmap_layer()` for all ASC grid layers
- Transparent `scatterplot_layer` overlay for hover tooltips
- Supports continuous (linear gradient) and categorical (discrete lookup) schemes

### Documentation
- Updated in-app Help modal with all DEPONS 3.2 features
- Added Reproduction & Mortality section (pregnancy FSM)
- Added Reference Memory System section
- Updated Performance section (3→10 kernels, 6 with prange)
- Updated Scientific Background and Model Validation sections
- Updated sediment status: "Active when WestonFlux enabled"
- Fixed distance-to-coast units (kilometres → metres)
- Created CHANGELOG.md with full version history
- Updated README.md, API.md, USER_GUIDE.md for v2.1

### Code Quality
- 515+ automated tests across 24 test files
- Population stability smoke test
- DEPONS/JASMINE survival consistency test
- Per-cell WestonFlux integration tests (NODATA fallback, month selection)
- Porpoise trails layer tests

## [2.0.0] - 2025

### Added
- JASMINE simulation mode with dual-mode configuration
- Hybrid behavioral FSM (5 states: FORAGING, TRAVELING, RESTING, DISPERSING, DISTURBED)
- Dynamic Energy Budget with body mass scaling (Kleiber)
- Disturbance memory with learned avoidance and habituation
- Physics-based movement (hydrodynamic drag, thrust, ocean currents)
- 6-tab Shiny web interface (Dashboard, Settings, Population, Disturbance, Landscape, Export)
- DeckGL map with porpoise positions, depth heatmap, turbine locations, noise contours
- Collapsible chart panel with population, births/deaths, and energy time series

## [0.1.0] - 2024

### Added
- Initial release with core simulation functionality
- Core modules: Landscape, Population, Energetics, PSM
- DEPONS-compatible output format
- Shiny web interface
- Data preview with message handler
