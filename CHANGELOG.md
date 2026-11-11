# Changelog

All notable changes to CENOP-JASMINE are documented in this file.

## [Unreleased]

### Added
- DEPONS 3.2 reference harness (`validation/depons_java/`, roadmap Track A1). It runs the
  official, unmodified DEPONS 3.2 release headless on Linux from one command: it fetches the
  release jar and Kattegat landscape (size + SHA-256 pinned), installs them unattended, runs
  each scenario seed, and writes a provenance `manifest.json` per run. Stdlib only, 85 fast
  tests; needs Java 21 for the model and Java 11 for the release's installer. See
  `validation/depons_java/README.md` and
  `docs/superpowers/specs/2026-11-10-track-a1-spike-report.md`

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
- **JASMINE social vector scaling.** When enabled, the social vector overrides the DEPONS random
  walk under realistic food. It needs rescaling before social behaviour is used in science runs.
- **Non-CRW movement modules keep the old dispersal completion.** The JASMINE physics and
  hybrid modules (`_update_movement_module_generic`) still end dispersal on straight-line
  displacement from the start point, not on path length. DEPONS parity applies to the CRW paths
  only.
- Not ported from DEPONS dispersal: the order of the reference-memory turn relative to the
  dispersal move (spec row D16).
- **Remainders after the D15 port (spec Finding 4, Step 5).**
  - Energy undershoot: with dispersal off, mean energy at days 90 / 120 / 150 / 165 is
    6.32 / 5.13 / 3.92 / 3.77 (s42) and 6.23 / 5.08 / 3.91 / 3.77 (s43), against DEPONS
    6.36 / 5.24 / 4.09 / 3.79 and 6.33 / 5.31 / 4.17 / 3.88: about 0.0-0.3 low (max 0.31, at
    s43 day 135), with N matching.
  - Late deaths: days 165-180 show 169 / 160 CENOP deaths against 203 / 191 in DEPONS, while
    the lactating share rises to 4-6%. Day 180 N with dispersal on is +13% / +8% above DEPONS.
  - D15 is no longer a candidate cause of either.
- **JAX backend is about 5x slower per tick on real landscapes after the land-avoidance port**
  (about 70 ms against 13 ms before, at 2000 animals; Numba about 7 ms). JAX stays opt-in; a
  follow-up can compact the still-needing agents in the land-check loop.
- **Known deviations from DEPONS introduced by the land-avoidance port:** beh-5 depth ties use a
  fixed neighbour order (DEPONS shuffles); look-ahead points are reflected at the grid edge as in
  DEPONS's BouncyBorders, but beh-5 neighbours outside the grid read as land.
- **Lithuania's 1 km cells conflict with CENOP's 400 m km-to-cell conversions.**
  `data/Lithuania/bathy.asc` has CELLSIZE 1000, but CENOP converts km to cells by dividing by
  0.4 (DEPONS's 400 m cells) for dispersal distances, the dispersal step and other movement
  scales, so on Lithuania these cover 2.5 times the intended distance. Pre-existing, and outside
  the dispersal port.
- `data/Kattegat` is the DEPONS **v3.0** Kattegat landscape, not v3.2 (all 33 files
  byte-identical to v3.0; prey, salinity, distance-to-coast and patch maps differ from
  v3.2). CENOP's loader cannot read the v3.2 files as shipped
- The simulation never reads the `dispersal` parameter: large-scale movement is always
  PSM-Type2, so the UI dispersal selector has no effect and `InnerDanishWaters` (which
  DEPONS TRACE 3.2 §3.2.4 specifies for Kattegat) cannot be run

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
