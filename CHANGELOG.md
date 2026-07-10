# Changelog

All notable changes to CENOP-JASMINE are documented in this file.

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
