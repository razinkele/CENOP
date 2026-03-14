# Changelog

All notable changes to CENOP-JASMINE are documented in this file.

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
- Updated Performance section (3→7 kernels)
- Updated Scientific Background and Model Validation sections

### Code Quality
- 502+ automated tests across 24 test files
- Population stability smoke test
- DEPONS/JASMINE survival consistency test

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
