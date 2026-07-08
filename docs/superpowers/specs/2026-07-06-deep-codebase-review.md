# CENOP Deep Codebase Review — 2026-07-06

_Multi-agent review: 15 scoped finders (module × lens) → 2 adversarial verifiers per finding → synthesis. 102 agents, 0 errors, 43 raw findings → 32 survived verification → 28 fully confirmed (both lenses). Every finding below was read against the actual source; the top CRITICAL/HIGH default-path items were additionally re-verified by hand._

## Executive summary

The CENOP simulation core is broadly correct on its validated headless/Numba path, but this review surfaces a systemic split between what the DEPONS-parity work validated and what the shipped product actually runs: the interactive Shiny web app routes DEPONS mode through a simplified, unvalidated CRW module, applies a metre-scaled (~400x) turbine deterrence vector, and uses stale DEPONS-3.0 parameter defaults (rS/rR=0.04, PSM preferred-distance mean 300 km) so default web runs diverge materially from the committed baselines. Two defects also hit the validated headless path: food logistic regrowth runs twice per simulated day, and newborn calves inherit dead agents' stale memory/dispersal state on recycled array slots — both biasing multi-year population dynamics (the model's primary use case). Reproducibility is compromised by an unseeded PSM RNG, and the Numba parallel kernels oversubscribe threads (net-slower than single-threaded, invalidating the committed 2.22 ms/tick baseline). The alternative backends (JAX, Cython) and non-default modes (JASMINE energy/movement) carry real divergences and a guaranteed crash but are latent behind opt-in flags. Cross-referencing prior reviews: double regrowth, calf slot recycling, JAX food-floor + dispersal-heading divergence, Cython float32 crash + move divergence, and the server Stop->Run thread race all remain OPEN; the prior "prange write-race" concern was re-examined and found race-free (distinct-index writes), and the uncapped-simulation DoS did not resurface (prior 50K cap holds).

## The central finding: shipped ≠ validated

The DEPONS-parity work and the committed Kattegat baselines validated the **headless / Numba inline-kernel path** (`movement_module=None`). But the **interactive Shiny web app takes a different code path** for the same "DEPONS (Regulatory)" mode, and two bugs corrupt even the validated headless baselines. Findings are therefore split into **DEFAULT** (affects shipped behaviour and/or the committed baselines) vs **LATENT** (real, but gated behind an opt-in backend/mode that is off by default).


## CRITICAL

### #1 · CRITICAL · CONFIRMED · [DEFAULT] — Live Shiny web app runs an unvalidated simplified CRW (DEPONSCRWMovementVectorized), not the validated Numba/NumPy path
- **Where:** `src/cenop/movement/depons_crw.py:357`  ·  _backend-divergence_
- **What:** The controller always builds DEPONSCRWMovementVectorized for DEPONS mode and injects it into Population, so _update_movement delegates to compute_step. That module omits reference-memory food attraction (crwContrib+vt) and never calls _update_reference_memory, clips the turn angle instead of reject-and-redraw, drops the distance-dependent second angle loop, adds deterrence into displacement (inflating step past the max_mov cap), and uses a *0.3 dispersal turn instead of the SSLogis override. The validated inline kernels run only when movement_module is None (headless reference runner + tests).
- **Impact:** This is the DEFAULT path of the shipped web app: every interactive DEPONS run uses a model different from the one the Kattegat baselines and all DEPONS-parity work validated, so trajectories, spatial distribution, and disturbance response diverge from the validated reference. Compounds with rank 2 (400x deterrence added directly to displacement) to produce teleporting agents near noise sources.
- **Fix:** Return None from create_movement_module for DEPONS_CRW so Population uses the validated kernels, or make _update_movement ignore _movement_module in DEPONS mode; add a test asserting a controller-built DEPONS sim does not call DEPONSCRWMovementVectorized.compute_step.

### #2 · CRITICAL · CONFIRMED · [DEFAULT] — Vectorized turbine deterrence uses metre displacement (x cell_size ~400) instead of grid units
- **Where:** `src/cenop/agents/turbine.py:483`  ·  _depons-parity_
- **What:** calculate_aggregate_deterrence_vectorized builds the deterrence vector from metre displacements (dx_m = (porp - turb) * cell_size, then vec_x = dx_m * s * deter_coeff); DEPONS Porpoise.java:1290-1291 and CENOP's own scalar oracle use GRID displacement. deter_coeff = 0.012 matches DEPONS exactly, so the vector is ~400x too large, uncompensated.
- **Impact:** Confirmed 400x on the production turbine path (turbines != off). On the headless/validated path the vector only feeds heading composition, so effect is bounded over-avoidance (heading points ~directly away from turbines, distorting spatial distribution near wind farms). On the Shiny web path (rank 1's module adds deterrence straight to displacement) the 400x vector flings porpoises hundreds of cells/tick. Latent in count-based baselines because displacement does not kill.
- **Fix:** Build the vector from grid displacement (dx_m/cell_size), keeping metres only for distance/TL, so vectorized == scalar == DEPONS; strengthen the weak vectorized test (currently only asserts dx>1.0).


## HIGH

### #3 · HIGH · CONFIRMED · [DEFAULT] — Food logistic regrowth runs twice per day boundary (double regrow)
- **Where:** `src/cenop/core/simulation.py:568`  ·  _depons-parity_
- **What:** On each day boundary step() calls _daily_tasks() (which calls replenish_food, gated only on _cell_data is not None, NOT on scalar _porpoises) and then calls replenish_food AGAIN at 566-572. The inline comment claiming _daily_tasks only regrows scalar porpoises is factually wrong. replenish_food is non-idempotent logistic growth on the shared _food_value array the population eats from. (Merged: reported independently by the landscape-io and sim-scheduler passes.)
- **Impact:** Every simulated day of every run (headless + web, homogeneous + real ASC) regrows food twice (~96 vs 48 growth iterations), inflating post-depletion recovery -> energy intake -> survival/reproduction. Equilibrium carrying capacity K is unchanged (both calls clamp), but the transient spatial food availability that governs foraging is biased high; corrupts the committed Kattegat baselines vs DEPONS (FoodTask scheduled once/day). Empirically a depleted 0.01 cell reaches ~0.35 after two calls vs ~0.011 after one.
- **Fix:** Delete one of the two replenish_food calls (prefer removing the inline block at 566-572); add a regression test asserting replenish_food fires exactly once per day boundary.

### #4 · HIGH · CONFIRMED · [DEFAULT] — Newborn calves inherit dead agents' stale state on recycled array slots
- **Where:** `src/cenop/agents/population.py:2142`  ·  _correctness_
- **What:** _update_pregnancy_status places weaned calves into recycled inactive slots and resets only 12 fields; _check_mortality resets only active_mask. Calves inherit the dead occupant's is_dispersing/dispersal target, reference-memory (_stored_util/_pos_history/_mem_count), psm_buffer, CRW state (prev_log_mov/prev_angle), energy history, and _prev_x/_prev_y. Unconditionally, _update_reference_memory computes the calf's attraction vector from the dead agent's food memory (has_history = _mem_count>=2); conditionally the calf is treated as dispersing toward a dead agent's stale target.
- **Impact:** Silent wrong-result on the DEFAULT path across all backends in every multi-year run (the model's primary use case) where deaths and weaning co-occur continuously. Alters foraging movement, spatial distribution, and dispersal vs DEPONS, where each calf is a fresh Porpoise with empty memory and dispersal inactive.
- **Fix:** Fully reset all per-agent arrays for new_slots to newborn defaults (or zero all per-agent state at death in _check_mortality): is_dispersing=False, dispersal_*=0, prev_log_mov=0.8, prev_angle=10.0, zero memory rows/_mem_ptr/_mem_count/psm_buffer/energy history, _prev_x/_prev_y = calf position.

### #5 · HIGH · CONFIRMED · [DEFAULT] — UI defaults rS/rR = 0.04 (DEPONS 3.0) shadow the correct DEPONS 3.2 value 0.03
- **Where:** `src/cenop/ui/tabs/settings.py:399`  ·  _depons-parity_
- **What:** Energy-panel inputs param_rS/param_rR default to 0.04; create_simulation_from_inputs reads them verbatim (r_s=input.param_rS(), r_r=input.param_rR()), overriding the correct dataclass defaults 0.03. Verified DEPONS-3.2 parameters.xml has rS=rR=0.03; 0.04 is the stale DEPONS-3.0/master value. The values feed get_work/ref_mem_strength_table decay tables.
- **Impact:** Every default interactive DEPONS run decays satiation + reference memory ~33% faster than DEPONS 3.2, changing foraging movement, energy intake, and population trajectory away from the claimed-parity target. Confined to the web-UI path (headless runner constructs SimulationParameters directly and gets 0.03).
- **Fix:** Set settings.py:399/407 input defaults to 0.03 and correct the tooltips at settings.py:43-44.

### #6 · HIGH · CONFIRMED · [DEFAULT] — PSM preferred dispersal distance drawn from an unseeded RNG (breaks reproducibility)
- **Where:** `src/cenop/behavior/psm.py:80`  ·  _depons-parity_
- **What:** PersistentSpatialMemory.__init__ falls back to np.random.default_rng() when no rng is passed and immediately draws preferred_distance from it; production constructs every agent's PSM without an rng (population.py:214-215), so each porpoise's preferred distance comes from independent OS entropy and ignores random_seed. Reached in default DEPONS mode; preferred_distance is never reassigned.
- **Impact:** Two runs with an identical seed draw different preferred distances; once any agent disperses, targets and trajectories diverge and population output is non-reproducible, despite the rest of the sim being seeded. Breaks the reproducibility contract the reference baselines (seed=42) depend on; propagates to both NumPy and JAX backends.
- **Fix:** Thread the population's seeded generator into every PSM construction: PersistentSpatialMemory(w, h, rng=self.rng), including calf/birth creation.

### #7 · HIGH · CONFIRMED · [DEFAULT] — Stop/Reset then Run reactivates the old worker via shared stop_event.clear(), running two sims onto one queue
- **Where:** `src/cenop/server/main.py:1155`  ·  _concurrency_
- **What:** A single stop_event/result_queue/sim_thread is shared across runs. stop_simulation sets running=False synchronously while the worker is still alive (it re-checks stop_event only after a full 48-tick batch + up to 0.3s sleep). start_simulation guards only on running()==False, drains the queue, calls stop_event.clear() (re-arming the old thread), and starts a second daemon on the same queue. No join()/is_alive() anywhere; reset_simulation has the same hole.
- **Impact:** A normal Stop->Run (or Reset->Run) gesture during an active run spawns two threads putting interleaved updates on one queue: non-monotonic tick history, oscillating population/birth/death counters, flickering progress, plus an orphaned CPU-bound thread leaking a core until max_ticks. Confined to interactive UI state (does not corrupt on-disk outputs), but corrupts the live dashboard.
- **Fix:** Join or refuse-while-alive on the previous thread and use a fresh stop_event/result_queue per run; gate start on thread liveness, and never clear() a shared event an old thread may still observe.

### #8 · HIGH · CONFIRMED · [DEFAULT] — Six per-agent @njit(parallel=True) kernels oversubscribe threads (no cap) - net slower than 1 thread
- **Where:** `src/cenop/optimizations/kernels.py:193`  ·  _concurrency_
- **What:** Six kernels parallelize over the ~N-agent axis (reflect_boundaries, turn_position, depons_bmr_cost, compute_ve_total, compute_attraction, heading_position_reflect); nothing sets NUMBA_NUM_THREADS/set_num_threads, so each per-tick call forks a 28-way region over a few hundred elements. Measured whole-tick 6.7-8.5x slower at 28 threads than 1; per-kernel 2-14x slower. regrow_food_kernel (~1e5 cells) legitimately keeps parallel=True.
- **Impact:** Performance/stability only - prange writes distinct indices so outputs are bit-identical (no correctness/parity impact; the prior review's 'write-race' concern is not borne out). But every default multi-core run regresses and the committed 2.22 ms/tick baseline is invalid. One verifier rated this MEDIUM as a pure perf issue.
- **Fix:** Pin threads at startup to a small fixed count, or strip parallel=True from the six small-N kernels and keep it only on regrow_food_kernel; then re-measure per-tick baselines.

### #9 · HIGH · CONFIRMED · [LATENT] — JASMINE energy mode crashes: JASMINEEnergyModule missing compute_food_intake/compute_bmr_cost used by the tick
- **Where:** `src/cenop/physiology/energy_budget.py:490`  ·  _correctness_
- **What:** The production split energy path calls _energy_module.compute_food_intake/compute_bmr_cost, defined only on DEPONSEnergyModule. Selecting 'JASMINE (Research)' builds a JASMINEEnergyModule lacking both; since _energy_module is non-None the Cython fast path is skipped and the NumPy path raises AttributeError on tick 1. The combined compute_energy_update JASMINE implements is documented as no longer called by step(). Reproduced at runtime.
- **Impact:** Guaranteed immediate crash whenever the documented JASMINE energy mode is selected - the entire mode is unusable. Default DEPONS mode is unaffected (its module has the methods). Non-default path, hence HIGH not CRITICAL.
- **Fix:** Implement compute_food_intake/compute_bmr_cost on JASMINEEnergyModule, add base-class defaults delegating to compute_energy_update, or make the split path fall back to compute_energy_update when the split methods are absent.

### #15 · HIGH · CONFIRMED · [LATENT] — JAX dispersal heading uses a different formula (wrong constants, inverted distance, deterministic vs random)
- **Where:** `src/cenop/optimizations/jax_kernels.py:337`  ·  _backend-divergence_
- **What:** jax_heading_composition's dispersal override diverges from the NumPy/DEPONS reference on four points: max_angle hardcoded 120 vs psm_type2_random_angle 20; dist_percent from distance-remaining-to-target vs distance-traveled-from-start (inverted); SSLogis phi3=1.0 vs 0.6; deterministic sign-toward-target steering vs uniform-random delta in [-20,20]. Net: directed PSM-Type3-like steering instead of PSM-Type2 logistic-scaled random walk.
- **Impact:** Every dispersing agent follows a materially different trajectory under JAX (e.g. ~98deg deterministic corrective turns vs ~1.5deg random jitter early in dispersal), sharply changing spatial distribution and survival of dispersers. Latent: JAX is opt-in (use_jax default False) and slower than Numba, so default runs use the correct NumPy path.
- **Fix:** Match the reference: max_angle=psm_type2_random_angle, dist_percent from distance traveled from dispersal_start, divide the logistic by psm_log (0.6), and use a JAX uniform-random delta scaled by the logistic added to prev_step_heading.

### #18 · HIGH · CONFIRMED · [LATENT] — Cython food_grid typed float32 but step() passes float64 _food_value uncast -> ValueError on first tick
- **Where:** `src/cenop/optimizations/tick_cython.pyx:43`  ·  _api-misuse_
- **What:** cython_depons_post_crw declares float32 food_grid; the call site passes self.landscape._food_value (float64 for homogeneous/DEPONS-external landscapes) uncast; Cython buffer acquisition raises 'Buffer dtype mismatch, expected float but got double' at function entry. Reproduced empirically.
- **Impact:** Any config engaging the Cython gate (energy_module None + comm off + homogeneous/all-water) crashes on tick 1. Latent/off in production: communication defaults True and an energy module is always attached, so the gate never fires in the shipped app. Already documented as a Track B defect and guarded by an xfail(strict) equivalence test.
- **Fix:** Cast at the call site (np.ascontiguousarray(self.landscape._food_value, dtype=np.float32)) or store _food_value as float32 in the homogeneous/DEPONS-external constructors (preserving the in-place depletion write-back).

### #19 · HIGH · CONFIRMED · [LATENT] — Cython omits the post-move land/depth rollback -> agents step onto land, ~3.6-cell x/y divergence
- **Where:** `src/cenop/optimizations/tick_cython.pyx:131`  ·  _depons-parity_
- **What:** The Cython move section reflects at world boundaries and writes x/y but never checks destination depth; the reference _apply_positions rolls agents that land on land back to their pre-move position (gated on landscape is not None, NOT on _skip_land_avoidance), and even 'Homogeneous' landscapes have land edges at depth -10. Empirically the displacement math is bit-identical, so this missing rollback is the sole cause of the documented ~3.7-cell single-tick divergence (correcting the docs' 'formula/units bug' attribution).
- **Impact:** When the Cython path runs, boundary/edge agents diverge from the NumPy/Numba reference and DEPONS by up to a step distance and remain on land. Latent: Cython off in production (comm defaults True + energy module always present).
- **Fix:** In the .pyx move section, look up destination depth and restore pre-move x/y when <=0, matching _apply_positions (pass the depth grid into the kernel).

### #21 · HIGH · DISPUTED · [LATENT] — Cython mortality draws from global np.random instead of seeded self.rng
- **Where:** `src/cenop/optimizations/tick_cython.pyx:79`  ·  _correctness_
- **What:** rand_mort = np.random.random(n) uses the process-global RNG, not self.rng (which the reference _check_mortality uses). One verifier confirmed non-reproducibility + cross-backend RNG desync; the other refuted the reproducibility claim because Simulation.step() re-seeds np.random with base_seed+tick each tick (simulation.py:453) before the Cython branch, so same-seed runs are actually reproducible on the production driver - leaving only an expected, already-documented cross-backend stream divergence.
- **Impact:** DISPUTED. At minimum a cross-backend RNG stream divergence (why the equivalence test is single-tick); the within-run non-reproducibility hinges on whether step()'s per-tick global re-seed is in effect. Latent: Cython off in production.
- **Fix:** Pass rand_mort = self.rng.random(n) (or the rng) into the kernel from population.py and use it there instead of np.random.random.


## MEDIUM

### #10 · MEDIUM · CONFIRMED · [DEFAULT] — Interactive DEPONS energy path adds swimming + disturbance costs absent from DEPONS and the headless reference
- **Where:** `src/cenop/physiology/energy_budget.py:456`  ·  _depons-parity_
- **What:** DEPONSEnergyModule.compute_bmr_cost (and depons_bmr_cost_kernel) return bmr + activity(current_speed*0.0001*scaling) + disturbance(0.002*deter_mag*scaling); DEPONS E_USE_PER_KM=0.0 (no swimming term) and has no disturbance energy term. The server always creates the module, so the interactive DEPONS path uses it; the headless runner passes energy_module=None -> inline BMR-only path that matches DEPONS.
- **Impact:** Interactive 'DEPONS (Regulatory)' runs drain extra energy (~1% of BMR per moving porpoise, plus a disturbance surcharge when deterred) not present in DEPONS or the committed baselines - systematic and always-on in the regulatory-labeled mode, compounding over multi-year runs via exponential starvation survival. Confined to the web path.
- **Fix:** Drop the activity/disturbance terms (or gate them behind params.e_use_per_km defaulting to 0) in both compute_bmr_cost and the kernel so the module path equals the inline/headless path.

### #11 · MEDIUM · CONFIRMED · [DEFAULT] — Births/deaths undercounted (net-diff) and death-cause columns always 0 in output
- **Where:** `src/cenop/core/simulation.py:544`  ·  _correctness_
- **What:** In vectorized mode _porpoises is empty so scalar per-agent counters never fire; births/deaths derive only from net population delta (diff = current - prev), so births and deaths co-occurring in a tick record only the net. deaths_starvation/old_age/bycatch are never set and stay 0, yet are written to the DEPONS-format population output and the dashboard.
- **Impact:** Output/reporting fidelity only - the population trajectory itself stays correct (read from active_mask). In a stable ~2000-pop multi-year run births and deaths co-occur on essentially every day boundary -> systematic undercount of both columns; cause-of-death columns are always 0 despite real mortality. True per-cause mortality IS captured separately in population.death_causes / Mortality.txt.
- **Fix:** Have population_manager.step() expose true per-tick birth and per-cause death counts and accumulate those into state.births/deaths/deaths_* instead of inferring from population size.

### #12 · MEDIUM · CONFIRMED · [DEFAULT] — Turbine deterrence strength scaled by a logistic probability (deter_probabilistic default True); DEPONS turbine deterrence is deterministic
- **Where:** `src/cenop/agents/turbine.py:452`  ·  _depons-parity_
- **What:** With deter_probabilistic=True (default) and slope 0.2, turbine strength is multiplied by response_probability_from_rl(rl, threshold, slope); DEPONS applies full strength deterministically for turbines (only ships have a Bernoulli reaction draw). Only unit tests set it False; the committed Kattegat turbine baseline was generated with scaling ON.
- **Impact:** Default DEPONS-parity turbine runs weaken near-threshold turbine avoidance vs Java (~50% at the 152 dB threshold, ~17% at 160 dB). Active on every in-range turbine event when turbines are enabled (default off), most at the outer avoidance edge.
- **Fix:** Apply full strength when strength>0 on the turbine path (do not probability-scale), or default deter_probabilistic False; confirm whether the scaling is an intended JASMINE-only extension.

### #13 · MEDIUM · CONFIRMED · [DEFAULT] — Paused ships still produce deterrence; DEPONS suppresses deterrence while paused
- **Where:** `src/cenop/agents/ship.py:578`  ·  _depons-parity_
- **What:** The vectorized deterrence loop iterates get_active_ships() (filters only _is_active, no pause check). A paused ship stays _is_active with _prev==current, so all 30 interpolated sub-positions collapse to the stationary point and it deters porpoises for the whole pause. DEPONS Ship.deterPorpoise returns immediately when ticksStillPaused>0 (and when currentBuoyIdx<0).
- **Impact:** Kattegat ships.json has 811 pause>0 waypoints (NorthSea 1590); whenever a ship stalls it keeps emitting deterrence DEPONS suppresses, diverging ship-scenario trajectories/deter_strength during every pause. Bounded because paused buoys are often near ports/shallow water with lower porpoise density. Active only when ships are enabled.
- **Fix:** Exclude paused ships (ticks_paused>0 or current_buoy_idx<0) from the deterrence set, mirroring DEPONS deterPorpoise's pause gate.

### #14 · MEDIUM · CONFIRMED · [DEFAULT] — Preferred dispersal distance is hardcoded N(300;100) km, ignoring configured/DEPONS 3.2 N(350;100)
- **Where:** `src/cenop/behavior/psm.py:97`  ·  _depons-parity_
- **What:** generate_preferred_distance defaults mean=300.0; production PSM construction passes no mean/sd, and params.psm_dist_mean/psm_dist_sd (default 350, also parsed from the PSM_dist UI string whose default is N(300;100)) are set on SimulationParameters but never read by any consumer. So preferred_distance is always N(300,100) regardless of config. (Merged: the settings.py UI-default finding's 'shadowing' mechanism was refuted by verifiers - the dataclass 350 default is dead; the real bug is the psm.py hardcode plus missing plumbing.)
- **Impact:** Dispersal targets are ~50 km (~14-17%) closer than DEPONS 3.2 on every default run once dispersal fires; drives where animals settle (dispersal completes at >=0.95*target). Configured PSM_dist is silently ignored.
- **Fix:** Plumb params.psm_dist_mean/psm_dist_sd into PersistentSpatialMemory construction (and calf copy); set the UI default (settings.py:360) and controller parse-fallback (simulation_controller.py:96) to N(350;100).

### #16 · MEDIUM · CONFIRMED · [LATENT] — JAX food floor is u_min (0.001) instead of DEPONS ADD_ARTIFICIAL_FOOD 0.01
- **Where:** `src/cenop/optimizations/jax_kernels.py:631`  ·  _depons-parity_
- **What:** jax_eat_food floors depleted cells at min_food, and _step_jax passes params.u_min (default 0.001); all NumPy/Numba/scalar paths hardcode 0.01, matching DEPONS. u_min (the utility floor) is conflated with the artificial-food floor, which DEPONS keeps separate.
- **Impact:** Under JAX, grazed cells deplete 10x lower (0.001 vs 0.01); the floor persists across ticks (written back to _food_value), systematically lowering intake in busy patches -> higher starvation and lower equilibrium population vs the reference backend. Latent: JAX opt-in (default off).
- **Fix:** Pass 0.01 (the ADD_ARTIFICIAL_FOOD constant) as min_food to jax_tick_energy/jax_eat_food in _step_jax, not params.u_min.

### #17 · MEDIUM · CONFIRMED · [LATENT] — JASMINE simplified movement freezes step length (no stochastic step-length draw)
- **Where:** `src/cenop/movement/jasmine_physics.py:326`  ·  _correctness_
- **What:** Population always creates a base MovementState (not JASMINEMovementState), so JASMINEPhysicsMovement.compute_step falls through to _compute_simplified_step, which derives speed = 10^prev_log_mov/4 with no new draw; the write-back leaves prev_log_mov unchanged. Absent currents/deterrence, step length is pinned at 10^0.8/4 ~= 1.577 cells/tick forever. The full velocity-Verlet physics path is dead.
- **Impact:** In JASMINE movement mode every agent moves a near-constant distance each tick with no log-normal step-length variability, unlike DEPONS (which redraws presLogMov each step). Non-default, user-selectable mode.
- **Fix:** Draw a fresh step length each call in _compute_simplified_step (e.g. the DEPONS log-normal step model), or document/disable JASMINE simplified mode.

### #20 · MEDIUM · CONFIRMED · [LATENT] — batch_runner _run_parallel exception fallback references undefined 'progress' -> NameError instead of sequential fallback
- **Where:** `src/cenop/core/batch_runner.py:257`  ·  _correctness_
- **What:** _run_parallel has no progress parameter/attribute, yet its except handler calls self._run_sequential(combinations, progress=progress). Any exception in the ProcessPoolExecutor block (e.g. a non-picklable progress_callback) raises NameError instead of the advertised graceful fallback. The original exception IS logged first (exc_info), so it is not fully masked. Reproduced.
- **Impact:** The parallel batch path's graceful degradation is broken - a hard crash on the exact pickling failure the try/except exists to catch. Latent: parallel defaults False and no in-repo caller sets it; reachable only via the opt-in public API.
- **Fix:** Add progress: bool = True to _run_parallel and thread it from run(), or use a literal (progress=True) in the fallback call.

### #23 · MEDIUM · DISPUTED · [LATENT] — batch_runner _run_sequential references param_str unbound when progress=False with a progress_callback set
- **Where:** `src/cenop/core/batch_runner.py:224`  ·  _correctness_
- **What:** param_str is assigned only inside 'if progress:', but passed to config.progress_callback unconditionally; progress=False + a callback -> UnboundLocalError on the first iteration. DISPUTED: both verifiers confirmed the source defect but found no in-repo code ever sets progress_callback and all callers use progress=True, so the two triggering conditions never co-occur on any exercised path.
- **Impact:** Latent robustness gap reachable only via the public API (set progress_callback + run(progress=False)); no current path triggers it. Compounds the rank-20 parallel fallback bug (a fallback running with progress=False would hit this too).
- **Fix:** Compute param_str unconditionally before the 'if progress:' block (or default it to '').


## LOW

### #22 · LOW · CONFIRMED · [DEFAULT] — Renderers read live population_manager arrays on the session thread while the worker mutates them (no lock)
- **Where:** `src/cenop/server/main.py:1882`  ·  _data-race_
- **What:** age_histogram/energy_histogram/vital_stats_table read pm.active_mask/age/energy/is_female/with_calf and sim.get_statistics() directly every 0.2s poll while the background sim.step() mutates them in place, with no lock. Fixed-capacity arrays mean lengths always match, so no shape mismatch/crash.
- **Impact:** Downgraded to LOW (both verifiers judged MEDIUM overstated): read-only torn snapshots cause transient, self-correcting display inaccuracies (a mean over ~2000 agents off by ~0.05%, or a count off by +/-1) for a single frame. Scientific output is computed on the worker and delivered as immutable queue snapshots, so trajectory/parity are unaffected.
- **Fix:** Publish an immutable stats snapshot from the worker over result_queue (as porpoise positions already are) and render from that, or guard sim access with a lock held around sim.step().

### #24 · LOW · CONFIRMED · [LATENT] — User-supplied seeds shorter than replicates causes IndexError
- **Where:** `src/cenop/core/batch_runner.py:212`  ·  _correctness_
- **What:** When config.seeds is provided, __init__ stores it without length validation; both run paths index self.seeds[rep] for rep in range(replicates). Fewer seeds than replicates -> IndexError. Reproduced with seeds=[42,43], replicates=5.
- **Impact:** Latent public-API robustness gap; fails fast (no output corruption). Not reachable from the Shiny app or the internal convenience functions, which auto-generate correct-length seeds.
- **Fix:** Validate len(config.seeds) >= config.replicates (clear ValueError) or cycle/extend seeds deterministically.

### #25 · LOW · CONFIRMED · [DEFAULT] — AGE_DISTRIBUTION_FREQUENCY has 311 entries (54 ones) vs DEPONS' 312 (55 ones)
- **Where:** `src/cenop/parameters/demography.py:13`  ·  _depons-parity_
- **What:** One age-1 weight was dropped in the port of the DEPONS ageDistribution[] table; rng.choice over the array samples by index, so P(age=1)=54/311 vs 55/312 and every other age gets x/311 vs x/312. Used on the production init path (population.py:491, simulation.py:241). Transcription error, not intended.
- **Impact:** Slight bias in initial age structure at t=0 (mature/breeding-age and near-max-age fractions) relative to DEPONS; washes out over multi-year runs.
- **Fix:** Add the missing '1' to the young-adult block so the table has 312 entries with 55 ones, matching DEPONS.

### #26 · LOW · CONFIRMED · [DEFAULT] — jomopans_spl raises math domain error for ship length <= 0
- **Where:** `src/cenop/behavior/jomopans_spl.py:111`  ·  _numerical_
- **What:** 20*log10(length_m/L_REF) is computed without validating length_m, which flows unvalidated from ships.json/ship-file. A moving ship (speed != 0) with length <= 0 crashes get_source_level (called per in-range ship every tick on the default JOMOPANS path) with ValueError. Reproduced.
- **Impact:** Malformed-config crash on the default source-level path; requires a non-positive length in input (default is 100.0 and real DEPONS files use positive lengths).
- **Fix:** Clamp length_m to a positive minimum before the log, or reject non-positive length at load time in load_from_json / Ship construction.

### #27 · LOW · CONFIRMED · [LATENT] — DISTURBED->FORAGING recovery gate (recovery_ticks) is inert unless JASMINE memory is active
- **Where:** `src/cenop/behavior/hybrid_fsm.py:177`  ·  _correctness_
- **What:** The recovery rule needs time_since_disturbance > recovery_ticks, built from last_disturbance_tick, which only JASMINEMemoryModule.record_disturbance writes; in DEPONS memory mode (default) it stays -9999 so time_since is permanently 9999 and the gate is always bypassed (agents leave DISTURBED the first tick deterrence drops).
- **Impact:** In default all-DEPONS mode DISTURBED is display-only (no dynamics effect). Under a JASMINE-energy + DEPONS-memory override, losing the recovery window drops a real elevated-activity energetic cost (DISTURBED multiplier 2.0). A cross-subsystem coupling defect.
- **Fix:** Maintain last_disturbance_tick independent of memory mode (update whenever deter_strength>threshold, or in the FSM itself).

### #28 · LOW · CONFIRMED · [LATENT] — JAX tick path never refreshes _active_idx, so reference memory is updated for dead slots
- **Where:** `src/cenop/agents/population.py:2035`  ·  _correctness_
- **What:** _active_idx is refreshed only in the Numba/NumPy/Cython paths; _step_jax never assigns it, so it stays arange(count). _update_reference_memory then stores food and advances _mem_ptr/_mem_count for dead slots. Active-agent outputs are gated on mask, so they are not corrupted.
- **Impact:** Latent (JAX off by default); wasted work on dead slots plus compounding the calf-recycling memory bug (rank 4) by changing which stale memory a recycled slot carries. No active-trajectory corruption.
- **Fix:** Set self._active_idx = np.flatnonzero(self.active_mask) at the start of _step_jax, mirroring the Numba path.

### #29 · LOW · DISPUTED · [LATENT] — Cython reflected (boundary-bounced) agents keep their outward heading; reference recomputes it inward
- **Where:** `src/cenop/optimizations/tick_cython.pyx:119`  ·  _depons-parity_
- **What:** Boundary reflection flips the position but leaves heading unchanged; the reference recomputes heading from the sign-flipped displacement (DEPONS forward()). A bounced Cython agent keeps pointing outward and re-bounces. DISPUTED: one verifier deemed it effectively unreachable (Cython gated off in production, and even if engaged the float32 crash of rank 18 fires first), invisible to the single-tick equivalence test.
- **Impact:** Latent multi-tick trajectory divergence confined to edge-bouncing agents; Cython off in production and preceded by an earlier crash. Zero effect on the shipped Numba/NumPy/JAX paths.
- **Fix:** After reflecting nx/ny, recompute heading = atan2(flipped_ddx, flipped_ddy) mod 360, mirroring _handle_land_avoidance.

### #30 · LOW · CONFIRMED · [DEFAULT] — build_turbine_blade_layer ignores client_animated; blades never rotate while the rAF loop burns CPU
- **Where:** `src/cenop/server/map_layers.py:309`  ·  _api-misuse_
- **What:** client_animated is never referenced; angle is baked statically (rotation defaults 0) with getAngle='@@d.angle'; the layout.py requestAnimationFrame handler increments window._cenopBladeRotation and calls setProps every frame but never feeds it into any angle accessor. BLADE_ANIMATION_JS/STOP constants are dead code.
- **Impact:** Cosmetic/perf: enabling the default-on blade animation re-invokes setProps ~60/s for zero visual rotation. No simulation impact.
- **Fix:** Implement client_animated (bind getAngle to a JS expression reading window._cenopBladeRotation with an updateTriggers/animation tick) or drop the parameter, dead constants, and the effect-free rAF loop.


## Coverage gaps / recommended follow-ups

- Multi-year runtime drift: findings are static-analysis + single-tick empirics; the cumulative population-trajectory divergence from double food regrowth, calf slot recycling, and the extra energy costs was not quantified against a running DEPONS Java reference over full multi-year runs.
- End-to-end DEPONS Java cross-validation (Track A) was not performed — all parity verdicts rest on code reading against parameters.xml/Porpoise.java, not a live Java oracle; a headless Java run would confirm/deny the compounded biases.
- GPU-only JAX behavior: the JAX dispersal-heading, food-floor, and _active_idx findings were verified on CPU/logic; GPU-specific numerics, the fixed-iteration rejection sampling, and np<->jax transfer effects on real hardware were not exercised.
- Concurrency was audited statically only: the Stop/Run thread race and renderer read races were not stress/fuzz-tested (rapid Stop->Run gestures, ThreadSanitizer-equivalent) to measure real corruption frequency; social_accumulate and other parallel=True kernels were not exhaustively checked for index aliasing beyond the six confirmed race-free ones.
- The batch_runner ProcessPoolExecutor parallel path was never run: pickling of Simulation/BatchRunner objects and the actual fallback-crash sequence were only reasoned about, not executed.
- Landscape/ASC I/O edge cases (malformed headers, NODATA handling in real ASC/food/depth files, monthly-file selection) and broader uploaded-file (ships.json/turbine) fuzzing beyond the jomopans length<=0 case were only lightly touched.
- Performance baselines are stale: no re-measured per-tick timings after accounting for thread oversubscription, and no measurement of the interactive Shiny path (which uses a different movement module than the profiled headless path).
- Cross-finding interactions were not systematically explored beyond the confirmed compound of the simplified web CRW (rank 1) with the 400x turbine deterrence (rank 2); other mode/backend/parameter combinations (e.g. JASMINE movement + JASMINE energy + DEPONS memory) may hide further coupling defects.
