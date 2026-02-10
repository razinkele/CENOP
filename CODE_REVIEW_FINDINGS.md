# CENOP-JASMINE Deep Code Review: Inconsistencies & Optimizations

**Date:** 2025-06-19  
**Scope:** Full review of `cenop-jasmine/src/cenop/` (54 Python files, ~12,000+ LOC)

---

## Executive Summary

The codebase is well-structured with clean separation of concerns (SoA population, modular movement/energy/behavior systems, DEPONS/JASMINE dual-mode architecture). However, this review identified **5 critical bugs**, **12 high-severity issues**, and **20+ medium/low issues** across inconsistencies, dead code, and performance bottlenecks.

---

## 1. CRITICAL BUGS (Runtime Failures)

### 1.1 `porpoise.py` L625 — `params.beta` Does Not Exist (**AttributeError**)

The scalar `Porpoise._check_mortality()` method references `params.beta` which does not exist on `SimulationParameters`. The correct attribute is `params.x_survival_const` (default 0.15). Additionally, `m_mort_prob_const` is hardcoded to `1.0` (line 623) instead of using `params.m_mort_prob_const` (default 0.5).

**Impact:** The scalar Porpoise class (used in legacy `_daily_tasks()` path) will crash on any mortality check.

**Fix:** Replace `params.beta` → `params.x_survival_const` and use `params.m_mort_prob_const`.

> **Note:** The vectorized `PorpoisePopulation._check_mortality()` (population.py L1098) correctly uses `getattr(self.params, 'x_survival_const', 0.15)` — this bug only affects the legacy scalar path.

### 1.2 `server/main.py` L166-167 — Duplicate `return rows`

```python
    return rows
    return rows  # Dead code — second return is unreachable
```

Harmless but indicates copy-paste error.

### 1.3 `server/main.py` L285-301 — Lithuania Turbines Missing

The `turbine_selector()` function defines its own local `LANDSCAPE_TURBINE_COMPATIBILITY` dict that is **missing the "Lithuania" key**. Since the sidebar defaults to `selected="Lithuania"`, users selecting Lithuania get a fallback `{"off": "No turbines"}` and cannot access the Curonian Nord wind park scenarios (`CuronianNord_35_15MW`, `CuronianNord_60_10MW`) that are defined in `ui/sidebar.py`.

**Fix:** Add Lithuania to the server's dict, or better yet, unify the two definitions.

### 1.4 Duplicate `LANDSCAPE_TURBINE_COMPATIBILITY` — Type Mismatch

| Location | Type | Landscapes |
|---|---|---|
| `ui/sidebar.py` L12-18 | `dict[str, list[str]]` | Includes Lithuania |
| `server/main.py` L285-301 | `dict[str, dict[str, str]]` | Missing Lithuania |

The server version uses `dict` values (for Shiny `input_select` labels), while the sidebar uses `list` values. Two sources of truth with different types and different content guarantees they'll diverge.

### 1.5 Mortality Formula Wrong in `porpoise.py` (L625-627)

```python
yearly_survival = 1 - (1.0 * exp(-energy * params.beta))  # params.beta = 20.0 (sound propagation!)
```

Using `beta_hat=20.0` (the sound propagation spreading factor) instead of `x_survival_const=0.15` makes `exp(-energy * 20)` vanish for any `energy > 0.5`, so **no porpoise ever dies of starvation** through this code path.

---

## 2. HIGH-SEVERITY ISSUES (Wrong Behavior)

### 2.1 Ship Scalar vs Vectorized Deterrence Divergence

- **Scalar** (`ship.py` `calculate_aggregate_deterrence`): Uses `ShipDeterrenceModel` with probabilistic deterrence (`random() < prob`)
- **Vectorized** (`ship.py` `calculate_aggregate_deterrence_vectorized`): Uses simple RL-threshold check, **skipping the probabilistic model entirely**

For production runs (which use vectorized), ships will behave differently than expected from the scalar model.

### 2.2 Turbine `should_deter()` vs `get_received_level()` Inconsistency

- `should_deter()` (turbine.py L148): Uses `self.impact` directly as source level
- `get_received_level()` (turbine.py L107): Uses `self.noise.get_source_level(is_construction)` with dB conversions

Two code paths yield different deterrence distances for the same turbine.

### 2.3 FSM Rule Ordering Race Condition (`hybrid_fsm.py` L149-156)

In `_update_depons()` and `_update_jasmine()`:
1. Rule 1 sets agents to `DISTURBED`
2. Rule 2 immediately checks DISTURBED agents for recovery **in the same tick**

If `time_since_disturbance` hasn't been reset, newly-disturbed agents can instantly recover. Similar issue in JASMINE mode where Rules 4-6 read modified state from earlier rules.

### 2.4 Heading Convention Mismatch Across Modules

| Module | Convention | 0° Direction |
|---|---|---|
| `memory.py` | `arctan2(dy, dx)` | East (mathematical) |
| `psm.py` | `arctan2(dx, dy)` | North (navigation) |
| `population.py` | `arctan2(dx, dy)` | North (navigation) |
| `dispersal.py` | `heading % 360` | Unspecified |

`memory.py` uses the wrong convention for DEPONS-style headings, causing `get_memories_in_direction()` angle comparisons to be incorrect.

### 2.5 Memory Eviction Strategy Mismatch (`memory.py` L78)

`RefMem.add()` uses FIFO eviction (`self._entries.pop(0)`) — removing the **oldest** entry. DEPONS Java removes the **weakest** entry. Strong recent memories can be lost if they were the first added.

### 2.6 `EnergyState` Fields Never Updated (`energy_budget.py`)

These fields are allocated but **never written to** by either energy module:
- `fat_reserve` (L68)
- `distance_traveled` (L71)
- `time_under_disturbance` (L74)
- `days_in_negative_balance` (L78)

`get_fitness_metrics()` returns stale initial values for all JASMINE fitness tracking.

### 2.7 `output_writer.py` L356 — Hardcoded Age Threshold

`_detect_mortality()` uses `age >= 24` for old-age classification but `SimulationParameters.max_age = 30.0`. Deaths at ages 24-30 are incorrectly classified. Bycatch deaths are classified as "unknown".

### 2.8 `dispersal.py` — Liskov Substitution Violation

`PSMType2Dispersal.start_dispersal(rng, target_heading, start_position)` has a different signature than base class `DispersalBehavior.start_dispersal(rng)`. Callers using the abstract interface won't pass extra args.

### 2.9 Hybrid FSM Mode Bug (`hybrid_fsm.py` L258-264)

`_update_hybrid()` checks `np.any(disturbance present)` then calls either `_update_depons()` or `_update_jasmine()` for **ALL agents**. If even one agent is disturbed, all agents switch to DEPONS rules. Should apply per-agent mode selection.

### 2.10 `simulation.py` — `_daily_tasks()` Uses Legacy Scalar Path (L470-520)

The `_daily_tasks()`, `_yearly_tasks()` methods iterate over `self._porpoises` (the legacy scalar list), which is empty when using `PorpoisePopulation`. These methods effectively do nothing:
- No food replenishment via `cell_data.replenish_food()`
- No yearly aging of scalar porpoises
- No calf weaning

The vectorized `PorpoisePopulation.step()` handles aging/reproduction internally, but **`cell_data.replenish_food()` is never called** from the vectorized path.

### 2.11 `cell_data.py` — `eat_food_vectorized` Race Condition (L220-240)

When multiple agents occupy the same cell, food is read once for all agents but consumed independently. The last-write-wins pattern causes over-consumption.

### 2.12 `app_ui` Created at Import Time (`layout.py` L261)

`app_ui = create_app_ui()` executes at module import time, constructing the entire UI tree including sidebar and all tabs. Side effects in constructor functions will execute prematurely.

---

## 3. MEDIUM-SEVERITY ISSUES

### 3.1 Dead Code

| File | Location | Description |
|---|---|---|
| `sound.py` L16-22 | `NoiseSourceType` enum | Defined, never used |
| `sound.py` L12 | `Optional` import | Never used |
| `psm.py` L7-8 | `field`, `Dict`, `List` imports | Never used |
| `states.py` L37-55 | Most `StateTransition` enum members | Never matched |
| `states.py` L128 | `transition_counts` dict | Never read/written |
| `memory.py` L3 | `field` import | Never used |
| `disturbance_memory.py` L48-54 | `DisturbanceEvent` dataclass | Never instantiated |
| `disturbance_memory.py` L287-288 | `DEFAULT_DECAY_HALF_LIFE`, `HABITUATION_RATE` | Never used |
| `energy_budget.py` L422-424 | `jasmine_body_mass_scaling` etc. | Always use defaults |
| `porpoise.py` L96 | `stored_util_list` | Never populated |
| `porpoise.py` L66-67 | `energy_consumed_daily`, `food_eaten_daily` | Set, never read |
| `porpoise.py` L103 | `dispersal_type` | Set, never read |
| `cell_data.py` L322-378 | `load_bathymetry_from_asc()` | Duplicates `loader._load_asc()` |
| `chart_helpers.py` L159-201 | `create_map_figure()` | Superseded by `create_pydeck_map()` |
| `profiler.py` L115-130 | `profile_detailed()` | Profiles synthetic ops, not sim |
| `agents/base.py` L114 | `if heading < 0` guard | Unreachable (`x % 360` always ≥ 0) |
| `output_writer.py` L148-149 | `_mortality_events`, `_dispersal_events` | Never used |

### 3.2 Duplicate Constants

| Constant | Location 1 | Location 2 |
|---|---|---|
| `MEM_CELL_SIZE = 5` | `psm.py` | `disturbance_memory.py` |
| `LANDSCAPE_TURBINE_COMPATIBILITY` | `ui/sidebar.py` | `server/main.py` (different types!) |
| `load_bathymetry_from_asc()` | `cell_data.py` | `loader.py` (`_load_asc`) |

### 3.3 Distance/Unit Inconsistencies

| Module | Unit System |
|---|---|
| `sound.py` `SoundPropagationParams.max_deter_distance` | km (50.0) |
| `sound.py` `calculate_deterrence_distance` default | meters (50000.0) |
| `psm.py` | grid cells with `cell_size` parameter |
| `dispersal.py` | km for `dist_mean` |
| `population.py` | mixed (cell units, meters, km) |

### 3.4 `simulation_controller.py` — Misleading Energy Proxy (L162-166)

```python
avg_food_eaten = avg_energy  # Actually mean energy RESERVES, not food intake
avg_energy_expended = 4.5 * 48  # Constant 216.0 — adds no information
```

The Energy Balance chart is effectively meaningless.

### 3.5 `disturbance_memory.py` L316-319 — Fragile Grid Initialization

Grid dimensions are inferred from agent positions via `_ensure_grid_dims`. If agents move beyond initial bounds later, the grid stays undersized and `_position_to_cell` silently clamps to boundary cells.

### 3.6 `server/__init__.py` — Misleading `__all__`

Exports `["main", "reactive_state", "simulation_controller", "renderers"]` but imports nothing. `from cenop.server import *` yields no names.

---

## 4. PERFORMANCE OPTIMIZATIONS

### 4.1 Critical Performance Issues

| File | Location | Issue | Estimated Impact |
|---|---|---|---|
| `disturbance_memory.py` L340-360 | `decay_memory()` | O(N×M) pure Python loops over agents × cells | **Very High** at scale |
| `disturbance_memory.py` L375-430 | `compute_avoidance()` | Triple-nested Python loop | **Very High** |
| `output_writer.py` — statistics | All write methods | Python loops + `flush()` per write | High for N>5000 |
| `psm.py` L157-174 | `get_target_cell_for_dispersal()` | Per-cell `sqrt` in Python loop | Moderate |
| `chart_helpers.py` L208-578 | `create_pydeck_map()` | Regenerates full HTML (370+ lines) per call | High for frequent updates |

### 4.2 Recommended Optimizations

1. **Vectorize `disturbance_memory.py`**: Replace per-agent Python dicts with NumPy sparse matrices. Use `scipy.sparse` or flat arrays with agent offsets for batch decay/avoidance computation.

2. **Vectorize `psm.py` target selection**: Extract all cell positions into NumPy arrays for batch distance computation (already done for the buffer in `population.py`).

3. **Use `np.bincount` in `BehaviorStateVector.get_statistics()`**: Replace loop over `BehaviorState` enum with single `np.bincount` call.

4. **Use lookup arrays for FSM multipliers**: Replace `hybrid_fsm.py` `get_speed_multiplier()` / `get_energy_cost_multiplier()` per-state loops with `LOOKUP[state_vector.state]` fancy indexing.

5. **Binary search optimization in `calculate_deterrence_distance()`**: Replace 50-iteration bisection with `scipy.optimize.brentq` or Newton's method.

6. **Buffer pydeck map HTML**: Cache the iframe template and only update the data payload via `postMessage` instead of regenerating the full HTML document. The computed `data_hash` (L249) is currently dead code.

7. **Batch ASC file loading**: `LandscapeLoader._load_monthly()` calls `_load_asc()` 12 times. Use `np.loadtxt` with proper skip_header instead of Python `for line / split()` loops.

8. **Pre-compute `_ensure_loaded()` gate**: All scalar getters (`get_depth`, `get_food_level`, etc.) call `_ensure_loaded()` on every access. Set a flag and check it once.

9. **`PorpoisePopulation.__init__`**: Creates `_psm_instances` list of N `PersistentSpatialMemory` objects (line ~170). These are mostly unused since the vectorized PSM buffer replaced them — only `preferred_distance` is read. Replace with a simple `np.ndarray` of preferred distances.

10. **`simulation.py` `_daily_tasks()`**: The legacy scalar daily/yearly task methods iterate empty lists. Either wire them to the vectorized population or remove to avoid confusion.

---

## 5. CROSS-REFERENCE WITH `cenop/` BRANCH FIXES

Fixes applied to `cenop/` that may need porting:

| Fix | `cenop/` Status | `cenop-jasmine/` Status |
|---|---|---|
| `append_with_limit()` in `reactive_state.py` | ✅ Added | ❌ Missing (but not imported/used by server/main.py) |
| Config constants (`LANDSCAPE_BOUNDS`, etc.) | ✅ Added to `config.py` | ⚠️ Not needed — imported from `ui/sidebar.py` directly |
| `list-vs-dict` turbine fix (`main.py` L298) | ✅ Fixed | ⚠️ Different bug — server uses local dict, but Lithuania is missing |
| `time_manager.advance()` in `simulation.step()` | ✅ Added | ✅ Already present |

---

## 6. ARCHITECTURE OBSERVATIONS

### 6.1 Dual Implementation Pattern

Many subsystems have both a legacy scalar implementation and a vectorized/modular replacement:
- `Porpoise` class vs `PorpoisePopulation`
- `_daily_tasks()` vs `_check_mortality()`/`_handle_reproduction()` in population
- `RefMem` vs vectorized PSM buffer
- `DEPONSEnergyModule` vs inline energy in population

The scalar implementations contain bugs (params.beta) while vectorized versions are correct. Consider marking the scalar path as deprecated or removing it.

### 6.2 PSM-Type2 Duplicate

Both `psm.py` (`PSMDispersalType2`) and `dispersal.py` (`PSMType2Dispersal`) implement the same SSLogis-based dispersal with slightly different APIs. One should be removed.

### 6.3 Memory Eviction Strategies

Three different memory modules use three different eviction strategies:
- `memory.py` (RefMem): FIFO (oldest removed first)
- `psm.py`: Dict-based (never evicts)
- `disturbance_memory.py`: Threshold-based pruning

No shared pattern or documented rationale for the differences.

---

## 7. PRIORITY FIX LIST

| # | Severity | File | Fix Description |
|---|---|---|---|
| 1 | **CRITICAL** | `porpoise.py` L623-627 | Fix `params.beta` → `params.x_survival_const`, use `params.m_mort_prob_const` |
| 2 | **CRITICAL** | `server/main.py` L285-301 | Add Lithuania to turbine compatibility dict |
| 3 | **CRITICAL** | `server/main.py` L167 | Remove duplicate `return rows` |
| 4 | **HIGH** | `ship.py` L398-440 | Align vectorized deterrence with probabilistic model |
| 5 | **HIGH** | `hybrid_fsm.py` L149-156 | Fix rule ordering race condition |
| 6 | **HIGH** | `memory.py` L78 | Change FIFO to weakest-entry eviction |
| 7 | **HIGH** | `simulation.py` L470-520 | Wire `replenish_food()` to vectorized path |
| 8 | **MEDIUM** | `energy_budget.py` L68-78 | Implement or remove dead `EnergyState` fields |
| 9 | **MEDIUM** | `output_writer.py` L356 | Use `params.max_age` instead of hardcoded 24 |
| 10 | **MEDIUM** | `disturbance_memory.py` | Vectorize `decay_memory()` and `compute_avoidance()` |
| 11 | **MEDIUM** | `memory.py` L126-130 | Fix heading convention (math→navigation) |
| 12 | **MEDIUM** | Unify `LANDSCAPE_TURBINE_COMPATIBILITY` | Single source of truth in shared location |

---

*Generated by automated deep code review. All line numbers reference `cenop-jasmine/src/cenop/` unless otherwise noted.*
