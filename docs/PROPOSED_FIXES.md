# Proposed Fixes & Optimizations — CENOP (CENOP-JASMINE branch)

> **Codebase**: ~18 500 lines of Python across `src/cenop/` (42 modules)
> **Test suite**: 320 passed, 1 skipped, 1 warning (~3 m 38 s)
> **Runtime**: Python 3.13.11 in micromamba `shiny` environment (`/opt/micromamba/envs/shiny`)
> **Branch**: `CENOP-JASMINE` (PR #2 — Merge JASMINE capabilities, Phases 1-5)

---

## 1. Critical Bugs

### 1.1 Dead code — duplicate `return rows` (server/main.py:167)

`src/cenop/server/main.py` lines 166-167 contain **two consecutive `return rows`** statements.
The second `return` is unreachable dead code. It should be removed.

```python
# Line 166-167
    return rows
    return rows       # <-- dead code, delete this line
```

### 1.2 Dead legacy code path — `_daily_tasks()` operates on empty list

`src/cenop/core/simulation.py:527-556` — `_daily_tasks()` iterates over `self._porpoises`, which is always `[]` when. the `population_manager` is active (which is the only current code path). This means:

- **Weaning / calf creation** via the legacy `Porpoise` class never executes.
- **Dead porpoise removal** via list comprehension never fires.
- **Food replenishment** via `self._cell_data.replenish_food()` never runs through this path (it is handled separately by the population manager).

These ~30 lines are dead code that should either be removed or explicitly wired to the vectorized population manager.

### 1.3 Entry point uses stale top-level `server/` directory

`app.py` (the Shiny entry point) imports `from server.main import server` — this pulls from the **top-level `server/main.py`** (1 516 lines, older API), NOT from `src/cenop/server/main.py` (2 267 lines, canonical). This means the app may run an outdated server depending on PYTHONPATH ordering.

---

## 2. High Priority — Repository Hygiene

### 2.1 Three duplicate server code trees

| Path | Lines | Status |
|---|---|---|
| `src/cenop/server/main.py` | 2 267 | **Canonical (package)** |
| `server/main.py` | 1 516 | Stale copy (different imports, older API) |
| `cenop/src/cenop/server/main.py` | 106 | Stale skeleton |

Similarly, `server/simulation_controller.py` (329 lines) and `server/reactive_state.py` (69 lines) are duplicated at both levels. The top-level `server/` and `cenop/` directories should be removed and `app.py` should import from `cenop.server.main`.

### 2.2 Stale top-level files to remove or archive

| File | Reason |
|---|---|
| `app_backup.py` | Old backup with ~15 print statements |
| `test_app.py`, `test_energy.py`, `test_minimal.py`, `test_shiny.py` | Ad-hoc scripts, not pytestcompatible, duplicate tests already in `tests/` |
| `minimal_app.py` | Development scaffold |
| `cenop/` directory (entire tree) | Stale copy of package |

### 2.3 85 `print()` debug statements in source (not tests)

Breakdown by file:

| File | Count |
|---|---|
| `src/cenop/server/main.py` | 34 |
| `src/cenop/core/profiler.py` | 21 |
| `src/cenop/server/simulation_controller.py` | 9 |
| `src/cenop/core/batch_runner.py` | 8 |
| `src/cenop/core/simulation.py` | 4 |
| `src/cenop/landscape/cell_data.py` | 4 |
| `src/cenop/agents/ship.py` | 3 |
| `src/cenop/core/time_manager.py` | 1 |
| `src/cenop/core/output_writer.py` | 1 |

**Action**: Replace all with `logger.debug()` / `logger.info()` / `logger.warning()` calls using `logging.getLogger(__name__)`.

### 2.4 Only 4 of 42 modules have logging configured

Files **with** logging: `config.py`, `server/main.py`, `server/simulation_controller.py`, `agents/population.py`.

All other 38 source modules have **zero** logging. Add `logger = logging.getLogger(__name__)` to every module and replace `print()` calls.

### 2.5 33 broad `except Exception` catches

| File | Count | Notes |
|---|---|---|
| `src/cenop/server/main.py` | 24 | Most silently log then continue |
| `src/cenop/agents/population.py` | 5 | Fallback for Numba/SciPy failures |
| `src/cenop/core/simulation.py` | 2 | |
| Other | 2 | |

Many of these swallow errors silently (e.g. `except Exception: pass` at `population.py:905`). Each should be:
- Narrowed to the specific exception type expected.
- Always logged with `logger.exception(...)` to preserve tracebacks.
- The bare `pass` at line 905 (adaptive recompute interval) is particularly dangerous — it silently hides any bug in the movement tracking EMA calculation.

---

## 3. Thread Safety & Reproducibility

### 3.1 ~~Global `np.random.seed()` is not thread-safe~~ ✅ FIXED

Replaced `np.random.seed()` with `np.random.default_rng(seed)` throughout:
- `Simulation.__init__` and `step()` create/re-create `self.rng`
- `PorpoisePopulation` accepts and uses `rng` parameter (all 25+ call sites converted)
- Movement modules (`DEPONSCRWMovement`, `JASMINEPhysicsMovement`, `MovementState.create()`) accept `rng`
- `PersistentSpatialMemory` and `PSMDispersalType2` accept `rng`
- `batch_runner.py` uses `np.random.default_rng().integers()` for seed generation
- Legacy `porpoise.py` and `profiler.py` calls left as-is (dead/benchmark code)

---

## 4. Architecture & Code Organisation

### 4.1 `agents/population.py` is too large (1 875 lines)

This single file handles: array initialization, movement, energy, PSM updates, dispersal targeting, reproduction, mortality, social communication (KDTree neighbours), behavioral FSM integration, disturbance memory, and DataFrame export.

**Suggested split**:
- `agents/population_core.py` — Array allocation, lifecycle (birth/death/aging)
- `agents/population_movement.py` — CRW steps, heading updates, boundary enforcement
- `agents/population_social.py` — KDTree neighbour finding, social vector accumulation
- `agents/population_psm.py` — PSM buffer updates, dispersal targeting
- `agents/population.py` — Thin orchestrator that delegates to the above

### 4.2 `server/main.py` is too large (2 267 lines)

All Shiny render callbacks, data loading, map rendering, noise propagation, and simulation polling live in one file.

**Suggested split**:
- `server/data_loaders.py` — Landscape table building, depth/foraging/ship/turbine data loading
- `server/map_renderers.py` — Porpoise map, depth overlay, foraging overlay, noise overlay
- `server/chart_renderers.py` — Population plot, energy plot, behavior plot, mortality plot
- `server/simulation_loop.py` — `run_simulation_loop()` and polling logic
- `server/main.py` — Wire everything together

### 4.3 Dual PSM storage wastes memory

`PorpoisePopulation` maintains **both**:
1. `psm_buffer` — np.ndarray shape `(count, rows, cols, 2)` — used for all PSM updates.
2. `_psm_instances` — `List[PersistentSpatialMemory]` — only used for `preferred_distance` access (3 call sites).

The `preferred_distance` property could be computed directly from `psm_buffer` data, eliminating the per-agent object list and saving significant memory for large populations.

### 4.4 `_compute_social_vectors` defined inside `__init__` then bound via `__get__`

At `population.py:~275`, `_compute_social_vectors` is defined as a local function inside `_initialize_population()` then attached to the instance via `types.MethodType`. This is an unusual pattern that:
- Makes the method invisible to IDE autocompletion and static analysis.
- Prevents easy unit testing in isolation.
- Could be a regular method with no change in behavior.

### 4.5 `simulation.porpoises` property creates `SimpleNamespace` objects per call

`simulation.py:670` creates a fresh `SimpleNamespace` for every agent on every access. This is extremely slow for per-tick monitoring but is labelled as legacy compatibility. Consider caching the result per tick or deprecating the property.

---

## 5. Performance Optimizations

### 5.1 PSM memory compression

For large worlds the `psm_buffer` shape `(N, rows, cols, 2)` can be enormous. Options:
- Increase `psm_cell_size` to reduce grid dimensions.
- Switch to sparse representation (dict-of-keys or COO) for agents that visit few cells.
- Use bounded-size hash maps per agent.
- Benchmark current memory use first (add a `psm_memory_bytes` property).

### 5.2 `disturbance_memory.py` uses per-agent dicts (not vectorized)

`DisturbanceMemory.memory_grids` stores a `Dict[int, Dict[Tuple[int,int], MemoryEntry]]` — one Python dict per agent. For large populations this creates GC pressure and prevents vectorization.

Consider a pre-allocated `(N, grid_rows, grid_cols)` array similar to `psm_buffer`, with a parallel decay timestamp array.

### 5.3 Additional Numba targets

The `optimizations.py` module already has `accumulate_psm_updates` and `vectorized_distance`. Additional candidates:
- Social neighbour accumulation loop (currently NumPy `bincount` fallback)
- CRW step kernel (trig operations over all agents)
- Energy budget update loop

### 5.4 Consistent dtypes

Some arrays are created without explicit dtype so they default to `float64`. Standardise on `float32` where full precision is not needed (positions, headings, energy) to halve memory and improve cache/SIMD performance.

### 5.5 Use `np.hypot` where appropriate

`optimizations.vectorized_distance` and several other locations compute `np.sqrt(dx**2 + dy**2)`. Replace with `np.hypot(dx, dy)` for numerical stability and clarity.

---

## 6. Packaging & CI

### 6.1 `pyproject.toml` classifiers missing Python 3.13

Classifiers list 3.10-3.12 but the runtime environment uses 3.13. Add:

```toml
"Programming Language :: Python :: 3.13",
```

Also update `[tool.black]` `target-version` and `[tool.mypy]` `python_version` accordingly.

### 6.2 No CI workflow exists

Add a GitHub Actions workflow that:
1. Uses `mamba-org/setup-micromamba` to create the `shiny` environment.
2. Runs `pip install -e .`
3. Runs `pytest -q`
4. Optionally runs `ruff check` and `mypy`.
5. Tests against Python 3.10, 3.12, and 3.13.

### 6.3 No `environment.yml` for the micromamba environment

The project provides `requirements.txt` and `pyproject.toml` but no `environment.yml` or `conda` spec to reproduce the exact micromamba environment. Add one for CI and onboarding.

### 6.4 `optimizations.numba_helpers` import path mismatch

`population.py` uses `from cenop.optimizations.numba_helpers import ...` (a subpackage path), but only `cenop/optimizations.py` (a single file) exists — there is no `optimizations/` directory. The imports work because they are inside `try/except` blocks, but the fallback path is always taken. Either:
- Create `optimizations/numba_helpers.py` as a subpackage, or
- Fix the import to `from cenop.optimizations import ...` (the actual module path).

---

## 7. Code Quality & Developer Experience

### 7.1 Linting and formatting

Add `pre-commit` config with `ruff` + `black`. The project already declares these as dev dependencies in `pyproject.toml` but there is no `.pre-commit-config.yaml` or CI enforcement.

### 7.2 Type annotations

Many core functions lack type annotations (especially in `population.py`, `simulation.py`, and `server/main.py`). Run `mypy --strict` incrementally — start with `parameters/` and `behavior/` which are smaller and more self-contained.

### 7.3 Developer documentation

Add `CONTRIBUTING.md` and/or `DEVELOPMENT.md` with:
- How to set up the micromamba environment
- How to install editable and run tests
- Architecture overview (SoA pattern, DEPONS vs JASMINE modes)

```bash
# Quick start commands
/usr/local/bin/micromamba run -p /opt/micromamba/envs/shiny pip install -e .
/usr/local/bin/micromamba run -p /opt/micromamba/envs/shiny pytest -q
```

---

## 8. Summary Action Items by Priority

| Priority | Item | Section |
|---|---|---|
| **Critical** | Remove duplicate `return rows` | 1.1 |
| **Critical** | Remove or reconnect dead `_daily_tasks()` code | 1.2 |
| **Critical** | Fix `app.py` to import from `cenop.server.main` | 1.3 |
| **High** | Delete stale `server/`, `cenop/`, and backup files | 2.1, 2.2 |
| **High** | Replace 85 `print()` with logging | 2.3 |
| **High** | Add logging to 38 modules | 2.4 |
| **High** | Narrow 33 broad `except Exception` catches | 2.5 |
| **High** | ~~Replace `np.random.seed()` with per-sim Generator~~ | 3.1 ✅ |
| **Medium** | Split `population.py` (1 875 lines) | 4.1 |
| **Medium** | Split `server/main.py` (2 267 lines) | 4.2 |
| **Medium** | Eliminate dual PSM storage | 4.3 |
| **Medium** | Add CI workflow + `environment.yml` | 6.2, 6.3 |
| **Medium** | Fix `optimizations.numba_helpers` import path | 6.4 |
| **Low** | Add Python 3.13 classifier | 6.1 |
| **Low** | Add `pre-commit`, linting, type annotations | 7.1, 7.2 |
| **Low** | Add developer documentation | 7.3 |
| **Low** | PSM sparse storage, disturbance memory vectorization | 5.1, 5.2 |
| **Low** | Consistent `float32` dtypes, `np.hypot` usage | 5.4, 5.5 |
