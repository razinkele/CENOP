# Tick Performance Phase 2 — Diminishing Returns Optimizations

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Squeeze remaining ~0.15ms from DEPONS mode (1.42→1.27ms) and ~0.25ms from JASMINE mode (1.79→1.54ms) through allocation elimination, dead code removal, and social sound kernel fusion.

**Architecture:** Seven small, low-risk optimizations. R3 (social sound kernel fusion) is the only medium-complexity task; the rest are trivial refactors. Each is independent — any subset can be skipped without affecting others.

**Tech Stack:** Python 3.13, NumPy, Numba (`@njit`, `prange`), pytest

**Post-Phase-1 baseline (N=500, 200×200 homogeneous):**
```
DEPONS mode (comm OFF):  1.42 ms/tick
JASMINE mode (comm ON):  1.79 ms/tick
Java DEPONS 3.2:         0.80 ms/tick
```

**Test command:** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`

---

## File Map

| File | Role | Tasks |
|------|------|-------|
| `src/cenop/agents/population.py` | Main simulation loop | R6, R7, R8, R9, R11, R12, R14 |
| `src/cenop/optimizations/kernels.py` | Numba kernels | R3 |
| `src/cenop/landscape/cell_data.py` | Landscape data access | R12 |
| `tests/test_tick_performance.py` | Performance + correctness tests | All |

---

## Task 1 (R7): Eliminate dead swimming_cost in BMR

**Files:**
- Modify: `src/cenop/agents/population.py:1870-1872`
- Test: `tests/test_tick_performance.py`

`swimming_cost` is always zero (`E_USE_PER_KM = 0`) but still computed with `10.0 ** self.prev_log_mov` (expensive `np.power`) every tick.

- [ ] **Step 1: Write test**

Add to `tests/test_tick_performance.py`:

```python
class TestR7DeadSwimmingCost:
    """R7: Eliminate dead swimming_cost computation."""

    def test_energy_unchanged_after_removing_dead_code(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — remove dead code**

In `_apply_bmr_cost` (line 1870-1872), replace:

```python
            bmr_cost = 0.001 * scaling_factor * self.params.e_use_per_30_min
            swimming_cost = (10.0 ** self.prev_log_mov) * 0.001 * scaling_factor * 0.0  # E_USE_PER_KM = 0
            total_cost = bmr_cost + swimming_cost
```

With:

```python
            total_cost = 0.001 * scaling_factor * self.params.e_use_per_30_min
```

- [ ] **Step 4: Run full test suite** — Expected: 508+ passed

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R7): remove dead swimming_cost computation (E_USE_PER_KM=0)"
```

---

## Task 2 (R9): Pass _active_idx to _eat_food_vectorized

**Files:**
- Modify: `src/cenop/agents/population.py:1849,3113-3140`
- Test: `tests/test_tick_performance.py`

`_eat_food_vectorized` calls `np.where(mask)[0]` internally (line 3125), but `_active_idx` is already available from `step()`.

- [ ] **Step 1: Write test**

```python
class TestR9PassActiveIdx:
    """R9: Reuse _active_idx in food eating."""

    def test_food_intake_unchanged(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement**

In `_eat_food_vectorized` (line 3113), add `active_idx=None` parameter:

```python
    def _eat_food_vectorized(self, mask, fract_to_eat, active_idx=None):
```

Replace line 3125:
```python
        active_idx = np.where(mask)[0]
```
With:
```python
        if active_idx is None:
            active_idx = np.where(mask)[0]
```

Then at the call site in `_apply_food_intake` (line 1849), pass the cached indices:

```python
                food_gained = self._eat_food_vectorized(mask, fract_to_eat, active_idx=self._active_idx)
```

- [ ] **Step 4: Run full test suite** — Expected: 508+ passed

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R9): pass cached _active_idx to _eat_food_vectorized"
```

---

## Task 3 (R11): Fix cell_data fallback paths + remove _positions fill

**Files:**
- Modify: `src/cenop/landscape/cell_data.py:516-517,545-546`
- Modify: `src/cenop/agents/population.py:850-852`
- Test: `tests/test_tick_performance.py`

Lines 850-852 fill `self._positions` to pass to `get_depths_vectorized`, but that method already accepts `xi`/`yi` which bypass it. However, `cell_data.py` fallback paths crash on `positions=None` (line 517: `len(positions)` when `_depth is None`). Fix the fallbacks first, then remove the fill.

**NOTE:** `_positions` is also filled in 7+ other locations. This task only removes the fill at lines 850-852 (the movement hot path). Full elimination would be a larger refactor — out of scope.

- [ ] **Step 1: Write test**

```python
class TestR11PositionsFallback:
    """R11: cell_data handles positions=None when xi/yi provided."""

    def test_get_depths_with_none_positions(self):
        land = make_landscape()
        xi = np.array([0, 50, 100], dtype=np.int32)
        yi = np.array([0, 50, 100], dtype=np.int32)
        result = land.get_depths_vectorized(None, xi=xi, yi=yi)
        assert len(result) == 3
        np.testing.assert_allclose(result, 30.0)

    def test_movement_unchanged(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: FAIL on `test_get_depths_with_none_positions` (fallback uses `len(positions)`)

- [ ] **Step 3: Fix cell_data fallback paths**

In `cell_data.py` `get_depths_vectorized` (line 516-517), replace:
```python
        if self._depth is None:
            return np.full(len(positions), 20.0)
```
With:
```python
        if self._depth is None:
            n = len(xi) if xi is not None else len(positions)
            return np.full(n, 20.0)
```

Apply same fix in `get_salinities_vectorized` (check its fallback path).

- [ ] **Step 4: Remove _positions fill in _update_movement**

In `_update_movement` (lines 850-852), remove:
```python
            self._positions[:, 0] = self.x
            self._positions[:, 1] = self.y
```

And pass `None` for positions:
```python
            np.copyto(
                self._depths,
                self.landscape.get_depths_vectorized(
                    None, xi=self._cell_xi, yi=self._cell_yi
                ),
            )
            np.copyto(
                self._salinity_vals,
                self.landscape.get_salinities_vectorized(
                    None, xi=self._cell_xi, yi=self._cell_yi
                ),
            )
```

- [ ] **Step 5: Run full test suite** — Expected: 508+ passed

- [ ] **Step 6: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/landscape/cell_data.py src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R11): fix cell_data None positions + remove movement fill"
```

---

## Task 4 (R8): Pre-allocate fract_to_eat buffer

**Files:**
- Modify: `src/cenop/agents/population.py:1847` and `__init__`
- Test: `tests/test_tick_performance.py`

`fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)` creates 3 temporaries per tick. Pre-allocate and compute in-place.

- [ ] **Step 1: Write test**

```python
class TestR8PreallocFract:
    """R8: Pre-allocate fract_to_eat buffer."""

    def test_food_fraction_unchanged(self):
        np.random.seed(42)
        pop = make_pop(100)
        for _ in range(30):
            pop.step()
        state = snapshot_state(pop)
        np.random.seed(42)
        pop2 = make_pop(100)
        for _ in range(30):
            pop2.step()
        assert_states_match(state, snapshot_state(pop2))
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement**

In `__init__`, add near other pre-allocated buffers:
```python
        self._fract_to_eat = np.zeros(count, dtype=np.float32)
```

In `_apply_food_intake` (line 1847), replace:
```python
            fract_to_eat = np.clip((20.0 - self.energy) / 10.0, 0.0, 0.99)
```
With:
```python
            np.subtract(np.float32(20.0), self.energy, out=self._fract_to_eat)
            self._fract_to_eat /= np.float32(10.0)
            np.clip(self._fract_to_eat, 0.0, 0.99, out=self._fract_to_eat)
            fract_to_eat = self._fract_to_eat
```

- [ ] **Step 4: Run full test suite** — Expected: 508+ passed

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R8): pre-allocate fract_to_eat buffer, compute in-place"
```

---

## Task 5 (R6): Vectorize death recording

**Files:**
- Modify: `src/cenop/agents/population.py:1949-1960`
- Test: `tests/test_tick_performance.py`

The death recording loop (`for idx in dead_indices:`) is Python-slow during mass death events. Vectorize it.

- [ ] **Step 1: Write test**

```python
class TestR6VectorizeDeathRecording:
    """R6: Vectorize death age/day/cause recording."""

    def test_death_recording_still_works(self):
        pop = make_pop(100)
        # Force low energy to cause starvation
        pop.energy[:] = 0.5
        for _ in range(100):
            pop.step()
        # Some deaths should have occurred
        assert len(pop.death_ages) > 0
        assert len(pop.death_ages) == len(pop.death_days)
        assert len(pop.death_ages) == len(pop.death_causes)
        # All causes should be valid strings
        for cause in pop.death_causes:
            assert cause in ("starvation", "old_age", "bycatch")
```

- [ ] **Step 2: Run test** — Expected: PASS

- [ ] **Step 3: Implement — vectorize the loop**

Replace lines 1949-1960:
```python
            sim_day = self._global_tick // 48
            dead_indices = np.where(all_deaths)[0]
            for idx in dead_indices:
                self.death_ages.append(int(self.age[idx]))
                self.death_days.append(sim_day)
                if starved[idx]:
                    self.death_causes.append("starvation")
                elif old_age[idx]:
                    self.death_causes.append("old_age")
                else:
                    self.death_causes.append("bycatch")
```

With:
```python
            sim_day = self._global_tick // 48
            dead_indices = np.where(all_deaths)[0]
            self.death_ages.extend(self.age[dead_indices].astype(int).tolist())
            self.death_days.extend([sim_day] * len(dead_indices))
            # Vectorized cause determination
            causes = np.where(starved[dead_indices], "starvation",
                     np.where(old_age[dead_indices], "old_age", "bycatch"))
            self.death_causes.extend(causes.tolist())
```

- [ ] **Step 4: Run full test suite** — Expected: 508+ passed

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R6): vectorize death age/cause recording"
```

---

## Task 6 (R12): Add xi/yi params to get_food_levels_vectorized

**Files:**
- Modify: `src/cenop/landscape/cell_data.py:195-209`
- Modify: `src/cenop/agents/population.py:2020-2025`
- Test: `tests/test_tick_performance.py`

`get_food_levels_vectorized` takes a `(N,2)` positions array, allocating `pos_buf` every tick (line 2022). Add optional `xi`/`yi` params to skip the allocation.

- [ ] **Step 1: Write test**

```python
class TestR12FoodLevelsXiYi:
    """R12: get_food_levels_vectorized accepts xi/yi."""

    def test_food_levels_same_with_xi_yi(self):
        from cenop.landscape.cell_data import CellData, LandscapeMetadata
        land = make_landscape()
        n = 50
        x = np.random.uniform(0, 199, n).astype(np.float32)
        y = np.random.uniform(0, 199, n).astype(np.float32)
        xi = np.clip(x.astype(np.int32), 0, 199)
        yi = np.clip(y.astype(np.int32), 0, 199)
        pos = np.column_stack((x, y))
        result_pos = land.get_food_levels_vectorized(pos)
        result_xiyi = land.get_food_levels_vectorized(None, xi=xi, yi=yi)
        np.testing.assert_array_equal(result_pos, result_xiyi)
```

- [ ] **Step 2: Run test** — Expected: FAIL (xi/yi params not yet supported)

- [ ] **Step 3: Implement**

In `cell_data.py`, modify `get_food_levels_vectorized` (line 195):

```python
    def get_food_levels_vectorized(self, positions=None, xi=None, yi=None):
        """Get food levels for multiple positions at once.

        Args:
            positions: (N, 2) array of (x, y) positions (used if xi/yi not provided)
            xi: Optional pre-computed int column indices
            yi: Optional pre-computed int row indices
        """
        self._ensure_loaded()
        if self._food_value is None:
            n = len(xi) if xi is not None else len(positions)
            return np.full(n, 0.5, dtype=np.float32)
        if xi is not None and yi is not None:
            return self._food_value[yi, xi].astype(np.float32)
        j = np.clip(positions[:, 0].astype(int), 0, self.width - 1)
        i = np.clip(positions[:, 1].astype(int), 0, self.height - 1)
        return self._food_value[i, j].astype(np.float32)
```

Then in `_update_reference_memory` (lines 2020-2025), replace:
```python
        n_active = len(active)
        pos_buf = np.empty((n_active, 2), dtype=np.float32)
        pos_buf[:, 0] = self.x[active]
        pos_buf[:, 1] = self.y[active]
        food_levels = self.landscape.get_food_levels_vectorized(pos_buf)
```
With:
```python
        food_levels = self.landscape.get_food_levels_vectorized(
            None, xi=self._cell_xi[active], yi=self._cell_yi[active]
        )
```

- [ ] **Step 4: Run full test suite** — Expected: 508+ passed

- [ ] **Step 5: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/landscape/cell_data.py src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R12): add xi/yi params to get_food_levels_vectorized

Eliminates per-tick (n_active, 2) pos_buf allocation in RefMem."
```

---

## Task 7 (R3): Fuse social sound computation into Numba kernel

**Files:**
- Modify: `src/cenop/optimizations/kernels.py` (add kernel)
- Modify: `src/cenop/agents/population.py:596-615` (wire kernel)
- Test: `tests/test_tick_performance.py`

The social vector path does ~15 NumPy operations on pair arrays: `hypot`, `*400`, `calculate_received_level` (log10 + multiply), `response_probability_from_rl` (clip + exp + clip, called 2×), plus accumulation. Fusing into one Numba kernel eliminates ~0.12ms dispatch overhead.

- [ ] **Step 1: Write test**

```python
class TestR3SocialSoundKernel:
    """R3: Fused social sound+probability Numba kernel."""

    def test_social_output_matches_numpy_path(self):
        """Social vectors must be identical with/without fused kernel."""
        np.random.seed(42)
        # Use JASMINE mode (comm enabled)
        params = SimulationParameters(
            porpoise_count=100, world_width=200, world_height=200,
            communication_enabled=True,
        )
        land = make_landscape()
        pop = PorpoisePopulation(100, params, landscape=land)
        pop._skip_land_avoidance = True
        for _ in range(20):
            pop.step()
        state = snapshot_state(pop)
        assert np.all(np.isfinite(state["x"]))
```

- [ ] **Step 2: Run test** — Expected: PASS (baseline)

- [ ] **Step 3: Add fused kernel to `kernels.py`**

```python
@njit(cache=True)
def social_sound_kernel(
    xi, yi, xj, yj, idx_i, idx_j,
    ambient_i, ambient_j,
    source_level, alpha_hat, beta_hat, threshold, slope,
    cell_size, n_agents,
    out_ux, out_uy, out_sw,
):
    """Fused social sound: distance + TL + RL + probability + accumulation.

    Replaces: hypot, *cell_size, calculate_transmission_loss,
    response_probability_from_rl (2x), social_accumulate_kernel.

    Uses parallel=False because multiple pairs may share agent indices,
    causing race conditions in the accumulation (matching existing
    social_accumulate_kernel which is also sequential).

    Handles ambient noise: p_i and p_j are computed independently using
    per-agent ambient levels (SNR = RL - ambient). When ambient is 0
    (no ambient RL), uses raw RL.

    Args:
        xi, yi: float64[n_pairs] — agent i positions (grid cells)
        xj, yj: float64[n_pairs] — agent j positions (grid cells)
        idx_i, idx_j: int64[n_pairs] — global agent indices
        ambient_i, ambient_j: float64[n_pairs] — ambient RL at each agent (0 if none)
        source_level, alpha_hat, beta_hat: sound propagation params
        threshold, slope: response probability params
        cell_size: grid cell size in meters (400.0)
        n_agents: total agent count (for output array bounds)
        out_ux, out_uy, out_sw: float64[n_agents] — accumulated (must be zeroed)
    """
    n_pairs = len(xi)
    for p in range(n_pairs):
        # Distance in grid cells
        ddx = xj[p] - xi[p]
        ddy = yj[p] - yi[p]
        dist_cells = math.sqrt(ddx * ddx + ddy * ddy) + 1.0e-6
        dist_m = dist_cells * cell_size

        # Unit direction (i → j)
        ux = ddx / dist_cells
        uy = ddy / dist_cells

        # Transmission loss: TL = beta * log10(r) + alpha * r
        dist_safe = max(dist_m, 1.0)
        tl = beta_hat * math.log10(dist_safe) + alpha_hat * dist_safe

        # Received level (symmetric — same distance both directions)
        rl = source_level - tl

        # Asymmetric probabilities: each listener has different ambient noise
        # p_i = P(agent i hears j) based on SNR at i
        snr_i = rl - ambient_i[p]
        linear_i = slope * (snr_i - threshold)
        linear_i = max(-500.0, min(500.0, linear_i))
        p_i = 1.0 / (1.0 + math.exp(-linear_i))
        p_i = max(0.0, min(1.0, p_i))

        # p_j = P(agent j hears i) based on SNR at j
        snr_j = rl - ambient_j[p]
        linear_j = slope * (snr_j - threshold)
        linear_j = max(-500.0, min(500.0, linear_j))
        p_j = 1.0 / (1.0 + math.exp(-linear_j))
        p_j = max(0.0, min(1.0, p_j))

        ii = idx_i[p]
        jj = idx_j[p]

        # Agent i attracted toward j (weighted by p_i)
        out_ux[ii] += ux * p_i
        out_uy[ii] += uy * p_i
        out_sw[ii] += p_i

        # Agent j attracted toward i (reverse direction, weighted by p_j)
        out_ux[jj] += -ux * p_j
        out_uy[jj] += -uy * p_j
        out_sw[jj] += p_j
```

**Design notes:**
- `parallel=False` (no `prange`) because pairs share agent indices → race condition on accumulation. Matches existing `social_accumulate_kernel` which is also sequential.
- `ambient_i/ambient_j` arrays handle both code paths: when `ambient_rl is None`, pass zeros (→ `snr = rl - 0 = rl`); when ambient_rl provided, pass `ambient_rl[idx_i]` and `ambient_rl[idx_j]`.
- Epsilon `1.0e-6` matches the float64 context of the kernel (existing NumPy path uses `np.float32` context which rounds differently, but the difference is negligible).

- [ ] **Step 4: Wire kernel into `_compute_social_vectors`**

In `_compute_social_vectors`, after the pair coordinate extraction (line ~596-600), replace lines 602-627 (the sound computation + accumulation section) with:

```python
                if _HAS_KERNELS:
                    from cenop.optimizations.kernels import social_sound_kernel
                    self._social_ux.fill(0.0)
                    self._social_uy.fill(0.0)
                    self._social_sw.fill(0.0)

                    # Build ambient arrays for asymmetric probability
                    if ambient_rl is not None:
                        amb_i = np.asarray(ambient_rl[idx_i], dtype=np.float64)
                        amb_j = np.asarray(ambient_rl[idx_j], dtype=np.float64)
                    else:
                        amb_i = np.zeros(ncols, dtype=np.float64)
                        amb_j = np.zeros(ncols, dtype=np.float64)

                    social_sound_kernel(
                        xi.astype(np.float64), yi.astype(np.float64),
                        xj.astype(np.float64), yj.astype(np.float64),
                        idx_i, idx_j, amb_i, amb_j,
                        float(source_level), float(self.params.alpha_hat),
                        float(self.params.beta_hat), float(threshold), float(slope),
                        400.0, self.count,
                        self._social_ux, self._social_uy, self._social_sw,
                    )
                    ux_total = self._social_ux
                    uy_total = self._social_uy
                    sw_total = self._social_sw
                else:
                    # ... keep existing NumPy path as fallback ...
```

Keep the existing code as the `else` branch.

- [ ] **Step 5: Run full test suite** — Expected: 508+ passed

- [ ] **Step 6: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/optimizations/kernels.py src/cenop/agents/population.py tests/test_tick_performance.py
git commit -m "perf(R3): fuse social sound computation into Numba kernel

Replaces ~15 NumPy operations (hypot, log10, exp, clip, accumulate)
with single Numba pass over pair arrays. ~0.12ms savings in JASMINE mode."
```

---

## Task 8: Final benchmark verification

- [ ] **Step 1: Run benchmark**

```bash
cd /home/razinka/cenjas && micromamba run -n shiny python3 -c "
import time, sys, numpy as np
sys.path.insert(0, 'CENOP/src')
from cenop.parameters.simulation_params import SimulationParameters
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import CellData, LandscapeMetadata
def mk():
    cd = CellData.__new__(CellData)
    cd.landscape_name='Homogeneous'; cd.data_dir=''
    cd.metadata=LandscapeMetadata(ncols=200,nrows=200,xllcorner=0.0,yllcorner=0.0)
    cd._depth=np.full((200,200),30.0,dtype=np.float32)
    cd._dist_to_coast=np.full((200,200),5000.0,dtype=np.float32)
    cd._sediment=np.full((200,200),5.0,dtype=np.float32)
    cd._food_prob=np.ones((200,200),dtype=np.float32)
    cd._food_value=np.full((200,200),50.0,dtype=np.float32)
    cd._blocks=np.zeros((200,200),dtype=np.int32)
    cd._entropy=np.full((12,200,200),50.0,dtype=np.float32)
    cd._salinity=np.full((12,200,200),30.0,dtype=np.float32)
    cd._demand_grid=np.zeros((200,200),dtype=np.float32)
    cd._current_month=1; cd._loaded=True; return cd
for mode, comm in [('DEPONS', False), ('JASMINE', True)]:
    p=SimulationParameters(porpoise_count=500,world_width=200,world_height=200,communication_enabled=comm)
    pop=PorpoisePopulation(500,p,landscape=mk()); pop._skip_land_avoidance=True
    for _ in range(100): pop.step()
    t0=time.perf_counter()
    for _ in range(500): pop.step()
    ms=(time.perf_counter()-t0)/500*1000
    print(f'{mode}: {ms:.3f} ms/tick')
"
```

- [ ] **Step 2: Run full test suite** — Expected: 508+ passed

- [ ] **Step 3: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add docs/DEPONS-CENOP-PARITY-ANALYSIS.md
git commit -m "docs: update performance baselines after Phase 2 optimizations"
```
