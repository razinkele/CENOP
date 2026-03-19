# JAX Performance Fix — Vectorize CRW Kernel

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the JAX tick performance from 1132ms/tick to <2ms/tick at N=500 by vectorizing the CRW kernel and fixing secondary issues.

**Problem:** The current JAX CRW kernel uses `lax.scan(_crw_single_agent, ..., length=n)` which processes each agent **sequentially** with 3 `lax.while_loop`s per agent for rejection sampling. This serializes all computation and costs ~1076ms/tick. A simple vectorized CRW without while_loops takes 0.35ms.

**Key insight:** Array transfer overhead (NumPy<->JAX) is only ~2ms/tick — not the bottleneck. The serial per-agent processing is 99.8% of the problem.

**Spec:** `docs/superpowers/specs/2026-03-19-jax-jit-tick-design.md`
**Previous plan:** `docs/superpowers/plans/2026-03-19-jax-jit-tick.md`

**Testing:** `cd /home/razinka/cenjas && eval "$(micromamba shell hook --shell bash)" && micromamba activate shiny && python3 -m pytest CENOP/tests/test_jax_tick.py -x -q`

**Commit convention:** Commit from within `CENOP/` (nested git repo, branch `CENOP-JASMINE`).

---

## Profiling Baseline (N=500 Kattegat)

| Component | Current | Target |
|-----------|---------|--------|
| `jax_tick_movement` | 1076ms | <2ms |
| `jax_tick_energy` | 0.8ms | 0.8ms (no change) |
| Array transfers | ~2ms | ~2ms (no change) |
| Python post-processing | ~0.5ms | ~0.5ms (no change) |
| **Full tick** | **1132ms** | **<5ms** |

For reference: Numba path = 2.34ms/tick, Java DEPONS = 0.84ms/tick.

---

## Task V1: Vectorize CRW Kernel (eliminates 99% of the problem)

**File:** `CENOP/src/cenop/optimizations/jax_kernels.py`

**What to change:** Replace `_crw_single_agent` + `lax.scan` + per-agent `lax.while_loop` with fully vectorized batch operations.

### Current architecture (slow):
```
lax.scan over N agents:
  for each agent:
    while_loop 1: angle rejection (|angle| > 180)
    while_loop 2: distance-dependent modulation
    while_loop 3: step length rejection (log_mov > max_mov)
```
Total: N × 3 while_loops = 1500 sequential loop evaluations.

### New architecture (fast):
```
Vectorized over all N agents simultaneously:
  1. Compute angles for all agents at once
  2. Replace while_loop rejection with bounded retry + clip fallback
  3. Compute step lengths for all agents at once
  4. Replace while_loop rejection with bounded retry + clip fallback
```

### Implementation strategy

The rejection sampling loops almost never execute (measured 0/500 violations for angle loop with typical parameters). Replace with:

**Option A — Fixed retry + clip (recommended):**
```python
# Angle: compute once, clip any violations
pres_angle = (corr_angle_base * prev_angle + rand_angle) * env_mod
# Retry once for any violations
violations = jnp.abs(pres_angle) > 180.0
key, k2 = jax.random.split(key)
retry_angle = (corr_angle_base * prev_angle + jax.random.normal(k2, (n,)) * r2_sd + r2_mean) * env_mod
pres_angle = jnp.where(violations, retry_angle, pres_angle)
# Final clip (matches emergency fallback in current code)
pres_angle = jnp.clip(pres_angle, -180.0, 180.0)
```

**Option B — Vectorized while_loop (if exact Java matching needed):**
```python
# while_loop over all agents simultaneously — retry only violating agents
def angle_cond(state):
    return jnp.any(state.violations) & (state.retry < 200)
def angle_body(state):
    # Resample ONLY violating agents, keep others
    ...
```
Option B is slower than A (~5ms vs ~0.5ms) but matches Java semantics exactly. Option A is recommended since the statistical properties are equivalent.

### Steps

- [ ] **Step 1:** Write a benchmark test that measures CRW kernel time at N=500, asserting < 5ms.

- [ ] **Step 2:** Rewrite `jax_crw_kernel` to use vectorized operations:
  - Remove `_crw_single_agent` and `_crw_active` functions
  - Replace `lax.scan` with direct vectorized computation
  - Use 1 retry pass + `jnp.clip` fallback for all 3 rejection loops
  - Keep the same function signature (no API change)

- [ ] **Step 3:** Run existing CRW tests — all 7 in `TestJaxCRWKernel` must pass.

- [ ] **Step 4:** Run full JAX test suite (52 unit tests, excluding the 2 slow integration tests):
  ```bash
  python3 -m pytest CENOP/tests/test_jax_tick.py -x -q -k "not TestJaxFullTick"
  ```

- [ ] **Step 5:** Benchmark CRW kernel standalone at N=500. Target: < 1ms.

- [ ] **Step 6:** Commit.
  ```bash
  cd /home/razinka/cenjas/CENOP
  git add src/cenop/optimizations/jax_kernels.py
  git commit -m "perf: vectorize JAX CRW kernel — replace lax.scan with batch ops"
  ```

---

## Task V2: Fix Land Avoidance Dtype Bug

**File:** `CENOP/src/cenop/optimizations/jax_kernels.py`

**Bug:** `jax_land_avoidance` has a float64/float32 mismatch in its `lax.fori_loop` carry. The loop body outputs float32 for a carry component that was initialized as float64. This works in the composed `jax_tick_movement` (where inputs are cast), but fails when called standalone with float64 inputs.

### Steps

- [ ] **Step 1:** Ensure all carry components in `jax_land_avoidance`'s `lax.fori_loop` match input/output dtypes. Cast outputs to match inputs explicitly.

- [ ] **Step 2:** Run land avoidance tests:
  ```bash
  python3 -m pytest CENOP/tests/test_jax_tick.py::TestJaxLandAvoidance -v
  ```

- [ ] **Step 3:** Commit.

---

## Task V3: Benchmark Full Tick and Assess On-Device State Need

**Files:** None modified — profiling only.

### Steps

- [ ] **Step 1:** After V1+V2, benchmark the full JAX tick at N=500:
  ```python
  # Warmup 5 ticks, then measure 100 ticks
  # Report: mean, median, p95 ms/tick
  ```

- [ ] **Step 2:** Profile the breakdown:
  - JAX movement kernel time (block_until_ready)
  - JAX energy kernel time
  - Array transfer time (jnp.asarray + np.asarray)
  - Python post-processing time
  - Total tick time

- [ ] **Step 3:** Decision gate:
  - If total tick < 3ms → **Done.** On-device state not needed. Close this plan.
  - If total tick 3-5ms → Acceptable. Document results, close plan.
  - If total tick > 5ms → Investigate remaining bottleneck, consider V4.

- [ ] **Step 4:** Commit benchmark results as a comment in this plan or in the design spec.

---

## Task V4 (Conditional): On-Device State Optimization

**Only proceed if V3 shows total tick > 5ms and array transfers are the bottleneck.**

### Analysis of what on-device state requires

7 Python methods run between JAX ticks, touching ~30 arrays:

| Method | Frequency | Feasibility to port to JAX |
|--------|-----------|---------------------------|
| `_update_reference_memory` | Every tick | Hard — circular buffer with Python indexing |
| `_recompute_cell_indices` | Every tick | Easy — just `int32(clip(x/y))` |
| `_update_behavior_fsm` | Every tick | Hard — module dispatch, conditional logic |
| `_update_psm` | Every tick | Medium — grid accumulation |
| `_check_dispersal_trigger` | 1/48 ticks | Medium — energy history comparison |
| `_update_aging` | Every tick | Trivial — `age += increment` |
| `_handle_reproduction` | 1/48 ticks | Very hard — slot allocation, births |

External consumers expecting NumPy: UI dashboard, simulation.py deterrence, output writer, cKDTree social vectors.

### Strategy if needed

1. Keep SoA arrays as JAX DeviceArrays permanently
2. Port `_recompute_cell_indices` and `_update_aging` into JAX tick functions (trivial)
3. Port `_update_reference_memory` into `jax_tick_movement` (medium — vectorized circular buffer)
4. Keep `_handle_reproduction`, `_update_behavior_fsm`, `_update_psm` in Python — convert to/from JAX only for these (infrequent or complex)
5. Lazy NumPy conversion for dashboard/UI (only when requested, not every tick)

Estimated effort: ~2-3 sessions. Expected gain: ~2ms/tick.

---

## Execution Order

```
V1 (vectorize CRW) → V2 (dtype fix) → V3 (benchmark) → V4 (conditional)
```

V1 and V2 are independent and can be done in parallel. V3 depends on both. V4 only if V3 indicates need.

## Expected Results

| Metric | Current | After V1-V3 | Java |
|--------|--------:|------------:|-----:|
| N=500 ms/tick | 1132 | ~3-5 | 0.84 |
| vs Java | 1348x | ~3-6x | 1.0x |
| vs Numba | 484x | ~1.3-2.1x | — |

Note: Matching Java performance (~0.84ms) is unlikely on CPU JAX for N=500 due to Python orchestration overhead from the 7 post-processing methods. GPU JAX could close the gap at larger N (>5000).
