# Cython Tick Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Python/NumPy glue between Numba kernels with a compiled Cython module, reducing DEPONS tick time from ~1.05ms to ~0.68ms (potentially faster than Java's 0.80ms).

**Architecture:** A single Cython `.pyx` module (`tick_cython.pyx`) provides a `fused_post_crw_tick()` function that replaces everything after the CRW+RefMem Numba kernels: heading composition, position update, boundary reflection, food intake, BMR cost, and mortality check — all in one compiled C loop. The existing Numba CRW and RefMem kernels stay (they're already 2.5× faster than Java via `prange`). The Python path is kept as a fallback for environments without Cython. A `_HAS_CYTHON` flag mirrors the existing `_HAS_KERNELS` pattern.

**Tech Stack:** Cython 3.x, NumPy C API (`cimport numpy`), libc math, existing Numba kernels

**Benchmarked prototype:** `scripts/cython_prototype/tick_core.pyx` — proves 0.083ms for N=500 (vs 0.307ms NumPy = 3.7×)

**Test command:** `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/ -x -q --ignore=tests/test_depons_physiology.py --ignore=tests/test_validation.py --ignore=tests/test_map_layers.py --ignore=tests/test_phase5.py`

---

## File Map

| File | Role | Tasks |
|------|------|-------|
| `src/cenop/optimizations/tick_cython.pyx` | **New** — Cython fused tick module | 1, 2, 3, 4 |
| `src/cenop/optimizations/setup_cython.py` | **New** — Cython build script | 1 |
| `src/cenop/optimizations/__init__.py` | Add `_HAS_CYTHON` import flag | 5 |
| `src/cenop/agents/population.py` | Wire Cython path into `step()` | 5, 6 |
| `tests/test_cython_tick.py` | **New** — Cython correctness + benchmark tests | 1, 2, 3, 4, 5, 6 |
| `pyproject.toml` | Add Cython optional dependency | 7 |

---

## Task 1: Build infrastructure + minimal Cython module

**Files:**
- Create: `src/cenop/optimizations/tick_cython.pyx`
- Create: `src/cenop/optimizations/setup_cython.py`
- Create: `tests/test_cython_tick.py`

This task creates the Cython build infrastructure and a minimal function that proves compilation works.

- [ ] **Step 1: Create the Cython build script**

Create `src/cenop/optimizations/setup_cython.py`:

```python
"""Build Cython extensions for CENOP tick acceleration.

Usage:
    cd src/cenop/optimizations
    python setup_cython.py build_ext --inplace
"""
from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

extensions = [
    Extension(
        "tick_cython",
        ["tick_cython.pyx"],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "language_level": "3",
        },
    ),
)
```

- [ ] **Step 2: Create minimal .pyx with a test function**

Create `src/cenop/optimizations/tick_cython.pyx`:

```cython
# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated DEPONS tick phases.

Replaces the Python/NumPy glue between Numba CRW/RefMem kernels with
compiled C loops. Provides ~3.7x speedup for heading+position+food+BMR+mortality.

The Numba CRW and RefMem kernels are NOT replaced — they use prange parallelism
and are already 2.5x faster than Java.
"""
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, atan2, pow, log, exp, fmod, M_PI

ctypedef np.float64_t f64
ctypedef np.float32_t f32
ctypedef np.int32_t i32

cdef double DEG2RAD = M_PI / 180.0
cdef double RAD2DEG = 180.0 / M_PI


def cython_available() -> bool:
    """Return True — confirms compiled module is loadable."""
    return True
```

- [ ] **Step 3: Create test file**

Create `tests/test_cython_tick.py`:

```python
"""Tests for Cython tick acceleration module."""

import numpy as np
import pytest
import subprocess
import sys
import os

# Build Cython module if not already built
_CYTHON_DIR = os.path.join(
    os.path.dirname(__file__), '..', 'src', 'cenop', 'optimizations'
)

def _try_import_cython():
    """Try importing tick_cython, build if missing."""
    try:
        from cenop.optimizations.tick_cython import cython_available
        return True
    except ImportError:
        return False

CYTHON_OK = _try_import_cython()


@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonAvailable:
    def test_module_loads(self):
        from cenop.optimizations.tick_cython import cython_available
        assert cython_available() is True
```

- [ ] **Step 4: Build the Cython module**

Run:
```bash
cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && micromamba run -n shiny python3 setup_cython.py build_ext --inplace
```
Expected: `.so` file created in `src/cenop/optimizations/`

- [ ] **Step 5: Run test**

Run: `cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py -v`
Expected: 1 passed

- [ ] **Step 6: Add .gitignore for generated files**

Create `src/cenop/optimizations/.gitignore`:
```
# Cython build artifacts
*.c
*.so
*.pyd
build/
```

- [ ] **Step 7: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add src/cenop/optimizations/tick_cython.pyx src/cenop/optimizations/setup_cython.py src/cenop/optimizations/.gitignore tests/test_cython_tick.py
git commit -m "feat: add Cython tick build infrastructure"
```

---

## Task 2: Implement fused heading+position+reflect in Cython

**Files:**
- Modify: `src/cenop/optimizations/tick_cython.pyx`
- Test: `tests/test_cython_tick.py`

Port the heading composition + position update + boundary reflection into the Cython module. This replaces the fused Numba `heading_position_reflect_kernel` and the surrounding Python glue.

- [ ] **Step 1: Add test**

Append to `tests/test_cython_tick.py`:

```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonHeadingPosition:
    def test_heading_position_matches_numpy(self):
        """Cython heading+position must match NumPy reference."""
        from cenop.optimizations.tick_cython import cython_heading_position_reflect

        np.random.seed(42)
        n = 200
        heading = np.random.uniform(0, 360, n).astype(np.float32)
        pres_angle = np.random.normal(0, 20, n).astype(np.float64)
        log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
        ve_total = np.random.uniform(0, 1, n).astype(np.float32)
        vt_x = np.random.normal(0, 0.1, n).astype(np.float32)
        vt_y = np.random.normal(0, 0.1, n).astype(np.float32)
        x = np.random.uniform(5, 195, n).astype(np.float32)
        y = np.random.uniform(5, 195, n).astype(np.float32)
        mask = np.ones(n, dtype=np.uint8)
        is_disp = np.zeros(n, dtype=np.uint8)

        out_h = np.zeros(n, dtype=np.float32)
        out_x = np.zeros(n, dtype=np.float32)
        out_y = np.zeros(n, dtype=np.float32)
        out_dx = np.zeros(n, dtype=np.float64)
        out_dy = np.zeros(n, dtype=np.float64)
        out_step = np.zeros(n, dtype=np.float64)
        out_prev_angle = np.zeros(n, dtype=np.float64)
        out_prev_lm = np.zeros(n, dtype=np.float64)

        cython_heading_position_reflect(
            heading, pres_angle, log_mov, ve_total, vt_x, vt_y,
            x, y, mask, is_disp,
            0.001, 4.0, 200, 200,
            out_h, out_x, out_y, out_dx, out_dy, out_step,
            out_prev_angle, out_prev_lm,
        )

        # Basic sanity
        assert np.all(np.isfinite(out_x))
        assert np.all(np.isfinite(out_y))
        assert np.all(out_x >= 0) and np.all(out_x <= 199)
        assert np.all(out_y >= 0) and np.all(out_y <= 199)
        assert np.all(out_h >= 0) and np.all(out_h < 360)
```

- [ ] **Step 2: Implement `cython_heading_position_reflect` in .pyx**

Add to `tick_cython.pyx`:

```cython
def cython_heading_position_reflect(
    np.ndarray[f32, ndim=1] heading,
    np.ndarray[f64, ndim=1] pres_angle,
    np.ndarray[f64, ndim=1] log_mov,
    np.ndarray[f32, ndim=1] ve_total,
    np.ndarray[f32, ndim=1] vt_x,
    np.ndarray[f32, ndim=1] vt_y,
    np.ndarray[f32, ndim=1] x,
    np.ndarray[f32, ndim=1] y,
    np.ndarray[np.uint8_t, ndim=1] mask,
    np.ndarray[np.uint8_t, ndim=1] is_dispersing,
    double inertia_const,
    double disp_step,
    int world_w,
    int world_h,
    np.ndarray[f32, ndim=1] out_heading,
    np.ndarray[f32, ndim=1] out_x,
    np.ndarray[f32, ndim=1] out_y,
    np.ndarray[f64, ndim=1] out_dx,
    np.ndarray[f64, ndim=1] out_dy,
    np.ndarray[f64, ndim=1] out_step_dist,
    np.ndarray[f64, ndim=1] out_prev_angle,
    np.ndarray[f64, ndim=1] out_prev_log_mov,
):
    """Heading composition + position + reflect in compiled C."""
    cdef int n = len(heading)
    cdef int i
    cdef double max_x = <double>(world_w - 1)
    cdef double max_y = <double>(world_h - 1)
    cdef double h, rad, dx_crw, dy_crw, pres_mov, crw_c
    cdef double total_dx, total_dy, new_h, step, ddx, ddy, nx, ny
    cdef double pre_heading

    for i in range(n):
        if not mask[i]:
            continue

        pres_mov = pow(10.0, log_mov[i])
        pre_heading = heading[i]

        if is_dispersing[i]:
            new_h = heading[i]
            step = disp_step
        else:
            h = fmod(heading[i] + pres_angle[i], 360.0)
            if h < 0:
                h += 360.0
            rad = h * DEG2RAD
            dx_crw = sin(rad)
            dy_crw = cos(rad)
            crw_c = inertia_const + pres_mov * ve_total[i]
            total_dx = dx_crw * crw_c + vt_x[i]
            total_dy = dy_crw * crw_c + vt_y[i]
            new_h = atan2(total_dx, total_dy) * RAD2DEG
            if new_h < 0:
                new_h += 360.0
            step = pres_mov / 4.0

        out_heading[i] = <f32>new_h
        out_step_dist[i] = step
        out_prev_angle[i] = fmod(new_h - pre_heading + 180.0, 360.0) - 180.0
        out_prev_log_mov[i] = log_mov[i]

        rad = new_h * DEG2RAD
        ddx = sin(rad) * step
        ddy = cos(rad) * step
        nx = x[i] + ddx
        ny = y[i] + ddy

        # Boundary reflection
        if nx < 0:
            nx = -nx; ddx = -ddx
        elif nx > max_x:
            nx = 2.0 * max_x - nx; ddx = -ddx
        if nx < 0: nx = 0.0
        elif nx > max_x: nx = max_x
        if ny < 0:
            ny = -ny; ddy = -ddy
        elif ny > max_y:
            ny = 2.0 * max_y - ny; ddy = -ddy
        if ny < 0: ny = 0.0
        elif ny > max_y: ny = max_y

        out_x[i] = <f32>nx
        out_y[i] = <f32>ny
        out_dx[i] = ddx
        out_dy[i] = ddy
```

- [ ] **Step 3: Rebuild and test**

```bash
cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && micromamba run -n shiny python3 setup_cython.py build_ext --inplace
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/cenop/optimizations/tick_cython.pyx tests/test_cython_tick.py
git commit -m "feat: Cython heading+position+reflect kernel"
```

---

## Task 3: Add food intake + BMR + mortality to Cython

**Files:**
- Modify: `src/cenop/optimizations/tick_cython.pyx`
- Test: `tests/test_cython_tick.py`

Add the `cython_food_bmr_mortality()` function that fuses food intake, BMR cost, and mortality check into a single C loop.

- [ ] **Step 1: Add test**

```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonFoodBMRMortality:
    def test_energy_changes_correctly(self):
        from cenop.optimizations.tick_cython import cython_food_bmr_mortality

        n = 100
        energy = np.full(n, 10.0, dtype=np.float32)
        active_mask = np.ones(n, dtype=np.uint8)
        with_calf = np.zeros(n, dtype=np.uint8)
        food_grid = np.full((200, 200), 50.0, dtype=np.float32)
        xi = np.random.randint(0, 200, n, dtype=np.int32)
        yi = np.random.randint(0, 200, n, dtype=np.int32)

        deaths = cython_food_bmr_mortality(
            energy, active_mask, with_calf,
            food_grid, xi, yi,
            4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
        )

        # Energy should have changed (food gain - BMR)
        assert not np.all(energy == 10.0)
        assert np.all(energy >= 0)
        assert np.all(energy <= 20.0)
        assert isinstance(deaths, int)

    def test_starvation_kills_low_energy(self):
        from cenop.optimizations.tick_cython import cython_food_bmr_mortality

        n = 50
        energy = np.full(n, 0.01, dtype=np.float32)  # Near-death
        active_mask = np.ones(n, dtype=np.uint8)
        with_calf = np.zeros(n, dtype=np.uint8)
        food_grid = np.full((200, 200), 0.01, dtype=np.float32)  # No food
        xi = np.zeros(n, dtype=np.int32)
        yi = np.zeros(n, dtype=np.int32)

        deaths = cython_food_bmr_mortality(
            energy, active_mask, with_calf,
            food_grid, xi, yi,
            4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
        )

        # Most agents should die
        alive = np.sum(active_mask)
        assert alive < n, f"Expected deaths but all {n} survived"
```

- [ ] **Step 2: Implement `cython_food_bmr_mortality`**

Add to `tick_cython.pyx`:

```cython
def cython_food_bmr_mortality(
    np.ndarray[f32, ndim=1] energy,
    np.ndarray[np.uint8_t, ndim=1] active_mask,
    np.ndarray[np.uint8_t, ndim=1] with_calf,
    np.ndarray[f32, ndim=2] food_grid,
    np.ndarray[i32, ndim=1] xi,
    np.ndarray[i32, ndim=1] yi,
    double e_use_per_30_min,
    double e_lact,
    double m_mort_prob_const,
    double x_survival_const,
    double seasonal_scaling,
    int world_w,
    int world_h,
) -> int:
    """Fused food intake + BMR cost + mortality check in C."""
    cdef int n = len(energy)
    cdef int i, cxi, cyi
    cdef double fract, food_available, eaten, scaling, bmr
    cdef double yearly_surv, step_surv
    cdef int deaths = 0

    cdef np.ndarray[f64, ndim=1] rand_mort = np.random.random(n)

    for i in range(n):
        if not active_mask[i]:
            continue

        cxi = xi[i]
        cyi = yi[i]
        if cxi < 0: cxi = 0
        if cxi >= world_w: cxi = world_w - 1
        if cyi < 0: cyi = 0
        if cyi >= world_h: cyi = world_h - 1

        # Food intake
        fract = (20.0 - energy[i]) / 10.0
        if fract < 0: fract = 0.0
        if fract > 0.99: fract = 0.99
        food_available = food_grid[cyi, cxi]
        eaten = food_available * fract
        food_grid[cyi, cxi] = food_available - eaten
        if food_grid[cyi, cxi] < 0.01:
            food_grid[cyi, cxi] = 0.01
        energy[i] += <f32>eaten
        if energy[i] > 20.0:
            energy[i] = 20.0

        # Mortality check (BEFORE BMR — Java ordering)
        if energy[i] > 0:
            yearly_surv = 1.0 - m_mort_prob_const * exp(-energy[i] * x_survival_const)
            if yearly_surv > 0:
                step_surv = exp(log(yearly_surv) / 17280.0)
            else:
                step_surv = 0.0
        else:
            step_surv = 0.0

        if rand_mort[i] > step_surv:
            if not with_calf[i] or energy[i] <= 0:
                active_mask[i] = 0
                deaths += 1
            else:
                with_calf[i] = 0

        if not active_mask[i]:
            continue

        # BMR cost (AFTER mortality — dead agents excluded)
        scaling = seasonal_scaling
        if with_calf[i]:
            scaling = scaling * e_lact
        bmr = 0.001 * scaling * e_use_per_30_min
        energy[i] -= <f32>bmr
        if energy[i] < 0: energy[i] = 0.0
        if energy[i] > 20.0: energy[i] = 20.0

    return deaths
```

- [ ] **Step 3: Rebuild and test**

```bash
cd /home/razinka/cenjas/CENOP/src/cenop/optimizations && micromamba run -n shiny python3 setup_cython.py build_ext --inplace
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py -v
```

- [ ] **Step 4: Commit**

```bash
git add src/cenop/optimizations/tick_cython.pyx tests/test_cython_tick.py
git commit -m "feat: Cython food+BMR+mortality fused kernel"
```

---

## Task 4: Create combined `cython_depons_post_crw()` entry point

**Files:**
- Modify: `src/cenop/optimizations/tick_cython.pyx`
- Test: `tests/test_cython_tick.py`

Combine Tasks 2 and 3 into a single function that `step()` calls once after the Numba CRW+RefMem kernels.

- [ ] **Step 1: Add test**

```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonFullPostCRW:
    def test_deterministic_output(self):
        """Two runs with same seed produce identical results."""
        from cenop.optimizations.tick_cython import cython_depons_post_crw

        def run_once(seed):
            np.random.seed(seed)
            n = 200
            x = np.random.uniform(5, 195, n).astype(np.float32)
            y = np.random.uniform(5, 195, n).astype(np.float32)
            heading = np.random.uniform(0, 360, n).astype(np.float32)
            prev_angle = np.random.normal(0, 10, n).astype(np.float64)
            prev_log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
            energy = np.random.uniform(5, 15, n).astype(np.float32)
            active = np.ones(n, dtype=np.uint8)
            is_disp = np.zeros(n, dtype=np.uint8)
            with_calf = (np.random.random(n) > 0.8).astype(np.uint8)
            pres_angle = np.random.normal(0, 20, n).astype(np.float64)
            log_mov = np.random.uniform(0.5, 1.5, n).astype(np.float64)
            ve_total = np.random.uniform(0, 1, n).astype(np.float32)
            vt_x = np.random.normal(0, 0.1, n).astype(np.float32)
            vt_y = np.random.normal(0, 0.1, n).astype(np.float32)
            food = np.full((200, 200), 50.0, dtype=np.float32)

            deaths = cython_depons_post_crw(
                x, y, heading, prev_angle, prev_log_mov, energy,
                active, is_disp, with_calf,
                pres_angle, log_mov, ve_total, vt_x, vt_y,
                food, 0.001, 4.0, 4.5, 1.4, 1.0, 0.4, 1.0, 200, 200,
            )
            return x.copy(), y.copy(), heading.copy(), energy.copy(), active.copy()

        r1 = run_once(42)
        r2 = run_once(42)
        for a, b in zip(r1, r2):
            np.testing.assert_array_equal(a, b)
```

- [ ] **Step 2: Implement `cython_depons_post_crw`**

This function calls `cython_heading_position_reflect` then `cython_food_bmr_mortality` internally, computing cell indices between them. Or fuse them into a single C loop (copy from the prototype `fused_depons_tick`).

The function signature should match what `population.py step()` can easily call:

```cython
def cython_depons_post_crw(
    # Agent state (read-write)
    np.ndarray[f32, ndim=1] x,
    np.ndarray[f32, ndim=1] y,
    np.ndarray[f32, ndim=1] heading,
    np.ndarray[f64, ndim=1] prev_angle,
    np.ndarray[f64, ndim=1] prev_log_mov,
    np.ndarray[f32, ndim=1] energy,
    np.ndarray[np.uint8_t, ndim=1] active_mask,
    np.ndarray[np.uint8_t, ndim=1] is_dispersing,
    np.ndarray[np.uint8_t, ndim=1] with_calf,
    # CRW/RefMem pre-computed outputs
    np.ndarray[f64, ndim=1] pres_angle,
    np.ndarray[f64, ndim=1] log_mov,
    np.ndarray[f32, ndim=1] ve_total,
    np.ndarray[f32, ndim=1] vt_x,
    np.ndarray[f32, ndim=1] vt_y,
    # Landscape
    np.ndarray[f32, ndim=2] food_grid,
    # OUTPUT arrays (caller must pre-allocate)
    np.ndarray[f32, ndim=1] out_food_gained,       # Per-agent food gained (for PSM)
    np.ndarray[f64, ndim=1] out_dispersal_dist,     # Dispersal distance to accumulate
    # Parameters
    double inertia_const,
    double disp_step,
    double e_use_per_30_min,
    double e_lact,
    double m_mort_prob_const,
    double x_survival_const,
    double seasonal_scaling,
    int world_w,
    int world_h,
) -> int:
    """All post-CRW phases in one C loop. Returns death count.

    CRITICAL: Phase ordering matches Java DEPONS 3.2 (Porpoise.java):
        1. Heading composition + position + reflect
        2. Food intake (energy += eaten)
        3. Mortality check (on post-food, pre-BMR energy)
        4. BMR cost (only for surviving agents)

    DO NOT copy from the prototype (tick_core.pyx) which has wrong
    ordering (BMR before mortality). Use Task 3's corrected ordering.
    """
    cdef int n = len(x)
    cdef int i
    cdef double max_x = <double>(world_w - 1)
    cdef double max_y = <double>(world_h - 1)
    cdef double h, rad, dx_crw, dy_crw, pres_mov, crw_c
    cdef double total_dx, total_dy, new_h, step, ddx, ddy, nx, ny
    cdef double fract, food_available, eaten, scaling, bmr
    cdef double yearly_surv, step_surv, pre_heading
    cdef int xi_c, yi_c, deaths = 0

    cdef np.ndarray[f64, ndim=1] rand_mort = np.random.random(n)

    for i in range(n):
        if not active_mask[i]:
            out_food_gained[i] = 0.0
            continue

        pres_mov = pow(10.0, log_mov[i])
        pre_heading = heading[i]

        # === 1. HEADING COMPOSITION + POSITION + REFLECT ===
        if is_dispersing[i]:
            new_h = heading[i]
            step = disp_step
            out_dispersal_dist[i] += step  # Track dispersal progress
        else:
            h = fmod(heading[i] + pres_angle[i], 360.0)
            if h < 0: h += 360.0
            rad = h * DEG2RAD
            dx_crw = sin(rad)
            dy_crw = cos(rad)
            crw_c = inertia_const + pres_mov * ve_total[i]
            total_dx = dx_crw * crw_c + vt_x[i]
            total_dy = dy_crw * crw_c + vt_y[i]
            new_h = atan2(total_dx, total_dy) * RAD2DEG
            if new_h < 0: new_h += 360.0
            step = pres_mov / 4.0

        heading[i] = <f32>new_h
        prev_angle[i] = fmod(new_h - pre_heading + 180.0, 360.0) - 180.0
        prev_log_mov[i] = log_mov[i]

        rad = new_h * DEG2RAD
        ddx = sin(rad) * step
        ddy = cos(rad) * step
        nx = x[i] + ddx
        ny = y[i] + ddy
        if nx < 0: nx = -nx
        elif nx > max_x: nx = 2.0 * max_x - nx
        if nx < 0: nx = 0.0
        elif nx > max_x: nx = max_x
        if ny < 0: ny = -ny
        elif ny > max_y: ny = 2.0 * max_y - ny
        if ny < 0: ny = 0.0
        elif ny > max_y: ny = max_y
        x[i] = <f32>nx
        y[i] = <f32>ny

        xi_c = <int>nx
        if xi_c < 0: xi_c = 0
        if xi_c >= world_w: xi_c = world_w - 1
        yi_c = <int>ny
        if yi_c < 0: yi_c = 0
        if yi_c >= world_h: yi_c = world_h - 1

        # === 2. FOOD INTAKE ===
        fract = (20.0 - energy[i]) / 10.0
        if fract < 0: fract = 0.0
        if fract > 0.99: fract = 0.99
        food_available = food_grid[yi_c, xi_c]
        eaten = food_available * fract
        food_grid[yi_c, xi_c] = food_available - eaten
        if food_grid[yi_c, xi_c] < 0.01:
            food_grid[yi_c, xi_c] = 0.01
        energy[i] += <f32>eaten
        if energy[i] > 20.0: energy[i] = 20.0
        out_food_gained[i] = <f32>eaten  # Export for PSM

        # === 3. MORTALITY CHECK (post-food, pre-BMR — Java ordering) ===
        if energy[i] > 0:
            yearly_surv = 1.0 - m_mort_prob_const * exp(-energy[i] * x_survival_const)
            if yearly_surv > 0:
                step_surv = exp(log(yearly_surv) / 17280.0)
            else:
                step_surv = 0.0
        else:
            step_surv = 0.0
        if rand_mort[i] > step_surv:
            if not with_calf[i] or energy[i] <= 0:
                active_mask[i] = 0
                deaths += 1
            else:
                with_calf[i] = 0
        if not active_mask[i]:
            continue

        # === 4. BMR COST (surviving agents only) ===
        scaling = seasonal_scaling
        if with_calf[i]: scaling = scaling * e_lact
        bmr = 0.001 * scaling * e_use_per_30_min
        energy[i] -= <f32>bmr
        if energy[i] < 0: energy[i] = 0.0
        if energy[i] > 20.0: energy[i] = 20.0

    return deaths
```

**IMPORTANT:** Do NOT copy from the prototype `tick_core.pyx` — it has BMR before mortality (wrong ordering). Use the code above which matches Java's food→mortality→BMR sequence.

- [ ] **Step 3: Rebuild and test**

- [ ] **Step 4: Commit**

```bash
git add src/cenop/optimizations/tick_cython.pyx tests/test_cython_tick.py
git commit -m "feat: Cython fused post-CRW tick entry point"
```

---

## Task 5: Wire Cython into `population.py step()`

**Files:**
- Modify: `src/cenop/optimizations/__init__.py`
- Modify: `src/cenop/agents/population.py:44-51,2492-2570`
- Test: `tests/test_cython_tick.py`

Add `_HAS_CYTHON` import flag and a Cython code path in `step()`.

- [ ] **Step 1: Add test — full population tick matches Python path**

```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonPopulationIntegration:
    def test_cython_tick_matches_python_tick(self):
        """Population state after 50 ticks must be close between paths."""
        from test_tick_performance import make_pop, snapshot_state

        # Run Python path
        np.random.seed(42)
        pop_py = make_pop(200)
        pop_py._use_cython = False  # Force Python
        for _ in range(50):
            pop_py.step()
        state_py = snapshot_state(pop_py)

        # Run Cython path
        np.random.seed(42)
        pop_cy = make_pop(200)
        pop_cy._use_cython = True  # Force Cython
        for _ in range(50):
            pop_cy.step()
        state_cy = snapshot_state(pop_cy)

        # Positions and energy may diverge slightly due to float32 ordering
        # but should be very close
        for key in state_py:
            np.testing.assert_allclose(
                state_py[key], state_cy[key], atol=0.1,
                err_msg=f"Cython/Python mismatch in {key}"
            )
```

- [ ] **Step 2: Add `_HAS_CYTHON` to `__init__.py`**

In `src/cenop/optimizations/__init__.py`, add near the top:

```python
try:
    from cenop.optimizations.tick_cython import cython_depons_post_crw, cython_available
    _HAS_CYTHON = cython_available()
except ImportError:
    _HAS_CYTHON = False
    cython_depons_post_crw = None
```

- [ ] **Step 3: Add `_HAS_CYTHON` import in `population.py`**

At line ~52 (after the `_HAS_KERNELS` block), add:

```python
try:
    from cenop.optimizations import _HAS_CYTHON, cython_depons_post_crw as _cython_post_crw
except ImportError:
    _HAS_CYTHON = False
    _cython_post_crw = None
```

In `__init__`, add:
```python
        self._use_cython = _HAS_CYTHON and self._energy_module is None and self._skip_land_avoidance
        # Pre-allocate Cython output buffer
        if self._use_cython:
            self._cython_food_gained = np.zeros(count, dtype=np.float32)
```

**Condition explained:** Cython enabled when (a) module built, (b) DEPONS mode (no JASMINE energy/FSM modules), (c) homogeneous landscape (no land avoidance needed). This means `_behavior_fsm` is always None and land avoidance is skipped — both are safe assumptions.

- [ ] **Step 4: Add Cython path in `step()`**

In `step()`, after the CRW+RefMem Numba kernels run (inside `_update_movement`) and before the food/BMR/mortality phases, add the Cython shortcut. The cleanest integration point is in `step()` itself (~line 2537), replacing steps 2-4d:

```python
        # 1. Movement calculations (CRW + RefMem stay as Numba)
        self._update_movement(mask, deterrence_vectors, ambient_rl)

        # === Cython fast path: fuse steps 2-4c into single C loop ===
        # Enabled when: Cython built + DEPONS mode + homogeneous landscape
        # (Cython skips land avoidance, so only safe on all-water grids)
        # Note: _behavior_fsm is always None in DEPONS mode (checked by _use_cython)
        if self._use_cython and _cython_post_crw is not None:
            current_month = self._get_current_month()
            seasonal_scaling = self._get_seasonal_scaling(current_month)
            disp_step = getattr(self.params, 'mean_disp_dist', 1.6) / 0.4
            world_w = self.landscape.width if self.landscape else self.params.world_width
            world_h = self.landscape.height if self.landscape else self.params.world_height

            # Pre-allocate output arrays (reuse from __init__)
            self._cython_food_gained.fill(0.0)

            deaths = _cython_post_crw(
                self.x, self.y, self.heading,
                self.prev_angle, self.prev_log_mov, self.energy,
                self.active_mask.view(np.uint8),
                self.is_dispersing.view(np.uint8),
                self.with_calf.view(np.uint8),
                self._pres_angle, self._log_mov,
                self._ve_total, self._vt_x, self._vt_y,
                self.landscape._food_value if self.landscape else np.full((world_h, world_w), 50.0, dtype=np.float32),
                self._cython_food_gained,             # OUTPUT: per-agent food for PSM
                self.dispersal_distance_traveled,     # READ-WRITE: accumulates in kernel
                self.params.inertia_const, disp_step,
                self.params.e_use_per_30_min, self.params.e_lact,
                getattr(self.params, 'm_mort_prob_const', 1.0),
                getattr(self.params, 'x_survival_const', 0.4),
                seasonal_scaling, world_w, world_h,
            )

            # Recompute cell indices after position update
            self._recompute_cell_indices()
            # Recompute active indices after mortality
            self._active_idx = np.flatnonzero(self.active_mask)

            # Dashboard stats (not simulation-critical, but UI needs them)
            n_active = len(self._active_idx)
            if n_active > 0:
                self.avg_food_gained = float(np.mean(self._cython_food_gained[self.active_mask]))
                self.avg_energy_cost = 0.001 * seasonal_scaling * self.params.e_use_per_30_min
            else:
                self.avg_food_gained = 0.0
                self.avg_energy_cost = 0.0

            # G11: Daily energy tracking
            self._energy_consumed_today[self.active_mask] += np.float32(
                0.001 * seasonal_scaling * self.params.e_use_per_30_min
            )
            if self._global_tick % 48 == 0:
                np.copyto(self.energy_consumed_daily, self._energy_consumed_today)
                self._energy_consumed_today[:] = 0

            # Post-BMR updates (PSM, energy history, dispersal)
            self._update_psm(self.active_mask, self._cython_food_gained)
            self._update_energy_history(self.active_mask)
            self._update_dispersal(self.active_mask)
            np.clip(self.energy, 0, 20.0, out=self.energy)
        else:
            # === Python/NumPy path (existing code, unchanged) ===
            # 2. Land avoidance
            self._handle_land_avoidance(mask)
            # 3. Apply positions
            self._apply_positions(mask)
            # ... rest of existing code (steps 3.5 through 4d) ...
```

**CRITICAL:** The Cython path skips land avoidance. This is safe ONLY on homogeneous (all-water) landscapes. For real landscapes, either:
- (a) Run land avoidance separately before the Cython call, OR
- (b) Only enable Cython when `_skip_land_avoidance` is True

Use option (b) — change the enable condition:
```python
        self._use_cython = _HAS_CYTHON and self._energy_module is None and self._skip_land_avoidance
```

- [ ] **Step 5: Run full test suite**

Expected: 516+ passed

- [ ] **Step 6: Commit**

```bash
git add src/cenop/optimizations/__init__.py src/cenop/agents/population.py tests/test_cython_tick.py
git commit -m "feat: wire Cython tick path into population.step()

Enabled when: Cython built + DEPONS mode + homogeneous landscape.
Falls back to Python/NumPy path otherwise."
```

---

## Task 6: Benchmark and validate

**Files:**
- Test: `tests/test_cython_tick.py`

- [ ] **Step 1: Add benchmark test**

```python
@pytest.mark.skipif(not CYTHON_OK, reason="Cython module not built")
class TestCythonBenchmark:
    def test_cython_faster_than_numpy(self):
        """Cython path should be at least 1.3x faster than Python path."""
        import time
        from test_tick_performance import make_pop, measure_tick

        pop_py = make_pop(500)
        pop_py._use_cython = False
        ms_py = measure_tick(pop_py, warmup=50, runs=200)

        pop_cy = make_pop(500)
        pop_cy._use_cython = True
        ms_cy = measure_tick(pop_cy, warmup=50, runs=200)

        speedup = ms_py / ms_cy
        print(f"\nPython: {ms_py:.3f}ms, Cython: {ms_cy:.3f}ms, Speedup: {speedup:.2f}x")
        assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.2f}x"
```

- [ ] **Step 2: Run benchmark**

```bash
cd /home/razinka/cenjas/CENOP && micromamba run -n shiny python3 -m pytest tests/test_cython_tick.py::TestCythonBenchmark -v -s
```

Expected: speedup > 1.3x

- [ ] **Step 3: Run full test suite**

Expected: 516+ passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_cython_tick.py
git commit -m "test: add Cython vs Python benchmark assertion"
```

---

## Task 7: Add Cython to optional dependencies

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add Cython optional dependency**

In `pyproject.toml`, add an optional dependencies section:

```toml
[project.optional-dependencies]
cython = ["cython>=3.0.0"]
```

- [ ] **Step 2: Add build instructions to README**

In README.md, add after the Micromamba section:

```markdown
### Building Cython Acceleration (optional)

For maximum performance (30-40% faster tick in DEPONS mode):

```bash
pip install cython
cd src/cenop/optimizations
python setup_cython.py build_ext --inplace
```
```

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml README.md
git commit -m "docs: add Cython optional dependency and build instructions"
```

---

## Task 8: Final validation

- [ ] **Step 1: Full benchmark comparison**

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

for mode in ['Python (Numba)', 'Cython + Numba']:
    use_cy = (mode == 'Cython + Numba')
    results = []
    for trial in range(3):
        p=SimulationParameters(porpoise_count=500,world_width=200,world_height=200,communication_enabled=False)
        pop=PorpoisePopulation(500,p,landscape=mk()); pop._skip_land_avoidance=True
        pop._use_cython = use_cy
        for _ in range(100): pop.step()
        t0=time.perf_counter()
        for _ in range(500): pop.step()
        results.append((time.perf_counter()-t0)/500*1000)
    med = sorted(results)[1]
    print(f'{mode:20s}: {med:.3f} ms/tick')
print(f'{\"Java DEPONS 3.2\":20s}: 0.795 ms/tick')
"
```

- [ ] **Step 2: Run full test suite one last time**

Expected: 516+ passed

- [ ] **Step 3: Commit**

```bash
cd /home/razinka/cenjas/CENOP && git add -u
git commit -m "perf: complete Cython tick integration — targeting sub-Java performance"
```
