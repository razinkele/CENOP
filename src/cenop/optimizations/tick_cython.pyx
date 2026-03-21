# cython: boundscheck=False, wraparound=False, cdivision=True, language_level=3
"""
Cython-accelerated DEPONS tick phases.

Replaces the Python/NumPy glue between Numba CRW/RefMem kernels with
compiled C loops. Provides ~3.7x speedup for heading+position+food+BMR+mortality.
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


def cython_depons_post_crw(
    # Agent state (read-write, modified in-place)
    np.ndarray[f32, ndim=1] x,
    np.ndarray[f32, ndim=1] y,
    np.ndarray[f32, ndim=1] heading,
    np.ndarray[f64, ndim=1] prev_angle,
    np.ndarray[f64, ndim=1] prev_log_mov,
    np.ndarray[f32, ndim=1] energy,
    np.ndarray[np.uint8_t, ndim=1] active_mask,
    np.ndarray[np.uint8_t, ndim=1] is_dispersing,
    np.ndarray[np.uint8_t, ndim=1] with_calf,
    # CRW/RefMem pre-computed (read-only)
    np.ndarray[f64, ndim=1] pres_angle,
    np.ndarray[f64, ndim=1] log_mov,
    np.ndarray[f32, ndim=1] ve_total,
    np.ndarray[f32, ndim=1] vt_x,
    np.ndarray[f32, ndim=1] vt_y,
    # Landscape (read-write for food consumption)
    np.ndarray[f32, ndim=2] food_grid,
    # OUTPUT arrays (caller pre-allocates)
    np.ndarray[f32, ndim=1] out_food_gained,
    np.ndarray[f32, ndim=1] dispersal_distance_traveled,
    # Scalar parameters
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
    """All post-CRW DEPONS phases in one compiled C loop.

    Phase ordering (matches Java Porpoise.java):
      1. Heading composition + position + boundary reflect
      2. Food intake (energy += eaten)
      3. Mortality check (on post-food, pre-BMR energy)
      4. BMR cost (surviving agents only)

    Returns: number of deaths this tick.
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

    # Pre-generate mortality random draws (vectorized NumPy, fast)
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
            dispersal_distance_traveled[i] += <f32>step
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

        heading[i] = <f32>new_h
        prev_angle[i] = fmod(new_h - pre_heading + 180.0, 360.0) - 180.0
        prev_log_mov[i] = log_mov[i]

        rad = new_h * DEG2RAD
        ddx = sin(rad) * step
        ddy = cos(rad) * step
        nx = x[i] + ddx
        ny = y[i] + ddy

        if nx < 0:
            nx = -nx
        elif nx > max_x:
            nx = 2.0 * max_x - nx
        if nx < 0:
            nx = 0.0
        elif nx > max_x:
            nx = max_x
        if ny < 0:
            ny = -ny
        elif ny > max_y:
            ny = 2.0 * max_y - ny
        if ny < 0:
            ny = 0.0
        elif ny > max_y:
            ny = max_y

        x[i] = <f32>nx
        y[i] = <f32>ny

        # Cell index
        xi_c = <int>nx
        if xi_c < 0: xi_c = 0
        if xi_c >= world_w: xi_c = world_w - 1
        yi_c = <int>ny
        if yi_c < 0: yi_c = 0
        if yi_c >= world_h: yi_c = world_h - 1

        # === 2. FOOD INTAKE ===
        fract = (20.0 - energy[i]) / 10.0
        if fract < 0:
            fract = 0.0
        if fract > 0.99:
            fract = 0.99
        food_available = food_grid[yi_c, xi_c]
        eaten = food_available * fract
        food_grid[yi_c, xi_c] = food_available - eaten
        if food_grid[yi_c, xi_c] < 0.01:
            food_grid[yi_c, xi_c] = 0.01
        energy[i] += <f32>eaten
        if energy[i] > 20.0:
            energy[i] = 20.0
        out_food_gained[i] = <f32>eaten

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
        if with_calf[i]:
            scaling = scaling * e_lact
        bmr = 0.001 * scaling * e_use_per_30_min
        energy[i] -= <f32>bmr
        if energy[i] < 0:
            energy[i] = 0.0
        if energy[i] > 20.0:
            energy[i] = 20.0

    return deaths
