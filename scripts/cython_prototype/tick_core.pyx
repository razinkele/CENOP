# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython prototype: fused DEPONS tick core.

Replaces the Python/NumPy glue between Numba kernels with a single
compiled C loop. Benchmarks whether eliminating Python dispatch
overhead closes the remaining gap to Java.
"""
import numpy as np
cimport numpy as np
from libc.math cimport sin, cos, atan2, sqrt, pow, log, exp, fabs, fmod
from libc.math cimport M_PI

ctypedef np.float64_t DTYPE_f64
ctypedef np.float32_t DTYPE_f32
ctypedef np.int32_t DTYPE_i32

cdef double DEG2RAD = M_PI / 180.0
cdef double RAD2DEG = 180.0 / M_PI


def fused_depons_tick(
    # Agent state (read-write)
    np.ndarray[DTYPE_f32, ndim=1] x,
    np.ndarray[DTYPE_f32, ndim=1] y,
    np.ndarray[DTYPE_f32, ndim=1] heading,
    np.ndarray[DTYPE_f64, ndim=1] prev_angle,
    np.ndarray[DTYPE_f64, ndim=1] prev_log_mov,
    np.ndarray[DTYPE_f32, ndim=1] energy,
    np.ndarray[np.uint8_t, ndim=1] active_mask,  # bool as uint8
    np.ndarray[np.uint8_t, ndim=1] is_dispersing,
    np.ndarray[np.uint8_t, ndim=1] with_calf,
    # CRW output (pre-computed by Numba kernel)
    np.ndarray[DTYPE_f64, ndim=1] pres_angle,
    np.ndarray[DTYPE_f64, ndim=1] log_mov,
    # RefMem output (pre-computed by Numba kernel)
    np.ndarray[DTYPE_f32, ndim=1] ve_total,
    np.ndarray[DTYPE_f32, ndim=1] vt_x,
    np.ndarray[DTYPE_f32, ndim=1] vt_y,
    # Landscape (read-only)
    np.ndarray[DTYPE_f32, ndim=2] depth_grid,
    np.ndarray[DTYPE_f32, ndim=2] food_grid,
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
):
    """
    Fused DEPONS tick phases: heading composition + position + reflect +
    food intake + BMR cost + mortality check.

    CRW angle/step and RefMem are pre-computed by Numba kernels (they
    already run at near-native speed). This function replaces the Python
    glue code between those kernels and the energy phases.
    """
    cdef int n = len(x)
    cdef int i
    cdef double max_x = <double>(world_w - 1)
    cdef double max_y = <double>(world_h - 1)
    cdef double h, rad, dx_crw, dy_crw, pres_mov, crw_c
    cdef double total_dx, total_dy, new_h, step, ddx, ddy, nx, ny
    cdef double fract, food_available, eaten, scaling, bmr
    cdef double yearly_surv, step_surv, rand_val
    cdef int xi_c, yi_c
    cdef double pre_heading

    # Pre-generate random numbers (vectorized, fast)
    cdef np.ndarray[DTYPE_f64, ndim=1] rand_mort = np.random.random(n)

    cdef int deaths = 0

    for i in range(n):
        if not active_mask[i]:
            continue

        pres_mov = pow(10.0, log_mov[i])
        pre_heading = heading[i]

        # === HEADING COMPOSITION ===
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

        heading[i] = <DTYPE_f32>new_h

        # prev_angle update
        prev_angle[i] = fmod(new_h - pre_heading + 180.0, 360.0) - 180.0
        prev_log_mov[i] = log_mov[i]

        # === POSITION UPDATE + REFLECT ===
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

        x[i] = <DTYPE_f32>nx
        y[i] = <DTYPE_f32>ny

        # === CELL INDEX ===
        xi_c = <int>nx
        if xi_c < 0: xi_c = 0
        if xi_c >= world_w: xi_c = world_w - 1
        yi_c = <int>ny
        if yi_c < 0: yi_c = 0
        if yi_c >= world_h: yi_c = world_h - 1

        # === FOOD INTAKE ===
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

        energy[i] += <DTYPE_f32>eaten
        if energy[i] > 20.0:
            energy[i] = 20.0

        # === BMR COST ===
        scaling = seasonal_scaling
        if with_calf[i]:
            scaling *= e_lact
        bmr = 0.001 * scaling * e_use_per_30_min
        energy[i] -= <DTYPE_f32>bmr

        # === MORTALITY CHECK ===
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
                with_calf[i] = 0  # abandon calf

        # Clamp energy
        if energy[i] < 0:
            energy[i] = 0.0
        if energy[i] > 20.0:
            energy[i] = 20.0

    return deaths
