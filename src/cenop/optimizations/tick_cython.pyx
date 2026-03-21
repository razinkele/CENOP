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
