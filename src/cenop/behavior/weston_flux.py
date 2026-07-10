"""WestonFlux physics-based transmission loss model for ship noise.

Ported from: WestonFlux.java (DEPONS 3.2)
Original: Copyright (C) 2022-2023 Jacob Nabe-Nielsen <jnn@bios.au.dk> (GPL v2)

Computes propagation loss in shallow water using the Weston flux integral
method with range-independent bathymetry. Accounts for:
- Bottom reflection loss (sediment grain size → sound speed ratio, density ratio)
- Water absorption (Francois-Garrison equation)
- Geometric spreading (r^(-3/2) cylindrical-to-spherical transition)
"""

import math

try:
    from numba import njit

    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):
        def decorator(func):
            return func

        return decorator


# Default frequency: 10^(12/10) * 1000 Hz ≈ 15848.93 Hz
FREQUENCY = 10.0 ** (12.0 / 10.0) * 1000.0
# Sound speed in sediment (m/s)
SPEED_IN_SEDIMENT = 1700.0
# pH of seawater
PH = 8.0


@njit(cache=True)
def weston_flux_tl(
    distance: float,
    depth: float,
    grain_size: float,
    temperature: float,
    salinity: float,
    frequency: float = FREQUENCY,
    ph: float = 8.0,
    c_s: float = 1700.0,
) -> float:
    """Calculate transmission loss using WestonFlux model.

    Args:
        distance: Distance from source in meters
        depth: Water depth at source in meters
        grain_size: Sediment grain size (phi scale, typically -1 to 9)
        temperature: Water temperature in °C
        salinity: Salinity in PSU (practical salinity units)
        frequency: Sound frequency in Hz (default: DEPONS 3.2 value)

    Returns:
        Propagation loss in dB re 1m²
    """
    if distance <= 0 or depth <= 0:
        return 0.0

    ssp_ratio = _ssp_ratio(grain_size)
    beta_s = _beta(grain_size, ssp_ratio)
    gamma_w = _gamma(frequency, temperature, salinity, ph, 0.0)
    rho_ratio = _rho_ratio(grain_size)

    return _range_independent(
        distance, depth, frequency, c_s, beta_s, gamma_w, ssp_ratio, rho_ratio
    )


@njit(cache=True)
def _range_independent(
    r: float,
    h: float,
    f: float,
    c_s: float,
    beta_s: float,
    gamma_w: float,
    ssp_ratio: float,
    rho_ratio: float,
) -> float:
    """Compute propagation loss using range-independent Weston flux integral."""
    # Convert water absorption from dB/m to Np/m
    alpha_w = math.log(10) / 20.0 * gamma_w

    # Clamp SSP ratio
    if ssp_ratio < 1.04:
        ssp_ratio = 1.04

    # Determine eta (effective bottom loss parameter)
    if ssp_ratio > 1:
        epsilon = math.log(10) / (40.0 * math.pi) * beta_s
        eta = 2.0 * rho_ratio * (ssp_ratio / ((ssp_ratio**2 - 1) ** 1.5)) * epsilon
    elif ssp_ratio < 1:
        eta = 2.0 * rho_ratio * ssp_ratio / math.sqrt(1 - ssp_ratio**2)
    else:
        eta = 0.0

    if eta <= 0:
        # Avoid division by zero / invalid erf argument
        return 0.0

    # Critical angle (range-independent: theta_limit == theta_crit)
    theta_crit = math.acos(1.0 / ssp_ratio)
    theta_limit = theta_crit

    # Effective water depth (range-independent)
    h_eff = h

    # Propagation factor F
    erf_arg = math.sqrt(eta * r / h) * theta_limit
    # Clamp erf argument to avoid overflow (erf saturates near ±6)
    erf_arg = min(erf_arg, 6.0)

    f_val = (
        r ** (-1.5)
        * math.sqrt(math.pi / (eta * h_eff))
        * math.erf(erf_arg)
        * math.exp(-2.0 * alpha_w * r)
    )

    if f_val <= 0:
        return 300.0  # Very large loss for effectively zero propagation

    # Propagation loss in dB re 1m²
    pl = -10.0 * math.log10(f_val)
    return pl


@njit(cache=True)
def _ssp_ratio(grain_size: float) -> float:
    """Sound speed ratio (high frequency) from sediment grain size."""
    gs = grain_size
    if gs < -8.0:
        gs = -8.0  # Clamp rather than raise
    if gs < 1.0:
        return 1.2778 - 0.056452 * gs + 0.002709 * gs**2
    elif gs < 5.3:
        return 1.3425 - 0.1382798 * gs + 0.0213937 * gs**2 - 0.0014881 * gs**3
    elif gs <= 9.0:
        return 1.0019 - 0.0024324 * gs
    else:
        return 1.0019 - 0.0024324 * 9.0  # Capped at 9


@njit(cache=True)
def _rho_ratio(grain_size: float) -> float:
    """Density ratio (high frequency) from sediment grain size."""
    gs = grain_size
    if gs < -8.0:
        gs = -8.0
    if gs < 1.0:
        return 2.3139 - 0.17057 * gs + 0.007797 * gs**2
    elif gs < 5.3:
        return 3.0455 - 1.1069031 * gs + 0.2290201 * gs**2 - 0.0165406 * gs**3
    elif gs <= 9.0:
        return 1.1565 - 0.0012973 * gs
    else:
        return 1.1565 - 0.0012973 * 9.0


@njit(cache=True)
def _beta(grain_size: float, ssp_ratio_high: float) -> float:
    """Sediment attenuation coefficient (high frequency)."""
    gs = grain_size
    if gs < -8.0:
        gs = -8.0
    if gs < 0.0:
        return 1.490 * 0.4556
    elif gs < 2.6:
        return 1.490 * ssp_ratio_high * (0.4556 + 0.0245 * gs)
    elif gs < 4.5:
        return 1.490 * ssp_ratio_high * (0.1978 + 0.1245 * gs)
    elif gs < 6.0:
        return 1.490 * ssp_ratio_high * (8.0399 - 2.5228 * gs + 0.20098 * gs**2)
    elif gs < 9.5:
        return 1.490 * ssp_ratio_high * (0.9431 - 0.2041 * gs + 0.0117 * gs**2)
    else:
        return 1.490 * ssp_ratio_high * 0.0601


@njit(cache=True)
def _gamma(f: float, temp: float, salinity: float, ph: float, depth_at_source: float) -> float:
    """Water absorption coefficient (dB/m) using Francois-Garrison equation."""
    f1 = 0.91 * (salinity / 35.0) ** 0.5 * math.exp(temp / 33.0)
    f2 = 46.6 * math.exp(temp / 18.0)

    if temp <= 20:
        a3 = 4.937e-4 - 2.59e-5 * temp + 9.11e-7 * temp**2 - 1.5e-8 * temp**3
    else:
        a3 = 3.964e-4 - 1.146e-5 * temp + 1.45e-7 * temp**2 - 6.5e-10 * temp**3

    p3 = 1 - 3.83e-5 * depth_at_source + 4.9e-4 * (depth_at_source / 1000.0) ** 2

    f_khz = f / 1000.0

    y1 = 0.101 * (f1 * f_khz**2) / (f1**2 + f_khz**2) * math.exp((ph - 8) / 0.57)
    y2 = (
        0.56
        * (1 + temp / 76.0)
        * (salinity / 35.0)
        * (f2 * f_khz**2)
        / (f2**2 + f_khz**2)
        * math.exp(-depth_at_source / 4900.0)
    )
    y3 = a3 * p3 * f_khz**2

    absorption = (y1 + y2 + y3) / 1000.0
    return absorption
