"""JOMOPANS calibrated ship source level model.

Ported from: JomopansEchoSPL.java (DEPONS 3.2)
Original: Copyright (C) 2022-2023 Jacob Nabe-Nielsen <jnn@bios.au.dk> (GPL v2)

Computes broadband source level for 13 JOMOPANS vessel classes using
the ECHO/JOMOPANS spectral model. The model uses vessel-class-specific
reference speeds, damping coefficients, and cargo hump flags.
"""

import math

from cenop.agents.ship import VesselClass

# Default frequency band: 10^(12/10) * 1000 Hz ≈ 15848.93 Hz
DEFAULT_BAND = 12

# Reference vessel speeds (knots) per class — Java lookupVC()
_VC_SPEED = {
    VesselClass.BULKER: 13.9,
    VesselClass.CARGO: 18.0,  # Same as CONTAINERSHIP
    VesselClass.CHEMICAL_TANKER: 12.4,  # Same as TANKER
    VesselClass.CONTAINER: 18.0,
    VesselClass.CRUISE: 17.1,
    VesselClass.DREDGER: 9.5,
    VesselClass.FISHING: 6.4,
    VesselClass.GOVERNMENT: 8.0,
    VesselClass.NAVAL: 11.1,
    VesselClass.PASSENGER: 9.7,
    VesselClass.RECREATIONAL: 10.6,
    VesselClass.TANKER: 12.4,
    VesselClass.TUG: 3.7,
    VesselClass.VEHICLE_CARRIER: 15.8,
    VesselClass.OTHER: 7.4,
}

# Is this a "cargo" class with low-frequency hump? — Java lookupCargo()
_IS_CARGO = {
    VesselClass.BULKER: True,
    VesselClass.CARGO: True,
    VesselClass.CHEMICAL_TANKER: False,
    VesselClass.CONTAINER: True,
    VesselClass.CRUISE: False,
    VesselClass.DREDGER: False,
    VesselClass.FISHING: False,
    VesselClass.GOVERNMENT: False,
    VesselClass.NAVAL: False,
    VesselClass.PASSENGER: False,
    VesselClass.RECREATIONAL: False,
    VesselClass.TANKER: True,
    VesselClass.TUG: False,
    VesselClass.VEHICLE_CARRIER: True,
    VesselClass.OTHER: False,
}

# Low-frequency damping — Java lookupDlo()
_DLO = {
    VesselClass.BULKER: 0.8,
    VesselClass.CARGO: 0.8,
    VesselClass.CONTAINER: 0.8,
}
# All others default to 1.0

# High-frequency damping — Java lookupDhi()
_DHI = {
    VesselClass.CRUISE: 4.0,
}
# All others default to 3.0

# Reference length: 300 ft in meters
L_REF = 300.0 / 3.28084

# Minimum vessel length (m). A non-positive length from a malformed ship file / ships.json
# is clamped to this floor so the math.log10(length / L_REF) term below cannot raise a
# "math domain error" and crash the per-tick source-level call (defense-in-depth; load-time
# validation in ShipManager.load_from_json is the primary guard).
_MIN_LENGTH_M = 1.0


def jomopans_spl(
    vessel_class: VesselClass,
    speed_knots: float,
    length_m: float,
    band: int = DEFAULT_BAND,
) -> float:
    """Calculate source level using JOMOPANS/ECHO polynomial model.

    Args:
        vessel_class: One of the 15 VesselClass enum values
        speed_knots: Ship speed in knots
        length_m: Ship length in meters
        band: Decidecade band number (default 12, → ~15849 Hz)

    Returns:
        Source level in dB re 1 µPa @ 1m
    """
    if speed_knots == 0:
        return 0.0

    # Defense-in-depth: clamp a non-positive length to a positive floor before the
    # log10(length_eff / L_REF) term so a bad ship file cannot crash the simulation.
    length_eff = length_m if length_m > 0.0 else _MIN_LENGTH_M

    frequency = 10.0 ** (band / 10.0) * 1000.0
    d_vc = _VC_SPEED.get(vessel_class, 7.4)
    is_cargo = _IS_CARGO.get(vessel_class, False)
    lf_hump = is_cargo and (frequency < 100)

    i_val = 208.0 if lf_hump else 191.0
    j_val = 2.0 if lf_hump else 0.0
    k_val = _DLO.get(vessel_class, 1.0) if lf_hump else _DHI.get(vessel_class, 3.0)
    l_val = (600.0 / d_vc) if lf_hump else (480.0 / d_vc)

    # Spectral density
    sp = (
        i_val
        - 10.0 * (j_val + 2) * math.log10(l_val)
        + 5.0 * j_val * math.log10(frequency)
        - 10.0 * math.log10((1 - (frequency / l_val) ** (0.5 * (j_val + 2))) ** 2 + k_val**2)
        + 60.0 * math.log10(speed_knots / d_vc)
        + 20.0 * math.log10(length_eff / L_REF)
    )

    # Convert spectral density to decidecade band level
    return sp + 10.0 * math.log10(0.231 * frequency)
