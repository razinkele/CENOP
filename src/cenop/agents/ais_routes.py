"""Build DEPONS ship routes (ships.json) from raw AIS positions.

Follows the encoding of DEPONS's Kattegat ship file, which was built from Danish AIS: every
route is one passage of one ship, sampled at 30-minute ticks (one buoy per tick, speed in
knots); a stationary spell is one buoy with a pause; the pause at the first buoy places the
passage in time and the pause at the last buoy fills the rest of the cycle, so every route
spans the same number of ticks (721 in Kattegat: 15 days plus one) and the scenario repeats.

In CENOP (as in DEPONS Ship.move) a ship arriving at a buoy with pause p stays there for 1 + p
ticks and does not deter while it waits; a first-buoy pause of s therefore puts the ship at its
first real position in tick s.
"""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

TICK_S = 1800
SOG_UNAVAILABLE = 102.0  # AIS: 102.3 knots means "not available"


def ais_type_to_class(code) -> str:
    """DEPONS ship-type string (JOMOPANS class) for an AIS ship-type code.

    AIS codes cannot tell bulk carriers, container ships and vehicle carriers apart (70-79),
    nor cruise ships from ferries (60-69); they map to Bulker and Passenger.
    """
    try:
        c = int(code)
    except (TypeError, ValueError):
        return "Other"
    if c == 30:
        return "Fishing"
    if c in (31, 32, 52):
        return "Tug"
    if c == 33:
        return "Dredger"
    if c == 35:
        return "Naval"
    if c in (36, 37):
        return "Recreational"
    if 40 <= c <= 49 or 60 <= c <= 69:
        return "Passenger"
    if c in (50, 51, 53, 54, 55, 58):
        return "Government/Research"
    if 70 <= c <= 79:
        return "Bulker"
    if 80 <= c <= 89:
        return "Tanker"
    return "Other"


def split_passages(t: np.ndarray, max_gap_s: float = 7200.0) -> list[slice]:
    """Slices of a sorted time series broken where two positions are more than max_gap_s apart."""
    if t.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(t) > max_gap_s) + 1
    edges = [0, *breaks.tolist(), t.size]
    return [slice(a, b) for a, b in zip(edges[:-1], edges[1:]) if b > a]


def route_from_passage(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sog: np.ndarray,
    t0: float,
    cycle_ticks: int = 721,
    still_m: float = 50.0,
    still_kn: float = 0.5,
    min_move_m: float = 500.0,
) -> list[dict] | None:
    """Buoys for one passage (positions sorted by time, in metres), or None if it is too short
    (fewer than two ticks) or the ship never moves (below 1 knot and within min_move_m)."""
    first = max(math.ceil((t[0] - t0) / TICK_S), 0)
    last = min(math.floor((t[-1] - t0) / TICK_S), cycle_ticks - 2)
    if last - first < 1:
        return None
    ticks = np.arange(first, last + 1)
    tt = t0 + ticks * TICK_S
    xs = np.interp(tt, t, x)
    ys = np.interp(tt, t, y)
    s = np.where(np.isfinite(sog) & (sog < SOG_UNAVAILABLE), sog, np.nan)
    ok = np.isfinite(s)
    if ok.sum() >= 2:
        ss = np.interp(tt, t[ok], s[ok])
    else:  # no usable speed over ground: from the displacement between ticks, in knots
        step = np.hypot(np.diff(xs), np.diff(ys)) / TICK_S * 3600 / 1852
        ss = np.concatenate([step, step[-1:]])
    extent = max(np.ptp(xs), np.ptp(ys))
    if np.nanmax(ss) < 1.0 and extent < min_move_m:
        return None
    buoys: list[dict] = []
    for k in range(ticks.size):
        if buoys and ss[k] < still_kn:
            b = buoys[-1]
            if math.hypot(xs[k] - b["x"], ys[k] - b["y"]) < still_m:
                b["pause"] += 1
                continue
        buoys.append(
            {
                "x": round(float(xs[k]), 2),
                "y": round(float(ys[k]), 2),
                "speed": round(float(ss[k]), 3),
                "pause": 0,
            }
        )
    buoys[0]["pause"] += int(first)
    used = sum(1 + b["pause"] for b in buoys)
    buoys[-1]["pause"] += cycle_ticks - used
    if len(buoys) == 1:  # a single buoy cannot sweep anywhere
        return None
    return buoys


def build_ships(
    mmsi: np.ndarray,
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    sog: np.ndarray,
    ais_type: np.ndarray,
    t0: float,
    cycle_ticks: int = 721,
    lengths: dict | None = None,
    default_lengths: dict | None = None,
    max_gap_s: float = 7200.0,
) -> dict:
    """ships.json content ({"routes": [...], "ships": [...]}) from AIS rows of any order.

    ``lengths`` maps MMSI to a known length in metres; otherwise the ship gets
    ``default_lengths[type]`` (or 50 m).
    """
    lengths = lengths or {}
    default_lengths = default_lengths or {}
    order = np.lexsort((t, mmsi))
    mmsi, t, x, y, sog, ais_type = (a[order] for a in (mmsi, t, x, y, sog, ais_type))
    routes, ships = [], []
    starts = np.flatnonzero(np.r_[True, mmsi[1:] != mmsi[:-1]])
    ends = np.r_[starts[1:], mmsi.size]
    for a, b in zip(starts, ends):
        vid = int(mmsi[a])
        codes = [c for c in ais_type[a:b] if c is not None and np.isfinite(c) and c > 0]
        vtype = ais_type_to_class(Counter(codes).most_common(1)[0][0]) if codes else "Other"
        length = lengths.get(vid) or default_lengths.get(vtype, 50)
        tv, xv, yv, sv = t[a:b], x[a:b], y[a:b], sog[a:b]
        for n, sl in enumerate(split_passages(tv, max_gap_s)):
            buoys = route_from_passage(tv[sl], xv[sl], yv[sl], sv[sl], t0, cycle_ticks)
            if buoys is None:
                continue
            name = f"{vid}-{n + 1}"
            routes.append({"name": f"Route_{name}", "route": buoys})
            ships.append(
                {
                    "name": name,
                    "type": vtype,
                    "length": int(round(length)),
                    "route": f"Route_{name}",
                }
            )
    return {"routes": routes, "ships": ships}
