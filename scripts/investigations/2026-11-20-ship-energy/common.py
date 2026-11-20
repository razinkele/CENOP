"""Shared bins and accumulators for the ship-energy comparison (A2 run 4, days 110-150)."""

import numpy as np

# (first day, last day) of each bin; a tick t belongs to day (t - 1) // 48 + 1
BINS = [(20, 30), (60, 70), (100, 110), (110, 120), (120, 130), (130, 140), (140, 150), (150, 160)]
KEYS = [
    "ticks",  # animal-ticks alive
    "exposed",  # ... with ship deterrence strength > 0
    "strength",  # sum of strength over exposed ticks
    "disp",  # animal-ticks dispersing
    "disp_exposed",  # ... dispersing and exposed
    "step_exp",  # sum of step length (cells), exposed and not dispersing
    "n_step_exp",
    "step_free",  # sum of step length (cells), not exposed and not dispersing
    "n_step_free",
    "e_exp",  # sum of energy on exposed ticks
    "e_free",  # sum of energy on unexposed ticks
]


def bin_of_day(day):
    for b, (lo, hi) in enumerate(BINS):
        if lo < day <= hi:
            return b
    return -1


def empty():
    return {k: np.zeros(len(BINS)) for k in KEYS}


def add(acc, b, alive, strength, dispersing, step, energy):
    """Add one tick (or any flat batch) of per-animal values belonging to bin ``b``."""
    exp = alive & (strength > 0)
    free = alive & ~exp
    acc["ticks"][b] += alive.sum()
    acc["exposed"][b] += exp.sum()
    acc["strength"][b] += strength[exp].sum()
    acc["disp"][b] += (alive & dispersing).sum()
    acc["disp_exposed"][b] += (exp & dispersing).sum()
    ok = np.isfinite(step) & ~dispersing
    acc["step_exp"][b] += step[exp & ok].sum()
    acc["n_step_exp"][b] += (exp & ok).sum()
    acc["step_free"][b] += step[free & ok].sum()
    acc["n_step_free"][b] += (free & ok).sum()
    acc["e_exp"][b] += energy[exp].sum()
    acc["e_free"][b] += energy[free].sum()


def table(label, acc):
    print(label)
    print(
        "  days      exposed  strength  dispersing  disp|exposed  step exp  step free  E exp   E free  E all"
    )
    for b, (lo, hi) in enumerate(BINS):
        t = acc["ticks"][b]
        if t == 0:
            continue
        ex = acc["exposed"][b]
        print(
            f"  {lo:3d}-{hi:3d}   {ex / t:6.3f}   {acc['strength'][b] / max(ex, 1):7.4f}   "
            f"{acc['disp'][b] / t:8.3f}   {acc['disp_exposed'][b] / max(ex, 1):10.3f}   "
            f"{acc['step_exp'][b] / max(acc['n_step_exp'][b], 1):7.3f}   "
            f"{acc['step_free'][b] / max(acc['n_step_free'][b], 1):8.3f}   "
            f"{acc['e_exp'][b] / max(ex, 1):5.2f}   {acc['e_free'][b] / max(t - ex, 1):5.2f}   "
            f"{(acc['e_exp'][b] + acc['e_free'][b]) / t:5.2f}"
        )
