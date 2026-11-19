"""A2 metrics and pass bands (pure functions; no file access).

The bands were fixed on 2026-11-19, before the disturbed scenario was run with more than one
seed. Design: docs/superpowers/specs/2026-11-19-a2-cross-validation-design.md

``java`` and ``cenop`` arguments are arrays with one row per seed unless stated otherwise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

PASS, FAIL, INCONCLUSIVE = "PASS", "FAIL", "INCONCLUSIVE"

# Bands
MAX_SE = 2.0  # |difference of means| in standard errors
MAX_REL_POPULATION = 0.10  # M1: |difference| relative to the Java mean
MAX_ABS_ENERGY = 0.25  # M3
MIN_DAYS_INSIDE = 0.95  # M2: share of days with the CENOP mean inside Java mean +/- 2 SD
MAX_TVD = 0.06  # M4: total variation distance between mean block shares
MIN_CORRELATION = 0.98  # M4
MIN_KS_P = 0.01  # M5
MAX_REL_EXPOSURE = 0.20  # M7
DETECTABLE = 0.10  # a result is inconclusive when 2 SE exceeds this share of the Java mean


@dataclass
class Row:
    """One checkpoint of one metric."""

    label: str
    java: float
    cenop: float
    diff: float
    se: float
    verdict: str
    note: str = ""

    @property
    def in_se(self) -> float:
        return self.diff / self.se if self.se > 0 else 0.0


def _means(java: np.ndarray, cenop: np.ndarray) -> tuple[float, float, float, float]:
    j, c = float(np.mean(java)), float(np.mean(cenop))
    se = float(np.sqrt(np.var(java, ddof=1) / len(java) + np.var(cenop, ddof=1) / len(cenop)))
    return j, c, c - j, se


def checkpoint(
    label: str,
    java: np.ndarray,
    cenop: np.ndarray,
    *,
    max_rel: float | None = None,
    max_abs: float | None = None,
) -> Row:
    """Compare seed samples at one checkpoint.

    Passes when the difference of means is within MAX_SE standard errors and within the
    relative (``max_rel``, of the Java mean) or absolute (``max_abs``) limit. A pass is
    downgraded to INCONCLUSIVE when 2 SE is larger than that limit: the comparison could not
    have shown a difference of the size the band forbids.
    """
    j, c, diff, se = _means(java, cenop)
    limit = max_abs if max_abs is not None else (max_rel or 0.0) * abs(j)
    ok = abs(diff) <= MAX_SE * se and abs(diff) <= limit
    verdict = PASS if ok else FAIL
    note = ""
    if ok and MAX_SE * se > limit:
        verdict, note = INCONCLUSIVE, "2 SE exceeds the band: too noisy to resolve it"
    return Row(label, j, c, diff, se, verdict, note)


def population_rows(
    java: np.ndarray, cenop: np.ndarray, days: list[int], labels: list[str]
) -> list[Row]:
    """M1. ``java``/``cenop``: (seeds, days + 1) daily population, column d = end of day d."""
    return [
        checkpoint(lab, java[:, d], cenop[:, d], max_rel=MAX_REL_POPULATION)
        for d, lab in zip(days, labels)
    ]


def energy_rows(java: np.ndarray, cenop: np.ndarray, days: list[int]) -> list[Row]:
    """M3. Daily population-wide mean energy, same layout as population_rows."""
    return [checkpoint(f"day {d}", java[:, d], cenop[:, d], max_abs=MAX_ABS_ENERGY) for d in days]


def trajectory(java: np.ndarray, cenop: np.ndarray, first_day: int = 1) -> tuple[float, str]:
    """M2. Share of days on which the CENOP mean lies inside Java mean +/- 2 SD.

    The Java SD is floored at 1 animal, so days on which all Java seeds agree exactly (the
    first days, before anyone dies) do not fail on a difference of one animal.
    """
    j_mean = java[:, first_day:].mean(axis=0)
    j_sd = np.maximum(java[:, first_day:].std(axis=0, ddof=1), 1.0)
    c_mean = cenop[:, first_day:].mean(axis=0)
    inside = float(np.mean(np.abs(c_mean - j_mean) <= 2.0 * j_sd))
    return inside, (PASS if inside >= MIN_DAYS_INSIDE else FAIL)


def block_shares(per_block: np.ndarray, first_day: int, last_day: int) -> np.ndarray:
    """Mean share of animals per block over a window. ``per_block``: (days + 1, blocks)."""
    w = per_block[first_day : last_day + 1]
    total = w.sum(axis=1, keepdims=True)
    return (w / np.where(total == 0, 1, total)).mean(axis=0)


def location(java_shares: np.ndarray, cenop_shares: np.ndarray) -> tuple[float, float, str]:
    """M4. ``*_shares``: (seeds, blocks). Total variation distance and correlation of the means."""
    j, c = java_shares.mean(axis=0), cenop_shares.mean(axis=0)
    tvd = float(0.5 * np.abs(j - c).sum())
    corr = float(np.corrcoef(j, c)[0, 1])
    return tvd, corr, (PASS if tvd <= MAX_TVD and corr >= MIN_CORRELATION else FAIL)


def age_at_death(java_ages: np.ndarray, cenop_ages: np.ndarray) -> tuple[float, float, str]:
    """M5. Two-sample Kolmogorov-Smirnov test on pooled ages at death (whole years)."""
    from scipy import stats

    res = stats.ks_2samp(java_ages, cenop_ages)
    return float(res.statistic), float(res.pvalue), (PASS if res.pvalue > MIN_KS_P else FAIL)


def ratio_row(
    label: str,
    java_num: np.ndarray,
    java_den: np.ndarray,
    cenop_num: np.ndarray,
    cenop_den: np.ndarray,
) -> Row:
    """M6. Ratio of means (disturbed / undisturbed) per model, SE by the delta method."""

    def ratio(num, den):
        r = np.mean(num) / np.mean(den)
        var = r**2 * (
            np.var(num, ddof=1) / len(num) / np.mean(num) ** 2
            + np.var(den, ddof=1) / len(den) / np.mean(den) ** 2
        )
        return float(r), float(var)

    rj, vj = ratio(java_num, java_den)
    rc, vc = ratio(cenop_num, cenop_den)
    se = float(np.sqrt(vj + vc))
    diff = rc - rj
    return Row(label, rj, rc, diff, se, PASS if abs(diff) <= MAX_SE * se else FAIL)


def exposure_row(java: np.ndarray, cenop: np.ndarray) -> Row:
    """M7. Share of animal-ticks with ship deterrence, one value per seed."""
    j, c, diff, se = _means(java, cenop)
    ok = abs(diff) <= MAX_REL_EXPOSURE * abs(j)
    return Row("share of animal-ticks with ship deterrence", j, c, diff, se, PASS if ok else FAIL)


def overall(verdicts: list[str]) -> str:
    """A metric fails if any checkpoint fails, is inconclusive if any is, else passes."""
    if FAIL in verdicts:
        return FAIL
    if INCONCLUSIVE in verdicts:
        return INCONCLUSIVE
    return PASS
