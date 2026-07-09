"""Pure population-statistics snapshot builder and renderer helpers.

Finding #22 (data-race): the age/energy histograms and the vital-stats table
used to read ``pm.active_mask``/``age``/``energy``/``is_female``/``with_calf``
and ``sim.get_statistics()`` directly on the session thread while the background
worker mutated those arrays in place, producing torn (transiently wrong)
displays.  The worker now builds an immutable snapshot with
``build_population_stats_snapshot`` (called on the same thread that steps the
sim, so it is coherent) and publishes it over the result queue.  The renderers
consume that snapshot dict and never touch the live Simulation.
"""

import logging
from typing import Any, Dict

import numpy as np
import pandas as pd

from .chart_helpers import create_histogram_chart, no_data_placeholder

logger = logging.getLogger(__name__)


def build_population_stats_snapshot(sim: Any) -> Dict[str, Any]:
    """Build an immutable population-stats snapshot from a live ``sim``.

    Must be called on the worker thread (the one that steps the sim) so the
    numpy reductions and ``.tolist()`` copies see a coherent, non-torn state.
    Returns ``{"ages": list, "energies": list, "stats": dict}`` — plain Python
    containers fully decoupled from the sim's arrays.
    """
    ages: list = []
    energies: list = []
    stats: Dict[str, Any] = {}

    pm = getattr(sim, "population_manager", None)
    active = getattr(pm, "active_mask", None) if pm is not None else None

    if active is not None:
        if hasattr(pm, "age") and np.any(active):
            ages = pm.age[active].tolist()
        if hasattr(pm, "energy") and np.any(active):
            energies = pm.energy[active].tolist()
    elif pm is None:
        df = getattr(sim, "agents_df", None)
        if df is not None and not df.empty:
            if "age" in df.columns:
                ages = df["age"].tolist()
            if "energy" in df.columns:
                energies = df["energy"].tolist()

    try:
        stats = dict(sim.get_statistics())
    except (AttributeError, TypeError, ValueError, KeyError) as e:
        logger.warning("build_population_stats_snapshot: get_statistics() failed: %s", e)
        stats = {}

    if active is not None and np.any(active):
        stats["avg_age"] = float(np.mean(pm.age[active]))
        stats["avg_energy"] = float(np.mean(pm.energy[active]))
        stats["females"] = int(np.sum(pm.is_female[active]))
        stats["with_calf"] = int(np.sum(pm.with_calf[active]))

    return {"ages": ages, "energies": energies, "stats": stats}


def render_age_histogram(snapshot: Dict[str, Any]):
    """Render the age-distribution histogram from a stats snapshot dict."""
    ages = (snapshot or {}).get("ages") or []
    if not ages:
        return no_data_placeholder("No age data available.")
    return create_histogram_chart(
        data=ages,
        title="Porpoise Age Distribution",
        x_title="Age (years)",
        y_title="Count",
        x_range=(0, 30),
        nbins=30,
        color="red",
        height=300,
    )


def render_energy_histogram(snapshot: Dict[str, Any]):
    """Render the energy-level histogram from a stats snapshot dict."""
    energies = (snapshot or {}).get("energies") or []
    if not energies:
        return no_data_placeholder("No energy data available.")
    return create_histogram_chart(
        data=energies,
        title="Energy Level Distribution",
        x_title="Energy",
        y_title="Porpoise Count",
        x_range=(0, 20),
        nbins=20,
        color="red",
        height=300,
    )


def build_vital_stats_df(snapshot: Dict[str, Any]) -> pd.DataFrame:
    """Build the vital-stats DataFrame from a stats snapshot dict."""
    stats = (snapshot or {}).get("stats") or {}
    if not stats:
        return pd.DataFrame(columns=["Statistic", "Value"])
    return pd.DataFrame(
        [
            {
                "Statistic": k,
                "Value": f"{v:.2f}" if isinstance(v, float) else str(v),
            }
            for k, v in stats.items()
        ]
    )
