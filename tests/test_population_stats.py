"""Tests for the population-stats snapshot builder and renderer helpers.

These helpers publish an immutable stats snapshot from the background worker
thread and let the histogram/vital-stats renderers read that snapshot instead of
reaching into the live, concurrently-mutated Simulation (Finding #22 data-race).
"""

import logging
from types import SimpleNamespace

import numpy as np
import pandas as pd

from cenop.server.renderers.population_stats import (
    build_population_stats_snapshot,
    render_age_histogram,
    render_energy_histogram,
    build_vital_stats_df,
)


def _fake_sim_with_pm():
    pm = SimpleNamespace(
        active_mask=np.array([True, True, False, True]),
        age=np.array([2.0, 4.0, 99.0, 6.0], dtype=np.float32),
        energy=np.array([10.0, 12.0, 0.0, 8.0], dtype=np.float32),
        is_female=np.array([True, False, True, True]),
        with_calf=np.array([True, False, False, False]),
    )
    sim = SimpleNamespace(
        population_manager=pm,
        get_statistics=lambda: {
            "tick": 48, "day": 1, "year": 0, "population": 3,
            "births_total": 0, "deaths_total": 0,
        },
    )
    return sim, pm


class TestBuildSnapshot:
    def test_snapshot_extracts_active_ages_and_energies(self):
        sim, _pm = _fake_sim_with_pm()
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert snap["energies"] == [10.0, 12.0, 8.0]

    def test_snapshot_stats_merge_active_aggregates(self):
        sim, _pm = _fake_sim_with_pm()
        stats = build_population_stats_snapshot(sim)["stats"]
        assert stats["population"] == 3
        assert stats["avg_age"] == 4.0
        assert stats["avg_energy"] == 10.0
        assert stats["females"] == 2
        assert stats["with_calf"] == 1

    def test_snapshot_is_decoupled_from_later_mutation(self):
        # The core defect: the snapshot must be an immutable copy taken at one
        # instant. Mutating the live arrays afterwards must NOT change the snapshot.
        sim, pm = _fake_sim_with_pm()
        snap = build_population_stats_snapshot(sim)
        pm.age[:] = 999.0
        pm.energy[:] = -1.0
        pm.active_mask[:] = False
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert snap["energies"] == [10.0, 12.0, 8.0]
        assert snap["stats"]["avg_age"] == 4.0

    def test_snapshot_agents_df_fallback(self):
        df = pd.DataFrame({"age": [1.0, 3.0], "energy": [5.0, 7.0]})
        sim = SimpleNamespace(
            population_manager=None,
            agents_df=df,
            get_statistics=lambda: {"population": 2},
        )
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == [1.0, 3.0]
        assert snap["energies"] == [5.0, 7.0]
        assert snap["stats"]["population"] == 2

    def test_snapshot_empty_population(self):
        pm = SimpleNamespace(
            active_mask=np.zeros(3, dtype=bool),
            age=np.zeros(3, dtype=np.float32),
            energy=np.zeros(3, dtype=np.float32),
            is_female=np.zeros(3, dtype=bool),
            with_calf=np.zeros(3, dtype=bool),
        )
        sim = SimpleNamespace(population_manager=pm,
                              get_statistics=lambda: {"population": 0})
        snap = build_population_stats_snapshot(sim)
        assert snap["ages"] == []
        assert snap["energies"] == []
        assert "avg_age" not in snap["stats"]

    def test_snapshot_logs_and_recovers_when_get_statistics_raises(self, caplog):
        # FIX B: a failing get_statistics() must not be swallowed silently — it
        # is logged at WARNING and the snapshot still returns (stats == {} merged
        # with any pm-derived aggregates).
        def _boom():
            raise ValueError("boom")

        sim, pm = _fake_sim_with_pm()
        sim.get_statistics = _boom
        with caplog.at_level(logging.WARNING):
            snap = build_population_stats_snapshot(sim)
        # pm is present with active agents, so aggregates are still merged in even
        # though get_statistics() failed (proving the fallback path ran).
        assert snap["stats"] == {
            "avg_age": 4.0,
            "avg_energy": 10.0,
            "females": 2,
            "with_calf": 1,
        }
        assert snap["ages"] == [2.0, 4.0, 6.0]
        assert any(
            "get_statistics() failed" in r.getMessage() and r.levelno == logging.WARNING
            for r in caplog.records
        )


class TestRenderHelpers:
    def test_age_histogram_from_snapshot_returns_html(self):
        out = render_age_histogram({"ages": [1.0, 2.0, 3.0], "energies": [], "stats": {}})
        assert type(out).__name__ == "HTML"

    def test_age_histogram_empty_returns_placeholder(self):
        out = render_age_histogram({"ages": [], "energies": [], "stats": {}})
        assert type(out).__name__ == "Tag"

    def test_energy_histogram_from_snapshot_returns_html(self):
        out = render_energy_histogram({"ages": [], "energies": [4.0, 5.0], "stats": {}})
        assert type(out).__name__ == "HTML"

    def test_energy_histogram_empty_returns_placeholder(self):
        out = render_energy_histogram({"ages": [], "energies": [], "stats": {}})
        assert type(out).__name__ == "Tag"

    def test_vital_stats_df_from_snapshot(self):
        df = build_vital_stats_df(
            {"ages": [], "energies": [],
             "stats": {"population": 3, "avg_age": 4.0}}
        )
        assert list(df.columns) == ["Statistic", "Value"]
        rows = {r["Statistic"]: r["Value"] for _, r in df.iterrows()}
        assert rows["population"] == "3"
        assert rows["avg_age"] == "4.00"

    def test_vital_stats_df_empty_stats(self):
        df = build_vital_stats_df({"ages": [], "energies": [], "stats": {}})
        assert df.empty
        assert list(df.columns) == ["Statistic", "Value"]

    def test_helpers_never_touch_a_live_sim(self):
        # Passing a plain dict (no Simulation) must fully work — proves the
        # renderers no longer depend on state.simulation().
        snap = {"ages": [2.0], "energies": [3.0], "stats": {"population": 1}}
        assert type(render_age_histogram(snap)).__name__ == "HTML"
        assert type(render_energy_histogram(snap)).__name__ == "HTML"
        assert not build_vital_stats_df(snap).empty
