"""DEPONS ship routes from raw AIS positions (cenop.agents.ais_routes)."""

import numpy as np
import pytest

from cenop.agents import ais_routes as ar
from cenop.agents.ship import Buoy, Route, Ship, _vessel_class_from_type

T0 = 1_723_420_800.0  # 2024-08-12 00:00 UTC


@pytest.mark.parametrize(
    "code, cls",
    [(70, "Bulker"), (79, "Bulker"), (84, "Tanker"), (60, "Passenger"), (41, "Passenger"),
     (30, "Fishing"), (52, "Tug"), (33, "Dredger"), (35, "Naval"), (37, "Recreational"),
     (51, "Government/Research"), (0, "Other"), (99, "Other"), (None, "Other")],
)  # fmt: skip
def test_ais_codes_map_to_classes_cenop_accepts(code, cls):
    assert ar.ais_type_to_class(code) == cls
    _vessel_class_from_type(cls)  # raises on an unknown class


def test_split_passages_at_long_gaps():
    t = np.array([0, 600, 1200, 20000, 20600])
    assert [(s.start, s.stop) for s in ar.split_passages(t, 7200)] == [(0, 3), (3, 5)]


def _track(start_s, hours, knots=10.0, still_from=None, still_hours=0):
    t = T0 + start_s + np.arange(0, hours * 3600 + 1, 60.0)
    v = knots * 1852 / 3600  # m/s
    x = 4_800_000 + v * (t - t[0])
    sog = np.full(t.size, knots)
    if still_from is not None:
        a = still_from
        b = a + still_hours * 60
        x[a:b] = x[a]
        x[b:] = x[b:] - (x[b] - x[a])
        sog[a:b] = 0.0
    y = np.full(t.size, 3_600_000.0)
    return t, x, y, sog


def test_route_places_the_passage_in_time_and_fills_the_cycle():
    t, x, y, sog = _track(start_s=5 * 1800, hours=3)
    buoys = ar.route_from_passage(t, x, y, sog, T0, cycle_ticks=721)
    assert buoys[0]["pause"] == 5  # first real position in tick 5
    assert sum(1 + b["pause"] for b in buoys) == 721
    assert len(buoys) == 7  # ticks 5..11
    assert buoys[1]["x"] - buoys[0]["x"] == pytest.approx(10 * 1852 / 2, rel=1e-6)
    assert buoys[2]["speed"] == pytest.approx(10.0)


def test_a_stationary_spell_becomes_a_pause():
    t, x, y, sog = _track(start_s=0, hours=6, still_from=90, still_hours=2)
    buoys = ar.route_from_passage(t, x, y, sog, T0, cycle_ticks=721)
    assert max(b["pause"] for b in buoys[:-1]) >= 3
    assert sum(1 + b["pause"] for b in buoys) == 721


def test_a_moored_ship_and_a_too_short_passage_are_dropped():
    t = T0 + np.arange(0, 6 * 3600, 60.0)
    x = np.full(t.size, 4_800_000.0)
    assert ar.route_from_passage(t, x, x * 0 + 3.6e6, np.zeros(t.size), T0) is None
    t2, x2, y2, s2 = _track(start_s=100, hours=0.4)
    assert ar.route_from_passage(t2, x2, y2, s2, T0) is None


def test_unavailable_speed_is_taken_from_the_displacement():
    t, x, y, sog = _track(start_s=0, hours=3, knots=12.0)
    buoys = ar.route_from_passage(t, x, y, np.full(t.size, 102.3), T0)
    assert buoys[2]["speed"] == pytest.approx(12.0, rel=1e-3)


def test_build_ships_names_types_lengths_and_runs_in_cenop():
    t1, x1, y1, s1 = _track(start_s=0, hours=3)
    t2, x2, y2, s2 = _track(start_s=40 * 1800, hours=2)
    mmsi = np.r_[np.full(t1.size, 111), np.full(t2.size, 222)]
    typ = np.r_[np.full(t1.size, 80.0), np.full(t2.size, np.nan)]
    out = ar.build_ships(
        mmsi, np.r_[t1, t2], np.r_[x1, x2], np.r_[y1, y2], np.r_[s1, s2], typ, T0,
        lengths={111: 183}, default_lengths={"Other": 19},
    )  # fmt: skip
    assert [s["name"] for s in out["ships"]] == ["111-1", "222-1"]
    assert [s["type"] for s in out["ships"]] == ["Tanker", "Other"]
    assert [s["length"] for s in out["ships"]] == [183, 19]
    r = out["routes"][1]["route"]
    ship = Ship(
        id=1, x=r[0]["x"], y=r[0]["y"], vessel_type=_vessel_class_from_type("Other"),
        route=Route(name="r", buoys=[Buoy(b["x"], b["y"], b["speed"], b["pause"]) for b in r]),
    )  # fmt: skip
    deterring = []
    for tick in range(721):
        ship.update(tick)
        deterring.append(ship.is_deterring)
    assert not any(deterring[:40]) and any(deterring[40:45]) and not any(deterring[60:700])
