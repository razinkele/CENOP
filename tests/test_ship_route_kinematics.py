"""Ship movement along a route, as in DEPONS Ship.move() and deterPorpoise().

DEPONS moves a ship one buoy per tick (the route is sampled at 30-minute positions; the buoy
speed is only used for the noise level), waits ``pause`` ticks at a buoy, and starts the route
again after the last buoy. In a tick it deters along the straight line from its buoy to the next
one, unless it is still waiting. The Kattegat ship file relies on this: every route is one
passage in a 721-tick cycle, placed in time by the wait at its first and last buoy.

CENOP convention: after ``update(tick)``, ``(_prev_x, _prev_y)`` is where the ship is in this
tick (its buoy) and ``(x, y)`` is where the tick's sweep ends.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cenop.agents.ship import Buoy, Route, Ship, VesselClass


def _ship(pauses, speeds=None):
    n = len(pauses)
    speeds = speeds or [10.0 + i for i in range(n)]
    buoys = [Buoy(x=10.0 * i, y=5.0 * i, speed=speeds[i], pause_ticks=pauses[i]) for i in range(n)]
    return Ship(
        id=1, x=0.0, y=0.0, route=Route(name="r", buoys=buoys), vessel_type=VesselClass.CARGO
    )


def _trace(ship, ticks):
    out = []
    for t in range(ticks):
        ship.update(t)
        out.append(
            (
                ship.current_buoy_idx,
                ship.is_deterring,
                (ship._prev_x, ship._prev_y),
                (ship.x, ship.y),
            )
        )
    return out


def _pos(i):
    return (10.0 * i, 5.0 * i)


class TestRouteCycle:
    def test_one_buoy_per_tick_with_pauses_and_restart(self):
        # pauses 2, 0, 1 -> a cycle of (2 + 1) + (0 + 1) + (1 + 1) = 6 ticks
        got = _trace(_ship([2, 0, 1]), 13)
        cycle = [
            (0, False, _pos(0), _pos(0)),  # arrives at buoy 0 and waits
            (0, False, _pos(0), _pos(0)),  # waiting
            (0, True, _pos(0), _pos(1)),  # wait over: sweeps towards buoy 1
            (1, True, _pos(1), _pos(2)),  # at buoy 1, sweeps towards buoy 2
            (2, False, _pos(2), _pos(2)),  # arrives at the last buoy and waits
            (2, True, _pos(2), _pos(2)),  # wait over, no next buoy: deters from where it is
        ]
        assert got[:6] == cycle
        assert got[6:12] == cycle  # the route starts again
        assert got[12] == cycle[0]

    def test_the_first_buoys_wait_is_not_skipped(self):
        got = _trace(_ship([5, 0, 0]), 6)
        assert [g[1] for g in got] == [False] * 5 + [True]
        assert all(g[0] == 0 for g in got)

    def test_no_pause_moves_every_tick(self):
        got = _trace(_ship([0, 0, 0]), 4)
        assert [g[0] for g in got] == [0, 1, 2, 0]
        assert got[0][2:] == (_pos(0), _pos(1))
        assert got[2][2:] == (_pos(2), _pos(2))  # last buoy: nothing to sweep to

    def test_last_buoy_with_zero_speed_does_not_strand_the_ship(self):
        # the Kattegat routes end on a buoy with speed 0
        ship = _ship([1, 0, 3], speeds=[12.0, 9.0, 0.0])
        idx = [g[0] for g in _trace(ship, 3 * 7)]
        assert idx.count(0) == idx.count(1) * 2  # back at buoy 0 in every cycle
        assert idx[7] == 0 and idx[14] == 0

    def test_noise_speed_is_the_current_buoys(self):
        ship = _ship([0, 0, 0], speeds=[12.0, 9.0, 4.0])
        speeds = []
        for t in range(3):
            ship.update(t)
            speeds.append(ship.noise.speed)
        assert speeds == [12.0, 9.0, 4.0]

    def test_before_start_and_after_end(self):
        ship = _ship([0, 0, 0])
        ship.tick_start, ship.tick_end = 3, 5
        got = _trace(ship, 8)
        assert [g[1] for g in got] == [False, False, False, True, True, True, False, False]
        assert [g[0] for g in got[3:6]] == [0, 1, 2]
        assert got[6][0] == -1  # DEPONS resets the buoy index after tickEnd


def test_kattegat_traffic_is_spread_over_the_cycle():
    """About the same number of ships sail in every tick; none are lost after the first passage."""
    path = Path(__file__).resolve().parents[1] / "data" / "Kattegat" / "ships.json"
    if not path.exists():
        pytest.skip("data/Kattegat not checked out")
    from cenop.agents.ship import ShipManager

    data = json.loads(path.read_text())
    manager = ShipManager()
    manager.load_from_json(
        str(path), utm_origin_x=529473.0, utm_origin_y=5972242.0, cell_size=400.0
    )
    assert manager.count == len(data["ships"])
    manager.set_enabled(True)
    deterring = []
    for t in range(3 * 721):
        manager.update(t)
        deterring.append(len(manager.get_deterring_ships()))
    deterring = np.array(deterring)
    first, third = deterring[:721], deterring[2 * 721 :]
    assert first.mean() == pytest.approx(third.mean(), rel=0.02)  # same traffic in every cycle
    # Never the whole fleet at once, except in the last tick of the cycle, when every ship's
    # wait at its last buoy ends and it "deters" for one tick from there (DEPONS does the
    # same; those buoys have speed 0).
    assert np.sort(first)[-2] < 0.25 * manager.count
    assert first[720] == manager.count
    assert first.mean() > 50  # ships are sailing at all
    assert (third > 0).mean() > 0.95
