"""The ship deterrence pass over all ships at once, and the waiting-ship shortcut."""

import copy

import numpy as np
import pytest

from cenop.agents import ship as ship_mod
from cenop.agents.ship import Buoy, Route, Ship, ShipManager, VesselClass
from cenop.parameters.simulation_params import SimulationParameters


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_compiled_pairs_equal_the_numpy_reference(dtype):
    rng = np.random.default_rng(3)
    n, ships, steps = 300, 12, 30
    px = rng.uniform(0, 80, n).astype(dtype)
    py = rng.uniform(0, 80, n).astype(dtype)
    sub_x = rng.uniform(0, 80, (ships, 1)) + rng.uniform(-3, 3, (ships, steps)).cumsum(axis=1) / 10
    sub_y = rng.uniform(0, 80, (ships, 1)) + rng.uniform(-3, 3, (ships, steps)).cumsum(axis=1) / 10
    cand_ship = np.repeat(np.arange(ships), 40)
    cand_porp = np.concatenate([np.sort(rng.choice(n, 40, replace=False)) for _ in range(ships)])
    px[cand_porp[0]], py[cand_porp[0]] = sub_x[0, 0], sub_y[0, 0]  # a pair closer than 1 m
    args = (px, py, cand_ship, cand_porp, sub_x, sub_y, 400.0, 0.5, 10000.0)
    ref = ship_mod._in_range_pairs_numpy(*args)
    got = ship_mod._in_range_pairs(*args)
    assert ref[0].size > 100
    assert (ref[2] == 1.0).any()  # the floor at 1 m is exercised
    for a, b in zip(ref, got):
        assert np.array_equal(a, b)


def _route_ship(pauses, ident=1):
    buoys = [
        Buoy(x=10.0 * i, y=5.0 * i, speed=10.0 + i, pause_ticks=p) for i, p in enumerate(pauses)
    ]
    return Ship(
        id=ident, x=0.0, y=0.0, route=Route(name="r", buoys=buoys), vessel_type=VesselClass.CARGO
    )


def _state(s):
    return (
        s.current_buoy_idx, s.ticks_paused, s.is_deterring, s._is_active,
        s._prev_x, s._prev_y, s.x, s.y, s.current_speed, s.noise.speed, s.heading,
    )  # fmt: skip


@pytest.mark.parametrize("pauses", [[5, 0, 3], [0, 0, 0], [1, 2, 0, 7], [40, 0]])
def test_waiting_shortcut_leaves_the_ship_as_the_full_update_does(pauses):
    fast, slow = _route_ship(pauses), _route_ship(pauses)
    fast.tick_end = slow.tick_end = 60  # the period ends mid-run as well
    for t in range(90):
        fast.update(t)
        slow._waiting = False  # never take the shortcut
        slow.update(t)
        assert _state(fast) == _state(slow), t


def _params():
    return SimulationParameters(porpoise_count=10, random_seed=1, weston_flux_percell=False)


def _moving_ship(ident, x0, y0, x1, y1, level):
    s = Ship(id=ident, x=x1, y=y1, vessel_type=VesselClass.CARGO)
    s._prev_x, s._prev_y = x0, y0
    s._is_active = True
    s.current_buoy_idx = 0
    s.noise.base_source_level = level
    return s


def test_ships_with_different_models_use_their_own_coefficients():
    """Two ships far apart, each near its own animals: the pass over both equals each alone."""
    p = _params()
    a = _moving_ship(1, 20.0, 20.0, 22.0, 21.0, 190.0)
    b = _moving_ship(2, 400.0, 400.0, 402.0, 399.0, 190.0)
    b.deterrence_model = copy.copy(b.deterrence_model)
    b.deterrence_model.cship_int_day = b.deterrence_model.cship_int_day + 1.0
    px = np.array([24.0, 25.0, 404.0, 405.0])
    py = np.array([22.0, 18.0, 401.0, 398.0])

    def run(ships):
        m = ShipManager(ships)
        m.enabled = True
        return m.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)

    both, only_a, only_b = run([a, b]), run([a]), run([b])
    assert np.abs(only_a[0][:2]).min() > 0 and np.abs(only_b[0][2:]).min() > 0
    for axis in (0, 1):
        assert np.array_equal(both[axis][:2], only_a[axis][:2])
        assert np.array_equal(both[axis][2:], only_b[axis][2:])
    same_model = _moving_ship(2, 400.0, 400.0, 402.0, 399.0, 190.0)
    assert not np.array_equal(run([same_model])[0][2:], only_b[0][2:])  # the coefficient matters


def test_equally_loud_ships_lowest_id_wins_whatever_the_list_order():
    p = _params()
    px, py = np.array([50.0]), np.array([50.0])
    first = _moving_ship(1, 46.0, 50.0, 46.0, 50.0, 190.0)
    second = _moving_ship(2, 54.0, 50.0, 54.0, 50.0, 190.0)  # same distance, other side

    def run(ships):
        m = ShipManager(ships)
        m.enabled = True
        return m.calculate_aggregate_deterrence_vectorized(px, py, p, _force_u=0.0)

    alone = run([first])
    assert alone[0][0] > 0  # pushed east, away from ship 1
    assert run([first, second])[0][0] == alone[0][0]
    assert run([second, first])[0][0] == alone[0][0]
