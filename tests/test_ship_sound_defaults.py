"""Ship sound as in DEPONS by default: Weston flux at the ship's position, real day length.

DEPONS always computes ship transmission loss with WestonFlux, from depth, sediment and salinity
at the interpolated ship position and a water temperature of 10 degrees
(Ship.calculateReceivedLevelFor), and takes day or night from the landscape's suntimes.csv.
CENOP used a spreading formula unless a setting was switched on, read the environment at the
animal's cell, and used a fixed 6-18 h day; with those, animals reacted to ships on 28% of
animal-ticks against 19% in DEPONS (A2 run 2).
"""

from pathlib import Path

import numpy as np
import pytest

import cenop.agents.ship as ship_mod
from cenop.agents.ship import Buoy, Route, Ship, ShipManager, VesselClass
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters

DATA = Path(__file__).resolve().parents[1] / "data"


def test_defaults_follow_depons():
    params = SimulationParameters()
    assert params.weston_flux_percell is True
    assert params.weston_flux_default_temperature == 10.0


def test_environment_is_read_at_the_ship_not_at_the_animal(monkeypatch):
    land = create_homogeneous_landscape(width=60, height=60, depth=20.0, food_prob=0.5)
    land._depth = land._depth.copy()
    land._depth[:, 30:] = 80.0  # deep water in the east, where the ship is
    assert land._sediment is not None
    params = SimulationParameters(random_seed=1)
    route = Route(
        name="r", buoys=[Buoy(x=40.5, y=10.5, speed=12.0), Buoy(x=40.5, y=14.5, speed=12.0)]
    )
    ship = Ship(id=3, x=40.5, y=10.5, route=route, vessel_type=VesselClass.CARGO)
    manager = ShipManager(ships=[ship])
    manager.set_enabled(True)
    manager.update(0)
    assert ship.is_deterring

    seen = {}
    real = ship_mod._ship_received_level_from_env

    def spy(source_level, dist_m, depths, grains, sal, params, weston):
        seen["weston"], seen["depths"] = weston, np.array(depths)
        return real(source_level, dist_m, depths, grains, sal, params, weston)

    monkeypatch.setattr(ship_mod, "_ship_received_level_from_env", spy)
    px = np.array([25.5, 26.5])  # animals in the shallow west, 6 km from the ship
    py = np.array([12.5, 12.5])
    manager.calculate_aggregate_deterrence_vectorized(
        px, py, params, cell_data=land, month=1, base_seed=1, tick=0, _force_u=0.0
    )
    assert seen["weston"] is True
    assert seen["depths"].size > 0 and (seen["depths"] == 80.0).all()


def test_simulation_uses_the_landscapes_sunrise_table():
    if not (DATA / "Kattegat" / "suntimes.csv").exists():
        pytest.skip("data/Kattegat not checked out")
    from cenop.core.simulation import Simulation
    from cenop.landscape.cell_data import CellData

    cd = CellData("Kattegat", DATA)
    cd.load()
    params = SimulationParameters(
        porpoise_count=10, sim_years=1, random_seed=1, landscape="Kattegat"
    )
    sim = Simulation(params=params, cell_data=cd, seed=1)
    sim.initialize()
    suntimes = sim.time_manager.suntimes
    assert suntimes.is_loaded
    # 1 January in the file: sunrise at tick 18, sunset at tick 32
    assert [suntimes.is_daytime(t) for t in (17, 18, 31, 32)] == [False, True, True, False]


def test_landscape_without_a_sunrise_table_keeps_the_fixed_day():
    from cenop.core.simulation import Simulation

    params = SimulationParameters(
        porpoise_count=10, sim_years=1, random_seed=1, landscape="Homogeneous"
    )
    sim = Simulation(params=params)
    assert not sim.time_manager.suntimes.is_loaded
