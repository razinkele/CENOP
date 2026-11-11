"""Initial placement: Lithuania starts animals only in its main connected water body.

The Curonian Lagoon is a separate water body on the 1 km Lithuania grid; animals placed there are
trapped by the DEPONS land check and starve. Placement for Lithuania is therefore restricted to
the largest connected water body. Every other landscape (Kattegat included) keeps DEPONS's rule:
any water cell, isolated pockets included.
"""

import numpy as np

import cenop.agents.population as P
from cenop.agents.population import PorpoisePopulation
from cenop.landscape.cell_data import create_homogeneous_landscape
from cenop.parameters.simulation_params import SimulationParameters

LAND_X0, LAND_X1 = 38, 44  # land strip separating the two water bodies


def _two_body_landscape(name):
    """60x60 grid: a large water body (x < 38) and a smaller one (x >= 44), split by land."""
    land = create_homogeneous_landscape(width=60, height=60)
    land._depth[:, LAND_X0:LAND_X1] = -10.0
    land.landscape_name = name
    return land


def _pop(name, count=600, seed=7):
    params = SimulationParameters(random_seed=seed, communication_enabled=False)
    return PorpoisePopulation(count, params, _two_body_landscape(name))


def test_lithuania_places_every_animal_in_the_main_water_body():
    pop = _pop("Lithuania")
    assert np.all(pop.x < LAND_X0)


def test_kattegat_still_places_animals_in_every_water_body():
    pop = _pop("Kattegat")
    assert np.any(pop.x >= LAND_X1)
    assert np.any(pop.x < LAND_X0)


def test_kattegat_placement_does_not_use_the_main_body_restriction(monkeypatch):
    def fail(mask):
        raise AssertionError("main-water-body restriction used for a DEPONS landscape")

    monkeypatch.setattr(P, "_main_water_body", fail)
    _pop("Kattegat")


def test_main_water_body_keeps_the_largest_diagonally_connected_body():
    mask = np.zeros((6, 6), dtype=bool)
    mask[0, 0] = mask[1, 1] = mask[2, 2] = True  # 3 cells touching only diagonally
    mask[5, 4] = mask[5, 5] = True  # 2-cell body
    kept = P._main_water_body(mask)
    assert kept[0, 0] and kept[1, 1] and kept[2, 2]
    assert not kept[5, 4] and not kept[5, 5]
