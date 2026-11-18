"""The dispersal setting (params.dispersal): validation, off, Undirected."""

import pytest

from cenop.parameters.simulation_params import DISPERSAL_MODES, SimulationParameters


def test_modes_constant():
    assert DISPERSAL_MODES == ("off", "PSM-Type2", "Undirected")


@pytest.mark.parametrize("mode", ["off", "PSM-Type2", "Undirected"])
def test_valid_modes_accepted(mode):
    assert SimulationParameters(dispersal=mode).dispersal == mode


def test_inner_danish_waters_not_implemented():
    with pytest.raises(ValueError, match="InnerDanishWaters.*not implemented"):
        SimulationParameters(dispersal="InnerDanishWaters")


@pytest.mark.parametrize("bad", ["Off", "undirected", " off", "PSM-Typ2", ""])
def test_unknown_mode_lists_valid_values(bad):
    with pytest.raises(ValueError, match="off.*PSM-Type2.*Undirected"):
        SimulationParameters(dispersal=bad)
