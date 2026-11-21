"""Code no simulation used, removed 2026-11-21.

behavior/dispersal.py held per-animal ports of eight DEPONS dispersal classes (the model
disperses in agents/population.py; PSM-Type1 and the PSM-Type3 variants are not offered by
DEPONS 3.2 either), psm.PSMDispersalType2 duplicated that path, sound.SoundPropagationParams
carried defaults the model never read (beta 20, threshold 152.9), and sound.Hydrophone was a
port of DEPONS's hydrophone output that no simulation created.
"""

import importlib

import pytest

import cenop.behavior
from cenop.behavior import psm, sound


def test_unused_dispersal_module_is_gone():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("cenop.behavior.dispersal")
    for name in ("DispersalBehavior", "DispersalType", "PSMDispersalType2"):
        assert not hasattr(cenop.behavior, name)
    assert not hasattr(psm, "PSMDispersalType2")


def test_unused_sound_classes_are_gone():
    for name in ("SoundPropagationParams", "Hydrophone", "HydrophoneRecord"):
        assert not hasattr(sound, name)
