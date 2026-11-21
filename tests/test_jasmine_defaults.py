"""JASMINE mode runs DEPONS movement and energy unless a subsystem override asks otherwise.

The JASMINE movement fallback and energy budget are experimental and uncalibrated (see the
changelog's known issues); learned avoidance and behaviour states stay JASMINE.
"""

from cenop.parameters.simulation_params import SimulationParameters


def test_jasmine_mode_defaults_to_depons_movement_and_energy():
    p = SimulationParameters(simulation_mode="JASMINE")
    assert p.get_effective_movement_mode() == "DEPONS"
    assert p.get_effective_energy_mode() == "DEPONS"
    assert p.get_effective_memory_mode() == "JASMINE"
    assert p.get_effective_fsm_mode() == "JASMINE"
    assert p.get_effective_time_mode() == "JASMINE"


def test_overrides_still_select_the_jasmine_movement_and_energy():
    p = SimulationParameters(
        simulation_mode="JASMINE", movement_mode="JASMINE", energy_mode="JASMINE"
    )
    assert p.get_effective_movement_mode() == "JASMINE"
    assert p.get_effective_energy_mode() == "JASMINE"


def test_app_builds_depons_movement_and_energy_in_jasmine_mode():
    from cenop.movement.depons_crw import DEPONSCRWMovement
    from cenop.physiology.energy_budget import DEPONSEnergyModule
    from cenop.server.simulation_controller import create_simulation_from_inputs
    from tests.test_settings_defaults import _make_mock_input

    inp = _make_mock_input(0.03, 0.03)
    inp.simulation_mode = lambda: "JASMINE"
    sim = create_simulation_from_inputs(inp)
    pm = sim.population_manager if hasattr(sim, "population_manager") else None
    if pm is None:
        sim.initialize()
        pm = sim.population_manager
    assert isinstance(pm._movement_module, DEPONSCRWMovement)
    assert isinstance(pm._energy_module, DEPONSEnergyModule)
    assert pm._memory_module is not None and type(pm._memory_module).__name__.startswith("JASMINE")


def test_build_simulation_gives_code_the_modules_the_app_uses():
    from cenop.movement.depons_crw import DEPONSCRWMovement
    from cenop.server.simulation_controller import build_simulation

    sim = build_simulation(
        SimulationParameters(simulation_mode="JASMINE", landscape="Homogeneous", porpoise_count=5)
    )
    sim.initialize()
    pm = sim.population_manager
    assert isinstance(pm._movement_module, DEPONSCRWMovement)
    assert type(pm._memory_module).__name__.startswith("JASMINE")
    assert pm._behavior_fsm is not None
