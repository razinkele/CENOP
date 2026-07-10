"""Finding #5: Energy-panel rS/rR UI defaults must match DEPONS 3.2 parameters.xml (0.03)."""

import re
from types import SimpleNamespace

import pytest


def _energy_panel_numeric_default(input_id: str) -> float:
    """Return the declared ``value=`` of a numeric input in the Energy settings panel."""
    from cenop.ui.tabs.settings import _energy_settings_panel

    html = str(_energy_settings_panel().content.tagify())
    tag_match = re.search(rf'<input[^>]*id="{input_id}"[^>]*>', html)
    assert tag_match, f"input {input_id!r} not found in Energy panel HTML"
    value_match = re.search(r'value="([^"]*)"', tag_match.group(0))
    assert value_match, f"no value attribute on input {input_id!r}"
    return float(value_match.group(1))


def _make_mock_input(param_rS: float, param_rR: float) -> SimpleNamespace:
    """Shiny-input stand-in returning UI defaults; rS/rR are parameterized."""

    def const(v):
        return lambda: v

    values = dict(
        random_seed=1,
        psm_dist="N(300;100)",
        porpoise_count=5,
        sim_years=1,
        simulation_mode="DEPONS",
        time_mode_override="",
        movement_mode_override="",
        fsm_mode_override="",
        energy_mode_override="",
        memory_mode_override="",
        jasmine_mass_kg=50.0,
        jasmine_drag_coeff=0.01,
        jasmine_max_thrust=100.0,
        jasmine_current_weight=0.5,
        jasmine_bmr_scale=1.0,
        jasmine_activity_cost=2.0,
        jasmine_disturbance_cost=1.5,
        jasmine_memory_decay_rate=0.001,
        jasmine_avoidance_strength=0.8,
        jasmine_avoidance_radius=20.0,
        landscape="Homogeneous",
        turbines="off",
        ships_enabled=False,
        weston_flux_percell=False,
        dispersal="off",
        tracked_porpoise_count=1,
        tdisp=3,
        psm_log=0.6,
        psm_tol=5.0,
        psm_angle=40.0,
        param_rS=param_rS,
        param_rR=param_rR,
        param_rU=0.1,
        bycatch_prob=0.018,
        param_k=0.001,
        param_a0=0.35,
        param_a1=0.0005,
        param_a2=-0.02,
        param_b0=-0.024,
        param_b1=-0.008,
        param_b2=0.93,
        param_b3=-14.0,
        communication_enabled=False,
        communication_range_km=1.0,
        communication_source_level=130.0,
        communication_threshold=80.0,
        communication_response_slope=0.1,
        social_weight=0.3,
    )
    return SimpleNamespace(**{k: const(v) for k, v in values.items()})


class TestEnergyPanelDecayDefaults:
    """rS/rR must default to DEPONS 3.2 value 0.03 (was stale DEPONS-3.0 0.04)."""

    def test_rS_ui_default_is_003(self):
        assert _energy_panel_numeric_default("param_rS") == pytest.approx(0.03)

    def test_rR_ui_default_is_003(self):
        assert _energy_panel_numeric_default("param_rR") == pytest.approx(0.03)

    def test_rS_tooltip_states_003(self):
        from cenop.ui.tabs.settings import TOOLTIPS

        assert "0.03" in TOOLTIPS["param_rS"]
        assert "0.04" not in TOOLTIPS["param_rS"]

    def test_rR_tooltip_states_003(self):
        from cenop.ui.tabs.settings import TOOLTIPS

        assert "0.03" in TOOLTIPS["param_rR"]
        assert "0.04" not in TOOLTIPS["param_rR"]

    def test_controller_propagates_ui_defaults_to_params(self):
        """End-to-end: UI energy defaults flow through create_simulation_from_inputs."""
        from cenop.server.simulation_controller import create_simulation_from_inputs

        ui_rS = _energy_panel_numeric_default("param_rS")
        ui_rR = _energy_panel_numeric_default("param_rR")
        sim = create_simulation_from_inputs(_make_mock_input(ui_rS, ui_rR))
        assert sim.params.r_s == pytest.approx(0.03)
        assert sim.params.r_r == pytest.approx(0.03)
