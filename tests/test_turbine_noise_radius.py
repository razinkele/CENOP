"""The map's noise circles use the turbine model's own deterrence range.

They used a fixed 158 dB threshold with 20 log10(r) spreading (the model: deter_threshold
152 dB, beta_hat log10(r) + alpha_hat r, capped at deter_max_distance), and drew 500 m
"operational" circles although operating turbines (145 dB) never reach the threshold.
"""

from cenop.agents.turbine import Turbine, TurbinePhase, deterrence_radius_m
from cenop.parameters.simulation_params import SimulationParameters


def _deters_at(params, impact, metres):
    t = Turbine(id=0, x=0.0, y=0.0, name="t", impact=impact, start_tick=0, end_tick=10)
    t.update_phase(5)
    assert t.phase == TurbinePhase.CONSTRUCTION
    return t.should_deter(metres / 400.0, 0.0, params)[0]


def test_radius_is_where_the_turbine_stops_deterring():
    p = SimulationParameters()
    r = deterrence_radius_m(234.0, p)
    assert 1_000 < r < p.deter_max_distance * 1000
    assert _deters_at(p, 234.0, r - 5.0) and not _deters_at(p, 234.0, r + 5.0)


def test_radius_is_capped_and_zero_below_the_threshold():
    p = SimulationParameters(deter_max_distance=10.0)
    assert deterrence_radius_m(234.0, p) == 10_000.0
    assert deterrence_radius_m(145.0, SimulationParameters()) == 0.0  # operating turbines


def test_map_legend_ramps_match_the_layer_colour_schemes():
    from cenop.server import map_legend
    from cenop.server.map_layers import GIS_COLOR_SCHEMES

    assert map_legend._BATHYMETRY_RAMP == GIS_COLOR_SCHEMES["viridis"]  # was a blue ramp
    assert map_legend._FORAGING_RAMP == GIS_COLOR_SCHEMES["green"]


class _NoTransform:
    def transform(self, x, y):
        return x, y


def test_map_reads_turbine_timing_in_ticks_like_the_model(tmp_path):
    """The map compared the file's start/end with the tick; 'start/end' headers are days."""
    from cenop.server.map_layers import turbines_for_map

    days = tmp_path / "days.txt"
    days.write_text("id\tx\ty\timpact\tstart\tend\nA\t10\t20\t234\t20\t23\n")
    ticks = tmp_path / "ticks.txt"
    ticks.write_text("id\tx\ty\timpact\ttick.start\ttick.end\nB\t10\t20\t220\t480\t500\n")
    (a,) = turbines_for_map(str(days), _NoTransform())
    (b,) = turbines_for_map(str(ticks), _NoTransform())
    assert (a["start"], a["end"], a["position"], a["impact"]) == (960, 1104, [10.0, 20.0], 234.0)
    assert (b["start"], b["end"], b["impact"]) == (480, 500, 220.0)  # was (0, 0)


def test_map_phase_follows_the_model_rule():
    from cenop.server.map_layers import turbine_phase_at

    assert turbine_phase_at(0, 144, 96) == "construction"  # was operational (96 > 3 days)
    assert turbine_phase_at(960, 1104, 959) == "planned"
    assert turbine_phase_at(960, 1104, 1104) == "operational"  # end tick: piling stopped
