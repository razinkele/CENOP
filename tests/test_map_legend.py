"""Layer legend for the dashboard map (shiny_deckgl layer_legend_widget entries)."""

from cenop.server import map_legend as ml


def test_entries_list_the_basemap_and_only_loaded_layers_in_order():
    entries = ml.legend_entries({"porpoises", "depth-bitmap"})
    assert [e["layer_id"] for e in entries] == ["basemap", "depth-bitmap", "porpoises"]
    assert all({"layer_id", "label", "shape"} <= set(e) for e in entries)


def test_grid_layers_carry_a_colour_ramp_and_others_one_colour():
    by_id = {e["layer_id"]: e for e in ml.legend_entries(set(ml.LEGEND_ORDER))}
    assert by_id["depth-bitmap"]["shape"] == "gradient" and len(by_id["depth-bitmap"]["colors"]) > 2
    assert by_id["foraging-bitmap"]["shape"] == "gradient"
    assert by_id["porpoises"]["shape"] == "circle" and "color" in by_id["porpoises"]
    assert "turbine-blades" not in by_id and "depth-tooltip" not in by_id


def test_a_toggle_carries_its_companion_layers():
    assert ml.toggled_ids("depth-bitmap") == ["depth-bitmap", "depth-tooltip"]
    assert ml.toggled_ids("turbine-poles") == ["turbine-poles", "turbine-blades"]
    assert ml.toggled_ids("porpoises") == ["porpoises"]


def test_user_choice_overrides_the_layer_visible_prop():
    layer = {"id": "porpoises", "visible": True, "data": [1]}
    assert ml.with_choice(layer, {"porpoises": False})["visible"] is False
    assert ml.with_choice(layer, {"porpoises": False})["data"] == [1]
    assert layer["visible"] is True  # not mutated
    assert ml.with_choice(layer, {}) is layer


def test_a_stale_choice_for_an_unloaded_layer_is_ignored():
    """A layer the app has unloaded is sent hidden; an old 'show' choice must not undo it."""
    choice = {"foraging-bitmap": True, "foraging-tooltip": True, "porpoises": False}
    assert ml.active_choice(choice, {"porpoises"}) == {"porpoises": False}
    assert ml.active_choice(choice, {"foraging-bitmap"}) == {
        "foraging-bitmap": True,
        "foraging-tooltip": True,
    }
