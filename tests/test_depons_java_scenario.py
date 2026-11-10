"""Tests for A1 scenario loading, DEPONS parameter merging and Repast sweep rendering."""

import json
import xml.etree.ElementTree as ET

import pytest

from validation.depons_java.scenario import (
    SCENARIO_DIR,
    ParamSpec,
    ScenarioError,
    effective_parameters,
    load_scenario,
    parse_parameters_xml,
    render_batch_params,
)

PARAMS_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<parameters>
\t\t<parameter name="randomSeed" displayName="randomSeed - RNG" type="int"
\t\t\t\t\t\tdefaultValue="__NULL__"
\t\t\t\t\t\tisReadOnly="false"
\t\t\t\t\t\tconverter="repast.simphony.parameter.StringConverterFactory$IntConverter"
\t\t/>
\t\t<parameter name="porpoiseCount" displayName="porpoiseCount" type="int"
\t\t\t\t\t\tdefaultValue="10000" isReadOnly="false"/>
\t\t<parameter name="ships" displayName="ships" type="boolean"
\t\t\t\t\t\tdefaultValue="false" isReadOnly="false"/>
\t\t<parameter name="PSM_dist" displayName="PSM_dist" type="string"
\t\t\t\t\t\tdefaultValue="N(300;100)" isReadOnly="false"/>
</parameters>
"""


@pytest.fixture
def specs(tmp_path):
    p = tmp_path / "parameters.xml"
    p.write_text(PARAMS_XML)
    return parse_parameters_xml(p)


def test_parse_parameters_xml_types_defaults_order(specs):
    assert list(specs) == ["randomSeed", "porpoiseCount", "ships", "PSM_dist"]
    assert specs["randomSeed"] == ParamSpec("randomSeed", "int", "__NULL__")
    assert specs["porpoiseCount"] == ParamSpec("porpoiseCount", "int", "10000")
    assert specs["ships"] == ParamSpec("ships", "boolean", "false")
    assert specs["PSM_dist"] == ParamSpec("PSM_dist", "string", "N(300;100)")


def test_effective_parameters_merges_and_seeds(specs):
    assert effective_parameters(specs, {"porpoiseCount": "200"}, 7) == {
        "randomSeed": "7",
        "porpoiseCount": "200",
        "ships": "false",
        "PSM_dist": "N(300;100)",
    }


def test_unknown_override_rejected(specs):
    with pytest.raises(ScenarioError, match="dispersl"):
        effective_parameters(specs, {"dispersl": "x"}, 1)


def test_bad_int_rejected(specs):
    with pytest.raises(ScenarioError, match="porpoiseCount"):
        effective_parameters(specs, {"porpoiseCount": "2k"}, 1)


def test_bad_boolean_rejected(specs):
    with pytest.raises(ScenarioError, match="ships"):
        effective_parameters(specs, {"ships": "yes"}, 1)


VALUES_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<parameters>
\t\t<parameter name="dispersal" displayName="Large-scale movement type" type="string"
\t\t\t\t\t\tdefaultValue="PSM-Type2" isReadOnly="false"
\t\t\t\t\t\tvalues="'off' 'PSM-Type2' 'Undirected' 'InnerDanishWaters'"
\t\t/>
\t\t<parameter name="PSM_dist" displayName="PSM_dist" type="string"
\t\t\t\t\t\tdefaultValue="N(300;100)" isReadOnly="false"/>
</parameters>
"""


@pytest.fixture
def value_specs(tmp_path):
    p = tmp_path / "values.xml"
    p.write_text(VALUES_XML)
    return parse_parameters_xml(p)


def test_parse_parameters_xml_reads_allowed_values(value_specs):
    assert value_specs["dispersal"].allowed == (
        "off",
        "PSM-Type2",
        "Undirected",
        "InnerDanishWaters",
    )
    assert value_specs["PSM_dist"].allowed is None


def test_value_outside_allowed_list_rejected(value_specs):
    # A typo in a constrained string parameter would make DEPONS fall back silently
    # (DispersalFactory -> Off), so it must fail before Java starts.
    with pytest.raises(ScenarioError, match="dispersal.*PSM-Typ2.*InnerDanishWaters"):
        effective_parameters(value_specs, {"dispersal": "PSM-Typ2"}, 1)


def test_value_inside_allowed_list_accepted(value_specs):
    params = effective_parameters(value_specs, {"dispersal": "InnerDanishWaters"}, 1)
    assert params["dispersal"] == "InnerDanishWaters"


def test_render_batch_params_roundtrip(specs):
    params = effective_parameters(specs, {"PSM_dist": "A&B"}, 3)
    root = ET.fromstring(render_batch_params(specs, params))
    assert root.tag == "sweep"
    assert root.get("runs") == "1"
    elems = root.findall("parameter")
    assert [e.get("name") for e in elems] == list(specs)
    assert all(e.get("type") == "constant" for e in elems)
    ctype = {e.get("name"): e.get("constant_type") for e in elems}
    assert ctype == {
        "randomSeed": "int",
        "porpoiseCount": "int",
        "ships": "boolean",
        "PSM_dist": "java.lang.String",
    }
    values = {e.get("name"): e.get("value") for e in elems}
    assert values["PSM_dist"] == "A&B"
    assert values["randomSeed"] == "3"


def _write_scenario(tmp_path, **changes):
    data = {
        "id": "s",
        "description": "d",
        "overrides": {"porpoiseCount": "200"},
        "seeds": [1, 2],
        "determinism_seed": 1,
    }
    data.update(changes)
    data = {k: v for k, v in data.items() if v is not None}
    p = tmp_path / "s.json"
    p.write_text(json.dumps(data))
    return p


@pytest.mark.parametrize(
    "changes",
    [
        {"seeds": None},
        {"seeds": []},
        {"seeds": [1, 1]},
        {"determinism_seed": 9},
    ],
    ids=["missing-seeds", "empty-seeds", "duplicate-seeds", "determinism-not-in-seeds"],
)
def test_load_scenario_validation(tmp_path, changes):
    with pytest.raises(ScenarioError):
        load_scenario(_write_scenario(tmp_path, **changes))


def test_load_scenario_ok(tmp_path):
    sc = load_scenario(_write_scenario(tmp_path))
    assert sc.seeds == (1, 2)
    assert sc.determinism_seed == 1
    assert sc.overrides == {"porpoiseCount": "200"}


def test_shipped_scenarios_differ_only_in_dispersal():
    psm2 = load_scenario(SCENARIO_DIR / "kattegat-psm2-undisturbed-v1.json")
    idw = load_scenario(SCENARIO_DIR / "kattegat-idw-undisturbed-v1.json")
    assert psm2.id == "kattegat-psm2-undisturbed-v1"
    assert idw.id == "kattegat-idw-undisturbed-v1"
    assert psm2.overrides["dispersal"] == "PSM-Type2"
    assert idw.overrides["dispersal"] == "InnerDanishWaters"
    strip = lambda o: {k: v for k, v in o.items() if k != "dispersal"}  # noqa: E731
    assert strip(psm2.overrides) == strip(idw.overrides)
    assert strip(psm2.overrides) == {
        "landscape": "Kattegat",
        "ships": "false",
        "turbines": "off",
        "porpoiseCount": "2000",
        "simYears": "5",
    }
    assert psm2.seeds == idw.seeds == (42, 43)
    assert psm2.determinism_seed == idw.determinism_seed == 42
