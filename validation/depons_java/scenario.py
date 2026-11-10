"""Scenario files, DEPONS parameter merging and Repast batch sweep rendering.

A scenario is a small JSON file holding overrides of the DEPONS 3.2 ``parameters.xml``
defaults plus an explicit seed list. The effective parameter set for a run is the full
default set with those overrides and ``randomSeed`` applied. It is rendered as a Repast
``batch_params.xml`` sweep that pins every parameter as a constant.

XML parsing uses the stdlib parser on trusted input only: the checksum-verified DEPONS
release's own ``parameters.xml`` and XML this module generates.
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

SCENARIO_DIR = Path(__file__).resolve().parent / "scenarios"

_CONSTANT_TYPE = {"int": "int", "boolean": "boolean", "string": "java.lang.String"}
_REQUIRED_KEYS = ("id", "description", "overrides", "seeds", "determinism_seed")


class ScenarioError(Exception):
    """A scenario file or parameter override is invalid."""


@dataclass(frozen=True)
class ParamSpec:
    name: str
    type: str
    default: str
    allowed: tuple[str, ...] | None = None  # from parameters.xml `values`, if constrained


@dataclass(frozen=True)
class Scenario:
    id: str
    description: str
    overrides: dict[str, str]
    seeds: tuple[int, ...]
    determinism_seed: int


def load_scenario(path: Path) -> Scenario:
    data = json.loads(Path(path).read_text())
    missing = [k for k in _REQUIRED_KEYS if k not in data]
    if missing:
        raise ScenarioError(f"{path}: missing keys {missing}")
    seeds = data["seeds"]
    if not isinstance(seeds, list) or not seeds:
        raise ScenarioError(f"{path}: 'seeds' must be a non-empty list")
    if not all(isinstance(s, int) and not isinstance(s, bool) for s in seeds):
        raise ScenarioError(f"{path}: every seed must be an integer")
    if len(set(seeds)) != len(seeds):
        raise ScenarioError(f"{path}: duplicate seeds {seeds}")
    if data["determinism_seed"] not in seeds:
        raise ScenarioError(f"{path}: determinism_seed {data['determinism_seed']} not in seeds")
    overrides = data["overrides"]
    if not isinstance(overrides, dict):
        raise ScenarioError(f"{path}: 'overrides' must be an object")
    return Scenario(
        id=str(data["id"]),
        description=str(data["description"]),
        overrides={str(k): str(v) for k, v in overrides.items()},
        seeds=tuple(seeds),
        determinism_seed=int(data["determinism_seed"]),
    )


def parse_parameters_xml(path: Path) -> dict[str, ParamSpec]:
    root = ET.parse(path).getroot()
    specs: dict[str, ParamSpec] = {}
    for elem in root.iter("parameter"):
        name, ptype = elem.get("name"), elem.get("type")
        if ptype not in _CONSTANT_TYPE:
            raise ScenarioError(f"{path}: parameter {name!r} has unsupported type {ptype!r}")
        values = elem.get("values")
        specs[name] = ParamSpec(
            name=name,
            type=ptype,
            default=elem.get("defaultValue", ""),
            allowed=tuple(re.findall(r"'([^']*)'", values)) if values else None,
        )
    return specs


def _check_value(spec: ParamSpec, value: str) -> None:
    if spec.type == "int":
        try:
            int(value)
        except ValueError:
            raise ScenarioError(f"override {spec.name}={value!r} is not an integer") from None
    elif spec.type == "boolean" and value not in ("true", "false"):
        raise ScenarioError(f"override {spec.name}={value!r} must be 'true' or 'false'")
    if spec.allowed is not None and value not in spec.allowed:
        raise ScenarioError(
            f"override {spec.name}={value!r} is not one of the allowed values {list(spec.allowed)}"
        )


def effective_parameters(
    specs: dict[str, ParamSpec], overrides: dict[str, str], seed: int
) -> dict[str, str]:
    unknown = sorted(set(overrides) - set(specs))
    if unknown:
        raise ScenarioError(f"unknown DEPONS parameter(s) in overrides: {unknown}")
    params = {name: spec.default for name, spec in specs.items()}
    for name, value in overrides.items():
        _check_value(specs[name], value)
        params[name] = value
    params["randomSeed"] = str(seed)
    return params


def render_batch_params(specs: dict[str, ParamSpec], params: dict[str, str]) -> str:
    sweep = ET.Element("sweep", runs="1")
    for name, spec in specs.items():
        ET.SubElement(
            sweep,
            "parameter",
            name=name,
            type="constant",
            constant_type=_CONSTANT_TYPE[spec.type],
            value=params[name],
        )
    return '<?xml version="1.0" ?>' + ET.tostring(sweep, encoding="unicode")
