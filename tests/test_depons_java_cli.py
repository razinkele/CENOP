"""End-to-end tests of the A1 `run`/`compare` CLI with a stub in place of Java (offline)."""

import hashlib
import json

import pytest

from validation.depons_java import cli
from validation.depons_java.artifacts import load_registry
from validation.depons_java.install import inspect_install_dir
from validation.depons_java.runner import RunResult

PARAMS_XML = """<?xml version="1.0" encoding="UTF-8" ?>
<parameters>
  <parameter name="randomSeed" type="int" defaultValue="__NULL__"/>
  <parameter name="porpoiseCount" type="int" defaultValue="10000"/>
  <parameter name="simYears" type="int" defaultValue="50"/>
  <parameter name="landscape" type="string" defaultValue="NorthSea"
             values="'NorthSea' 'Kattegat' 'UserDefined'"/>
  <parameter name="dispersal" type="string" defaultValue="PSM-Type2"
             values="'off' 'PSM-Type2' 'Undirected' 'InnerDanishWaters'"/>
  <parameter name="ships" type="boolean" defaultValue="false"/>
  <parameter name="turbines" type="string" defaultValue="off" values="'off' 'User-def'"/>
</parameters>
"""
ZIP_BYTES = b"fake kattegat zip"


@pytest.fixture
def env(tmp_path, monkeypatch):
    root = tmp_path / "install"
    (root / "repast.simphony" / "repast.simphony.runtime_2.10.0").mkdir(parents=True)
    (root / "DEPONS" / "DEPONS.rs").mkdir(parents=True)
    (root / "DEPONS" / "DEPONS.rs" / "parameters.xml").write_text(PARAMS_XML)
    (root / "DEPONS" / "data").mkdir()
    (root / "DEPONS" / "data" / "Kattegat.zip").write_bytes(ZIP_BYTES)
    (root / "groovylib").mkdir()
    (root / "groovylib" / "groovy-3.0.17-indy.jar").write_bytes(b"jar")
    pin = load_registry()["depons_jar"].sha256
    (root / ".cenop-install.json").write_text(json.dumps({"jar_sha256": pin}))
    layout = inspect_install_dir(root)

    scenario = tmp_path / "scenario.json"
    scenario.write_text(
        json.dumps(
            {
                "id": "kattegat-test",
                "description": "test",
                "overrides": {
                    "landscape": "Kattegat",
                    "dispersal": "PSM-Type2",
                    "ships": "false",
                    "turbines": "off",
                    "porpoiseCount": "200",
                    "simYears": "1",
                },
                "seeds": [1, 2],
                "determinism_seed": 1,
            }
        )
    )
    calls = []
    behaviour = {"rc": 0, "last_tick": 17279, "timed_out": False}

    def fake_run_batch(cmd, cwd, log_path, timeout_s):
        calls.append(cmd)
        ts = f"2026.Nov.10.12_00_{len(calls):02d}"
        (cwd / f"Statistics.{ts}.csv").write_text(
            '"run","tick","PorpoiseCount","FoodEnergyLevel","PorpoiseEnergyLevel"\n'
            f"1,1.0,200,1.0,2.0\n1,{behaviour['last_tick']}.0,150,1.0,2.0\n"
        )
        log_path.write_text("stub run\n")
        return RunResult(behaviour["rc"], behaviour["timed_out"], 1.0, log_path)

    monkeypatch.setattr(cli, "ensure_installed", lambda cache: layout)
    monkeypatch.setattr(cli, "java_version", lambda: "test-java 21")
    monkeypatch.setattr(cli, "run_batch", fake_run_batch)
    out = tmp_path / "out"
    base = ["run", "--scenario", str(scenario), "--out", str(out), "--cache", str(tmp_path)]
    return {
        "layout": layout,
        "out": out,
        "base": base,
        "calls": calls,
        "behaviour": behaviour,
        "pin": pin,
    }


def _manifest(env, seed):
    d = env["out"] / "kattegat-test" / "N200-Y1" / f"seed-{seed}"
    return json.loads((d / "manifest.json").read_text())


def test_cmd_run_success_writes_ok_manifests(env):
    assert cli.main(env["base"]) == 0
    assert len(env["calls"]) == 2
    m = _manifest(env, 1)
    assert m["status"] == "ok"
    assert m["ticks_expected"] == m["ticks_found"] == 17279
    assert m["locale"] == "C.UTF-8"
    assert m["jar_sha256"] == env["pin"]
    assert m["landscape_sha256"] == hashlib.sha256(ZIP_BYTES).hexdigest()
    assert m["parameters"]["randomSeed"] == "1"
    assert m["output_files"]["Statistics.csv"].startswith("Statistics.2026")
    assert _manifest(env, 2)["parameters"]["randomSeed"] == "2"
    leftovers = [p for p in env["layout"].model_dir.iterdir() if p.suffix == ".csv"]
    assert leftovers == []


def test_cmd_run_nonzero_exit_records_failed(env):
    env["behaviour"]["rc"] = 1
    assert cli.main(env["base"]) == 1
    assert _manifest(env, 1)["status"] == "failed"
    assert _manifest(env, 1)["returncode"] == 1


def test_cmd_run_timeout_records_failed(env):
    env["behaviour"].update(rc=-9, timed_out=True)
    assert cli.main(env["base"]) == 1
    assert _manifest(env, 1)["timed_out"] is True
    assert _manifest(env, 1)["status"] == "failed"


def test_cmd_run_incomplete_run_records_failed(env):
    env["behaviour"]["last_tick"] = 9000
    assert cli.main(env["base"]) == 1
    m = _manifest(env, 1)
    assert m["status"] == "failed"
    assert m["ticks_found"] == 9000


def test_cmd_run_checks_every_seed_dir_before_running(env):
    busy = env["out"] / "kattegat-test" / "N200-Y1" / "seed-2"
    busy.mkdir(parents=True)
    (busy / "Statistics.csv").write_text("hours of earlier work")
    assert cli.main(env["base"]) == 1
    assert env["calls"] == []  # refused before seed 1 started
    assert not (env["out"] / "kattegat-test" / "N200-Y1" / "seed-1").exists()


def test_cmd_run_duplicate_seeds_refused(env):
    assert cli.main(env["base"] + ["--seeds", "1", "1"]) == 1
    assert env["calls"] == []


def test_cmd_run_overwrite_keeps_old_run_when_landing_dirty(env):
    old = env["out"] / "kattegat-test" / "N200-Y1" / "seed-1"
    old.mkdir(parents=True)
    (old / "Statistics.csv").write_text("previous good run")
    (env["layout"].model_dir / "Statistics.2026.Nov.10.09_00_00.csv").write_text("leftover")
    assert cli.main(env["base"] + ["--overwrite"]) == 1
    assert (old / "Statistics.csv").read_text() == "previous good run"
    assert env["calls"] == []


def test_cmd_run_rejects_mistyped_dispersal_before_running(env, tmp_path):
    bad = tmp_path / "bad.json"
    data = json.loads((tmp_path / "scenario.json").read_text())
    data["overrides"]["dispersal"] = "PSM-Typ2"
    bad.write_text(json.dumps(data))
    argv = ["run", "--scenario", str(bad), "--out", str(env["out"]), "--cache", str(tmp_path)]
    assert cli.main(argv) == 1
    assert env["calls"] == []


def test_git_failure_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "REPO_ROOT", tmp_path)  # not a git repo
    assert cli._git("rev-parse", "HEAD") is None


def test_compare_refuses_vacuous_comparison(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    assert cli.main(["compare", str(a), str(b)]) == 2
