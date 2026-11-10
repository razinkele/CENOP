"""Tests for the A1 headless Repast batch runner (offline; no DEPONS run)."""

import shlex
import shutil
import sys
import time
from pathlib import Path

import pytest

from validation.depons_java.artifacts import default_cache_dir
from validation.depons_java.install import InstallLayout
from validation.depons_java.runner import (
    JVM_FLAGS,
    REPAST_BATCH_MAIN,
    build_batch_command,
    build_classpath,
    java_version,
    run_batch,
)


def _layout(root=Path("/inst")):
    rt = root / "repast.simphony" / "repast.simphony.runtime_2.10.0"
    model = root / "DEPONS"
    return InstallLayout(
        root=root,
        repast_version="2.10.0",
        repast_root=root / "repast.simphony",
        runtime_dir=rt,
        model_dir=model,
        scenario_dir=model / "DEPONS.rs",
        groovy_jar=root / "groovylib" / "groovy-3.0.17-indy.jar",
    )


def test_build_classpath_matches_launcher():
    lay = _layout()
    assert build_classpath(lay).split(":") == [
        f"{lay.runtime_dir}/bin",
        f"{lay.runtime_dir}/lib/*",
        str(lay.groovy_jar),
        f"{lay.model_dir}/lib/*",
    ]


def test_build_batch_command_shape(tmp_path):
    lay = _layout()
    params = tmp_path / "batch_params.xml"
    cmd = build_batch_command(lay, params)
    assert cmd[0] == "java"
    assert "-Xmx3072M" in cmd
    assert "-Djava.awt.headless=true" in cmd
    start = cmd.index(JVM_FLAGS[0])
    assert tuple(cmd[start : start + len(JVM_FLAGS)]) == JVM_FLAGS
    cp_at = cmd.index("-cp")
    assert cmd[cp_at + 1] == build_classpath(lay)
    assert cmd[cp_at + 2 :] == [REPAST_BATCH_MAIN, "-params", str(params), str(lay.scenario_dir)]
    assert REPAST_BATCH_MAIN == "repast.simphony.runtime.RepastBatchMain"


def test_build_batch_command_resolves_relative_params(tmp_path, monkeypatch):
    # Java runs with cwd = model dir, so a relative params path would resolve there.
    monkeypatch.chdir(tmp_path)
    cmd = build_batch_command(_layout(), Path("out/batch_params.xml"))
    assert cmd[cmd.index("-params") + 1] == str(tmp_path / "out" / "batch_params.xml")


def test_run_batch_captures_output_and_exit_code(tmp_path):
    log = tmp_path / "java.log"
    res = run_batch(
        [sys.executable, "-c", "print('hello'); raise SystemExit(3)"], tmp_path, log, 30
    )
    assert res.returncode == 3
    assert res.timed_out is False
    assert res.log_path == log
    assert "hello" in log.read_text()
    assert res.wall_seconds >= 0


def test_run_batch_timeout_kills(tmp_path):
    t0 = time.monotonic()
    res = run_batch(
        [sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, tmp_path / "l", 0.5
    )
    assert res.timed_out is True
    assert res.returncode == -9
    assert time.monotonic() - t0 < 5


@pytest.mark.skipif(shutil.which("java") is None, reason="no java on PATH")
def test_java_version_reports_version():
    assert "version" in java_version()


_LAUNCHER = default_cache_dir() / "install" / "start_model.command"


@pytest.mark.skipif(not _LAUNCHER.is_file(), reason="DEPONS release not installed locally")
def test_jvm_flags_match_installed_launcher():
    java_line = next(
        line for line in _LAUNCHER.read_text().splitlines() if line.startswith("java ")
    )
    tokens = shlex.split(java_line)
    flags = tokens[tokens.index("-Xmx3072M") + 1 : tokens.index("-cp")]
    assert tuple(flags) == JVM_FLAGS
