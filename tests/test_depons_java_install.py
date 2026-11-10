"""Tests for A1 DEPONS install-layout inspection and landscape placement (offline)."""

from pathlib import Path

import pytest

from validation.depons_java.install import (
    InstallError,
    find_installer_java,
    inspect_install_dir,
    install_landscape,
)


def _fake_install(root, *, runtime=True):
    rs = root / "repast.simphony"
    if runtime:
        (rs / "repast.simphony.runtime_2.10.0" / "bin").mkdir(parents=True)
        (rs / "repast.simphony.runtime_2.10.0" / "lib").mkdir()
    (rs / "repast.simphony.batch_2.10.0").mkdir(parents=True)
    (root / "DEPONS" / "DEPONS.rs").mkdir(parents=True)
    (root / "DEPONS" / "DEPONS.rs" / "parameters.xml").write_text("<parameters/>")
    (root / "DEPONS" / "lib").mkdir()
    (root / "groovylib").mkdir()
    (root / "groovylib" / "groovy-3.0.17-indy.jar").write_bytes(b"jar")
    return root


def test_inspect_install_dir_complete(tmp_path):
    root = _fake_install(tmp_path / "inst")
    layout = inspect_install_dir(root)
    assert layout.root == root
    assert layout.repast_version == "2.10.0"
    assert layout.repast_root == root / "repast.simphony"
    assert layout.runtime_dir == root / "repast.simphony" / "repast.simphony.runtime_2.10.0"
    assert layout.model_dir == root / "DEPONS"
    assert layout.scenario_dir == root / "DEPONS" / "DEPONS.rs"
    assert layout.groovy_jar == root / "groovylib" / "groovy-3.0.17-indy.jar"


def test_inspect_install_dir_returns_absolute_paths(tmp_path, monkeypatch):
    # The runner uses cwd = model dir, so every layout path must be absolute.
    _fake_install(tmp_path / "inst")
    monkeypatch.chdir(tmp_path)
    layout = inspect_install_dir(Path("inst"))
    assert layout.root.is_absolute()
    assert layout.runtime_dir.is_absolute()
    assert layout.scenario_dir.is_absolute()
    assert layout.groovy_jar.is_absolute()


def test_inspect_install_dir_missing_runtime_names_it(tmp_path):
    root = _fake_install(tmp_path / "inst", runtime=False)
    with pytest.raises(InstallError, match="runtime"):
        inspect_install_dir(root)


def test_install_landscape_copies_to_data_dir(tmp_path):
    layout = inspect_install_dir(_fake_install(tmp_path / "inst"))
    src = tmp_path / "Kattegat-2614f56a.zip"
    src.write_bytes(b"zipbytes")
    target = install_landscape(src, layout)
    assert target == layout.model_dir / "data" / "Kattegat.zip"
    assert target.read_bytes() == b"zipbytes"


def test_find_installer_java_prefers_env(monkeypatch, tmp_path):
    fake = tmp_path / "java"
    fake.write_text("#!/bin/sh\n")
    monkeypatch.setenv("CENOP_DEPONS_INSTALLER_JAVA", str(fake))
    assert find_installer_java() == str(fake)


def test_find_installer_java_missing_explains(monkeypatch, tmp_path):
    monkeypatch.setenv("CENOP_DEPONS_INSTALLER_JAVA", str(tmp_path / "nope"))
    with pytest.raises(InstallError, match="Java 11"):
        find_installer_java()
