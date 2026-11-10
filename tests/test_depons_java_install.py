"""Tests for A1 DEPONS install-layout inspection and landscape placement (offline)."""

import hashlib
import os
from pathlib import Path

import pytest

from validation.depons_java.install import (
    InstallError,
    find_installer_java,
    inspect_install_dir,
    install_landscape,
    install_release,
    read_install_marker,
    verify_install,
)

FAKE_INSTALLER = """#!/bin/bash
props="${@: -1}"
dest="$(sed -n 's/^INSTALL_PATH=//p' "$props")"
mkdir -p "$dest/repast.simphony/repast.simphony.runtime_2.10.0/bin" \\
         "$dest/DEPONS/DEPONS.rs" "$dest/groovylib"
echo '<parameters/>' > "$dest/DEPONS/DEPONS.rs/parameters.xml"
touch "$dest/groovylib/groovy-3.0.17-indy.jar"
exit "${FAKE_EXIT:-0}"
"""


def _fake_java(tmp_path):
    script = tmp_path / "fake-java"
    script.write_text(FAKE_INSTALLER)
    script.chmod(0o755)
    jar = tmp_path / "DEPONS-3.2-setup.jar"
    jar.write_bytes(b"not really a jar")
    return str(script), jar


def test_install_release_with_fake_installer_writes_marker(tmp_path):
    java, jar = _fake_java(tmp_path)
    layout = install_release(jar, tmp_path / "install", installer_java=java)
    assert layout.root == (tmp_path / "install").resolve()
    marker = read_install_marker(layout.root)
    assert marker["jar_sha256"] == hashlib.sha256(jar.read_bytes()).hexdigest()
    assert not (tmp_path / "install.partial").exists()


def test_install_release_failing_installer_leaves_no_dest(tmp_path, monkeypatch):
    java, jar = _fake_java(tmp_path)
    monkeypatch.setenv("FAKE_EXIT", "1")
    with pytest.raises(InstallError, match="exited 1"):
        install_release(jar, tmp_path / "install", installer_java=java)
    assert not (tmp_path / "install").exists()


def test_install_release_refuses_incomplete_existing_dest(tmp_path):
    java, jar = _fake_java(tmp_path)
    (tmp_path / "install").mkdir()
    with pytest.raises(InstallError, match="delete it"):
        install_release(jar, tmp_path / "install", installer_java=java)


def test_verify_install_binds_install_to_pinned_jar(tmp_path):
    java, jar = _fake_java(tmp_path)
    layout = install_release(jar, tmp_path / "install", installer_java=java)
    good = hashlib.sha256(jar.read_bytes()).hexdigest()
    assert verify_install(layout, good) == good
    with pytest.raises(InstallError, match="reinstall"):
        verify_install(layout, "0" * 64)
    (layout.root / ".cenop-install.json").unlink()
    with pytest.raises(InstallError, match="marker"):
        verify_install(layout, good)


def test_install_landscape_skips_identical_copy(tmp_path):
    layout = inspect_install_dir(_fake_install(tmp_path / "inst"))
    src = tmp_path / "Kattegat.zip"
    src.write_bytes(b"zipbytes")
    target = install_landscape(src, layout)
    os.utime(target, ns=(1, 1))
    install_landscape(src, layout)  # identical content: must not rewrite a live file
    assert target.stat().st_mtime_ns == 1


def test_install_landscape_replaces_differing_atomically(tmp_path):
    layout = inspect_install_dir(_fake_install(tmp_path / "inst"))
    data = layout.model_dir / "data"
    data.mkdir()
    (data / "Kattegat.zip").write_bytes(b"stale")
    src = tmp_path / "Kattegat.zip"
    src.write_bytes(b"fresh")
    install_landscape(src, layout)
    assert (data / "Kattegat.zip").read_bytes() == b"fresh"
    assert sorted(p.name for p in data.iterdir()) == ["Kattegat.zip"]


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
