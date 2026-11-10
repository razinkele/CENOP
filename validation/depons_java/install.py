"""Unattended install of the DEPONS 3.2 release and inspection of the installed layout.

The release is an IzPack 5.1.3 installer. IzPack 5.1.3 cannot start on Java 21: its
PicoContainer bytecode reader fails on modern class files. The *installer* is therefore
run with a Java 11 (or 8) runtime. It only unpacks files, so this has no effect on the
model, which runs later on the system Java.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

_INSTALLER_JAVA_CANDIDATES = (
    "/usr/lib/jvm/java-11-openjdk-amd64/bin/java",
    "/usr/lib/jvm/java-11-openjdk/bin/java",
    "/usr/lib/jvm/java-8-openjdk-amd64/bin/java",
    "/usr/lib/jvm/java-1.8.0-openjdk/bin/java",
)
_RUNTIME_PREFIX = "repast.simphony.runtime_"


class InstallError(Exception):
    """The DEPONS release could not be installed or the install tree is incomplete."""


@dataclass(frozen=True)
class InstallLayout:
    root: Path
    repast_version: str
    repast_root: Path
    runtime_dir: Path
    model_dir: Path
    scenario_dir: Path
    groovy_jar: Path


def inspect_install_dir(root: Path) -> InstallLayout:
    root = Path(root).resolve()  # absolute: runs use cwd = model dir
    if not root.is_dir():
        raise InstallError(f"install dir {root} does not exist")
    repast_root = root / "repast.simphony"
    if not repast_root.is_dir():
        raise InstallError(f"{repast_root} (Repast plugins dir) not found")
    runtimes = sorted(p for p in repast_root.glob(f"{_RUNTIME_PREFIX}*") if p.is_dir())
    if len(runtimes) != 1:
        raise InstallError(
            f"expected exactly one Repast runtime ({_RUNTIME_PREFIX}<version>) in "
            f"{repast_root}, found {[p.name for p in runtimes]}"
        )
    runtime_dir = runtimes[0]
    model_dir = root / "DEPONS"
    scenario_dir = model_dir / "DEPONS.rs"
    if not (scenario_dir / "parameters.xml").is_file():
        raise InstallError(f"{scenario_dir}/parameters.xml (DEPONS model scenario) not found")
    groovy = sorted((root / "groovylib").glob("groovy-*.jar"))
    if len(groovy) != 1:
        raise InstallError(f"expected exactly one groovy jar in {root / 'groovylib'}")
    return InstallLayout(
        root=root,
        repast_version=runtime_dir.name[len(_RUNTIME_PREFIX) :],
        repast_root=repast_root,
        runtime_dir=runtime_dir,
        model_dir=model_dir,
        scenario_dir=scenario_dir,
        groovy_jar=groovy[0],
    )


def find_installer_java() -> str:
    env = os.environ.get("CENOP_DEPONS_INSTALLER_JAVA")
    candidates = (env,) if env else _INSTALLER_JAVA_CANDIDATES
    for cand in candidates:
        if cand and Path(cand).is_file():
            return cand
    raise InstallError(
        "the DEPONS 3.2 IzPack installer needs a Java 11 (or 8) runtime; it fails on "
        "Java 21. Install OpenJDK 11 or set CENOP_DEPONS_INSTALLER_JAVA to its java binary "
        f"(checked: {list(candidates)})"
    )


def install_release(
    jar: Path, dest: Path, *, installer_java: str | None = None, timeout_s: float = 1800
) -> InstallLayout:
    dest = Path(dest)
    if dest.exists():
        try:
            return inspect_install_dir(dest)
        except InstallError as exc:
            raise InstallError(
                f"{dest} exists but is not a complete install ({exc}); delete it and retry"
            ) from exc
    java = installer_java or find_installer_java()
    partial = dest.with_name(dest.name + ".partial")
    if partial.exists():
        shutil.rmtree(partial)
    dest.parent.mkdir(parents=True, exist_ok=True)
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    with tempfile.TemporaryDirectory() as tmp:
        props = Path(tmp) / "install.properties"
        props.write_text(f"INSTALL_PATH={partial}\n")
        proc = subprocess.run(
            [java, "-Djava.awt.headless=true", "-jar", str(jar), "-options", str(props)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        raise InstallError(f"installer exited {proc.returncode}:\n{tail}")
    inspect_install_dir(partial)
    os.rename(partial, dest)
    return inspect_install_dir(dest)


def install_landscape(zip_path: Path, layout: InstallLayout, landscape: str = "Kattegat") -> Path:
    data_dir = layout.model_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / f"{landscape}.zip"
    shutil.copyfile(zip_path, target)
    return target
