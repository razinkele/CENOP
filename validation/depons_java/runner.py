"""Headless DEPONS 3.2 runs through Repast Simphony's batch entry point.

The classpath and JVM flags mirror the release's ``start_model.command`` exactly. Only
the main class differs: ``RepastBatchMain``, the headless twin of ``RepastMain``, which
reads a parameter sweep file. Runs execute with the model dir as working directory,
because DEPONS resolves ``data/<landscape>.zip`` and writes its outputs relative to it.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from validation.depons_java.install import InstallLayout

REPAST_BATCH_MAIN = "repast.simphony.runtime.RepastBatchMain"

# Verbatim from DEPONS 3.2 start_model.command (minus -Xmx, which is a parameter).
JVM_FLAGS: tuple[str, ...] = (
    "-XX:+IgnoreUnrecognizedVMOptions",
    "--add-modules=ALL-SYSTEM",
    "--add-exports=java.base/jdk.internal.ref=ALL-UNNAMED",
    "--add-exports=java.base/java.lang=ALL-UNNAMED",
    "--add-exports=java.desktop/sun.awt=ALL-UNNAMED",
    "--add-exports=java.desktop/sun.java2d=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.lang=ALL-UNNAMED",
    "--add-opens",
    "java.base/java.util=ALL-UNNAMED",
    "--add-exports",
    "java.xml/com.sun.org.apache.xpath.internal=ALL-UNNAMED",
    "--add-exports",
    "java.xml/com.sun.org.apache.xpath.internal.objects=ALL-UNNAMED",
)


@dataclass(frozen=True)
class RunResult:
    returncode: int
    timed_out: bool
    wall_seconds: float
    log_path: Path


def build_classpath(layout: InstallLayout) -> str:
    return ":".join(
        [
            f"{layout.runtime_dir}/bin",
            f"{layout.runtime_dir}/lib/*",
            str(layout.groovy_jar),
            f"{layout.model_dir}/lib/*",
        ]
    )


def build_batch_command(
    layout: InstallLayout, batch_params: Path, *, java: str = "java", xmx: str = "3072M"
) -> list[str]:
    return [
        java,
        "-Djava.awt.headless=true",
        f"-Xmx{xmx}",
        *JVM_FLAGS,
        "-cp",
        build_classpath(layout),
        REPAST_BATCH_MAIN,
        "-params",
        # Absolute: Java runs with cwd = model dir, where a relative path would resolve.
        str(Path(batch_params).resolve()),
        str(layout.scenario_dir),
    ]


RUN_LOCALE = "C.UTF-8"  # DEPONS timestamps and number formats follow the JVM default locale


def run_batch(cmd: list[str], cwd: Path, log_path: Path, timeout_s: float) -> RunResult:
    env = {k: v for k, v in os.environ.items() if k != "DISPLAY"}
    env["LC_ALL"] = env["LANG"] = RUN_LOCALE
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=log, stderr=subprocess.STDOUT)
        try:
            returncode = proc.wait(timeout=timeout_s)
            timed_out = False
        except subprocess.TimeoutExpired:
            returncode, timed_out = -9, True
        finally:
            # Also on Ctrl-C/SIGTERM: never leave a JVM writing into the landing dir.
            if proc.poll() is None:
                proc.kill()
                proc.wait()
    return RunResult(returncode, timed_out, time.monotonic() - start, log_path)


def java_version(java: str = "java") -> str:
    proc = subprocess.run([java, "-version"], capture_output=True, text=True, check=False)
    lines = (proc.stderr or proc.stdout).splitlines()
    return lines[0].strip() if lines else ""
