"""Collecting DEPONS outputs into per-run directories, plus provenance manifests.

DEPONS/Repast write timestamped files (e.g. ``Statistics.2026.Nov.10.12_16_51.csv``)
into one landing directory, the model dir. Run isolation:

- a run may only start when the landing dir holds no timestamped outputs;
- collection *moves* outputs out under stable names (``Statistics.csv``).

Each run directory therefore holds exactly the files that run produced.
"""

from __future__ import annotations

import filecmp
import json
import re
import shutil
from pathlib import Path

_TS_NAME = re.compile(
    r"^(?P<base>.+?)\.\d{4}\.[^.]+\.\d{2}\.\d{2}_\d{2}_\d{2}(?:_\d+)?(?P<rest>(?:\.[^.]+)+)$"
)
# Non-timestamped files DEPONS writes into the landing dir that belong to the run.
EXTRA_LANDING_FILES = ("debug.log",)
# Files the harness itself writes into a run dir (ignored by compare_runs).
HARNESS_FILES = ("manifest.json", "java.log", "batch_params.xml")

MANIFEST_FIELDS: tuple[str, ...] = (
    "scenario_id",
    "seed",
    "parameters",
    "depons_version",
    "depons_commit",
    "jar_sha256",
    "landscape_sha256",
    "java_version",
    "repast_version",
    "harness_commit",
    "harness_dirty",
    "host",
    "started_utc",
    "wall_seconds",
    "returncode",
    "timed_out",
    "status",
    "output_files",
)


class CollectError(Exception):
    """Run outputs could not be collected without mixing or losing files."""


def normalise_output_name(name: str) -> str | None:
    match = _TS_NAME.match(name)
    return f"{match['base']}{match['rest']}" if match else None


def landing_outputs(raw_dir: Path) -> list[Path]:
    return sorted(
        p for p in Path(raw_dir).iterdir() if p.is_file() and normalise_output_name(p.name)
    )


def assert_landing_clean(raw_dir: Path) -> None:
    leftovers = landing_outputs(raw_dir)
    if leftovers:
        names = ", ".join(p.name for p in leftovers)
        raise CollectError(
            f"landing dir {raw_dir} still holds outputs from an earlier run: {names}. "
            "Move or delete them before starting a new run."
        )


def prepare_run_dir(dest: Path, *, overwrite: bool) -> None:
    dest = Path(dest)
    if dest.exists() and any(dest.iterdir()):
        if not overwrite:
            raise CollectError(f"run dir {dest} is not empty; pass --overwrite to replace it")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)


def collect_run(raw_dir: Path, dest: Path) -> dict[str, str]:
    raw_dir, dest = Path(raw_dir), Path(dest)
    plan: dict[str, Path] = {}
    for src in landing_outputs(raw_dir):
        target = normalise_output_name(src.name)
        if target in plan:
            raise CollectError(
                f"{plan[target].name} and {src.name} both normalise to {target}; "
                "the landing dir holds outputs from more than one run"
            )
        plan[target] = src
    for extra in EXTRA_LANDING_FILES:
        if (raw_dir / extra).is_file():
            plan[extra] = raw_dir / extra
    clashes = sorted(t for t in plan if (dest / t).exists())
    if clashes:
        raise CollectError(f"run dir {dest} already contains {clashes}")
    dest.mkdir(parents=True, exist_ok=True)
    for target, src in plan.items():
        shutil.move(str(src), str(dest / target))
    return {target: src.name for target, src in sorted(plan.items())}


def run_status(returncode: int, timed_out: bool, collected: dict[str, str]) -> str:
    ok = returncode == 0 and not timed_out and "Statistics.csv" in collected
    return "ok" if ok else "failed"


def write_manifest(dest: Path, manifest: dict) -> Path:
    missing = [f for f in MANIFEST_FIELDS if f not in manifest]
    if missing:
        raise CollectError(f"manifest missing required fields: {missing}")
    path = Path(dest) / "manifest.json"
    path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    return path


def compare_runs(a: Path, b: Path) -> list[str]:
    def files(d: Path) -> set[str]:
        return {p.name for p in Path(d).iterdir() if p.is_file() and p.name not in HARNESS_FILES}

    names_a, names_b = files(a), files(b)
    differing = names_a ^ names_b
    for name in names_a & names_b:
        if not filecmp.cmp(Path(a) / name, Path(b) / name, shallow=False):
            differing.add(name)
    return sorted(differing)
