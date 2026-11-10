"""Collecting DEPONS outputs into per-run directories, plus provenance manifests.

DEPONS/Repast write timestamped files (e.g. ``Statistics.2026.Nov.10.12_16_51.csv``)
into one landing directory, the model dir. Run isolation:

- a run may only start when the landing dir holds no timestamped outputs;
- collection *moves* outputs out under stable names (``Statistics.csv``).

Each run directory therefore holds exactly the files that run produced.
"""

from __future__ import annotations

import fcntl
import filecmp
import json
import re
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

# yyyy.MMM.dd.HH_mm_ss[_SSS]; MMM is locale-dependent and may end in a dot ("Nov.").
_TS_NAME = re.compile(
    r"^(?P<base>.+?)\.\d{4}\.[^.]+\.?\.\d{2}\.\d{2}_\d{2}_\d{2}(?:_\d+)?(?P<rest>(?:\.[^.]+)+)$"
)
TICKS_PER_YEAR = 360 * 48  # DEPONS: 360-day years, 48 half-hour ticks per day
LOCK_NAME = "harness.lock"
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
    "ticks_expected",
    "ticks_found",
    "status",
    "locale",
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


def check_run_dirs(dirs: list[Path], *, overwrite: bool) -> None:
    """Refuse up front (before any run starts) if a run dir would need overwriting."""
    if overwrite:
        return
    busy = [str(d) for d in dirs if Path(d).exists() and any(Path(d).iterdir())]
    if busy:
        raise CollectError(f"run dir(s) not empty: {busy}; pass --overwrite to replace them")


@contextmanager
def run_lock(cache_dir: Path) -> Iterator[None]:
    """Exclusive per-install lock: one install has one landing dir, so runs are serial."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with open(cache_dir / LOCK_NAME, "w") as fh:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise CollectError(
                f"another harness process holds {cache_dir / LOCK_NAME}; runs sharing an "
                "install must be sequential"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


def expected_last_tick(sim_years: int) -> int:
    return sim_years * TICKS_PER_YEAR - 1


def read_last_tick(statistics_csv: Path) -> int | None:
    path = Path(statistics_csv)
    if not path.is_file():
        return None
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        fh.seek(max(0, fh.tell() - 4096))
        lines = [ln for ln in fh.read().decode().splitlines() if ln.strip()]
    if not lines or lines[-1].startswith('"'):
        return None
    try:
        return int(float(lines[-1].split(",")[1]))
    except (IndexError, ValueError):
        return None


def run_status(
    returncode: int,
    timed_out: bool,
    collected: dict[str, str],
    last_tick: int | None,
    expected_tick: int,
) -> str:
    ok = (
        returncode == 0
        and not timed_out
        and "Statistics.csv" in collected
        and last_tick == expected_tick
    )
    return "ok" if ok else "failed"


def assert_comparable(run_dir: Path) -> None:
    """A run dir only counts in a comparison if it holds a completed run."""
    run_dir = Path(run_dir)
    if not (run_dir / "Statistics.csv").is_file():
        raise CollectError(f"{run_dir} has no Statistics.csv; not a completed run")
    manifest = run_dir / "manifest.json"
    if manifest.is_file():
        status = json.loads(manifest.read_text()).get("status")
        if status != "ok":
            raise CollectError(f"{run_dir} manifest status is {status!r}, not 'ok'")


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
