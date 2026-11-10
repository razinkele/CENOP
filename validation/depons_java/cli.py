"""Command-line entry point for the A1 DEPONS-Java reference harness."""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from validation.depons_java.artifacts import (
    REPO_ROOT,
    ArtifactError,
    default_cache_dir,
    fetch_artifact,
    load_registry,
    verify_artifact,
)
from validation.depons_java.collect import (
    CollectError,
    assert_landing_clean,
    collect_run,
    compare_runs,
    prepare_run_dir,
    run_status,
    write_manifest,
)
from validation.depons_java.install import (
    InstallError,
    InstallLayout,
    install_landscape,
    install_release,
)
from validation.depons_java.runner import build_batch_command, java_version, run_batch
from validation.depons_java.scenario import (
    ScenarioError,
    effective_parameters,
    load_scenario,
    parse_parameters_xml,
    render_batch_params,
)

DEPONS_VERSION = "3.2"
DEPONS_COMMIT = "8f533f8ef59f3652056b4f23c6abecbe6cd80acb"


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, check=False
    )
    return proc.stdout.strip()


def _cmd_run(args: argparse.Namespace) -> int:
    cache = Path(args.cache) if args.cache else default_cache_dir()
    try:
        scenario = load_scenario(Path(args.scenario))
        layout = ensure_installed(cache)
        registry = load_registry()
        specs = parse_parameters_xml(layout.scenario_dir / "parameters.xml")
        overrides = dict(scenario.overrides)
        if args.porpoise_count is not None:
            overrides["porpoiseCount"] = str(args.porpoise_count)
        if args.sim_years is not None:
            overrides["simYears"] = str(args.sim_years)
        effective_parameters(specs, overrides, 0)  # validate before any run starts
        count = overrides.get("porpoiseCount", specs["porpoiseCount"].default)
        years = overrides.get("simYears", specs["simYears"].default)
        seeds = args.seeds or list(scenario.seeds)
        jvm = java_version()
        failed = 0
        for seed in seeds:
            seed_dir = Path(args.out) / scenario.id / f"N{count}-Y{years}" / f"seed-{seed}"
            prepare_run_dir(seed_dir, overwrite=args.overwrite)
            assert_landing_clean(layout.model_dir)
            params = effective_parameters(specs, overrides, seed)
            batch_params = seed_dir / "batch_params.xml"
            batch_params.write_text(render_batch_params(specs, params))
            cmd = build_batch_command(layout, batch_params)
            started = datetime.now(timezone.utc).isoformat(timespec="seconds")
            result = run_batch(cmd, layout.model_dir, seed_dir / "java.log", args.timeout_s)
            collected = collect_run(layout.model_dir, seed_dir)
            status = run_status(result.returncode, result.timed_out, collected)
            failed += status != "ok"
            write_manifest(
                seed_dir,
                {
                    "scenario_id": scenario.id,
                    "seed": seed,
                    "parameters": params,
                    "depons_version": DEPONS_VERSION,
                    "depons_commit": DEPONS_COMMIT,
                    "jar_sha256": registry["depons_jar"].sha256,
                    "landscape_sha256": registry["kattegat_landscape"].sha256,
                    "java_version": jvm,
                    "repast_version": layout.repast_version,
                    "harness_commit": _git("rev-parse", "HEAD"),
                    "harness_dirty": bool(_git("status", "--porcelain")),
                    "host": platform.node(),
                    "started_utc": started,
                    "wall_seconds": round(result.wall_seconds, 3),
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "status": status,
                    "output_files": collected,
                    "command": cmd,
                },
            )
            print(f"seed {seed}: {status} in {result.wall_seconds:.1f} s -> {seed_dir}")
    except (ArtifactError, InstallError, ScenarioError, CollectError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 1 if failed else 0


def _cmd_compare(args: argparse.Namespace) -> int:
    differing = compare_runs(Path(args.run_a), Path(args.run_b))
    for name in differing:
        print(name)
    if not differing:
        print("identical")
    return 1 if differing else 0


def ensure_installed(cache: Path) -> InstallLayout:
    """Fetch + verify the pinned artifacts, install the release, place the landscape."""
    registry = load_registry()
    jar = fetch_artifact(registry["depons_jar"], cache)
    landscape_zip = fetch_artifact(registry["kattegat_landscape"], cache)
    layout = install_release(jar, cache / "install")
    install_landscape(landscape_zip, layout)
    return layout


def _cmd_install(args: argparse.Namespace) -> int:
    cache = Path(args.cache) if args.cache else default_cache_dir()
    try:
        layout = ensure_installed(cache)
    except (ArtifactError, InstallError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    for field in (
        "root",
        "repast_version",
        "runtime_dir",
        "model_dir",
        "scenario_dir",
        "groovy_jar",
    ):
        print(f"{field}: {getattr(layout, field)}")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    cache = Path(args.cache) if args.cache else default_cache_dir()
    try:
        for artifact in load_registry().values():
            path = fetch_artifact(artifact, cache, allow_unpinned=args.allow_unpinned)
            sha = verify_artifact(path, artifact, allow_unpinned=args.allow_unpinned)
            print(f"{artifact.name} {path} {sha}")
    except ArtifactError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python3 -m validation.depons_java")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="download and verify the pinned DEPONS artifacts")
    fetch.add_argument("--allow-unpinned", action="store_true")
    fetch.add_argument("--cache", help="artifact cache dir (default $CENOP_DEPONS_CACHE)")
    fetch.set_defaults(func=_cmd_fetch)

    install = sub.add_parser("install", help="install the DEPONS release + Kattegat landscape")
    install.add_argument("--cache", help="artifact cache dir (default $CENOP_DEPONS_CACHE)")
    install.set_defaults(func=_cmd_install)

    run = sub.add_parser("run", help="run a scenario for each seed (fetch/install as needed)")
    run.add_argument("--scenario", required=True, help="scenario JSON file")
    run.add_argument("--seeds", type=int, nargs="+", help="override the scenario's seeds")
    run.add_argument("--porpoise-count", type=int, help="override porpoiseCount")
    run.add_argument("--sim-years", type=int, help="override simYears")
    run.add_argument("--out", default=str(REPO_ROOT / "output" / "depons_java"))
    run.add_argument("--timeout-s", type=float, default=21600.0, help="per-seed timeout")
    run.add_argument("--overwrite", action="store_true", help="replace existing run dirs")
    run.add_argument("--cache", help="artifact cache dir (default $CENOP_DEPONS_CACHE)")
    run.set_defaults(func=_cmd_run)

    compare = sub.add_parser("compare", help="compare two run dirs byte-wise")
    compare.add_argument("run_a")
    compare.add_argument("run_b")
    compare.set_defaults(func=_cmd_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
