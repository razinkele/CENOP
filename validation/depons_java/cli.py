"""Command-line entry point for the A1 DEPONS-Java reference harness."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from validation.depons_java.artifacts import (
    ArtifactError,
    default_cache_dir,
    fetch_artifact,
    load_registry,
    verify_artifact,
)
from validation.depons_java.install import (
    InstallError,
    InstallLayout,
    install_landscape,
    install_release,
)


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
