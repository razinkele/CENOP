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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)
