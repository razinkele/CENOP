"""Pinned DEPONS release artifacts and checksum-verified fetching.

Every artifact is verified by size and SHA-256 before use. Downloads go to a ``.part``
file and are only moved onto the final name once verified, so an interrupted or corrupt
download can never be mistaken for a good cached copy.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

REGISTRY_PATH = Path(__file__).resolve().parent / "artifacts.json"
REPO_ROOT = Path(__file__).resolve().parents[2]
_CHUNK = 1024 * 1024


class ArtifactError(Exception):
    """An artifact failed verification or could not be fetched."""


@dataclass(frozen=True)
class Artifact:
    name: str
    url: str
    size: int
    sha256: str | None
    filename: str


def load_registry(path: Path = REGISTRY_PATH) -> dict[str, Artifact]:
    raw = json.loads(Path(path).read_text())
    return {
        name: Artifact(
            name=name,
            url=entry["url"],
            size=int(entry["size"]),
            sha256=entry["sha256"],
            filename=entry["filename"],
        )
        for name, entry in raw.items()
    }


def default_cache_dir() -> Path:
    env = os.environ.get("CENOP_DEPONS_CACHE")
    return Path(env) if env else REPO_ROOT / ".depons-cache"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_artifact(path: Path, artifact: Artifact, *, allow_unpinned: bool = False) -> str:
    size = Path(path).stat().st_size
    if size != artifact.size:
        raise ArtifactError(f"{artifact.name}: size {size} != expected {artifact.size} ({path})")
    actual = sha256_file(path)
    if artifact.sha256 is None:
        if not allow_unpinned:
            raise ArtifactError(
                f"{artifact.name}: no pinned SHA-256 (actual {actual}); "
                "pin it in artifacts.json or pass --allow-unpinned"
            )
        return actual
    if actual != artifact.sha256.lower():
        raise ArtifactError(
            f"{artifact.name}: SHA-256 {actual} != pinned {artifact.sha256} ({path})"
        )
    return actual


def fetch_artifact(artifact: Artifact, cache_dir: Path, *, allow_unpinned: bool = False) -> Path:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    final = cache_dir / artifact.filename
    if final.exists():
        try:
            verify_artifact(final, artifact, allow_unpinned=allow_unpinned)
            return final
        except ArtifactError:
            final.unlink()
    part = cache_dir / f"{artifact.filename}.part"
    try:
        with urllib.request.urlopen(artifact.url) as resp, open(part, "wb") as out:
            shutil.copyfileobj(resp, out, _CHUNK)
        verify_artifact(part, artifact, allow_unpinned=allow_unpinned)
        os.replace(part, final)
    except BaseException as exc:
        part.unlink(missing_ok=True)
        if isinstance(exc, ArtifactError):
            raise
        if isinstance(exc, OSError):
            raise ArtifactError(f"{artifact.name}: download failed: {exc}") from exc
        raise
    return final
