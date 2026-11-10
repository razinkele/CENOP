"""Tests for the A1 DEPONS-Java harness artifact registry and verified fetch (offline)."""

import hashlib

import pytest

from validation.depons_java.artifacts import (
    Artifact,
    ArtifactError,
    fetch_artifact,
    load_registry,
    sha256_file,
    verify_artifact,
)

KATTEGAT_SHA = "2614f56a771bf9c486a3186779040b2426b6027cfa4bb50aa61f9cd60ff94913"


def _source(tmp_path, data=b"landscape-bytes"):
    src = tmp_path / "src" / "thing.zip"
    src.parent.mkdir()
    src.write_bytes(data)
    return src, Artifact(
        name="thing",
        url=src.as_uri(),
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
        filename="thing.zip",
    )


def test_sha256_file_known_value(tmp_path):
    p = tmp_path / "abc"
    p.write_bytes(b"abc")
    assert sha256_file(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_verify_rejects_wrong_size(tmp_path):
    src, art = _source(tmp_path)
    bad = Artifact(art.name, art.url, art.size + 1, art.sha256, art.filename)
    with pytest.raises(ArtifactError):
        verify_artifact(src, bad)


def test_verify_rejects_wrong_sha(tmp_path):
    src, art = _source(tmp_path)
    bad = Artifact(art.name, art.url, art.size, "0" * 64, art.filename)
    with pytest.raises(ArtifactError):
        verify_artifact(src, bad)


def test_verify_unpinned_requires_flag(tmp_path):
    src, art = _source(tmp_path)
    unpinned = Artifact(art.name, art.url, art.size, None, art.filename)
    with pytest.raises(ArtifactError):
        verify_artifact(src, unpinned)
    assert verify_artifact(src, unpinned, allow_unpinned=True) == art.sha256


def test_fetch_downloads_and_verifies(tmp_path):
    _, art = _source(tmp_path)
    cache = tmp_path / "cache"
    out = fetch_artifact(art, cache)
    assert out == cache / "thing.zip"
    assert out.exists()
    assert verify_artifact(out, art) == art.sha256
    assert not list(cache.glob("*.part"))


def test_fetch_skips_when_cached_valid(tmp_path):
    _, art = _source(tmp_path)
    cache = tmp_path / "cache"
    first = fetch_artifact(art, cache)
    mtime = first.stat().st_mtime_ns
    second = fetch_artifact(art, cache)
    assert second.stat().st_mtime_ns == mtime


def test_fetch_replaces_corrupt_cached_file(tmp_path):
    src, art = _source(tmp_path)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "thing.zip").write_bytes(b"truncated")
    out = fetch_artifact(art, cache)
    assert out.read_bytes() == src.read_bytes()


def test_fetch_size_mismatch_leaves_nothing(tmp_path):
    _, art = _source(tmp_path)
    bad = Artifact(art.name, art.url, art.size + 5, art.sha256, art.filename)
    cache = tmp_path / "cache"
    with pytest.raises(ArtifactError):
        fetch_artifact(bad, cache)
    assert not (cache / "thing.zip").exists()
    assert not list(cache.glob("*.part"))


def test_registry_pins_landscape():
    reg = load_registry()
    land = reg["kattegat_landscape"]
    assert land.url == "https://depons.eu/files/landscapes/v3.2/Kattegat-2614f56a.zip"
    assert land.size == 38927699
    assert land.sha256 == KATTEGAT_SHA
    assert reg["depons_jar"].size == 287091000
