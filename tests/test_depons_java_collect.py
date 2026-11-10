"""Tests for A1 output collection, run isolation, manifests and run comparison (offline)."""

import json

import pytest

from validation.depons_java.collect import (
    MANIFEST_FIELDS,
    CollectError,
    assert_landing_clean,
    collect_run,
    compare_runs,
    normalise_output_name,
    prepare_run_dir,
    run_status,
    write_manifest,
)

TS_A = "2026.Nov.10.12_16_51"
TS_B = "2026.Nov.10.13_02_07"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Statistics.2026.Nov.10.12_00_01.csv", "Statistics.csv"),
        ("YearlyMortality.2026.Nov.10.12_00_01_42.csv", "YearlyMortality.csv"),
        ("Statistics.2026.lis.10.12_00_01.csv", "Statistics.csv"),
        (
            "Statistics.2026.Nov.10.12_00_01.batch_param_map.csv",
            "Statistics.batch_param_map.csv",
        ),
        ("notes.txt", None),
        ("debug.log", None),
        # Real names from the Task 5 smoke run (NOTES.md):
        ("Hydrophones.2026.Nov.10.12_16_51.csv", "Hydrophones.csv"),
        (
            "PorpoisePerBlock.2026.Nov.10.12_16_51.batch_param_map.csv",
            "PorpoisePerBlock.batch_param_map.csv",
        ),
        ("RandomPorpoise.2026.Nov.10.12_16_51.csv", "RandomPorpoise.csv"),
        ("Reproduction.2026.Nov.10.12_16_51.csv", "Reproduction.csv"),
        ("YearlyMortality.2026.Nov.10.12_16_52_230.csv", "YearlyMortality.csv"),
    ],
)
def test_normalise_names(name, expected):
    assert normalise_output_name(name) == expected


def _put(raw, ts, tag):
    raw.mkdir(exist_ok=True)
    (raw / f"Statistics.{ts}.csv").write_text(f"stats-{tag}")
    (raw / f"YearlyMortality.{ts}_100.csv").write_text(f"mort-{tag}")


def test_collect_duplicate_normalised_names_rejected(tmp_path):
    raw = tmp_path / "raw"
    _put(raw, TS_A, "a")
    (raw / f"Statistics.{TS_B}.csv").write_text("stale")
    with pytest.raises(CollectError, match="Statistics.csv"):
        collect_run(raw, tmp_path / "d")
    assert len(list(raw.iterdir())) == 3  # nothing moved
    assert not (tmp_path / "d").exists() or not any((tmp_path / "d").iterdir())


def test_sequential_runs_collect_only_their_own_files(tmp_path):
    raw = tmp_path / "raw"
    _put(raw, TS_A, "a")
    mapping = collect_run(raw, tmp_path / "d1")
    assert mapping == {
        "Statistics.csv": f"Statistics.{TS_A}.csv",
        "YearlyMortality.csv": f"YearlyMortality.{TS_A}_100.csv",
    }
    assert_landing_clean(raw)
    _put(raw, TS_B, "b")
    collect_run(raw, tmp_path / "d2")
    assert (tmp_path / "d1" / "Statistics.csv").read_text() == "stats-a"
    assert (tmp_path / "d2" / "Statistics.csv").read_text() == "stats-b"
    assert (tmp_path / "d2" / "YearlyMortality.csv").read_text() == "mort-b"


def test_assert_landing_clean_lists_leftovers(tmp_path):
    raw = tmp_path / "raw"
    _put(raw, TS_A, "a")
    with pytest.raises(CollectError, match=f"Statistics.{TS_A}.csv"):
        assert_landing_clean(raw)


def test_prepare_run_dir_refuses_nonempty_without_overwrite(tmp_path):
    d = tmp_path / "seed-42"
    d.mkdir()
    (d / "Statistics.csv").write_text("old")
    with pytest.raises(CollectError):
        prepare_run_dir(d, overwrite=False)
    prepare_run_dir(d, overwrite=True)
    assert d.is_dir()
    assert not any(d.iterdir())


def test_collect_refuses_existing_target_name(tmp_path):
    raw, d = tmp_path / "raw", tmp_path / "d"
    _put(raw, TS_A, "a")
    d.mkdir()
    (d / "Statistics.csv").write_text("old")
    (d / "batch_params.xml").write_text("<sweep/>")
    with pytest.raises(CollectError, match="Statistics.csv"):
        collect_run(raw, d)
    assert (d / "Statistics.csv").read_text() == "old"


def test_collect_moves_debug_log_and_tolerates_harness_files(tmp_path):
    raw, d = tmp_path / "raw", tmp_path / "d"
    _put(raw, TS_A, "a")
    (raw / "debug.log").write_text("")
    d.mkdir()
    (d / "batch_params.xml").write_text("<sweep/>")
    (d / "java.log").write_text("log")
    mapping = collect_run(raw, d)
    assert mapping["debug.log"] == "debug.log"
    assert not (raw / "debug.log").exists()
    assert (d / "debug.log").exists()


def test_run_status():
    ok = {"Statistics.csv": "Statistics.x.csv"}
    assert run_status(0, False, ok) == "ok"
    assert run_status(1, False, ok) == "failed"
    assert run_status(0, True, ok) == "failed"
    assert run_status(0, False, {}) == "failed"


def test_write_manifest_requires_all_fields(tmp_path):
    full = {f: None for f in MANIFEST_FIELDS}
    path = write_manifest(tmp_path, full)
    assert path == tmp_path / "manifest.json"
    assert set(json.loads(path.read_text())) == set(MANIFEST_FIELDS)
    partial = dict(full)
    del partial["jar_sha256"]
    with pytest.raises(CollectError, match="jar_sha256"):
        write_manifest(tmp_path, partial)


def test_compare_runs_identical_differing_missing(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    for d in (a, b):
        d.mkdir()
        (d / "Statistics.csv").write_text("same")
        (d / "manifest.json").write_text(str(d))  # ignored
        (d / "java.log").write_text(str(d))  # ignored
        (d / "batch_params.xml").write_text(str(d))  # ignored
    assert compare_runs(a, b) == []
    (b / "Statistics.csv").write_text("different")
    (a / "YearlyMortality.csv").write_text("only in a")
    assert compare_runs(a, b) == ["Statistics.csv", "YearlyMortality.csv"]
