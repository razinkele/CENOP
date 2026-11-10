"""Tests for A1 output collection, run isolation, manifests and run comparison (offline)."""

import json

import pytest

from validation.depons_java.collect import (
    MANIFEST_FIELDS,
    CollectError,
    assert_comparable,
    assert_landing_clean,
    check_run_dirs,
    collect_run,
    compare_runs,
    expected_last_tick,
    normalise_output_name,
    prepare_run_dir,
    read_last_tick,
    run_lock,
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
        # Locales whose month abbreviations carry a trailing dot (de/fr/lt on modern JDKs):
        ("Statistics.2026.Nov..10.12_16_51.csv", "Statistics.csv"),
        ("YearlyMortality.2026.lapkr..10.12_00_01_42.csv", "YearlyMortality.csv"),
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
    assert run_status(0, False, ok, 17279, 17279) == "ok"
    assert run_status(1, False, ok, 17279, 17279) == "failed"
    assert run_status(0, True, ok, 17279, 17279) == "failed"
    assert run_status(0, False, {}, None, 17279) == "failed"
    # exit 0 + Statistics.csv but the run stopped early: not a completed run
    assert run_status(0, False, ok, 9000, 17279) == "failed"


def test_expected_last_tick_matches_depons_calendar():
    # 360-day years x 48 ticks; Statistics records ticks 1..N-1 (NOTES: 17,279 for 1 yr)
    assert expected_last_tick(1) == 17279
    assert expected_last_tick(5) == 86399


def test_read_last_tick(tmp_path):
    stats = tmp_path / "Statistics.csv"
    stats.write_text(
        '"run","tick","PorpoiseCount","FoodEnergyLevel","PorpoiseEnergyLevel"\n'
        "1,1.0,200,1.0,2.0\n1,17279.0,107,1.0,2.0\n"
    )
    assert read_last_tick(stats) == 17279
    assert read_last_tick(tmp_path / "missing.csv") is None
    header_only = tmp_path / "h.csv"
    header_only.write_text('"run","tick"\n')
    assert read_last_tick(header_only) is None


def test_check_run_dirs_refuses_any_nonempty_before_running(tmp_path):
    a, b = tmp_path / "seed-42", tmp_path / "seed-43"
    b.mkdir()
    (b / "Statistics.csv").write_text("old")
    with pytest.raises(CollectError, match="seed-43"):
        check_run_dirs([a, b], overwrite=False)
    check_run_dirs([a, b], overwrite=True)  # allowed; nothing deleted yet
    assert (b / "Statistics.csv").exists()


def test_run_lock_is_exclusive(tmp_path):
    with run_lock(tmp_path):
        with pytest.raises(CollectError, match="another harness"):
            with run_lock(tmp_path):
                pass
    with run_lock(tmp_path):  # released after the block
        pass


def test_assert_comparable_rejects_empty_or_failed_runs(tmp_path):
    good, empty, failed = tmp_path / "good", tmp_path / "empty", tmp_path / "failed"
    for d in (good, empty, failed):
        d.mkdir()
    (good / "Statistics.csv").write_text("x")
    (good / "manifest.json").write_text(json.dumps({"status": "ok"}))
    (failed / "Statistics.csv").write_text("x")
    (failed / "manifest.json").write_text(json.dumps({"status": "failed"}))
    assert_comparable(good)
    with pytest.raises(CollectError, match="Statistics.csv"):
        assert_comparable(empty)
    with pytest.raises(CollectError, match="failed"):
        assert_comparable(failed)


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
