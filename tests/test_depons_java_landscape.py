"""Tests for the A1 landscape identity check (offline)."""

from validation.depons_java.landscape import LandscapeReport, compare_dirs, render_report


def _tree(root, files):
    for rel, data in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    return root


def test_compare_dirs_classifies_all_five(tmp_path):
    ref = _tree(
        tmp_path / "ref",
        {
            "bathy.asc": b"same",
            "blocks.asc": b"ref-version",
            "only_ref.asc": b"r",
            "prey0000_01_XX.asc": b"prey-jan",
        },
    )
    cand = _tree(
        tmp_path / "cand",
        {
            "bathy.asc": b"same",
            "blocks.asc": b"cand-version",
            "only_cand.json": b"c",
            "prey0000_01.asc": b"prey-jan",
        },
    )
    report = compare_dirs(ref, cand)
    assert report == LandscapeReport(
        identical=["bathy.asc"],
        differing=["blocks.asc"],
        renamed_identical=["prey0000_01_XX.asc -> prey0000_01.asc"],
        missing_in_candidate=["only_ref.asc"],
        extra_in_candidate=["only_cand.json"],
    )


def test_compare_dirs_recurses_with_posix_paths(tmp_path):
    ref = _tree(tmp_path / "ref", {"sub/a.asc": b"x"})
    cand = _tree(tmp_path / "cand", {"sub/a.asc": b"x"})
    assert compare_dirs(ref, cand).identical == ["sub/a.asc"]


def test_render_report_lists_differing_files(tmp_path):
    report = LandscapeReport(
        identical=["bathy.asc"],
        differing=["blocks.asc"],
        renamed_identical=["p_XX.asc -> p.asc"],
        missing_in_candidate=[],
        extra_in_candidate=["ships.json"],
    )
    text = render_report(report)
    assert "| identical | 1 |" in text
    assert "| differing | 1 |" in text
    assert "blocks.asc" in text
    assert "p_XX.asc -> p.asc" in text
    assert "ships.json" in text
