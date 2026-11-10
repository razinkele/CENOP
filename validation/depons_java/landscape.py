"""Landscape identity check: the verified DEPONS v3.2 landscape vs CENOP's data dir.

Files are compared by SHA-256. A file whose content matches a reference file under a
different name (e.g. DEPONS ``prey0000_01_XX.asc`` vs CENOP ``prey0000_01.asc``) is
reported as ``renamed_identical`` rather than as missing plus extra.
"""

from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from validation.depons_java.artifacts import sha256_file


@dataclass(frozen=True)
class LandscapeReport:
    identical: list[str]
    differing: list[str]
    renamed_identical: list[str]
    missing_in_candidate: list[str]
    extra_in_candidate: list[str]


def _hashes(root: Path) -> dict[str, str]:
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256_file(p) for p in root.rglob("*") if p.is_file()}


def compare_dirs(reference: Path, candidate: Path) -> LandscapeReport:
    ref, cand = _hashes(reference), _hashes(candidate)
    common = ref.keys() & cand.keys()
    identical = sorted(n for n in common if ref[n] == cand[n])
    differing = sorted(n for n in common if ref[n] != cand[n])
    only_ref = sorted(ref.keys() - cand.keys())
    only_cand = sorted(cand.keys() - ref.keys())
    renamed, matched_cand = [], set()
    for r in only_ref:
        match = next((c for c in only_cand if c not in matched_cand and cand[c] == ref[r]), None)
        if match:
            renamed.append(f"{r} -> {match}")
            matched_cand.add(match)
    matched_ref = {entry.split(" -> ")[0] for entry in renamed}
    return LandscapeReport(
        identical=identical,
        differing=differing,
        renamed_identical=renamed,
        missing_in_candidate=[r for r in only_ref if r not in matched_ref],
        extra_in_candidate=[c for c in only_cand if c not in matched_cand],
    )


def extract_landscape(zip_path: Path, dest: Path) -> Path:
    """Extract the verified landscape zip; return the dir holding the landscape files."""
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(dest)
    entries = [p for p in dest.iterdir()]
    if len(entries) == 1 and entries[0].is_dir():
        return entries[0]
    return dest


def render_report(report: LandscapeReport) -> str:
    lines = ["| category | files |", "|---|---|"]
    sections = [
        ("identical", report.identical),
        ("differing", report.differing),
        ("renamed (identical content)", report.renamed_identical),
        ("missing in candidate", report.missing_in_candidate),
        ("extra in candidate", report.extra_in_candidate),
    ]
    lines += [f"| {name} | {len(items)} |" for name, items in sections]
    for name, items in sections[1:]:
        if items:
            lines += ["", f"**{name}:**"] + [f"- `{item}`" for item in items]
    return "\n".join(lines)
