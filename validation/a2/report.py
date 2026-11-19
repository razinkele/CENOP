"""Build the A2 parity report from the run outputs.

python3 -m validation.a2.report [--out docs/validation/A2-parity-report.md]

Reads output/a2/java/a2-<scenario>/N2000-Y<years>/seed-*/ (Track A1 harness) and
output/a2/cenop/<scenario>/seed-*.csv (validation.a2.run_cenop). Day d is the end of day d:
Java tick 48 d - 1 (its ticks of day d are 48 (d - 1) .. 48 d - 1), CENOP after tick 48 d.
Metrics and bands: validation/a2/metrics.py.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from validation.a2 import metrics as m

ROOT = Path(__file__).resolve().parents[2]
JAVA = ROOT / "output" / "a2" / "java"
CENOP = ROOT / "output" / "a2" / "cenop"
SCENARIOS = ["u-idw", "u-psm", "u-off", "d-ships"]
TITLES = {
    "u-idw": "Undisturbed, InnerDanishWaters, 5 years",
    "u-psm": "Undisturbed, PSM-Type2, 5 years",
    "u-off": "Undisturbed, dispersal off, 1 year",
    "d-ships": "Ships, InnerDanishWaters, 1 year",
}
ENERGY_DAYS = [30, 90, 150, 240, 359]
WINDOW = (30, 360)


def scenario_spec(name: str) -> dict:
    return json.loads((ROOT / "validation/a2/scenarios" / f"{name}.json").read_text())


def population_days(years: int) -> tuple[list[int], list[str]]:
    if years == 1:
        days = [90, 180, 270, 359]
        return days, [f"day {d}" for d in days]
    days = [360 * k for k in range(1, years + 1)]
    return days, [f"end of year {k}" for k in range(1, years + 1)]


def load_java_seed(d: Path, n_days: int) -> dict:
    pop = np.full(n_days + 1, np.nan)
    energy = np.full(n_days + 1, np.nan)
    with open(d / "Statistics.csv") as fh:
        rd = csv.reader(fh)
        next(rd)
        for _, tick, count, _food, total_energy in rd:
            t = int(float(tick))
            if (t + 1) % 48 == 0:
                day = (t + 1) // 48
                pop[day] = int(count)
                energy[day] = float(total_energy) / max(int(count), 1)
    per_block = np.zeros((n_days + 1, 60))
    with open(d / "PorpoisePerBlock.csv") as fh:
        rd = csv.reader(fh)
        next(rd)
        for _, tick, block, count in rd:
            t = int(float(tick))
            if (t + 1) % 48 == 0 and int(block) >= 1:
                per_block[(t + 1) // 48, int(block) - 1] = float(count)
    ages = []
    with open(d / "Reproduction.csv") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        i_age = head.index("AgeAtDeath")
        ages = [int(float(row[i_age])) for row in rd]
    return {"pop": pop, "energy": energy, "blocks": per_block, "ages": np.array(ages)}


def java_exposure(d: Path) -> float:
    """Share of a seed's tracked animal-ticks, while alive, with ship deterrence > 0.

    RandomPorpoise.csv keeps writing a dead animal's last row; an animal is taken as alive up
    to its last tick with a change in position or energy.
    """
    last_state: dict[str, tuple] = {}
    last_change: dict[str, int] = {}
    ship_ticks: dict[str, list[int]] = {}
    with open(d / "RandomPorpoise.csv") as fh:
        rd = csv.reader(fh)
        head = next(rd)
        i_t, i_id = head.index("tick"), head.index("Id")
        i_x, i_e, i_s = (
            head.index("UtmX"),
            head.index("EnergyLevel"),
            head.index("DeterShipStrength"),
        )
        for row in rd:
            t = int(float(row[i_t]))
            key = row[i_id]
            state = (row[i_x], row[i_e])
            if last_state.get(key) != state:
                last_state[key] = state
                last_change[key] = t
            if float(row[i_s]) > 0:
                ship_ticks.setdefault(key, []).append(t)
    alive = sum(last_change.values()) + len(last_change)  # ticks 0 .. last_change inclusive
    exposed = sum(sum(1 for t in ts if t <= last_change[k]) for k, ts in ship_ticks.items())
    return exposed / alive


def load_cenop_seed(f: Path) -> dict:
    a = np.loadtxt(f, delimiter=",", skiprows=1)
    n_days = int(a[-1, 0])
    pop = np.full(n_days + 1, np.nan)
    energy = np.full(n_days + 1, np.nan)
    per_block = np.zeros((n_days + 1, 60))
    days = a[:, 0].astype(int)
    pop[days], energy[days], per_block[days] = a[:, 1], a[:, 2], a[:, 5:65]
    deaths = np.loadtxt(f.with_name(f.stem + "-deaths.csv"), delimiter=",", skiprows=1, ndmin=2)
    return {
        "pop": pop,
        "energy": energy,
        "blocks": per_block,
        "ages": deaths[:, 1].astype(int) if deaths.size else np.array([], dtype=int),
        "exposure": float(a[:, 3].sum() / a[:, 4].sum()),
    }


def load(scenario: str) -> tuple[list[dict], list[dict], int]:
    years = int(scenario_spec(scenario)["overrides"]["simYears"])
    n_days = 360 * years
    java_dirs = sorted((JAVA / f"a2-{scenario}" / f"N2000-Y{years}").glob("seed-*"))
    java = []
    for d in java_dirs:
        run = load_java_seed(d, n_days)
        # the last day of the run ends at Java's last tick (48 n_days - 1): already covered
        if scenario == "d-ships":
            run["exposure"] = java_exposure(d)
        java.append(run)
    cenop = [
        load_cenop_seed(f)
        for f in sorted((CENOP / scenario).glob("seed-[0-9]*.csv"))
        if "deaths" not in f.name
    ]
    return java, cenop, years


def stack(runs: list[dict], key: str) -> np.ndarray:
    return np.array([r[key] for r in runs])


def fmt_rows(rows: list[m.Row], digits: int) -> list[str]:
    out = [
        "| Checkpoint | Java | CENOP | Difference | In SE | Verdict |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        rel = f" ({100 * r.diff / r.java:+.1f}%)" if digits == 1 and r.java else ""
        note = f" ({r.note})" if r.note else ""
        out.append(
            f"| {r.label} | {r.java:.{digits}f} | {r.cenop:.{digits}f} | {r.diff:+.{digits}f}{rel} | "
            f"{r.in_se:+.1f} | {r.verdict}{note} |"
        )
    return out


def build() -> tuple[str, dict]:
    lines: list[str] = []
    summary: dict[str, dict[str, str]] = {}
    data = {s: load(s) for s in SCENARIOS}
    for scenario in SCENARIOS:
        java, cenop, years = data[scenario]
        verdicts: dict[str, str] = {}
        lines += [
            f"## {TITLES[scenario]}",
            "",
            f"Java seeds: {len(java)}. CENOP seeds: {len(cenop)}.",
            "",
        ]
        if not java or not cenop:
            lines += ["No runs found for one side; scenario skipped.", ""]
            summary[scenario] = {}
            continue
        jp, cp = stack(java, "pop"), stack(cenop, "pop")
        je, ce = stack(java, "energy"), stack(cenop, "energy")

        days, labels = population_days(years)
        rows = m.population_rows(jp, cp, days, labels)
        verdicts["M1 population"] = m.overall([r.verdict for r in rows])
        lines += ["**M1 Population** (mean over seeds)", ""] + fmt_rows(rows, 1) + [""]

        inside, v = m.trajectory(jp, cp)
        verdicts["M2 trajectory"] = v
        lines += [
            f"**M2 Trajectory.** CENOP's mean is inside Java's mean +/- 2 SD on {100 * inside:.1f}% "
            f"of days (band: at least {100 * m.MIN_DAYS_INSIDE:.0f}%). {v}.",
            "",
        ]

        rows = m.energy_rows(je, ce, ENERGY_DAYS)
        verdicts["M3 mean energy"] = m.overall([r.verdict for r in rows])
        lines += ["**M3 Mean energy** (population-wide)", ""] + fmt_rows(rows, 2) + [""]

        if scenario != "u-off":
            js = np.array([m.block_shares(r["blocks"], *WINDOW) for r in java])
            cs = np.array([m.block_shares(r["blocks"], *WINDOW) for r in cenop])
            tvd, corr, v = m.location(js, cs)
            verdicts["M4 location"] = v
            lines += [
                f"**M4 Location.** Share of animals per 40 km block, days {WINDOW[0]}-{WINDOW[1]}: total "
                f"variation distance {tvd:.3f} (band: at most {m.MAX_TVD}), correlation {corr:.3f} "
                f"(band: at least {m.MIN_CORRELATION}). {v}.",
                "",
            ]

        ja = np.concatenate([r["ages"] for r in java])
        ca = np.concatenate([r["ages"] for r in cenop])
        stat, p, v = m.age_at_death(ja, ca)
        verdicts["M5 age at death"] = v
        lines += [
            f"**M5 Age at death.** {len(ja)} Java and {len(ca)} CENOP deaths, mean age "
            f"{ja.mean():.2f} and {ca.mean():.2f}; Kolmogorov-Smirnov statistic {stat:.4f}, "
            f"p = {p:.3g} (band: p above {m.MIN_KS_P}). {v}.",
            "",
        ]

        if scenario == "d-ships":
            uj, uc, _ = data["u-idw"]
            if uj and uc:
                row = m.ratio_row(
                    "N with ships / N undisturbed, day 359",
                    jp[:, 359],
                    stack(uj, "pop")[:, 359],
                    cp[:, 359],
                    stack(uc, "pop")[:, 359],
                )
                verdicts["M6 ship effect"] = row.verdict
                lines += ["**M6 Ship effect on population**", ""] + fmt_rows([row], 3) + [""]
            row = m.exposure_row(
                np.array([r["exposure"] for r in java]), np.array([r["exposure"] for r in cenop])
            )
            verdicts["M7 exposure"] = row.verdict
            lines += (
                [
                    "**M7 Exposure to ship noise** (Java: its 200 tracked animals per seed while alive; "
                    "CENOP: all animals)",
                    "",
                ]
                + fmt_rows([row], 4)
                + ["", f"Band: within {100 * m.MAX_REL_EXPOSURE:.0f}% of Java.", ""]
            )
        summary[scenario] = verdicts
    return "\n".join(lines), summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "docs/validation/A2-parity-report-tables.md"))
    args = parser.parse_args()
    body, summary = build()
    metrics = sorted({k for v in summary.values() for k in v})
    head = [
        "# A2 parity tables (generated)",
        "",
        "Generated by `python3 -m validation.a2.report`; do not edit by hand.",
        "",
    ]
    head += ["## Verdicts", "", "| Metric | " + " | ".join(TITLES[s] for s in SCENARIOS) + " |"]
    head += ["|---|" + "---|" * len(SCENARIOS)]
    for k in metrics:
        head.append(f"| {k} | " + " | ".join(summary[s].get(k, "n/a") for s in SCENARIOS) + " |")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(head) + "\n\n" + body + "\n")
    print("\n".join(head[4:]))


if __name__ == "__main__":
    main()
