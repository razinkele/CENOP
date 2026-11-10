# Track A1 — DEPONS-Java Headless Reference Harness — Spike Report

**Date:** 2026-11-10 · **Spec:** `2026-11-10-track-a1-depons-java-harness-design.md` ·
**Plan:** `docs/superpowers/plans/2026-11-10-track-a1-depons-java-harness.md` ·
**Harness:** `validation/depons_java/` (README there) · **Raw log:** `validation/depons_java/NOTES.md`

## Verdict

**The primary route worked.** The unmodified, official DEPONS 3.2 release runs headless on
Linux, scripted end to end, from one command. Inputs are checksum-pinned, seeds explicit,
and every run carries a provenance manifest. No fallback rung was needed. The time box
held: the first headless run succeeded in session 1. DEPONS's **simulated dynamics are
seed-deterministic**; two reporting outputs vary at floating-point/row-order level (§3).
The spike also surfaced CENOP-side A2 prerequisites (§7) and a large DEPONS-vs-CENOP
population difference that A2 must explain (§9).

## 1. Route taken (spec §6)

| Step | Outcome |
|---|---|
| Fetch | Release jar `DEPONS-3.2-setup.jar` (287,091,000 B) pinned trust-on-first-use, SHA-256 `bfbe73ea…68bea` (GitHub publishes no digest). Kattegat v3.2 landscape verified against the pin in `DownloadLandscapes.java` (`2614f56a…94913`). |
| Inspect | IzPack 5.1.3 installer. Payload in `resources/packs/pack-Model` (337 MB). The pack index lists the complete Repast Simphony **2.10.0** runtime incl. `RepastBatchMain` and the batch plugin; Groovy 3.0.17. |
| Install | **Blocker → solved.** IzPack 5.1.3 crashes on Java 21 in every mode: `ArrayIndexOutOfBoundsException` in PicoContainer's `BytecodeReadingParanamer`, a 2018-era class-file parser. Running the *installer only* under Java 11 works unattended: `-options` with `INSTALL_PATH`, exit 0, ~3 s. The installer just unpacks files. |
| Run | `repast.simphony.runtime.RepastBatchMain -params <sweep.xml> DEPONS.rs`, with the release launcher's exact classpath and JVM flags, cwd = model dir, system **OpenJDK 21.0.12**. **Worked on the first attempt.** |

Fallback rungs (a) xvfb, (b) source build and (c) published output: **not needed**.

## 2. Confirmed invocation

```
cwd: <install>/DEPONS        DISPLAY unset
java -Djava.awt.headless=true -Xmx3072M <15 JVM flags verbatim from start_model.command> \
  -cp "<rt>/bin:<rt>/lib/*:<install>/groovylib/groovy-3.0.17-indy.jar:<install>/DEPONS/lib/*" \
  repast.simphony.runtime.RepastBatchMain -params <abs>/batch_params.xml <install>/DEPONS/DEPONS.rs
```

`<rt>` = `<install>/repast.simphony/repast.simphony.runtime_2.10.0`. The flags are guarded
by a test against the installed launcher. Parameters go only through a generated
`batch_params.xml` sweep (all 70 pinned as constants). Each sink's
`*.batch_param_map.csv` echoes the effective values, confirming the sweep was applied.

## 3. Determinism

**The simulated dynamics are seed-deterministic. Two reporting outputs are not
byte-stable.**

- Smoke scale (N=200, 1 yr, seed 42): two runs gave byte-identical outputs for every file.
- Full scale (N=2000, 5 yr, seed 42): four independent runs, two before and two after
  the review fix pass, spanning the `C.UTF-8` locale pin.
  - **Byte-identical in all four:** `tick`/`PorpoiseCount`/`FoodEnergyLevel` at all
    86,399 ticks; the multiset of deaths (`Reproduction.csv` rows); `YearlyMortality.csv`;
    `PorpoisePerBlock.csv` (counts per block per tick); and `RandomPorpoise.csv` (one
    animal's full position + energy trajectory).
  - **`Statistics.csv:PorpoiseEnergyLevel`** differs in ~33 k rows by **≤ 6.3 × 10⁻¹⁶
    relative** (2–3 ULP). It is a Repast `SUM` aggregate iterated in a per-JVM agent
    order, so the summation order varies; three variants were seen in four runs.
    `FoodEnergyLevel` sums over a fixed array and never varies.
  - **`Reproduction.csv`:** deaths in the same tick are written in varying order.
- Meaning for A2: any DEPONS reference run can be regenerated from its manifest. Compare
  population/food exactly, `PorpoiseEnergyLevel` with ~1e-15 relative tolerance, and
  deaths as multisets. The harness's `compare` is deliberately byte-level (strict), so it
  flags these two files on a repeat. A dynamics-aware compare mode is a possible
  follow-up.

## 4. Outputs written by default (spec §3.5 / §9)

In batch mode **every file sink in `DEPONS.rs` writes by default**, into the model dir,
with a timestamp in the name:

| File | Columns | Frequency |
|---|---|---|
| `Statistics.csv` | run, tick, PorpoiseCount, FoodEnergyLevel, PorpoiseEnergyLevel | every tick (48/day; 17,280/yr) |
| `YearlyMortality.csv` | run, year, age, count, deaths | yearly, per age class |
| `Reproduction.csv` (Repast name "Dead Porpoises") | run, tick, TickAtDeath, AgeAtDeath, CalvesBorn, CalvesWeaned, LongDistMovType | one row per death (`Porpoise.die()`) |
| `PorpoisePerBlock.csv` | run, tick, Block, PorpoiseCount | per tick × block (large) |
| `RandomPorpoise.csv` | run, tick, Id, UtmX, UtmY, EnergyLevel, DeterTurbineStrength, DeterShipStrength, LoudestShipSPL, DispersalMode, PSMActive, PSMTargetUtmX/Y | per tick for `trackedPorpoiseCount` animals (default 1) |
| `Hydrophones.csv` | run, tick, name, source/received SL, ship… | header only when no hydrophones |
| `*.batch_param_map.csv` | run + all 70 parameters | once |

`ReproductiveLog*` and `PersistentSpatialMemory*`, named in the developer's example batch
config, are **not** default outputs. Column semantics, from the `DEPONS.rs` data set:
`PorpoiseEnergyLevel` and `FoodEnergyLevel` are **sums** (aggregation `SUM` over porpoises
or food cells), not means.

## 5. Variable-availability matrix for A2 (spec §3.5)

| A2 metric | DEPONS source | CENOP source | Mapping / gap |
|---|---|---|---|
| Daily population size | `Statistics.PorpoiseCount` (per tick) | `Population.txt: population` (daily, tick 48k) | subsample DEPONS at day boundaries; **tick alignment to verify in A2** |
| Mean porpoise energy | `Statistics.PorpoiseEnergyLevel / PorpoiseCount` (per tick) | `Energy.txt: mean_energy` (daily; also std/min/max, pct_hungry/starving) | direct after dividing the DEPONS sum by the count; DEPONS gives no spread by default |
| Food level | `Statistics.FoodEnergyLevel` (sum over food cells, per tick) | **none** in CENOP's default outputs | **CENOP gap**: add a landscape food-sum output |
| Annual mortality by age | `YearlyMortality` (year, age, count, deaths) | `Mortality.txt` (one row per death: age, cause) | aggregate CENOP deaths by year × age class; the `count` denominator needs CENOP's population age structure (`PorpoiseStatistics.txt`, gitignored) |
| Deaths by cause | **none** by default (`CauseOfDeath` exists internally but isn't written) | `Population.txt: deaths_*` / `Mortality.txt: cause` | **DEPONS gap**: compare totals only, or add a sink (would modify the DEPONS config, an anti-goal) |
| Births | indirect: `Reproduction.CalvesBorn` per dead female (lifetime) | `Population.txt: births` (daily) | **different granularity**: compare lifetime reproductive output of dead females, or rates derived from population change |
| Dispersal | `RandomPorpoise` DispersalMode/PSMActive/PSMTarget for `trackedPorpoiseCount` animals | `Dispersal.txt` (one row per dispersal event, all animals) | raise `trackedPorpoiseCount`, a supported parameter that changes no behaviour, to track more animals; output size grows ~linearly |

## 6. Cost and ensemble size (spec §3.6)

Measured (one process, this workstation, OpenJDK 21.0.12):

| run | wall | per simulated year |
|---|---|---|
| N=200, 1 yr (psm2 / idw smoke) | 31.7–32.8 s | ~32 s |
| N=2000, 5 yr (psm2, 3 runs) | 232–255 s | ~48 s average |

Cost scales with the **live** population. At N=2000 the first months run at ~0.98 s per
simulated day (≈ 6 min/yr). Runs get faster as the population falls; it reached ~50
here (§9). A scenario that keeps ~2000 animals alive costs up to **~30 min per 5-year
run**.

**Ensemble recommendation for A2:**
- **30 replicates per scenario per model.** For a two-sample KS test on per-run summary
  statistics this gives reasonable power; A2 must define its pass bands before running.
- DEPONS cost per scenario: ≈ 2 h at the observed speed, ≤ 15 h worst case.
- Parallelism: one install = one landing directory = serial runs, enforced by the lock.
  Run k independent caches (`CENOP_DEPONS_CACHE=…/cache-k`, each with its own install)
  to parallelise across cores.

## 7. Landscape identity and CENOP-side prerequisites for A2 (spec §3.7, §8)

- **CENOP's `data/Kattegat` is the DEPONS v3.0 landscape, not v3.2.** All 33 files are
  byte-identical to `depons.eu/files/landscapes/v3.0/Kattegat-51c36e01.zip` (SHA-256
  `51c36e01…19c84`). Against the v3.2 landscape (`2614f56a…`), only 5 files match.
  `disttocoast.asc`, `patches.asc`, all 12 monthly prey maps and all 12 salinity maps
  differ in data, not formatting. The v3.2 release itself bundles the v3.2 Kattegat zip,
  byte-identical to the depons.eu download.
- CENOP's landscape loader cannot read DEPONS 3.2's rolling-date names
  (`prey0000_MM_XX.asc`, `ships0000_XX_XX.json`).
- CENOP ignores `params.dispersal` and always runs PSM-Type2 (spec §8), so the IDW
  scenario is DEPONS-only for now.

**A2 prerequisites, in order:**
1. Decide the comparison landscape. Recommended: v3.2, canonical for the DEPONS 3.2 code
   CENOP is synced to. Then teach CENOP's loader the `_XX` names, or add a renaming step.
2. Re-run CENOP on that landscape. The committed Kattegat baselines (v3.0 landscape,
   PSM-Type2) are regression fixtures, not comparison data.
3. Wire `InnerDanishWaters` into CENOP before the TRACE-canonical half of A2.
4. Add a food-level output to CENOP.

## 8. Review and acceptance provenance

- A whole-branch code review returned "with fixes": 1 Critical (a mistyped constrained
  value, e.g. `dispersal`, silently ran a different model and was recorded `ok`) and 9
  Important.
  - **Provenance:** completion check by final tick; install marker binding the tree to
    the jar; hash of the landscape actually used; git state captured up front.
  - **Safety:** run lock; validate-before-delete ordering; atomic landscape placement;
    JVM killed on interrupt; locale-robust file names.
  - All were fixed test-first in `9b79cc7` (fast suite 895 passed). The ledger lists the
    deferred minors.
- Acceptance runs were redone on `9b79cc7` from a clean tree. Every manifest has
  `status: ok`, `harness_dirty: false`, `ticks_found == ticks_expected`, jar `bfbe73ea…`,
  landscape `2614f56a…`, locale `C.UTF-8`.

## 9. Observation for A2 (not judged here)

Under the like-for-like psm2 scenario on the v3.2 landscape, DEPONS's undisturbed Kattegat
population fell **2000 → 51 (seed 42) / 57 (seed 43)** within 5 years. In one-year smoke
runs at N=200 it fell to 107 (psm2) and 125 (IDW). CENOP's committed baseline stays
≈ 1999, but it uses the v3.0 landscape. The comparison is therefore confounded by:
1. landscape (v3.0 vs v3.2, §7);
2. PSM-Type2 being non-canonical for Kattegat (TRACE §3.2.4);
3. initial-condition differences.

Separating these is A2's first job; it may be the most consequential finding for the
paper.

## 10. Hand-off to A2

- Scenario files: `validation/depons_java/scenarios/kattegat-{psm2,idw}-undisturbed-v1.json`
  (overrides + seeds; A2 should generate the CENOP configuration from the same files).
- Reference runs: `output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5/seed-{42,43}/`
  (regenerable exactly from the manifests; not committed).
- Recommended scale and replicates: see §6.
- Not yet compared: no statistics were computed; that is A2.
