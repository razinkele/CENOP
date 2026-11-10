# Track A1 — DEPONS-Java Headless Reference Harness (feasibility spike) — Design

**Date:** 2026-11-10 · **Roadmap item:** A1 (`2026-06-12-cenop-roadmap-design.md` §5)
**Status:** design. Next step is an implementation plan (`writing-plans`).

## 1. Intent

CENOP is a Python port of DEPONS. Every parity claim so far rests on reading the Java
source and `parameters.xml`, not on running the Java model. The **audience is paper
reviewers**. They will ask whether the port reproduces DEPONS, and a convincing answer
needs DEPONS output that we actually generated: from the official release, with
documented provenance, in a way a reviewer can rerun.

A1 builds the reference side of that comparison: a **reproducible, headless harness that
runs the unmodified DEPONS 3.2 release** on Linux and captures its population and energy
output for a fixed scenario and a recorded set of seeds. A2 (a separate spec) does the
statistical comparison against CENOP.

A1 is a **time-boxed feasibility spike**. Either it ends with a working harness, or it
ends with a documented decision to use published DEPONS output instead (the roadmap's
fallback).

## 2. Facts this design builds on (verified 2026-11-10)

- **Upstream:** `jacobnabe/DEPONS` latest release is **v3.2** (published 2026-02-05). Tag
  `v3.2` = `master` = commit `8f533f8`, identical to the local `DEPONS-3.2/` clone. The
  contributor fork's `dev-3.2` branch has an identical tree. No newer release exists.
  `dev-4.0` (process-based energetics, alpha, last commit 2025-12-18) is unreleased and
  **out of scope**.
- **Release artifact:** `DEPONS-3.2-setup.jar` (287,091,000 bytes), an **IzPack 5**
  installer. Its config (`installer/installation_components.xml`) **indicates** it bundles
  the Repast Simphony runtime plugins, including `repast.simphony.batch`, the Groovy jar,
  and the model. Still unconfirmed against the jar itself, and the jar does **not**
  include a Java runtime. GitHub publishes **no checksum** for the asset.
- **Launchers:** only `start_model.command` (GUI; `repast.simphony.runtime.RepastMain`,
  `-Xmx3072M`, plus a set of `--add-exports/--add-opens` flags) and `.bat`. No headless
  Linux launcher exists, so **headless operation is untested**.
- **Batch mode:** the repo has `batch/batch_params.xml` (a Repast parameter sweep) and
  `batch_configuration.properties`. The latter is a developer's file with hard-coded
  `/home/jonas/...` paths; treat it as an example only.
- **Landscape:** `test/.../util/DownloadLandscapes.java` pins
  `https://depons.eu/files/landscapes/v3.2/Kattegat-2614f56a.zip`, 38,927,699 bytes,
  SHA-256 `2614f56a771bf9c486a3186779040b2426b6027cfa4bb50aa61f9cd60ff94913`.
- **Java:** OpenJDK 21.0.12 is installed. `ant` is not.
- **DEPONS outputs:**
  - `Statistics.csv` (Repast file sink; every tick: `tick, PorpoiseCount, FoodEnergyLevel,
    PorpoiseEnergyLevel`; timestamped filename).
  - `YearlyMortality.<timestamp>.csv` (`run, year, age, count, deaths`).
  - Other file sinks defined in `DEPONS.rs`: `RandomPorpoise.csv` (tracked tracks),
    `PorpoisePerBlock.csv`, `Reproduction.csv`, `Hydrophones.csv`.
  - **Unconfirmed:** `ReproductiveLog*` and `PersistentSpatialMemory*` appear only in the
    developer's example batch config. Whether they, and the sinks above, are written by
    default, need enabling, or are debug-only is not yet established.
- **Seeding:** `parameters.xml` `randomSeed` defaults to `__NULL__` (randomised per run),
  so the harness must set seeds explicitly.
- **Scenario guidance from the DEPONS 3.2 TRACE** (`DEPONS 3.2 TRACE-2026-02-05.pdf`):
  - **"In the Kattegat landscape the large-scale movements must be simulated using a
    behaviour developed specifically for this area (InnerDanishWaters movements)"**
    (§3.2.4 "Parameters related to large-scale movements", p. 44; the behaviour itself is
    described in §2.7.2.3).
  - The Kattegat runs in §3.2.3 ("Spatial memory component of fine-scale movements",
    pp. 42–43) used **200 animals, 11 years, the first 10 as burn-in, and included
    ships**. They are a **movement calibration** (home range, net squared displacement,
    residence time against 17 tagged animals), **not** an undisturbed
    population-dynamics scenario.
  - `parameters.xml` defaults are `landscape=NorthSea`, `porpoiseCount=10000`,
    `dispersal=PSM-Type2`, `simYears=50`, ships/turbines off.
- **CENOP side (verified 2026-11-10):** CENOP's simulation **ignores the `dispersal`
  parameter**.
  - Nothing in `agents/population.py` or `core/simulation.py` reads `params.dispersal`.
    `_apply_dispersal_heading` always applies the PSM-Type2 formula.
  - `behavior/dispersal.py` defines an `InnerDanishWatersDispersal` class, but nothing in
    the simulation calls it and no test covers it. The UI selector (`off` / `PSM-Type2` /
    `Undirected` / `InnerDanishWaters`) therefore has no effect.
  - CENOP's committed Kattegat baselines (`scripts/run_kattegat_reference.py`; N=2000,
    5 years, seed 42) are PSM-Type2 runs.
  - Consequence: **CENOP currently cannot run the TRACE-mandated Kattegat
    configuration** (see §8).

## 3. Goals

1. **Run DEPONS 3.2 headless on Linux from the official release binary**, with no GUI and
   no display. Fallbacks are in §6.
2. **One scripted, idempotent entry point** that fetches and verifies the artifacts,
   installs, runs a scenario × seed list, and collects the outputs into a stable layout.
3. **Provenance a reviewer can audit.** Every run writes a manifest covering: DEPONS
   version and commit, release-jar SHA-256, landscape SHA-256, Java version, Repast
   version, the full effective parameter set, seed, host, wall-clock time, and the harness
   git commit.
4. **A determinism check:** the same scenario and seed run twice gives byte-identical
   `Statistics.csv` content (timestamps ignored). If it does not, document why and what
   that means for A2.
5. **A variable-availability matrix** for A2. For each metric A2 will compare (daily
   population size, mean porpoise energy, food level, annual mortality by age, births,
   dispersal), record whether DEPONS emits it, in which file and column, at what
   frequency, and how it maps to CENOP's output (`Population.txt`, `Energy.txt`,
   `Mortality.txt`, `Dispersal.txt`). Record any optional sink that has to be enabled to
   get a metric.
6. **A cost measurement** (wall-clock time per simulated year at the chosen N), so A2 can
   size its replicate ensemble realistically.
7. **Input identity check.** Hash-compare every file of the unpacked, checksum-verified
   v3.2 Kattegat landscape against CENOP's `data/Kattegat/`, whose provenance is
   undocumented. Report identical, differing, and missing files. Comparing like with like
   (the scenario-drift anti-goal) depends on both models reading the same landscape.

## 4. Non-goals

- The statistical comparison, pass bands, and fidelity verdicts (all A2).
- Any change to CENOP model code (`src/cenop/`). A1 adds harness scripts and docs only.
  In particular, wiring `InnerDanishWaters` (or the `dispersal` parameter at all) into
  CENOP's simulation is a separate work item (§8).
- Disturbed scenarios (ships, turbines). A2 adds one disturbed scenario once the
  undisturbed one works.
- North Sea or any other landscape.
- DEPONS 4.0 / `dev-4.0`.
- Running the harness in GitHub CI. It is too heavy (287 MB jar, multi-year Java runs).
  CI integration is a possible later item, not A1.

## 5. Anti-goals (would make A1 a failure even if it technically runs)

- **Modifying DEPONS code or configuration to make its output easier to compare.** The
  reference must be the released model. Scenario parameters are set only through
  Repast's supported parameter mechanism (`batch_params.xml` / `parameters.xml` values),
  never by patching Java.
- **Building DEPONS from source when the release binary can be made to work.** A reviewer
  should be able to say "this is the published v3.2". A source build is a fallback (§6)
  and must be labelled as such, with commit `8f533f8` recorded.
- **Unpinned or unverified inputs**: a download without a checksum check, or a seed that
  isn't recorded.
- **Scenario drift between the two models.** The DEPONS scenario file is the single
  source of truth, and A2 must be able to configure CENOP identically from it.
- Vendoring the 287 MB jar or the landscape zip into git.

## 6. Approach (decided) and fallback ladder

**Primary — official release, headless Repast batch:**

1. **Fetch** `DEPONS-3.2-setup.jar` from the GitHub release into its own empty cache
   directory, outside the repo and gitignored. Before running anything, `unzip -l` it and
   confirm the Repast runtime and `repast.simphony.batch` are inside. Compute its SHA-256.
   No upstream digest exists, so pin it **trust-on-first-use**: record it in the harness
   config and verify it on every later fetch.
2. **Install unattended** (IzPack 5 headless or auto-install mode) into the cache
   directory. If unattended install misbehaves, extracting the installer's packs directly
   is an acceptable equivalent, as long as the resulting tree matches what the installer
   would produce.
3. **Fetch the Kattegat landscape** zip, verify the SHA-256 pinned in
   `DownloadLandscapes.java`, and unpack it where DEPONS expects landscapes.
4. **Run headless** through Repast Simphony's batch entry point, on the classpath and JVM
   flags taken from `start_model.command`. The exact main class (expected
   `repast.simphony.batch.BatchMain` or equivalent) and arguments must be confirmed from
   the bundled Repast version. Use a generated `batch_params.xml` holding the scenario
   parameters and the seed list.
5. **Collect** each run's outputs into
   `<output-root>/<scenario-id>/seed-<seed>/` with a `manifest.json`. Normalise
   timestamped filenames to stable names and keep the originals' names in the manifest.

**Fallback ladder** (take the next rung only when the current one is blocked):

- (a) **GUI launcher under a virtual display** (`xvfb-run` with `start_model.command`),
  driving Repast's batch/run options. This is still the official binary.
- (b) **Source build** of commit `8f533f8` against a Repast Simphony distribution of the
  matching version. This is the official source, built by us, and labelled as such in
  every manifest.
- (c) **Published reference output**: a documented decision record to compare against
  results published in the TRACE documents or DEPONS papers, stating what that loses
  (no seed control, fewer metrics, no ensemble).

## 7. Scenario definitions (my call; check this)

Both scenarios start from **DEPONS 3.2 `parameters.xml` defaults** and override only
`landscape=Kattegat`, `ships=false`, `turbines=off`, `dispersal` (below), `porpoiseCount`,
`simYears`, and `randomSeed` (per replicate).

1. **`kattegat-psm2-undisturbed-v1` (like-for-like, primary for A1).**
   `dispersal=PSM-Type2`, the only large-scale movement CENOP can currently run. This is
   the scenario A2 can compare **today**. It is a valid comparison of the two codebases on
   identical inputs, but **not** the TRACE-recommended Kattegat configuration, and any
   paper text must say so.
2. **`kattegat-idw-undisturbed-v1` (TRACE-canonical, target).**
   `dispersal=InnerDanishWaters`, per TRACE §3.2.4. A1 produces DEPONS output for it so
   the reference is ready, but A2 can only use it once CENOP wires `InnerDanishWaters`
   into its simulation (§8).

- **Scale:** **N and years are harness parameters**, not hard-coded. The acceptance runs
  use a short smoke configuration (≤ 1 simulated year) plus one full-length run.
  - Default full-length scale: N=2000 for 5 years, matching CENOP's existing baseline
    scale.
  - Note: DEPONS publishes **no canonical undisturbed Kattegat population scenario**. The
    TRACE's Kattegat N=200 / 11-year runs were a movement calibration with ships (§2).
  - A2 fixes the final scale and burn-in using A1's cost measurement and the statistical
    power each option gives. Small-N runs are noisy, so they need more replicates.
- **Seeds:** an explicit list, recorded in the scenario file. A1 needs at least 2 seeds,
  plus one repeated seed for the determinism check.

Scenario files must be machine-readable (YAML/JSON) so A2 can generate the matching
CENOP configuration from the same file.

## 8. Recorded side-finding: CENOP's dispersal parameter has no effect (not fixed in A1)

Found while writing this spec, and recorded per the roadmap's "no silent scope creep"
rule. CENOP's simulation **hard-codes PSM-Type2 large-scale movement** and never reads
`params.dispersal`. This has two consequences:

1. **Model gap.** CENOP cannot run the TRACE-mandated Kattegat configuration
   (`InnerDanishWaters`), nor `Undirected` or `off`. The `InnerDanishWatersDispersal`
   class in `behavior/dispersal.py` is untested and not called by the simulation.
2. **Silent UI no-op.** The web app's dispersal selector accepts four choices, but the
   simulation runs PSM-Type2 whatever the user picks.

CENOP's committed Kattegat baselines are therefore PSM-Type2 runs. They are still valid as
regression fixtures, but they are not a TRACE-canonical scenario. A2 must produce fresh
CENOP runs from §7's scenario files rather than reuse them.

Fixing this means wiring the dispersal type into the vectorized step, with DEPONS-parity
tests for `InnerDanishWaters`. That is model work, and it needs its own spec. It is a
**prerequisite for the TRACE-canonical half of A2**, but not for A1.

## 9. Deliverables

- Harness scripts in CENOP under a dedicated directory (e.g. `validation/depons_java/`).
  Python, matching the repo's scripts; thin shell wrappers are fine. The harness includes
  the pinned artifact config (URLs, sizes, SHA-256s) and the scenario file(s).
- A reviewer-facing `README` in that directory: prerequisites (Java 21, disk space,
  network), a one-command rerun, the expected outputs, and how to check the provenance.
- `.gitignore` entries for the artifact cache and the run outputs.
- A short **spike report**
  (`docs/superpowers/specs/YYYY-MM-DD-track-a1-spike-report.md`) covering:
  - which rung of §6 worked, and the blockers met on the way
  - the confirmed Repast version and batch invocation
  - the determinism result
  - the variable-availability matrix (§3.5)
  - the cost measurement and an A2 ensemble-size recommendation
  - the landscape identity result (§3.7)
  - which output sinks are written by default and which had to be enabled
- If the fallback is taken (rung c): the decision record in place of the harness.

## 10. Acceptance

A1 is done when **either**:

- running the harness entry point on a clean machine with Java 21 fetches and verifies
  the artifacts, then:
  - completes `kattegat-psm2-undisturbed-v1` for ≥ 2 seeds plus one repeat, and
  - completes at least a smoke run of `kattegat-idw-undisturbed-v1`,
  - writing outputs and manifests for both, with the spike report (including the
    landscape identity result) committed; **or**
- the spike report documents why rungs (a) and (b) failed and commits the decision to use
  published output (rung c).

## 11. Time box

**Two working sessions.** If the primary approach has not produced one headless run by
the end of session 1, move down the fallback ladder instead of debugging Repast
internals open-endedly.

## 12. Left to the builder

- Exact IzPack invocation, Repast batch main class and arguments, classpath assembly.
- The directory layout inside the harness, script structure, and CLI flag names.
- The manifest schema (it must cover every field in §3.3).
- How to normalise timestamped output names.
- Cache location (default: a gitignored path inside the repo, or `~/.cache/cenop/`),
  overridable by an environment variable.
