# DEPONS 3.2 reference harness (Track A1)

This directory holds a small, stdlib-only Python harness. It runs the **official,
unmodified DEPONS 3.2 release** (Java / Repast Simphony 2.10.0) headless on Linux and
records full provenance for every run. It produces the DEPONS reference output that CENOP
is compared against (roadmap Track A2).

## Prerequisites

- Linux with Python ≥ 3.11 (stdlib only; CENOP's `shiny` env works).
- **Java 21** to run the model (any recent OpenJDK; recorded in each manifest).
- **Java 11** (or 8) to run the DEPONS *installer* only. The release's IzPack 5.1.3
  installer crashes on Java 21. It only unpacks files, so its JRE cannot affect the model.
  The harness finds `/usr/lib/jvm/java-11-openjdk*/bin/java`, or set
  `CENOP_DEPONS_INSTALLER_JAVA=/path/to/java11`.
- ~1 GB free disk (287 MB jar, 39 MB landscape, 341 MB install, run outputs).
- Network access for the first run (github.com, depons.eu).

## Reproduce the reference runs (one command)

From the CENOP repository root:

```bash
python3 -m validation.depons_java run \
    --scenario validation/depons_java/scenarios/kattegat-psm2-undisturbed-v1.json
```

On first use this:
1. downloads the release jar and the Kattegat landscape, verifying size and SHA-256
   against `artifacts.json`;
2. installs the release unattended into the cache;
3. runs every seed in the scenario file.

Later runs reuse the verified cache. Other subcommands:

| command | purpose |
|---|---|
| `fetch` | download + verify artifacts only |
| `install` | fetch + unattended install + place the landscape |
| `run --scenario F [--seeds …] [--porpoise-count N] [--sim-years Y] [--overwrite]` | run a scenario (fetch/install as needed) |
| `compare RUN_A RUN_B` | byte-compare two completed run directories; exit 0 identical, 1 differ, 2 not comparable (missing `Statistics.csv` or status ≠ ok) |
| `compare-landscape [--candidate DIR]` | hash-compare the verified DEPONS landscape with a directory |

The cache defaults to `.depons-cache/` in the repo (gitignored); override it with
`CENOP_DEPONS_CACHE`. Outputs default to `output/depons_java/` (gitignored).

## Scenarios

| file | large-scale movement | status |
|---|---|---|
| `kattegat-psm2-undisturbed-v1.json` | PSM-Type2 | like-for-like with CENOP today |
| `kattegat-idw-undisturbed-v1.json` | InnerDanishWaters (TRACE 3.2 §3.2.4) | TRACE-canonical; CENOP cannot run it yet |

Both start from the release's `parameters.xml` defaults and override only `landscape`,
`dispersal`, `ships`, `turbines`, `porpoiseCount`, `simYears` and `randomSeed`. Parameters
are passed **only** through a generated Repast `batch_params.xml` sweep (every parameter
pinned as a constant). No DEPONS code or configuration is modified.

## Output layout and run isolation

```
output/depons_java/<scenario-id>/N<count>-Y<years>/seed-<seed>/
    batch_params.xml   # the exact sweep given to Repast
    java.log           # DEPONS/Repast stdout+stderr
    manifest.json      # provenance (below)
    Statistics.csv  YearlyMortality.csv  Reproduction.csv  PorpoisePerBlock.csv
    RandomPorpoise.csv  Hydrophones.csv  *.batch_param_map.csv  debug.log
```

DEPONS writes timestamped files into its model directory. The harness:
- refuses to start a run while that directory holds outputs from an earlier run;
- moves each run's files out under stable names, so every run directory holds exactly
  what that run produced;
- holds an exclusive lock (`<cache>/harness.lock`), so a second concurrent `run` against
  the same install is refused;
- before deleting or launching anything, checks every seed directory (`--overwrite` is
  needed to replace existing ones), rejects duplicate seeds, and validates every scenario
  value against the release's `parameters.xml`, including the allowed-value lists for
  `landscape`, `dispersal` and `turbines`.

If the harness is interrupted, the JVM is killed. The child process runs with
`LC_ALL=LANG=C.UTF-8`, so output names and formatting are identical across machines.

## Checking provenance

Each `manifest.json` records:
- `scenario_id`, `seed`, and the full effective `parameters` (all 70)
- `depons_version` (3.2) and `depons_commit` (`8f533f8…`)
- `jar_sha256`: read from the install's `.cenop-install.json` marker and verified against
  the pin, so it binds the run to the jar the install tree came from
- `landscape_sha256`: the hash of the landscape zip actually loaded
- `java_version`, `repast_version` and `locale`
- `harness_commit` and `harness_dirty`, captured before the first run (`null` if git is
  unavailable); reference runs must show `false`
- `host`, `started_utc`, `wall_seconds`
- `returncode`, `timed_out`, `ticks_expected`, `ticks_found` and `status`. `status` is
  `ok` only when Java exited 0, did not time out, and `Statistics.csv` reached the final
  tick (`simYears × 17,280 − 1`).
- `output_files` (normalised name → original timestamped name)
- the exact java `command`

To verify the inputs yourself:

```bash
sha256sum .depons-cache/DEPONS-3.2-setup.jar .depons-cache/Kattegat-2614f56a.zip
```

The two hashes must match `artifacts.json`. The jar hash was pinned trust-on-first-use,
because GitHub publishes no digest for the release asset.

## Determinism

Same seed + same release + same JVM → the **same simulated trajectory**. Across four
independent seed-42 runs these were byte-identical:
- population and food at every tick;
- the multiset of deaths;
- `YearlyMortality.csv`, `PorpoisePerBlock.csv` and `RandomPorpoise.csv`.

Two *reporting* outputs are not byte-stable, because DEPONS's data collector iterates
agents in an order that is not fixed across JVM runs:
- `Statistics.csv:PorpoiseEnergyLevel` is a floating-point sum and differs at ≤ 6.3e-16
  relative (2–3 ULP);
- `Reproduction.csv` lists deaths that share a tick in varying order.

`compare` is deliberately byte-level, so a repeat run reports those two files as
differing. Compare them as described in the spike report: the energy sum with a ~1e-15
relative tolerance, deaths as a multiset.

See `docs/superpowers/specs/2026-11-10-track-a1-spike-report.md` for the findings and the
A2 hand-off.
