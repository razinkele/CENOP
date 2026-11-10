# Track A1 — DEPONS-Java Headless Reference Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A scripted, reproducible harness that fetches and verifies the official DEPONS 3.2
release, runs it headless on Linux for two Kattegat scenarios × explicit seeds, and writes
normalised outputs plus a provenance manifest per run. It ends with a committed spike
report.

**Architecture:**
- A stdlib-only Python package `validation/depons_java/` with one module per
  responsibility: artifacts, scenario, install, runner, collect, landscape, cli.
- Pure logic (checksums, parameter merging, sweep-XML rendering, name normalisation,
  manifests, directory comparison) is unit-tested offline in the fast suite.
- Everything that touches Java, the network or the 287 MB jar is a **discovery step** run
  by hand through the CLI. Its findings go into a running log, `validation/depons_java/NOTES.md`,
  which becomes the spike report.

**Tech Stack:** Python 3.13 stdlib (`hashlib`, `urllib.request`, `zipfile`,
`xml.etree.ElementTree`, `json`, `subprocess`, `argparse`, `platform`), OpenJDK 21,
Repast Simphony (bundled in the release), IzPack 5 installer, pytest.

**Spec:** `docs/superpowers/specs/2026-11-10-track-a1-depons-java-harness-design.md`

## Global Constraints

- Run the **unmodified DEPONS 3.2 release**. Set parameters only through Repast's
  parameter mechanism (generated `batch_params.xml`); never patch Java or DEPONS config.
- **No changes under `src/cenop/`.**
- Never commit the jar, the landscape zip, the install tree or run outputs. Cache:
  `$CENOP_DEPONS_CACHE`, default `<repo>/.depons-cache/` (gitignored). Outputs: default
  `output/depons_java/` (gitignored).
- Pinned artifacts (verbatim):
  - Release jar `https://github.com/jacobnabe/DEPONS/releases/download/v3.2/DEPONS-3.2-setup.jar`,
    size `287091000`, SHA-256 pinned **trust-on-first-use** in Task 2 (no upstream digest).
  - Landscape `https://depons.eu/files/landscapes/v3.2/Kattegat-2614f56a.zip`, size `38927699`,
    SHA-256 `2614f56a771bf9c486a3186779040b2426b6027cfa4bb50aa61f9cd60ff94913`.
- DEPONS source commit for manifests: `8f533f8ef59f3652056b4f23c6abecbe6cd80acb`.
- Every download is verified (size + SHA-256) before use. Every seed is explicit and
  recorded.
- JVM flags are copied verbatim from the release's `start_model.command`
  (`-XX:+IgnoreUnrecognizedVMOptions --add-modules=ALL-SYSTEM` plus its `--add-exports` and
  `--add-opens` list); `-Xmx3072M` by default.
- Unit tests: offline, no Java, no network, under 2 s total, in `tests/` (so they run in CI).
- Stdlib only — **no new dependencies**.
- Run all commands from `/home/razinka/cenjas/CENOP`. Commit inside the CENOP repo on branch
  `track-a1-depons-harness`. **Do not push.**
- Before each commit: `micromamba run -n shiny black --check --line-length 100 validation tests`
  and `micromamba run -n shiny ruff check validation tests` are clean. The fast suite stays
  green: `micromamba run -n shiny python3 -m pytest tests/ -q`.
- **Time box: two working sessions.** If Task 5 has not produced one headless DEPONS run by
  the end of session 1, stop debugging and execute **Task F** (fallback ladder).
- **Run isolation:** DEPONS/Repast write timestamped files into one landing directory
  (found in Task 5). `run` refuses to start a seed while that directory holds timestamped
  outputs, and collection **moves** files out of it. Each run therefore owns exactly the
  files it produced.
- **Clean-tree provenance:** acceptance runs (Task 8) start from a committed, clean tree,
  so every manifest has `harness_dirty: false`. Write NOTES and the report *after* the
  runs.

## Review Focus

1. **A partial or corrupt cached download** (interrupted fetch, wrong size) must be
   re-downloaded, never used. A temp file must never end up under the final name.
   → test in Task 1.
2. **A typo or invalid value in a scenario override** (`dispersl`, `porpoiseCount="2k"`,
   `ships="yes"`) must fail before Java starts, with a message naming the key.
   → test in Task 3.
3. **A Java run that crashes or times out but leaves partial files** must be recorded as
   `status: "failed"` with a non-zero CLI exit, never collected as a success.
   → tests in Tasks 5 and 6.
4. **Re-running the same seed into an existing output directory** must not mix old and new
   files. → test in Task 6.
5. **Several timestamped files normalising to the same name** (left over from an earlier
   run in Repast's working directory) must be an error, not a silent pick.
   → test in Task 6.

---

### Task 1: Package scaffold, artifact registry, verified fetch, `fetch` CLI

**Files:**
- Create: `validation/__init__.py` (empty), `validation/depons_java/__init__.py` (empty)
- Create: `validation/depons_java/artifacts.py`, `validation/depons_java/artifacts.json`
- Create: `validation/depons_java/cli.py`, `validation/depons_java/__main__.py`
- Create: `validation/depons_java/NOTES.md` (heading `# A1 spike log`, then empty)
- Modify: `.gitignore` (append `.depons-cache/` and `output/depons_java/`)
- Test: `tests/test_depons_java_artifacts.py`

**Interfaces:**
- Produces:
  - `class ArtifactError(Exception)`
  - `@dataclass(frozen=True) class Artifact: name: str; url: str; size: int; sha256: str | None; filename: str`
  - `REGISTRY_PATH: Path` (= `artifacts.json` beside the module)
  - `load_registry(path: Path = REGISTRY_PATH) -> dict[str, Artifact]`. Keys:
    `"depons_jar"`, `"kattegat_landscape"`.
  - `default_cache_dir() -> Path` (`$CENOP_DEPONS_CACHE`, else repo root `/.depons-cache`)
  - `sha256_file(path: Path) -> str` (lowercase hex)
  - `verify_artifact(path: Path, artifact: Artifact, *, allow_unpinned: bool = False) -> str`
    returns the SHA-256. Raises `ArtifactError` on a size mismatch, a SHA mismatch, or an
    unpinned artifact (`sha256 is None`) unless `allow_unpinned`.
  - `fetch_artifact(artifact: Artifact, cache_dir: Path, *, allow_unpinned: bool = False) -> Path`
    returns `cache_dir / artifact.filename`.
  - CLI: `python3 -m validation.depons_java fetch [--allow-unpinned] [--cache DIR]` prints
    `<name> <path> <sha256>` per artifact. Exits 1 on `ArtifactError`.

- [ ] **Step 1: Write the failing tests** in `tests/test_depons_java_artifacts.py`. Use
  `tmp_path`. Serve source files via `file://` URLs (`Path.as_uri()`); no network.
  - `test_sha256_file_known_value`: for `b"abc"` assert
    `"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"`.
  - `test_verify_rejects_wrong_size` / `test_verify_rejects_wrong_sha`: `pytest.raises(ArtifactError)`.
  - `test_verify_unpinned_requires_flag`: raises without `allow_unpinned`; with it, returns
    the hash.
  - `test_fetch_downloads_and_verifies`: the returned file exists, verifies, and no `*.part`
    file remains in the cache.
  - `test_fetch_skips_when_cached_valid`: a second fetch leaves `st_mtime_ns` unchanged.
  - `test_fetch_replaces_corrupt_cached_file` (Review Focus 1): pre-write wrong bytes at the
    final name. Fetch, then assert the content equals the source.
  - `test_fetch_size_mismatch_leaves_nothing`: the artifact declares the wrong size. Fetch
    raises `ArtifactError`; neither the final file nor `*.part` exists afterwards.
  - `test_registry_pins_landscape`: `load_registry()["kattegat_landscape"]` has the
    spec's URL, size `38927699` and SHA-256; `["depons_jar"].size == 287091000`.

- [ ] **Step 2: Run** `micromamba run -n shiny python3 -m pytest tests/test_depons_java_artifacts.py -q`.
  Expected: FAIL (import error).

- [ ] **Step 3: Implement** `artifacts.py`, `artifacts.json` (values from Global Constraints;
  jar `"sha256": null`), `cli.py` with an `argparse` subparser `fetch`, and `__main__.py`
  calling `cli.main()`.
  - Download to `<filename>.part`, verify, then `os.replace` onto the final name.
  - Delete the `.part` on any failure.
  - A cached file that fails verification is deleted and re-fetched.

- [ ] **Step 4: Run** the tests again. Expected: PASS. Then the fast suite and lint per
  Global Constraints.

- [ ] **Step 5: Commit** `feat(a1): artifact registry with verified fetch and fetch CLI`.

---

### Task 2: Fetch the real artifacts, pin the jar hash, inspect the jar (discovery)

**Files:**
- Modify: `validation/depons_java/artifacts.json` (pin the jar SHA-256)
- Modify: `validation/depons_java/NOTES.md`

**Interfaces:**
- Consumes: the Task 1 CLI.
- Produces: a pinned `depons_jar.sha256`; NOTES section `## Release jar`.

- [ ] **Step 1: Fetch.** Run `micromamba run -n shiny python3 -m validation.depons_java fetch --allow-unpinned`.
  Expected: the landscape verifies against its pinned hash, and the jar prints its SHA-256.
- [ ] **Step 2: Pin.** Write the printed jar hash into `artifacts.json`. Re-run `fetch`
  **without** `--allow-unpinned`. Expected: both verify; no download (cached).
- [ ] **Step 3: Inspect.** Run `unzip -l .depons-cache/DEPONS-3.2-setup.jar | grep -i -E "repast|pack|groovy" | head -50`.
  Record in NOTES whether the Repast runtime and the `repast.simphony.batch` plugin are
  visible as entries, or packed inside IzPack pack blobs, and record the Repast version
  string if visible.
- [ ] **Step 4: Gate.** If the listing shows the jar cannot contain a Repast runtime (no
  pack blobs and no repast entries), note it and go to Task F rung (b). Otherwise continue.
- [ ] **Step 5: Commit** `chore(a1): pin DEPONS 3.2 release jar sha256 (TOFU) + jar notes`.

---

### Task 3: Scenario files, parameter merge, sweep-XML rendering

**Files:**
- Create: `validation/depons_java/scenario.py`
- Create: `validation/depons_java/scenarios/kattegat-psm2-undisturbed-v1.json`
- Create: `validation/depons_java/scenarios/kattegat-idw-undisturbed-v1.json`
- Test: `tests/test_depons_java_scenario.py`

**Interfaces:**
- Produces:
  - `class ScenarioError(Exception)`
  - `@dataclass(frozen=True) class ParamSpec: name: str; type: str; default: str`
    (`type` ∈ `{"int", "boolean", "string"}`)
  - `@dataclass(frozen=True) class Scenario: id: str; description: str; overrides: dict[str, str]; seeds: tuple[int, ...]; determinism_seed: int`
  - `SCENARIO_DIR: Path`
  - `load_scenario(path: Path) -> Scenario`
  - `parse_parameters_xml(path: Path) -> dict[str, ParamSpec]` (insertion order = file order)
  - `effective_parameters(specs: dict[str, ParamSpec], overrides: dict[str, str], seed: int) -> dict[str, str]`.
    Every spec name maps to a string; overrides are applied; `randomSeed` is set to
    `str(seed)`.
  - `render_batch_params(specs: dict[str, ParamSpec], params: dict[str, str]) -> str`
- Scenario JSON (exact). The psm2 file:
  `{"id": "kattegat-psm2-undisturbed-v1", "description": "...", "overrides": {"landscape": "Kattegat", "dispersal": "PSM-Type2", "ships": "false", "turbines": "off", "porpoiseCount": "2000", "simYears": "5"}, "seeds": [42, 43], "determinism_seed": 42}`.
  The idw file is identical except `id` = `kattegat-idw-undisturbed-v1`, `"dispersal": "InnerDanishWaters"`,
  and a description citing TRACE §3.2.4.

- [ ] **Step 1: Write the failing tests.** Use an inline 4-parameter `parameters.xml`
  written to `tmp_path`, in the release's format:
  `randomSeed` int `__NULL__`, `porpoiseCount` int `10000`, `ships` boolean `false`,
  `PSM_dist` string `N(300;100)`.
  - `test_parse_parameters_xml_types_defaults_order`
  - `test_effective_parameters_merges_and_seeds`: overriding `porpoiseCount="200"` with seed
    7 gives `{"randomSeed": "7", "porpoiseCount": "200", "ships": "false", "PSM_dist": "N(300;100)"}`.
  - `test_unknown_override_rejected` (Review Focus 2): `{"dispersl": "x"}` raises
    `ScenarioError`, and the message contains `dispersl`.
  - `test_bad_int_rejected` (`porpoiseCount="2k"`) and `test_bad_boolean_rejected`
    (`ships="yes"`; only `true`/`false` are accepted).
  - `test_render_batch_params_roundtrip`: parse the output with `ElementTree`. The root tag
    is `sweep` with `runs="1"`. One `parameter` per spec, in order, each with
    `type="constant"`. `constant_type` maps int→`int`, boolean→`boolean`,
    string→`java.lang.String`. A value containing `&` round-trips intact.
  - `test_load_scenario_validation`: raises for a missing `seeds`, empty seeds, duplicate
    seeds, and a `determinism_seed` not in `seeds`.
  - `test_shipped_scenarios_differ_only_in_dispersal`: load both shipped files and assert
    the overrides are equal except `dispersal` (`PSM-Type2` vs `InnerDanishWaters`), and
    the seeds are equal.

- [ ] **Step 2: Run** `... pytest tests/test_depons_java_scenario.py -q`. Expected: FAIL.
- [ ] **Step 3: Implement** `scenario.py` and the two JSON files.
- [ ] **Step 4: Run** the tests, the fast suite and lint. Expected: PASS.
- [ ] **Step 5: Commit** `feat(a1): scenario files, parameter merge and Repast sweep rendering`.

---

### Task 4: Unattended install, install layout, landscape placement, `install` CLI

**Files:**
- Create: `validation/depons_java/install.py`
- Modify: `validation/depons_java/cli.py` (add `install`)
- Modify: `validation/depons_java/NOTES.md` (section `## Install`)
- Test: `tests/test_depons_java_install.py`

**Interfaces:**
- Consumes: `fetch_artifact`, `load_registry`, `default_cache_dir` (Task 1).
- Produces:
  - `class InstallError(Exception)`
  - `@dataclass(frozen=True) class InstallLayout: root: Path; repast_version: str; repast_root: Path; runtime_dir: Path; model_dir: Path; scenario_dir: Path; groovy_jar: Path`
    - `runtime_dir` = `repast_root / f"repast.simphony.runtime_{repast_version}"`
    - `model_dir` = the directory that `start_model.command` `cd`s into (`DEPONS`)
    - `scenario_dir` = `model_dir / "DEPONS.rs"`
  - `inspect_install_dir(root: Path) -> InstallLayout`. Raises `InstallError` naming the
    first missing piece.
  - `install_release(jar: Path, dest: Path) -> InstallLayout`. Idempotent: if `dest`
    already inspects cleanly, return the layout.
  - `install_landscape(zip_path: Path, layout: InstallLayout, landscape: str = "Kattegat") -> Path`.
    Copies the verified zip to `layout.model_dir / "data" / f"{landscape}.zip"` (DEPONS's
    `LandscapeLoader` reads `data/<landscape>.zip` relative to the working directory) and
    returns that path.
  - CLI: `install [--cache DIR]` installs into `<cache>/install/`, places the Kattegat zip,
    and prints the layout.

- [ ] **Step 1: Discover the unattended install (manual).** In a scratch copy, try IzPack 5
  modes in this order:
  1. `java -jar DEPONS-3.2-setup.jar -options-template t.properties`, then
     `-options t.properties` with `INSTALL_PATH` set
  2. `-console`
  3. extracting the packs directly

  Record the working method and the resulting tree (`find <dest> -maxdepth 3 -type d`) in
  NOTES, including the Repast version from the `repast.simphony.runtime_<ver>` dirname.
- [ ] **Step 2: Write the failing tests** (fake trees in `tmp_path`, mirroring the real tree
  recorded in Step 1):
  - `test_inspect_install_dir_complete`
  - `test_inspect_install_dir_missing_runtime_names_it`: the `InstallError` message contains
    `runtime`.
  - `test_install_landscape_copies_to_data_dir`: the target is `model_dir/data/Kattegat.zip`
    and its bytes are equal to the source.
- [ ] **Step 3: Run** the tests. Expected: FAIL.
- [ ] **Step 4: Implement** `install.py` with the method found in Step 1, and the CLI
  subcommand.
- [ ] **Step 5: Run** the tests, then the real
  `micromamba run -n shiny python3 -m validation.depons_java install`. Expected: the tests
  pass and the printed layout points at existing paths. Run lint and the fast suite.
- [ ] **Step 6: Commit** `feat(a1): unattended DEPONS install, layout inspection, landscape placement`.

---

### Task 5: Headless Repast batch run (the crux; discovery + code)

**Files:**
- Create: `validation/depons_java/runner.py`
- Modify: `validation/depons_java/NOTES.md` (section `## Headless run`)
- Test: `tests/test_depons_java_runner.py`

**Interfaces:**
- Consumes: `InstallLayout` (Task 4); `render_batch_params` and `effective_parameters`
  (Task 3).
- Produces:
  - `JVM_FLAGS: tuple[str, ...]` (verbatim from `start_model.command`, without `-Xmx`)
  - `REPAST_BATCH_MAIN: str` (the class confirmed in Step 1)
  - `build_classpath(layout: InstallLayout) -> str`
  - `build_batch_command(layout: InstallLayout, batch_params: Path, *, java: str = "java", xmx: str = "3072M") -> list[str]`
  - `@dataclass(frozen=True) class RunResult: returncode: int; timed_out: bool; wall_seconds: float; log_path: Path`
  - `run_batch(cmd: list[str], cwd: Path, log_path: Path, timeout_s: float) -> RunResult`.
    stdout and stderr go to `log_path`. On timeout the process is killed and
    `timed_out=True`, `returncode=-9`.
  - `java_version(java: str = "java") -> str` (first line of `java -version`)

- [ ] **Step 1: Discover the batch invocation (manual).**
  - Locate the batch main class with `unzip -l` on the jars under
    `repast.simphony.batch_<ver>` (expected `repast.simphony.batch.BatchMain`).
  - Generate a smoke sweep: the psm2 scenario with `porpoiseCount=200`, `simYears=1`,
    seed 42, rendered with Task 3 functions from `<scenario_dir>/parameters.xml`.
  - Run it from `layout.model_dir` with the classpath and flags from `start_model.command`
    plus the batch main class, its params-file argument, and `DEPONS.rs`.
  - Iterate on the classpath and arguments until it finishes.
  - Record in NOTES: the exact command, the cwd, the **landing directory** where output
    files appear, their exact names (including timestamp format and any
    `batch_param_map` companions), the wall-clock time, and every error that had to be
    solved.
  - Then move the smoke outputs out of the landing directory into the scratchpad, leaving
    it clean (see Run isolation).
- [ ] **Step 2: Gate.** No completed headless run by the end of session 1 → stop and go to
  **Task F**.
- [ ] **Step 3: Write the failing tests:**
  - `test_build_classpath_matches_launcher`: it contains `runtime_dir/bin`,
    `runtime_dir/lib/*`, `groovy_jar` and `model_dir/lib/*` in that order, joined by `:`,
    plus any extra entries Step 1 proved necessary.
  - `test_build_batch_command_shape`: `cmd[0] == "java"`; it contains `-Xmx3072M`; `JVM_FLAGS`
    appears as a contiguous ordered subsequence; then `REPAST_BATCH_MAIN`, the params file
    and `str(layout.scenario_dir)` in the order found in Step 1.
  - `test_run_batch_captures_output_and_exit_code` (Review Focus 3): cmd =
    `[sys.executable, "-c", "print('hello'); raise SystemExit(3)"]` gives `returncode == 3`
    and a log containing `hello`.
  - `test_run_batch_timeout_kills`: `[sys.executable, "-c", "import time; time.sleep(30)"]`
    with `timeout_s=0.5` gives `timed_out is True`, and returns in < 5 s.
- [ ] **Step 4: Run** the tests. Expected: FAIL. **Step 5: Implement** `runner.py`.
  **Step 6: Run** the tests, lint and the fast suite. Expected: PASS.
- [ ] **Step 7: Commit** `feat(a1): headless Repast batch runner for DEPONS 3.2`.

---

### Task 6: Output collection, manifests, `run` and `compare` CLI

**Files:**
- Create: `validation/depons_java/collect.py`
- Modify: `validation/depons_java/cli.py` (add `run`, `compare`)
- Test: `tests/test_depons_java_collect.py`

**Interfaces:**
- Consumes: Tasks 1, 3, 4 and 5 interfaces.
- Produces:
  - `class CollectError(Exception)`
  - `normalise_output_name(name: str) -> str | None`. Strips the timestamp segment matching
    `\.\d{4}\.[^.]+\.\d{2}\.\d{2}_\d{2}_\d{2}(?:_\d+)?` before the extension, keeps any
    suffix such as `.batch_param_map`, and returns `None` for names without a timestamp
    segment.
  - `landing_outputs(raw_dir: Path) -> list[Path]` returns the files in `raw_dir` for
    which `normalise_output_name` is not `None`.
  - `assert_landing_clean(raw_dir: Path) -> None` raises `CollectError`, listing the
    leftover names, if `landing_outputs(raw_dir)` is non-empty.
  - `collect_run(raw_dir: Path, dest: Path, *, overwrite: bool = False) -> dict[str, str]`
    **moves** every landing output into `dest` under its normalised name and returns
    `{normalised_name: original_name}`. It raises `CollectError` if `dest` is non-empty
    and not `overwrite` (`overwrite` clears `dest` first), or if two raw files normalise
    to the same name; on that error nothing is moved.
  - `MANIFEST_FIELDS: tuple[str, ...]` =
    `("scenario_id", "seed", "parameters", "depons_version", "depons_commit", "jar_sha256", "landscape_sha256", "java_version", "repast_version", "harness_commit", "harness_dirty", "host", "started_utc", "wall_seconds", "returncode", "timed_out", "status", "output_files")`
  - `run_status(returncode: int, timed_out: bool, collected: dict[str, str]) -> str`
    returns `"ok"` only if returncode 0, not timed out, and `"Statistics.csv"` collected;
    otherwise `"failed"`.
  - `write_manifest(dest: Path, manifest: dict) -> Path`. Writes `dest/manifest.json`
    (sorted keys, indent 2) and raises `CollectError` if any `MANIFEST_FIELDS` key is
    missing.
  - `compare_runs(a: Path, b: Path) -> list[str]` returns the sorted normalised names that
    differ byte-wise or exist in only one run, ignoring `manifest.json`, `java.log` and
    `batch_params.xml`.
  - CLI `run --scenario PATH [--seeds S ...] [--porpoise-count N] [--sim-years Y] [--out DIR] [--timeout-s T] [--overwrite] [--cache DIR]`
    is the **single entry point**:
    - It fetches and verifies the artifacts and installs (release and landscape) when
      they are missing or invalid, idempotently, via the Task 1 and Task 4 functions.
    - Defaults: `--out output/depons_java`, `--timeout-s 21600` (6 h per seed).
    - Layout: `<out>/<scenario_id>/N<count>-Y<years>/seed-<seed>/`.
    - Per seed: `assert_landing_clean`, write `batch_params.xml`, run, collect (move),
      write the manifest.
    - Exit non-zero if any seed's status is `"failed"`.
    - `--porpoise-count` and `--sim-years` override the scenario's values, and the
      manifest records the effective ones.
  - CLI `compare RUN_A RUN_B` prints the differing names and exits 1 if any differ.

- [ ] **Step 1: Write the failing tests:**
  - `test_normalise_names`:
    - `Statistics.2026.Nov.10.12_00_01.csv` → `Statistics.csv`
    - `YearlyMortality.2026.Nov.10.12_00_01_42.csv` → `YearlyMortality.csv`
    - non-English month `Statistics.2026.lis.10.12_00_01.csv` → `Statistics.csv`
    - `Statistics.2026.Nov.10.12_00_01.batch_param_map.csv` → `Statistics.batch_param_map.csv`
    - `notes.txt` → `None`
    - Plus one case per real filename recorded in NOTES (Task 5).
  - `test_collect_duplicate_normalised_names_rejected` (Review Focus 5): nothing is moved
    on error.
  - `test_sequential_runs_collect_only_their_own_files`:
    1. Put set A (`Statistics.<tsA>.csv`, `YearlyMortality.<tsA>.csv`) in `raw`, then
       `collect_run(raw, d1)`.
    2. `assert_landing_clean(raw)` passes.
    3. Put set B in `raw`, then `collect_run(raw, d2)`.
    4. `d2` holds only B's contents and `d1` only A's.
  - `test_assert_landing_clean_lists_leftovers`: the error message contains the leftover
    filename.
  - `test_collect_refuses_nonempty_dest_without_overwrite` (Review Focus 4), and with
    `overwrite=True` stale files are gone.
  - `test_run_status` (Review Focus 3): `(0, False, {"Statistics.csv": ...})` → ok;
    `(1, …)` → failed; `(0, True, …)` → failed; `(0, False, {})` → failed.
  - `test_write_manifest_requires_all_fields`
  - `test_compare_runs_identical_differing_missing`
- [ ] **Step 2: Run** the tests. Expected: FAIL. **Step 3: Implement** `collect.py` and the
  CLI subcommands.
  - `harness_commit` comes from `git rev-parse HEAD`; `harness_dirty` from
    `git status --porcelain` being non-empty.
  - `host` is `platform.node()`; `started_utc` is ISO-8601.
  - `parameters` is the full `effective_parameters` dict; `depons_version` is `"3.2"`.
- [ ] **Step 4: Run** the tests, lint and the fast suite. Expected: PASS. Then a real smoke
  run:
  `micromamba run -n shiny python3 -m validation.depons_java run --scenario validation/depons_java/scenarios/kattegat-psm2-undisturbed-v1.json --seeds 42 --porpoise-count 200 --sim-years 1`.
  Expected: exit 0, and `manifest.json` has `"status": "ok"`.
- [ ] **Step 5: Early determinism probe.** Repeat the same smoke command with
  `--out output/depons_java/repeat`, then `compare` the two `seed-42` directories. Record
  the result in NOTES. If outputs differ, investigate now (minutes at smoke scale), before
  any full-scale run.
- [ ] **Step 6: Commit** `feat(a1): output collection, provenance manifests, run/compare CLI`.

---

### Task 7: Landscape identity check

**Files:**
- Create: `validation/depons_java/landscape.py`
- Modify: `validation/depons_java/cli.py` (add `compare-landscape`)
- Test: `tests/test_depons_java_landscape.py`

**Interfaces:**
- Consumes: `fetch_artifact`, `sha256_file` (Task 1).
- Produces:
  - `@dataclass(frozen=True) class LandscapeReport: identical: list[str]; differing: list[str]; missing_in_candidate: list[str]; extra_in_candidate: list[str]`
    (sorted relative POSIX paths)
  - `compare_dirs(reference: Path, candidate: Path) -> LandscapeReport` (SHA-256 per file,
    recursive)
  - `render_report(report: LandscapeReport) -> str`: a markdown table of the four counts,
    followed by the non-identical file lists.
  - CLI `compare-landscape [--candidate data/Kattegat]` extracts the verified zip into
    `<cache>/landscape/Kattegat/` (if the zip has a single top-level directory, compare
    from inside it), compares, and prints the report.

- [ ] **Step 1: Write the failing tests:** `test_compare_dirs_classifies_all_four`, built
  from a reference/candidate pair in `tmp_path` that has one file of each kind;
  `test_render_report_lists_differing_files`.
- [ ] **Step 2: Run.** Expected: FAIL. **Step 3: Implement.** **Step 4: Run** the tests,
  lint and the fast suite. Expected: PASS.
- [ ] **Step 5: Real check.** Run
  `micromamba run -n shiny python3 -m validation.depons_java compare-landscape` and paste
  the report into NOTES under `## Landscape identity`.
- [ ] **Step 6: Commit** `feat(a1): landscape identity check (DEPONS v3.2 zip vs data/Kattegat)`.

---

### Task 8: Acceptance runs, cost, variable matrix, README, spike report

**Files:**
- Create: `validation/depons_java/README.md`
- Create: `docs/superpowers/specs/<date>-track-a1-spike-report.md` (date of completion)
- Modify: `validation/depons_java/NOTES.md`

**Interfaces:**
- Consumes: the full CLI.

- [ ] **Step 0: Clean tree.** `git status --porcelain` must be empty before any acceptance
  run (Global Constraints: clean-tree provenance). Hold NOTES edits until Step 5.
- [ ] **Step 1: Full psm2 runs.** `run --scenario …/kattegat-psm2-undisturbed-v1.json`
  (seeds 42 and 43, N=2000, 5 years). Expected: both manifests have `"status": "ok"` and
  `"harness_dirty": false`.
- [ ] **Step 2: Determinism.** Re-run seed 42 with `--out output/depons_java/repeat`, then
  `compare output/depons_java/kattegat-psm2-undisturbed-v1/N2000-Y5/seed-42 output/depons_java/repeat/kattegat-psm2-undisturbed-v1/N2000-Y5/seed-42`.
  Record the result. If files differ, record which ones and why (e.g. a `run` column or
  wall-clock fields), and whether the population/energy columns themselves differ.
- [ ] **Step 3: IDW smoke.** Run the idw scenario with `--seeds 42 --porpoise-count 200 --sim-years 1`.
  Expected: `"status": "ok"`.
- [ ] **Step 4: Cost.** From the manifests, compute wall seconds per simulated year at
  N=200 and N=2000. Recommend an A2 replicate count that fits one overnight run.
- [ ] **Step 5: Variable-availability matrix.** Inspect the collected files' headers.
  - For each A2 metric (daily population, mean porpoise energy, food level, annual
    mortality by age, births, dispersal), record: DEPONS file/column/frequency, the
    matching CENOP file/column (`output/kattegat_ref/Population.txt`, `Energy.txt`,
    `Mortality.txt`, `Dispersal.txt`), and any sink that had to be enabled.
  - Record which `DEPONS.rs` sinks wrote files by default.
- [ ] **Step 6: README** (reviewer-facing):
  - prerequisites (Java 21, ~1 GB disk, network)
  - the one-command reproduction (`python3 -m validation.depons_java run --scenario …`,
    which fetches and installs as needed; `fetch` and `install` are optional sub-steps)
  - the output layout and the run-isolation rule
  - how to check manifests and checksums
  - the statement that the DEPONS release ran unmodified
- [ ] **Step 7: Spike report.** Fold NOTES into the report.
  - Cover each item of spec §9: which rung worked, blockers, the Repast version and batch
    invocation, the determinism result, the variable matrix, cost and ensemble size, the
    landscape identity result, and sink defaults.
  - End with the A2 hand-off: scenario files, recommended scale and replicates, and the
    §8 dispersal gap as a prerequisite for the IDW half.
- [ ] **Step 8: Verify.** Run the fast suite, lint, and `git status` (no cache or outputs
  staged).
- [ ] **Step 9: Commit** `docs(a1): spike report, reviewer README, acceptance results`.

---

### Task F (conditional): Fallback ladder

Executed only when a gate in Task 2 or Task 5 sends you here. Take one rung at a time and
record each attempt in NOTES.

- **(a) GUI launcher under a virtual display.** Install `xvfb` if absent (ask the user
  before using `sudo`). Run the release's `start_model.command` under `xvfb-run -a`, driving
  Repast's batch/run options (Repast reference manual "run options"). If it produces the
  outputs, add `build_gui_command(layout) -> list[str]` to `runner.py` (tested like
  `build_batch_command`) and continue with Task 6.
- **(b) Source build** of commit `8f533f8` against a Repast Simphony distribution of the
  matching version. Every manifest records `"depons_build": "source-8f533f8"`, and the
  README labels the runs accordingly. Continue with Task 6.
- **(c) Published output.** Write a decision record in place of the spike report:
  - why (a) and (b) failed
  - which published DEPONS results (TRACE figures/tables, papers) A2 will compare against
  - what is lost: no seed control, fewer metrics, no ensemble

  Commit it; Tasks 6–8 are then skipped.
