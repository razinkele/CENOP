# A1 spike log

## Release jar (Task 2, 2026-11-10)

- `fetch --allow-unpinned` downloaded both artifacts in ~19 s.
  - Kattegat landscape verified against its pinned SHA-256 `2614f56a…94913`.
  - Release jar SHA-256 (trust-on-first-use, now pinned in `artifacts.json`):
    `bfbe73ea977902708cbc41182e82a23a61cda3cfa496bfdc02df8a621cc68bea`
    (287,091,000 bytes; GitHub publishes no digest for this asset).
  - A re-run without `--allow-unpinned` verified from cache with no re-download (mtime
    unchanged).
- The jar is an **IzPack 5.1.3** installer (`resources/vars`: `izpack.version=5.1.3`).
  Its payload is not visible as plain zip entries but packed into blobs:
  `resources/packs/pack-Model` (337,160,819 bytes) and `resources/packs/pack-Sources`
  (3,236,713 bytes).
- Installer variables: `REPAST_VERSION=2.10.0`, `Groovy_Jar=groovy-3.0.17-indy.jar`.
- The pack index `resources/packs.info` lists install targets under
  `$INSTALL_PATH/repast.simphony/`, including:
  - `repast.simphony.runtime_2.10.0` (with `bin/repast/simphony/runtime/RepastBatchMain.class`)
  - `repast.simphony.batch_2.10.0` (with `bin/repast/simphony/batch/BatchMain.class`,
    `lib/batch_groovy.jar`)
  - all other Repast 2.10.0 plugins (core, data, scenario, gis, …) and `saf.core.ui_2.10.0`
- **Gate:** passed. The release bundles the complete Repast 2.10.0 runtime, including
  both headless batch entry points. No Java runtime is bundled (system OpenJDK 21.0.12 is
  used).

## Install (Task 4, 2026-11-10)

- **IzPack 5.1.3 does not run on Java 21.** Every mode (`-options-template`, `-options`)
  fails before installing anything with `ArrayIndexOutOfBoundsException: Index 70131 out of
  bounds for length 22674`. Stack trace (`-DDEBUG=true -DSTACKTRACE=true`): inside
  `org.picocontainer.paranamer.BytecodeReadingParanamer$ClassReader`. This is IzPack's
  dependency-injection container parsing class bytecode with a 2018-era parser that
  cannot read Java 21 class files. An installer/JDK incompatibility, not a DEPONS defect.
- **Works with Java 11** (system `/usr/lib/jvm/java-11-openjdk-amd64`, OpenJDK 11.0.32.1):
  `java -jar DEPONS-3.2-setup.jar -options install.properties`, where the properties file
  contains only `INSTALL_PATH=<dest>`. Exit 0; unpacks Model, Sources and Documentation
  packs (~341 MB). Unattended and headless (no `DISPLAY`, `-Djava.awt.headless=true`).
  The installer only unpacks files, so the JRE used to run it does not affect the model.
- Installed tree (`<dest>`):
  - `start_model.command`, `start_model.bat` (with `$REPAST_VERSION`/`$Groovy_Jar`
    substituted), `Uninstaller/`
  - `groovylib/groovy-3.0.17-indy.jar`
  - `repast.simphony/<plugin>_2.10.0/` for every Repast plugin, including
    `repast.simphony.runtime_2.10.0/{bin,lib,boot.properties,plugin_jpf.xml}` and
    `repast.simphony.batch_2.10.0/`
  - `DEPONS/` = the model dir that the launcher `cd`s into: `DEPONS.rs/`, `bin/`, `lib/`
    (jackson jars), `data/`, `output/`, `src/`, the TRACE PDFs, …
- **The release already bundles all five v3.2 landscapes** in `DEPONS/data/` (`DanTysk`,
  `Gemini`, `Homogeneous`, `Kattegat`, `NorthSea` `.zip`). The bundled `Kattegat.zip` is
  **byte-identical** to the separately pinned depons.eu download (SHA-256 `2614f56a…94913`).
  `install_landscape` still places the verified copy explicitly.
- Launcher classpath note: `start_model.command` adds the relative entry `lib/*` and then
  `cd DEPONS`. The JVM expands it from its working directory, so the model jars come from
  `DEPONS/lib/*`.

## Headless run (Task 5, 2026-11-10)

- **Worked on the first attempt** (well within session 1, so the time-box gate passed).
  Entry point: `repast.simphony.runtime.RepastBatchMain`, the batch twin of the
  launcher's `RepastMain`, in `repast.simphony.runtime_2.10.0/bin`. Its options (from the
  class constants): `-params <sweep file>`, `-opt <file>`, `-interactive`,
  `-bootprops <file>`, `-help`. It boots `repast.simphony.batch.BatchInit` via the JPF
  plugin system.
- Exact command (cwd = `<install>/DEPONS`, `DISPLAY` unset, system OpenJDK 21.0.12):
  `java -Djava.awt.headless=true -Xmx3072M <start_model.command JVM flags> -cp
  "<rt>/bin:<rt>/lib/*:<install>/groovylib/groovy-3.0.17-indy.jar:lib/*"
  repast.simphony.runtime.RepastBatchMain -params <abs batch_params.xml> ./DEPONS.rs`.
  Here `<rt>` = `<install>/repast.simphony/repast.simphony.runtime_2.10.0`. This is the
  same classpath as `start_model.command`; no extra entries were needed.
- Smoke (psm2 scenario, N=200, simYears=1, seed 42): **exit 0, wall 31.8 s**
  (~0.10 s per simulated day). The log confirms `Loading landscape files from
  …/DEPONS/data/Kattegat.zip` and monthly prey files. The only stderr noise is
  `WARNING: Using incubator modules: jdk.incubator.vector`.
- **Landing directory = the model dir `<install>/DEPONS/`** (the working directory).
  Files written, with timestamp `yyyy.MMM.dd.HH_mm_ss` (YearlyMortality adds `_SSS`):
  - `Statistics.<ts>.csv`: `"run","tick","PorpoiseCount","FoodEnergyLevel","PorpoiseEnergyLevel"`,
    one row per tick (17,279 rows for 1 year = 360 days × 48 ticks)
  - `YearlyMortality.<ts_ms>.csv`: `"run","year","age","count","deaths"`
  - `PorpoisePerBlock.<ts>.csv`, `RandomPorpoise.<ts>.csv`, `Reproduction.<ts>.csv`,
    `Hydrophones.<ts>.csv`. **All `DEPONS.rs` file sinks write by default in batch mode.**
  - a `<Sink>.<ts>.batch_param_map.csv` for each Repast sink. It is the full effective
    parameter row (`run` plus all 70 parameters) and confirms the sweep was applied:
    `randomSeed=42`, `simYears=1`, `porpoiseCount=200`, `dispersal=PSM-Type2`,
    `PSM_dist=N(350;100)`.
  - `debug.log` (not timestamped; empty in this run)
  - `ReproductiveLog*` / `PersistentSpatialMemory*` were **not** written. They are not
    default outputs.
- Observation for A2: in the smoke run the population fell 200 → 107 within year 1, and
  `PorpoiseEnergyLevel` (a population sum) went 2008 → 1224. A first-year decline at this
  initial N/age structure. Not judged here.
- The smoke outputs were moved to the scratchpad afterwards; the landing dir is clean.
