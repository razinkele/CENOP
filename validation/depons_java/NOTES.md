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
