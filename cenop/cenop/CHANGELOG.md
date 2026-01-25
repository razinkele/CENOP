# Changelog

All notable changes to this project will be documented in this file.

## 2026-01-24 — Data Preview refactor & bug fixes (by Arturas Razinkovas-Baziukas)

### Summary

- Reworked the *Data Preview* feature (pydeck-based) to remove inline `<script>` injection and prevent accumulation of script tags when switching files.
- Eliminated a reactive loop that caused app reload/crash when changing the preview file selector.
- Added robust client-server communication using `session.send_custom_message` with a persistent `Shiny.addCustomMessageHandler('preview_data_update')` on the client side.
- Converted the previous server-side `preview_data_loader` into a pure data computation (`@reactive.calc`: `preview_data_source`) and a small async effect (`preview_map_updater`) that sends messages to the UI. Added `preview_stats_text` renderer for metadata.
- Implemented caching for preview data to avoid repeated heavy loads and added smart downsampling for large ASC grids (adaptive sampling based on grid size to limit points sent to the client).
- Added detailed debug logging (`[PREVIEW DEBUG]`) and loop-detection guardrails.
- Fixed NameError by ensuring required imports (e.g., `Path`) are present where needed.

### Files changed (high-level)

- `src/cenop/server/main.py` — refactor `preview_data_loader` into `preview_data_source`, add `preview_map_updater`, `preview_stats_text`, caching, logging, and safety checks
- `src/cenop/ui/tabs/settings.py` — replaced script-injection pattern with a persistent `Shiny.addCustomMessageHandler('preview_data_update', ...)` and added `preview_stats_text` output
- `src/cenop/ui/tabs/settings.py` — removed inline `preview_data_loader` script injection point to avoid repeated script tags
- `src/cenop/ui/tabs/dashboard.py` — reviewed for related issues and confirmed no reactive loop present; minor logging recommendations retained
- `README.md` — added Contact information

### Rationale & Notes

- Script injection via renderers led to repeated inline `<script>` blocks being injected into the DOM on each file change; this caused duplicated handlers and extraneous `postMessage` events (`Message received: undefined`) and risked degrading client performance or causing reloads. Moving to a single custom message handler eliminates these issues.
- Using a `@reactive.calc` for heavy lifting (pure computation) and a `@reactive.effect` for side-effects (send to session) separates concerns and reduces accidental reactivity loops.

### Testing & Verification

- Manual testing performed: switching landscapes and files in the Data Preview no longer injects scripts and no longer triggers reloads.
- Browser console logs show `[PREVIEW] Received data update via custom message` and `[PREVIEW] Data sent to iframe` confirming successful flow.

---

(If you want this changelog updated in another format or included in the project root, tell me and I will add it.)
