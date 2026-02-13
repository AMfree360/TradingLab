# Session Notes

## Snapshot (as of 2026-02-12)

### What shipped / changed today
- Guided Strategy Builder v2 (GUI): exposed risk controls in **Advanced (Step 5)** (not Step 4):
  - Sizing: `sizing_mode` (`account_size|daily_equity|equity`) + `account_size`.
  - Daily limits: `max_trades_per_day` + optional `max_daily_loss_pct` (checkbox-enabled; null disables).
  - Backtest costs: optional overrides `commissions` and `slippage_ticks` (checkbox-enabled; null uses market defaults).

- Research spec + compiler pipeline:
  - `StrategySpec` carries the new fields so they persist through YAML → compile.
  - Compiler emits:
    - `risk.sizing_mode` + `risk.account_size`
    - `trade_limits.max_trades_per_day` + `trade_limits.max_daily_loss_pct`
    - `backtest.commissions` / `backtest.slippage_ticks` only when explicitly overridden.

- Market-aware defaults/hints:
  - Advanced defaults resolver + `/api/advanced-defaults` now include commission hints (`commission_rate`, `commission_per_contract`) alongside slippage defaults.
  - Advanced Step 5 shows market-profile hint values to reduce confusion.

- Guided edit persistence:
  - `.guided.json` sidecar meta now round-trips the new fields.
  - Edit-guided reconstructs the advanced risk settings from the spec as the source of truth.

### Current status
- Test suite: `116 passed`.

## Snapshot (as of 2026-02-11)

### What shipped / changed today
- Phase 4 (engine/spec enhancements):
  - Stop-distance guardrails (min + optional max) enforced at entry.
  - Context invalidation handling (`execution.exit_if_context_invalid`):
    - `mode: immediate` emits an exit intent with reason `context_invalid`.
    - `mode: tighten_stop` emits a stop-tightening intent (breakeven/entry by default).
  - Per-level metadata for partial exits (levels carry context for reporting).
  - Exit reason propagation from strategy intent → engine → trade records.

- Guided Strategy Wizard (Advanced Step 5):
  - Added UI controls to enable/disable context invalidation handling and pick `immediate` vs `tighten_stop`.
  - Ensured the setting is serialized into the generated spec on both Review preview and Create+Compile paths.
  - Guided edit/reopen round-trips the new fields via the `.guided.json` sidecar metadata and YAML hydration.

- GUI Launcher usability:
  - Added pagination for Strategies and Runs pages (`page` + `per_page` query params).
  - Added the ability to delete GUI run bundles from:
    - Runs list page
    - Run detail page
  - Deletion uses a typed-name confirmation step and deletes the run bundle directory under `data/logs/gui_runs/`.

- Guided Strategy Builder v2 (Step 4) preview UX:
  - Context visual (preview): candlestick chart under the “Primary context type” selector with context overlays + shading.
  - Setup visual (preview): entry timeframe candles with context shading and markers where ALL(Context + Signals + Triggers) align.
  - Mixed-timeframe clarity when `context_tf != entry_tf`:
    - API message explicitly states forward-fill (`Context TF: X (forward-filled → entry)`).
    - In-chart badge annotation reinforces the same.
  - Optional marker layers for “signals (all)” and “triggers (all)” are present but hidden by default (toggle via legend).

- Builder v2 regime alignment + MA filter generalization:
  - Fixed alignment semantics so `align_with_context` only affects context/regime (signals/triggers keep correct LONG=bull, SHORT=bear polarity).
  - Added optional MA slope + separation/distance filters to MA-based context types (e.g., price-vs-MA, MA cross, MA spread).

- Bug fix:
  - Fixed `/runs/{run_name}/delete` 500 error (missing `urllib.parse` import).

### Current status
- Test suite: `109 passed`.

## Snapshot (as of 2026-02-10)

### What shipped / changed today
- Filter system correctness (root-cause fix for “why is ADXFilter running?”):
  - Updated filter config merging so Pydantic schema defaults don’t accidentally override master filter defaults (dump with `exclude_unset=True`).
  - Updated `config/master_filters.yml` defaults to be safer: regime ADX + news filter disabled unless explicitly enabled per strategy.

- Backtest report Trade Chart usability:
  - Candles are readable by default (limited initial x-window + wider report layout + taller chart).
  - Added range slider and range-selector buttons for quick navigation.
  - Added wheel interactions for 2-axis control:
    - Wheel pans price (y)
    - Shift+Wheel pans time (x)
    - Ctrl/Cmd+Wheel zooms price (y)
    - Ctrl/Cmd+Shift+Wheel zooms time (x)

### Current status
- Test suite: `96 passed`.

## Snapshot (as of 2026-02-09)

### What shipped / changed today
- Controlled-English parsing hardening:
  - Added strict vs lenient parse modes.
  - Improved robustness for compound long/short lines (e.g., “go long when …, go short when …”).
  - RSI parsing now supports above/below templates and defaults RSI length to 14 when omitted.
  - Failure UX improved: if no LONG/SHORT entry rule can be extracted, CLI exits with code 2 and a friendly message (no traceback).
- GUI Launcher (thin orchestrator over canonical scripts):
  - FastAPI web launcher runs the same scripts as the CLI and records reproducible run bundles.
  - Run bundles stored under `data/logs/gui_runs/` (inputs, command, stdout/stderr, meta).
  - Runs list + run detail pages added with safe path handling.
  - Backtest reliability fixes:
    - GUI runs are now non-interactive (prevents “stuck running forever” when a script calls `input()`).
    - GUI backtests always pass `--auto-resample`.
    - Added split policy controls to the backtest UI.
    - Added `--split-policy-mode {enforce|auto|none}` in `scripts/run_backtest.py`; GUI defaults to `auto` so custom uploads don’t get filtered to 0 bars by canonical policy dates.
    - Added dataset inspection endpoint that detects uploaded dataset date range/timeframe and auto-applies the detected start/end range by default (still editable).
- Guided Strategy Wizard (GUI):
  - Tight pipeline that generates a validated `StrategySpec`, writes `research_specs/<name>.yml`, then compiles to `strategies/<name>/`.
  - Market-aware UX: futures-only settings (margin mode) only appear for futures; FX/futures market type is locked to prevent inconsistent combos.
  - Entry rules UX clarified: up to 2 conditions per side (AND); condition #2 is optional and hidden by default.
- Futures support upgrades:
  - Added/expanded micro + mini futures instruments in market profiles and guided instrument lists.
  - Added execution flag `execution.use_intraday_margin` and wired it end-to-end (schema → compiler → backtest engine → broker affordability).
- Research layer exit rules (controlled-English → DSL → spec → compiled strategy):
  - Supports explicit “Exit …” rules and templates like “exit on opposite breakout/breakdown”.
  - Compiler emits exit signals and the engine consumes them as `signal_exit` (without breaking existing stop/target/trailing behavior).
- Backtest report visual upgrades:
  - HTML report can render a Trade Chart (candlesticks + indicator overlays + entries/exits/SL).
  - Trade chart UX improved: auto-zoom to the trade window and dedicated partial-exit markers.

- Post-backtest iteration loop (GUI):
  - “Tune + rebuild” flow loads `spec.yml` from a run bundle, edits knobs safely, writes a new `research_specs/<new>.yml`, recompiles a v2 strategy, and links it from the run page.
  - Run detail includes “Quick diagnosis + suggested tweaks” powered by report JSON (enhanced with MFE/MAE exit-reason counts when available).

### Day summary (2026-02-09)
- Stabilized GUI backtests end-to-end: no interactive prompts, deterministic args, and reproducible bundles.
- Removed the biggest foot-gun for uploads: canonical split policies no longer silently date-filter custom datasets to 0 bars in the GUI.
- Made the system’s assumptions visible: dataset range/timeframe detection is shown and the detected date range is applied by default (but remains editable).
- Improved iteration speed: run → diagnose → apply preset/tune → rebuild v2 → rerun, all from the run detail.
- Kept the core pipeline deterministic (spec validation + compilation + scripts) while improving UX and guardrails.

### Current status
- Test suite: `96 passed`.

### What’s left to do (next actions)
- Optional wizard branching polish:
  - Consider beginner vs advanced paths.
  - Skip/shorten futures-only concepts for non-futures flows where possible.
- Optional chart enhancements (if desired): show TP/trailing events explicitly when available; add per-trade filtering/legend grouping.
- Optional backtest UX polish (if desired): suggest market/profile defaults from filename/symbol when uploading custom datasets.

## Snapshot (as of 2026-02-08)

### What shipped / changed today
- Standardized split-policy enforcement across the pipeline:
  - Central policy file: `config/data_splits.yml` (Phase 1 / Phase 2 / final holdout).
  - Phase scripts refuse ad-hoc date ranges unless explicitly overridden.
- Data pipeline hardening + history expansion:
  - Rebuilt canonical processed datasets from 15m sources into 4h (left-edge labeling).
  - Refreshed multi-year BTCUSDT history and corrected timestamp unit issues in 2025 ingest.
- Phase 2 walk-forward correctness:
  - Fixed window generation to drop truncated trailing windows that were producing 0-trade OOS segments and distorting WF stats.
- Multi-provider market data download capability:
  - Providers support crypto bulk archives + generic equity/FX sources (pluggable adapter model).
- Config-profile support for phase scripts (enables clean tuning iterations):
  - Phase 1: `scripts/run_training_validation.py --config-profile <name>`
  - Phase 2: `scripts/run_oos_validation.py --config-profile <name>`
  - Final OOS: `scripts/run_final_oos_test.py --config-profile <name>`
- Tuning harness for `donchian_breakout`:
  - New script: `scripts/tune_donchian_breakout.py`.
  - Writes a profile override to `strategies/donchian_breakout/configs/profiles/<profile>.yml`.
  - Extended to include a lightweight block-bootstrap precheck to target the Phase 1 Monte Carlo “individual tests” gate.

### Current status
- Pipeline is now “policy-driven” and reproducible end-to-end.
- `donchian_breakout` can be tuned to pass the basic Phase 1 backtest gates (PF/Sharpe/DD/Trades), but still fails Phase 1 due to Monte Carlo bootstrap individual p-values not meeting the strict majority rule.

### What’s left to do (next actions)
- Fix the `scripts/tune_donchian_breakout.py` crash encountered during candidate printing when trailing stop params are missing/disabled (KeyError on `activation_r`).
- Reduce noisy filter logging during tuning runs (ADX filter rejects spam) so tuner output is usable.
- Run the enhanced tuner at higher budget to find candidates that pass the bootstrap individual-tests gate (p ≤ 0.05 on ≥2/3 metrics).
- Once Phase 1 passes with a profile, proceed with Phase 2 walk-forward and final holdout using the same profile (respecting one-shot consumption semantics).

## Snapshot (as of 2026-02-06)

### What shipped today (Phase 1 validation + reporting + ops)
- Monte Carlo suite refactor:
  - Tests split into **universal** vs **conditional** categories.
  - Suitability-aware execution: universal tests always attempted; conditional tests run only when suitable.
  - Phase 1 gating uses **only suitable + universal** Monte Carlo tests.
  - Bar-count/suitability bug fixed (uses true dataset bar count, not a derived slice).
- Monte Carlo combined robustness now has **two views**:
  - Universal combined (used for Phase 1 gating)
  - All-suitable combined (informational: universal + conditional that ran)
- Registry operationalization:
  - SQLite registry outcomes now persist pass/fail status and structured reasons in `outcome_json`.
  - CLI gained a “catalog” view to quickly see latest pass/fail by strategy + phase.
- Reports updated to match the new MC policy:
  - Each MC test is labeled universal/conditional and whether it was used for gating.
  - Both combined blocks are displayed (universal gating + all-suitable info).
- Docs updated under `docs/` (user + dev) to match the new semantics and CLI.
- CLI alignment: Phase 1 script default `--mc-iterations` restored to 1000.

### What shipped
- Execution realism contract: strategy emits intent, engine enforces fill timing (default decision at bar close, fill next bar open).
- Anti-lookahead hardening + tests (strategies cannot use forward data to decide trades).
- MarketSpec consolidation: canonical implementation in `engine/market.py`; `engine/market_spec.py` is a silent compatibility shim.
- Dataset governance:
  - Deterministic dataset manifests with stable hashing.
  - Phase locks: Phase 1/2/holdout refuse to proceed if dataset slice identity changes.
  - Phase 2 OOS is consumed on attempt (not on success).
  - Walkforward warmup data bounded to train-start → oos-end to prevent holdout leakage.
  - Manifests written to `data/manifests/` and the exact path printed.
- Experiment tracking:
  - SQLite registry at `.experiments/registry.sqlite`.
  - Runs are auto-recorded from phase scripts.
  - CLI: `scripts/runs.py list|compare`.
  - CLI: `scripts/runs.py catalog`.

### Key decisions / invariants (do not break)
- Fill timing: evaluate at bar close, fill next bar open (unless explicitly configured otherwise).
- No lookahead: no shifting signals forward to confirm in the future.
- Monte Carlo policy:
  - Universal tests are eligible for gating if suitable.
  - Conditional tests are diagnostics by default (reported, not gated).
  - Combined robustness is emitted in two views (universal gating + all-suitable info).
- Phase locks:
  - Phase 1 lock binds training slice identity.
  - Phase 2 lock binds OOS slice identity.
  - Holdout lock binds final OOS slice identity.
  - If the manifest changes mid-phase, the run must stop.
- OOS semantics:
  - Phase 2 OOS is consumed on attempt.
  - Holdout is one-shot (consumed on attempt).
- Walkforward input window: must not include holdout; allow only train-start → oos-end.

### Where to look (anchors)
- Fill timing + execution contract: `engine/backtest_engine.py`, `config/schema.py`, tests.
- Phase orchestration + locking: `validation/pipeline.py`, `validation/state.py`.
- Dataset manifest: `repro/dataset_manifest.py`.
- Experiment registry + CLI: `experiments/registry.py`, `scripts/runs.py`.
- Monte Carlo policy + suitability: `validation/suitability.py`, `validation/training_validator.py`, `validation/monte_carlo/runner.py`.
- Report rendering: `reports/report_generator.py`.

### Verification
- Tests: `./.venv/bin/python -m pytest -q`
- Registry quick check:
  - `./.venv/bin/python scripts/runs.py list --limit 10`
  - `./.venv/bin/python scripts/runs.py catalog --limit 20`

## Next session (focus)

### 1) Optimize `donchian_breakout` toward profitability
- Use Phase 1 reports to identify the failure mode (especially randomized-entry diagnostics vs the strategy’s edge).
- Iterate on:
  - Entry trigger quality (false breakouts, regime filters, volatility filters)
  - Exit logic (ATR stop/target policy, trailing behavior, time stop)
  - Parameter search ranges (ensure sensible defaults + bounded overfitting)
- Rerun Phase 1 with fast settings while iterating (`--mc-iterations 100`), then confirm with 1000.

### 2) Build a better data downloader (make backtests/validation easy)
- Define a single “dataset request” interface: symbol, venue, market-type (spot/futures), timeframe, start/end, timezone, and storage paths.
- Add/extend a downloader that can:
  - Fetch raw OHLCV, validate schema, write to `data/raw/`
  - Produce canonical processed datasets in `data/processed/` (and optionally auto-resample)
  - Persist manifests + provenance (source, params, retrieval time)

### 3) Inventory market data sources
- Gather and document sources for crypto/FX/equities/futures (cost, licensing, rate limits, historical depth).
- Pick the initial “supported sources” list and build adapters incrementally.

## Longer-term themes (still relevant)
- Portfolio layer:
  - Multi-strategy/multi-market allocation.
  - Correlation-aware risk caps.
  - Exposure constraints.
  - Portfolio-level performance attribution.
- Execution realism upgrades:
  - Partial fills.
  - Fee schedules (tiered maker/taker).
  - Funding/borrow models.
  - Spread models.
  - Latency models.

### Strategy follow-ups (after portfolio layer exists)
- Re-review `ema_crossover` under portfolio sizing + constraints.
- Re-review `donchian_breakout` under portfolio constraints + execution realism.

### Immediate “first task” suggestion
- Keep on ice until Donchian + data pipeline are stable.

