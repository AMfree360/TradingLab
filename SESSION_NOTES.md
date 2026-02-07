# Session Notes

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

