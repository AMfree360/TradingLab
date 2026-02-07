# Runbook

Copy/paste commands to run the research pipeline.

## Execution timing (important)

- Default model: **decide at bar close, fill at next bar open**.
- This is intentional to reduce lookahead bias.

## Setup

- Run tests:

  `./.venv/bin/python -m pytest -q`

## Phase 1: Training validation

Phase 1 includes a **Monte Carlo suite**:
- **Universal tests** (eligible for gating when suitable): permutation, block bootstrap
- **Conditional tests** (diagnostic by default): randomized-entry

In reports you’ll see two combined robustness views:
- **Universal combined** (used for Phase 1 gating)
- **All-suitable combined** (informational)

- Fast iteration (small MC):

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --data data/raw/BTCUSDT-4h-2023.parquet --market BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31 --mc-iterations 100`

- Full run (production-ish):

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --data data/raw/BTCUSDT-4h-2023.parquet --market BTCUSDT --start-date 2023-01-01 --end-date 2023-12-31 --mc-iterations 1000`

Expected artifacts:
- Report(s) under `reports/`.
- Dataset manifest JSON written under `data/manifests/` (path printed).
- Run recorded in `.experiments/registry.sqlite`.

## Phase 2: OOS validation (walkforward)

⚠️ Phase 2 OOS is consumed on attempt.

- Example:

  `./.venv/bin/python scripts/run_oos_validation.py --strategy donchian_breakout --data data/raw/BTCUSDT-4h-2023.parquet --market BTCUSDT --start-date 2024-01-01 --end-date 2024-12-31 --wf-training-period "1 year" --wf-test-period "6 months"`

Expected artifacts:
- Report(s) under `reports/`.
- Dataset manifest JSON under `data/manifests/`.
- Run recorded in `.experiments/registry.sqlite`.

## Final OOS (holdout) test

⚠️ Holdout is one-shot (consumed on attempt).

- Example:

  `./.venv/bin/python scripts/run_final_oos_test.py --strategy donchian_breakout --data data/raw/BTCUSDT-4h-2023.parquet --market BTCUSDT --holdout-start 2025-01-01 --holdout-end 2025-12-31`

Expected artifacts:
- Backtest-style report under `reports/`.
- Holdout manifest JSON under `data/manifests/`.
- Run recorded in `.experiments/registry.sqlite`.

## Experiment registry

Runs are recorded automatically by the phase scripts.

- Registry DB: `.experiments/registry.sqlite`

- List recent runs:

  `./.venv/bin/python scripts/runs.py list --limit 20`

- Filter by strategy/phase:

  `./.venv/bin/python scripts/runs.py list --strategy donchian_breakout --phase phase1_training --limit 50`

- Compare two runs by id:

  `./.venv/bin/python scripts/runs.py compare --a 12 --b 15`

- Catalog latest status by strategy + phase:

  `./.venv/bin/python scripts/runs.py catalog --limit 30`

- Catalog only failures for a strategy:

  `./.venv/bin/python scripts/runs.py catalog --strategy donchian_breakout --status fail --limit 50`

## Portfolio risk controls (v1)

TradingLab supports a simple, deterministic portfolio layer centered on **open risk / risk-to-stop**.

### Open-risk cap (risk-to-stop)

When enabled, the engine computes total open risk across all open positions and **scales down new entries** so the new trade fits inside the remaining risk budget.

- Open risk definition: worst-case loss if every open position is stopped at its current stop.
- Behavior when cap is exceeded: new entry size is scaled down; if no size can fit, the entry is skipped.

Config (strategy YAML):

```yaml
portfolio:
  enabled: true
  # Example: never have more than 2% of the account at risk across all open positions
  max_open_risk_pct: 2.0
```

### Stop-loss fallback (last resort)

If a strategy cannot produce a valid stop-loss price, a portfolio-level fallback can be enabled.

- Fallback stop distance: `multiplier × ATR(period)`
- Default: `3 × ATR(14)`

Config (strategy YAML):

```yaml
portfolio:
  enabled: true
  stop_fallback_enabled: true
  stop_fallback_atr_period: 14
  stop_fallback_atr_multiplier: 3.0
```

Live signal example (simplified):

```python
# Strategy emits a live signal dict
signal = {
  "action": "BUY",
  "quantity": 0.05,
  # Recommended: always include an explicit stop_price
  "stop_price": 41250.0,
}

# If stop_price is missing/invalid AND stop fallback is enabled,
# the engine will compute a last-resort stop using 3×ATR(14).
```

### Live trading caveat (stop tracking)

In live mode, portfolio open-risk calculation depends on having a known stop price for every open position.

- Stops placed/updated by the engine are tracked locally.
- If the process restarts, any already-open positions may have unknown stops; the engine will **block new entries** until stop tracking is restored.

## Calendar filters (opt-in)

TradingLab supports calendar-based filters (day-of-week, trading sessions, time blackouts). These are **off by default** and only apply when explicitly enabled in the strategy config.

Key switches:

- `calendar_filters.master_filters_enabled`: master on/off switch for applying filters in `StrategyBase.apply_filters()`.
- `calendar_filters.day_of_week.enabled`: enables weekday allowlist.
- `calendar_filters.trading_sessions_enabled`: enables `TradingSessionFilter` (defining sessions alone does not activate filtering).

Note on `config/master_filters.yml`:

- Master filter config is merged into the filter chain, but filter *application* is gated by the strategy’s `calendar_filters.master_filters_enabled`.
- To keep behavior predictable, master config alone will not “turn on” calendar filtering for a strategy.

### Day-of-week filter

`allowed_days` uses Python weekday numbering: `0=Mon ... 6=Sun`.

```yaml
calendar_filters:
  master_filters_enabled: true
  day_of_week:
    enabled: true
    allowed_days: [0, 1, 2, 3, 4]  # weekdays only
```

### Trading sessions

Trading sessions are evaluated in UTC+00. A signal passes if it is inside **any enabled session**.

```yaml
calendar_filters:
  master_filters_enabled: true
  trading_sessions_enabled: true
  trading_sessions:
    London:
      enabled: true
      start: "07:00"
      end: "16:00"
```

If `trading_sessions_enabled: false`, the `trading_sessions` definitions are ignored for filtering.

## Troubleshooting

- If Phase 2 is blocked because OOS is consumed:
  - This is expected and by design.
  - Use a different OOS slice/data file, or explicitly reset state only for dev/testing.

- If a phase aborts due to dataset lock mismatch:
  - The dataset slice identity changed (file contents or slice boundaries).
  - Treat it as a stop-the-line event; decide whether this is a legitimate new dataset version or an accidental mutation.
