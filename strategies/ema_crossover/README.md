EMA Crossover Strategy (Canonical Example)

Overview

This folder contains the canonical `ema_crossover` example strategy (fast EMA / slow EMA).
The strategy implements a simple EMA 5 / EMA 100 crossover with optional entry filters and ATR-based stops.

[![CI](https://github.com/AMfree360/Dev/actions/workflows/ci.yml/badge.svg)](https://github.com/AMfree360/Dev/actions/workflows/ci.yml)

Key configuration options

- `moving_averages.ema_fast.length` (default: 5)
- `moving_averages.ema_slow.length` (default: 100)
- `entry.confirmation_required` (bool) — require a confirmation bar after crossover
- `entry.ema_separation_pct` (float) — minimum percent separation between EMAs to accept entry
- `entry.trend_slope_bars` (int) — number of bars to measure trend slope for filtering
- `stop_loss.atr_enabled` (bool) — enable ATR-based stop placement
- `stop_loss.atr_length` (int) — ATR period length
- `stop_loss.atr_multiplier` (float) — ATR multiplier for stop distance

Usage

- Quick smoke backtest:
  - `python3 scripts/run_backtest.py --strategy ema_crossover --config strategies/ema_crossover/config.yml`
- Run Phase 1 training validation:
  - `python3 scripts/run_training_validation.py --strategy ema_crossover --config strategies/ema_crossover/config.yml --start-date YYYY-MM-DD --end-date YYYY-MM-DD`

Notes

- ATR stop settings are backward compatible with existing `stop_loss` schema. Enable ATR using `stop_loss.atr_enabled: true`.
- If Phase 1 yields too few trades or Monte Carlo randomized-entry runs are skipped, relax entry filters or increase the timeframe/date range.

See also

- `docs/user/06-validation.md` — Validation guide and recent notes
- `scripts/reset_validation_state.py` — Reset Phase 1 state if needed
