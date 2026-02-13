# Strategy Development (User Guide)

This guide explains how to create and run a new strategy in TradingLab without needing to understand the internal engine architecture.

If you want the technical deep-dive (base classes, signals, engine lifecycle), see: [Developer Strategy Development Guide](../dev/05-strategy-development.md).

---

## What you need to create

A strategy lives in `strategies/<strategy_name>/` and typically contains:

- `config.yml` — the strategy configuration (market, timeframes, risk, trade management)
- `strategy.py` — the trading logic
- `README.md` (optional) — notes about the strategy

Recommended naming:
- Folder name: lowercase with underscores (e.g., `donchian_breakout`)

---

## Step 1 — Copy an existing strategy (fastest path)

The easiest way to start is to copy an existing strategy folder and modify it.

Example:

```bash
cp -r strategies/ema_crossover strategies/my_strategy
```

Then edit:
- `strategies/my_strategy/config.yml`
- `strategies/my_strategy/strategy.py`

---

## Step 2 — Configure your strategy (`config.yml`)

Your `config.yml` controls how the strategy behaves.

Minimum you should set:

- `strategy_name` and `description`
- `market.symbol` (e.g., `BTCUSDT`)
- `timeframes.signal_tf` and `timeframes.entry_tf`
- risk and trade management parameters appropriate to the market

See the full reference: [Configuration Guide](04-configuration.md).

### Timeframes: the most common early mistake

- Your dataset has a bar interval (e.g., 15m, 1h, 4h).
- Your config’s `timeframes.entry_tf` should match what you intend to trade.

If you load 1h data but `entry_tf` is 15m, results can be invalid unless you resample.

---

## Step 3 — Implement your logic (`strategy.py`)

A strategy is Python code that:

- reads indicators or patterns from price bars
- decides when a trade is allowed
- returns decisions at bar close (the engine fills at next bar open by default)

Practical approach:
- Start simple (one clear entry + one clear exit)
- Avoid adding multiple filters at once
- Make sure the strategy produces trades in backtest before optimizing

---

## Step 4 — Run a dev backtest

Run a backtest using the backtest script:

```bash
python3 scripts/run_backtest.py \
  --strategy my_strategy \
  --data data/processed/BTCUSDT-4h.parquet
```

Optional:
- Add `--report-visuals` to include equity + drawdown charts.

---

## Common issues and best responses

### 1) “I downloaded data in yearly/monthly chunks”

Backtests normally expect a single file.

Consolidate first (recommended):

```bash
python3 scripts/consolidate_data.py --input "data/raw/BTCUSDT-15m-*.parquet" --output data/processed/BTCUSDT-15m.parquet
```

Or let the runner do it:

```bash
python3 scripts/run_backtest.py --strategy my_strategy --data "data/raw/BTCUSDT-15m-*.parquet" --auto-consolidate
```

### 2) “My files are mixed frequency (daily + hourly)”

Don’t merge mixed intervals directly. Resample to one timeframe, then consolidate.

```bash
python3 scripts/resample_ohlcv.py --src data/raw/BTCUSDT-1h-2022.parquet --dst data/raw/BTCUSDT-4h-2022.parquet --rule 4h
python3 scripts/consolidate_data.py --input "data/raw/BTCUSDT-4h-*.parquet" --output data/processed/BTCUSDT-4h.parquet
```

### 3) “My backtest has 0 trades”

Common causes:
- entry conditions never occur (too strict)
- filters block everything
- timeframe mismatch (strategy logic expects more granular data)

Debug tips:
- temporarily loosen one condition at a time
- print (or log) the reason entries are rejected

### 4) “Indicators look wrong / results look too good”

Most often:
- timeframe mismatch (strategy uses 15m logic on 1h data)
- accidental lookahead in custom code
- mixed-frequency data merged incorrectly

Use the workflow guides:
- Data sanity: [Data Management Guide](03-data-management.md)
- Backtesting sanity: [Backtesting Guide](05-backtesting.md)

---

## Next step: Validation

Once the strategy runs and produces trades on the dev slice, move to validation:

- Phase 1: `python3 scripts/run_training_validation.py ...`
- Phase 2 (WF OOS): `python3 scripts/run_oos_validation.py ...`
- Final holdout: `python3 scripts/run_final_oos_test.py ...`

For the standardized reproducible workflow, see: [Golden Path: Split Policy + Config Profiles](11-split-policy-and-profiles.md).
