# Research Layer (No-Code Strategy Builder)

TradingLab’s research layer lets you define strategies in a **human-readable spec** (YAML) and compile them into a standard TradingLab strategy folder.

Goal:
- You describe the strategy in a structured, readable format.
- TradingLab summarizes it and generates the necessary `strategies/<name>/` files.
- You run the normal backtest + validation pipeline.

This is **systematic**, not discretionary: you specify rules and filters that can be tested and validated.

---

## 1) What this gives you

- A “no-code” front-end for strategy research.
- Repeatable compilation: edit the YAML spec, re-generate.
- Compatibility with the rest of TradingLab (backtests, filters, validation, reports).

---

## 2) The Strategy Spec format

A Strategy Spec is a YAML file (see `research_specs/`).

Core fields:
- `name`, `description`
- `market.symbol` (and optional `exchange`, `market_type`)
- `entry_tf` and optional `extra_tfs`
- `long` / `short` rules:
  - `conditions_all`: list of conditions (timeframe + expression)
  - optional stop definition (`percent` or `atr`)
- `filters`: pass-through TradingLab filter configuration (calendar/regime/news/volume)

Optional:
- `trade_management`: overrides for engine-managed exits (merged with `config/master_trade_management.yml`)

Example (EMA cross):

```yaml
name: research_ema_cross
market:
  symbol: BTCUSDT
entry_tf: 4h

long:
  conditions_all:
    - tf: 4h
      expr: "crosses_above(ema(close, 20), ema(close, 50))"
```

### Trade management (engine-managed exits)

The research spec can override TradingLab’s trade management config. These settings are compiled into the generated `strategies/<name>/config.yml` and merged with `config/master_trade_management.yml` by the engine.

Supported (currently compiled as **R-based** levels):
- `trade_management.trailing_stop`
- `trade_management.take_profit.levels` (`target_r`, `exit_pct`)
- `trade_management.partial_exit.levels` (`level_r`, `exit_pct`)

Example:

```yaml
trade_management:
  trailing_stop:
    enabled: true
    type: EMA
    length: 21
    activation_type: r_based
    activation_r: 0.5
    stepped: true
    min_move_pips: 7.0

  take_profit:
    enabled: true
    levels:
      - target_r: 2.0
        exit_pct: 50
      - target_r: 3.0
        exit_pct: 50

  partial_exit:
    enabled: true
    levels:
      - level_r: 1.5
        exit_pct: 80
```

---

## 3) Condition expression DSL

Expressions are a safe subset of Python-like syntax over OHLCV series.

Base series:
- `open`, `high`, `low`, `close`, `volume`

Derived series:
- `hl2` or `hl2()` ( (high + low) / 2 )
- `hlc3` or `hlc3()` ( (high + low + close) / 3 )
- `ohlc4` or `ohlc4()` ( (open + high + low + close) / 4 )

Supported functions:
- `ema(close, n)`
- `sma(close, n)`
- `rsi(close, n)`
- `atr(n)`
- `donchian_high(n)`
- `donchian_low(n)`
- `crosses_above(a, b)`
- `crosses_below(a, b)`
- `highest(series, n)`
- `lowest(series, n)`
- `shift(series, n)`
- `nz(series, value)`

### Multi-timeframe references

You can reference another timeframe inside a single expression.

Supported forms:
- `at("1h", close)`
- Shorthand: `close@1h` (equivalent to `at("1h", close)`)
- Shorthand (general): `<atom>@1h` where `<atom>` can be an indicator call or a parenthesized expression

When you use `at("1h", ...)`, TradingLab evaluates the inner expression on the `1h` dataframe and forward-fills it onto the current timeframe.

Examples:

```text
close > close@2h
ema(close, 20) > at("4h", ema(close, 50))
(ema(close, 20) > ema(close, 50))@4h
ema(close, 20)@4h > ema(close, 50)
ema(close@4h, 20) > ema(close, 50)
```

Notes:
- `ema(close@4h, 20)` is normalized internally to `at("4h", ema(close, 20))`.

---

## Best Practices

### 1) Prefer clarity for multi-timeframe logic

- Use `at("4h", ...)` when the intent is “evaluate this on 4h” (especially when nesting).
- Use `@4h` shorthand when it stays readable.

Good:

```text
ema(close, 20) > at("4h", ema(close, 50))
```

Also good (shorter):

```text
ema(close, 20) > ema(close, 50)@4h
```

### 2) Put indicators on the timeframe you mean

- If you want a 4h EMA, write `ema(close@4h, 20)` or `at("4h", ema(close, 20))`.
- Avoid patterns like `ema(at("4h", close), 20)` (it looks like 4h intent, but is easy to misread).

### 3) Understand alignment (forward-fill)

`at("4h", ...)` is aligned onto the current timeframe using forward-fill.

Implication:
- On `1h`, a `4h` value persists for the next 4 `1h` bars.

This is usually what you want for confirmations, but it also means your signal can “stay true” until the higher timeframe updates.

### 4) Keep it causal

- Avoid writing rules that implicitly rely on “future” bars.
- Favor crossover functions like `crosses_above(a, b)` which use only current/previous bars.

### 5) Start simple, then add gates

- First build an entry rule that trades.
- Then add 1–2 confirmations (often higher timeframe trend/regime).
- Then add filters (sessions/regime/news) last.

### 6) Debugability tip

When a condition gets complex, split it into separate `conditions_all` lines (even if they’re on the same timeframe). It makes summaries and troubleshooting much easier.

Supported operators:
- comparisons: `>`, `>=`, `<`, `<=`, `==`, `!=`
- boolean: `and`, `or`, `not`
- arithmetic: `+`, `-`, `*`, `/`

---

## 4) Build (compile) a strategy folder

```bash
python3 scripts/build_strategy_from_spec.py --spec research_specs/example_ema_cross.yml --print-summary
```

### Plain English (template-based, no ML)

If you want to hide code entirely, you can start from controlled plain-English notes and have TradingLab generate the YAML spec for you.

This is **not** free-form NLP; it’s a deterministic parser for a small template. If your text is ambiguous or missing required details (symbol, timeframe, etc.), TradingLab will ask targeted follow-up questions.

Example notes (put in a `.txt` file or paste via stdin):

```text
Symbol: BTCUSDT
Exchange: binance

Go long when EMA(20) crosses above EMA(50) on 4h
Enter short when close < SMA(200) on 4h

# Optional confirmations/gates (applied to both long/short)
Only trade when EMA(50) > EMA(200) on 1d

# Optional calendar/session filters (UTC+00)
Only trade during London session

Stop ATR(14) * 3
Take profit 2R 50%, 3R 50%
Partial exit 1.5R 80%
Trailing EMA 21 after 0.5R

# Breakout example (Donchian)
Go long when breakout above Donchian high 20 on 4h

# Breakdown example (Donchian)
Go short when breakdown below 20-bar low on 4h
```

Generate a YAML spec (interactive if needed):

```bash
python3 scripts/build_strategy_from_english.py --name my_english_strategy --in notes.txt
```

Generate and compile to `strategies/<name>/`:

```bash
python3 scripts/build_strategy_from_english.py --name my_english_strategy --in notes.txt --compile
```

This creates:
- `strategies/<name>/config.yml`
- `strategies/<name>/strategy.py`

---

## 5) Backtest the generated strategy

```bash
python3 scripts/run_backtest.py --strategy research_ema_cross --data data/processed/BTCUSDT-4h.parquet
```

Then proceed to validation as usual.

---

## Notes / roadmap

- This layer is designed to hide Python from users, but the generated strategy code is still a normal TradingLab strategy (for transparency and reproducibility).
- Over time, we can add:
  - more indicators
  - multi-timeframe helpers
  - a menu/interactive builder that writes the YAML spec
  - richer stop/exit/position sizing primitives
