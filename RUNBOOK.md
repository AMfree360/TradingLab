# Runbook

Copy/paste commands to run the research pipeline.

## Execution timing (important)

- Default model: **decide at bar close, fill at next bar open**.
- This is intentional to reduce lookahead bias.

## Exit signals (strategy behavior)

Some strategies (including generated research strategies) may emit explicit exit intent.

- A strategy can emit an exit signal (e.g., `action="exit"`).
- The engine treats these as “signal exits” and closes matching open positions at the current bar’s price.
- Hard stops / trade-management exits still take precedence based on the engine’s exit ordering.

### Context invalidation handling (optional)

Research specs can optionally enable a safety behavior: when the strategy’s **context/regime flips from true → false** while in a position, the compiler emits an intent based on the configured mode.

Config shape:

```yaml
execution:
  exit_if_context_invalid:
    enabled: true
    mode: immediate   # or: tighten_stop
```

Modes:
- `immediate`: compiler emits `action="exit"` with reason `context_invalid`.
- `tighten_stop`: compiler emits `action="tighten_stop"` (engine tightens stop in-place; default tighten mode is breakeven/entry if no explicit stop price is provided).

Notes:
- Exit intents are only consumed when the signal timestamp is <= the current bar timestamp (mirrors entry intent handling for mixed TF).
- Stop-tightening intent never closes the position directly; it only updates the stop and then normal stop/target/trailing logic applies.

### Stop-distance guardrails (engine)

If enabled in the spec/config, the engine enforces stop-distance constraints at entry (e.g., minimum stop distance, optional maximum stop distance). This helps avoid unrealistically tight or excessively wide stops when a strategy’s generated stop is out of bounds.

## Setup

- Run tests:

  `./.venv/bin/python -m pytest -q`

## GUI Launcher (local web)

Thin local web UI that runs the same TradingLab scripts.

- Install GUI deps:

  `./.venv/bin/python -m pip install -r requirements-gui.txt`

- Run the launcher:

  `./.venv/bin/python scripts/run_gui_launcher.py`

- Optional flags:
  - Bind to a specific host/port:

    `./.venv/bin/python scripts/run_gui_launcher.py --host 127.0.0.1 --port 8000`

  - Dev auto-reload:

    `./.venv/bin/python scripts/run_gui_launcher.py --reload`

Key pages:
- `/` home
- `/strategies` browse/edit strategy specs
- `/create-strategy-guided` guided wizard (recommended)
- `/backtest` run a backtest (supports file uploads + advanced options)
- `/runs` run bundles (commands, inputs, logs)

Artifacts (commands + logs + inputs) are stored under `data/logs/gui_runs/`.

Runs page UX:
- Pagination is supported via query params: `/runs?page=2&per_page=50`.
- Deleting a run is supported from both the runs list and the run detail page:
  - Click “Delete…”
  - Type the run name to confirm
  - This permanently deletes the run bundle directory

Strategies page UX:
- Pagination is supported via query params: `/strategies?page=2&per_page=50`.

Backtest UX notes:
- GUI backtests are **non-interactive** (no `input()` prompts). The launcher passes `--auto-resample` so the run can’t hang waiting for a terminal response.
- If you upload a custom dataset, the GUI will inspect it and **auto-apply the detected start/end range** by default (still editable in Advanced options).
- Split policy is configurable in Advanced options; the GUI defaults to `--split-policy-mode auto` so custom uploads don’t get accidentally constrained by canonical policy dates.

## Guided Strategy Wizard (GUI)

The guided wizard is the “tight pipeline” path: it generates a validated research spec, writes it to `research_specs/<name>.yml`, then compiles via the existing scripts.

- The wizard is market-aware:
  - FX forces `market_type=spot`.
  - Futures forces `market_type=futures` and exposes futures-only controls.

- Entry rules UX:
  - Legacy guided flow: up to 2 conditions per side (AND).
  - Builder v2 (Step 4) supports multiple Context + Signals + Triggers items per side (combined with AND), with add-buttons and JSON persistence.

- Builder v2 alignment semantics:
  - `align_with_context` aligns **context/regime** bullish rules to LONG and bearish rules to SHORT when both sides are enabled.
  - Signals/triggers remain direction-correct: LONG uses bull polarity; SHORT uses bear polarity.

- Visual previews (Builder v2 Step 4):
  - Context visual (preview): entry candles + primary context overlays/shading.
  - Setup visual (preview): entry candles + context shading + markers where ALL(Context + Signals + Triggers) align.
  - Data source: both previews use the newest GUI dataset under `data/logs/gui_runs/**/dataset_*`.
    - If previews show “No dataset found…”, run any GUI backtest (or upload a dataset) first.
  - Mixed timeframes: when `context_tf != entry_tf`, context is evaluated on `context_tf` and forward-filled to the entry chart index (so shading may look “stepped”).
  - Legend toggles: the setup chart includes optional marker layers (signals/triggers) that can be enabled via the Plotly legend.

- Risk/limits/cost overrides (Advanced Step 5):
  - Step 4 keeps the core “setup” decisions (context/signal/trigger + risk per trade).
  - Step 5 now hosts optional account/risk controls:
    - Sizing: `sizing_mode` and `account_size`
    - Daily limits: `max_trades_per_day` and optional `max_daily_loss_pct`
    - Backtest costs: optional overrides for `commissions` and `slippage_ticks`
  - If commission/slippage overrides are left disabled, TradingLab uses market profile defaults from `config/market_profiles.yml`.

### Quick verification checklist (GUI + YAML + CLI)

Use this after making changes to builder/compiler/engine behavior.

1) Sanity: run tests

   `./.venv/bin/python -m pytest -q`

2) Builder v2 alignment semantics (Step 4)

   - Start: `/create-strategy-guided` → pick “Builder v2”.
   - In Step 4:
     - Ensure both LONG + SHORT are enabled.
     - Toggle “Align direction with context”.
     - Pick a primary context type (e.g., “Price vs MA”).
   - In “Effective rules (preview)” confirm:
     - LONG context shows the bullish regime expression (e.g., “Price above MA…”).
     - SHORT context shows the bearish regime expression (e.g., “Price below MA…”).
     - Signals/triggers remain direction-correct (LONG uses bull polarity, SHORT uses bear polarity).

3) MA slope + separation filters on MA-based context rules

   - In Step 4 → “More context filters (optional)” → Add a context rule:
     - `Trend: Price vs MA`:
       - Enable “Require MA slope filter” and set `Slope lookback bars`.
       - Set `Min price/MA distance` (optional).
   - Confirm the Effective Rules include the added slope + distance expressions.

4) Context invalidation handling (Advanced Step 5)

   - Continue to Step 5 → Execution:
     - Enable “Exit if context invalidates”.
     - Pick mode `immediate` or `tighten_stop`.
   - On Review, confirm the generated spec includes:

     ```yaml
     execution:
       exit_if_context_invalid:
         enabled: true
         mode: immediate   # or: tighten_stop
     ```

5) End-to-end: compile + run one backtest

   - From Review: Save → Compile → Backtest.
   - Or via CLI (example):

     `./.venv/bin/python scripts/run_backtest.py --strategy <your_strategy_name> --data <path_to_dataset> --split-policy-mode none --report-visuals`

   - Inspect the run bundle under `data/logs/gui_runs/` for the exact command, inputs, and generated report artifacts.

6) Risk/limits/cost overrides (Advanced Step 5)

   - Continue to Step 5 → “Risk & limits”:
     - Set `sizing_mode` and `account_size`.
     - Set `max_trades_per_day`.
     - Optionally enable `max_daily_loss_pct`.
     - Optionally override `commissions` and/or `slippage_ticks`.
   - On Review, confirm the generated spec includes those fields at the top level.

## Build a strategy from controlled-English (CLI)

This is deterministic parsing (no LLM). It outputs a YAML spec and optionally compiles to `strategies/<name>/`.

- Parse only (writes `research_specs/<name>.yml`):

  `./.venv/bin/python scripts/build_strategy_from_english.py --name my_strategy --in notes.txt --entry-tf 1h`

- Parse + compile:

  `./.venv/bin/python scripts/build_strategy_from_english.py --name my_strategy --in notes.txt --entry-tf 1h --compile`

- Strict vs lenient parsing:
  - `--parse-mode strict` (default): requires explicit controlled-English templates.
  - `--parse-mode lenient`: accepts a broader set of entry lines without guessing values.

- Non-interactive automation:
  - `--non-interactive` fails with exit code 2 if clarifications are required.
  - `--answers-json answers.json` supplies clarification answers deterministically.

Example `answers.json` (shape only):

```json
{
  "exchange": "binance",
  "symbol": "BTCUSDT"
}
```

Note: If no LONG or SHORT entry rule is extracted, the script exits with code 2 and a friendly message (no traceback).

## Futures notes (micros/minis, margin, slippage)

- Micros/minis are represented as distinct symbols (e.g., `MES` vs `ES`) with their own contract/tick/margin/cost specs in `config/market_profiles.yml`.
- Slippage is modeled in ticks (or price units depending on market profile) so the $ impact naturally scales with tick value/contract size.
- Margin enforcement in backtests supports intraday vs overnight via `execution.use_intraday_margin`:
  - `true` (default): use intraday margin for affordability checks (day-trading style).
  - `false`: use initial/overnight margin (swing/hold behavior).

## Reports (visuals)

By default, HTML reports are generated in a compact mode (no embedded charts).

To include charts (equity/drawdown + trade chart), pass `--report-visuals`.

- Backtest:

  `./.venv/bin/python scripts/run_backtest.py --strategy ema_crossover --data data/processed/BTCUSDT-4h-2018-2021.parquet --report-visuals`

- Phase 1:

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --report-visuals --mc-iterations 100`

- Phase 2:

  `./.venv/bin/python scripts/run_oos_validation.py --strategy donchian_breakout --report-visuals`

- Final holdout:

  `./.venv/bin/python scripts/run_final_oos_test.py --strategy donchian_breakout --report-visuals`

Notes:
- Visual charts use Plotly (loaded from a CDN in the HTML report).
- The Trade Chart shows candlesticks plus overlays (when available) and trade annotations:
  - entries + exits
  - stop-loss line segments (when `stop_price` exists)
  - partial exit markers (when recorded)

Trade Chart navigation tips (backtest report):
- Use the range buttons (e.g. `15m`, `30m`, `1h`, `4h`, `1D`, `1W`, `1M`, `3M`, `All`) to jump the visible window.
- Use the range slider (mini-chart under the x-axis) to scrub horizontally.
- Mouse wheel controls:
  - Wheel: pan price (y)
  - Shift+Wheel: pan time (x)
  - Ctrl/Cmd+Wheel: zoom price (y)
  - Ctrl/Cmd+Shift+Wheel: zoom time (x)

## AI-assisted drafting (optional)

TradingLab’s core pipeline is deterministic (parse/validate/compile/backtest). AI can be used **only as a drafting layer** to rewrite messy notes into controlled-English that the existing parser understands.

- Script: `./.venv/bin/python scripts/build_strategy_from_notes_ai.py`
- Provider: currently supports `--provider ollama` (local)
- Artifacts: writes prompt + raw response + output hashes under `data/logs/ai/`
- Parsing: `--parse-mode strict|lenient` (default: strict)

Example:

`./.venv/bin/python scripts/build_strategy_from_notes_ai.py --name my_strategy --in notes.txt --provider ollama --model llama3.1 --entry-tf 1h --parse-mode lenient --compile --print-controlled-english --auto-resolve-clarifications --max-clarify-repairs 1`

## Downloading market data

TradingLab can download raw OHLCV for different asset classes via pluggable providers.

Download script: `scripts/download_data.py`

Providers:
- `binance_api` (crypto, REST; good for small ranges)
- `binance_vision` (crypto, bulk monthly archives; best for multi-year)
- `yfinance` (stocks / indices / FX / futures as supported by Yahoo symbols; optional dependency)
- `stooq` (daily-only equities/indices/FX where available; stooq symbol format)
- `csv_url` (any market/vendor that can expose OHLCV as CSV)

Examples:

- Crypto (fast multi-year via Binance Data Vision, then resample 15m → 4h):

  `./.venv/bin/python scripts/download_data.py --provider binance_vision --market-type spot --symbol BTCUSDT --interval 15m --start 2018-01-01 --end 2021-12-31 --resample-to 4h --processed-output data/processed/BTCUSDT-4h-2018-2021.parquet`

- Stocks (Yahoo Finance, daily):

  `./.venv/bin/python scripts/download_data.py --provider yfinance --symbol AAPL --interval 1d --start 2018-01-01 --end 2021-12-31 --output data/raw/AAPL-1d-2018-01-01_to_2021-12-31.parquet`

- Indices (Yahoo Finance):

  `./.venv/bin/python scripts/download_data.py --provider yfinance --symbol '^GSPC' --interval 1d --start 2018-01-01 --end 2021-12-31 --output 'data/raw/^GSPC-1d-2018-01-01_to_2021-12-31.parquet'`

- FX (Yahoo Finance):

  `./.venv/bin/python scripts/download_data.py --provider yfinance --symbol EURUSD=X --interval 1h --start 2023-01-01 --end 2023-06-30 --resample-to 4h --processed-output data/processed/EURUSD=X-4h-2023H1.parquet`

Note: provider symbol formats vary (e.g. Yahoo uses `EURUSD=X`, futures like `ES=F`).
Note: if you use `yfinance`, you may need to install it with `pip install yfinance`.

## Standard data split policy (required)

All research/validation runs should follow a single standardized split policy to avoid accidental data leakage.

- Policy file: `config/data_splits.yml`
- Default policy name: `default_btcusdt_4h`

Current BTCUSDT 4h splits:

- Backtest/Dev (iterative): 2019-01-01 → 2019-12-31
- Phase 1 (training validation gate): 2019-01-01 → 2020-12-31
- Phase 2 (walk-forward OOS gate): 2021-01-01 → 2024-12-31 (WF 12mo train / 12mo test, expanding)
- Final OOS holdout (one-shot): 2025-01-01 → 2025-12-31

The phase scripts enforce these ranges by default. If you pass a different `--data`/date range, the script will error unless you explicitly use `--override-split-policy`.

Backtest script split-policy modes:
- `scripts/run_backtest.py` supports `--split-policy-mode`:
  - `enforce` (default): enforce the chosen policy range (and error unless overridden when conflicting inputs are provided)
  - `auto`: enforce policy only when using canonical/downloader flows (i.e., no custom `--data` upload path)
  - `none`: do not apply a split policy range

Examples:
- Canonical behavior (policy-enforced backtest):

  `./.venv/bin/python scripts/run_backtest.py --strategy ema_crossover --split-policy-mode enforce --split-policy-name default_btcusdt_4h`

- Custom dataset backtest without policy constraints:

  `./.venv/bin/python scripts/run_backtest.py --strategy ema_crossover --data data/processed/MY_DATA.parquet --split-policy-mode none`

## Phase 1: Training validation

Phase 1 includes a **Monte Carlo suite**:
- **Universal tests** (eligible for gating when suitable): permutation, block bootstrap
- **Conditional tests** (diagnostic by default): randomized-entry

In reports you’ll see two combined robustness views:
- **Universal combined** (used for Phase 1 gating)
- **All-suitable combined** (informational)

- Fast iteration (small MC):

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --mc-iterations 100`

- Run with a config profile override (recommended for tuned configs):

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --config-profile phase1_tuned --mc-iterations 1000`

- Full run (production-ish):

  `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --mc-iterations 1000`

Expected artifacts:
- Report(s) under `reports/`.
- Dataset manifest JSON written under `data/manifests/` (path printed).
- Run recorded in `.experiments/registry.sqlite`.

## Phase 2: OOS validation (walkforward)

⚠️ Phase 2 OOS is consumed on attempt.

- Example:

  `./.venv/bin/python scripts/run_oos_validation.py --strategy donchian_breakout`

- Example with config profile:

  `./.venv/bin/python scripts/run_oos_validation.py --strategy donchian_breakout --config-profile phase1_tuned`

Expected artifacts:
- Report(s) under `reports/`.
- Dataset manifest JSON under `data/manifests/`.
- Run recorded in `.experiments/registry.sqlite`.

## Final OOS (holdout) test

⚠️ Holdout is one-shot (consumed on attempt).

- Example:

  `./.venv/bin/python scripts/run_final_oos_test.py --strategy donchian_breakout`

- Example with config profile:

  `./.venv/bin/python scripts/run_final_oos_test.py --strategy donchian_breakout --config-profile phase1_tuned`

## Tuning (non-consuming)

To iterate on strategy parameters without consuming Phase 2 / holdout validation state, use:

- Donchian tuner (writes a config profile under `strategies/donchian_breakout/configs/profiles/`):

  `./.venv/bin/python scripts/tune_donchian_breakout.py --max-evals 300 --top-n 10 --write-profile phase1_tuned`

Then run Phase 1 using the profile:

- `./.venv/bin/python scripts/run_training_validation.py --strategy donchian_breakout --config-profile phase1_tuned --mc-iterations 1000`

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

Note on non-calendar master filters (regime/news/volume):
- The master config may define defaults for additional filters (e.g., regime filters like ADX).
- These should be treated as **opt-in per strategy**. If you want a strategy to use ADX, explicitly enable it in the strategy config (or via your guided builder advanced options) rather than relying on implicit schema defaults.

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
