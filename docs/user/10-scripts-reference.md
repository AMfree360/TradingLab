# TradingLab Scripts Reference

This document lists all available scripts in the TradingLab system and their purposes.

## Location

All scripts are located in the `scripts/` directory.

## Main Scripts

### Backtesting

#### `run_backtest.py`
**Purpose**: Run a single backtest on historical data.

**Usage**:
```bash
python scripts/run_backtest.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --market <market_symbol>
```

**Notes**:
- `--data` can be a single file, a directory, or a glob (e.g., `data/raw/BTCUSDT-15m-*.parquet`).
- If `--data` resolves to multiple files, either consolidate first or pass `--auto-consolidate`.

**Useful flags**:
- `--auto-consolidate`: Consolidate multiple input files into a single parquet automatically
- `--consolidated-output`: Optional explicit output path for the consolidated parquet
- `--report-visuals`: Include optional charts (equity + drawdown) in the HTML report

**Output**: HTML report in `reports/` directory.

---

### Research Layer

#### `build_strategy_from_spec.py`
**Purpose**: Compile a human-readable research spec (YAML) into a standard TradingLab strategy folder under `strategies/<name>/`.

**Usage**:
```bash
python scripts/build_strategy_from_spec.py \
  --spec research_specs/example_ema_cross.yml \
  --print-summary
```

**Output**:
- `strategies/<name>/config.yml`
- `strategies/<name>/strategy.py`

**Notes**:
- Generated strategies are normal TradingLab strategies; backtest/validation workflows are unchanged.

---

### Validation

#### `run_training_validation.py`
**Purpose**: Run Phase 1 validation (Training Validation) on training data.

**Usage**:
```bash
python scripts/run_training_validation.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --market <market_symbol>
```

**What it does**:
- Backtest quality check
- Monte Carlo suite (permutation, block bootstrap, randomized entry)
- Parameter sensitivity analysis
- Strategy suitability assessment

**Output**: Validation report and state tracking.

---

#### `run_oos_validation.py`
**Purpose**: Run Phase 2 validation (OOS Validation) with walk-forward analysis.

**Usage**:
```bash
python scripts/run_oos_validation.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --wf-training-period "1 year" \
  --wf-test-period "6 months" \
  --wf-holdout-end-date "2025-01-01" \
  --market <market_symbol>
```

**What it does**:
- Walk-forward analysis on OOS data
- Calculates WFE, OOS Consistency, Binomial test
- Excludes periods with insufficient trades
- Generates walk-forward visualizations

**⚠️ WARNING**: OOS data can only be used once!

**Output**: Validation report with walk-forward analysis and visualizations.

---

#### `run_final_oos_test.py`
**Purpose**: Run one-time final OOS test on holdout period.

**Usage**:
```bash
python scripts/run_final_oos_test.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --holdout-start <YYYY-MM-DD> \
  --holdout-end <YYYY-MM-DD> \
  --market <market_symbol>
```

**Requirements**:
- Phase 1 must have PASSED
- Phase 2 must have PASSED
- Holdout period must not have been tested before

**⚠️ CRITICAL**: This test can only be run ONCE per strategy/data combination!

**Output**: Final OOS test report.

---

#### `run_stationarity.py`
**Purpose**: Run Phase 3 validation (Enhanced Stationarity Analysis) to determine retraining frequency.

**Usage**:
```bash
python scripts/run_stationarity.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --market <market_symbol> \
  --training-period-days 365 \
  --max-days 30
```

**When to use**: After strategy has been live trading for some time (e.g., 3-6 months of live data).

**What it tests**:
- **Conditional Metrics**: Win rate, payoff ratio, signal frequency, big win frequency
- **Filter-Aware Regime Analysis**: Only tests enabled regime filters
- **3-Window Baseline**: Uses robust baseline from stabilized windows (configurable)
- **Dual-Threshold Logic**: Absolute (PF < 1.5) + relative degradation

**Output**: 
- Recommended retraining frequency with detailed reason
- PF range (min-max observed)
- Conditional metrics stability (checked only in baseline windows)
- Regime-conditioned performance (if filters enabled)
- Comprehensive metric definitions

**Configuration**: All parameters configurable in `config/validation_criteria.yml` under `stationarity:` section.

---

#### `run_validation.py`
**Purpose**: Run complete validation pipeline (Phase 1 + Phase 2).

**Usage**:
```bash
python scripts/run_validation.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --phase <backtest|phase1|phase2|final_oos> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --market <market_symbol> \
  --config-profile <profile_name>
```

**Split policy (recommended)**:
- By default, TradingLab enforces a standard split policy from `config/data_splits.yml`.
- If a split policy is present and you are not overriding it, you must provide `--phase` so the script can select the correct policy range.
- To run ad-hoc dates/data (not recommended), use `--override-split-policy`.

**What it does**: Runs both Phase 1 and Phase 2 validation sequentially.

**Notes**:
- For the most reproducible workflow, prefer the phase-specific scripts:
  - `scripts/run_training_validation.py` (Phase 1)
  - `scripts/run_oos_validation.py` (Phase 2)
  - `scripts/run_final_oos_test.py` (holdout)

---

### Optimization

#### `optimize_strategy.py`
**Purpose**: Optimize strategy parameters using grid search.

**Usage**:
```bash
python scripts/optimize_strategy.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD> \
  --param <param_path> <value1> <value2> ... \
  --metric <metric_name> \
  --output <output_config_file>
```

**Example**:
```bash
python scripts/optimize_strategy.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2024.csv \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \
  --metric profit_factor \
  --output strategies/ema_crossover/config_optimized.yml
```

**⚠️ IMPORTANT**: Only optimize on training data! Save OOS data for validation.

---

### Data Management

#### `download_data.py`
**Purpose**: Download historical data from exchanges/APIs.

**Usage**:
```bash
python scripts/download_data.py \
  --symbol <symbol> \
  --interval <interval> \
  --start <YYYY-MM-DD> \
  --end <YYYY-MM-DD> \
  --output <output_file>
```

**Example**:
```bash
python scripts/download_data.py \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --output data/raw/BTCUSDT-15m-2020-2024.parquet
```

---

#### `consolidate_data.py`
**Purpose**: Consolidate multiple data files into a single file.

**Usage**:
```bash
python scripts/consolidate_data.py \
  --input <file1> <file2> ... \
  --output <output_file>
```

**Example**:
```bash
python scripts/consolidate_data.py \
  --input data/raw/EURUSD_2020.csv data/raw/EURUSD_2021.csv data/raw/EURUSD_2022.csv \
  --output data/raw/EURUSD_2020-2022.csv
```

**Additional options**:
- `--dedupe-keep {first,last}`: When removing duplicate timestamps, keep first or last (default: first)
- `--allow-mixed-frequency`: Allow inputs with different inferred bar intervals (default: off; recommended to resample first)

---

### State Management

#### `reset_validation_state.py`
**Purpose**: Reset validation state for a strategy (use with caution!).

**Usage**:
```bash
python scripts/reset_validation_state.py \
  --strategy <strategy_name> \
  --data <data_file>
```

**⚠️ WARNING**: This will reset validation progress. Use only if you need to restart validation from scratch.

---

## Experiment registry

Trading Lab records runs (phase scripts, backtests, etc.) into a local SQLite registry:

- Database: `.experiments/registry.sqlite`
- CLI: `scripts/runs.py`

### `runs.py`

**List recent runs**:

```bash
python scripts/runs.py list --limit 20
```

**Filter by strategy/phase**:

```bash
python scripts/runs.py list --strategy ema_crossover --phase phase1_training --limit 50
```

**Compare two runs**:

```bash
python scripts/runs.py compare --a 12 --b 15
```

**Show latest pass/fail per strategy + phase (catalog)**:

```bash
python scripts/runs.py catalog
python scripts/runs.py catalog --status pass
python scripts/runs.py catalog --strategy ema_crossover
python scripts/runs.py catalog --json
```

Notes:
- The registry is meant to make runs comparable and auditable.
- Phase scripts store a structured outcome payload (pass/fail and reasons) in the registry so you can quickly see where a strategy failed.
- Validation semantics still apply (e.g., Phase 2 OOS / holdout are one-shot and consumed on attempt).

---

### Debug Scripts

#### `debug_backtest_comparison.py`
**Purpose**: Compare backtest results between different implementations.

**Location**: `scripts/debug_backtest_comparison.py`

---

#### `debug_trades.py`
**Purpose**: Debug individual trades and execution logic.

**Location**: `scripts/debug_trades.py` (root directory)

---

#### `debug_trades_detailed.py`
**Purpose**: Detailed trade debugging with full execution trace.

**Location**: `scripts/debug_trades_detailed.py` (root directory)

---

#### `debug_position_sizing.py`
**Purpose**: Debug position sizing calculations.

**Location**: `scripts/debug_position_sizing.py` (root directory)

---

### Debug Utilities (scripts/debug/)

#### `debug_capital.py`
**Purpose**: Debug capital tracking and accounting.

#### `debug_strategy.py`
**Purpose**: Debug strategy signal generation and execution.

#### `inspect_data.py`
**Purpose**: Inspect data file structure and contents.

#### `inspect_parquet.py`
**Purpose**: Inspect Parquet file format and schema.

---

### Testing

#### `test_failure_counting.py`
**Purpose**: Test failure counting logic in validation.

---

## Script Categories

### Production Scripts (Regular Use)
- `run_backtest.py`
- `run_training_validation.py`
- `run_oos_validation.py`
- `run_final_oos_test.py`
- `run_stationarity.py`
- `optimize_strategy.py`
- `download_data.py`
- `consolidate_data.py`

### Administrative Scripts
- `reset_validation_state.py`

### Debug Scripts (Development)
- `debug_backtest_comparison.py`
- `debug_trades.py`
- `debug_trades_detailed.py`
- `debug_position_sizing.py`
- `scripts/debug/*.py`

### Testing Scripts
- `test_failure_counting.py`

---

## Common Workflows

### Quick Strategy Test
```bash
# 1. Run backtest
python scripts/run_backtest.py --strategy my_strategy --data data.csv --start-date 2020-01-01 --end-date 2021-12-31 --market EURUSD

# 2. Review results in reports/
```

### Full Validation Workflow
```bash
# 1. Phase 1: Training Validation
python scripts/run_training_validation.py --strategy my_strategy --data data.csv --start-date 2020-01-01 --end-date 2021-12-31 --market EURUSD

# 2. Phase 2: OOS Validation
python scripts/run_oos_validation.py --strategy my_strategy --data data.csv --start-date 2020-01-01 --end-date 2024-12-31 --wf-training-period "1 year" --wf-test-period "6 months" --market EURUSD

# 3. (Optional) Final OOS Test on Holdout
python scripts/run_final_oos_test.py --strategy my_strategy --data data.csv --holdout-start 2025-01-01 --holdout-end 2025-12-31 --market EURUSD
```

### Optimization Workflow
```bash
# 1. Optimize parameters
python scripts/optimize_strategy.py --strategy my_strategy --data data.csv --start-date 2020-01-01 --end-date 2021-12-31 --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 --metric profit_factor --output strategies/my_strategy/config_optimized.yml

# 2. Run validation with optimized parameters
python scripts/run_training_validation.py --strategy my_strategy --data data.csv --config strategies/my_strategy/config_optimized.yml --start-date 2020-01-01 --end-date 2021-12-31 --market EURUSD
```

---

### Benchmarking

#### `run_benchmark.py`
**Purpose**: Benchmark Trading Lab against external platforms (MT5, NinjaTrader, TradingView).

**What it does**:
- Runs the same strategy in Trading Lab
- Imports trades from external platform export
- Compares trades trade-by-trade (engine equivalence)
- Compares metrics (metric parity)

**Usage**:
```bash
python scripts/run_benchmark.py \
  --strategy <strategy_name> \
  --data <data_file> \
  --external-trades <external_trades.csv> \
  --platform <mt5|ninjatrader|tradingview> \
  --start-date <YYYY-MM-DD> \
  --end-date <YYYY-MM-DD>
```

**Parameters**:
- `--strategy`: Strategy name
- `--data`: Market data file (same as used in external platform)
- `--external-trades`: Path to external platform trade export (CSV)
- `--platform`: Platform name (`mt5`, `ninjatrader`, `tradingview`)
- `--start-date` / `--end-date`: Date range (must match external platform)
- `--time-tolerance`: Time tolerance for trade matching in seconds (default: 60)
- `--price-tolerance-pct`: Price tolerance percentage (default: 0.01%)
- `--metric-tolerance-pct`: Metric comparison tolerance (default: 5.0%)
- `--output`: Output directory for results (default: `reports/`)
- `--json`: Output results as JSON instead of HTML

**Output**: JSON file with comparison results in `reports/` directory.

**Example**:
```bash
# Benchmark against MT5
python scripts/run_benchmark.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_M15_2021_2025.csv \
  --external-trades mt5_trades.csv \
  --platform mt5 \
  --start-date 2021-01-01 \
  --end-date 2023-12-31
```

**See also**: [Validation Guide - Benchmarking](06-validation.md#benchmarking-against-external-platforms)

---

## Getting Help

For detailed usage of each script, use the `--help` flag:
```bash
python scripts/run_backtest.py --help
```

For workflow guidance, see:
- [Complete Workflow Guide](02-complete-workflow.md)
- [Validation Guide](VALIDATION.md)
- [Getting Started Guide](GETTING_STARTED.md)
