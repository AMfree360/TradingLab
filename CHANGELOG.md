# Changelog

All notable changes to Trading Lab will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Planned
- Live trading execution engine
- Binance API adapter
- Additional exchange adapters
- Advanced reporting features
- Strategy marketplace/templates

### Added - 2026-02-12
- Guided Strategy Builder v2 (GUI): added risk/limits/cost overrides to Advanced (Step 5):
  - Sizing: `sizing_mode` + `account_size`
  - Daily limits: `max_trades_per_day` + optional `max_daily_loss_pct`
  - Backtest costs: optional overrides `commissions` and `slippage_ticks` (market-profile defaults when unset)
- Guided edit: `.guided.json` sidecar now round-trips the new advanced risk fields and edit-guided rehydrates from the spec.
- Research pipeline: `StrategySpec` + compiler carry the fields through to generated strategy config (`risk`, `trade_limits`, and optional `backtest` overrides).
- Advanced defaults API: `/api/advanced-defaults` now exposes commission hints (`commission_rate`, `commission_per_contract`) for market-aware UI guidance.

### Added - Recent
- Canonical `ema_crossover` example strategy (EMA 5/100) with ATR stop option and relaxed configs
- ATR stop config keys: `stop_loss.atr_enabled`, `stop_loss.atr_length`, `stop_loss.atr_multiplier`
- Fix: `reports/report_generator.py` syntax error and more robust HTML generation for backtest/validation
- Fix: Monte Carlo randomized-entry watchdog and runner edge-case handling
- Docs: Updated validation guide with recent changes and notes about config keys

### Added - Reproducibility & Governance
- Dataset manifests with stable hashing and JSON artifacts written under `data/manifests/`
- Phase dataset locks (Phase 1/Phase 2/holdout): refuse to proceed if manifest changes mid-phase
- Experiment registry (SQLite) with run recording and CLI compare/list (`scripts/runs.py`)
- Resume artifacts: `SESSION_NOTES.md` (snapshot + invariants) and `RUNBOOK.md` (copy/paste commands)

### Changed
- Phase 2 OOS semantics: OOS is consumed on attempt (state is written before validation runs)
- Walkforward warmup data is bounded to train-start → oos-end to prevent holdout leakage

## [0.2.0] - 2024-12-09

### Added - Phased Validation System
- **Phase 1: Training Validation** - Validates strategy on training data only
  - Monte Carlo Permutation tests to prove edge > random
  - Parameter Sensitivity analysis for robustness
  - Training data quality checks
  - Prevents wasting time on strategies without real edge
- **Phase 2: Out-of-Sample (OOS) Validation** - One-time use of unseen data
  - Walk-Forward Analysis on OOS data
  - Consistency scoring across periods
  - OOS data protection (can only be used once)
- **Phase 3: Stationarity Analysis** - Post-live monitoring
  - Edge erosion detection
  - Recommended retraining frequency
  - Performance stability analysis
- Validation state tracking to prevent OOS data reuse
- **Smart Phase 1 re-run logic**: 
  - Allows re-running Phase 1 if it failed (reoptimization is part of development)
  - Prevents re-running Phase 1 if it passed (prevents data mining)
  - Phase 2 remains strict (OOS data can only be used once)
- Configurable pass/fail criteria via `config/validation_criteria.yml`
- New CLI scripts:
  - `scripts/run_training_validation.py` - Phase 1 validation
  - `scripts/run_oos_validation.py` - Phase 2 validation
  - `scripts/run_stationarity.py` - Phase 3 analysis

### Added - Market-Agnostic Strategy Design
- Market profiles system (`config/market_profiles.yml`)
- Automatic market profile application via `--market` flag
- Market-specific settings: exchange, market_type, leverage, commissions, slippage
- Strategies work seamlessly across crypto, forex, and stocks
- Market loader module for profile management

### Added - Enhanced Data Handling
- Universal data loader supporting multiple formats
  - CSV files (various separators: comma, tab, semicolon)
  - Parquet files (including transposed format)
  - Automatic column name detection and mapping
  - Handles DATE/TIME columns, angle brackets, case variations
  - Automatic timestamp detection and parsing
- Data consolidation script (`scripts/consolidate_data.py`)
  - Combines multiple yearly data files into single file
  - Removes duplicates and sorts chronologically
  - Supports wildcard patterns

### Added - Advanced Execution Features
- **Flatten Time**: Close all positions at specified time (e.g., 9:30 PM GMT)
  - Prevents holding positions overnight
  - Configurable via `execution.flatten_time`
- **Max Wait Bars**: Skip trades if entry condition not met within N bars
  - Prevents stale signals from executing
  - Configurable via `execution.max_wait_bars`
- **MA-based Stop Loss**: Dynamic stop loss based on moving averages
  - Supports SMA/EMA with configurable length
  - Buffer in pips or price units
  - Configurable via `stop_loss.buffer_pips` and `stop_loss.buffer_unit`
- **Session Filtering**: Individual session toggles
  - Each trading session can be enabled/disabled independently
  - Dictionary format: `{enabled: true/false, start: "HH:MM", end: "HH:MM"}`

### Added - Date Range Filtering
- `--start-date` and `--end-date` arguments for backtest and validation scripts
- Allows subsetting data for training/testing periods
- Enables proper train/test split for validation workflow
- Improved error messages for date range mismatches

### Fixed - Capital Tracking and Metrics
- **Capital Mismatch Detection**: Validates `current_capital` vs. calculated from trade P&Ls
  - Automatically rebuilds equity curve from accurate trade P&Ls when mismatch detected
  - Ensures final metrics are accurate even if internal tracking has bugs
- **Max Drawdown Calculation**: Industry-standard formula
  - Formula: `(Peak - Trough) / Peak * 100%`
  - Uses running maximum (peak-to-trough method)
  - Capped at 100% with validation warnings
- **CAGR Calculation**: Improved accuracy
  - Uses actual time between first and last trade
  - Validates sign matches total return
  - Warns on unrealistic values (>500% with short time spans)
  - Capped at 1000% for display
- **Sortino Ratio**: Enhanced calculation
  - Uses equity curve returns (period-based) instead of trade returns
  - Calculates `periods_per_year` from actual data frequency
  - Shows uncapped value in warnings when capped at 20.0
  - Better handling of edge cases (no downside returns, near-zero deviation)

### Fixed - Report Generation
- **Validation Reports**: Complete rewrite to handle phased validation structure
  - Supports `phase1_training`, `phase2_oos`, `phase3_stationarity`
  - Displays all sections: backtest, Monte Carlo, sensitivity, criteria checks, walk-forward
  - Color-coded pass/fail status
  - Failure reasons display
- **JSON Serialization**: Fixed validation state saving
  - Converts numpy types to native Python types
  - Handles nested dictionaries and lists
  - Prevents "Object of type bool is not JSON serializable" errors

### Enhanced - Documentation
- **New Workflow Guide** (`docs/WORKFLOW.md`): Complete step-by-step workflow
  - Phase 1: Development (data, strategy creation, backtesting, optimization)
  - Phase 2: Validation (training validation, OOS validation, stationarity)
  - Phase 3: Live Trading (deployment, monitoring, maintenance)
  - Decision points and gates at each phase
  - Common workflow patterns and troubleshooting
- **Updated Validation Guide** (`docs/VALIDATION.md`): Reflects new phased methodology
- **Updated Architecture Guide** (`docs/ARCHITECTURE.md`): Added debug scripts and market-agnostic design
- **Updated Data Management Guide** (`docs/DATA_MANAGEMENT.md`): Added data consolidation section
- **Updated Backtesting Guide** (`docs/BACKTESTING.md`): Added date range filtering section
- **Updated README** (`docs/README.md`): Added workflow guide to index

### Technical Improvements
- Debug scripts organized into `scripts/debug/` subdirectory
- Improved error messages throughout
- Better handling of edge cases in calculations
- Enhanced logging and warnings for suspicious values
- Type safety improvements

### Breaking Changes
- None - all changes are backward compatible

### Migration Notes
- Validation workflow has changed - see `docs/VALIDATION.md` for new methodology
- Market profiles are optional - existing strategies continue to work
- Date range filtering is optional - existing commands work without it

## [0.1.0] - 2024-01-XX

### Added - Phase 1: Core Infrastructure
- Core backtesting engine with position management
- Strategy base class interface
- Multi-timeframe data resampling
- CSV data loader
- Configuration system with YAML support
- Basic metrics calculation (PF, Sharpe, win rate, etc.)
- Trade tracking and equity curve generation
- Commission and slippage modeling

### Added - Phase 2: Trailing Stops & Partial Exits
- Trailing stop implementation with SMA/EMA support
- R-multiple based trailing stop activation
- Partial exit functionality at specified R-multiple levels
- Position class for tracking open positions
- Enhanced trade tracking with partial exits
- Support for multiple concurrent positions

### Added - Phase 3: Validation & Testing
- Walk-forward analysis with expanding and rolling windows
- Monte Carlo permutation tests for trade sequences
- Enhanced metrics module (Sortino, CAGR, Calmar, Kelly, etc.)
- Validation runner for orchestrating tests
- Stationarity detection for retraining frequency
- Parameter sensitivity analysis

### Added - Phase 4: Reporting & Analysis
- HTML report generation with interactive charts
- JSON export for programmatic access
- Backtest reports with comprehensive metrics
- Validation reports with walk-forward and Monte Carlo results
- Command-line scripts for backtesting (`run_backtest.py`)
- Command-line scripts for validation (`run_validation.py`)
- Comprehensive documentation in `/docs`

### Documentation
- Getting Started Guide
- Strategy Development Guide
- Backtesting Guide
- Validation Guide
- Reports and Interpretation Guide
- Optimization Guide
- Configuration Guide
- Live Trading Guide
- Data Management Guide
- Documentation index and navigation

### Technical
- Pydantic-based configuration validation
- Type hints throughout codebase
- Modular architecture for extensibility
- Error handling and validation
- Progress bars for long-running operations (tqdm)

## [0.0.1] - Initial Release

### Added
- Project structure
- Basic README
- Initial project setup

---

## Version History

### Phase 1 (v0.1.0-alpha)
- Core backtesting functionality
- Basic strategy interface
- Configuration system

### Phase 2 (v0.1.0-beta)
- Trailing stops
- Partial exits
- Enhanced position management

### Phase 3 (v0.1.0-rc1)
- Validation framework
- Enhanced metrics
- Testing tools

### Phase 4 (v0.1.0)
- Reporting system
- Documentation
- Command-line tools
- Ready for use

---

## Upgrade Notes

### From Phase 1 to Phase 2
- No breaking changes
- New configuration options for trailing stops and partial exits
- Enhanced Trade dataclass with partial_exits field

### From Phase 2 to Phase 3
- No breaking changes
- New validation modules available
- Enhanced metrics can be calculated from existing BacktestResult objects

### From Phase 3 to Phase 4
- No breaking changes
- New reporting modules available
- Scripts added for easier command-line usage

---

## Known Issues

- Short position capital tracking is simplified (margin requirements not fully modeled)
- Multiple partial exits per position not yet supported
- Live trading execution not yet implemented

---

## Future Roadmap

### Phase 5: Live Trading (Planned)
- Binance API adapter
- Paper trading mode
- Real-time execution engine
- Order management
- Position monitoring

### Phase 6: Advanced Features (Planned)
- Additional exchange adapters
- Portfolio management
- Multi-strategy execution
- Advanced risk management
- Performance attribution

### Phase 7: User Interface (Planned)
- Web dashboard
- Strategy builder UI
- Real-time monitoring
- Report viewer

---

## Contributing

See the main README for contribution guidelines.

## License

MIT License - see LICENSE file for details.

