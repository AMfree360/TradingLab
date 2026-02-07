# Trading Lab Architecture Documentation

This document describes the architecture, design decisions, and technical implementation of Trading Lab. It's intended for developers and engineers who will work on or extend the system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture Diagram](#architecture-diagram)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Design Patterns](#design-patterns)
6. [Extension Points](#extension-points)
7. [Technical Decisions](#technical-decisions)
8. [API Reference](#api-reference)
9. [Development Guidelines](#development-guidelines)

## System Overview

Trading Lab is a modular, strategy-agnostic backtesting and validation platform built in Python. It follows a layered architecture with clear separation of concerns.

### Key Principles

- **Strategy-Agnostic**: Strategies implement a simple interface, engine doesn't know strategy details
- **Modular**: Components are independent and can be extended/replaced
- **Config-Driven**: Behavior controlled via YAML configuration files
- **Reproducible**: All runs are deterministic and traceable
- **Extensible**: Easy to add new strategies, metrics, validators, adapters

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│  (CLI Scripts: run_backtest.py, run_validation.py)          │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Strategy Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ StrategyBase │  │ MAAlignment   │  │ YourStrategy │      │
│  │  (Interface) │  │ (Example)     │  │  (Custom)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Engine Layer                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Backtest     │  │ Resampler    │  │ Position     │      │
│  │ Engine       │  │              │  │ Manager      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                  Validation Layer                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Walk-Forward │  │ Monte Carlo  │  │ Stationarity │      │
│  │ Analyzer     │  │ Permutation  │  │ Analyzer     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                   Adapter Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ CSV Loader   │  │ Binance API  │  │ Future       │      │
│  │              │  │ (Planned)    │  │ Adapters     │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└──────────────────────┬──────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────┐
│                    Data Layer                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Raw Data     │  │ Processed    │  │ Manifests    │      │
│  │ (CSV/Parquet)│  │ Data         │  │ (Metadata)   │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│              Supporting Components                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Metrics      │  │ Reports      │  │ Config       │      │
│  │ Calculator   │  │ Generator    │  │ Schema       │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Strategy Layer

**Location**: `strategies/`

**Purpose**: Defines trading logic

**Key Classes**:
- `StrategyBase` (abstract base class)
  - `generate_signals()`: Core method that produces trading signals
  - `get_indicators()`: Calculates technical indicators
  - `get_required_timeframes()`: Returns needed timeframes

**Design Pattern**: Strategy Pattern - algorithms (strategies) are interchangeable

**Extension**: Create new strategy by:
1. Inheriting from `StrategyBase`
2. Implementing required methods
3. Creating `config.yml` file

### 2. Engine Layer

**Location**: `engine/`

**Purpose**: Executes backtests and manages positions

#### BacktestEngine

**File**: `engine/backtest_engine.py`

**Responsibilities**:
- Load and prepare data
- Execute trades based on signals
- Manage positions (entry, exit, trailing stops, partial exits)
- Track equity curve and trade returns (Monte Carlo compatible)
- Calculate basic metrics

**Key Methods**:
- `run()`: Main entry point for backtesting
- `enter_position()`: Opens new position
- `_update_trailing_stop()`: Updates trailing stops
- `_check_partial_exit()`: Handles partial exits
- `_close_position()`: Closes position and creates Trade record
- `_create_result()`: Creates BacktestResult with integer-indexed equity curve

**State Management**:
- `account`: AccountState object (cash, equity, positions)
- `open_positions`: List of active Position objects
- `trades`: List of completed Trade objects
- `equity_list`: Equity after each trade (for integer-indexed equity curve)
- `trade_returns_list`: Precomputed trade returns (for MC compatibility)

**Monte Carlo Compatibility**:
- Produces integer-indexed equity curves (0..N, not time-indexed)
- Precomputes trade returns during execution
- Trade objects include `pnl_after_costs` field
- All data structures compatible with MC validation tests

#### Resampler

**File**: `engine/resampler.py`

**Purpose**: Converts data between timeframes

**Key Functions**:
- `resample_ohlcv()`: Resamples single timeframe
- `resample_multiple()`: Resamples to multiple timeframes

**Algorithm**: Uses pandas resampling with OHLCV aggregation rules

### 3. Validation Layer

**Location**: `validation/`

**Purpose**: Rigorous testing of strategies

#### WalkForwardAnalyzer

**File**: `validation/walkforward.py`

**Purpose**: Tests strategy on out-of-sample data

**Algorithm**:
1. Split data into training/test windows
2. Run backtest on training data
3. Run backtest on test data (unseen)
4. Repeat with expanding/rolling windows
5. Aggregate results

**Key Classes**:
- `WalkForwardAnalyzer`: Main analyzer
- `WalkForwardResult`: Contains all steps and summary
- `WalkForwardStep`: Single train/test period result

#### ValidationSuitabilityAssessor

**File**: `validation/suitability.py`

**Purpose**: Determines which validation tests are suitable for a given strategy

**Algorithm**:
1. Assess strategy characteristics (return CV, exit uniformity, sample size, etc.)
2. Run quick permutation test (20 iterations) to check if order matters
3. Determine suitability for each MC test based on characteristics
4. Return suitability information with reasons and alternatives

**Key Classes**:
- `ValidationSuitabilityAssessor`: Main assessor
- `StrategyProfile`: Strategy characteristics (return_cv, exit_uniformity, n_trades, etc.)
- `TestSuitability`: Suitability information for each test

**Key Features**:
- Prevents false negatives by skipping inappropriate tests
- Quick permutation test checks if order actually matters (final_equity_cv)
- Provides clear reasons and alternatives for skipped tests
- Industry best practice: adaptive test selection

#### MonteCarloPermutation

**File**: `validation/monte_carlo/permutation.py`

**Purpose**: Tests if trade order matters (temporal skill vs. luck)

**Algorithm**:
1. Extract precomputed trade_returns from BacktestResult
2. Shuffle returns (not P&Ls) using seeded RNG
3. Apply returns with compounding: `equity_{t+1} = equity_t * (1 + return_perm[t])`
4. Rebuild integer-indexed equity curve
5. Recompute metrics using same pipeline
6. Calculate p-values and percentiles

**Key Features**:
- Uses return-based compounding (not P&L shuffling)
- Works with integer-indexed equity curves
- Uses precomputed trade_returns from engine
- Produces varying distributions (no zero variance)
- Stores full equity curves (if N ≤ 1000) for probability cone visualization

**Key Classes**:
- `MonteCarloPermutation`: Main tester
- `PermutationResult`: Contains p-values, distributions, and equity curves

#### StationarityAnalyzer

**File**: `validation/stationarity.py`

**Purpose**: Determines optimal retraining frequency

**Algorithm**:
1. Train on initial period
2. Test on 1-day OOS windows
3. Test on 2-day, 3-day, ... N-day windows
4. Find when performance degrades
5. Recommend retrain frequency

### 4. Adapter Layer

**Location**: `adapters/`

**Purpose**: Interface with external data sources and exchanges

#### CSVDataLoader

**File**: `adapters/data/csv_loader.py`

**Purpose**: Loads historical data from CSV files

**Future Adapters**:
- Binance API adapter (live data)
- Other exchange adapters
- Database adapters

**Design**: Adapter Pattern - uniform interface for different data sources

### 5. Metrics Layer

**Location**: `metrics/`

**Purpose**: Calculate performance metrics

**File**: `metrics/metrics.py`

**Key Functions**:
- `calculate_enhanced_metrics()`: All metrics from BacktestResult
- `calculate_sortino_ratio()`: Downside risk-adjusted return
- `calculate_cagr()`: Compound annual growth rate
- `calculate_recovery_factor()`: P&L / max drawdown
- And more...

**Design**: Stateless functions - pure calculations

### 6. Reporting Layer

**Location**: `reports/`

**Purpose**: Generate human-readable reports

**File**: `reports/report_generator.py`

**Key Class**: `ReportGenerator`

**Output Formats**:
- HTML (interactive charts with Plotly)
- JSON (machine-readable)

**Features**:
- Backtest reports
- Validation reports
- Interactive equity curves
- Metric dashboards

### 7. Configuration Layer

**Location**: `config/`

**Purpose**: Validate and manage configuration

**File**: `config/schema.py`

**Key Classes**:
- `StrategyConfig`: Complete strategy configuration
- `WalkForwardConfig`: Walk-forward analysis settings
- Various config classes for different components

**Design**: Pydantic models for validation and type safety

## Data Flow

### Backtest Flow

```
1. User runs script with strategy + data
   ↓
2. Load strategy config (YAML → Pydantic model)
   ↓
3. Load data (CSV → DataFrame)
   ↓
4. Create strategy instance
   ↓
5. BacktestEngine.run():
   a. Prepare multi-timeframe data (resample)
   b. Strategy.generate_signals() → signals DataFrame
   c. For each bar:
      - Check for new signals → enter_position()
      - Update open positions (trailing stops, partial exits)
      - Check for exits → _close_position()
   d. Calculate metrics
   ↓
6. Generate BacktestResult
   ↓
7. Calculate enhanced metrics
   ↓
8. Generate report (HTML + JSON)
```

### Validation Flow

```
1. User runs validation script
   ↓
2. ValidationRunner.validate_strategy():
   a. Walk-Forward Analysis:
      - Split data into windows
      - Run backtest on each train/test pair
      - Aggregate results
   b. Strategy Suitability Assessment:
      - Assess strategy characteristics
      - Determine which MC tests are suitable
   c. Monte Carlo Suite (Conditional):
      - Run only suitable tests
      - Permutation: Shuffle returns and recalculate (if suitable)
      - Bootstrap: Block bootstrap with synthetic returns (if suitable)
      - Randomized Entry: Random entries with same risk management (always suitable)
      - Calculate p-values and percentiles
   d. Enhanced Metrics:
      - Calculate all metrics
   ↓
3. Generate validation report (with suitability assessment and conditional test results)
```

## Design Patterns

### 1. Strategy Pattern

**Where**: Strategy layer

**Purpose**: Make trading algorithms interchangeable

**Implementation**: `StrategyBase` abstract class, concrete strategies inherit

**Benefits**: Easy to add new strategies without modifying engine

### 2. Adapter Pattern

**Where**: Adapter layer

**Purpose**: Uniform interface for different data sources

**Implementation**: `CSVDataLoader`, future `BinanceAdapter`, etc.

**Benefits**: Engine doesn't care about data source

### 3. Template Method Pattern

**Where**: BacktestEngine

**Purpose**: Define algorithm skeleton, let strategies fill details

**Implementation**: Engine calls `generate_signals()`, strategy implements logic

**Benefits**: Consistent execution flow, flexible strategy logic

### 4. Factory Pattern

**Where**: Strategy loading

**Purpose**: Create strategy instances from config

**Implementation**: Scripts dynamically load strategy classes

**Benefits**: No hardcoded strategy references

### 5. Observer Pattern (Future)

**Where**: Live trading (planned)

**Purpose**: Notify components of trade events

**Implementation**: Event system for trade execution

## Extension Points

### Adding a New Strategy

1. **Create folder**: `strategies/your_strategy/`
2. **Create `config.yml`**: Define parameters
3. **Create `strategy.py`**: Implement `StrategyBase`
4. **Implement methods**:
   - `get_indicators()`: Calculate indicators
   - `generate_signals()`: Generate trading signals
5. **Test**: Run backtest script

**Example**: See `strategies/ema_crossover/` for reference

### Adding a New Metric

1. **Add function**: `metrics/metrics.py`
2. **Add to `calculate_enhanced_metrics()`**: Include in aggregation
3. **Update reports**: Add to report templates (optional)

**Example**:
```python
def calculate_new_metric(result: BacktestResult) -> float:
    # Your calculation
    return value
```

### Adding a New Validator

1. **Create class**: `validation/your_validator.py`
2. **Implement interface**: Similar to `WalkForwardAnalyzer`
3. **Add to ValidationRunner**: Integrate into validation flow
4. **Update reports**: Add results to validation reports

### Adding a New Data Adapter

1. **Create class**: `adapters/data/your_adapter.py`
2. **Implement interface**: Similar to `CSVDataLoader`
3. **Update engine**: Use new adapter for data loading

**Interface**:
```python
class YourDataLoader:
    def load(self, source) -> pd.DataFrame:
        # Load and return DataFrame with OHLCV columns
        pass
```

### Adding a New Exchange Adapter (Live Trading)

1. **Create class**: `adapters/execution/your_exchange.py`
2. **Implement methods**:
   - `place_order()`
   - `get_position()`
   - `cancel_order()`
3. **Integrate**: Connect to live trading engine (when implemented)

## Technical Decisions

### Why Pydantic for Configuration?

**Decision**: Use Pydantic models for configuration validation

**Rationale**:
- Type safety
- Automatic validation
- Clear error messages
- IDE support
- Easy to extend

**Alternatives Considered**: Plain dicts (less safe), dataclasses (no validation)

### Why Separate Strategy and Engine?

**Decision**: Strategy logic separate from execution engine

**Rationale**:
- Strategy-agnostic engine
- Easy to test strategies independently
- Clear separation of concerns
- Multiple strategies can use same engine

**Trade-offs**: Slight overhead from interface, but worth it for flexibility

### Why Bar-by-Bar Execution?

**Decision**: Process each bar sequentially, not vectorized

**Rationale**:
- More realistic (can't look ahead)
- Supports complex position management
- Easier to debug
- Supports trailing stops and partial exits

**Trade-offs**: Slower than vectorized, but more accurate

### Why YAML for Configuration?

**Decision**: YAML files for strategy configuration

**Rationale**:
- Human-readable
- Easy to edit
- Supports comments
- Standard format
- Version control friendly

**Alternatives Considered**: JSON (no comments), Python files (too complex)

### Why Dataclasses for Results?

**Decision**: Use dataclasses for Trade, Position, BacktestResult

**Rationale**:
- Type hints
- Immutable (frozen dataclasses)
- Easy to serialize
- Clear structure
- Less boilerplate than classes

## API Reference

### StrategyBase Interface

```python
class StrategyBase(ABC):
    @abstractmethod
    def generate_signals(
        self, 
        df_by_tf: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df_by_tf: Dict mapping timeframe to DataFrame with indicators
        
        Returns:
            DataFrame with columns: direction, entry_price, stop_price
            Index: timestamp
        """
        pass
    
    @abstractmethod
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with added indicator columns
        """
        pass
```

### BacktestEngine API

```python
class BacktestEngine:
    def __init__(
        self,
        strategy: StrategyBase,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0
    )
    
    def run(self, data_path: Path) -> BacktestResult:
        """Run backtest and return results."""
        pass
    
    def load_data(self, data_path: Path) -> pd.DataFrame:
        """Load data from file."""
        pass
    
    def prepare_data(self, df_base: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Prepare multi-timeframe data."""
        pass
```

### ValidationRunner API

```python
class ValidationRunner:
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0
    )
    
    def validate_strategy(
        self,
        data: pd.DataFrame,
        strategy_config: Dict,
        wf_config: Optional[WalkForwardConfig] = None,
        run_monte_carlo: bool = True,
        monte_carlo_iterations: int = 1000
    ) -> Dict:
        """Run complete validation suite."""
        pass
```

## Development Guidelines

### Code Style

- **Type Hints**: Use type hints everywhere
- **Docstrings**: All public methods/classes need docstrings
- **Naming**: 
  - Classes: PascalCase
  - Functions: snake_case
  - Constants: UPPER_SNAKE_CASE
- **Line Length**: 100 characters (Black formatter)

### Testing

- **Unit Tests**: Test individual functions/classes
- **Integration Tests**: Test component interactions
- **Location**: `tests/` directory
- **Naming**: `test_*.py` files, `test_*` functions

### Adding Features

1. **Design First**: Document design decisions
2. **Implement**: Follow existing patterns
3. **Test**: Add tests for new code
4. **Document**: Update relevant docs
5. **Validate**: Run existing tests

### Error Handling

- **Validation**: Validate inputs early
- **Clear Messages**: Error messages should be helpful
- **Logging**: Use logging for important events
- **Exceptions**: Use appropriate exception types

### Debugging Tools

The `scripts/debug/` directory contains diagnostic scripts for troubleshooting:

- **`debug_strategy.py`**: Analyzes why a strategy isn't generating signals
  - Shows indicator values
  - Checks alignment conditions
  - Displays signal generation statistics
  
- **`debug_capital.py`**: Traces capital calculations to find calculation errors
  - Shows capital before/after each trade
  - Tracks position entry/exit
  - Helps identify compounding or calculation bugs
  
- **`inspect_data.py`**: Inspects data file structure and format
  - Shows raw file structure
  - Identifies timestamp formats
  - Checks column names and types
  
- **`inspect_parquet.py`**: Detailed Parquet file inspection
  - Analyzes Parquet-specific structure
  - Checks for transposed data
  - Identifies timestamp column issues

**Usage**: When encountering issues, run the appropriate debug script to diagnose the problem before modifying code.

### Performance Considerations

- **Data Loading**: Use Parquet for large datasets
- **Vectorization**: Use pandas/numpy where possible
- **Caching**: Cache expensive calculations
- **Profiling**: Profile before optimizing

### Security (Live Trading)

- **API Keys**: Never commit to git
- **Permissions**: Minimum required permissions
- **Validation**: Validate all inputs
- **Rate Limiting**: Respect exchange limits

## File Structure

```
trading-lab/
├── adapters/          # External interfaces
│   ├── data/         # Data loaders
│   └── execution/    # Exchange adapters (future)
├── config/           # Configuration management
│   ├── schema.py     # Pydantic models
│   └── defaults.yml  # Default values
├── docs/             # Documentation
├── engine/           # Core backtesting engine
│   ├── backtest.py   # Main engine
│   └── resampler.py  # Timeframe conversion
├── metrics/          # Performance metrics
│   └── metrics.py    # Metric calculations
├── reports/          # Report generation
│   └── report_generator.py
├── scripts/          # CLI tools
│   ├── run_backtest.py
│   ├── run_validation.py
│   ├── download_data.py
│   └── debug/        # Debug and diagnostic scripts
│       ├── debug_strategy.py    # Debug strategy signal generation
│       ├── debug_capital.py     # Debug capital calculations
│       ├── inspect_data.py      # Inspect data file structure
│       └── inspect_parquet.py   # Inspect Parquet file format
├── strategies/       # Trading strategies
│   ├── base/         # Base classes
│   └── ema_crossover/ # Example strategy
├── tests/            # Unit tests
├── validation/       # Validation tests
│   ├── walkforward.py
│   ├── monte_carlo.py
│   ├── stationarity.py
│   ├── sensitivity.py
│   └── runner.py
└── data/             # Data storage
    ├── raw/          # Raw data files
    ├── processed/    # Processed data
    └── manifests/    # Data metadata
```

## Future Architecture Considerations

### Planned Components

1. **Live Trading Engine**
   - Order management
   - Position tracking
   - Risk limits
   - Event system

2. **Portfolio Management**
   - Multiple strategies
   - Capital allocation
   - Risk aggregation

3. **Database Layer**
   - Store trades
   - Store results
   - Query interface

4. **Web Interface**
   - Dashboard
   - Strategy builder
   - Real-time monitoring

### Scalability

- **Current**: Single-threaded, suitable for single strategies
- **Future**: May need parallelization for:
  - Multiple strategies
  - Large parameter grids
  - Real-time data processing

### Extensibility

The architecture is designed for extension:
- New strategies: Just implement interface
- New metrics: Add functions
- New validators: Add classes
- New adapters: Implement interface

## Contributing

When contributing:

1. **Follow Architecture**: Don't break separation of concerns
2. **Add Tests**: New code needs tests
3. **Update Docs**: Update relevant documentation
4. **Type Hints**: Always use type hints
5. **Review**: Code should be reviewed before merge

## Summary

Trading Lab uses a modular, layered architecture:
- **Strategy Layer**: Trading logic (extensible)
- **Engine Layer**: Execution (strategy-agnostic)
- **Validation Layer**: Testing (comprehensive)
- **Adapter Layer**: External interfaces (pluggable)
- **Supporting**: Metrics, reports, config

Key design principles:
- Separation of concerns
- Strategy pattern for flexibility
- Configuration-driven behavior
- Type safety with Pydantic
- Extensibility at every layer

This architecture supports:
- Easy strategy development
- Rigorous validation
- Future live trading
- Continuous extension

For questions or clarifications, refer to code comments or open an issue.

