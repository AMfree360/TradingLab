# API Reference

Complete API documentation for Trading Lab components.

## Table of Contents

1. [Strategy Base](#strategy-base)
2. [Backtest Engine](#backtest-engine)
3. [Validation](#validation)
4. [Metrics](#metrics)
5. [Filters](#filters)
6. [Configuration](#configuration)

## Strategy Base

### StrategyBase

Abstract base class for all trading strategies.

```python
class StrategyBase(ABC):
    """Abstract base class for all trading strategies."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration."""
    
    @abstractmethod
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals from multi-timeframe data.
        
        Args:
            df_by_tf: Dictionary mapping timeframe strings to DataFrames
        
        Returns:
            DataFrame with columns: direction, entry_price, stop_price
        """
    
    @abstractmethod
    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            tf: Optional timeframe label
        
        Returns:
            DataFrame with indicators added
        """
    
    def get_required_timeframes(self) -> list[str]:
        """Get list of required timeframes."""
    
    def prepare_data(self, df_by_tf: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """Prepare data by computing indicators."""
```

## Backtest Engine

### BacktestEngine

Core backtesting engine.

```python
class BacktestEngine:
    """Core backtesting engine."""
    
    def __init__(
        self,
        strategy: StrategyBase,
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0
    ):
        """Initialize engine."""
    
    def run(self, data: pd.DataFrame) -> BacktestResult:
        """
        Execute backtest on historical data.
        
        Args:
            data: DataFrame with OHLCV data and datetime index
        
        Returns:
            BacktestResult with metrics, trades, and equity curve
        """
    
    def enter_position(
        self,
        direction: str,
        entry_price: float,
        stop_price: float,
        timestamp: pd.Timestamp
    ) -> Optional[Position]:
        """Enter a new position."""
    
    def _close_position(
        self,
        position: Position,
        exit_price: float,
        exit_reason: str,
        timestamp: pd.Timestamp
    ) -> Trade:
        """Close position and create trade record."""
```

### BacktestResult

Result container from backtest execution.

```python
@dataclass
class BacktestResult:
    """Backtest execution result."""
    strategy_name: str
    trades: List[Trade]
    equity_curve: List[float]  # Integer-indexed
    trade_returns: List[float]  # Precomputed returns
    metrics: Dict[str, float]
    initial_capital: float
    final_capital: float
```

### Trade

Individual trade record.

```python
@dataclass
class Trade:
    """Completed trade record."""
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    pnl_after_costs: float
    commission: float
    slippage: float
    exit_reason: str
```

### Position

Open position representation.

```python
@dataclass
class Position:
    """Open trading position."""
    direction: str
    entry_price: float
    stop_price: float
    initial_stop: float
    size: float
    timestamp: pd.Timestamp
    partial_exit_done: bool = False
```

## Validation

### WalkForwardAnalyzer

Walk-forward analysis validator.

```python
class WalkForwardAnalyzer:
    """Walk-forward analysis for OOS validation."""
    
    def __init__(
        self,
        training_period: str = "1 year",
        test_period: str = "6 months",
        window_type: str = "expanding"
    ):
        """Initialize analyzer."""
    
    def analyze(
        self,
        strategy_class: type[StrategyBase],
        strategy_config: StrategyConfig,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.
        
        Returns:
            WalkForwardResult with all steps and summary
        """
```

### TrainingValidator

Phase 1 validation (training data).

```python
class TrainingValidator:
    """Phase 1 validation on training data."""
    
    def validate(
        self,
        strategy_class: type[StrategyBase],
        strategy_config: StrategyConfig,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> TrainingValidationResult:
        """Run Phase 1 validation."""
```

### OOSValidator

Phase 2 validation (out-of-sample data).

```python
class OOSValidator:
    """Phase 2 validation on OOS data."""
    
    def validate(
        self,
        strategy_class: type[StrategyBase],
        strategy_config: StrategyConfig,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> OOSValidationResult:
        """Run Phase 2 validation."""
```

### MonteCarloPermutation

Monte Carlo permutation test.

```python
class MonteCarloPermutation:
    """Monte Carlo permutation test."""
    
    def __init__(self, n_iterations: int = 1000, random_seed: Optional[int] = None):
        """Initialize tester."""
    
    def test(
        self,
        result: BacktestResult,
        metrics: List[str] = ['final_pnl', 'sharpe', 'pf']
    ) -> PermutationResult:
        """
        Run permutation test.
        
        Returns:
            PermutationResult with p-values and distributions
        """

### MonteCarloSuite

Orchestrates the full Monte Carlo suite (permutation, block bootstrap, randomized entry) and computes combined robustness.

Notes:
- Results include suitability-aware skipping and a **dual combined** output:
  - Universal combined (used for Phase 1 gating)
  - All-suitable combined (informational)

```python
class MonteCarloSuite:
    def __init__(
        self,
        seed: int = 42,
        weights: Optional[Dict[str, float]] = None,
        randomized_entry_config: Optional[RandomizedEntryConfig] = None,
    ):
        ...

    def run_conditional(
        self,
        backtest_result: BacktestResult,
        strategy: StrategyBase,
        price_data: pd.DataFrame,
        test_suitability: Dict[str, Any],
        metrics: List[str] = ['final_pnl', 'sharpe_ratio', 'profit_factor'],
        n_iterations: int = 1000,
    ) -> MonteCarloSuiteResult:
        ...
```

`MonteCarloSuiteResult.combined` keys (representative):
- Universal (gating): `robust`, `score`, `percentile`, `p_value`, `normalized_weights`, `n_suitable_tests`
- All suitable (info): `robust_all`, `score_all`, `percentile_all`, `p_value_all`, `normalized_weights_all`, `n_suitable_tests_all`
```

## Metrics

### calculate_enhanced_metrics

Calculate all performance metrics.

```python
def calculate_enhanced_metrics(result: BacktestResult) -> Dict[str, float]:
    """
    Calculate all enhanced metrics from backtest result.
    
    Returns:
        Dictionary of metric names to values
    """
```

### Individual Metric Functions

```python
def calculate_profit_factor(result: BacktestResult) -> float:
    """Calculate profit factor (gross profit / gross loss)."""

def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sharpe ratio."""

def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """Calculate Sortino ratio."""

def calculate_cagr(
    initial_capital: float,
    final_capital: float,
    years: float
) -> float:
    """Calculate Compound Annual Growth Rate."""

def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """Calculate maximum drawdown percentage."""
```

## Filters

### BaseFilter

Abstract base class for filters.

```python
class BaseFilter(ABC):
    """Abstract base class for all filters."""
    
    def __init__(self, config: dict):
        """Initialize filter with configuration."""
    
    @abstractmethod
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if trade should be allowed.
        
        Args:
            context: FilterContext with price data and timestamp
        
        Returns:
            FilterResult with allowed flag and reason
        """
```

### FilterContext

Context passed to filters.

```python
@dataclass
class FilterContext:
    """Context for filter evaluation."""
    timestamp: pd.Timestamp
    price_data: pd.Series  # Current bar data
    market_data: pd.DataFrame  # Historical data up to timestamp
    symbol: Optional[str] = None
```

### FilterResult

Result from filter check.

```python
@dataclass
class FilterResult:
    """Result from filter check."""
    allowed: bool
    reason: str
```

### FilterManager

Manages multiple filters.

```python
class FilterManager:
    """Manages filter execution."""
    
    def __init__(self, master_config: dict, strategy_config: dict):
        """Initialize with master and strategy configs."""
    
    def should_allow_trade(self, context: FilterContext) -> bool:
        """Check if trade should be allowed (all filters must pass)."""
    
    def get_filter_results(self, context: FilterContext) -> Dict[str, FilterResult]:
        """Get results from all filters."""
```

## Configuration

### StrategyConfig

Complete strategy configuration.

```python
class StrategyConfig(BaseModel):
    """Strategy configuration."""
    strategy_name: str
    description: Optional[str] = None
    market: MarketConfig
    timeframes: TimeframeConfig
    risk: RiskConfig
    execution: ExecutionConfig
    backtest: BacktestConfig
    # ... other sections
```

### Loading Configuration

```python
def load_and_validate_strategy_config(
    config_path: str
) -> StrategyConfig:
    """
    Load and validate strategy configuration.
    
    Args:
        config_path: Path to config.yml file
    
    Returns:
        Validated StrategyConfig object
    
    Raises:
        ValidationError: If configuration is invalid
    """
```

### Configuration Schema

See `config/schema.py` for complete schema definition.

## Data Loading

### Universal Data Loader

```python
def load_data(
    file_path: str,
    start_date: Optional[pd.Timestamp] = None,
    end_date: Optional[pd.Timestamp] = None
) -> pd.DataFrame:
    """
    Load data from file (CSV or Parquet).
    
    Args:
        file_path: Path to data file
        start_date: Optional start date filter
        end_date: Optional end date filter
    
    Returns:
        DataFrame with OHLCV data and datetime index
    """
```

## Resampling

### Resample Functions

```python
def resample_ohlcv(
    df: pd.DataFrame,
    target_tf: str
) -> pd.DataFrame:
    """
    Resample OHLCV data to target timeframe.
    
    Args:
        df: DataFrame with OHLCV data
        target_tf: Target timeframe (e.g., '1h', '15m')
    
    Returns:
        Resampled DataFrame
    """

def resample_multiple(
    df: pd.DataFrame,
    timeframes: List[str]
) -> Dict[str, pd.DataFrame]:
    """
    Resample to multiple timeframes.
    
    Returns:
        Dictionary mapping timeframe to DataFrame
    """
```

## Reporting

### ReportGenerator

Generate HTML/JSON reports.

```python
class ReportGenerator:
    """Generate backtest and validation reports."""
    
    def generate_backtest_report(
        self,
        result: BacktestResult,
        output_path: str
    ) -> None:
        """Generate HTML backtest report."""
    
    def generate_validation_report(
        self,
        result: ValidationResult,
        output_path: str
    ) -> None:
        """Generate HTML validation report."""
```

## Type Definitions

### Common Types

```python
from typing import Dict, List, Optional, Union
import pandas as pd

# Common type aliases
TimeframeDict = Dict[str, pd.DataFrame]
MetricDict = Dict[str, float]
ConfigDict = Dict[str, Any]
```

## Error Classes

### Custom Exceptions

```python
class InsufficientDataError(Exception):
    """Raised when data is insufficient."""

class ValidationError(Exception):
    """Raised when validation fails."""

class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
```

## Constants

### Default Values

```python
DEFAULT_INITIAL_CAPITAL = 10000.0
DEFAULT_COMMISSION_RATE = 0.0004
DEFAULT_SLIPPAGE_TICKS = 0.0
MIN_TRADES_FOR_VALIDATION = 30
```

---

For detailed implementation, see source code in respective modules.
