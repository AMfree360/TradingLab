# Extension Points

This guide explains how to extend Trading Lab by adding new strategies, metrics, validators, filters, and adapters.

## Table of Contents

1. [Adding a New Strategy](#adding-a-new-strategy)
2. [Adding a New Metric](#adding-a-new-metric)
3. [Adding a New Validator](#adding-a-new-validator)
4. [Adding a New Filter](#adding-a-new-filter)
5. [Adding a New Data Adapter](#adding-a-new-data-adapter)
6. [Adding a New Report Format](#adding-a-new-report-format)

## Adding a New Strategy

### Overview

Strategies implement the `StrategyBase` interface and define trading logic. The engine calls your strategy methods to generate signals.

### Step-by-Step Guide

#### Step 1: Create Strategy Folder

```bash
mkdir strategies/my_new_strategy
cd strategies/my_new_strategy
```

#### Step 2: Create Configuration File

Create `config.yml`:

```yaml
strategy_name: my_new_strategy
description: "My custom trading strategy"

market:
  exchange: binance
  symbol: BTCUSDT
  base_timeframe: 1m

timeframes:
  signal_tf: "1h"
  entry_tf: "15m"

# Your custom parameters
my_custom_param: 42
my_custom_threshold: 0.5

risk:
  sizing_mode: equity
  risk_per_trade_pct: 1.0
  account_size: 10000.0

# ... other standard config sections
```

#### Step 3: Implement Strategy Class

Create `strategy.py`:

```python
"""My custom trading strategy."""

import pandas as pd
import numpy as np
from typing import Dict, Optional
from strategies.base import StrategyBase
from config.schema import StrategyConfig


class MyNewStrategy(StrategyBase):
    """Custom strategy implementation."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy."""
        super().__init__(config)
        # Access custom config values
        self.custom_param = config.extra.get('my_custom_param', 42)
        self.custom_threshold = config.extra.get('my_custom_threshold', 0.5)
    
    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        Args:
            df: DataFrame with OHLCV data
            tf: Optional timeframe label
        
        Returns:
            DataFrame with indicators added
        """
        # Calculate your indicators
        df = df.copy()
        
        # Example: Simple moving average
        df['sma_20'] = df['close'].rolling(window=20).mean()
        
        # Example: RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Add your custom indicators here
        # ...
        
        return df
    
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals.
        
        Args:
            df_by_tf: Dictionary of timeframes to DataFrames
        
        Returns:
            DataFrame with signals (columns: direction, entry_price, stop_price)
        """
        # Get timeframes
        signal_tf = self.config.timeframes.signal_tf
        entry_tf = self.config.timeframes.entry_tf
        
        # Get dataframes
        df_signal = df_by_tf[signal_tf].copy()
        df_entry = df_by_tf[entry_tf].copy()
        
        # Calculate indicators
        df_signal = self.get_indicators(df_signal, signal_tf)
        df_entry = self.get_indicators(df_entry, entry_tf)
        
        signals = []
        
        # Your trading logic here
        for timestamp, row in df_signal.iterrows():
            # Example: Long signal when RSI < 30 and price > SMA
            if (row['rsi'] < 30 and 
                row['close'] > row['sma_20'] and
                not pd.isna(row['rsi']) and 
                not pd.isna(row['sma_20'])):
                
                # Get entry price from entry timeframe
                entry_row = df_entry.loc[df_entry.index <= timestamp]
                if len(entry_row) > 0:
                    entry_row = entry_row.iloc[-1]
                    
                    # Calculate stop loss (example: 2% below entry)
                    stop_price = entry_row['close'] * 0.98
                    
                    signals.append({
                        'timestamp': timestamp,
                        'direction': 'long',
                        'entry_price': entry_row['close'],
                        'stop_price': stop_price,
                    })
        
        # Convert to DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df.set_index('timestamp', inplace=True)
            return signals_df
        else:
            return self._create_signal_dataframe()
```

#### Step 4: Test Your Strategy

```bash
# Run a backtest
python scripts/run_backtest.py \
  --strategy my_new_strategy \
  --data data/raw/BTCUSDT_1m.csv \
  --market BTCUSDT
```

### Key Points

- **Inherit from `StrategyBase`**: Provides common functionality
- **Implement `get_indicators()`**: Calculate all indicators
- **Implement `generate_signals()`**: Return DataFrame with signals
- **Signal format**: Must have `direction`, `entry_price`, `stop_price`
- **Use `self.config`**: Access configuration values
- **Handle missing data**: Check for NaN values before using indicators

### Advanced: Using Filters

If your strategy uses filters:

```python
def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    # ... calculate indicators ...
    
    for timestamp, row in df_signal.iterrows():
        # Create filter context
        context = FilterContext(
            timestamp=timestamp,
            price_data=row,
            market_data=df_signal.loc[:timestamp]
        )
        
        # Check filters
        if self.filter_manager.should_allow_trade(context):
            # Generate signal
            # ...
```

## Adding a New Metric

### Overview

Metrics are stateless functions that calculate performance measures from backtest results.

### Step-by-Step Guide

#### Step 1: Add Metric Function

Edit `metrics/metrics.py`:

```python
def calculate_my_custom_metric(result: BacktestResult) -> float:
    """
    Calculate my custom metric.
    
    Args:
        result: BacktestResult with trades and equity curve
    
    Returns:
        Metric value
    """
    if len(result.trades) == 0:
        return 0.0
    
    # Your calculation here
    # Example: Average win/loss ratio
    winning_trades = [t for t in result.trades if t.pnl_after_costs > 0]
    losing_trades = [t for t in result.trades if t.pnl_after_costs < 0]
    
    if len(losing_trades) == 0:
        return float('inf') if len(winning_trades) > 0 else 0.0
    
    avg_win = sum(t.pnl_after_costs for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = abs(sum(t.pnl_after_costs for t in losing_trades) / len(losing_trades))
    
    return avg_win / avg_loss if avg_loss > 0 else 0.0
```

#### Step 2: Add to Enhanced Metrics

Edit `calculate_enhanced_metrics()` in `metrics/metrics.py`:

```python
def calculate_enhanced_metrics(result: BacktestResult) -> Dict[str, float]:
    """Calculate all enhanced metrics."""
    metrics = {
        # ... existing metrics ...
        'my_custom_metric': calculate_my_custom_metric(result),
    }
    return metrics
```

#### Step 3: Update Reports (Optional)

Edit report templates to display your metric:

```python
# In reports/report_generator.py
def _generate_metrics_table(self, metrics: Dict[str, float]) -> str:
    # Add your metric to the table
    html = f"""
    <tr>
        <td>My Custom Metric</td>
        <td>{metrics.get('my_custom_metric', 0.0):.2f}</td>
    </tr>
    """
    return html
```

### Key Points

- **Stateless**: Metrics should be pure functions
- **Handle edge cases**: Empty trades, division by zero, etc.
- **Use BacktestResult**: Access trades, equity curve, etc.
- **Return float**: All metrics return numeric values

## Adding a New Validator

### Overview

Validators test strategies for robustness. They implement specific validation logic.

### Step-by-Step Guide

#### Step 1: Create Validator Class

Create `validation/my_validator.py`:

```python
"""Custom validator implementation."""

from typing import Dict, Any
from engine.backtest_engine import BacktestEngine
from engine.backtest_engine import BacktestResult
from strategies.base import StrategyBase
from config.schema import StrategyConfig
import pandas as pd


class MyCustomValidator:
    """Custom validation test."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize validator.
        
        Args:
            config: Validator configuration
        """
        self.config = config or {}
        self.min_threshold = self.config.get('min_threshold', 1.5)
    
    def validate(
        self,
        strategy_class: type[StrategyBase],
        strategy_config: StrategyConfig,
        data: pd.DataFrame,
        initial_capital: float = 10000.0
    ) -> Dict[str, Any]:
        """
        Run validation test.
        
        Args:
            strategy_class: Strategy class to test
            strategy_config: Strategy configuration
            data: Historical data
            initial_capital: Starting capital
        
        Returns:
            Dictionary with validation results
        """
        # Create strategy instance
        strategy = strategy_class(strategy_config)
        
        # Run backtest
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=initial_capital
        )
        result = engine.run(data)
        
        # Your validation logic
        passed = result.metrics.get('profit_factor', 0.0) >= self.min_threshold
        
        return {
            'passed': passed,
            'profit_factor': result.metrics.get('profit_factor', 0.0),
            'threshold': self.min_threshold,
            'message': 'Validation passed' if passed else 'Profit factor too low'
        }
```

#### Step 2: Integrate into Validation Pipeline

Edit `validation/pipeline.py`:

```python
from validation.my_validator import MyCustomValidator

class ValidationPipeline:
    def __init__(self):
        # ... existing validators ...
        self.my_validator = MyCustomValidator()
    
    def run_phase_1(self, ...):
        # ... existing validation ...
        
        # Add your validator
        my_result = self.my_validator.validate(
            strategy_class=strategy_class,
            strategy_config=strategy_config,
            data=data,
            initial_capital=initial_capital
        )
        
        if not my_result['passed']:
            # Handle failure
            pass
```

### Key Points

- **Consistent interface**: Follow existing validator patterns
- **Return structured results**: Dictionary with pass/fail and details
- **Handle errors**: Gracefully handle exceptions
- **Document requirements**: What data/conditions are needed

## Adding a New Filter

### Overview

Filters prevent trades in certain conditions (e.g., news events, low volatility).

### Step-by-Step Guide

#### Step 1: Create Filter Class

Create `strategies/filters/my_filter/__init__.py`:

```python
"""My custom filter."""

from strategies.filters.base import BaseFilter, FilterContext, FilterResult
from typing import Optional


class MyCustomFilter(BaseFilter):
    """Custom filter implementation."""
    
    def __init__(self, config: dict):
        """
        Initialize filter.
        
        Args:
            config: Filter configuration
        """
        super().__init__(config)
        self.threshold = config.get('threshold', 0.5)
        self.enabled = config.get('enabled', True)
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if trade should be allowed.
        
        Args:
            context: Filter context with price data, timestamp, etc.
        
        Returns:
            FilterResult with allowed flag and reason
        """
        if not self.enabled:
            return FilterResult(allowed=True, reason="Filter disabled")
        
        # Your filter logic
        # Example: Check if volatility is too low
        if len(context.market_data) < 20:
            return FilterResult(allowed=True, reason="Insufficient data")
        
        volatility = context.market_data['close'].pct_change().std()
        
        if volatility < self.threshold:
            return FilterResult(
                allowed=False,
                reason=f"Volatility too low: {volatility:.4f} < {self.threshold}"
            )
        
        return FilterResult(allowed=True, reason="Volatility acceptable")
```

#### Step 2: Register Filter

Edit `strategies/filters/__init__.py`:

```python
from strategies.filters.my_filter import MyCustomFilter

FILTER_REGISTRY = {
    # ... existing filters ...
    'my_custom_filter': MyCustomFilter,
}
```

#### Step 3: Add to Configuration Schema

Edit `config/schema.py`:

```python
class FilterConfig(BaseModel):
    # ... existing filters ...
    my_custom_filter: Optional[Dict[str, Any]] = None
```

#### Step 4: Use in Strategy Config

```yaml
filters:
  my_custom_filter:
    enabled: true
    threshold: 0.5
```

### Key Points

- **Inherit from `BaseFilter`**: Provides common functionality
- **Implement `check()`**: Return FilterResult
- **Use FilterContext**: Access price data, timestamp, etc.
- **Handle edge cases**: Insufficient data, missing values, etc.

## Adding a New Data Adapter

### Overview

Adapters load data from different sources (CSV, APIs, databases).

### Step-by-Step Guide

#### Step 1: Create Adapter Class

Create `adapters/data/my_adapter.py`:

```python
"""Custom data adapter."""

import pandas as pd
from typing import Optional
from pathlib import Path


class MyDataAdapter:
    """Adapter for loading data from custom source."""
    
    def load(
        self,
        source: str,
        symbol: Optional[str] = None,
        start_date: Optional[pd.Timestamp] = None,
        end_date: Optional[pd.Timestamp] = None
    ) -> pd.DataFrame:
        """
        Load data from source.
        
        Args:
            source: Data source identifier (file path, API endpoint, etc.)
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with OHLCV data and datetime index
        """
        # Your loading logic
        # Example: Load from custom API
        import requests
        
        response = requests.get(f"https://api.example.com/data/{symbol}")
        data = response.json()
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        
        # Filter by date if needed
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]
        
        # Ensure required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df[required_cols]
```

#### Step 2: Register Adapter

Edit data loading code to use your adapter:

```python
# In scripts or engine code
from adapters.data.my_adapter import MyDataAdapter

def load_data(source: str, adapter_type: str = 'auto'):
    if adapter_type == 'my_adapter' or source.startswith('api://'):
        adapter = MyDataAdapter()
        return adapter.load(source)
    # ... other adapters ...
```

### Key Points

- **Consistent interface**: Return DataFrame with OHLCV columns
- **Datetime index**: Must have datetime index
- **Handle errors**: Gracefully handle missing data, API failures, etc.
- **Date filtering**: Support optional date range filtering

## Adding a New Report Format

### Overview

Reports can be generated in different formats (HTML, PDF, JSON, etc.).

### Step-by-Step Guide

#### Step 1: Create Report Generator

Create `reports/my_format_generator.py`:

```python
"""Custom report format generator."""

from engine.backtest_engine import BacktestResult
from typing import Dict, Any


class MyFormatGenerator:
    """Generate reports in custom format."""
    
    def generate(self, result: BacktestResult, output_path: str) -> None:
        """
        Generate report.
        
        Args:
            result: BacktestResult to report on
            output_path: Where to save report
        """
        # Your report generation logic
        # Example: Generate JSON report
        import json
        
        report_data = {
            'strategy': result.strategy_name,
            'total_trades': len(result.trades),
            'profit_factor': result.metrics.get('profit_factor', 0.0),
            'sharpe_ratio': result.metrics.get('sharpe_ratio', 0.0),
            # ... more metrics ...
        }
        
        with open(output_path, 'w') as f:
            json.dump(report_data, f, indent=2)
```

#### Step 2: Integrate into Report System

Edit `reports/report_generator.py`:

```python
from reports.my_format_generator import MyFormatGenerator

class ReportGenerator:
    def generate(self, result: BacktestResult, format: str = 'html'):
        if format == 'my_format':
            generator = MyFormatGenerator()
            generator.generate(result, self.output_path)
        # ... other formats ...
```

### Key Points

- **Use BacktestResult**: Access all backtest data
- **Handle all metrics**: Include all available metrics
- **Error handling**: Gracefully handle missing data
- **Documentation**: Document format specification

## Best Practices

### General

1. **Follow existing patterns**: Look at similar implementations
2. **Write tests**: Test your extensions thoroughly
3. **Document**: Document your code and configuration
4. **Handle errors**: Gracefully handle edge cases
5. **Type hints**: Use type hints for better code clarity

### Performance

1. **Optimize calculations**: Use vectorized operations where possible
2. **Cache results**: Cache expensive calculations
3. **Lazy loading**: Load data only when needed
4. **Memory efficiency**: Be mindful of memory usage

### Testing

1. **Unit tests**: Test individual components
2. **Integration tests**: Test with real data
3. **Edge cases**: Test boundary conditions
4. **Error cases**: Test error handling

## Examples

See existing implementations for reference:

- **Strategies**: `strategies/ema_crossover/`
- **Metrics**: `metrics/metrics.py`
- **Validators**: `validation/walkforward.py`, `validation/monte_carlo/`
- **Filters**: `strategies/filters/regime/`, `strategies/filters/calendar/`
- **Adapters**: `adapters/data/`

## Getting Help

If you need help extending the system:

1. **Check examples**: Look at similar implementations
2. **Read architecture docs**: Understand system design
3. **Ask questions**: Open an issue for discussion
4. **Review code**: Study existing code for patterns

---

**Remember**: Extensions should follow existing patterns and maintain system consistency. When in doubt, look at similar implementations!
