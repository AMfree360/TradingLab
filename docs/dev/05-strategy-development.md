# Strategy Development Guide

This guide will teach you how to create your own trading strategies in Trading Lab. Don't worry if you're not a programmer - we'll explain everything step by step.

## What is a Trading Strategy?

A trading strategy is a set of rules that tells the system:
- **When to buy** (enter a long position)
- **When to sell** (enter a short position)
- **When to exit** (close a position)
- **How much to risk** (position sizing)

## Strategy Structure

Every strategy in Trading Lab has three parts:

1. **Configuration file** (`config.yml`) - Settings and parameters
2. **Strategy code** (`strategy.py`) - The trading logic
3. **Optional README** - Documentation for your strategy

## Creating Your First Strategy

### Step 1: Create Strategy Folder

1. Go to the `strategies/` folder
2. Create a new folder with your strategy name (e.g., `my_strategy`)
3. Use lowercase letters and underscores (no spaces)

### Step 2: Create Configuration File

Create a file called `config.yml` in your strategy folder. Here's a template:

```yaml
strategy_name: my_strategy
description: "My first trading strategy"

market:
  exchange: binance
  symbol: BTCUSDT
  base_timeframe: 1m

timeframes:
  signal_tf: "1h"    # Timeframe for generating signals
  entry_tf: "15m"    # Timeframe for entering trades

moving_averages:
  ema5:
    enabled: true
    length: 5
  ema20:
    enabled: true
    length: 20

alignment_rules:
  long:
    condition: "ema5 > ema20"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2
  short:
    condition: "ema5 < ema20"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2

stop_loss:
  type: "SMA"
  length: 50
  offset_points: 0.0

trailing_stop:
  enabled: false
  type: "EMA"
  length: 21
  activation_r: 0.5

partial_exit:
  enabled: false
  level_r: 1.5
  exit_pct: 80.0

risk:
  sizing_mode: equity
  risk_per_trade_pct: 1.0
  account_size: 10000.0

execution:
  max_positions: 1

backtest:
  commissions: 0.0004
  slippage_ticks: 0.0
```

### Step 3: Create Strategy Code

Create a file called `strategy.py` in your strategy folder. Here's a simple example:

```python
"""My first trading strategy."""

import pandas as pd
import numpy as np
from strategies.base import StrategyBase
from config.schema import StrategyConfig


class MyStrategy(StrategyBase):
    """Simple moving average crossover strategy."""
    
    def __init__(self, config: StrategyConfig):
        """Initialize strategy with configuration."""
        super().__init__(config)
    
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators.
        
        This function adds indicators to your data.
        Common indicators: moving averages, MACD, RSI, etc.
        """
        # Add moving averages
        if 'ema5' in self.config.moving_averages:
            ma_config = self.config.moving_averages['ema5']
            if ma_config.enabled:
                df['ema5'] = df['close'].ewm(span=ma_config.length, adjust=False).mean()
        
        if 'ema20' in self.config.moving_averages:
            ma_config = self.config.moving_averages['ema20']
            if ma_config.enabled:
                df['ema20'] = df['close'].ewm(span=ma_config.length, adjust=False).mean()
        
        # Add MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        df['macd_hist'] = macd - signal
        
        return df
    
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Generate trading signals.
        
        This is where your trading logic goes!
        Returns a DataFrame with signals.
        """
        # Get timeframes
        signal_tf = self.config.timeframes.signal_tf
        entry_tf = self.config.timeframes.entry_tf
        
        # Get dataframes
        df_signal = df_by_tf[signal_tf]
        df_entry = df_by_tf[entry_tf]
        
        # Calculate indicators
        df_signal = self.get_indicators(df_signal)
        df_entry = self.get_indicators(df_entry)
        
        signals = []
        
        # Check each bar in signal timeframe
        for timestamp, row in df_signal.iterrows():
            # Check if we have enough data
            if pd.isna(row.get('ema5')) or pd.isna(row.get('ema20')):
                continue
            
            # Long signal: EMA5 crosses above EMA20
            if row['ema5'] > row['ema20']:
                # Check MACD confirmation
                macd_bars = self._count_macd_bars(df_signal, timestamp, 'long')
                if macd_bars >= self.config.alignment_rules['long'].macd_bars_required_signal_tf:
                    # Check entry timeframe
                    entry_row = df_entry.loc[df_entry.index <= timestamp].iloc[-1] if len(df_entry.loc[df_entry.index <= timestamp]) > 0 else None
                    if entry_row is not None:
                        entry_macd_bars = self._count_macd_bars(df_entry, entry_row.name, 'long')
                        if entry_macd_bars >= self.config.alignment_rules['long'].macd_bars_required_entry_tf:
                            # Calculate stop loss
                            stop_price = self._calculate_stop_loss(entry_row, 'long')
                            
                            signals.append({
                                'timestamp': timestamp,
                                'direction': 'long',
                                'entry_price': entry_row['close'],
                                'stop_price': stop_price,
                            })
            
            # Short signal: EMA5 crosses below EMA20
            elif row['ema5'] < row['ema20']:
                macd_bars = self._count_macd_bars(df_signal, timestamp, 'short')
                if macd_bars >= self.config.alignment_rules['short'].macd_bars_required_signal_tf:
                    entry_row = df_entry.loc[df_entry.index <= timestamp].iloc[-1] if len(df_entry.loc[df_entry.index <= timestamp]) > 0 else None
                    if entry_row is not None:
                        entry_macd_bars = self._count_macd_bars(df_entry, entry_row.name, 'short')
                        if entry_macd_bars >= self.config.alignment_rules['short'].macd_bars_required_entry_tf:
                            stop_price = self._calculate_stop_loss(entry_row, 'short')
                            
                            signals.append({
                                'timestamp': timestamp,
                                'direction': 'short',
                                'entry_price': entry_row['close'],
                                'stop_price': stop_price,
                            })
        
        # Convert to DataFrame
        if signals:
            signals_df = pd.DataFrame(signals)
            signals_df.set_index('timestamp', inplace=True)
            return signals_df
        else:
            return pd.DataFrame(columns=['direction', 'entry_price', 'stop_price'])
    
    def _count_macd_bars(self, df: pd.DataFrame, timestamp: pd.Timestamp, direction: str) -> int:
        """Count consecutive MACD bars in direction."""
        # Get data up to timestamp
        df_before = df.loc[df.index <= timestamp].tail(10)
        
        if direction == 'long':
            return len(df_before[df_before['macd_hist'] > 0])
        else:
            return len(df_before[df_before['macd_hist'] < 0])
    
    def _calculate_stop_loss(self, row: pd.Series, direction: str) -> float:
        """Calculate stop loss price."""
        stop_config = self.config.stop_loss
        
        if stop_config.type == "SMA":
            # Use SMA for stop loss
            # This is simplified - you'd calculate SMA from historical data
            # For now, use a percentage-based stop
            if direction == 'long':
                return row['close'] * 0.98  # 2% stop
            else:
                return row['close'] * 1.02  # 2% stop
        else:
            # EMA-based stop
            if direction == 'long':
                return row['close'] * 0.98
            else:
                return row['close'] * 1.02
```

## Understanding the Code

### The `get_indicators()` Method

This method calculates technical indicators (moving averages, MACD, etc.) and adds them to your data. The system calls this automatically for each timeframe.

### The `generate_signals()` Method

This is where your trading logic lives. It:
1. Gets data from different timeframes
2. Checks your conditions (e.g., EMA crossover)
3. Returns a list of signals (when to buy/sell)

### Signal Format

Each signal must have:
- `timestamp`: When the signal occurs
- `direction`: 'long' or 'short'
- `entry_price`: Price to enter at
- `stop_price`: Stop loss price

## Testing Your Strategy

Once you've created your strategy:

1. Make sure you have data (see [Data Management Guide](DATA_MANAGEMENT.md))
2. Run a backtest:
   ```bash
   python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv
   ```

## Common Patterns

### Moving Average Crossover

```python
if row['ema_fast'] > row['ema_slow']:
    # Long signal
elif row['ema_fast'] < row['ema_slow']:
    # Short signal
```

### MACD Confirmation

```python
if row['macd_hist'] > 0:
    # Bullish
elif row['macd_hist'] < 0:
    # Bearish
```

### RSI Overbought/Oversold

```python
# Calculate RSI (add to get_indicators)
delta = df['close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
rs = gain / loss
df['rsi'] = 100 - (100 / (1 + rs))

# Use in signals
if row['rsi'] < 30:
    # Oversold - potential buy
elif row['rsi'] > 70:
    # Overbought - potential sell
```

## Best Practices

1. **Start Simple**: Begin with basic strategies and add complexity gradually
2. **Test Thoroughly**: Always backtest before going live
3. **Document Your Logic**: Add comments explaining your decisions
4. **Use Configuration**: Put parameters in `config.yml`, not hardcoded
5. **Handle Edge Cases**: Check for missing data, division by zero, etc.

## Next Steps

After creating your strategy:

1. **Follow the Complete Workflow**: [../user/02-complete-workflow.md](../user/02-complete-workflow.md) - This provides a step-by-step process for testing and validating your strategy
2. **Run Initial Backtest**: [Backtesting Guide](BACKTESTING.md) - Test your strategy on historical data
3. **Validate Your Strategy**: [Validation Guide](VALIDATION.md) - Ensure it has a real edge
4. **Optimize Parameters**: [Optimization Guide](OPTIMIZATION.md) - Improve performance if needed
5. **Understand Configuration**: [Configuration Guide](CONFIGURATION.md) - Learn all available options

## Getting Help

If you're stuck:
1. Look at the `ema_crossover` example strategy
2. Check the [Configuration Guide](CONFIGURATION.md)
3. Review error messages carefully
4. Test with simple logic first

Happy strategy development!

