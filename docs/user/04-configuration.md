# Configuration Guide

This guide explains how to configure your trading strategies in Trading Lab. Configuration files control all aspects of your strategy's behavior.

## Configuration File Structure

Each strategy has a `config.yml` file in its folder. This YAML file contains all settings for your strategy.

## Basic Structure

```yaml
strategy_name: my_strategy
description: "Description of my strategy"

market:
  exchange: binance
  symbol: BTCUSDT
  base_timeframe: 1m

timeframes:
  signal_tf: "1h"
  entry_tf: "15m"

# ... more sections below
```

## Configuration Sections

### 1. Strategy Information

```yaml
strategy_name: my_strategy
description: "My trading strategy description"
```

**Purpose**: Identifies your strategy

**Fields**:
- `strategy_name`: Unique name (lowercase, underscores)
- `description`: Human-readable description

### 2. Market Configuration

```yaml
market:
  exchange: binance
  symbol: BTCUSDT
  base_timeframe: 1m
```

**Purpose**: Defines the market you're trading

**Fields**:
- `exchange`: Exchange name (e.g., "binance")
- `symbol`: Trading pair (e.g., "BTCUSDT")
- `base_timeframe`: Smallest timeframe in your data (e.g., "1m", "5m")

### 3. Timeframes

```yaml
timeframes:
  signal_tf: "1h"    # Timeframe for generating signals
  entry_tf: "15m"    # Timeframe for entering trades
```

**Purpose**: Defines which timeframes to use

**Fields**:
- `signal_tf`: Timeframe for signal generation (e.g., "1h", "4h", "1d")
- `entry_tf`: Timeframe for trade entry (e.g., "15m", "1h")

**Common combinations**:
- Signal: 1h, Entry: 15m (swing trading)
- Signal: 4h, Entry: 1h (position trading)
- Signal: 15m, Entry: 5m (scalping)

### 4. Moving Averages

```yaml
moving_averages:
  ema5:
    enabled: true
    length: 5
  ema20:
    enabled: true
    length: 20
  sma50:
    enabled: true
    length: 50
```

**Purpose**: Configure moving averages used in your strategy

**Fields**:
- `enabled`: true/false to turn on/off
- `length`: Number of periods (e.g., 5, 20, 50)

**Types**:
- `ema*`: Exponential Moving Average
- `sma*`: Simple Moving Average

**Example**: `ema5` with `length: 5` creates a 5-period EMA.

### 5. Alignment Rules

```yaml
alignment_rules:
  long:
    condition: "ema5 > ema20"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2
  short:
    condition: "ema5 < ema20"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2
```

**Purpose**: Define when to enter long/short positions

**Fields**:
- `condition`: Logic for signal (e.g., "ema5 > ema20")
- `macd_bars_required_signal_tf`: Minimum MACD bars on signal timeframe
- `macd_bars_required_entry_tf`: Minimum MACD bars on entry timeframe

**Common conditions**:
- `"ema5 > ema20"`: Fast EMA above slow EMA (bullish)
- `"ema5 < ema20"`: Fast EMA below slow EMA (bearish)
- `"sma50 > sma200"`: Golden cross (bullish)
- `"sma50 < sma200"`: Death cross (bearish)

### 6. Stop Loss

```yaml
stop_loss:
  type: "SMA"        # or "EMA"
  length: 50
  buffer_pips: 3.0   # Buffer in pips/points/price
  buffer_unit: "pip" # "pip", "price", or "points"
  pip_size: 0.0001   # Price value of one pip
  min_stop_distance:
    enabled: false
    type: "fixed"    # "atr" or "fixed"
    fixed_distance_pips: 6.0  # For fixed type
    atr_multiplier: 1.5        # For ATR type
    atr_period: 14             # For ATR type
    mode: "skip"     # "skip" (reject) or "use_atr"/"use_fixed"
```

**Purpose**: Define stop loss settings with minimum distance protection

**Fields**:
- `type`: "SMA" or "EMA" for moving average type
- `length`: Periods for moving average (e.g., 50)
- `buffer_pips`: Buffer in pips/points/price units (e.g., 3.0)
- `buffer_unit`: Unit type ("pip" for forex, "points" for futures, "price" for crypto/stocks)
- `pip_size`: Price value of one pip (e.g., 0.0001 for EURUSD)
- `min_stop_distance`: Minimum stop distance configuration
  - `enabled`: Enable minimum stop distance check
  - `type`: "atr" (ATR-based) or "fixed" (fixed distance)
  - `fixed_distance_pips`: Fixed minimum distance (default: 6.0)
  - `atr_multiplier`: ATR multiplier for ATR type (default: 1.5)
  - `atr_period`: ATR period for ATR type (default: 14)
  - `mode`: "skip" (reject trade) or "use_atr"/"use_fixed" (use minimum distance)

**How it works**: 
- Stop loss is set at the moving average value (minus buffer for longs, plus buffer for shorts)
- Minimum stop distance ensures stop is never too close to entry
- Prevents zero-quantity trades and ensures proper risk management

### 7. Trade Management (NEW ARCHITECTURE)

Trade management uses a **master config pattern** - defaults in `config/master_trade_management.yml` can be overridden at strategy level.

#### 7a. Trailing Stop

```yaml
trailing_stop:
  enabled: true
  type: "EMA"        # "EMA", "SMA", "ATR", "percentage", "fixed_distance"
  length: 21         # For EMA/SMA
  activation_type: "r_based"  # "r_based", "atr_based", "price_based", "time_based"
  activation_r: 1.5  # For r_based activation
  stepped: true      # Use stepped R-based trailing
  min_move_pips: 2.0 # Minimum price movement before moving SL
```

**Purpose**: Automatically adjust stop loss as price moves favorably

**Fields**:
- `enabled`: true/false to enable/disable
- `type`: Trailing stop type ("EMA", "SMA", "ATR", "percentage", "fixed_distance")
- `length`: Periods for EMA/SMA (default: 21)
- `activation_type`: How trailing activates ("r_based", "atr_based", "price_based", "time_based")
- `activation_r`: R-multiple threshold for r_based activation (e.g., 1.5 = 1.5R)
- `stepped`: Use stepped R-based trailing (example: ema_crossover)
- `min_move_pips`: Minimum price movement before moving SL (default: 2.0)

**How it works**:
- Once trade reaches activation threshold, trailing stop activates
- Stop loss follows the moving average (only moves in favorable direction)
- Stepped trailing locks in profits at each R-multiple level
- Locks in profits as trade moves in your favor

#### 7b. Partial Exit (Multiple Levels Support)

```yaml
partial_exit:
  enabled: true
  levels:
    - enabled: true
      type: "r_based"  # "r_based", "atr_based", "price_based", "percentage_based", "time_based"
      level_r: 1.5
      exit_pct: 80.0
    - enabled: true
      type: "r_based"
      level_r: 2.0
      exit_pct: 50.0
  # Legacy support (auto-converted to levels):
  # level_r: 1.5
  # exit_pct: 80.0
```

**Purpose**: Exit parts of position at multiple profit targets

**Fields**:
- `enabled`: true/false to enable/disable
- `levels`: List of partial exit levels (executed in order)
  - `enabled`: Enable/disable this level
  - `type`: Level calculation type ("r_based", "atr_based", "price_based", "percentage_based", "time_based")
  - `level_r`: R-multiple to trigger (for r_based type)
  - `exit_pct`: Percentage of position to exit at this level

**How it works**:
- Multiple partial exits can be configured (e.g., 80% at 1.5R, then 50% of remaining at 2.0R)
- Levels execute in order as price reaches each target
- Remaining position continues with stop loss/trailing stop
- Locks in profits while letting winners run

#### 7c. Take Profit (Multiple Levels Support)

```yaml
take_profit:
  enabled: true
  levels:
    - enabled: true
      type: "r_based"
      target_r: 2.0
      exit_pct: 50.0  # Exit 50% at 2R
    - enabled: true
      type: "r_based"
      target_r: 3.0
      exit_pct: 50.0  # Exit remaining 50% at 3R
  # Legacy support (auto-converted to levels):
  # target_r: 3.0
```

**Purpose**: Define take profit levels (can have multiple levels)

**Fields**:
- `enabled`: true/false to enable/disable
- `levels`: List of take profit levels (executed in order)
  - `enabled`: Enable/disable this level
  - `type`: Level calculation type ("r_based", "atr_based", "price_based", "percentage_based", "time_based")
  - `target_r`: R-multiple target (for r_based type)
  - `exit_pct`: Percentage of position to exit at this level (100 = full exit)

**How it works**:
- Multiple TP levels allow scaling out (e.g., 50% at 2R, 50% at 3R)
- Levels execute in order as price reaches each target
- Partial TP exits reduce position size while keeping remainder active

### 9. Risk Management

```yaml
risk:
  sizing_mode: equity        # or "account_size"
  risk_per_trade_pct: 1.0
  account_size: 10000.0
```

**Purpose**: Control position sizing and risk

**Fields**:
- `sizing_mode`: 
  - `"equity"`: Risk based on current account value
  - `"account_size"`: Risk based on initial account size
- `risk_per_trade_pct`: Percentage of capital to risk per trade (e.g., 1.0 = 1%)
- `account_size`: Initial account size in dollars

**Example**: 
- Account: $10,000
- Risk: 1%
- Risk amount: $100 per trade
- If stop is $10 away, position size = $100 / $10 = 10 units

### 10. Execution

```yaml
execution:
  max_positions: 1
```

**Purpose**: Control trade execution

**Fields**:
- `max_positions`: Maximum concurrent positions (e.g., 1 = one at a time)

### 11. Backtest Settings

```yaml
backtest:
  commissions: 0.0004    # 0.04% per trade
  slippage_ticks: 0.0    # Slippage in price units
```

**Purpose**: Configure backtesting costs

**Fields**:
- `commissions`: Commission rate (0.0004 = 0.04% for Binance)
- `slippage_ticks`: Slippage in price units (0.0 = no slippage)

**Common commission rates**:
- Binance spot: 0.0004 (0.04%)
- Binance futures: 0.0002 (0.02%)
- Coinbase: 0.005 (0.5%)

## Complete Example Configuration

```yaml
strategy_name: ema_crossover
description: "Moving average alignment strategy for crypto"

market:
  exchange: binance
  symbol: BTCUSDT
  base_timeframe: 1m

timeframes:
  signal_tf: "1h"
  entry_tf: "15m"

moving_averages:
  ema5:
    enabled: true
    length: 5
  ema15:
    enabled: true
    length: 15
  ema30:
    enabled: true
    length: 30
  sma50:
    enabled: true
    length: 50
  ema100:
    enabled: true
    length: 100

alignment_rules:
  long:
    condition: "ema5 > ema15 > ema30 > sma50 > ema100"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2
  short:
    condition: "ema5 < ema15 < ema30 < sma50 < ema100"
    macd_bars_required_signal_tf: 1
    macd_bars_required_entry_tf: 2

stop_loss:
  type: "SMA"
  length: 50
  offset_points: 0.0

trailing_stop:
  enabled: true
  type: "EMA"
  length: 21
  activation_r: 0.5

partial_exit:
  enabled: true
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

## Configuration Best Practices

### 1. Start with Defaults

Begin with default/reasonable values, then optimize if needed.

### 2. Document Changes

Add comments explaining why you changed values:

```yaml
risk:
  risk_per_trade_pct: 1.5  # Increased from 1.0 after optimization
```

### 3. Use Version Control

Keep config files in git to track changes over time.

### 4. Test Incrementally

Change one parameter at a time to understand its effect.

### 5. Validate After Changes

Always re-run validation after changing parameters.

## Common Configuration Patterns

### Conservative Strategy

```yaml
risk:
  risk_per_trade_pct: 0.5  # Lower risk
stop_loss:
  length: 30  # Tighter stops
trailing_stop:
  enabled: true
  activation_r: 0.3  # Activate early
```

### Aggressive Strategy

```yaml
risk:
  risk_per_trade_pct: 2.0  # Higher risk
stop_loss:
  length: 100  # Wider stops
trailing_stop:
  enabled: false  # Let winners run
```

### Scalping Strategy

```yaml
timeframes:
  signal_tf: "5m"
  entry_tf: "1m"
risk:
  risk_per_trade_pct: 0.5  # Smaller positions
stop_loss:
  length: 20  # Tight stops
```

### Swing Trading Strategy

```yaml
timeframes:
  signal_tf: "4h"
  entry_tf: "1h"
risk:
  risk_per_trade_pct: 1.5
stop_loss:
  length: 100  # Wider stops for swings
trailing_stop:
  enabled: true
  activation_r: 0.5
```

## Troubleshooting

### "Invalid configuration" error

**Check**:
- YAML syntax (indentation, colons, quotes)
- All required fields present
- Field values are correct type (numbers vs strings)

### Strategy not working as expected

**Check**:
- Timeframes are correct
- Moving averages enabled
- Alignment rules make sense
- Stop loss settings reasonable

### Performance issues

**Check**:
- Commission/slippage realistic
- Risk per trade not too high
- Max positions set correctly

## Next Steps

- Learn about [Strategy Development](STRATEGY_DEVELOPMENT.md)
- Understand [Optimization](OPTIMIZATION.md)
- Read about [Backtesting](BACKTESTING.md)

## Summary

Configuration files control everything about your strategy:
- Market and timeframes
- Indicators and signals
- Risk management
- Stop losses and exits
- Backtest settings

Take time to understand each section and configure appropriately for your strategy!

