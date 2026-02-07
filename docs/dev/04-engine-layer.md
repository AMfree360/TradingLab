# Engine Layer Details

This document provides detailed information about the backtesting engine implementation, including architecture, data flow, and key components.

## Table of Contents

1. [Overview](#overview)
2. [BacktestEngine](#backtestengine)
3. [Account Management](#account-management)
4. [Position Management](#position-management)
5. [Trade Execution](#trade-execution)
6. [Data Flow](#data-flow)
7. [Monte Carlo Compatibility](#monte-carlo-compatibility)

## Overview

The engine layer is responsible for:
- Executing backtests on historical data
- Managing account capital and positions
- Executing trades based on strategy signals
- Tracking equity curves and trade returns
- Calculating basic performance metrics

### Key Components

- **BacktestEngine**: Main execution engine
- **Account**: Capital and equity tracking
- **Broker**: Order execution simulation
- **Resampler**: Timeframe conversion
- **MarketSpec**: Market configuration

## BacktestEngine

### Class Structure

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
        # Initialize account, positions, trades
        pass
    
    def run(self, data: pd.DataFrame) -> BacktestResult:
        """Execute backtest on historical data."""
        # Main execution loop
        pass
```

### Execution Flow

1. **Data Preparation**
   - Load and validate data
   - Resample to required timeframes
   - Prepare data for strategy

2. **Signal Generation**
   - Call strategy's `generate_signals()`
   - Get trading signals for each bar

3. **Trade Execution**
   - Process signals in chronological order
   - Check account balance
   - Execute entries/exits
   - Update positions

4. **Position Management**
   - Update trailing stops
   - Check partial exits
   - Handle stop losses
   - Close positions when needed

5. **Result Compilation**
   - Calculate metrics
   - Build equity curve
   - Create BacktestResult

### Key Methods

#### `run()`

Main entry point for backtesting:

```python
def run(self, data: pd.DataFrame) -> BacktestResult:
    """
    Execute backtest on historical data.
    
    Args:
        data: DataFrame with OHLCV data and datetime index
    
    Returns:
        BacktestResult with metrics, trades, and equity curve
    """
    # 1. Prepare data
    df_by_tf = self._prepare_data(data)
    
    # 2. Generate signals
    signals = self.strategy.generate_signals(df_by_tf)
    
    # 3. Execute trades
    for timestamp, bar in data.iterrows():
        # Process signals
        # Update positions
        # Check exits
        pass
    
    # 4. Compile results
    return self._create_result()
```

#### `enter_position()`

Opens a new position:

```python
def enter_position(
    self,
    direction: str,
    entry_price: float,
    stop_price: float,
    timestamp: pd.Timestamp
) -> Optional[Position]:
    """
    Enter a new position.
    
    Args:
        direction: 'long' or 'short'
        entry_price: Entry price
        stop_price: Stop loss price
        timestamp: Entry timestamp
    
    Returns:
        Position object if successful, None if insufficient capital
    """
    # Calculate position size based on risk
    risk_amount = self.account.equity * self.strategy.config.risk.risk_per_trade_pct / 100.0
    risk_per_unit = abs(entry_price - stop_price)
    
    if risk_per_unit == 0:
        return None
    
    position_size = risk_amount / risk_per_unit
    
    # Check if we have enough capital
    required_capital = position_size * entry_price
    if required_capital > self.account.cash:
        return None
    
    # Create position
    position = Position(
        direction=direction,
        entry_price=entry_price,
        stop_price=stop_price,
        size=position_size,
        timestamp=timestamp
    )
    
    # Update account
    self.account.cash -= required_capital
    self.account.equity -= required_capital  # Will be updated on exit
    self.open_positions.append(position)
    
    return position
```

#### `_update_trailing_stop()`

Updates trailing stops for open positions:

```python
def _update_trailing_stop(
    self,
    position: Position,
    current_price: float,
    timestamp: pd.Timestamp
) -> None:
    """
    Update trailing stop for position.
    
    Args:
        position: Position to update
        current_price: Current market price
        timestamp: Current timestamp
    """
    if not self.strategy.config.trailing_stop.enabled:
        return
    
    # Calculate current R-multiple
    if position.direction == 'long':
        r_multiple = (current_price - position.entry_price) / abs(position.entry_price - position.initial_stop)
    else:
        r_multiple = (position.entry_price - current_price) / abs(position.entry_price - position.initial_stop)
    
    # Check if trailing stop should activate
    if r_multiple < self.strategy.config.trailing_stop.activation_r:
        return
    
    # Calculate new stop price
    trailing_config = self.strategy.config.trailing_stop
    # ... calculate based on EMA/SMA ...
    
    # Update stop (only in favorable direction)
    if position.direction == 'long':
        position.stop_price = max(position.stop_price, new_stop)
    else:
        position.stop_price = min(position.stop_price, new_stop)
```

#### `_check_partial_exit()`

Handles partial position exits:

```python
def _check_partial_exit(
    self,
    position: Position,
    current_price: float,
    timestamp: pd.Timestamp
) -> Optional[Trade]:
    """
    Check if partial exit should occur.
    
    Args:
        position: Position to check
        current_price: Current market price
        timestamp: Current timestamp
    
    Returns:
        Trade object if partial exit occurred, None otherwise
    """
    if not self.strategy.config.partial_exit.enabled:
        return None
    
    # Calculate current R-multiple
    r_multiple = self._calculate_r_multiple(position, current_price)
    
    # Check if partial exit level reached
    if r_multiple >= self.strategy.config.partial_exit.level_r:
        # Check if already partially exited
        if position.partial_exit_done:
            return None
        
        # Calculate exit size
        exit_pct = self.strategy.config.partial_exit.exit_pct / 100.0
        exit_size = position.size * exit_pct
        
        # Execute partial exit
        trade = self._close_position_partial(position, exit_size, current_price, timestamp)
        position.partial_exit_done = True
        
        return trade
    
    return None
```

## Account Management

### AccountState Class

Tracks account capital and equity:

```python
@dataclass
class AccountState:
    """Account state tracking."""
    cash: float  # Available cash
    equity: float  # Total equity (cash + positions value)
    initial_capital: float  # Starting capital
    
    def update_equity(self, positions: List[Position], current_prices: Dict[str, float]):
        """Update equity based on current position values."""
        positions_value = sum(
            p.size * current_prices.get(p.symbol, p.entry_price)
            for p in positions
        )
        self.equity = self.cash + positions_value
```

### Capital Tracking

- **Cash**: Available capital for new positions
- **Equity**: Total account value (cash + open positions)
- **Initial Capital**: Starting capital (for metrics)

## Position Management

### Position Class

Represents an open position:

```python
@dataclass
class Position:
    """Open trading position."""
    direction: str  # 'long' or 'short'
    entry_price: float
    stop_price: float
    initial_stop: float  # Original stop loss
    size: float  # Position size
    timestamp: pd.Timestamp
    partial_exit_done: bool = False
```

### Position Lifecycle

1. **Entry**: Created when signal triggers
2. **Active**: Monitored for exits
3. **Exit**: Closed when stop/target hit or signal reverses

## Trade Execution

### Order Types

- **Market Orders**: Executed at current price
- **Stop Orders**: Executed when price reaches stop level
- **Limit Orders**: Executed at specified price (future)

### Execution Logic

```python
def _execute_trade(
    self,
    position: Position,
    exit_price: float,
    exit_reason: str,
    timestamp: pd.Timestamp
) -> Trade:
    """
    Execute trade exit.
    
    Args:
        position: Position to close
        exit_price: Exit price
        exit_reason: Reason for exit ('stop', 'target', 'signal')
        timestamp: Exit timestamp
    
    Returns:
        Trade object with P&L and metrics
    """
    # Calculate P&L
    if position.direction == 'long':
        pnl = (exit_price - position.entry_price) * position.size
    else:
        pnl = (position.entry_price - exit_price) * position.size
    
    # Calculate costs
    commission = (position.entry_price + exit_price) * position.size * self.commission_rate
    slippage = self.slippage_ticks * position.size
    
    # Net P&L
    pnl_after_costs = pnl - commission - slippage
    
    # Update account
    self.account.cash += position.size * exit_price + pnl_after_costs
    
    # Create trade record
    trade = Trade(
        entry_time=position.timestamp,
        exit_time=timestamp,
        direction=position.direction,
        entry_price=position.entry_price,
        exit_price=exit_price,
        size=position.size,
        pnl=pnl,
        pnl_after_costs=pnl_after_costs,
        commission=commission,
        slippage=slippage,
        exit_reason=exit_reason
    )
    
    # Remove from open positions
    self.open_positions.remove(position)
    
    # Track equity
    self.account.update_equity(self.open_positions, {})
    self.equity_list.append(self.account.equity)
    self.trade_returns_list.append(pnl_after_costs / self.account.initial_capital)
    
    return trade
```

## Data Flow

### Input Data

```python
# Input: DataFrame with OHLCV
data = pd.DataFrame({
    'open': [...],
    'high': [...],
    'low': [...],
    'close': [...],
    'volume': [...]
}, index=pd.DatetimeIndex([...]))
```

### Multi-Timeframe Processing

```python
# Resample to required timeframes
df_by_tf = {
    '1h': resample_to_1h(data),
    '15m': resample_to_15m(data),
    '1m': data  # Base timeframe
}

# Strategy generates signals
signals = strategy.generate_signals(df_by_tf)
```

### Output Data

```python
# BacktestResult contains:
result = BacktestResult(
    strategy_name='my_strategy',
    trades=[...],  # List of Trade objects
    equity_curve=[...],  # Integer-indexed equity values
    trade_returns=[...],  # Precomputed returns
    metrics={...}  # Performance metrics
)
```

## Monte Carlo Compatibility

The engine is designed to be compatible with Monte Carlo validation tests:

### Integer-Indexed Equity Curve

Instead of time-indexed, uses integer index (0, 1, 2, ...):

```python
# Not: {timestamp: equity_value}
# But: [equity_0, equity_1, equity_2, ...]

equity_curve = [10000.0, 10100.0, 9950.0, ...]  # After each trade
```

### Precomputed Trade Returns

Returns are precomputed during execution:

```python
# Return for each trade
trade_returns = [
    pnl_after_costs / initial_capital,
    pnl_after_costs / initial_capital,
    ...
]
```

### Why This Design?

- **Monte Carlo tests** shuffle trade returns, not P&Ls
- **Compounding** is applied correctly: `equity[t+1] = equity[t] * (1 + return[t])`
- **No lookahead bias** in MC tests
- **Efficient** for large numbers of iterations

## Performance Considerations

### Optimization Tips

1. **Vectorize operations**: Use pandas/numpy vectorized operations
2. **Avoid loops**: Use DataFrame operations where possible
3. **Cache calculations**: Cache expensive indicator calculations
4. **Lazy evaluation**: Only calculate what's needed

### Memory Management

- **Stream processing**: Process data in chunks for large datasets
- **Delete unused data**: Clear intermediate DataFrames
- **Use generators**: For large trade lists

## Error Handling

### Common Errors

1. **Insufficient Capital**: Handle gracefully, skip trade
2. **Invalid Signals**: Validate before processing
3. **Data Issues**: Check for NaN, missing columns
4. **Division by Zero**: Check denominators before division

### Error Recovery

```python
try:
    position = self.enter_position(...)
    if position is None:
        logger.warning("Insufficient capital for trade")
        continue
except Exception as e:
    logger.error(f"Error entering position: {e}", exc_info=True)
    # Continue with next signal
    continue
```

## Testing

### Unit Tests

Test individual methods:

```python
def test_enter_position_sufficient_capital():
    engine = BacktestEngine(strategy, initial_capital=10000.0)
    position = engine.enter_position('long', 100.0, 95.0, timestamp)
    assert position is not None
    assert engine.account.cash < 10000.0
```

### Integration Tests

Test full backtest execution:

```python
def test_full_backtest():
    engine = BacktestEngine(strategy, initial_capital=10000.0)
    result = engine.run(data)
    assert result is not None
    assert len(result.trades) > 0
    assert result.metrics['total_pnl'] is not None
```

## Summary

The engine layer provides:
- **Robust execution**: Handles all edge cases
- **Accurate simulation**: Realistic trade execution
- **Performance tracking**: Complete metrics
- **MC compatibility**: Ready for validation tests

Understanding the engine layer is crucial for:
- **Debugging**: Know where issues occur
- **Optimization**: Identify bottlenecks
- **Extension**: Add new features correctly

---

For more details, see:
- [Architecture Overview](01-architecture.md)
- [Monte Carlo Implementation](08-monte-carlo.md)
- [API Reference](11-api-reference.md)
