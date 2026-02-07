# Engine Layer - Monte Carlo Compatibility

## Overview

The TradingLab engine layer has been redesigned to be 100% compatible with Monte Carlo validation tests. This document describes the engine's data structures and execution logic that ensure seamless integration with permutation, bootstrap, and randomized-entry tests.

---

## Table of Contents

1. [Trade Class Structure](#trade-class-structure)
2. [BacktestResult Structure](#backtestresult-structure)
3. [Execution Logic](#execution-logic)
4. [Equity and Returns Tracking](#equity-and-returns-tracking)
5. [Monte Carlo Integration](#monte-carlo-integration)

---

## Trade Class Structure

### Required Fields

The `Trade` class exposes the following fields for Monte Carlo compatibility:

```python
@dataclass
class Trade:
    # Required fields
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # "long" or "short"
    entry_price: float
    exit_price: float
    quantity: float  # REQUIRED - position size
    
    # P&L fields (REQUIRED for MC)
    pnl_raw: float = 0.0  # Raw P&L before costs
    pnl_after_costs: float = 0.0  # REQUIRED - P&L after all costs
    
    # Cost fields
    commission: float = 0.0
    slippage: float = 0.0
    
    # Optional fields
    stop_price: Optional[float] = None
    exit_reason: Optional[str] = None
    partial_exits: List = field(default_factory=list)
```

### Key Points

- **`quantity`**: Always a float, even for FX or futures
- **`pnl_after_costs`**: REQUIRED field - includes commission, slippage, and spread
- **`pnl_raw`**: Raw P&L before costs (for reference)
- **Backward Compatibility**: Properties provide `.size`, `.net_pnl`, `.gross_pnl` aliases

### P&L Calculation

During trade execution:
```python
# Raw P&L (before costs)
pnl_raw = (exit_price - entry_price) * quantity * direction_mult

# P&L after costs
pnl_after_costs = pnl_raw - total_commission
# Note: Slippage is already included in price adjustments
```

---

## BacktestResult Structure

### Required Fields

The `BacktestResult` class contains:

```python
@dataclass
class BacktestResult:
    # Core fields
    initial_capital: float
    final_capital: float
    
    # MONTE CARLO COMPATIBLE: Integer-indexed equity curve
    equity_curve: pd.Series  # Index: 0..N (integer, not datetime)
        # equity_curve[0] = initial_capital
        # equity_curve[i] = equity after trade i
    
    # MONTE CARLO COMPATIBLE: Precomputed trade returns
    trade_returns: np.ndarray  # Shape (N,)
        # trade_returns[i] = pnl_after_costs_i / equity_before_trade_i
    
    # Trade data
    trades: List[Trade]
    
    # Summary metrics
    total_pnl: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    total_commission: float
    total_slippage: float
    max_drawdown: float
    
    # Metadata
    strategy_name: str = ""
    symbol: str = ""
```

### Key Points

- **`equity_curve`**: Integer-indexed (0..N), NOT time-indexed
  - Length = N + 1 (initial capital + after each trade)
  - Index: `range(len(equity_curve))` = `[0, 1, 2, ..., N]`
  
- **`trade_returns`**: Precomputed during execution
  - Length = N (matches number of trades)
  - Formula: `return_i = pnl_after_costs_i / equity_before_trade_i`
  - Used directly by permutation test (no recomputation needed)

- **All fields have defaults**: Allows synthetic BacktestResult construction for MC tests

---

## Execution Logic

### Equity and Returns Tracking

During backtest execution, the engine tracks:

1. **Equity List**: `self.equity_list`
   - Starts with `[initial_capital]`
   - Appends equity after each trade closes
   - Final length = N + 1

2. **Trade Returns List**: `self.trade_returns_list`
   - Computes and appends return for each trade
   - Formula: `return = pnl_after_costs / equity_before`
   - Final length = N

### Trade Closing Logic

When a trade closes (`_close_position`):

```python
# 1. Track equity before trade
equity_before = self.account.equity

# 2. Calculate P&L after costs
pnl_after_costs = realized_pnl - total_commission

# 3. Update account (cash, commission, etc.)
self.account.cash += exit_proceeds
# ... account updates ...

# 4. Track equity after trade
equity_after = self.account.equity

# 5. Calculate trade return
if equity_before > 0:
    trade_return = pnl_after_costs / equity_before
else:
    trade_return = 0.0

# 6. Append to tracking lists
self.equity_list.append(equity_after)
self.trade_returns_list.append(trade_return)

# 7. Create Trade object
trade = Trade(
    quantity=position.size,
    pnl_after_costs=pnl_after_costs,
    # ... other fields
)
```

### Result Construction

At backtest end (`_create_result`):

```python
# 1. Build integer-indexed equity curve
equity_curve = pd.Series(
    self.equity_list,  # Length N+1
    index=range(len(self.equity_list))  # [0, 1, 2, ..., N]
)

# 2. Build trade_returns array
trade_returns = np.array(self.trade_returns_list, dtype=float)  # Length N

# 3. Calculate final capital
final_capital = float(equity_curve.iloc[-1])
total_pnl = final_capital - initial_capital

# 4. Create BacktestResult
result = BacktestResult(
    initial_capital=initial_capital,
    final_capital=final_capital,
    equity_curve=equity_curve,  # Integer-indexed
    trade_returns=trade_returns,  # Precomputed
    trades=self.trades,
    total_pnl=total_pnl,
    # ... other fields
)
```

---

## Equity and Returns Tracking

### Equity Curve Structure

**Integer-Indexed Equity Curve**:
- Index: `[0, 1, 2, ..., N]` (not timestamps)
- Values: `[initial_capital, equity_after_trade_1, ..., equity_after_trade_N]`
- Length: N + 1 (where N = number of trades)

**Why Integer Index?**
- Monte Carlo tests don't depend on timestamps
- Permutation test shuffles trade order, not time
- Eliminates timestamp alignment issues
- Works identically for all MC tests

### Trade Returns Structure

**Precomputed Trade Returns**:
- Array length: N (matches number of trades)
- Formula: `trade_returns[i] = pnl_after_costs[i] / equity_before_trade[i]`
- Computed during execution (not post-processed)

**Why Precomputed?**
- Ensures consistency: same calculation for all uses
- Eliminates recomputation errors
- Permutation test can use directly
- No need to access equity_curve during MC

---

## Monte Carlo Integration

### Permutation Test Integration

The permutation test uses precomputed `trade_returns`:

```python
# PREFERRED: Use precomputed trade_returns
if hasattr(backtest_result, 'trade_returns'):
    trade_returns = backtest_result.trade_returns  # Use directly
else:
    # FALLBACK: Compute from trades (shouldn't happen with new engine)
    trade_returns = compute_from_trades(backtest_result.trades)

# Shuffle returns
permuted_returns = rng.permutation(trade_returns)

# Apply compounding
equity_path = [initial_capital]
for ret in permuted_returns:
    equity_path.append(equity_path[-1] * (1 + ret))

# Build synthetic BacktestResult
synthetic_result = BacktestResult(
    equity_curve=pd.Series(equity_path, index=range(len(equity_path))),
    trade_returns=permuted_returns,  # Shuffled returns
    trades=synthetic_trades,
    # ... other fields
)
```

### Metrics Pipeline Integration

The metrics pipeline works with integer-indexed equity curves:

```python
# calculate_enhanced_metrics handles both index types
def calculate_enhanced_metrics(result: BacktestResult):
    equity_curve = result.equity_curve
    
    # Works with both datetime and integer indexes
    if isinstance(equity_curve.index, pd.DatetimeIndex):
        # Use timestamps for time span calculation
        time_span = calculate_from_timestamps(equity_curve.index)
    else:
        # Integer index - use trade timestamps or default
        time_span = calculate_from_trades(result.trades) or 1.0
    
    # Returns calculation works identically for both
    returns = equity_curve.pct_change().dropna()  # Time-agnostic
    
    # Sharpe/Sortino work with returns (no timestamp dependency)
    sharpe = np.mean(returns) / np.std(returns)
    # ...
```

---

## Benefits

### 1. Deterministic Outputs

- Equity and returns computed during execution
- No post-processing required
- Consistent across all uses

### 2. Monte Carlo Ready

- Integer-indexed equity curves (no timestamp chaos)
- Precomputed trade returns (no recomputation)
- Clean data structures for MC tests

### 3. Backward Compatible

- Trade properties provide old API (`.size`, `.net_pnl`, etc.)
- Existing code continues to work
- Gradual migration path

### 4. Eliminates Common Bugs

- No timestamp length mismatches
- No zero variance distributions
- No recomputation errors
- No lookahead bias

---

## Migration Guide

### For Existing Code

If you have code that accesses Trade or BacktestResult:

**Trade Access**:
```python
# Old code (still works via properties)
trade.size  # → trade.quantity
trade.net_pnl  # → trade.pnl_after_costs
trade.gross_pnl  # → trade.pnl_raw

# New code (preferred)
trade.quantity  # Direct access
trade.pnl_after_costs  # Direct access
```

**BacktestResult Access**:
```python
# Old code (still works)
result.equity_curve  # Now integer-indexed
result.trades  # Same structure

# New code (use precomputed returns)
result.trade_returns  # Precomputed array
```

### For Monte Carlo Tests

**Permutation Test**:
```python
# Use precomputed trade_returns
trade_returns = backtest_result.trade_returns
permuted_returns = rng.permutation(trade_returns)
# Apply compounding...
```

**Bootstrap Test**:
```python
# Use integer-indexed equity_curve
equity_curve = backtest_result.equity_curve  # Integer index
# Map trades to synthetic prices...
```

---

## Verification

### Self-Check

After engine execution, verify:

1. **Equity Curve**:
   ```python
   assert len(result.equity_curve) == len(result.trades) + 1
   assert isinstance(result.equity_curve.index, pd.RangeIndex)  # Integer index
   assert result.equity_curve.iloc[0] == result.initial_capital
   ```

2. **Trade Returns**:
   ```python
   assert len(result.trade_returns) == len(result.trades)
   assert all(np.isfinite(result.trade_returns))
   ```

3. **Consistency**:
   ```python
   # Final capital should match equity curve
   assert abs(result.final_capital - result.equity_curve.iloc[-1]) < 1e-6
   
   # Total P&L should match
   assert abs(result.total_pnl - (result.final_capital - result.initial_capital)) < 1e-6
   ```

---

## Conclusion

The engine layer is now fully Monte Carlo compatible:

✅ Integer-indexed equity curves (no timestamp dependency)  
✅ Precomputed trade returns (no recomputation needed)  
✅ Clean Trade structure with `pnl_after_costs`  
✅ Deterministic execution (equity tracked during execution)  
✅ Backward compatible (properties maintain old API)  

This eliminates all variance collapse issues and ensures robust Monte Carlo validation.

---

*Last Updated: 2024*  
*Version: 2.0 - Monte Carlo Compatible*

