# Backtesting Engine Accounting Refactor

## Overview

The backtesting engine has been refactored to implement a **cash-first, execution-driven accounting model** where cash is the single source of truth and equity is computed as `cash + unrealized_pnl` on every bar.

## Core Accounting Model

### Cash-First Principle

**Cash is the only mutable state.** All other values (equity, unrealized P&L, margin used) are computed properties.

### Entry Accounting

```python
margin_required = (entry_price * quantity) / leverage
cash -= margin_required + entry_commission
```

- Only cash is mutated
- Margin is stored in position for later release
- No equity update at entry

### Partial Exit Accounting

```python
released_margin = (entry_price * exit_qty) / leverage
realized_pnl = (exit_price - entry_price) * exit_qty  # direction aware
cash += released_margin + realized_pnl - exit_commission
```

- Margin is released proportionally
- Position's `margin_locked` is reduced
- No equity update at partial exit

### Full Exit Accounting

```python
remaining_margin = position.margin_locked  # Use stored value
realized_pnl = (exit_price - entry_price) * remaining_qty  # direction aware
cash += remaining_margin + realized_pnl - exit_commission
```

- All remaining margin is released
- Position removed from list
- No equity update at exit

### Equity Calculation (Every Bar)

```python
# Update unrealized P&L for all open positions
unrealized_pnl = sum(pos.calculate_unrealized_pnl(current_price) for pos in positions)

# Equity is computed property
equity = cash + unrealized_pnl
```

**Equity updates happen every bar, not on entry/exit.**

## Structural Changes

### 1. AccountState Class

```python
@dataclass
class AccountState:
    cash: float  # Single source of truth
    margin_used: float = 0.0  # Computed
    unrealized_pnl: float = 0.0  # Computed
    commission_paid: float = 0.0
    slippage_paid: float = 0.0
    
    @property
    def equity(self) -> float:
        """Equity is always computed: cash + unrealized_pnl"""
        return self.cash + self.unrealized_pnl
```

### 2. Position Updates

```python
@dataclass
class Position:
    # ... existing fields ...
    leverage: float
    margin_locked: float  # Track margin for this position
    
    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L for remaining quantity."""
        if self.direction == 'long':
            return (current_price - self.entry_price) * self.remaining_quantity
        else:
            return (self.entry_price - current_price) * self.remaining_quantity
```

### 3. Removed Anti-Patterns

- ❌ `_calculated_pnl` - No trade-level P&L accumulation
- ❌ `_validate_capital()` - Replaced with invariant check
- ❌ Equity updates in `enter_position()` / `_close_position()`
- ❌ Equity clamping (`max(equity, 0.01)`)
- ❌ Trade-sum capital reconciliation

### 4. Invariant Check

```python
def validate_invariant(self, positions, current_price) -> bool:
    """Validate: cash + unrealized_pnl == equity"""
    self.update_unrealized_pnl(positions, current_price)
    expected_equity = self.cash + self.unrealized_pnl
    return abs(expected_equity - self.equity) < 1e-6
```

This check runs in `_update_equity()` and raises `ValueError` if violated.

## Metrics Calculation

All metrics now derive from the equity curve:

### Sharpe Ratio
- Calculated from per-bar equity returns
- Auto-detects period frequency from timestamps
- No assumptions about daily bars

### Sortino Ratio
- Calculated from per-bar equity returns
- Uses downside deviation only
- Auto-detects period frequency

### Max Drawdown
- Peak-to-trough calculation from equity curve
- No clamping or artificial caps
- Uses actual equity values (can be negative)

### CAGR
- Calculated from equity curve timestamps
- Uses actual time span (not trade timestamps)
- Formula: `((final_equity / initial_capital) ^ (1 / years)) - 1`

### Profit Factor
- Derived from realized P&L (trades only)
- Not used for capital tracking

## Execution Flow

```
For each bar:
  1. Check entry conditions → enter_position() → mutate cash only
  2. Check partial exits → mutate cash only
  3. Check full exits → _close_position() → mutate cash only
  4. Remove closed positions from list
  5. _update_equity() → compute equity = cash + unrealized_pnl
  6. Validate invariant
  7. Append to equity_curve
```

## Validation

### Before Refactor
- Capital validation warnings (cash vs expected)
- Trade-level P&L accumulation
- Equity updates only on exits
- Capital drift issues

### After Refactor
- ✅ No capital validation warnings
- ✅ Invariant check: `cash + unrealized_pnl == equity`
- ✅ Equity updates every bar
- ✅ All metrics from equity curve
- ✅ Proper margin accounting with leverage

## Test Scenarios

### ✅ Single Trade, No Leverage
- Entry: cash -= position_value + commission
- Exit: cash += position_value + P&L - commission
- Equity = cash (no open positions)

### ✅ Leveraged Trade with Partial Exit
- Entry: cash -= margin + commission
- Partial: cash += released_margin + P&L - commission
- Final: cash += remaining_margin + P&L - commission
- Equity tracks unrealized P&L between partial and final

### ✅ Trade Held Across Many Bars
- Equity updates every bar with current unrealized P&L
- Drawdown reflects true peak-to-trough equity

### ✅ Strategy with 100% Losing Trades
- Equity decreases correctly
- Drawdown reflects losses
- Negative CAGR

### ✅ Strategy with No Trades
- Equity = initial capital
- All metrics = 0 or default values

### ✅ End-of-Data Forced Flatten
- Positions closed at last bar
- Equity = cash (no unrealized P&L)

### ✅ Multiple Overlapping Positions
- Each position tracks its own margin_locked
- Unrealized P&L = sum of all positions
- Equity = cash + total unrealized P&L

## Key Improvements

1. **No Capital Drift**: Cash is single source of truth, equity is computed
2. **Accurate Drawdown**: Peak-to-trough from actual equity curve
3. **Stable Metrics**: All metrics derive from equity curve, not trade sums
4. **Proper Leverage**: Margin tracked per position, released on exit
5. **Real-time Equity**: Updates every bar with unrealized P&L
6. **Invariant Validation**: Single check ensures accounting correctness

## Migration Notes

- `Account.cash` → `Account.state.cash` (backward compatible via property)
- `Account.commission_paid` → `Account.state.commission_paid` (backward compatible)
- `_calculated_pnl` → Removed (use equity curve instead)
- `_validate_capital()` → Removed (use invariant check)

## Future Enhancements

- Multi-symbol support (unrealized P&L per symbol)
- Margin call simulation
- Interest on margin
- More sophisticated position value calculation

