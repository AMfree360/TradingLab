# Backtest Engine V2 - Industry Standard Implementation

## Overview

The backtesting engine has been completely rewritten using industry-standard practices from Backtrader, Zipline, and VectorBT.

## Key Improvements

### 1. **Separate Cash and Position Tracking**
- **Old**: Mixed capital tracking with complex state management
- **New**: Clear separation between `Account.cash` and positions
- **Benefit**: Easier to debug, validate, and maintain

### 2. **Independent P&L Calculation**
- **Old**: P&L calculated from capital changes (error-prone)
- **New**: P&L calculated independently for each trade, then validated against capital
- **Benefit**: Can detect and fix capital tracking bugs automatically

### 3. **Validation at Every Step**
- **Old**: Capital mismatch detected only at the end
- **New**: Validation after each trade, with clear error messages
- **Benefit**: Catch bugs early, easier debugging

### 4. **Clean Architecture**
- **Old**: Monolithic class with mixed concerns
- **New**: Separate `Account` class for state, clear method separation
- **Benefit**: Easier to test, maintain, and extend

## Architecture

```
BacktestEngineV2
├── Account (tracks cash, positions, costs)
├── Position (tracks open position state)
├── Trade (tracks completed trades)
└── Methods:
    ├── enter_position() - Enter new position
    ├── _close_position() - Close position and create Trade
    ├── _check_partial_exit() - Handle partial exits
    ├── _update_trailing_stop() - Update trailing stops
    └── _validate_capital() - Validate capital tracking
```

## Capital Tracking Logic

### Entry (Long Position)
```python
position_cost = entry_price * quantity + entry_commission
account.cash -= position_cost
```

### Partial Exit
```python
exit_proceeds = exit_price * exit_quantity - exit_commission
account.cash += exit_proceeds
```

### Final Exit
```python
exit_proceeds = exit_price * remaining_quantity - exit_commission
account.cash += exit_proceeds
```

### Validation
```python
expected_cash = initial_capital + sum(all_trade_pnls)
assert abs(account.cash - expected_cash) < tolerance
```

## Key Features

1. **Proper Commission Tracking**: All commissions tracked separately and included in P&L
2. **Slippage Handling**: Applied correctly on entry and exit
3. **Position Sizing**: Supports both fixed and compounding modes
4. **Trailing Stops**: Full support for stepped and EMA-based trailing
5. **Partial Exits**: Proper handling of partial position exits
6. **Validation**: Automatic validation of capital tracking

## Migration Notes

- Interface is identical to old engine (drop-in replacement)
- All existing scripts work without changes
- Results should be more accurate due to proper capital tracking
- Capital mismatch warnings should be eliminated

## Testing

The new engine has been designed to:
- Pass all existing tests
- Produce identical results to old engine (when old engine was correct)
- Catch and report capital tracking errors immediately
- Be easier to debug and maintain

## Future Enhancements

1. Add position value tracking for more accurate equity calculation
2. Support for multiple symbols
3. Portfolio-level risk management
4. More detailed validation and error reporting

