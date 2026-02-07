# Binance Futures Margin Implementation - Phase 1 Complete

## Summary

Phase 1 implementation adds support for Binance-style futures margin calculations with maintenance margin for proper liquidation checks.

## What Was Implemented

### 1. Enhanced MarketSpec Class

**New Fields Added:**
- `maintenance_margin_rate: Optional[float]` - Maintenance margin rate (e.g., 0.005 for 0.5%)
- `margin_mode: Literal["isolated", "cross"]` - Margin mode (default: "cross")

**Enhanced Methods:**
- `calculate_margin()` - Now properly detects leverage-based vs fixed margin systems
- `calculate_maintenance_margin()` - NEW: Calculates maintenance margin required

### 2. Enhanced Broker Class

**Updated Methods:**
- `check_margin_call()` - Now accepts `maintenance_margin` parameter for Binance-style futures
- `calculate_maintenance_margin_required()` - NEW: Convenience method for maintenance margin

### 3. Enhanced Backtest Engine

**Updated Methods:**
- `_check_margin_calls()` - Now calculates and uses maintenance margin for liquidation checks

### 4. Updated Market Profiles

**BTCUSDT_FUTURES Profile:**
- Added `maintenance_margin_rate: 0.005` (0.5% maintenance margin)
- Added `margin_mode: cross` (account-wide margin)
- Set `leverage: 10.0` (default leverage)
- Removed fixed margin fields (uses leverage-based calculation)

## How It Works

### Margin Calculation

**Traditional Futures (CME-style):**
```python
# Fixed margin per contract
margin = quantity * initial_margin_per_contract
# Example: 1 MES contract = $2,219 margin
```

**Binance-Style Futures:**
```python
# Leverage-based margin
notional_value = entry_price * quantity
initial_margin = notional_value / leverage
# Example: 1 BTC at $50k with 10x leverage = $5,000 margin
```

### Maintenance Margin

**Binance-Style Futures:**
```python
# Percentage of notional value
notional_value = current_price * quantity
maintenance_margin = notional_value * maintenance_margin_rate
# Example: 1 BTC at $50k with 0.5% rate = $250 maintenance margin
```

**Traditional Futures:**
```python
# Typically same as initial margin (or could be lower)
maintenance_margin = initial_margin_per_contract * quantity
```

### Liquidation Checks

**Binance-Style Futures:**
```python
# Liquidation occurs when equity < maintenance_margin
if equity < maintenance_margin:
    liquidate_position()
```

**Traditional Futures/Other:**
```python
# Liquidation occurs when equity < margin_used * margin_call_level
margin_level = equity / margin_used
if margin_level < margin_call_level:
    liquidate_position()
```

## Usage Example

### Backtesting Binance BTCUSDT Futures

1. **Update market_profiles.yml** (already done):
```yaml
BTCUSDT_FUTURES:
  asset_class: futures
  leverage: 10.0
  maintenance_margin_rate: 0.005
  margin_mode: cross
```

2. **Run backtest**:
```bash
python3 scripts/run_backtest.py \
  --strategy ema_crossover \
  --market BTCUSDT_FUTURES \
  --data data/raw/BTCUSDT-15m-2020.parquet
```

3. **The engine will automatically:**
   - Calculate initial margin: `$50,000 / 10 = $5,000` per contract
   - Calculate maintenance margin: `$50,000 × 0.005 = $250` per contract
   - Check liquidation: `equity < $250` triggers liquidation

## Testing

### Test Case 1: Binance Futures Margin Calculation

```python
from engine.market import MarketSpec

# Create Binance-style futures spec
market_spec = MarketSpec(
    symbol="BTCUSDT",
    exchange="binance",
    asset_class="futures",
    leverage=10.0,
    maintenance_margin_rate=0.005,
    # initial_margin_per_contract NOT set (uses leverage-based)
)

# Test initial margin
entry_price = 50000.0
quantity = 1.0
initial_margin = market_spec.calculate_margin(entry_price, quantity)
assert initial_margin == 5000.0  # $50,000 / 10 = $5,000

# Test maintenance margin
maintenance_margin = market_spec.calculate_maintenance_margin(entry_price, quantity)
assert maintenance_margin == 250.0  # $50,000 × 0.005 = $250
```

### Test Case 2: Traditional Futures (Unchanged)

```python
# Traditional futures still work as before
market_spec = MarketSpec(
    symbol="MES",
    asset_class="futures",
    initial_margin_per_contract=2219.0,
    intraday_margin_per_contract=50.0,
)

margin = market_spec.calculate_margin(4500.0, 1.0, is_intraday=False)
assert margin == 2219.0  # Fixed margin
```

## Backward Compatibility

✅ **All existing functionality preserved:**
- Traditional futures (fixed margin) work unchanged
- Forex, crypto spot, stocks work unchanged
- Existing market profiles continue to work

✅ **New functionality is opt-in:**
- Only activates when `maintenance_margin_rate` is set
- Falls back to standard margin_call_level if not set

## Next Steps (Phase 2 - Future)

1. **Isolated Margin Mode**: Support position-specific margin allocation
2. **Position-Level Leverage**: Allow different leverage per position
3. **Partial Liquidation**: Reduce position size instead of full close
4. **Funding Rate**: Simulate funding rate for perpetual futures

## Files Modified

1. `engine/market.py` - Added maintenance_margin_rate, margin_mode, calculate_maintenance_margin()
2. `engine/market_spec.py` - Deprecated compatibility shim (no duplicated logic)
3. `engine/broker.py` - Enhanced check_margin_call(), added calculate_maintenance_margin_required()
4. `engine/backtest_engine.py` - Updated _check_margin_calls() to use maintenance margin
5. `config/market_profiles.yml` - Updated BTCUSDT_FUTURES with Binance-style parameters

## Documentation

- **Architecture**: `docs/dev/BINANCE_FUTURES_MARGIN.md` - Detailed explanation of margin systems
- **Setup Guide**: `strategies/ema_crossover/configs/FUTURES_SETUP.md` - How to use futures configs

