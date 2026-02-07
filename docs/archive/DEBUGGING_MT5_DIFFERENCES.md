# Debugging MT5 vs Trading Lab Differences

## Current Situation
- **MT5 Strategy Tester**: PF 1.61 ✅
- **Trading Lab**: PF 0.64-0.70 ❌
- **Capital Mismatch Warning**: Detected (suggests calculation bug)

## Potential Issues Identified

### 1. **Capital Tracking Bug** (CRITICAL)
The warning shows:
```
Capital mismatch detected: current_capital=9051.27, expected=9399.53
```

This suggests the capital tracking logic has a bug. Possible causes:
- Partial exits not properly tracked
- Entry/exit capital calculations don't match
- Commission/slippage double-counted or missed

**Fix Needed**: Review capital tracking in `_close_position()` and `_check_partial_exit()`

### 2. **Entry Condition Checking** (FIXED BUT NEEDS VERIFICATION)
We just fixed the entry condition logic to match the historical reference (previous TitanEA behavior). Example implementation is now available as `strategies/ema_crossover/`.
However, we need to verify:
- Are entry conditions being checked correctly?
- Are we using the right bar data?
- Is the MACD progression check working?

### 3. **Position Sizing Differences**
Historical TitanEA used:
- `balance * RiskPercent / 100.0` for risk amount
- Lot size calculation based on SL distance in pips
- Uses `account.Balance()` (current balance)

Trading Lab uses:
- `capital * risk_pct / 100.0` for risk amount
- Direct quantity calculation: `risk_amount / price_risk`
- Uses `current_capital` or `initial_capital` based on sizing_mode

**Potential Issue**: If `sizing_mode` is "account_size" but TitanEA uses current balance, this could cause differences.

### 4. **Stop Loss Calculation**
TitanEA:
- Uses closed bar (index 1) for SL MA
- Adds buffer: `sl_level -= StopLossBuffer * pip_value` for long
- Uses Ask() for long entry, Bid() for short entry

Trading Lab:
- Uses current bar for SL MA (might be different)
- Adds buffer similarly
- Uses close price (not Ask/Bid)

**Potential Issue**: Using close vs Ask/Bid could cause entry price differences.

### 5. **Trailing Stop Logic**
Need to verify:
- Is trailing stop activating at the right R multiple?
- Is the stepped trailing working correctly?
- Is the minimum move being enforced?

### 6. **Data Alignment**
- Are we using the same data as MT5?
- Are bars aligned correctly (closed bars vs current bars)?
- Are timezones correct?

## Debugging Steps

### Step 1: Fix Capital Tracking
The capital mismatch suggests a bug. Review:
1. Entry: `current_capital -= (entry_price * quantity + commission)`
2. Partial Exit: `current_capital += (exit_price * quantity - commission)`
3. Final Exit: `current_capital += (exit_price * quantity - commission)`

Ensure these balance correctly.

### Step 2: Compare First Few Trades
Compare the first 5-10 trades between MT5 and Trading Lab:
- Entry time
- Entry price
- Exit time
- Exit price
- P&L

If they differ, identify why.

### Step 3: Check Signal Generation
Verify signals are being generated at the same times:
- Are we generating the same number of signals?
- Are signals at the same timestamps?
- Are entry conditions being met at the same times?

### Step 4: Verify Position Sizing
Check if position sizes match:
- Calculate expected position size for a few trades
- Compare with actual position size
- Verify risk calculation matches

### Step 5: Check Stop Losses
Verify stop losses are being hit correctly:
- Are stops at the right prices?
- Are they being hit at the right times?
- Is trailing stop working?

## Recommended Fixes

1. **Fix capital tracking bug** (highest priority)
2. **Verify entry condition checking** is working correctly
3. **Check position sizing** matches TitanEA logic
4. **Verify stop loss calculation** uses correct bar data
5. **Compare trade-by-trade** with MT5 to find differences

## Next Steps

1. Run diagnostic script to identify specific issues
2. Compare first 10 trades with MT5
3. Fix capital tracking bug
4. Verify all calculations match TitanEA
5. Re-run backtest and compare results

