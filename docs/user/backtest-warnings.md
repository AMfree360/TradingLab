# Backtest Warnings and Signal Rejections

This document explains the various warnings and signal rejection reasons you may encounter during backtesting.

## Signal Rejection Reasons

At the end of each backtest, you'll see a summary of signal rejections:

```
Cumulative Signal Rejections Summary:
  Total rejections: 24689
  expired: 120
  ema_alignment: 11034
  macd_progression: 10092
  invalid_stop: 3393
  zero_quantity: 50
  Pending signals at end: 0
  Daily Trade Limit Rejections: 2003
```

### `zero_quantity`

**What it means**: The calculated position size is zero or negative, so the trade cannot be executed.

**Why this happens**:
1. **Stop loss equals entry price**: The stop loss distance is zero, making it impossible to calculate position size
2. **Stop loss too close**: For forex, the minimum stop loss distance is 1 pip (0.0001 for most pairs). If your stop loss is closer than this, position sizing will return zero
3. **Price risk too small**: The difference between entry and stop prices is too small to calculate a meaningful position size

**Common causes** (FIXED in latest version):
- ~~**Stop loss MA equals entry price**: If the stop loss moving average (`sl_ma`) value equals or is very close to the entry price, even after adding the buffer, the stop loss may end up equal to the entry price~~ **FIXED**: The stop loss calculation now ensures minimum distance from entry equals buffer amount
- **Incorrect buffer configuration**: If `buffer_pips` is too small or `buffer_unit` is incorrect for your market, the stop loss may be too close
- **Market-specific issues**: For EURUSD, ensure `buffer_unit` is set to `"pip"` with `pip_size: 0.0001`, not `"points"` or `"price"`

**Note**: As of the latest update, the stop loss calculation ensures that the stop is always at least `buffer_pips` away from the entry price, even if the `sl_ma` equals the entry price. The implementation uses the more conservative (further from entry) of:
- Stop calculated from `sl_ma ± buffer`
- Stop calculated from `entry ± buffer`

This guarantees minimum distance while respecting the MA-based stop when it provides better protection.

**How to diagnose**:
The warning log includes detailed information:
```
zero_quantity - entry=1.06000, stop=1.06000, price_risk=0.000000, 
stop_loss_pips=0.0000, sl_ma=1.06000, buffer=3 pip (pip_size=0.0001), 
risk_amount=$50.00, account_size=$10000.00, direction=long
```

**What to check**:
1. **sl_ma value**: Is it equal to or very close to the entry price? This suggests the MA is crossing the price
2. **buffer configuration**: Verify `buffer_pips` and `buffer_unit` in your strategy config match your market profile
3. **buffer calculation**: For EURUSD, buffer should be in pips (e.g., `buffer_pips: 3` with `buffer_unit: "pip"` and `pip_size: 0.0001`)

**How to fix**:
1. **Increase buffer**: Increase `buffer_pips` to ensure minimum distance (e.g., 3-5 pips for EURUSD)
2. **Check market profile**: Ensure your market profile matches your data (EURUSD should use forex profile, not futures)
3. **Verify stop loss calculation**: Check that `sl_ma` is calculated correctly and isn't equal to entry price

**Example fix for EURUSD**:
```yaml
stop_loss:
  type: "SMA"
  length: 50
  buffer_pips: 3.0        # 3 pips minimum
  buffer_unit: "pip"      # Must be "pip" for EURUSD
  pip_size: 0.0001        # Standard pip size for EURUSD
```

### `invalid_stop`

**What it means**: The calculated stop loss is invalid (None or on the wrong side of entry).

**Why this happens**:
1. **Stop loss missing**: The `sl_ma` value is NaN or missing from the data
2. **Stop loss on wrong side**: For long trades, stop must be below entry. For short trades, stop must be above entry
3. **Buffer makes stop invalid**: After adding the buffer, the stop loss ends up on the wrong side of entry

**Common causes**:
- **MA not calculated**: The stop loss MA hasn't been calculated yet (insufficient data)
- **MA equals entry**: The MA value equals the entry price, so after buffer, stop is on wrong side
- **Incorrect buffer direction**: The buffer is being added in the wrong direction

**How to diagnose**:
The warning log shows:
```
invalid_stop LONG - entry=1.06000, stop=1.06000, sl_ma=1.06000, 
buffer=3 points, diff=0.00000
```

**What to check**:
1. **sl_ma value**: Is it NaN or equal to entry price?
2. **Buffer calculation**: Is the buffer being subtracted (longs) or added (shorts) correctly?
3. **Market profile**: Is the buffer unit correct for your market?

### `expired`

**What it means**: A signal waited too long for entry conditions and expired.

**Why this happens**:
- The signal was generated but entry conditions (EMA alignment, MACD progression) were never met within the maximum wait time
- The `max_wait_bars` limit was reached

**This is normal**: Some signals will naturally expire if market conditions don't align. This is expected behavior.

### `ema_alignment`

**What it means**: EMA alignment check failed on the entry timeframe.

**Why this happens**:
- The EMAs are not aligned in the expected direction on the entry timeframe bar
- This is a filter to ensure the trend is still valid when entering

**This is normal**: This filter prevents entering trades when the trend has reversed.

### `macd_progression`

**What it means**: MACD progression check failed on the entry timeframe.

**Why this happens**:
- The MACD histogram is not progressing in the expected direction
- This ensures momentum is still building when entering

**This is normal**: This filter prevents entering trades when momentum is weakening.

### `cannot_afford`

**What it means**: The account doesn't have enough margin/cash to open the position.

**Why this happens**:
- Insufficient free margin after accounting for existing positions
- Commission costs exceed available cash
- Position size is too large for account equity

**How to fix**:
- Reduce `risk_per_trade_pct` in risk config
- Increase initial capital
- Close existing positions
- Disable margin checks (not recommended for realistic backtesting)

## Understanding the Logs

### Log Format

The zero_quantity warning includes:
- `entry`: Entry price (5 decimal places for precision)
- `stop`: Stop loss price (5 decimal places)
- `price_risk`: Absolute difference between entry and stop
- `stop_loss_pips`: Stop loss distance in pips (for forex)
- `sl_ma`: The stop loss MA value used in calculation
- `buffer`: Buffer configuration (value + unit + pip_size if applicable)
- `risk_amount`: Dollar amount being risked
- `account_size`: Account size used for position sizing
- `direction`: Trade direction (long/short)

### Example Log Analysis

```
zero_quantity - entry=1.06000, stop=1.06000, price_risk=0.000000, 
stop_loss_pips=0.0000, sl_ma=1.06000, buffer=3 pip (pip_size=0.0001), 
risk_amount=$50.00, account_size=$10000.00, direction=long
```

**Analysis**:
- Entry and stop are equal (1.06000), so price_risk is 0
- sl_ma is 1.06000, same as entry price
- Buffer is 3 pips (0.0003), but stop still equals entry
- **Problem**: The MA value equals entry, so even after subtracting 3 pips, the stop should be 1.05970, not 1.06000

**Possible causes**:
1. The stop loss calculation is using the wrong bar (maybe using current bar instead of previous)
2. The buffer is not being applied correctly
3. There's a rounding issue

## Troubleshooting Checklist

When you see `zero_quantity` warnings:

1. ✅ Check market profile matches your data (EURUSD = forex, not futures)
2. ✅ Verify `buffer_unit` is correct (`"pip"` for EURUSD, not `"points"`)
3. ✅ Check `pip_size` is correct (0.0001 for EURUSD)
4. ✅ Verify `buffer_pips` is sufficient (minimum 3 for EURUSD)
5. ✅ Check if `sl_ma` values are being calculated correctly
6. ✅ Verify stop loss calculation uses correct bar (should use closed bar, not current)
7. ✅ Check if there's a rounding issue in stop loss calculation

## Related Configuration

### Stop Loss Config
```yaml
stop_loss:
  type: "SMA"              # or "EMA"
  length: 50               # MA period
  buffer_pips: 3.0        # Buffer amount
  buffer_unit: "pip"       # Unit: "pip", "price", or "points"
  pip_size: 0.0001        # Pip size for pip-based buffers
```

### Market Profile
Ensure your market profile matches your instrument:
- **EURUSD**: Use forex profile with `pip_value: 0.0001`
- **MES (Micro E-mini S&P)**: Use futures profile with `tick_size: 0.25`

## Getting Help

If you continue to see `zero_quantity` warnings after checking the above:

1. Check the detailed log output for the exact values
2. Verify your strategy config matches the market profile
3. Compare stop loss calculation with the example strategy implementation (e.g., ema_crossover)
4. Check if the issue is specific to certain market conditions (e.g., when price crosses MA)

