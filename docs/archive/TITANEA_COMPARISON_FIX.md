# TitanEA vs Trading Lab Implementation Comparison & Fixes

## Critical Issue Found

The Trading Lab backtest was producing PF < 1.0 while MT5 TitanEA produced PF 1.61 with the same parameters. This document outlines the differences found and fixes applied.

## Key Differences Identified

### 1. **Signal Execution Logic (CRITICAL FIX)**

**TitanEA Behavior:**
- Signals are generated on **signal timeframe** (1H) based on signal TF conditions only
- Entry conditions are checked **separately** on **entry timeframe** (15m)
- If entry conditions aren't met, signal goes to a "waiting list"
- Waiting signals are **rechecked on each new entry bar** until:
  - Entry conditions are met → trade executes
  - Max wait bars exceeded → signal expires

**Previous Trading Lab Behavior (WRONG):**
- Entry conditions were checked **once** at signal generation time
- Used **stale entry bar data** (latest entry bar up to signal time)
- Signals executed immediately without rechecking entry conditions
- This caused:
  - Missing trades that TitanEA would take
  - Taking trades at wrong times
  - Using outdated entry bar data

**Fix Applied:**
- Signals now generated on signal TF only (no entry condition checks)
- Entry conditions checked **on each entry bar** before execution
- Matches TitanEA's waiting signal logic

### 2. **Trailing Stop Implementation**

**Status:** ✅ Already matches TitanEA

- Stepped trailing based on R multiples (0.5R, 1R, 2R, 3R thresholds)
- EMA-based trailing combined with stepped logic
- Minimum move enforcement
- All logic verified against TitanEA source code

### 3. **Partial Exit Implementation**

**Status:** ✅ Already matches TitanEA

- Triggers at configured R multiple
- Exits configured percentage of position
- Remaining position continues with trailing stop

### 4. **Position Sizing**

**Status:** ✅ Already matches TitanEA

- R-based position sizing
- Supports both equity and account_size modes
- Risk percentage per trade

## Files Modified

1. **strategies/ema_crossover/strategy.py**
   - Removed entry condition checks from signal generation
   - Signals now only contain signal TF information
   - Entry conditions moved to execution time

2. **engine/backtest.py**
   - Added `_check_entry_conditions()` method
   - Modified signal execution to check entry conditions on each entry bar
   - Implements TitanEA's waiting signal pattern

## Testing Recommendations

After these fixes, backtest results should be much closer to TitanEA. However, some differences may still exist due to:

1. **Data differences**: MT5 vs CSV data sources
2. **Time zone handling**: Ensure GMT/UTC alignment
3. **Bar alignment**: MT5 uses closed bars, verify our implementation matches
4. **Commission/slippage**: Verify these match your MT5 settings

## Next Steps

1. Run backtest with same parameters as MT5
2. Compare:
   - Number of trades
   - Entry/exit times
   - Profit factor
   - Win rate
3. If discrepancies remain, check:
   - Data alignment (first/last bars)
   - Time zone conversions
   - Indicator calculations (EMA, MACD, ADX)

## Notes

- The fix ensures signals wait for proper entry conditions, matching TitanEA's behavior
- This should significantly improve profit factor and trade quality
- Entry conditions are now checked with fresh data on each bar, not stale data from signal time

