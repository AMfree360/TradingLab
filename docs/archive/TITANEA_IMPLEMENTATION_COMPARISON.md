# TitanEA Implementation Comparison: MQL5 vs Python

This document compares the historical TitanEA implementation with the Python implementation; relevant code references have been migrated to the `ema_crossover` example.

## 1. Signal Generation Logic

### MQL5 Implementation
- Checks Signal TF for EMA alignment
- Checks Signal TF for MACD progression
- Applies calendar and regime filters
- Stores signal direction and waits for Entry TF confirmation

### Python Implementation
- ✅ Checks Signal TF for EMA alignment
- ✅ Checks Signal TF for MACD progression  
- ✅ Applies calendar and regime filters
- ✅ Checks Entry TF for EMA alignment
- ✅ Checks Entry TF for MACD progression

**Status**: ✅ **CORRECT** - Python implementation matches MQL5 logic

## 2. Entry Conditions

### MQL5 Implementation (`CheckEntryConditions`)
1. EMA alignment on Entry TF
2. MACD progression on Entry TF
3. **Optional EMA check**: If `Entry_EMA_Optional > 0`, checks:
   - Long: `slowest[1] > opt[1]` (slowest SMA must be above optional EMA)
   - Short: `slowest[1] < opt[1]` (slowest SMA must be below optional EMA)

### Python Implementation
1. ✅ EMA alignment on Entry TF
2. ✅ MACD progression on Entry TF
3. ❌ **MISSING**: Optional EMA check (slowest vs optional)

**Status**: ⚠️ **MISSING FEATURE** - Optional EMA check not implemented

## 3. Stop Loss Calculation

### MQL5 Implementation
```cpp
double sl_level = sl_ma[1]; // Use closed bar
double pip_value = point * 10;

// Add buffer
if(direction == 1)
   sl_level -= StopLossBuffer * pip_value;
else
   sl_level += StopLossBuffer * pip_value;

// Fallback: if calculated stop is wrong side, use close - buffer
```

### Python Implementation
```python
sl_level = row['sl_ma']
buffer_price = sl_cfg.buffer_pips * self.pip_size

if direction == 1:  # Long
    stop = sl_level - buffer_price
    return stop if stop < row['close'] else row['close'] - buffer_price
else:  # Short
    stop = sl_level + buffer_price
    return stop if stop > row['close'] else row['close'] + buffer_price
```

**Status**: ✅ **CORRECT** - Python implementation matches MQL5 logic

## 4. MACD Progression Check

### MQL5 Implementation
```cpp
// Check from closed bar (index 1)
if(direction == 1) // Long
{
   for(int i = 0; i < min_bars; i++)
      if(hist[1 + i] <= hist[1 + i + 1]) return false;
}
else // Short
{
   for(int i = 0; i < min_bars; i++)
      if(hist[1 + i] >= hist[1 + i + 1]) return false;
}
```

This checks: `hist[1] > hist[2] > hist[3] > ...` for long (increasing histogram)

### Python Implementation
```python
start_idx = current_idx - min_bars
hist_values = df['macd_hist'].iloc[start_idx:current_idx + 1].values

if direction == 1:  # Long
    for i in range(min_bars):
        if hist_values[i] <= hist_values[i + 1]:
            return False
```

**Issue**: Python uses `current_idx` (current bar) while MQL5 uses closed bar (index 1).
- MQL5: Checks `hist[1] > hist[2] > hist[3]` (past to present)
- Python: Checks `hist[current-min_bars] > hist[current-min_bars+1] > ...` (should be same but indexing might differ)

**Status**: ⚠️ **NEEDS VERIFICATION** - Indexing might be different

## 5. Trailing Stop

### MQL5 Implementation
- **Activation**: Starts trailing when price reaches `entry_price ± (TrailingStartR * R_value)`
- **EMA-based trailing**: Uses trailing EMA value as stop loss
- **Stepped R-based trailing** (if enabled):
  - At 3R: Trail to +2R
  - At 2R: Trail to +1R
  - At 1R: Trail to +0.5R
  - At 0.5R: Trail to breakeven
- **Combined**: Uses max(stepped_sl, ema_sl) for long, min for short
- **Minimum move**: Only moves SL if new SL is at least `min_move_pips` away
- **Cooldown**: 60 seconds between modifications

### Python Implementation
- ❌ **NOT IMPLEMENTED** - TODO comment in code

**Status**: ❌ **MISSING FEATURE** - Trailing stop not implemented

## 6. Partial Exits

### MQL5 Implementation
- Checks if price reaches `partial_price = entry_price ± (PartialR * R_value)`
- Closes `PartialPercent` (e.g., 80%) of position
- Updates `current_lots` and marks `partial_taken = true`
- Logs partial execution

### Python Implementation
- ❌ **NOT IMPLEMENTED** - No partial exit logic

**Status**: ❌ **MISSING FEATURE** - Partial exits not implemented

## 7. Trade Management (During Open Position)

### MQL5 Implementation (`ManageActiveTrades`)
On every new bar:
1. **EMA Alignment Check**: Closes position if EMA alignment breaks
2. **Partial Execution Check**: Checks if partial level reached
3. **Take Profit Check**: Checks if TP level reached (when partials enabled)
4. **Trailing Stop Update**: Updates trailing stop on new bar

### Python Implementation
- ❌ **MISSING**: EMA alignment check during trade
- ❌ **MISSING**: Partial execution check
- ❌ **MISSING**: Trailing stop update

**Status**: ❌ **MISSING FEATURES** - Trade management not fully implemented

## 8. Optional EMA Check in Entry Conditions

### MQL5 Implementation
```cpp
if(Entry_EMA_Optional > 0)
{
   if(direction == 1)
   {
      if(slowest[1] <= opt[1]) return false;
   }
   else
   {
      if(slowest[1] >= opt[1]) return false;
   }
}
```

### Python Implementation
- ❌ **MISSING**: This check is not in `generate_signals()`

**Status**: ❌ **MISSING FEATURE**

## Summary

### ✅ Correctly Implemented
1. Signal generation logic
2. EMA alignment check (signal and entry TF)
3. MACD progression check (signal and entry TF) - **FIXED: Now matches MQL5 indexing**
4. Stop loss calculation with buffer
5. Calendar filters
6. Regime filters
7. **Optional EMA check** in entry conditions - **ADDED: Now checks slowest vs optional**

### ⚠️ Needs Verification
1. ~~MACD progression indexing~~ - **VERIFIED: Correct implementation**

### ❌ Missing Features (High Priority)
1. ~~**Optional EMA check** in entry conditions~~ - **FIXED**
2. **Trailing stop** (EMA-based and stepped R-based) - **NOT IMPLEMENTED**
3. **Partial exits** (close X% at R multiple) - **NOT IMPLEMENTED**
4. **EMA alignment check during trade** (close if alignment breaks) - **NOT IMPLEMENTED**
5. **Trade management on new bars** (partials, trailing, alignment checks) - **NOT IMPLEMENTED**

## Priority Fixes

1. **High Priority**: Add optional EMA check in entry conditions
2. **High Priority**: Implement trailing stop logic
3. **High Priority**: Implement partial exit logic
4. **Medium Priority**: Add EMA alignment check during trade management
5. **Low Priority**: Verify MACD progression indexing

