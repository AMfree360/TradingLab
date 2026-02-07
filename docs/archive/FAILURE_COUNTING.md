# Phase 1 Failure Counting System

## Overview

The Phase 1 validation system now includes automatic failure counting to enforce discipline in strategy development. This prevents excessive re-optimization and data mining.

## How It Works

### Failure Limits

- **1st Failure**: 2 attempts remaining (warning shown)
- **2nd Failure**: 1 attempt remaining (strong warning)
- **3rd Failure**: Blocked for 30 days with clear message
- **After 30 Days**: Count resets automatically, fresh attempt allowed

### Automatic Tracking

The system automatically tracks:
- Number of Phase 1 failures
- Date of last failure
- Previous failure reasons
- Remaining attempts

### Display in Reports

The HTML validation report shows:
- Date range of validation
- Failure count (X/3)
- Remaining attempts
- Last failure date
- Previous failure reasons
- Warning/blocked status with color coding

## Manual Reset (Special Cases)

For special cases where you need to reset manually (e.g., major strategy changes, bug fixes):

### Reset Everything (Including Failure Count)

```bash
python3 scripts/reset_validation_state.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_M15_2021_2025.csv
```

This gives you a fresh start with failure count reset to 0.

### Reset State But Keep Failure Count

```bash
python3 scripts/reset_validation_state.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_M15_2021_2025.csv \
  --keep-failure-count
```

This resets the validation state but preserves the failure count for tracking purposes.

## Best Practices

1. **Use manual reset sparingly**: Only when you've made significant changes (reoptimized parameters, changed logic, fixed bugs)
2. **Don't data mine**: The failure counting system is designed to prevent overfitting
3. **Consider discarding**: After 3 failures, seriously consider if the strategy concept is flawed
4. **Wait 30 days**: If you must retry after 3 failures, wait the full 30 days to ensure you're not just repeating the same mistakes

## Console Output Example

When re-running Phase 1 after a failure:

```
⚠️  Phase 1 previously completed but FAILED (completed: 2025-12-10T00:17:59.782593)
   Failure count: 1/3
   Remaining attempts: 2
Re-running Phase 1 validation (reoptimization is part of development process)...
Previous failure reasons:
  - Monte Carlo p-value too high (strategy not better than random)
  - Monte Carlo percentile too low (strategy not better than random)

============================================================
PHASE 1: TRAINING VALIDATION
============================================================
Date range: 2021-11-17 to 2022-12-30
Testing on training data only...
```

## HTML Report Example

The HTML report includes a prominent retry information box:

- **Yellow warning box** for 1-2 failures
- **Red blocked box** for 3 failures (within 30 days)
- Shows all failure information clearly

## Implementation Details

- Failure count is stored in `.validation_state/{strategy}_{data}.json`
- Count persists across sessions
- Automatically resets after 30 days
- Can be manually reset via script
- Integrated into validation pipeline

## Testing

A test script is available to verify the logic:

```bash
python3 scripts/test_failure_counting.py
```

This tests all failure counting scenarios without requiring a full validation run.

