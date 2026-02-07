# Capital Tracking Fix

## Problem
Capital is growing massively (646k vs expected 6.9k), suggesting we're adding capital incorrectly.

## Root Cause Analysis

The capital tracking logic should be:
1. **Entry (long)**: `capital -= (entry_price * quantity + entry_commission)`
2. **Partial Exit**: `capital += (exit_price * exit_quantity - exit_commission)`
3. **Final Exit**: `capital += (exit_price * remaining_quantity - exit_commission)`

**Expected final capital**: `initial_capital + total_pnl_after_costs`

Where `total_pnl_after_costs` should account for all commissions.

## Current Issue

The capital is growing way too much, which suggests:
1. We're adding exit proceeds multiple times
2. We're not properly deducting entry costs
3. We're creating positions when capital is insufficient (capital goes negative, we set to 0, but still create position)

## Fix Applied

1. **Prevent position creation when capital insufficient**: If capital would go negative, don't create the position
2. **Ensure entry commission is properly tracked**: Store `entry_commission` in Position
3. **Ensure P&L calculation accounts for entry commission**: Subtract `entry_commission` from `total_pnl_after_costs`

## Next Steps

Need to verify:
1. Are we creating positions when capital is 0 or negative?
2. Are we adding exit proceeds multiple times?
3. Is the position sizing calculation using wrong capital values?

