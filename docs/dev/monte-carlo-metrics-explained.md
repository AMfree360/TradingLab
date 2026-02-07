# Monte Carlo Test Metrics Explained

## What Metrics Are Calculated?

All three Monte Carlo tests calculate the **same metrics**:
- `final_pnl`: Total profit/loss
- `sharpe_ratio`: Risk-adjusted returns
- `profit_factor`: Gross profit / Gross loss

These metrics are calculated from:
1. **Baseline backtest** (observed values)
2. **Monte Carlo iterations** (distributions)

## How P-Values and Percentiles Are Calculated

### For Each Metric:

1. **Observed Value**: From your baseline backtest
2. **Distribution**: Array of values from N Monte Carlo iterations
3. **P-Value**: `p = (# simulated >= observed) / N`
   - Low p-value (< 0.05): Strategy significantly outperforms null hypothesis
   - High p-value (> 0.95): Strategy significantly underperforms null hypothesis
4. **Percentile**: `percentile = rank(observed) / N * 100`
   - High percentile (> 95%): Strategy is in top 5% of simulations
   - Low percentile (< 5%): Strategy is in bottom 5% of simulations

## Randomized Entry Test Metrics

### What You See in Output:

```
3. RANDOMIZED ENTRY TEST (Entry Contribution)
   final_pnl: p-value=1.0000, percentile=0.0% ✗
   sharpe_ratio: p-value=1.0000, percentile=0.0% ✗
   profit_factor: p-value=1.0000, percentile=0.0% ✗
```

**These ARE the test metrics.** They show:
- Your strategy's `final_pnl` = $X
- Random entries' `final_pnl` distribution = [values from 10 iterations]
- Your strategy's `final_pnl` is in the 0th percentile (worst)
- P-value = 1.0 means 100% of random entries performed better

### What the "Sanity Check" Shows:

```
RANDOMIZED ENTRY SANITY CHECK
  Unique Entry Bars: 26
  Unique Entry Prices: 26
```

**This is just diagnostics** - it confirms randomization is working. It's NOT the test metrics.

## How Combined Score Works

### Formula:

For each metric:
```
metric_score = 0.5 * (1 - p_value) + 0.5 * (percentile / 100)
```

For each test:
```
test_score = average(metric_scores for all metrics)
```

Combined score:
```
combined_score = weighted_average(test_scores)
  Base weights: Randomized Entry: 50%, Bootstrap: 30%, Permutation: 20%
  Normalized over tests actually included
```

### Universal vs Conditional (Gating)

The system reports two combined views:
- **Universal (gating)**: Combines only suitable **universal** tests (used for Phase 1 pass/fail)
- **All suitable (informational)**: Combines all tests that ran, including **conditional** diagnostics

This prevents conditional diagnostics (like randomized-entry) from dominating pass/fail decisions while still keeping their signal visible.

### Example Calculation:

**Bootstrap Test:**
- final_pnl: p=0.10, percentile=90% → score = 0.5*(1-0.10) + 0.5*(90/100) = 0.90
- sharpe_ratio: p=0.00, percentile=100% → score = 0.5*(1-0.00) + 0.5*(100/100) = 1.00
- profit_factor: p=0.00, percentile=100% → score = 1.00
- **Bootstrap test score = (0.90 + 1.00 + 1.00) / 3 = 0.967**

**Randomized Entry Test:**
- final_pnl: p=1.00, percentile=0% → score = 0.5*(1-1.00) + 0.5*(0/100) = 0.00
- sharpe_ratio: p=1.00, percentile=0% → score = 0.00
- profit_factor: p=1.00, percentile=0% → score = 0.00
- **Randomized Entry test score = (0.00 + 0.00 + 0.00) / 3 = 0.00**

**Combined Score (illustration):**
- For the **universal (gating)** view, weights are renormalized over universal tests only.
- For the **all-suitable** view, conditional tests may contribute, but the result is informational.

## ⚠️ Should We Combine Tests?

### Arguments FOR Combining:

1. **Quick Pass/Fail Threshold**: Single number for automated validation
2. **Comparing Multiple Strategies**: Easier to rank strategies
3. **Weighted Importance**: Preserves a consistent weighting scheme while still adapting to which tests are suitable

### Arguments AGAINST Combining:

1. **Tests Measure Different Things**: Combining masks critical insights
2. **Conflicting Results**: Bootstrap says "good", Randomized Entry says "bad" → treat randomized-entry as a diagnostic; don't let it dominate gating
3. **Loss of Information**: Individual test results are more actionable

### Industry Practice:

**Most quant funds report tests separately** and use combined scores only for:
- Automated pass/fail thresholds (with individual results still visible)
- Comparing multiple strategies (as a tie-breaker)
- Quick screening (but always review individual results)

## Recommendation

1. Always interpret individual tests first.
2. Use the **universal combined (gating)** for Phase 1 automation.
3. Use conditional tests (like randomized-entry) as diagnostics to guide iteration.

