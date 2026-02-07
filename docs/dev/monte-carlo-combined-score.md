# Should We Combine Monte Carlo Test Results?

## The Question

If tests answer different questions, is there a need to combine them?

## Short Answer

**For interpretation: NO** - Always interpret tests separately.

**For pass/fail thresholds: YES** - But only using the **universal (gating)** combined score; keep conditional tests informational.

## What Each Test Measures

| Test | Question | What It Tests |
|------|----------|---------------|
| **Permutation** | Does trade order matter? | Compounding effects, sequencing |
| **Bootstrap** | Does strategy work under resampled markets? | Market structure robustness |
| **Randomized Entry** | Does entry timing contribute to edge? | Entry logic effectiveness |

## What Metrics Are Combined?

All three tests calculate the **same metrics**:
- `final_pnl`: Total profit/loss
- `sharpe_ratio`: Risk-adjusted returns  
- `profit_factor`: Gross profit / Gross loss

### How They're Combined:

1. **For each metric** (final_pnl, sharpe_ratio, profit_factor):
   ```
   metric_score = 0.5 * (1 - p_value) + 0.5 * (percentile / 100)
   ```

2. **For each test** (Permutation, Bootstrap, Randomized Entry):
   ```
   test_score = average(metric_scores for all 3 metrics)
   ```

3. **Combined score**:
   ```
   combined_score = weighted_average(test_scores)
       Base weights: Randomized Entry: 50%, Bootstrap: 30%, Permutation: 20%
       Normalized over tests actually included
   ```

## Universal vs All-Suitable Combined

The system emits two combined views:

- **Universal combined (gating)**: only suitable **universal** tests contribute
- **All-suitable combined (informational)**: combines all tests that ran, including conditional diagnostics

This avoids rejecting strategies purely due to a conditional diagnostic while keeping that diagnostic visible for iteration.

## Example: Your Current Results

### Individual Test Scores:

**Bootstrap Test:**
- final_pnl: p=0.10, percentile=90% → score = 0.90
- sharpe_ratio: p=0.00, percentile=100% → score = 1.00
- profit_factor: p=0.00, percentile=100% → score = 1.00
- **Bootstrap test score = (0.90 + 1.00 + 1.00) / 3 = 0.967** ✅

**Randomized Entry Test:**
- final_pnl: p=1.00, percentile=0% → score = 0.00
- sharpe_ratio: p=1.00, percentile=0% → score = 0.00
- profit_factor: p=1.00, percentile=0% → score = 0.00
- **Randomized Entry test score = (0.00 + 0.00 + 0.00) / 3 = 0.00** ❌

### Combined Score:

With 2 tests (Bootstrap + Randomized Entry) as an **all-suitable (informational)** view:
- Normalized weights: Bootstrap = 30/(30+50) = 37.5%, Randomized Entry = 50/(30+50) = 62.5%
- Combined = 0.375 * 0.967 + 0.625 * 0.00 = **0.362**

### What This Means:

- **Bootstrap (0.967)**: Exits and risk management work excellently
- **Randomized Entry (0.00)**: Entry logic is hurting performance
- Use randomized-entry as a diagnostic: if it's weak, it's an actionable signal to revisit entries/filters

## Industry Practice

### What Quant Funds Do:

1. **Report tests separately** - Always show individual results
2. **Provide interpretation** - Explain what each test means
3. **Use combined scores sparingly** - Only for:
   - Automated pass/fail thresholds
   - Comparing multiple strategies (tie-breaker)
   - Quick screening (but always review individual results)

### What They DON'T Do:

- ❌ Hide individual results behind combined score
- ❌ Use combined score as primary interpretation
- ❌ Combine without context

## Recommendation

### Keep Combined Score BUT:

1. ✅ **Always show individual test results first**
2. ✅ **Always show interpretation of each test**
3. ✅ **Show breakdown of what's being combined**
4. ✅ **Add clear warnings** that tests measure different things
5. ✅ **Use combined score only for pass/fail threshold**
6. ✅ **Never hide individual results**

### Display Format:

```
COMBINED ROBUSTNESS (Universal / Gating):
   Use for automated Phase 1 pass/fail.
   Always interpret individual universal tests above.
  
   Score: <score>, Percentile: <percentile>%, P-value: <p_value>
   Weights are normalized over suitable universal tests only.

COMBINED ROBUSTNESS (All Suitable / Info):
   Informational view that may include conditional diagnostics.

   Score: <score_all>, Percentile: <percentile_all>%, P-value: <p_value_all>
```

## When Combined Score Is Useful

### ✅ Good Use Cases:

1. **Automated Validation**: Single pass/fail threshold
2. **Comparing Strategies**: Quick ranking (but review individual results)
3. **Screening**: Filter out clearly bad strategies

### ❌ Bad Use Cases:

1. **Primary Interpretation**: Always look at individual tests
2. **Hiding Individual Results**: Never hide what's being combined
3. **Ignoring Test Context**: When tests disagree, investigate the individual tests and categories (universal vs conditional)

## Conclusion

**Combined scores are useful for automation, but interpretation should always focus on individual test results.**

Your current results show:
- ✅ Bootstrap: Excellent (exits/risk work)
- ❌ Randomized Entry: Critical issue (entry logic hurts)

**Action:** Use the universal (gating) combined to decide pass/fail; use randomized-entry to drive entry improvements.

