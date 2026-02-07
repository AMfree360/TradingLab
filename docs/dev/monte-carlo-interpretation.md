# Monte Carlo Test Interpretation Guide

## Overview

This guide explains what each Monte Carlo test measures, how to interpret results, and industry-standard practices for presenting and evaluating test outcomes.

## ⚠️ Critical Principle: Tests Measure Different Things

**Each test answers a different question. They should be interpreted separately, not combined.**

Combining test results can mask critical insights. For example:
- A strategy might perform well under resampled market conditions (Bootstrap)
- But perform WORSE than random entries (Randomized Entry)
- This tells you: **Entry logic is hurting, but exits/risk management work**

---

## Test 1: Permutation Test

### What It Measures

**Null Hypothesis**: "Trade outcomes in random order produce similar results"

**Question Answered**: Does the **order** of trades matter? Is the strategy's performance due to lucky sequencing or real edge?

### How It Works

1. Takes your actual trades (with their P&L)
2. Randomly shuffles the order of trades
3. Recalculates metrics (final_pnl, Sharpe, etc.) from shuffled sequence
4. Repeats N times to build distribution

### Interpretation

**High Percentile (e.g., 95%)**:
- ✅ Strategy's actual trade sequence is better than random ordering
- ✅ Suggests: Compounding effects, drawdown management, or trade sequencing matters
- ✅ Edge may be in: Position sizing, risk management, trade timing

**Low Percentile (e.g., 5%)**:
- ⚠️ Strategy's actual sequence is WORSE than random ordering
- ⚠️ Suggests: Strategy is hurt by bad sequencing (e.g., large losses early)
- ⚠️ May indicate: Poor risk management or unlucky timing

**Near 50%**:
- Strategy's performance is order-independent
- Returns are too small relative to equity for compounding to matter
- Common in: High-frequency, small R-multiple strategies

### Industry Standard Interpretation

- **p-value < 0.05**: Strategy's sequence is significantly better than random (reject null)
- **Percentile > 95%**: Strategy is in top 5% of possible orderings
- **Percentile < 5%**: Strategy is in bottom 5% of possible orderings (bad sequencing)

### When to Use

✅ **Suitable for**: Strategies with large R-multiples, significant compounding effects
❌ **Not suitable for**: High-frequency strategies, small R-multiple trades (returns too small relative to equity)

---

## Test 2: Block Bootstrap Test

### What It Measures

**Null Hypothesis**: "Resampled market returns produce similar results"

**Question Answered**: Does the strategy perform well under **similar market conditions**? Is the edge robust to market structure?

### How It Works

1. Extracts returns from your price data
2. Resamples returns in blocks (preserves short-term autocorrelation)
3. Reconstructs synthetic price series from resampled returns
4. Maps your trades to synthetic prices (preserves entry/exit logic)
5. Recalculates metrics from synthetic trades
6. Repeats N times to build distribution

### Interpretation

**High Percentile (e.g., 95%)**:
- ✅ Strategy performs well even when market structure is resampled
- ✅ Suggests: Edge is robust to market conditions
- ✅ Edge may be in: Exit logic, risk management, trade selection (not just lucky market timing)

**Low Percentile (e.g., 5%)**:
- ⚠️ Strategy performs poorly under resampled conditions
- ⚠️ Suggests: Strategy was lucky with specific market conditions
- ⚠️ May indicate: Overfitting to specific market patterns

**Near 50%**:
- Strategy's performance is similar to resampled market conditions
- Edge is marginal or non-existent

### Industry Standard Interpretation

- **p-value < 0.05**: Strategy significantly outperforms resampled markets (reject null)
- **Percentile > 95%**: Strategy is in top 5% of possible market conditions
- **Percentile < 5%**: Strategy is in bottom 5% (lucky market timing)

### When to Use

✅ **Suitable for**: All strategies with sufficient data (≥1000 bars recommended)
❌ **Not suitable for**: Very short datasets (<500 bars)

---

## Test 3: Randomized Entry Test

### What It Measures

**Null Hypothesis**: "Random entries with identical risk management produce similar results"

**Question Answered**: Does **entry timing/location** contribute to edge? Or is the edge purely in exits, risk management, and trade structure?

### How It Works

1. Extracts baseline trades (entry bar, exit bar, direction, stop, size)
2. For each MC iteration:
   - Randomizes entry bar/price (preserves direction, exit logic, risk management)
   - Re-simulates trades with randomized entries
   - Recalculates metrics
3. Repeats N times to build distribution

### Interpretation

**High Percentile (e.g., 95%)**:
- ✅ Strategy significantly outperforms random entries
- ✅ Suggests: Entry timing/location IS contributing to edge
- ✅ Edge is in: Signal quality, entry selection, entry timing
- ⚠️ **Warning**: Strategy may be fragile to execution errors

**Low Percentile (e.g., 5%)**:
- ⚠️ Strategy performs WORSE than random entries
- ⚠️ **Critical Finding**: Entry logic is HURTING performance
- ⚠️ Suggests: Strategy should use random entries OR fix entry logic
- ✅ **Positive**: Exits and risk management work (random entries still profitable)

**Near 50%**:
- ✅ Strategy performs similarly to random entries
- ✅ **Excellent**: Entry has little/no edge, but strategy is still profitable
- ✅ Suggests: Edge is in exits, risk management, trade structure
- ✅ **Robust**: Strategy is not dependent on perfect entry execution

### Industry Standard Interpretation

- **p-value < 0.05**: Strategy significantly outperforms random entries (reject null)
- **Percentile > 95%**: Entry timing is critical (strategy is entry-dependent)
- **Percentile < 5%**: Entry logic is harmful (strategy should use random entries)
- **Percentile ≈ 50%**: Entry is irrelevant (edge is structural, not entry-dependent)

### When to Use

✅ **Suitable for**: ALL strategies (universal test)
❌ **Never skip**: This is the most important test for understanding edge source

---

## ⚠️ Critical Insight: Why Results Can Conflict

### Example: Your Current Results

```
Bootstrap:        p=0.0000, percentile=100%  ✅ Strategy beats resampled markets
Randomized Entry:  p=1.0000, percentile=0%     ⚠️ Strategy WORSE than random entries
```

### What This Means

1. **Bootstrap (100%)**: Strategy performs well under similar market conditions
   - Exits work
   - Risk management works
   - Trade structure is sound

2. **Randomized Entry (0%)**: Entry logic is HURTING performance
   - Random entries would perform better
   - Entry timing/selection is suboptimal
   - Strategy should either:
     - Fix entry logic
     - Use random entries (if exits/risk management are the edge)

### Action Items

1. **Investigate entry logic**: Why is it worse than random?
2. **Test random entries**: If random entries perform better, use them
3. **Focus on exits**: Your exits/risk management are working

---

## Industry Standard: Separate Interpretation

### Best Practice: Report Tests Separately

**DO NOT** combine tests into a single "robustness score" without context.

**DO** report:
1. Each test's results separately
2. What each test measures
3. Interpretation of each result
4. Action items based on each result

### Recommended Display Format

```
============================================================
MONTE CARLO VALIDATION RESULTS
============================================================

1. PERMUTATION TEST (Order Independence)
   Question: Does trade order matter?
   Result: p=0.0234, percentile=97.3% ✓
   Interpretation: Strategy's sequence is significantly better than random ordering.
   Action: Compounding effects are contributing to edge.

2. BLOCK BOOTSTRAP TEST (Market Structure Robustness)
   Question: Does strategy work under resampled market conditions?
   Result: p=0.0000, percentile=100.0% ✓
   Interpretation: Strategy performs well even when market structure is resampled.
   Action: Edge is robust to market conditions.

3. RANDOMIZED ENTRY TEST (Entry Contribution)
   Question: Does entry timing/location contribute to edge?
   Result: p=1.0000, percentile=0.0% ✗
   Interpretation: Strategy performs WORSE than random entries.
   Action: Entry logic is hurting performance. Consider:
     - Fixing entry signals
     - Using random entries (if exits/risk are the edge)
     - Investigating why entry selection is suboptimal

============================================================
OVERALL ASSESSMENT
============================================================

✅ Strengths:
   - Exits and risk management work (Bootstrap: 100%)
   - Trade structure is sound

⚠️ Weaknesses:
   - Entry logic is suboptimal (Randomized Entry: 0%)
   - Entry timing/selection is hurting performance

🎯 Recommendations:
   1. Investigate entry logic: Why is it worse than random?
   2. Test random entries: If better, use them
   3. Focus on exits: Your exits/risk management are working
```

---

## When to Combine Tests

### Only Combine When:

1. **All tests agree** (all high or all low)
2. **You need a single pass/fail threshold** (but still report individual results)
3. **You're comparing multiple strategies** (use combined score as tie-breaker)

### How to Combine (If Needed)

**Weighted Average** (current approach):

Base weights:
- Randomized Entry: 50%
- Bootstrap: 30%
- Permutation: 20%

Combination is reported in two views:
- **Universal (gating)**: combines suitable universal tests only (used for Phase 1 pass/fail)
- **All suitable (informational)**: may include conditional diagnostics like randomized-entry

**But Always Report**:
- Individual test results
- Individual interpretations
- Individual action items

---

## Industry Standards Summary

### 1. Statistical Significance

- **p-value < 0.05**: Statistically significant (reject null hypothesis)
- **p-value ≥ 0.05**: Not statistically significant (cannot reject null)

### 2. Percentile Interpretation

- **> 95%**: Top 5% (excellent)
- **90-95%**: Top 10% (very good)
- **50-90%**: Above median (good)
- **10-50%**: Below median (concerning)
- **< 10%**: Bottom 10% (critical issue)

### 3. Test Priority

1. **Universal gating tests**: Bootstrap and permutation (when suitable) decide Phase 1 pass/fail
2. **Randomized Entry**: High-value diagnostic for iteration, but conditional/informational by default

### 4. Reporting Standards

- ✅ Report each test separately
- ✅ Explain what each test measures
- ✅ Provide interpretation for each result
- ✅ Include action items based on each result
- ❌ Don't hide individual results in combined score
- ❌ Don't combine without context

---

## References

- **Quantitative Trading**: Ernie Chan - "Algorithmic Trading"
- **Statistical Testing**: "Advances in Financial Machine Learning" - Marcos López de Prado
- **Industry Practice**: Most quant funds report tests separately, not combined

