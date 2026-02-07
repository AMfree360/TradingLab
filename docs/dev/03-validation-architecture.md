# Validation Philosophy and Architecture

## Overview

This document serves as the definitive reference for the validation subsystem architecture, philosophy, and implementation specifications. It establishes the benchmark for all future development and refactoring of validation tests.

---

## Table of Contents

1. [Philosophy](#philosophy)
2. [Architecture Overview](#architecture-overview)
3. [Strategy Suitability Assessment](#strategy-suitability-assessment)
4. [Permutation Test](#permutation-test)
5. [Block Bootstrap Test](#block-bootstrap-test)
6. [Randomized Entry Test](#randomized-entry-test)
7. [Combined Robustness Score](#combined-robustness-score)
8. [Phase 1 Training Validation](#phase-1-training-validation)
9. [Metrics Pipeline](#metrics-pipeline)
10. [Common Pitfalls](#common-pitfalls)

---

## Philosophy

### Core Principles

1. **Statistical Rigor**: All tests must use correct statistical methods with proper null hypotheses
2. **Consistency**: All tests must use the SAME metrics pipeline to ensure comparability
3. **Reproducibility**: All tests use seeded random number generators
4. **Independence**: Each test addresses a different aspect of strategy robustness
5. **Completeness**: Tests must cover both temporal and structural aspects of strategy performance

### The Three Pillars of Validation

Our validation system is built on three independent Monte Carlo tests, each testing a different null hypothesis:

1. **Permutation Test**: Tests if trade order matters (temporal skill vs. luck)
2. **Block Bootstrap Test**: Tests robustness to different market regime sequences
3. **Randomized Entry Test**: Tests if strategy edge exceeds random entries with same risk management (**conditional/informational by default**)

These three tests are complementary and together provide a comprehensive assessment of strategy robustness.

---

## Architecture Overview

### System Components

```
validation/
├── monte_carlo/
│   ├── permutation.py      # Permutation test engine
│   ├── block_bootstrap.py   # Block bootstrap test engine
│   ├── randomized_entry.py # Randomized entry test engine
│   ├── runner.py           # Orchestrator and combined scoring
│   └── utils.py            # Shared utilities (metrics, p-values)
├── training_validator.py   # Phase 1 validation orchestrator
└── ...
```

### Data Flow

```
BacktestResult → Strategy Suitability Assessment → Conditional Test Execution → Distributions → P-values/Percentiles → Combined Score
```

**Conditional Execution**: Tests are only run if they are suitable for the strategy characteristics.

**Universal vs Conditional (Gating Policy)**:
- Tests can be categorized as **UNIVERSAL** (eligible for Phase 1 pass/fail) or **CONDITIONAL** (diagnostic/informational by default).
- Phase 1 Monte Carlo gating uses only **suitable + universal** tests.
- Conditional tests may still run and be reported, but do not affect pass/fail unless explicitly configured.

### Key Design Decisions

1. **Unified Metrics Pipeline**: All tests use `calculate_enhanced_metrics()` via `BacktestResult` objects
2. **Seeded RNG**: All random operations use `np.random.default_rng(seed)` for reproducibility
3. **Iterations**: N ≥ 1000 is recommended for stable estimates; lower counts are allowed for smoke tests/CI
4. **Distribution Validation**: All distributions are validated before p-value calculation
5. **KDE Smoothing**: Optional KDE smoothing for low-variance distributions
6. **Integer-Indexed Equity Curves**: All equity curves use integer index (0..N) for MC compatibility
7. **Precomputed Trade Returns**: Engine computes `trade_returns` during execution for MC compatibility
8. **Return-Based Compounding**: Permutation test uses return-based compounding, not P&L shuffling
9. **Conditional Test Execution**: Tests are only run if suitable for strategy characteristics (prevents false negatives)
10. **Adaptive Pass/Fail Logic**: Pass requirements adapt based on number of suitable tests
11. **Majority Rule for Metrics**: Individual tests pass if majority of metrics pass (not all)

---

## Permutation Test

### Philosophy

**Null Hypothesis**: The strategy's trade outcomes are independent and identically distributed (i.e., no temporal skill; only luck).

**What It Tests**: If a strategy has real edge, the order of trades matters. If it's just luck, random ordering should perform similarly.

**Key Insight**: This test breaks temporal dependencies while preserving actual trade outcomes. It answers: "Would this strategy perform as well if trades occurred in random order?"

### Architecture

#### Input
- `BacktestResult` with original trades and **precomputed trade_returns**
- Metrics to test (default: `['final_pnl', 'sharpe', 'profit_factor']`)
- Number of iterations (N ≥ 1000, default: 2000)

#### Process

1. **Extract Trade Returns** (PREFERRED: Use precomputed from BacktestResult)
   ```python
   # PREFERRED: Use precomputed trade_returns from engine
   if hasattr(backtest_result, 'trade_returns'):
       trade_returns = backtest_result.trade_returns  # Precomputed: return_i = pnl_after_costs_i / equity_before_i
   else:
       # FALLBACK: Compute from trades and equity_curve
       trade_returns = compute_returns_from_trades(trades, equity_curve)
   ```

2. **Shuffle Returns** (for each iteration)
   ```python
   permuted_returns = rng.permutation(trade_returns)  # Shuffle RETURNS, not P&Ls
   ```

3. **Apply Compounding** (for each iteration)
   ```python
   equity_path = [initial_capital]
   for return_i in permuted_returns:
       equity_next = equity_path[-1] * (1 + return_i)  # Compounding
       equity_path.append(equity_next)
   ```

4. **Reconstruct BacktestResult**
   - Create synthetic trades with shuffled returns applied
   - Build integer-indexed equity curve (0..N)
   - Use SAME metrics pipeline (`calculate_enhanced_metrics`)

5. **Calculate Metrics**
   - Use `calculate_enhanced_metrics(permuted_result)` for consistency

#### Output

- **Distributions**: Array of metric values from N permutations
- **P-values**: `p = (# simulated >= observed) / N`
- **Percentiles**: `percentile = rank(observed) / N * 100`

#### Correct Implementation Requirements

✅ **MUST DO**:
- Use precomputed `trade_returns` from BacktestResult if available
- Compute returns as: `return_i = pnl_after_costs_i / equity_at_entry_i`
- Shuffle **returns** (not raw P&Ls) using numpy RNG (seeded)
- Apply returns with compounding: `equity_{t+1} = equity_t * (1 + return_perm[t])`
- Build integer-indexed equity curves (0..N, not time-indexed)
- Use SAME metrics pipeline as real backtest
- Repeat N times (N ≥ 1000)
- Exclude PF as meaningful permutation metric (PF is order-insensitive)

❌ **MUST NOT DO**:
- Shuffle timestamps instead of trade returns
- Shuffle raw P&Ls instead of returns
- Recompute trade signals inside MC
- Use different metrics pipeline
- Use cumulative PnL instead of per-trade returns
- Build time-indexed equity curves (causes timestamp chaos)

### Example

```python
# CORRECT: Use precomputed trade_returns, shuffle returns, apply compounding
trade_returns = backtest_result.trade_returns  # Precomputed by engine
permuted_returns = rng.permutation(trade_returns)
equity_path = [initial_capital]
for ret in permuted_returns:
    equity_path.append(equity_path[-1] * (1 + ret))

# WRONG: Shuffle P&Ls or timestamps
shuffled_pnls = rng.permutation([t.net_pnl for t in trades])  # WRONG - no compounding
shuffled_times = rng.permutation(trade_times)  # WRONG - timestamps don't matter
```

---

## Block Bootstrap Test

### Philosophy

**Null Hypothesis**: Market returns have temporal structure but strategy behavior is random.

**What It Tests**: If a strategy has real edge, it should perform well across different orderings of market regimes, not just one specific sequence.

**Key Insight**: This test preserves time dependencies (volatility clustering, regime patterns) while testing robustness to different market sequences.

### Architecture

#### Input
- `BacktestResult` with original trades
- Price series (bar-level data)
- Metrics to test
- Number of iterations (N ≥ 1000)

#### Process

1. **Extract Bar-Level Returns**
   ```python
   returns = price_series.pct_change().dropna().values
   ```

2. **Block Bootstrap** (for each iteration)
   - Block size: 10 ≤ block_size ≤ 50 (for M15 data)
   - Sample blocks with replacement
   - Create synthetic return series

3. **Reconstruct Synthetic Price Series**
   ```python
   synthetic_prices = initial_price * np.cumprod(1.0 + bootstrap_returns)
   ```

4. **Map Original Trades to Synthetic Series**
   - Map original trade timestamps → synthetic return windows
   - Apply original trade sizes & SL/TP to synthetic prices
   - Compute PnL using synthetic prices but original position logic

5. **Build BacktestResult from Synthetic Trades**
   - Create trades with synthetic prices but original structure
   - Use SAME metrics pipeline

#### Output

- **Distributions**: Array of metric values from N bootstrap iterations
- **P-values**: `p = (# simulated >= observed) / N`
- **Percentiles**: `percentile = rank(observed) / N * 100`

#### Correct Implementation Requirements

✅ **MUST DO**:
- Use bar-level returns (not equity curve returns)
- Block bootstrap with 10 ≤ block_size ≤ 50
- Map original trade timestamps → synthetic return windows
- Apply original trade sizes & SL/TP to synthetic series
- Use SAME metrics pipeline

❌ **MUST NOT DO**:
- Run full strategy on synthetic data
- Use bootstrapped equity curves
- Use block_size=1 (destroys autocorrelation)
- Fail to reindex trades into synthetic timeline
- Produce zero variance distributions

### Block Size Selection

For M15 data:
- **Minimum**: 10 bars (preserves short-term dependencies)
- **Maximum**: 50 bars (allows sufficient resampling)
- **Auto-selection**: Based on data size, clamped to [10, 50]

```python
def _auto_select_block_length(n_bars: int, timeframe: str = 'M15') -> int:
    if n_bars < 100:
        return max(10, min(50, int(np.sqrt(n_bars))))
    # ... more logic
    return max(10, min(50, calculated_size))
```

### Example

```python
# Correct: Bootstrap returns, map trades to synthetic prices
bootstrap_returns = block_bootstrap(returns, block_length=20)
synthetic_prices = reconstruct_prices(initial_price, bootstrap_returns)
synthetic_trades = apply_trades_to_synthetic_prices(original_trades, synthetic_prices)

# Wrong: Re-run full backtest on synthetic data
synthetic_backtest = run_backtest(strategy, synthetic_prices)  # WRONG
```

---

## Randomized Entry Test

### Philosophy

**Null Hypothesis**: The alpha component of the strategy provides no predictive power; equivalent returns can be produced by random entry with the SAME sizing, SL, TP, exit logic, and execution model.

**What It Tests**: This is the most important test - it answers: "Is this strategy better than random entries with identical risk management?"

**Key Insight**: If a strategy has real edge, it should significantly outperform random entries with identical risk management, execution model, and exit logic.

### Architecture

#### Input
- `BacktestResult` with original trades
- Price data (OHLCV DataFrame)
- Strategy instance (for risk model and exit logic)
- Metrics to test
- Number of iterations (N ≥ 1000)

#### Process

1. **Generate Random Entry Signals**
   ```python
   entry_mask = rng.random(n_bars) < entry_probability  # Bernoulli
   ```

2. **For Each Random Entry**:
   - Randomly choose direction (long/short) based on strategy config
   - Calculate position size using SAME risk engine
   - Apply SAME SL/TP logic from strategy config
   - Simulate trade bar-by-bar (no lookahead)
   - Apply SAME execution model (slippage, commission)

3. **Build BacktestResult from Random Trades**
   - Create trades with same structure as original
   - Use SAME metrics pipeline

#### Output

- **Distributions**: Array of metric values from N random iterations
- **P-values**: `p = (# simulated >= observed) / N`
- **Percentiles**: `percentile = rank(observed) / N * 100`
- **Average Random Trades**: For validation

#### Correct Implementation Requirements

✅ **MUST DO**:
- Maintain ALL: SL/TP, position sizing, stop-out, exit timing, trade length restrictions
- Generate random entry signals: `entry[i] = Bernoulli(p)`
- Ensure random entries CANNOT look into future (only use information up to time i)
- Apply original risk engine to each random entry
- Apply original exit logic
- Use SAME metrics pipeline

❌ **MUST NOT DO**:
- Produce zero trades (broken random stream)
- Set trade directions incorrectly (always long/always flat)
- Produce zero PnL (flat distribution)
- Missing risk engine (size=0)
- Missing SL/TP application
- Not applying transaction costs
- Using future bars to exit (lookahead)

### Entry Probability

Default: Match original trade frequency
```python
if entry_probability is None:
    n_trades = len(backtest_result.trades)
    n_bars = len(price_data)
    entry_probability = min(0.1, (n_trades / n_bars) * 1.5)
```

### Example

```python
# Correct: Random entries with same risk management
for bar_idx in range(n_bars):
    if rng.random() < entry_probability:
        direction = 'long' if rng.random() < 0.5 else 'short'
        size = calculate_position_size(account, risk_cfg, entry_price, stop_price)
        trade = simulate_trade(entry_idx, direction, size, SL, TP, price_data)
        # No lookahead - only use bars up to current index

# Wrong: Using future information
if future_price > current_price:  # WRONG - lookahead
    enter_long()
```

---

## Combined Robustness Score

### Philosophy

**Purpose**: Combine results from multiple Monte Carlo tests into a unified robustness assessment.

**Key Insight**: Different tests have different null hypotheses, so we need a principled way to combine them while respecting their relative importance.

### Architecture

#### Formula

For each metric in each test:
```
metric_score = 0.5 * (1 - p_value) + 0.5 * (percentile / 100)
```

For each test:
```
test_score = mean(metric_scores across all metrics)
```

Combined score:
```
combined_score = Σ(weight_i * test_score_i)
```

#### Weights and normalization

Base weights are defined per specification:
- **Randomized Entry**: 50%
- **Bootstrap**: 30%
- **Permutation**: 20%

In practice, the combined score uses **normalized weights over the tests that are actually included**:
- **Universal combined (gating)**: normalize over suitable **universal** tests only
- **All-suitable combined (informational)**: normalize over all suitable tests that ran (universal + conditional)

```python
weights = {
    'permutation': 0.20,
    'bootstrap': 0.30,
    'randomized_entry': 0.50
}
```

#### Output

- **Combined Score / Percentile / P-value (Universal / Gating)**: Combined only from suitable universal tests
- **Combined Score / Percentile / P-value (All Suitable / Info)**: Combined from all tests that ran (universal + conditional)
- **Individual Test Scores**: For detailed analysis
- **Warnings**: Distribution validity checks

### Correctness Requirements

✅ **MUST DO**:
- Use identical metric pipeline across all tests
- Rescale metrics before computing percentiles
- Validate all distributions (no NaN, inf, or constant values)
- Use correct weights per specification

❌ **MUST NOT DO**:
- Combine inconsistent null hypotheses without normalization
- Use broken distributions (zero variance)
- Use p-values from incomparable tests
- Use wrong or mismatched metrics

---

## Phase 1 Training Validation

### Philosophy

**Purpose**: Validate strategy on training data only before proceeding to out-of-sample testing.

**Key Insight**: A strategy must pass rigorous validation on training data before it's worth testing on unseen data.

### Architecture

#### Validation Criteria

1. **Backtest Quality**
   - Profit Factor >= `min_profit_factor`
   - Sharpe Ratio >= `min_sharpe`
   - Minimum trades >= `training_min_trades`

2. **Strategy Suitability Assessment**
   - Assess strategy characteristics (return CV, exit uniformity, sample size, etc.)
   - Determine which MC tests are suitable for this strategy
   - Skip unsuitable tests (prevents false negatives)

3. **Monte Carlo Suite (Suitability-aware)**
     - Combined MC score >= `min_mc_score` (only from suitable + universal tests)
     - Combined MC percentile >= `min_mc_percentile` (only from suitable + universal tests)
   - **Adaptive Pass Requirement**:
         - If 3 suitable universal tests: require at least 2 to pass
         - If 2 suitable universal tests: require at least 1 to pass (majority rule)
         - If 1 suitable universal test: require it to pass
   - **Majority Rule for Metrics**: Each test passes if majority of its metrics pass
     - Conditional tests (e.g., randomized-entry baseline) are reported but not gated by default

4. **Parameter Sensitivity** (optional)
   - Coefficient of variation <= `sensitivity_max_cv`

#### Pass/Fail Logic

```python
# Step 1: Assess strategy suitability
profile = assessor.assess_strategy(backtest_result)
suitability = assessor.get_test_suitability(profile)

# Step 2: Run only suitable tests
mc_results = mc_suite.run_conditional(backtest_result, test_suitability=suitability)

# Step 3: Check pass/fail (adaptive based on suitable tests)
quality_pass = (PF >= min_PF) and (Sharpe >= min_Sharpe) and (trades >= min_trades)
mc_score_pass = (combined_score >= min_mc_score) and (combined_percentile >= min_mc_percentile)

# Adaptive individual test requirement
n_suitable = count_suitable_tests(suitability)
if n_suitable == 3:
    mc_tests_pass = (at_least_2_of_3_suitable_tests_pass)
elif n_suitable == 2:
    mc_tests_pass = (at_least_1_of_2_suitable_tests_pass)  # Majority rule
elif n_suitable == 1:
    mc_tests_pass = (suitable_test_passes)
else:
    mc_tests_pass = False

passed = quality_pass and mc_score_pass and mc_tests_pass
```

#### Correct Implementation Requirements

✅ **MUST DO**:
- Run backtest on training data only
- Assess strategy suitability before running MC tests
- Run only suitable Monte Carlo tests (use N ≥ 1000 for production-quality runs)
- Compute combined MC score from suitable + universal tests only for gating
- Apply adaptive pass requirement based on number of suitable tests
- Use majority rule for individual test metrics (test passes if majority of metrics pass)
- Meet all configurable thresholds

❌ **MUST NOT DO**:
- Running all tests regardless of suitability
- Passing if only one suitable test passes when 3 are suitable
- Using broken MC results
- Ignoring individual test failures
- Accepting strategy with contradictory MC outputs
- Requiring all metrics to pass (use majority rule instead)

---

## Metrics Pipeline

### Philosophy

**Core Principle**: All validation tests MUST use the SAME metrics pipeline to ensure comparability.

**Implementation**: All tests use `calculate_enhanced_metrics(BacktestResult)` via properly constructed `BacktestResult` objects.

### Architecture

#### Unified Pipeline

```python
# All tests follow this pattern:
1. Create BacktestResult from test data
2. Call calculate_enhanced_metrics(result)
3. Extract metrics for comparison
```

#### Metrics Calculated

- `final_pnl`: Total P&L (always present)
- `sharpe` / `sharpe_ratio`: Sharpe ratio (mean(returns) / std(returns))
- `profit_factor`: Gross profit / Gross loss (from trades, not equity_curve)
- `total_return_pct`: Percentage return
- `max_drawdown`: Maximum drawdown (computed from equity_curve)
- `win_rate`: Win rate percentage
- `sortino_ratio`: Sortino ratio (mean(returns) / std(negative_returns))

#### Key Metrics Calculation Details

**Sharpe Ratio**:
- Works with both datetime-indexed and integer-indexed equity curves
- Formula: `sharpe = mean(returns) / std(returns)` (no annualization by default)
- Returns computed as: `returns = equity_curve.pct_change().dropna()`
- Time-agnostic: works identically for both index types

**Sortino Ratio**:
- Uses negative returns only: `sortino = mean(returns) / std(negative_returns)`
- Works with both index types

**Profit Factor**:
- Computed from trades list (NOT from equity_curve)
- Formula: `PF = sum(winning_pnls) / abs(sum(losing_pnls))`

#### BacktestResult Construction

For each test, we construct a `BacktestResult` with:
- Same structure as original backtest
- Trades with appropriate P&Ls (using `pnl_after_costs`)
- Integer-indexed equity curve (0..N, not time-indexed)
- All required fields populated

#### Engine-Generated BacktestResult

The backtest engine now produces:
- `equity_curve`: Integer-indexed (0..N) where N = number of trades
  - `equity_curve[0] = initial_capital`
  - `equity_curve[i] = equity after trade i`
- `trade_returns`: Precomputed array (length N)
  - `trade_returns[i] = pnl_after_costs_i / equity_before_trade_i`
- `trades`: List of Trade objects with `pnl_after_costs` field

### Example

```python
# Permutation test - uses precomputed trade_returns
trade_returns = backtest_result.trade_returns  # Precomputed by engine
permuted_returns = rng.permutation(trade_returns)
equity_path = apply_compounding(permuted_returns, initial_capital)

shuffled_result = BacktestResult(
    initial_capital=initial_capital,
    final_capital=equity_path[-1],
    equity_curve=pd.Series(equity_path, index=range(len(equity_path))),  # Integer index
    trade_returns=np.array([...]),  # Synthetic trade returns
    trades=synthetic_trades,
    # ... other fields
)
metrics = calculate_enhanced_metrics(shuffled_result)  # SAME pipeline
```

---

## Common Pitfalls

### 1. Using Different Metrics Pipelines

❌ **Wrong**:
```python
# Permutation uses custom metrics
perm_metrics = custom_calculate_metrics(equity_curve)

# Bootstrap uses different metrics
bootstrap_metrics = another_calculate_metrics(trades)
```

✅ **Correct**:
```python
# Both use same pipeline
perm_metrics = calculate_enhanced_metrics(perm_result)
bootstrap_metrics = calculate_enhanced_metrics(bootstrap_result)
```

### 2. Re-running Full Backtest in Bootstrap

❌ **Wrong**:
```python
# Re-running strategy on synthetic data
synthetic_backtest = run_backtest(strategy, synthetic_prices)
```

✅ **Correct**:
```python
# Apply original trades to synthetic prices
synthetic_trades = apply_trades_to_synthetic_prices(original_trades, synthetic_prices)
```

### 3. Lookahead in Randomized Entry

❌ **Wrong**:
```python
# Using future information
if price_data.iloc[i+10]['close'] > current_price:
    enter_long()
```

✅ **Correct**:
```python
# Only use information up to current bar
if rng.random() < entry_probability:
    enter_long()  # Based on current bar only
```

### 4. Wrong Combined Score Formula

❌ **Wrong**:
```python
# Using arbitrary weights or formulas
score = 0.7 * percentile + 0.3 * (1 - p_value)
```

✅ **Correct**:
```python
# Per specification
metric_score = 0.5 * (1 - p_value) + 0.5 * (percentile / 100)
combined_score = 0.20 * perm_score + 0.30 * bootstrap_score + 0.50 * random_score
```

### 5. Insufficient Iterations

❌ **Wrong**:
```python
n_iterations = 100  # Too few
```

✅ **Correct**:
```python
n_iterations = 1000  # Minimum per specification
```

### 6. Zero Variance Distributions

❌ **Wrong**:
```python
# Not validating distributions
p_value = calculate_p_value(observed, distribution)  # May fail silently
```

✅ **Correct**:
```python
# Validate first
is_valid, error_msg = validate_distribution(distribution)
if not is_valid:
    warnings.warn(error_msg)
    return conservative_p_value
```

---

## Future Reference

### When Modifying Tests

1. **Always** use the SAME metrics pipeline
2. **Always** validate distributions before p-value calculation
3. **Always** use seeded RNG for reproducibility
4. **Always** require N ≥ 1000 iterations
5. **Never** introduce lookahead bias
6. **Never** re-run full backtest in bootstrap
7. **Never** use different metrics pipelines across tests

### When Adding New Tests

1. Define clear null hypothesis
2. Use SAME metrics pipeline
3. Validate distributions
4. Document philosophy and architecture
5. Update this document

### When Debugging

1. Check distribution validity first
2. Verify metrics pipeline consistency
3. Check for lookahead bias
4. Verify p-value calculation formula
5. Check combined score weights

---

## Conclusion

This document establishes the benchmark for all validation tests. Any future modifications must:

1. Adhere to the specifications outlined here
2. Maintain consistency with the unified metrics pipeline
3. Preserve statistical rigor
4. Document changes in this file

**Remember**: The goal is not just to pass tests, but to ensure strategies have genuine edge that will persist in live trading.

---

---

## Engine Layer Integration

### Trade Structure

The engine produces `Trade` objects with:
- `quantity: float` (REQUIRED - position size)
- `pnl_after_costs: float` (REQUIRED - P&L after all costs)
- `pnl_raw: float` (raw P&L before costs)
- All fields have defaults for synthetic construction

### BacktestResult Structure

The engine produces `BacktestResult` with:
- `equity_curve: pd.Series` - **Integer-indexed** (0..N), NOT time-indexed
  - `equity_curve[0] = initial_capital`
  - `equity_curve[i] = equity after trade i`
- `trade_returns: np.ndarray` - **Precomputed** during execution
  - `trade_returns[i] = pnl_after_costs_i / equity_before_trade_i`
- All fields have defaults for synthetic construction

### Execution Logic

During backtest execution:
1. Track equity before/after each trade
2. Compute trade return: `return = pnl_after_costs / equity_before`
3. Append to `equity_list` and `trade_returns_list`
4. Build integer-indexed equity curve at end
5. Build trade_returns array at end

See `docs/ENGINE_LAYER_MONTE_CARLO_COMPATIBILITY.md` for detailed documentation.

---

*Last Updated: 2024*
*Version: 2.0 - Engine Layer Integration*

