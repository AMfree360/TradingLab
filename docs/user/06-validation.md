# Validation Guide

Validation is the process of rigorously testing your strategy to ensure it's robust and not just lucky. This guide explains Trading Lab's **phased validation methodology** designed to avoid common pitfalls like overfitting, edge erosion, and lack of stationarity.

**💡 Tip**: For a complete step-by-step workflow that includes validation, see the [Complete Workflow Guide](02-complete-workflow.md). The workflow guide shows how validation fits into the overall strategy development process.

## Why Validation Matters

A good backtest doesn't guarantee your strategy will work in the future. Validation helps you:

- **Avoid overfitting**: Make sure your strategy works on data it hasn't seen
- **Prove edge**: Verify your strategy is better than random chance
- **Test robustness**: See if results are consistent across different time periods
- **Prevent data leakage**: Enforce strict separation of training and OOS data
- **Build confidence**: Multiple passing tests increase confidence in your strategy

## The Phased Validation Methodology

Trading Lab uses a **three-phase validation system** that enforces best practices:

### Phase 1: Training Validation (Training Data Only)
**Purpose**: Prove the strategy has a real edge before using OOS data.

**Tests** (Conditionally Executed):
1. **Strategy Suitability Assessment**: Determines which tests are appropriate for this strategy
2. **Backtest Quality Check**: Ensures strategy is profitable on training data
3. **Monte Carlo Suite** (Suitability-aware):
   - **Universal tests (used for pass/fail if suitable)**:
     - **Permutation Test**: Tests if trade order matters
     - **Block Bootstrap Test**: Tests robustness to market regime sequences
   - **Conditional tests (informational by default)**:
     - **Randomized Entry Test**: Entry contribution / baseline comparison
4. **Parameter Sensitivity**: Tests robustness to parameter changes

**Conditional Test Execution**:
- Not all tests are suitable for all strategies
- The system automatically assesses strategy characteristics and skips inappropriate tests
- This prevents false negatives and saves computational resources
- Skipped tests are clearly marked with reasons and alternatives

**Data Used**: Training data only (e.g., 2020-2021)

**Gate**: Must pass before Phase 2 can run

**Re-running Phase 1**:
- ✅ **If Phase 1 FAILED**: You can re-run after reoptimization
  - Reoptimize parameters (see [Optimization Guide](OPTIMIZATION.md))
  - Update strategy configuration
  - Run Phase 1 again - it will automatically allow re-running
  - This is part of the development process (training data can be reused)
- ❌ **If Phase 1 PASSED**: Cannot re-run (prevents data mining)
  - Once Phase 1 passes, proceed to Phase 2
  - If you want to test different parameters, create a new strategy variant

### Phase 2: Out-of-Sample Validation (OOS Data - Used Only Once)
**Purpose**: Test strategy on completely unseen data.

**Tests**:
1. **Walk-Forward Analysis**: Tests strategy on multiple unseen test periods
2. **Consistency Check**: Ensures results are consistent across periods
3. **Overfitting Detection**: Compares test PF to train PF

**Data Used**: Out-of-sample data (e.g., 2022-2023) - **used only once**

**Gate**: Must pass Phase 1 first. OOS data is marked as "used" after this test.

### Phase 3: Stationarity Analysis (Post-Live)
**Purpose**: Determine optimal retraining frequency to combat edge erosion.

**Tests**:
1. **Performance Degradation Analysis**: Measures how quickly strategy performance degrades
2. **Retraining Frequency Recommendation**: Suggests when to re-optimize parameters

**Data Used**: Live trading data (after strategy is qualified)

**Gate**: Must pass Phase 1 and Phase 2 first

## Recent Changes & Important Config Notes

- **ATR stops**: Added `stop_loss.atr_enabled`, `stop_loss.atr_length`, and `stop_loss.atr_multiplier` to enable ATR-based stops. `stop_loss.type` remains supported for backward compatibility.
- **Entry filters (ema_crossover)**: New keys available for the canonical EMA example: `entry.confirmation_required`, `entry.ema_separation_pct`, and `entry.trend_slope_bars`.
- **Monte Carlo reporting**: Reports label each MC test as UNIVERSAL vs CONDITIONAL and whether it was used for Phase 1 gating. Combined robustness is shown in two views: Universal (gating) and All Suitable (informational).
- **Randomized-entry watchdog**: The randomized-entry test may abort or be skipped when there are too few unique entry points. If you see frequent skips, increase the timeframe, relax entry filters, or lengthen the date range to generate more trades.
- **Validation state manager**: Run state and retry limits are stored under `validation_state/`. Use `scripts/reset_validation_state.py` or `ValidationStateManager.reset_phase1()` to reset Phase 1 state when appropriate.
- **Quick smoke tests**: For fast checks use `scripts/run_backtest.py --strategy ema_crossover`; reports are written to `reports/` for quick inspection.
- **Re-running Phase 1**: Phase 1 is re-runnable only after a FAILURE. If Phase 1 previously passed, create a new strategy variant to test different parameters.

### Reproducibility & dataset governance (important)

Trading Lab enforces strict separation between training, OOS, and holdout data.

- **Dataset manifests**: Each run writes a deterministic dataset manifest JSON under `data/manifests/` that captures slice identity (file + date range) with stable hashing.
- **Phase locks**: Phase 1 / Phase 2 / Holdout bind to the manifest identity. If the manifest changes mid-phase, the run **aborts**. Treat this as a stop-the-line event: either you intentionally created a new dataset version, or something mutated unexpectedly.
- **Consumed-on-attempt semantics**:
  - **Phase 2 OOS is consumed on attempt** (state is recorded before validation runs).
  - **Holdout is one-shot** (also consumed on attempt).
  If a run fails part-way through, the slice is still considered used by design.
- **Walk-forward warmup bounds**: Walk-forward uses only the bounded window from train-start → oos-end to prevent holdout leakage.

For run tracking and comparisons, see the experiment registry notes in the [Scripts Reference](10-scripts-reference.md).

### Monte Carlo Defaults & Recommendation

- **Default iterations**: All Monte Carlo engines (permutation, block bootstrap, randomized-entry) use a default of **1000 iterations**. This is considered a reasonable minimum for stable p-value estimates in most use cases.
- **Recommended minimum**: For publication-grade validation or final runs prefer **≥ 1000 iterations**; increase to 2000–5000 for very noisy strategies or when p-values are near decision thresholds.
- **CI consideration**: Unit tests and CI runs may use smaller iteration counts for speed. Use `RUN_SMOKE` or run locally to perform longer Monte Carlo runs when needed.

## Complete Workflow

### Step 1: Prepare Data

```bash
# Consolidate multiple data files if needed
python3 scripts/consolidate_data.py \
  --input data/raw/EURUSD_2020.csv data/raw/EURUSD_2021.csv data/raw/EURUSD_2022.csv \
  --output data/raw/EURUSD_2020-2022.csv
```

### Step 2: Optimize Parameters (Training Data Only)

```bash
# Optimize strategy parameters on training data
python3 scripts/optimize_strategy.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2022.csv \
  --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \
  --param moving_averages.ema5.length 5 10 15 \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --metric profit_factor \
  --output strategies/ema_crossover/config_optimized.yml
```

**Important**: Only optimize on training data! Save OOS data for Phase 2.

### Step 3: Run Phase 1 - Training Validation

```bash
# Validate on training data only
python3 scripts/run_training_validation.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2022.csv \
  --config strategies/ema_crossover/config_optimized.yml \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --sensitivity-param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \
  --mc-iterations 1000
```

**What it checks**:
- ✅ Training PF ≥ 1.5
- ✅ Training Sharpe ≥ 0.2
- ✅ Minimum 30 trades
- ✅ Strategy suitability assessment completed
\- ✅ Monte Carlo suite (only suitable UNIVERSAL tests are used for pass/fail):
  - Combined MC score ≥ 0.60
  - Combined MC percentile ≥ 70%
  - Adaptive pass requirement:
    - 3 suitable universal tests: at least 2 must pass
    - 2 suitable universal tests: at least 1 must pass (majority rule)
    - 1 suitable universal test: must pass
  - Conditional MC tests may still run and be reported, but do not count toward pass/fail by default
- ✅ Parameter sensitivity CV ≤ 0.3 (robust to changes)

**Note**: Individual MC tests use majority rule - a test passes if majority of its metrics pass (not all).

**If Phase 1 fails**: Strategy doesn't have a real edge. Go back to optimization or strategy development.

### Step 4: Run Phase 2 - OOS Validation (Only Once!)

**New Workflow with Holdout Period**:

The recommended workflow now includes a **holdout period** (e.g., 2025) that is completely locked away during development and walk-forward validation. This ensures a truly independent final test.

```bash
# Walk-forward validation on development data (excluding holdout)
# Example: Development data = 2020-2024, Holdout = 2025
python3 scripts/run_oos_validation.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --wf-training-period "1 year" \
  --wf-test-period "6 months" \
  --market EURUSD
```

**With Holdout Period** (recommended):
```bash
# Exclude holdout period from walk-forward (e.g., 2025)
python3 scripts/run_oos_validation.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --wf-training-period "1 year" \
  --wf-test-period "6 months" \
  --wf-holdout-end-date "2025-01-01" \
  --market EURUSD
```

**⚠️ CRITICAL**: OOS data can only be used **once**. After this test, the system will prevent reusing the same OOS data.

**What it checks**:
- ✅ Mean test PF ≥ 1.5
- ✅ Consistency score ≥ 0.7
- ✅ Test PF ≥ 80% of Train PF (prevents overfitting)
- ✅ Walk-Forward Efficiency (WFE) ≥ 60% (test PF ≥ 60% of train PF)
- ✅ OOS Consistency Score ≥ 60% (% of periods with PF ≥ threshold)
- ✅ Binomial test p-value ≤ 0.1 (statistical significance)
- ✅ At least 3 test periods
- ✅ Minimum trades per period (default: 30) - periods with insufficient trades are excluded from stats

**New Metrics Explained**:

1. **Walk-Forward Efficiency (WFE)**: Measures how well test performance matches training performance. WFE = mean(test_pf / train_pf). Higher is better (100% = perfect, 60%+ is acceptable).

2. **OOS Consistency Score**: Percentage of test periods where PF ≥ threshold (default 1.5). Measures how consistently the strategy meets minimum performance.

3. **Binomial Test**: Statistical test to verify that the success rate (PF ≥ threshold) is significantly better than random chance (50%). Requires sufficient periods for power.

4. **Period Exclusion**: Periods with fewer than minimum trades (default 30) are excluded from statistical calculations but still recorded in reports for transparency.

**If Phase 2 passes**: Strategy is **qualified for final OOS test** (Step 4.5) or **live trading**! 🎉

**If Phase 2 fails**: Strategy doesn't work on unseen data. Go back to development.

### Step 4.5: Run Final OOS Test on Holdout Period (Optional, One-Time)

**⚠️ CRITICAL**: This test can **only be run ONCE** per strategy/data combination. The holdout period is permanently marked as used.

After Phase 2 passes, you can run a final one-time test on the holdout period (e.g., 2025):

```bash
# Final OOS test on holdout period (e.g., 2025)
python3 scripts/run_final_oos_test.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --holdout-start 2025-01-01 \
  --holdout-end 2025-12-31 \
  --market EURUSD
```

**Requirements**:
- ✅ Phase 1 (Training Validation) must have PASSED
- ✅ Phase 2 (OOS Validation) must have PASSED
- ✅ Holdout period must not have been tested before

**What it does**:
- Runs a single backtest on the holdout period
- Uses the same strategy configuration that passed Phase 2
- Records results permanently - no re-runs allowed
- Provides final validation before live trading

**If Final OOS Test passes**: Strategy is **fully qualified for live trading**! 🎉

**If Final OOS Test fails**: Review strategy performance and consider revisions before going live.

### Step 5: Run Phase 3 - Stationarity Analysis (Post-Live)

After the strategy has been live trading, determine when to retrain:

```bash
# Analyze stationarity on live data
python3 scripts/run_stationarity.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2024.csv \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --training-period-days 365 \
  --max-days 30
```

**What Phase 3 Tests**:

The enhanced stationarity test measures **the conditions that make your strategy profitable**, not just profit factor. This works for ALL strategy types (trend-following, mean reversion, etc.).

**1. Conditional Metrics** (What Makes PF Possible):
- **Signal Frequency**: Trades per day (even if not triggered)
- **Conditional Win Rate**: P(Win | Signal) - probability of winning given a signal
- **Payoff Ratio**: E[Win] / E[Loss] - average win vs average loss
- **Big Win Frequency**: Frequency of large wins (>2R) per period
- **Consecutive Losses**: Run analysis - average losses between wins

**2. Filter-Aware Regime Analysis**:
- Only tests regime conditions for **enabled filters** in your strategy config
- If ADX filter enabled → Tests performance when ADX > threshold vs ADX < threshold
- If ATR filter enabled → Tests performance by volatility quintiles
- If no regime filters enabled → Regime analysis is skipped

**3. Dual-Threshold Retraining Logic**:
- **Absolute Threshold**: PF < 1.5 (minimum acceptable) → Retrain immediately
- **Relative Threshold**: PF < 85% of baseline (3-window average) → Only retrain if approaching absolute threshold
- **Conditional Degradation**: Win rate drops 20%+, payoff collapses 30%+, big wins disappear → Retrain (only in stabilized windows)
- **3-Window Baseline**: Uses average of 3 stabilized windows (15-20, 20-25, 25-30 days) as baseline instead of day 1, reducing false signals

**Key Insight**: For sparse trend-following strategies, PF variance often reflects natural trade distribution variance, not degradation. The enhanced test distinguishes between:
- ✅ **Normal variance**: PF ranges from 1.7-4.16, but all above 1.5 → No retraining needed
- ❌ **Actual degradation**: Conditional metrics degrading, big wins disappearing → Retraining needed

**Output**: 
- Recommended retraining frequency with detailed reason
- PF range (min-max observed, filters out invalid/infinite values)
- Conditional metrics stability (checked only in baseline windows, not short windows)
- Regime-conditioned performance (if filters enabled)
- Comprehensive metric definitions explaining what each metric measures

## Pass/Fail Criteria

All criteria are configurable in `config/validation_criteria.yml`. Default values follow industry standards:

### Phase 1: Training Validation

```yaml
training:
  monte_carlo_p_value_max: 0.05  # 95% confidence
  training_min_profit_factor: 1.5
  training_min_sharpe: 1.0
  training_min_trades: 30
  sensitivity_max_cv: 0.3  # 30% variation allowed
```

### Phase 2: OOS Validation

```yaml
oos:
  # Basic Walk-Forward Criteria
  wf_min_test_pf: 1.5  # Minimum PF on test periods
  wf_min_consistency_score: 0.7  # 70% consistency across periods
  wf_max_pf_std: 0.5  # Maximum standard deviation of test PF
  wf_train_test_pf_ratio_min: 0.8  # Test PF ≥ 80% of Train PF (prevents overfitting)
  wf_min_test_periods: 3  # Minimum number of test periods
  
  # New Industry-Standard Metrics
  wf_min_wfe: 60.0  # Minimum Walk-Forward Efficiency (60% = test PF ≥ 60% of train PF)
  wf_min_oos_consistency: 60.0  # Minimum OOS Consistency Score (% of periods with PF ≥ threshold)
  wf_min_binomial_p_value: 0.1  # Maximum p-value for binomial test (statistical significance)
  wf_min_trades_per_period: 30  # Minimum trades per period (periods with fewer are excluded)
  
  # Overall OOS Performance
  oos_min_sharpe: 0.2  # Minimum Sharpe ratio on OOS data
  oos_max_drawdown_pct: 50.0  # Maximum acceptable drawdown (50%)
```

### Phase 3: Stationarity Analysis

```yaml
stationarity:
  # Dual-Threshold Retraining Logic
  degradation_threshold_pct: 0.85  # Relative threshold (85% of baseline PF)
  min_acceptable_pf: 1.5  # Absolute threshold (retrain if PF < this)
  min_analysis_periods: 5  # Minimum periods for analysis
  
  # Walk-Forward Window Configuration
  max_days: 30  # Maximum window size to test (default: 30 days)
  step_days: 1  # Step size for window testing (default: 1 day)
  
  # Baseline Windows (3-window approach for robust baseline)
  # Uses 3 stabilized windows and averages them as baseline
  # Recommended for sparse strategies (30 trades/month or less)
  baseline_window1_start: 15  # First baseline window start (days)
  baseline_window1_end: 20   # First baseline window end (days)
  baseline_window2_start: 20 # Second baseline window start (days)
  baseline_window2_end: 25   # Second baseline window end (days)
  baseline_window3_start: 25 # Third baseline window start (days)
  baseline_window3_end: 30   # Third baseline window end (days)
  
  # Conditional Metrics Thresholds
  win_rate_degradation_pct: 0.20  # Win rate drop threshold (20% = 0.20)
  payoff_degradation_pct: 0.30    # Payoff ratio drop threshold (30% = 0.30)
  big_win_freq_degradation_pct: 0.50  # Big win frequency drop threshold (50% = 0.50)
  stability_cv_threshold: 0.20    # Coefficient of variation for stability (20% = 0.20)
  
  # Big Win Detection
  big_win_threshold_r: 2.0  # R-multiple threshold for "big win" (default: 2.0)
```

**How It Works**:
- **3-Window Baseline Approach**: Uses 3 stabilized windows (15-20, 20-25, 25-30 days) averaged as baseline. Statistically sound and recommended for sparse strategies.
- Tests **conditional metrics** (win rate, payoff ratio, signal frequency) that make PF possible
- Only tests **enabled regime filters** (filter-aware analysis)
- Uses **dual thresholds**: Absolute (PF < 1.5) OR relative degradation approaching absolute
- **Does NOT retrain** if PF variance is just distribution noise but absolute performance remains acceptable
- **Stability Check**: Only evaluates stability in baseline windows (15-30 days), ignoring short-window variance (1-5 days) which is normal

### Adjusting Criteria

Edit `config/validation_criteria.yml` to adjust thresholds:

```bash
# Use custom criteria
python3 scripts/run_training_validation.py \
  --strategy ema_crossover \
  --criteria config/my_custom_criteria.yml \
  ...
```

## Validation State Tracking

The system tracks validation progress in `.validation_state/` to prevent:

- **OOS data reuse**: Once OOS data is used in Phase 2, it cannot be used again
- **Skipping phases**: Phase 2 requires Phase 1 to pass first
- **Data leakage**: Clear separation between training and OOS periods

## Common Issues

### "Phase 1 FAILED - Monte Carlo p-value too high"

**Problem**: Strategy results are not better than random chance.

**Solutions**:
- Strategy doesn't have a real edge - go back to development
- Need more trades (at least 30-50)
- Check for data issues or bugs
- Review which MC tests were suitable and which passed

### "Permutation Test: NOT SUITABLE"

**Problem**: Permutation test was skipped because it's not suitable for this strategy.

**Why this happens**:
- Returns are too uniform (mechanical strategies with identical exits)
- Sample size too small (< 30 trades)
- Returns too small relative to equity (compounding effects negligible)

**Solutions**:
- This is normal for certain strategy types
- Review the alternatives suggested (bootstrap, randomized_entry)
- Focus on the tests that were suitable
- The combined score only uses suitable tests

### "Phase 1 FAILED - Parameter sensitivity too high"

**Problem**: Strategy is too sensitive to parameter changes (not robust).

**Solutions**:
- Simplify strategy (fewer parameters)
- Use wider parameter ranges
- Strategy may be overfitted to specific parameters

### "OOS data has already been used!"

**Problem**: Trying to reuse OOS data that was already tested.

**Solutions**:
- Use different OOS data (different date range or file)
- This is by design - OOS should only be used once
- If you need to retest, use a different data file

### "Phase 2 FAILED - Test PF much lower than Train PF"

**Problem**: Overfitting - strategy only works on training data.

**Solutions**:
- Simplify strategy (fewer parameters)
- Reduce optimization
- Test on more diverse data
- Strategy may not have a real edge

### "Phase 1 must be completed first!"

**Problem**: Trying to run Phase 2 before Phase 1 passes.

**Solutions**:
- Run Phase 1 validation first
- Ensure Phase 1 passes all criteria
- Then proceed to Phase 2

## Best Practices

### 1. Strict Data Separation

- **Training data**: Use for optimization and Phase 1 validation
- **OOS data**: Use **only once** for Phase 2 validation
- **Never** optimize on OOS data
- **Never** reuse OOS data after Phase 2

### 2. Complete the Full Workflow

Don't skip phases:
1. ✅ Optimize on training data
2. ✅ Phase 1: Training validation
3. ✅ Phase 2: OOS validation
4. ✅ Phase 3: Stationarity (post-live)

### 3. Use Realistic Criteria

Adjust criteria in `config/validation_criteria.yml` based on:
- Your risk tolerance
- Market conditions
- Strategy type
- Historical performance

### 4. Document Everything

Keep records of:
- Which data was used for training vs OOS
- Optimization results
- Validation results
- Live performance

### 5. Regular Re-validation

Markets change. After going live:
- Monitor performance
- Run Phase 3 (stationarity) periodically
- Retrain when recommended
- Re-run Phase 1 and Phase 2 on new data if needed

## Validation Checklist

Before considering a strategy "validated":

- [ ] Parameters optimized on training data only
- [ ] Phase 1: Training validation PASSED
  - [ ] Training PF ≥ 1.5
  - [ ] Monte Carlo p-value < 0.05
  - [ ] Parameter sensitivity acceptable
- [ ] Phase 2: OOS validation PASSED
  - [ ] Test PF ≥ 1.5 consistently
  - [ ] Consistency score ≥ 0.7
  - [ ] Test PF ≥ 80% of Train PF
- [ ] Strategy qualified for live trading
- [ ] Phase 3: Stationarity analysis completed (post-live)
- [ ] Retraining schedule established

## When to Proceed

A strategy is ready for live trading when:

1. ✅ Phase 1 PASSED (has edge on training data)
2. ✅ Phase 2 PASSED (works on unseen OOS data)
3. ✅ All criteria met
4. ✅ Confidence in strategy robustness

**Remember**: It's better to reject a strategy than to trade a bad one. The validation system is designed to protect you from overfitting and false confidence.

## Understanding Enhanced Stationarity Testing

### Why Enhanced Testing?

Traditional stationarity tests (like Masters' approach) have limitations for sparse, regime-dependent strategies:

**Problem**: For trend-following strategies with 10 trades/month:
- PF variance (1.7-4.16) often reflects **natural trade distribution variance**, not degradation
- Some periods have the "big win", some don't → PF varies naturally
- Relative degradation (85% threshold) triggers false retraining signals

**Solution**: Enhanced test measures **conditions that make PF possible**, not just PF itself.

### What Gets Tested

**1. Conditional Metrics** (Universal - Works for ALL strategies):
- **Signal Frequency**: Are you still getting setups? (Even if not all trigger)
- **Conditional Win Rate**: P(Win | Signal) - Is your edge in pattern recognition stable?
- **Payoff Ratio**: E[Win] / E[Loss] - Is your ability to capture trends stable?
- **Big Win Frequency**: Are you still catching big moves?
- **Consecutive Losses**: Run analysis - Is the pattern of losses between wins stable?

**2. Filter-Aware Regime Analysis**:
- **Only tests enabled filters**: If your strategy uses ADX filter, tests performance when ADX > threshold
- **Regime-conditioned performance**: Shows PF by volatility quintile, trend strength, etc.
- **No filters enabled?**: Regime analysis is skipped (not relevant)

**3. Dual-Threshold Logic**:
- **Absolute**: PF < 1.5 → Retrain (strategy no longer profitable)
- **Relative**: PF < 85% of baseline (3-window average) → Only retrain if approaching absolute threshold
- **Conditional**: Win rate drops 20%+, payoff collapses 30%+ → Retrain (edge eroding, only checked in stabilized windows)
- **3-Window Baseline**: Averages windows 15-20, 20-25, 25-30 days as baseline. More robust than using day 1 (which has high variance).

### Example: Your Case

**Your Results**:
- PF Range: 1.7 - 4.16 (all above 1.5)
- Recommended: 8 days (based on relative drop from 4.16 to 3.12)

**Enhanced Test Interpretation**:
- ✅ **Absolute performance acceptable**: All PF values above 1.5
- ✅ **Conditional metrics stable**: Win rate, payoff ratio consistent
- ✅ **No degradation detected**: Strategy remains profitable
- **Conclusion**: **Retraining NOT required** - The 8-day recommendation is ignored because absolute performance remains acceptable

**Why This is Correct**:
- PF variance (1.7-4.16) is just **distribution noise** (some periods have big win, some don't)
- Strategy hasn't degraded - it's still profitable (PF > 1.5)
- Conditional metrics (what makes PF possible) remain stable

### Reading Enhanced Stationarity Reports

**Key Sections**:

1. **Retraining Recommendation**:
   - Shows days and **reason**
   - "No degradation detected" → Don't retrain
   - "Absolute threshold breached" → Retrain immediately
   - "Conditional metrics degraded" → Edge eroding, retrain

2. **PF Range**:
   - Min-Max observed PF
   - If all above 1.5 → Strategy remains profitable

3. **Conditional Metrics Charts**:
   - X-axis: **Walk-Forward Window Size (days)** - NOT "days since training"
   - Win rate over window sizes
   - Payoff ratio over window sizes
   - Signal frequency over window sizes
   - Big win frequency over window sizes
   - **Note**: Short windows (1-5 days) show high variance (normal). Metrics converge by day 15.
   - **Stable lines in baseline windows (15-30 days)** = Edge persists
   - **Declining lines in baseline windows** = Edge eroding

4. **Regime-Conditioned Performance** (if filters enabled):
   - Performance by volatility quintile
   - Performance by trend strength (ADX)
   - Shows if strategy works in expected regimes

## Advanced Usage

### Custom Validation Criteria

Create your own criteria file:

```yaml
# config/my_criteria.yml
training:
  monte_carlo_p_value_max: 0.01  # Stricter: 99% confidence
  training_min_profit_factor: 2.0  # Higher bar

oos:
  wf_min_test_pf: 2.0  # Higher requirement
  wf_min_consistency_score: 0.8  # More consistent

stationarity:
  min_acceptable_pf: 1.8  # Stricter absolute threshold
  degradation_threshold_pct: 0.80  # Stricter relative threshold (80%)
  
  # For sparse strategies (10-30 trades/month), use longer windows
  max_days: 60
  baseline_window1_start: 30
  baseline_window1_end: 40
  baseline_window2_start: 40
  baseline_window2_end: 50
  baseline_window3_start: 50
  baseline_window3_end: 60
  
  # Adjust thresholds for your strategy
  payoff_degradation_pct: 0.25  # Stricter: 25% drop triggers retrain
  stability_cv_threshold: 0.15  # Stricter: 15% CV for stability
```

Use it:
```bash
python3 scripts/run_training_validation.py \
  --criteria config/my_criteria.yml \
  ...
```

### Understanding Stationarity Test Configuration

**Why 3-Window Baseline?**
- **Statistical Soundness**: Averaging multiple windows reduces sampling error (industry standard)
- **Sparse Strategy Support**: For strategies with ~30 trades/month, longer windows (15-30 days) provide more trades per window
- **Robust Baseline**: Day 1 has high variance due to short windows - not a good baseline. Using days 15-30 gives stable reference.

**Window Size Guidelines**:
- **Frequent strategies** (100+ trades/month): Default windows (15-30 days) work well
- **Sparse strategies** (10-30 trades/month): Use longer windows (30-60 days) for more stable metrics
- **Very sparse** (<10 trades/month): Consider 60-90 day windows

**Stability Check**:
- Only evaluates stability in **baseline windows** (15-30 days by default)
- **Ignores short-window variance** (days 1-5) which is normal and expected
- Uses coefficient of variation (CV) threshold (default: 20%) to determine stability
- Metrics are stable if CV < threshold across all conditional metrics in baseline windows

### Parameter Sensitivity Analysis

Test multiple parameters:

```bash
python3 scripts/run_training_validation.py \
  --strategy ema_crossover \
  --sensitivity-param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \
  --sensitivity-param moving_averages.ema5.length 5 10 15 \
  ...
```

### Walk-Forward Configuration

Customize walk-forward analysis:

```bash
python3 scripts/run_oos_validation.py \
  --wf-training-period "2 years" \
  --wf-test-period "3 months" \
  --wf-window-type rolling \
  ...
```

## Benchmarking Against External Platforms

Trading Lab includes a benchmarking tool to verify engine equivalence and metric parity against external platforms like MT5, NinjaTrader, and TradingView.

### Why Benchmark?

Benchmarking ensures:
- **Engine Correctness**: Trading Lab generates the same trades as external platforms
- **Metric Accuracy**: Calculated metrics (PF, Sharpe, etc.) match external platforms
- **Confidence**: One-time verification that your backtest engine is correct

**Important**: Benchmarking is about **engine equivalence**, not performance comparison. You're verifying that Trading Lab produces the same results, not trying to "beat" other platforms.

### How It Works

1. **Run strategy in external platform** (MT5, NinjaTrader, TradingView)
2. **Export trades** to CSV/Excel
3. **Run same strategy in Trading Lab** with same data and parameters
4. **Compare trades trade-by-trade** (≥99.5% match target)
5. **Compare metrics** (PF, Sharpe, etc. within tolerance)

### Running a Benchmark

```bash
python3 scripts/run_benchmark.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_M15_2021_2025.csv \
  --external-trades mt5_trades.csv \
  --platform mt5 \
  --start-date 2021-01-01 \
  --end-date 2023-12-31
```

**Parameters**:
- `--strategy`: Strategy name (must match folder in `strategies/`)
- `--data`: Market data file (same as used in external platform)
- `--external-trades`: Path to external platform trade export (CSV)
- `--platform`: Platform name (`mt5`, `ninjatrader`, `tradingview`)
- `--start-date` / `--end-date`: Date range (must match external platform)
- `--time-tolerance`: Time tolerance for trade matching in seconds (default: 60)
- `--price-tolerance-pct`: Price tolerance percentage (default: 0.01%)
- `--metric-tolerance-pct`: Metric comparison tolerance (default: 5.0%)

### Exporting Trades from External Platforms

#### MT5 (MetaTrader 5)
1. Run Strategy Tester
2. Right-click on results → "Export" or "Save Report"
3. Export as CSV
4. Ensure columns include: Entry Time, Entry Price, Exit Time, Exit Price, Volume, Type, Profit

#### NinjaTrader
1. Run Strategy Analyzer
2. Right-click on results → "Export"
3. Export as CSV
4. Ensure columns include: Entry Time, Exit Time, Entry Price, Exit Price, Quantity, Direction, P&L

#### TradingView
1. Run Strategy Tester
2. Click "Export" button
3. Export as CSV
4. Ensure columns include: Entry Time, Exit Time, Entry Price, Exit Price, Quantity, Direction, P&L

### Understanding Results

**Trade Comparison**:
- **Match Rate**: Percentage of trades that match between platforms (target: ≥99.5%)
- **Matched Trades**: Number of trades that matched
- **Unmatched Trades**: Trades that don't match (investigate differences)

**Metric Comparison**:
- **Metrics Match**: Whether all critical metrics are within tolerance
- **Differences**: Shows percentage difference for each metric

**Overall Result**:
- ✅ **EQUIVALENT**: Trading Lab matches external platform (engine is correct)
- ❌ **NOT EQUIVALENT**: Differences detected (investigate causes)

### Common Differences and Causes

**Acceptable Differences**:
- **Bar close assumptions**: Different platforms may use different bar close times
- **Fill model differences**: Slippage/commission models may differ
- **Rounding**: Small price/quantity rounding differences

**Unacceptable Differences**:
- **Different trade count**: Strategy logic may differ
- **Large price differences**: Data source or calculation error
- **Metric mismatch**: Calculation error in one platform

### When to Benchmark

- **One-time verification**: After implementing a new strategy
- **After engine changes**: When modifying backtest engine logic
- **Before live trading**: Final verification of engine correctness

**Note**: Once engine equivalence is proven, you don't need to keep comparing. Trust your validated engine.

## Summary

The phased validation methodology ensures:

1. **Edge exists** (Phase 1: Monte Carlo proves it's better than random)
2. **Edge persists** (Phase 2: Works on unseen data)
3. **Edge degrades predictably** (Phase 3: Know when to retrain)

By following this workflow, you avoid:
- ❌ Overfitting
- ❌ Data leakage
- ❌ False confidence
- ❌ Edge erosion surprises

**The system enforces best practices so you don't have to remember them.**
