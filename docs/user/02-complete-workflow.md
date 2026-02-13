# Complete Strategy Development Workflow

This guide provides a step-by-step workflow for developing, testing, and validating trading strategies using Trading Lab. Follow this workflow to ensure your strategies are robust and ready for live trading.

## Overview

The Trading Lab workflow consists of three main phases:

1. **Development Phase**: Create and test your strategy idea
2. **Validation Phase**: Rigorously test for robustness and edge
3. **Live Phase**: Deploy and monitor (when ready)

Each phase has specific steps and criteria that must be met before proceeding.

**Recommended standard workflow**:
- Use the standardized split policy (`config/data_splits.yml`) and config profiles so Phase 1, Phase 2, and the final holdout are reproducible and comparable.
- See: [Golden Path: Split Policy + Config Profiles](11-split-policy-and-profiles.md)

---

## Phase 1: Development

### Step 1.1: Get Your Data

**Goal**: Obtain historical data for your target market

**Actions**:
1. Identify your target market (e.g., BTCUSDT, EURUSD, AAPL)
2. Download historical data (see [Data Management Guide](03-data-management.md))

**Example - Downloading Data**:
```bash
# Download BTCUSDT 15-minute data for 2020-2024
python scripts/download_data.py \
  --symbol BTCUSDT \
  --interval 15m \
  --start 2020-01-01 \
  --end 2024-12-31 \
  --output data/raw/BTCUSDT-15m-2020-2024.parquet
```

**Example - Consolidating Multiple Files**:
If you have yearly files (2020.csv, 2021.csv, 2022.csv):
```bash
python scripts/consolidate_data.py \
  --input "data/raw/EURUSD_*.csv" \
  --output data/raw/EURUSD_M15_2020_2025.csv
```

#### Common data pitfalls (manual/chunked downloads)

If you manually download market data in chunks (daily/monthly/yearly files), backtests over many years can break or become misleading unless you standardize the dataset.

**1) Multiple files** (yearly/monthly chunks)

- Backtests normally expect a single file.
- You can consolidate first (recommended for reproducibility):

```bash
python3 scripts/consolidate_data.py --input "data/raw/BTCUSDT-15m-*.parquet" --output data/processed/BTCUSDT-15m.parquet
python3 scripts/run_backtest.py --strategy my_strategy --data data/processed/BTCUSDT-15m.parquet
```

- Or let the runner consolidate automatically:

```bash
python3 scripts/run_backtest.py --strategy my_strategy --data "data/raw/BTCUSDT-15m-*.parquet" --auto-consolidate
```

**2) Mixed-frequency files** (hourly + daily + monthly)

- Don’t merge different bar intervals directly.
- Resample to a single target timeframe first, then consolidate.

```bash
python3 scripts/resample_ohlcv.py --src data/raw/BTCUSDT-1h-2021.parquet --dst data/raw/BTCUSDT-4h-2021.parquet --rule 4h
python3 scripts/consolidate_data.py --input "data/raw/BTCUSDT-4h-*.parquet" --output data/processed/BTCUSDT-4h.parquet
```

**3) Missing candles (gaps)**

If your dataset has gaps and you want a perfect grid for research/validation:

```bash
python3 scripts/fill_timegrid.py --input data/processed/BTCUSDT-4h.parquet --output data/processed/BTCUSDT-4h-filled.parquet --freq 4h
```

See also: [Data Management Guide](03-data-management.md)

**Checkpoint**: ✅ You have data file(s) in `data/raw/` directory

---

### Step 1.2: Create Your Strategy

**Goal**: Implement your trading idea as a strategy

**Actions**:
1. Create strategy folder: `strategies/my_strategy/`
2. Create `config.yml` with your parameters
3. Create `strategy.py` with your trading logic

**Example - Strategy Structure**:
```
strategies/
  my_strategy/
    config.yml      # Strategy configuration
    strategy.py     # Trading logic
    README.md       # Optional documentation
```

**Detailed Guide**: See [Strategy Development Guide](13-strategy-development.md)

**Checkpoint**: ✅ Strategy code compiles without errors

---

### Step 1.3: Initial Backtest

**Goal**: Verify your strategy works and generates trades

**Actions**:
1. Run a backtest on a subset of your data
2. Check that trades are generated
3. Review basic metrics

**Example - Running Initial Backtest**:
```bash
# Test on 2020-2021 data (training period)
python scripts/run_backtest.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --start-date 2020-01-01 \
  --end-date 2021-12-31
```

**What to Look For**:
- ✅ Strategy generates trades (not 0 trades)
- ✅ No errors in execution
- ✅ Reasonable metrics (don't worry about perfection yet)

**If Issues**:
- 0 trades: Check signal generation logic
- Errors: Review error messages and fix code
- Unrealistic metrics: Check position sizing and capital tracking

**Checkpoint**: ✅ Strategy runs and generates trades

---

### Step 1.4: Review Initial Results

**Goal**: Understand what your strategy is doing

**Actions**:
1. Open the HTML report in `reports/`
2. Review equity curve
3. Check key metrics

**Key Metrics to Review**:
- **Total Trades**: Should be reasonable (not too few, not too many)
- **Win Rate**: Percentage of winning trades
- **Profit Factor**: Gross profit / Gross loss (should be > 1.0)
- **Max Drawdown**: Largest peak-to-trough decline (lower is better)
- **Equity Curve**: Should trend upward (if profitable)

**Example - Good Initial Results**:
```
Total Trades: 245
Win Rate: 52.00%
Profit Factor: 1.85
Max Drawdown: 18.50%
```

**Example - Needs Improvement**:
```
Total Trades: 12        # Too few trades
Win Rate: 25.00%        # Too low
Profit Factor: 0.65     # Losing strategy
Max Drawdown: 85.00%    # Too high
```

**Decision Point**:
- ✅ If results look reasonable → Proceed to Step 1.5
- ❌ If results are poor → Revise strategy logic, then repeat Step 1.3

**Checkpoint**: ✅ You understand your strategy's behavior

---

### Step 1.5: Optimize Parameters (Optional)

**Goal**: Find optimal parameter values for your strategy

**Actions**:
1. Identify parameters to optimize (e.g., risk_per_trade_pct, MA lengths)
2. Run optimization script
3. Review results and select best parameters

**Example - Optimizing Risk Per Trade**:
```bash
python scripts/optimize_strategy.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 2.5
```

**Example - Optimizing Multiple Parameters**:
```bash
python scripts/optimize_strategy.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \
  --param moving_averages.ema5.length 3 5 7 10
```

**What to Look For**:
- Best profit factor
- Reasonable drawdown
- Stable across parameter range (not just one value)

**Important**: 
- ⚠️ Don't over-optimize (can lead to overfitting)
- ⚠️ Test on training data only (save OOS data for Phase 2)
- ⚠️ If optimization doesn't improve results much, your strategy might not have an edge

**Checkpoint**: ✅ You have optimized parameters (or decided optimization isn't needed)

---

## Phase 2: Validation

**Critical**: Phase 2 validation is where most strategies fail. This is by design - it filters out strategies that don't have a real edge.

### Step 2.1: Phase 1 - Training Validation

**Goal**: Prove your strategy has a real edge (not just luck)

**Actions**:
1. Run Phase 1 validation on training data
2. Review Monte Carlo suite results (universal gating + conditional diagnostics)
3. Check if strategy passes all criteria

**Example - Running Phase 1 Validation**:
```bash
python scripts/run_training_validation.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --market BTCUSDT
```

**What Phase 1 Tests**:
1. **Backtest Quality**: Minimum profit factor, Sharpe ratio, trade count
2. **Monte Carlo Suite**: Suitability-aware robustness tests
  - **Universal tests (used for pass/fail if suitable)**: permutation, block bootstrap
  - **Conditional tests (informational by default)**: randomized-entry baseline
3. **Parameter Sensitivity**: Tests if strategy is robust to parameter changes

**Understanding Results**:

**Monte Carlo Suite (high level)**:
- **Suitability**: Some tests are skipped if they would be misleading for the current strategy/dataset.
- **Universal combined robustness (gating)**: Computed from suitable **universal** tests only.
- **All-suitable combined robustness (informational)**: Includes both universal + conditional tests that ran.

**Example - Passing Phase 1**:
```
Training Backtest:
  Profit Factor: 2.15
  Sharpe Ratio: 1.85
  Total Trades: 245

Monte Carlo Suite (Universal / Gating):
  Combined: score=0.72, percentile=84.1% ✓ PASS
  Permutation: ✓ PASS
  Block bootstrap: ✓ PASS
  Randomized entry (Conditional / Info): skipped or informational

✅ Phase 1 PASSED
```

**Example - Failing Phase 1**:
```
Training Backtest:
  Profit Factor: 2.79
  Sharpe Ratio: 3.26
  Total Trades: 506

Monte Carlo Suite (Universal / Gating):
  Combined: score=0.41, percentile=55.0% ✗ FAIL
  Permutation: ✗ FAIL
  Block bootstrap: ✓ PASS
  Randomized entry (Conditional / Info): informational

❌ Phase 1 FAILED
Failure reasons:
  - Universal Monte Carlo robustness score too low
  - Universal Monte Carlo percentile too low
  - Not enough suitable universal MC tests passed (majority rule)
```

**What This Means**:
- Even though the backtest shows PF 2.79, the Monte Carlo test shows it's not better than random
- This strategy is likely overfitted or got lucky with trade sequencing
- **Do not proceed to Phase 2** - the strategy doesn't have a real edge

**Decision Point**:
- ✅ **PASS**: Proceed to Step 2.2 (OOS Validation)
  - **Important**: Once Phase 1 passes, it cannot be re-run (prevents data mining)
  - If you want to test different parameters, create a new strategy variant
- ❌ **FAIL**: 
  - Strategy doesn't have a real edge on training data
  - **You can re-run Phase 1 after reoptimization**:
    - Reoptimize parameters (Step 1.5)
    - Update strategy configuration
    - Run Phase 1 again - it will automatically allow re-running if it previously failed
  - **Best Practice**: 
    - If Phase 1 fails after 2-3 reoptimization attempts, consider:
      - Strategy logic may be flawed
      - Market conditions may not suit the strategy
      - Strategy may not have a real edge
    - After multiple failures, it's often better to try a different approach
  - **Do not proceed to Phase 2** until Phase 1 passes

**Checkpoint**: ✅ Phase 1 validation passed

---

### Step 2.2: Phase 2 - Out-of-Sample Validation

**Goal**: Test strategy on completely unseen data (one-time use)

**Recommended Workflow with Holdout Period**:

The industry-standard approach is to **lock away a holdout period** (e.g., 2025) during development and walk-forward validation, then test on it once as a final validation step.

**Example - Walk-Forward on Development Data (Excluding Holdout)**:
```bash
# Development data: 2020-2024, Holdout: 2025 (locked away)
python scripts/run_oos_validation.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2025.parquet \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --wf-training-period "1 year" \
  --wf-test-period "6 months" \
  --wf-holdout-end-date "2025-01-01" \
  --market BTCUSDT
```

**Alternative - Traditional OOS Validation** (without holdout):
```bash
# OOS data: 2022-2024 (not used in training)
python scripts/run_oos_validation.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --start-date 2020-01-01 \
  --end-date 2024-12-31 \
  --wf-training-period "1 year" \
  --wf-test-period "6 months" \
  --market BTCUSDT
```

**What Phase 2 Tests**:
- **Walk-Forward Analysis**: Tests strategy on unseen data with expanding/rolling windows
- **Walk-Forward Efficiency (WFE)**: Measures test PF vs train PF ratio (≥60% required)
- **OOS Consistency Score**: % of periods with PF ≥ threshold (≥60% required)
- **Binomial Test**: Statistical significance of success rate (p-value ≤ 0.1)
- **Consistency**: Checks if performance is consistent across periods
- **Overfitting Detection**: Compares test PF to train PF (test ≥ 80% of train)
- **Period Exclusion**: Periods with <30 trades are excluded from stats but still recorded

**Understanding Results**:

**Walk-Forward Analysis**:
- Multiple train/test cycles on OOS data
- Each cycle: train on one period, test on next period
- Measures consistency across periods

**Example - Passing Phase 2**:
```
Walk-Forward Analysis:
  Total Steps: 5
  Included Steps: 5
  Excluded Steps: 0
  Mean Test PF: 1.85
  Consistency Score: 0.75
  Walk-Forward Efficiency: 78.5%
  OOS Consistency Score: 80.0% (4/5 periods ≥ 1.5 PF)
  Binomial p-value: 0.0625

Criteria Checks:
  min_test_pf: ✓ PASS
  min_consistency: ✓ PASS
  min_wfe: ✓ PASS (78.5% ≥ 60%)
  min_oos_consistency: ✓ PASS (80% ≥ 60%)
  binomial_test: ✓ PASS (0.0625 ≤ 0.1)

✅ Phase 2 PASSED
```

**Example - Failing Phase 2**:
```
Walk-Forward Analysis:
  Total Steps: 5
  Mean Test PF: 0.95
  Consistency Score: 0.35

Criteria Checks:
  min_test_pf: ✗ FAIL (0.95 < 1.2)
  min_consistency: ✗ FAIL (0.35 < 0.5)

❌ Phase 2 FAILED
Failure reasons:
  - OOS profit factor too low
  - Strategy not consistent across periods
```

**What This Means**:
- Strategy performed well on training data but failed on OOS data
- This indicates overfitting - strategy learned training data patterns that don't generalize
- **Do not proceed to live trading**

**Important**: 
- ⚠️ OOS data can only be used **once**
- ⚠️ If you fail Phase 2, you cannot retest on the same OOS data
- ⚠️ You must get new OOS data or revise strategy significantly

**Decision Point**:
- ✅ **PASS**: Strategy qualifies for live testing → Proceed to Phase 3
- ❌ **FAIL**: 
  - Strategy is overfitted
  - Revise strategy or try different approach
  - **Do not proceed** to live trading

**Checkpoint**: ✅ Phase 2 validation passed

---

### Step 2.2a: Final OOS Test on Holdout Period (Optional, One-Time)

**Goal**: Run a final one-time test on the locked-away holdout period

**⚠️ CRITICAL**: This test can **only be run ONCE** per strategy/data combination. The holdout period is permanently marked as used.

**Actions**:
1. Ensure Phase 1 and Phase 2 have both PASSED
2. Run final OOS test on holdout period (e.g., 2025)
3. Review final results

**Example - Running Final OOS Test**:
```bash
# Final test on holdout period (e.g., 2025)
python scripts/run_final_oos_test.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2025.parquet \
  --holdout-start 2025-01-01 \
  --holdout-end 2025-12-31 \
  --market BTCUSDT
```

**Requirements**:
- ✅ Phase 1 (Training Validation) must have PASSED
- ✅ Phase 2 (OOS Validation) must have PASSED
- ✅ Holdout period must not have been tested before

**What It Does**:
- Runs a single backtest on the holdout period
- Uses the same strategy configuration that passed Phase 2
- Records results permanently - no re-runs allowed
- Provides final validation before live trading

**Understanding Results**:
- Compare holdout results to walk-forward expectations
- If results are similar → Strategy is robust ✅
- If results are worse → Review strategy before going live ⚠️

**Decision Point**:
- ✅ **PASS**: Strategy is **fully qualified for live trading**! 🎉
- ❌ **FAIL**: Review strategy performance and consider revisions

**Checkpoint**: ✅ Final OOS test completed (if using holdout workflow)

---

### Step 2.3: Phase 3 - Stationarity Analysis (Post-Live)

**Goal**: Determine when to retrain/optimize strategy (after going live)

**Actions**:
1. Run stationarity analysis on live trading results
2. Review recommended retraining frequency
3. Plan optimization schedule

**Example - Running Stationarity Analysis**:
```bash
python scripts/run_stationarity.py \
  --strategy my_strategy \
  --data data/raw/BTCUSDT-15m-2020-2024.parquet \
  --market BTCUSDT
```

**When to Run**:
- After accumulating live trading data (e.g., 3-6 months)
- Periodically to check if strategy edge is eroding
- Before deciding to retrain/optimize

**Understanding Results**:
- **Enhanced Stationarity Analysis**: Tests conditions that make strategy profitable (not just PF)
- **3-Window Baseline**: Uses robust baseline from 3 stabilized windows (15-20, 20-25, 25-30 days)
- **Conditional Metrics**: Win rate, payoff ratio, signal frequency, big win frequency
- **Filter-Aware Regime Analysis**: Only tests enabled regime filters
- **Dual-Threshold Logic**: Absolute (PF < 1.5) + relative degradation
- **Recommended Retraining**: Suggests when to re-optimize with detailed reason
- **Edge Erosion Detection**: Alerts if conditional metrics degrade in stabilized windows
- **Metric Definitions**: Comprehensive explanations of all metrics in the report

**Checkpoint**: ✅ You understand when to retrain strategy

---

## Phase 3: Live Trading

**Prerequisites**:
- ✅ Phase 1 validation passed
- ✅ Phase 2 validation passed
- ✅ Strategy is qualified for live trading

### Step 3.1: Final Pre-Live Checklist

**Goal**: Ensure everything is ready for live trading

**Checklist**:
- [ ] Strategy passed Phase 1 and Phase 2 validation
- [ ] Risk parameters are appropriate for your capital
- [ ] Commission and slippage settings match your broker
- [ ] You understand the strategy's behavior
- [ ] You have a plan for monitoring and risk management
- [ ] You've tested with paper trading (if available)

**Review**:
- [Live Trading Guide](LIVE_TRADING.md) for detailed setup
- [Configuration Guide](CONFIGURATION.md) for risk settings
- [Reports Guide](REPORTS.md) for monitoring

**Checkpoint**: ✅ All checklist items completed

---

### Step 3.2: Deploy Strategy

**Goal**: Start live trading

**Actions**:
1. Configure live trading settings
2. Set up monitoring
3. Start execution

**Detailed Guide**: See [Live Trading Guide](LIVE_TRADING.md)

**Checkpoint**: ✅ Strategy is running live

---

### Step 3.3: Monitor and Maintain

**Goal**: Ensure strategy continues to perform

**Actions**:
1. Monitor performance regularly
2. Compare live results to backtest expectations
3. Run Phase 3 (Stationarity) periodically
4. Retrain/optimize when recommended

**Key Metrics to Monitor**:
- Profit factor (should match backtest)
- Win rate (should match backtest)
- Drawdown (should not exceed backtest max)
- Trade frequency (should match backtest)

**Red Flags**:
- Performance significantly worse than backtest
- Drawdown exceeding backtest max
- Strategy not generating expected trades
- Market conditions changed significantly

**Checkpoint**: ✅ Strategy is monitored and maintained

---

## Complete Workflow Summary

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: DEVELOPMENT                                     │
├─────────────────────────────────────────────────────────┤
│ 1.1 Get Data                                            │
│ 1.2 Create Strategy                                     │
│ 1.3 Initial Backtest                                    │
│ 1.4 Review Results                                      │
│ 1.5 Optimize Parameters (Optional)                      │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 2: VALIDATION                                     │
├─────────────────────────────────────────────────────────┤
│ 2.1 Phase 1: Training Validation                       │
│     ├─ Backtest Quality                                 │
│     ├─ Monte Carlo Suite (Universal + Conditional)      │
│     └─ Parameter Sensitivity                            │
│                                                          │
│     ✅ PASS → Continue                                  │
│     ❌ FAIL → Stop (No Real Edge)                       │
│                                                          │
│ 2.2 Phase 2: OOS Validation                            │
│     ├─ Walk-Forward Analysis                            │
│     └─ Consistency Check                                │
│                                                          │
│     ✅ PASS → Qualified for Live                       │
│     ❌ FAIL → Stop (Overfitted)                         │
│                                                          │
│ 2.3 Phase 3: Stationarity (Post-Live)                   │
│     └─ Retraining Schedule                             │
└─────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────┐
│ PHASE 3: LIVE TRADING                                   │
├─────────────────────────────────────────────────────────┤
│ 3.1 Pre-Live Checklist                                  │
│ 3.2 Deploy Strategy                                     │
│ 3.3 Monitor and Maintain                                │
└─────────────────────────────────────────────────────────┘
```

---

## Common Workflow Patterns

### Pattern 1: Quick Strategy Test

**Use Case**: You have an idea and want to quickly test it

**Workflow**:
1. Get data (Step 1.1)
2. Create strategy (Step 1.2)
3. Initial backtest (Step 1.3)
4. Review results (Step 1.4)
5. **Decision**: If results look good → Full validation, else → Revise or abandon

**Time**: 1-2 hours

---

### Pattern 2: Full Strategy Development

**Use Case**: You want to develop a strategy from scratch and validate it properly

**Workflow**:
1. Complete Phase 1 (Development)
2. Complete Phase 2 (Validation)
3. If passes → Proceed to Phase 3 (Live)

**Time**: Days to weeks

---

### Pattern 3: Strategy Optimization

**Use Case**: You have a working strategy and want to improve it

**Workflow**:
1. Run optimization (Step 1.5)
2. Update config with best parameters
3. Re-run Phase 1 validation
4. If still passes → Re-run Phase 2 (if OOS data available)

**Time**: Hours to days

---

## Decision Points and Gates

### Gate 1: After Initial Backtest
- **Question**: Does strategy generate trades and show promise?
- **Pass**: Proceed to optimization/validation
- **Fail**: Revise strategy logic

### Gate 2: After Phase 1 Validation
- **Question**: Does strategy have a real edge (not luck)?
- **Pass**: Proceed to Phase 2
- **Fail**: Strategy doesn't have edge - stop here

### Gate 3: After Phase 2 Validation
- **Question**: Does strategy work on unseen data?
- **Pass**: Qualified for live trading
- **Fail**: Strategy is overfitted - stop here

### Gate 4: Before Live Trading
- **Question**: Is everything ready and understood?
- **Pass**: Deploy strategy
- **Fail**: Complete checklist items

---

## Tips for Success

1. **Follow the Workflow**: Don't skip steps - each phase has a purpose
2. **Be Patient**: Good strategies take time to develop and validate
3. **Accept Failure**: Most strategies will fail validation - that's normal
4. **Don't Cheat**: Don't use OOS data for training or retest on same OOS data
5. **Document Everything**: Keep notes on what works and what doesn't
6. **Start Simple**: Begin with basic strategies, add complexity gradually
7. **Understand Your Strategy**: Know why it works, not just that it works

---

## Troubleshooting

### "Strategy generates 0 trades"
- Check signal generation logic
- Verify data has enough history for indicators
- Check if filters are too restrictive

### "Phase 1 validation fails"
- **You can re-run Phase 1 after reoptimization**:
  1. Reoptimize parameters (see Step 1.5)
  2. Update strategy configuration with new parameters
  3. Run Phase 1 validation again - it will automatically allow re-running if it previously failed
- **If it continues to fail after 2-3 attempts**:
  - Strategy likely doesn't have a real edge
  - Consider different approach or market
  - Review Monte Carlo results for insights
  - May be time to try a different strategy idea

### "Phase 1 already completed and PASSED"
- Once Phase 1 passes, it cannot be re-run (prevents data mining)
- If you want to test different parameters:
  - Create a new strategy variant (e.g., `my_strategy_v2`)
  - Or use a different data file
- Proceed to Phase 2 (OOS Validation) instead

### "Phase 2 validation fails"
- Strategy is overfitted to training data
- **Important**: OOS data can only be used once - you cannot retest on the same OOS data
- Options:
  - Simplify strategy or use different parameters (then re-run Phase 1, then get NEW OOS data)
  - Get new OOS data (different time period) if you want to retest
  - Consider that the strategy may not be robust enough for live trading

### "Results don't match backtest"
- Check commission and slippage settings
- Verify market conditions haven't changed
- Review live execution vs. backtest assumptions

---

## Next Steps

- **New to Trading Lab?**: Start with [Getting Started Guide](GETTING_STARTED.md)
- **Creating Strategies?**: See [Strategy Development Guide](13-strategy-development.md)
- **Understanding Results?**: See [Reports Guide](REPORTS.md)
- **Need Help?**: Review relevant guides in `/docs`

---

## Summary

This workflow ensures:
- ✅ Strategies are properly developed
- ✅ Strategies are rigorously validated
- ✅ Only strategies with real edges proceed to live trading
- ✅ Time is not wasted on strategies that won't work

**Remember**: The validation phases are designed to be strict. If your strategy fails, it's better to know early than to lose money in live trading.

Good luck with your strategy development!

