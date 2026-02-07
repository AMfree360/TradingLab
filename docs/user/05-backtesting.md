# Backtesting Guide

Backtesting is the process of testing your trading strategy on historical data to see how it would have performed. This guide will teach you everything you need to know about backtesting in Trading Lab.

**💡 Tip**: For a complete step-by-step workflow from development to validation, see the [Complete Workflow Guide](02-complete-workflow.md).

## What is Backtesting?

Backtesting simulates trading your strategy on past market data. It tells you:
- How much profit/loss you would have made
- How many trades were winners vs losers
- What your maximum drawdown was
- And much more!

**Important**: Past performance doesn't guarantee future results, but backtesting helps you understand if your strategy has potential.

## Execution timing model (important)

Trading Lab uses an execution realism contract designed to reduce lookahead bias:

- **Decision time**: strategy logic is evaluated at the **bar close**.
- **Fill time (default)**: if a trade is approved, the engine fills at the **next bar open**.

This means a signal that appears on the close of bar $t$ cannot be filled at bar $t$'s open (because that price was already in the past).

## Running a Backtest

### Basic Command

The simplest way to run a backtest:

```bash
python scripts/run_backtest.py --strategy ema_crossover --data data/raw/BTCUSDT_1m.csv
```

This will:
1. Load your strategy
2. Load historical data
3. Simulate trading
4. Generate a report

### Command Options

You can customize your backtest with these options:

```bash
python scripts/run_backtest.py \
  --strategy ema_crossover \
  --data data/raw/BTCUSDT_1m.csv \
  --capital 10000 \
  --commission 0.0004 \
  --slippage 0.0 \
  --output my_backtest_report
```

**Options explained**:
- `--strategy`: Name of your strategy folder
- `--data`: Path to your data file
- `--capital`: Starting capital (default: 10000)
- `--commission`: Commission rate per trade (default: 0.0004 = 0.04%)
- `--slippage`: Slippage in price units (default: 0.0)
- `--output`: Name for the report (optional)
- `--start-date`: Start date for backtest (YYYY-MM-DD). Filters data to this date and later
- `--end-date`: End date for backtest (YYYY-MM-DD). Filters data up to this date

### Date Range Filtering

You can specify a date range to test on a subset of your data, reserving the rest for out-of-sample (OOS) testing:

```bash
# Test on 2020-2021, keeping 2022-2025 for OOS validation
python scripts/run_backtest.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --start-date 2020-01-01 \
  --end-date 2021-12-31
```

**Why use date filtering?**
- **Reserve data for OOS testing**: Keep later data unseen for final validation
- **Test specific periods**: Focus on particular market conditions
- **Compare periods**: Test how strategy performs in different market regimes

**Example workflow**:
```bash
# Step 1: Backtest on training period (2020-2021)
python scripts/run_backtest.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --start-date 2020-01-01 \
  --end-date 2021-12-31 \
  --output training_period

# Step 2: Validate on OOS period (2022-2025) using validation script
python scripts/run_validation.py \
  --strategy ema_crossover \
  --data data/raw/EURUSD_2020-2025.csv \
  --start-date 2022-01-01 \
  --end-date 2025-12-31
```

**Important**: The data file still contains all years, but the backtest only uses the specified date range. This ensures your strategy hasn't "seen" the OOS data during development.

## Understanding Backtest Results

After running a backtest, you'll see output like this:

```
============================================================
BACKTEST RESULTS
============================================================
Strategy: ema_crossover
Total Trades: 150
Winning Trades: 90
Losing Trades: 60
Win Rate: 60.00%
Profit Factor: 1.85
Sharpe Ratio: 1.42
Sortino Ratio: 1.68
Total P&L: $2,450.00
Max Drawdown: 8.50%
CAGR: 24.50%
============================================================
```

### Key Metrics Explained

#### Profit Factor (PF)
- **What it is**: Gross profit ÷ Gross loss
- **Good value**: Above 1.5
- **Example**: PF of 1.85 means you made $1.85 for every $1 lost

#### Sharpe Ratio
- **What it is**: Risk-adjusted return measure
- **Good value**: Above 1.0 (above 2.0 is excellent)
- **Example**: Sharpe of 1.42 means good risk-adjusted returns

#### Sortino Ratio
- **What it is**: Like Sharpe, but only considers downside risk
- **Good value**: Above 1.0
- **Example**: Sortino of 1.68 means the strategy handles losses well

#### Win Rate
- **What it is**: Percentage of winning trades
- **Good value**: Depends on strategy (30-70% is common)
- **Example**: 60% win rate means 6 out of 10 trades were winners

#### Max Drawdown
- **What it is**: Largest peak-to-trough decline
- **Good value**: Lower is better (under 20% is good)
- **Example**: 8.5% drawdown means your account dropped 8.5% from its peak

#### CAGR (Compound Annual Growth Rate)
- **What it is**: Average yearly return
- **Good value**: Depends on risk tolerance (10-30% is good)
- **Example**: 24.5% CAGR means your account grew 24.5% per year on average

#### Edge Latency (NEW)
- **What it is**: Number of trades needed to statistically detect a positive edge
- **Good value**: < 150 trades (lower is better)
- **Example**: 28 trades means edge will be apparent after 28 trades
- **Interpretation**:
  - < 50 trades: Strong edge, easy to detect
  - 50-150 trades: Moderate edge, reasonable detection time
  - 150-300 trades: Weak edge, requires patience
  - > 300 trades: Edge may be too weak for practical use
- See [Edge Latency Guide](edge-latency-metric.md) for detailed interpretation

## Reading the HTML Report

After each backtest, an HTML report is generated in the `reports/` folder. Open it in your web browser to see:

1. **Key Metrics Dashboard**: Color-coded metrics at a glance
2. **Equity Curve**: Visual chart showing how your account value changed over time
3. **Detailed Metrics Table**: All calculated metrics

### Equity Curve Interpretation

The equity curve shows your account value over time:
- **Upward slope**: Good - you're making money
- **Smooth upward slope**: Excellent - consistent profits
- **Large dips**: Concerning - high drawdowns
- **Flat periods**: Strategy might not be working during those times

## Common Backtesting Scenarios

### Scenario 1: Testing a New Strategy

```bash
# Run initial backtest
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# Review the report
# If results look good, proceed to validation
```

### Scenario 2: Comparing Different Parameters

```bash
# Test with risk_per_trade_pct = 1.0
# Edit config.yml, then:
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv --output test_1pct

# Test with risk_per_trade_pct = 2.0
# Edit config.yml, then:
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv --output test_2pct

# Compare the reports
```

### Scenario 3: Testing with Different Costs

```bash
# Test with realistic commissions
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv --commission 0.001 --output with_commissions

# Test with slippage
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv --slippage 0.5 --output with_slippage
```

## What Makes a Good Backtest?

### ✅ Good Signs

- **Consistent profits**: Equity curve trends upward
- **Reasonable drawdowns**: Max drawdown under 20%
- **Good profit factor**: Above 1.5
- **Enough trades**: At least 30-50 trades for statistical significance
- **Realistic assumptions**: Includes commissions and slippage

### ⚠️ Warning Signs

- **Overfitting**: Perfect results might mean the strategy only works on this specific data
- **Too few trades**: Less than 20 trades isn't statistically meaningful
- **Large drawdowns**: Max drawdown over 30% is risky
- **Poor profit factor**: Below 1.2 means you're barely profitable
- **Unrealistic assumptions**: No commissions/slippage makes results too optimistic

## Backtesting Best Practices

### 1. Use Realistic Costs

Always include:
- **Commissions**: Real trading fees (e.g., 0.04% for Binance)
- **Slippage**: Price movement between signal and execution

### 2. Test on Multiple Time Periods

Don't just test on one year. Test on:
- Bull markets
- Bear markets
- Sideways markets
- Different volatility periods

### 3. Avoid Overfitting

- Don't optimize parameters too much
- Test on out-of-sample data (data not used for development)
- Simple strategies often work better than complex ones

### 4. Consider Market Conditions

- Crypto markets are different from stock markets
- High volatility periods need different risk management
- Market structure changes over time

## Troubleshooting

### "No signals generated"

**Possible causes**:
- Strategy conditions too strict
- Not enough data
- Indicators not calculated correctly

**Solutions**:
- Check your strategy logic
- Verify data has required columns
- Test with simpler conditions first

### "All trades are losers"

**Possible causes**:
- Strategy logic is inverted
- Stop losses too tight
- Entry conditions wrong

**Solutions**:
- Review your entry/exit logic
- Adjust stop loss settings
- Test entry conditions separately

### "Unrealistic results"

**Possible causes**:
- Forgot to include costs
- Data issues (lookahead bias)
- Strategy has bugs

**Solutions**:
- Add commissions and slippage
- Verify data is correct
- Debug your strategy code

## Next Steps

After backtesting:

1. **If results look good**: Proceed to [Validation](06-validation.md)
2. **If results need improvement**: Try [Optimization](07-optimization.md)
3. **To understand metrics better**: Read [Reports Guide](08-reports.md)

## Example Workflow

```bash
# 1. Initial backtest
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# 2. Review results - looks good!

# 3. Run validation
python scripts/run_validation.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# 4. Review validation report - passes!

# 5. Optimize parameters (see Optimization Guide)

# 6. Final validation on optimized parameters

# 7. If everything passes, consider live trading (see Live Trading Guide)
```

## Summary

Backtesting is your first step in strategy development. It helps you:
- Understand if your strategy has potential
- Identify issues before risking real money
- Compare different approaches

Remember: Good backtest results don't guarantee future profits, but they're a necessary first step!

