# Optimization Guide

Optimization is the process of finding the best parameters for your trading strategy. This guide explains when and how to optimize effectively.

## What is Optimization?

Optimization means testing different parameter values to find what works best. For example:
- Testing risk_per_trade_pct: 0.5%, 1.0%, 1.5%, 2.0%
- Testing EMA lengths: 5, 10, 15, 20
- Testing stop loss offsets: 0, 0.5%, 1.0%

## When to Optimize

### ✅ Good Times to Optimize

- **After initial backtest shows promise**: Strategy works but could be better
- **Before going live**: Fine-tune parameters for best results
- **After market changes**: Adapt to new market conditions
- **When validation passes but barely**: Improve margins

### ❌ Bad Times to Optimize

- **Before validation**: Optimize AFTER you know strategy works
- **On too little data**: Need sufficient data for meaningful results
- **Too aggressively**: Can lead to overfitting
- **On in-sample data only**: Must validate on out-of-sample data

## The Optimization Process

### Step 1: Identify Parameters to Optimize

Not all parameters should be optimized. Focus on:

**Good candidates**:
- Risk management (risk_per_trade_pct)
- Stop loss settings
- Indicator periods (if strategy is robust)
- Entry/exit thresholds

**Avoid optimizing**:
- Too many parameters at once (overfitting risk)
- Parameters that don't affect results much
- Parameters that make strategy too complex

### Step 2: Define Parameter Ranges

Choose reasonable ranges based on:
- **Market knowledge**: What makes sense for this market?
- **Initial backtest**: What values worked?
- **Literature**: What do experts recommend?

**Example**:
```python
param_grid = {
    'risk.risk_per_trade_pct': [0.5, 1.0, 1.5, 2.0],  # 0.5% to 2.0%
    'stop_loss.length': [20, 30, 40, 50],  # SMA periods
    'trailing_stop.activation_r': [0.3, 0.5, 0.7, 1.0]  # R-multiples
}
```

### Step 3: Run Grid Search

Use the sensitivity analyzer:

```python
from validation import SensitivityAnalyzer
from config.schema import load_and_validate_strategy_config
import pandas as pd

# Load data
df = pd.read_csv('data/raw/BTCUSDT_1m.csv', index_col=0, parse_dates=True)

# Load base config
base_config = load_and_validate_strategy_config('strategies/my_strategy/config.yml')
base_config_dict = base_config.model_dump()  # Convert to dict

# Define parameter grid
param_grid = {
    'risk.risk_per_trade_pct': [0.5, 1.0, 1.5, 2.0],
    'stop_loss.length': [30, 40, 50, 60],
}

# Create analyzer
analyzer = SensitivityAnalyzer(
    strategy_class=MyStrategy,
    initial_capital=10000.0
)

# Run grid search
results_df = analyzer.grid_search(
    data=df,
    base_config=base_config_dict,
    param_grid=param_grid,
    metric='profit_factor'  # Optimize for profit factor
)

# View results
print(results_df.sort_values('profit_factor', ascending=False).head(10))
```

### Step 4: Analyze Results

Look for:
- **Best combination**: Highest metric value
- **Stability**: Good results across similar parameters
- **Sensitivity**: How much results change with parameters

```python
# Analyze sensitivity
for param in param_grid.keys():
    sensitivity = analyzer.analyze_sensitivity(results_df, param, 'profit_factor')
    print(f"{param}: {sensitivity}")
```

### Step 5: Validate Optimized Parameters

**CRITICAL**: Always validate optimized parameters on out-of-sample data!

```bash
# Update config.yml with optimized parameters
# Then run validation
python scripts/run_validation.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv
```

## Optimization Strategies

### Strategy 1: Coarse to Fine

1. **Coarse search**: Test wide ranges with large steps
   ```python
   risk_per_trade_pct: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
   ```

2. **Fine search**: Narrow down to best region
   ```python
   risk_per_trade_pct: [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]  # Around 1.0-1.5
   ```

### Strategy 2: One at a Time

Optimize one parameter at a time:

1. Optimize risk_per_trade_pct (fix others)
2. Optimize stop_loss.length (fix others, use best risk)
3. Optimize trailing_stop.activation_r (fix others, use best so far)

**Advantage**: Less overfitting risk
**Disadvantage**: Might miss parameter interactions

### Strategy 3: Full Grid Search

Test all combinations:

**Advantage**: Finds best combination
**Disadvantage**: Can overfit, takes longer

**Use when**: 
- Few parameters (2-3)
- Parameters likely interact
- Have plenty of data

## Avoiding Overfitting

### What is Overfitting?

Overfitting means your strategy works perfectly on historical data but fails on new data. It's the #1 optimization mistake.

### Signs of Overfitting

- **Perfect results**: Too good to be true
- **Sharp parameter peaks**: Only one specific value works
- **Poor validation**: Test results much worse than train
- **Many parameters**: More parameters = more overfitting risk

### How to Avoid Overfitting

#### 1. Limit Parameters

- Optimize 2-3 parameters max
- Use coarse grids (fewer values)
- Focus on important parameters

#### 2. Use Out-of-Sample Validation

- Never optimize on validation data
- Always test optimized parameters on new data
- Use walk-forward analysis

#### 3. Prefer Simpler Solutions

- If two parameter sets perform similarly, choose simpler one
- Avoid complex parameter combinations
- Simpler strategies are more robust

#### 4. Check Stability

- Good parameters should work in a range, not just one value
- If only one specific value works, it's probably overfitted
- Prefer parameters that work across similar values

#### 5. Limit Optimization Rounds

- Don't optimize forever
- Set a limit (e.g., 2-3 rounds)
- Accept "good enough" results

## Parameter Sensitivity Analysis

Understanding how sensitive your strategy is to parameters helps avoid overfitting.

### Low Sensitivity (Good)

Parameters that work across a range:
```python
# These all work similarly - good!
risk_per_trade_pct: 0.8 → PF: 1.85
risk_per_trade_pct: 1.0 → PF: 1.87
risk_per_trade_pct: 1.2 → PF: 1.83
```

**Interpretation**: Strategy is robust, less likely to overfit.

### High Sensitivity (Concerning)

Parameters that only work at one value:
```python
# Only 1.0 works - concerning!
risk_per_trade_pct: 0.8 → PF: 1.2
risk_per_trade_pct: 1.0 → PF: 1.9  # Only this works
risk_per_trade_pct: 1.2 → PF: 1.1
```

**Interpretation**: Might be overfitted, test carefully.

## Optimization Workflow

### Complete Workflow

```bash
# 1. Initial backtest (no optimization)
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# 2. If results look good, run validation
python scripts/run_validation.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# 3. If validation passes, then optimize
# (Use Python script for grid search - see examples above)

# 4. Update config.yml with optimized parameters

# 5. Re-validate optimized parameters
python scripts/run_validation.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv

# 6. If validation still passes, proceed to live
```

## Example: Optimizing Risk Per Trade

### Step 1: Define Grid

```python
param_grid = {
    'risk.risk_per_trade_pct': [0.5, 1.0, 1.5, 2.0, 2.5]
}
```

### Step 2: Run Grid Search

```python
results = analyzer.grid_search(data, base_config, param_grid, 'profit_factor')
```

### Step 3: Analyze Results

```python
# Best result
best = results.loc[results['profit_factor'].idxmax()]
print(f"Best risk_per_trade_pct: {best['risk.risk_per_trade_pct']}")
print(f"Best PF: {best['profit_factor']}")

# Check sensitivity
sensitivity = analyzer.analyze_sensitivity(results, 'risk.risk_per_trade_pct', 'profit_factor')
print(f"Sensitivity: {sensitivity}")
```

### Step 4: Validate

Update config.yml and re-run validation.

## When to Stop Optimizing

Stop when:
- ✅ Validation results are good
- ✅ Parameters are stable (work in a range)
- ✅ Further optimization doesn't improve much
- ✅ You've reached your optimization limit (2-3 rounds)

Don't optimize:
- ❌ Forever (overfitting risk)
- ❌ On validation data
- ❌ Too many parameters at once
- ❌ When results are already good

## Common Mistakes

### Mistake 1: Optimizing Before Validation

**Wrong**: Optimize → Backtest → Done
**Right**: Backtest → Validate → Optimize → Re-validate

### Mistake 2: Optimizing Too Many Parameters

**Wrong**: Optimize 10 parameters at once
**Right**: Optimize 2-3 important parameters

### Mistake 3: Not Validating Optimized Parameters

**Wrong**: Optimize, update config, go live
**Right**: Optimize, update config, validate, then consider live

### Mistake 4: Over-Optimizing

**Wrong**: Keep optimizing until perfect
**Right**: Stop when "good enough" and validated

## Best Practices

1. **Optimize last**: Only after validation passes
2. **Limit parameters**: 2-3 max
3. **Use coarse grids**: Fewer values, wider ranges
4. **Always validate**: Test optimized parameters on new data
5. **Prefer simplicity**: Simpler is usually better
6. **Check stability**: Good parameters work in a range
7. **Set limits**: Don't optimize forever

## Next Steps

After optimization:

- **If validation still passes**: Consider [Live Trading](LIVE_TRADING.md)
- **If validation fails**: Simplify strategy or accept current parameters
- **To understand results**: Review [Reports Guide](REPORTS.md)

## Summary

Optimization can improve your strategy, but:
- Do it AFTER validation
- Limit parameters and rounds
- Always re-validate
- Prefer simple, stable solutions

Remember: A slightly worse but robust strategy beats a perfect but overfitted one!

