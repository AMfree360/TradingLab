# Edge Latency / Expected Time to Significance Metric

## Overview

The **Edge Latency** metric answers a critical question for traders:

> "How many trades do I need to see before I can be confident this strategy has a real edge?"

This metric uses statistical hypothesis testing to determine the number of trades required to detect a positive edge with a given confidence and power level. It then converts this to time (years/months) using observed trade frequency.

## Interpretation

### Trade Count Interpretation

- **Low latency (< 50 trades)**: Strong edge, easy to detect
  - Strategy has a clear statistical advantage
  - Edge will be apparent relatively quickly
  
- **Medium latency (50-150 trades)**: Moderate edge, reasonable detection time
  - Strategy has a real edge but requires patience
  - Typical for well-designed strategies
  
- **High latency (150-300 trades)**: Weak edge, requires patience
  - Edge exists but is subtle
  - May require months or years to confirm
  
- **Very high latency (> 300 trades)**: Edge may be too weak for practical use
  - Warning issued automatically
  - May indicate strategy needs refinement or position sizing adjustment

### Time Interpretation

When `trades_per_year` is provided, the metric also reports `years_to_significance`:

- **< 0.5 years**: Edge will be apparent within 6 months
- **0.5-2 years**: Reasonable timeframe for edge confirmation
- **2-5 years**: Long but potentially acceptable for patient traders
- **> 5 years**: Warning issued - may be impractical

## Formula

The metric uses a one-sided hypothesis test:

**H₀**: μ ≤ 0 (no edge)  
**H₁**: μ > 0 (positive edge)

**Required Trades**:
```
N_required = ((z_alpha + z_beta) / signal_ratio)²
```

Where:
- `signal_ratio = μ / σ` (mean R-multiple / std R-multiple)
- `z_alpha` = z-score for confidence level (default 1.645 for 95%)
- `z_beta` = z-score for power level (default 0.84 for 80%)

**Time to Significance**:
```
years_to_significance = required_trades / trades_per_year
```

## Usage

The metric is automatically calculated in `calculate_enhanced_metrics()` and included in backtest results:

```python
from metrics import calculate_enhanced_metrics
from engine.backtest_engine import BacktestResult

# After running backtest
metrics = calculate_enhanced_metrics(backtest_result)

# Access edge latency
edge_latency = metrics['edge_latency']
print(f"Required trades: {edge_latency['required_trades']:.1f}")
print(f"Years to significance: {edge_latency['years_to_significance']:.2f}")
```

## Output Structure

```python
{
    'mean_r': float,              # Mean R-multiple
    'std_r': float,               # Standard deviation of R-multiples
    'signal_ratio': float,        # μ / σ (mean / std)
    'required_trades': float,      # Number of trades needed (None if no edge)
    'years_to_significance': float | None,  # Time in years (None if trades_per_year not provided)
    'confidence': float,          # Confidence level used (default 0.95)
    'power': float                # Power level used (default 0.80)
}
```

## Edge Cases

### No Edge (mean ≤ 0)
- `signal_ratio`: `None`
- `required_trades`: `None`
- `years_to_significance`: `None`

### Zero Variance (all trades same R-multiple)
- `signal_ratio`: `None`
- `required_trades`: `None`
- This is a degenerate case (all wins or all losses with same size)

### Empty Trade List
- All fields set to `None`

## Warnings

The metric automatically issues warnings for:

1. **Low Signal Ratio (< 0.1)**: Very weak edge that may require hundreds of trades
2. **High Latency (> 300 trades)**: Edge may be too weak for practical use
3. **Long Time (> 5 years)**: May be impractical to wait for confirmation

## Example Output

For a realistic strategy with 100 trades:

```
Edge Latency:
  Mean R: 0.523
  Std R: 1.124
  Signal Ratio: 0.465
  Required Trades: 28.3
  Years to Significance: 0.14 (at 200 trades/year)
```

This indicates:
- Strong edge (signal ratio = 0.465)
- Only 28 trades needed to detect edge
- At 200 trades/year, edge will be apparent in ~1.7 months

## Best Practices

1. **Use with other metrics**: Edge latency complements Sharpe, expectancy, and profit factor
2. **Consider trade frequency**: Strategies with low trade frequency may have longer time to significance
3. **Monitor during live trading**: Track progress toward required trade count
4. **Adjust position sizing**: If latency is too high, consider adjusting position sizing to improve signal ratio
5. **Refine strategy**: Very high latency may indicate strategy needs optimization

## Technical Notes

- Uses one-sided hypothesis test (testing for positive edge only)
- Default confidence: 95% (α = 0.05)
- Default power: 80% (β = 0.20)
- Based on normal distribution assumption (valid for large sample sizes)
- R-multiples are extracted from trade objects automatically

## References

- Statistical power analysis for sample size determination
- One-sided hypothesis testing for edge detection
- Signal-to-noise ratio in trading strategy evaluation

