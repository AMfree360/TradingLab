"""Edge Latency / Expected Time to Significance metric.

This module computes the statistical edge latency - the number of trades required
to detect a positive edge with a given confidence and power level.

INTERPRETATION
--------------
Edge latency answers: "How many trades do I need to see before I can be confident
this strategy has a real edge?"

- Low latency (< 50 trades): Strong edge, easy to detect
- Medium latency (50-150 trades): Moderate edge, reasonable detection time
- High latency (150-300 trades): Weak edge, requires patience
- Very high latency (> 300 trades): Edge may be too weak for practical use

The metric converts trade count to time using observed trade frequency, helping
traders understand if they can realistically wait for edge confirmation.

FORMULA
-------
N_required = ((z_alpha + z_beta) / signal_ratio) ** 2

Where:
- signal_ratio = μ / σ (mean R-multiple / std R-multiple)
- z_alpha = z-score for confidence level (default 1.645 for 95%)
- z_beta = z-score for power level (default 0.84 for 80%)

This uses a one-sided hypothesis test:
H0: μ ≤ 0 (no edge)
H1: μ > 0 (positive edge)
"""

from typing import Optional, Dict
import numpy as np
from scipy import stats


def compute_edge_latency(
    r_multiples: np.ndarray,
    confidence: float = 0.95,
    power: float = 0.80,
    trades_per_year: Optional[float] = None
) -> Dict[str, Optional[float]]:
    """
    Compute edge latency metric for a trading strategy.
    
    Edge latency is the number of trades required to statistically detect
    a positive edge with given confidence and power levels.
    
    Args:
        r_multiples: Array of R-multiples from completed trades
        confidence: Confidence level (default 0.95 = 95%)
        power: Statistical power (default 0.80 = 80%)
        trades_per_year: Optional observed trade frequency for time conversion
    
    Returns:
        Dictionary with:
            - mean_r: Mean R-multiple
            - std_r: Standard deviation of R-multiples
            - signal_ratio: μ / σ (mean / std)
            - required_trades: Number of trades needed to detect edge (None if no edge)
            - years_to_significance: Time to significance in years (None if trades_per_year not provided)
            - confidence: Confidence level used
            - power: Power level used
    
    Examples:
        >>> # Strong edge: mean=0.5, std=1.0 → signal_ratio=0.5
        >>> r = np.array([0.5, 1.0, -0.5, 1.5, 0.0, 0.5, 1.0, -0.5])
        >>> result = compute_edge_latency(r)
        >>> result['required_trades']  # Should be relatively low (< 50)
        
        >>> # Weak edge: mean=0.1, std=1.0 → signal_ratio=0.1
        >>> r = np.array([0.1, 0.2, -0.1, 0.1, 0.0, 0.1, 0.2, -0.1])
        >>> result = compute_edge_latency(r, trades_per_year=100.0)
        >>> result['years_to_significance']  # Will be higher
    """
    # Validate inputs
    if len(r_multiples) == 0:
        return {
            'mean_r': None,
            'std_r': None,
            'signal_ratio': None,
            'required_trades': None,
            'years_to_significance': None,
            'confidence': confidence,
            'power': power
        }
    
    # Calculate statistics
    mean_r = float(np.mean(r_multiples))
    std_r = float(np.std(r_multiples, ddof=1)) if len(r_multiples) > 1 else 0.0
    
    # Edge cases: no edge or zero variance
    if mean_r <= 0:
        # No positive edge detected
        return {
            'mean_r': mean_r,
            'std_r': std_r,
            'signal_ratio': None,
            'required_trades': None,
            'years_to_significance': None,
            'confidence': confidence,
            'power': power
        }
    
    if std_r == 0 or std_r < 1e-10:
        # Zero or near-zero variance - all trades have same R-multiple
        # This is degenerate (all wins or all losses with same size)
        # Can't compute signal ratio
        return {
            'mean_r': mean_r,
            'std_r': std_r,
            'signal_ratio': None,
            'required_trades': None,
            'years_to_significance': None,
            'confidence': confidence,
            'power': power
        }
    
    # Calculate signal ratio
    signal_ratio = mean_r / std_r
    
    # Validate signal ratio
    # Note: Warnings removed - information is shown in HTML report
    # if signal_ratio < 0.1:
    #     warnings.warn(...)
    
    # Get z-scores for confidence and power
    # z_alpha: one-sided test, so use confidence directly
    z_alpha = stats.norm.ppf(confidence)
    # z_beta: power is 1 - beta, so beta = 1 - power
    z_beta = stats.norm.ppf(power)
    
    # Calculate required sample size
    # Formula: N = ((z_alpha + z_beta) / signal_ratio) ** 2
    required_trades = ((z_alpha + z_beta) / signal_ratio) ** 2
    
    # Validate required trades
    # Note: Warnings removed - information is shown in HTML report
    # if required_trades > 300:
    #     warnings.warn(...)
    
    # Convert to time if trades_per_year provided
    years_to_significance = None
    if trades_per_year is not None and trades_per_year > 0:
        years_to_significance = required_trades / trades_per_year
        
        # Warn if time is very long
        # Note: Warnings removed - information is shown in HTML report
        # if years_to_significance > 5.0:
        #     warnings.warn(...)
    
    return {
        'mean_r': mean_r,
        'std_r': std_r,
        'signal_ratio': float(signal_ratio),
        'required_trades': float(required_trades),
        'years_to_significance': float(years_to_significance) if years_to_significance is not None else None,
        'confidence': confidence,
        'power': power
    }

