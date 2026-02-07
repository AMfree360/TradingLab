"""Shared utilities for Monte Carlo validation engines.

This module provides:
1. Unified metric calculation (all engines use same pipeline)
2. KDE smoothing for p-value estimation
3. Z-score normalization for combining results
4. Distribution validation
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde

from engine.backtest_engine import BacktestResult, Trade
from metrics.metrics import calculate_enhanced_metrics


def calculate_unified_metrics(
    backtest_result: Optional[BacktestResult] = None,
    trades: Optional[List[Trade]] = None,
    equity_curve: Optional[pd.Series] = None,
    initial_capital: Optional[float] = None,
    final_capital: Optional[float] = None
) -> Dict[str, float]:
    """Calculate metrics using the unified pipeline.
    
    This ensures all MC engines use the same metric calculation,
    making p-values comparable across tests.
    
    Args:
        backtest_result: Full BacktestResult (preferred)
        trades: List of trades (alternative)
        equity_curve: Equity curve series (alternative)
        initial_capital: Initial capital (required if not in result)
        final_capital: Final capital (required if not in result)
    
    Returns:
        Dictionary of metrics (same keys as calculate_enhanced_metrics)
    """
    # Prefer BacktestResult if available
    if backtest_result is not None:
        return calculate_enhanced_metrics(backtest_result)
    
    # Otherwise construct minimal BacktestResult
    if trades is None:
        trades = []
    
    if initial_capital is None:
        raise ValueError("initial_capital required if backtest_result not provided")
    
    if final_capital is None:
        if equity_curve is not None and len(equity_curve) > 0:
            final_capital = float(equity_curve.iloc[-1])
        else:
            final_capital = initial_capital + sum(t.net_pnl for t in trades)
    
    # Calculate max drawdown from equity curve
    max_drawdown = 0.0
    if equity_curve is not None and len(equity_curve) > 1:
        running_max = np.maximum.accumulate(equity_curve.values)
        drawdowns = ((equity_curve.values - running_max) / running_max) * 100.0
        max_drawdown = float(np.min(drawdowns))
    elif trades:
        # Build equity curve from trades
        equity_values = [initial_capital]
        cumulative_pnl = 0.0
        for trade in trades:
            cumulative_pnl += trade.net_pnl
            equity_values.append(initial_capital + cumulative_pnl)
        
        if len(equity_values) > 1:
            equity_array = np.array(equity_values)
            running_max = np.maximum.accumulate(equity_array)
            drawdowns = ((equity_array - running_max) / running_max) * 100.0
            max_drawdown = float(np.min(drawdowns))
    
    # Ensure equity curve is valid (at least 2 points with different timestamps)
    if equity_curve is None or len(equity_curve) < 2:
        # Create a minimal equity curve with at least 2 points
        if equity_curve is not None and len(equity_curve) == 1:
            # Duplicate the single point with a slightly later timestamp
            first_time = equity_curve.index[0]
            second_time = first_time + timedelta(days=1)
            equity_curve = pd.Series([equity_curve.iloc[0], equity_curve.iloc[0]], index=[first_time, second_time])
        else:
            # Create default equity curve
            now = datetime.now()
            equity_curve = pd.Series(
                [initial_capital, final_capital],
                index=[now, now + timedelta(days=1)]
            )
    
    # Create minimal BacktestResult
    result = BacktestResult(
        strategy_name="MC_TEST",
        symbol="MC_TEST",
        initial_capital=initial_capital,
        final_capital=final_capital,
        total_trades=len(trades),
        winning_trades=sum(1 for t in trades if t.net_pnl > 0),
        losing_trades=sum(1 for t in trades if t.net_pnl <= 0),
        win_rate=(sum(1 for t in trades if t.net_pnl > 0) / len(trades) * 100.0) if trades else 0.0,
        total_pnl=final_capital - initial_capital,
        total_commission=sum(t.commission for t in trades),
        total_slippage=sum(t.slippage for t in trades),
        max_drawdown=max_drawdown,
        trades=trades,
        equity_curve=equity_curve
    )
    
    # Calculate metrics with error handling
    try:
        import warnings
        # Suppress NumPy warnings for edge cases (we handle them explicitly)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*degrees of freedom.*')
            warnings.filterwarnings('ignore', category=RuntimeWarning, message='.*invalid value encountered.*')
            return calculate_enhanced_metrics(result)
    except (ValueError, ZeroDivisionError) as e:
        # If metrics calculation fails, return empty metrics
        # This can happen with insufficient data
        return {
            'final_pnl': final_capital - initial_capital,
            'sortino_ratio': 0.0,
            'sharpe_ratio': 0.0,
            'cagr': 0.0,
            'recovery_factor': 0.0,
            'calmar_ratio': 0.0,
            'expectancy': 0.0,
            'kelly_percent': 0.0,
            'ulcer_index': 0.0,
            'total_return_pct': 0.0,
            'profit_factor': 0.0,
        }


def smooth_distribution_kde(
    values: np.ndarray,
    bandwidth: Optional[float] = None
) -> Tuple[np.ndarray, Optional[gaussian_kde]]:
    """Apply KDE smoothing to distribution.
    
    This helps when distributions have low variance or are sparse,
    making p-value estimation more reliable.
    
    Args:
        values: Distribution values
        bandwidth: KDE bandwidth (auto-select if None)
    
    Returns:
        Tuple of (smoothed_values, kde_object)
    """
    if len(values) < 3:
        # Not enough data for KDE
        return values, None
    
    # Remove infinite and NaN values
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 3:
        return values, None
    
    # Auto-select bandwidth using Scott's rule if not provided
    if bandwidth is None:
        # Scott's rule: bandwidth = n^(-1/5) * std
        # Need at least 2 values for std with ddof=1
        if len(finite_values) > 1:
            std = np.std(finite_values, ddof=1)
            if std > 0:
                bandwidth = std * (len(finite_values) ** (-1/5))
            else:
                # Very low variance - use small fixed bandwidth
                bandwidth = np.abs(np.mean(finite_values)) * 0.01 if np.mean(finite_values) != 0 else 0.01
        else:
            # Not enough values for std - use small fixed bandwidth
            bandwidth = np.abs(np.mean(finite_values)) * 0.01 if len(finite_values) > 0 and np.mean(finite_values) != 0 else 0.01
    
    try:
        kde = gaussian_kde(finite_values, bw_method=bandwidth)
        # Sample from KDE to get smoothed distribution
        smoothed = kde.resample(len(values))[0]
        return smoothed, kde
    except (np.linalg.LinAlgError, ValueError):
        # KDE failed (e.g., singular matrix) - return original
        return values, None


def calculate_p_value_with_kde(
    observed: float,
    distribution: np.ndarray,
    use_kde: bool = True,
    kde_bandwidth: Optional[float] = None
) -> Tuple[float, float]:
    """Calculate p-value with optional KDE smoothing.
    
    Args:
        observed: Observed value
        distribution: Null distribution
        use_kde: Whether to use KDE smoothing
        kde_bandwidth: KDE bandwidth (auto if None)
    
    Returns:
        Tuple of (p_value, percentile)
    """
    if len(distribution) == 0:
        return 1.0, 0.0
    
    # Remove infinite and NaN values
    finite_dist = distribution[np.isfinite(distribution)]
    if len(finite_dist) == 0:
        return 1.0, 0.0
    
    # Need at least 2 values for std with ddof=1
    if len(finite_dist) < 2:
        # Single value - can't calculate variance, use empirical p-value
        p_value = 1.0 if finite_dist[0] >= observed else 0.0
        percentile = 0.0 if finite_dist[0] < observed else 100.0
        return float(p_value), float(percentile)
    
    # Check if distribution has sufficient variance
    std = np.std(finite_dist, ddof=1)
    mean_val = np.mean(finite_dist)
    
    # If variance is too low, use KDE smoothing
    if use_kde and (std < abs(mean_val) * 1e-6 or len(finite_dist) < 10):
        smoothed_dist, kde = smooth_distribution_kde(finite_dist, kde_bandwidth)
        if kde is not None:
            # Use KDE to estimate probability
            try:
                # Calculate CDF at observed value
                # P(X >= observed) = 1 - CDF(observed)
                cdf_at_observed = kde.integrate_box_1d(-np.inf, observed)
                p_value = 1.0 - cdf_at_observed
                # Percentile: P(X < observed)
                percentile = cdf_at_observed * 100.0
                return float(p_value), float(percentile)
            except (ValueError, np.linalg.LinAlgError):
                # KDE integration failed, fall back to empirical
                pass
    
    # Empirical p-value (standard method)
    p_value = float((finite_dist >= observed).sum() / len(finite_dist))
    percentile = float((finite_dist < observed).sum() / len(finite_dist) * 100.0)
    
    # Apply minimum p-value floor to prevent exactly 0.0 (which is statistically suspicious)
    # Minimum is 1/n to account for the observed value itself
    min_p_value = 1.0 / len(finite_dist)
    if p_value < min_p_value:
        # P-value is suspiciously low - this could indicate:
        # 1. The observed value is truly exceptional (legitimate)
        # 2. A calculation mismatch between observed and bootstrap metrics (bug)
        # 3. Insufficient bootstrap iterations
        # We floor it at 1/n but keep the percentile accurate
        p_value = min_p_value
    
    return p_value, percentile


def validate_distribution(
    distribution: np.ndarray,
    min_samples: int = 10,
    min_variance_ratio: float = 1e-6
) -> Tuple[bool, str]:
    """Validate that distribution is suitable for p-value calculation.
    
    Args:
        distribution: Distribution to validate
        min_samples: Minimum number of samples required
        min_variance_ratio: Minimum variance relative to mean (for non-zero mean)
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if len(distribution) < min_samples:
        return False, f"Distribution has only {len(distribution)} samples, need at least {min_samples}"
    
    finite_dist = distribution[np.isfinite(distribution)]
    if len(finite_dist) < min_samples:
        return False, f"Distribution has only {len(finite_dist)} finite values"
    
    # Need at least 2 values for std with ddof=1
    if len(finite_dist) < 2:
        return False, f"Distribution has only {len(finite_dist)} finite values, need at least 2 for variance calculation"
    
    std = np.std(finite_dist, ddof=1)
    mean_val = np.mean(finite_dist)
    
    # Check variance
    if abs(mean_val) > 0:
        variance_ratio = std / abs(mean_val)
        if variance_ratio < min_variance_ratio:
            return False, f"Distribution variance too low (std/mean = {variance_ratio:.2e}, need >= {min_variance_ratio})"
    
    # Check if all values are identical
    if std == 0:
        return False, "Distribution has zero variance (all values identical)"
    
    return True, ""


def normalize_to_z_scores(
    values: np.ndarray,
    mean: Optional[float] = None,
    std: Optional[float] = None
) -> np.ndarray:
    """Normalize values to Z-scores.
    
    Args:
        values: Values to normalize
        mean: Mean (calculated if None)
        std: Standard deviation (calculated if None)
    
    Returns:
        Z-scores
    """
    if mean is None:
        mean = np.mean(values)
    if std is None:
        std = np.std(values, ddof=1)
    
    if std == 0:
        return np.zeros_like(values)
    
    return (values - mean) / std


def normalize_to_ranks(
    values: np.ndarray
) -> np.ndarray:
    """Normalize values to ranks (0-1 scale).
    
    Args:
        values: Values to normalize
    
    Returns:
        Rank-normalized values (0 = minimum, 1 = maximum)
    """
    if len(values) == 0:
        return values
    
    ranks = stats.rankdata(values, method='average')
    # Normalize to 0-1
    normalized = (ranks - 1) / (len(values) - 1) if len(values) > 1 else np.array([0.5])
    return normalized

