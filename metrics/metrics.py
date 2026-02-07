"""Enhanced performance metrics calculation."""

from typing import List, Optional
import pandas as pd
import numpy as np
from engine.backtest_engine import Trade, BacktestResult
from metrics.edge_latency import compute_edge_latency
from metrics.mfe_mfa_analysis import analyze_mfe_mae, get_mfe_mae_summary


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252
) -> float:
    """
    Calculate Sortino ratio (downside deviation only).
    
    Args:
        returns: Array of returns
        risk_free_rate: Risk-free rate (default 0.0)
        periods_per_year: Number of periods per year for annualization
    
    Returns:
        Sortino ratio
    """
    # #region agent log
    import json
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'B',
                'location': 'metrics.py:9',
                'message': 'calculate_sortino_ratio entry',
                'data': {
                    'returns_len': len(returns),
                    'returns_sample': returns.tolist()[:20] if len(returns) > 0 else [],
                    'returns_min': float(np.min(returns)) if len(returns) > 0 else None,
                    'returns_max': float(np.max(returns)) if len(returns) > 0 else None,
                    'returns_mean': float(np.mean(returns)) if len(returns) > 0 else None,
                    'risk_free_rate': risk_free_rate,
                    'periods_per_year': periods_per_year,
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    if len(returns) == 0:
        return 0.0
    
    excess_returns = returns - risk_free_rate / periods_per_year
    
    # #region agent log
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'C',
                'location': 'metrics.py:28',
                'message': 'calculate_sortino_ratio after excess_returns calc',
                'data': {
                    'excess_returns_len': len(excess_returns),
                    'excess_returns_min': float(np.min(excess_returns)) if len(excess_returns) > 0 else None,
                    'excess_returns_max': float(np.max(excess_returns)) if len(excess_returns) > 0 else None,
                    'excess_returns_mean': float(np.mean(excess_returns)) if len(excess_returns) > 0 else None,
                    'risk_free_adjustment': risk_free_rate / periods_per_year,
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    downside_returns = excess_returns[excess_returns < 0]
    
    # #region agent log
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'D',
                'location': 'metrics.py:29',
                'message': 'calculate_sortino_ratio downside_returns filter',
                'data': {
                    'downside_returns_len': len(downside_returns),
                    'downside_returns': downside_returns.tolist()[:20] if len(downside_returns) > 0 else [],
                    'excess_returns_negative_count': int(np.sum(excess_returns < 0)),
                    'excess_returns_zero_count': int(np.sum(excess_returns == 0)),
                    'excess_returns_positive_count': int(np.sum(excess_returns > 0)),
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    if len(downside_returns) == 0:
        # No downside returns - this can happen when:
        # 1. Insufficient data points (e.g., very few trades producing sparse equity curve)
        # 2. All returns are positive (all winning trades/periods)
        # 3. Overfitting (rare but possible)
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'E',
                    'location': 'metrics.py:31',
                    'message': 'calculate_sortino_ratio NO DOWNSIDE RETURNS',
                    'data': {
                        'excess_returns_mean': float(np.mean(excess_returns)) if len(excess_returns) > 0 else None,
                        'excess_returns_all': excess_returns.tolist()[:50] if len(excess_returns) > 0 else [],
                        'returns_all': returns.tolist()[:50] if len(returns) > 0 else [],
                        'will_return_capped': np.mean(excess_returns) > 0 if len(excess_returns) > 0 else False,
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
        if np.mean(excess_returns) > 0:
            import warnings
            # More accurate warning: insufficient data is more common than overfitting
            if len(returns) < 3:
                warnings.warn(f"Sortino ratio calculation: No downside returns detected with only {len(returns)} return(s). "
                             f"This indicates insufficient data points for meaningful Sortino calculation. "
                             f"Consider using trade-based returns instead. Returning capped value of 20.0.")
            else:
                warnings.warn(f"Sortino ratio calculation: No downside returns detected despite {len(returns)} returns. "
                             f"This may indicate overfitting or insufficient downside data. "
                             f"Returning capped value of 20.0.")
            return 20.0
        else:
            return 0.0
    
    # Need at least 2 values for std with ddof=1
    if len(downside_returns) < 2:
        # Not enough downside returns for variance calculation
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'N',
                    'location': 'metrics.py:139',
                    'message': 'calculate_sortino_ratio INSUFFICIENT_DOWNSIDE_RETURNS',
                    'data': {
                        'downside_returns_len': len(downside_returns),
                        'returns_len': len(returns),
                        'excess_returns_mean': float(np.mean(excess_returns)) if len(excess_returns) > 0 else None,
                        'will_warn': np.mean(excess_returns) > 0 if len(excess_returns) > 0 else False,
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        if np.mean(excess_returns) > 0:
            import warnings
            warnings.warn("Sortino ratio calculation: Insufficient downside returns for variance calculation. "
                         "This may indicate overfitting or insufficient data. "
                         "Returning capped value of 20.0.")
            return 20.0
        else:
            return 0.0
    
    downside_std = np.std(downside_returns, ddof=1)
    if downside_std == 0 or downside_std < 1e-10:  # Very small or zero
        # If no downside risk, return a high but reasonable value if profitable
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'run1',
                    'hypothesisId': 'M',
                    'location': 'metrics.py:151',
                    'message': 'calculate_sortino_ratio DOWNSIDE_STD_NEAR_ZERO',
                    'data': {
                        'downside_returns_len': len(downside_returns),
                        'downside_std': float(downside_std),
                        'excess_returns_mean': float(np.mean(excess_returns)) if len(excess_returns) > 0 else None,
                        'will_warn': np.mean(excess_returns) > 0 if len(excess_returns) > 0 else False,
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        if np.mean(excess_returns) > 0:
            import warnings
            warnings.warn("Sortino ratio calculation: Downside deviation is near zero. "
                         "This may indicate overfitting or insufficient downside data. "
                         "Returning capped value of 20.0.")
            return 20.0
        else:
            return 0.0
    
    mean_return = np.mean(excess_returns)
    sortino = mean_return / downside_std * np.sqrt(periods_per_year)
    
    # Cap Sortino at a reasonable maximum to avoid unrealistic values
    # This can happen with very few losing trades and very small losses
    # For crypto strategies, even excellent ones rarely exceed 5-10
    # Cap at 20.0 to flag potentially unrealistic scenarios
    if sortino > 20.0:
        import warnings
        warnings.warn(f"Sortino ratio capped at 20.0 (calculated value: {sortino:.2f}). "
                     f"This may indicate: 1) Very few losing trades, 2) Very small losses, "
                     f"3) Overfitting, or 4) Calculation issue. "
                     f"Verify strategy robustness and data quality.")
        return 20.0
    
    return float(sortino)


def calculate_cagr(
    initial_capital: float,
    final_capital: float,
    years: float
) -> float:
    """
    Calculate Compound Annual Growth Rate (CAGR).
    
    Args:
        initial_capital: Starting capital
        final_capital: Ending capital
        years: Number of years
    
    Returns:
        CAGR as a percentage
    """
    if initial_capital <= 0 or years <= 0:
        return 0.0
    
    if final_capital <= 0:
        return -100.0
    
    cagr = ((final_capital / initial_capital) ** (1.0 / years) - 1.0) * 100.0
    return float(cagr)


def calculate_recovery_factor(
    total_pnl: float,
    max_drawdown: float
) -> float:
    """
    Calculate recovery factor (total P&L / max drawdown).
    
    Args:
        total_pnl: Total profit/loss
        max_drawdown: Maximum drawdown
    
    Returns:
        Recovery factor
    """
    if max_drawdown == 0:
        return np.inf if total_pnl > 0 else 0.0
    
    return float(total_pnl / abs(max_drawdown))


def calculate_calmar_ratio(
    cagr: float,
    max_drawdown_pct: float
) -> float:
    """
    Calculate Calmar ratio (CAGR / max drawdown %).
    
    Args:
        cagr: Compound Annual Growth Rate (%)
        max_drawdown_pct: Maximum drawdown percentage
    
    Returns:
        Calmar ratio
    """
    if max_drawdown_pct == 0:
        return np.inf if cagr > 0 else 0.0
    
    return float(cagr / abs(max_drawdown_pct))


def calculate_expectancy(
    trades: List[Trade]
) -> float:
    """
    Calculate expectancy per trade.
    
    Args:
        trades: List of Trade objects
    
    Returns:
        Average expectancy per trade
    """
    if not trades:
        return 0.0
    
    total_pnl = sum(t.net_pnl for t in trades)
    return float(total_pnl / len(trades))


def calculate_kelly_percent(
    win_rate: float,
    avg_win: float,
    avg_loss: float
) -> float:
    """
    Calculate Kelly Criterion percentage.
    
    Args:
        win_rate: Win rate (0.0 to 1.0)
        avg_win: Average winning trade
        avg_loss: Average losing trade (as positive number)
    
    Returns:
        Kelly percentage
    """
    # Handle edge cases
    if avg_loss == 0:
        return 0.0
    
    if avg_win == 0:
        # No winning trades - Kelly would be negative, return 0
        return 0.0
    
    win_loss_ratio = avg_win / avg_loss
    
    # Check for division by zero or very small win_loss_ratio
    if win_loss_ratio == 0 or abs(win_loss_ratio) < 1e-10:
        return 0.0
    
    kelly = win_rate - ((1 - win_rate) / win_loss_ratio)
    
    # Return as percentage, capped at reasonable values
    return float(max(0.0, min(100.0, kelly * 100.0)))


def calculate_ulcer_index(equity_curve: np.ndarray) -> float:
    """
    Calculate Ulcer Index (measure of drawdown severity).
    
    Args:
        equity_curve: Array of equity values
    
    Returns:
        Ulcer Index
    """
    if len(equity_curve) < 2:
        return 0.0
    
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - running_max) / running_max) * 100.0
    
    # Square of drawdown percentages
    squared_dd = drawdown_pct ** 2
    
    # Average of squared drawdowns
    ulcer = np.sqrt(np.mean(squared_dd))
    
    return float(ulcer)


def calculate_sterling_ratio(
    total_return_pct: float,
    avg_max_dd: float,
    periods: int = 3
) -> float:
    """
    Calculate Sterling ratio (return / average max drawdown over periods).
    
    Args:
        total_return_pct: Total return percentage
        avg_max_dd: Average maximum drawdown over periods
        periods: Number of periods to average
    
    Returns:
        Sterling ratio
    """
    if avg_max_dd == 0:
        return np.inf if total_return_pct > 0 else 0.0
    
    return float(total_return_pct / abs(avg_max_dd))


def calculate_trade_returns(trades: List[Trade], initial_capital: float) -> np.ndarray:
    """
    Calculate returns for each trade.
    
    Args:
        trades: List of Trade objects
        initial_capital: Initial capital
    
    Returns:
        Array of returns (as decimals, e.g., 0.01 = 1%)
    """
    if not trades or initial_capital == 0:
        return np.array([])
    
    returns = np.array([t.net_pnl / initial_capital for t in trades])
    return returns


def calculate_equity_returns(equity_curve: pd.Series) -> np.ndarray:
    """
    Calculate percentage returns from equity curve.
    
    Works with both datetime-indexed and integer-indexed equity curves.
    Returns are ALWAYS computed as: returns = equity_curve.pct_change().dropna()
    This is time-agnostic and works identically for both index types.
    
    Args:
        equity_curve: Series with numeric or datetime index
    
    Returns:
        Array of percentage returns
    """
    # #region agent log
    import json
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'A',
                'location': 'metrics.py:271',
                'message': 'calculate_equity_returns entry',
                'data': {
                    'equity_curve_len': len(equity_curve),
                    'equity_curve_values': equity_curve.values.tolist()[:10] if len(equity_curve) > 0 else [],
                    'equity_curve_first': float(equity_curve.iloc[0]) if len(equity_curve) > 0 else None,
                    'equity_curve_last': float(equity_curve.iloc[-1]) if len(equity_curve) > 0 else None,
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    if len(equity_curve) < 2:
        return np.array([])
    
    # Calculate period returns
    returns = equity_curve.pct_change().dropna().values
    
    # #region agent log
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'run1',
                'hypothesisId': 'A',
                'location': 'metrics.py:285',
                'message': 'calculate_equity_returns after pct_change',
                'data': {
                    'returns_len': len(returns),
                    'returns': returns.tolist()[:20] if len(returns) > 0 else [],
                    'returns_min': float(np.min(returns)) if len(returns) > 0 else None,
                    'returns_max': float(np.max(returns)) if len(returns) > 0 else None,
                    'returns_mean': float(np.mean(returns)) if len(returns) > 0 else None,
                    'negative_count': int(np.sum(returns < 0)) if len(returns) > 0 else 0,
                    'zero_count': int(np.sum(returns == 0)) if len(returns) > 0 else 0,
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    return returns


def calculate_enhanced_metrics(
    result: BacktestResult,
    risk_free_rate: float = 0.0,
    periods_per_year: Optional[int] = None  # Auto-detect from equity curve
) -> dict:
    """
    Calculate all enhanced metrics from a BacktestResult.
    
    Args:
        result: BacktestResult object
        risk_free_rate: Risk-free rate for Sharpe/Sortino
        periods_per_year: Number of periods per year
    
    Returns:
        Dictionary of enhanced metrics
    """
    metrics = {}
    
    if not result.trades:
        return {
            'sortino_ratio': 0.0,
            'cagr': 0.0,
            'recovery_factor': 0.0,
            'calmar_ratio': 0.0,
            'expectancy': 0.0,
            'kelly_percent': 0.0,
            'ulcer_index': 0.0,
            'total_return_pct': 0.0,
        }
    
    # Calculate time period in years
    # Industry standard: Use equity curve timestamps for most accurate time span (Backtrader/VectorBT approach)
    # Handle both datetime-indexed and integer-indexed equity curves
    if len(result.equity_curve) > 1:
        # Check if equity curve has datetime index or integer index
        idx = result.equity_curve.index
        if isinstance(idx, pd.DatetimeIndex):
            # Use equity curve timestamps (most accurate - reflects actual backtest period)
            first_time = idx[0]
            last_time = idx[-1]
            time_span_seconds = (last_time - first_time).total_seconds()
            time_span = time_span_seconds / (365.25 * 24 * 3600)
        else:
            # Integer index (trade-level equity curve) - use trade timestamps or default
            if result.trades and len(result.trades) > 0:
                first_trade_time = result.trades[0].entry_time
                last_trade_time = result.trades[-1].exit_time
                time_span_seconds = (last_trade_time - first_trade_time).total_seconds()
                time_span = time_span_seconds / (365.25 * 24 * 3600)
            else:
                # No trades - default to 1 year for trade-level equity curves
                time_span = 1.0
    elif result.trades and len(result.trades) > 0:
        # Fallback to trade timestamps
        first_trade_time = result.trades[0].entry_time
        last_trade_time = result.trades[-1].exit_time
        
        time_span_seconds = (last_trade_time - first_trade_time).total_seconds()
        time_span = time_span_seconds / (365.25 * 24 * 3600)
        
        # Don't artificially inflate time span - use actual time
        # But ensure it's not zero or negative (safety check only)
        if time_span <= 0:
            # Fallback to equity curve if available
            if len(result.equity_curve) > 1:
                time_span_seconds = (result.equity_curve.index[-1] - result.equity_curve.index[0]).total_seconds()
                time_span = time_span_seconds / (365.25 * 24 * 3600)
            else:
                time_span = 1.0  # Default to 1 year if can't calculate
        
        # Only enforce minimum if time span is suspiciously small (< 1 day)
        # This prevents division by zero but doesn't distort short-term results
        # Use actual time span even if it's short - CAGR will be high but that's correct for short periods
        if time_span < (1.0 / 365.25):  # Less than 1 day - likely a calculation error
            import warnings
            warnings.warn(f"Time span calculated as {time_span*365.25:.2f} days, which seems too short. "
                         f"Using equity curve timestamps instead.")
            # Try equity curve as fallback
            if len(result.equity_curve) > 1:
                time_span_seconds = (result.equity_curve.index[-1] - result.equity_curve.index[0]).total_seconds()
                time_span = time_span_seconds / (365.25 * 24 * 3600)
                if time_span < (1.0 / 365.25):  # Still too short
                    warnings.warn(f"Equity curve time span also too short ({time_span*365.25:.2f} days). "
                                 f"Using 1 day minimum for CAGR calculation.")
                    time_span = 1.0 / 365.25  # 1 day minimum only as last resort
    elif len(result.equity_curve) > 1:
        # Fallback to equity curve timestamps
        time_span_seconds = (result.equity_curve.index[-1] - result.equity_curve.index[0]).total_seconds()
        time_span = time_span_seconds / (365.25 * 24 * 3600)
        
        # Only enforce minimum if suspiciously small (< 1 day)
        if time_span < (1.0 / 365.25):  # Less than 1 day
            import warnings
            warnings.warn(f"Equity curve time span calculated as {time_span*365.25:.2f} days, which seems too short. "
                         f"Using 1 day minimum for CAGR calculation.")
            time_span = 1.0 / 365.25  # 1 day minimum only as last resort
    else:
        # No trades or insufficient data - default to 1 year
        time_span = 1.0
    
    # Total return percentage
    total_return_pct = ((result.final_capital / result.initial_capital) - 1.0) * 100.0
    metrics['total_return_pct'] = total_return_pct
    
    # CAGR
    # Formula: CAGR = ((Final Value / Initial Value) ^ (1 / Years)) - 1
    # This correctly handles both positive and negative returns
    # Note: High CAGR (>100%) is possible in crypto, especially in bull markets
    # However, very high CAGR (>500%) may indicate:
    # 1. Over-optimization
    # 2. Look-ahead bias
    # 3. Unrealistic assumptions (slippage, commissions, etc.)
    # 4. Exceptional market conditions (e.g., 2020-2021 crypto bull run)
    # 5. Time span calculation error (too short period)
    
    # Calculate CAGR
    # For very short periods (< 0.1 years = ~36 days), CAGR is not meaningful
    # Skip CAGR calculation and set to None to avoid misleading annualized values
    if time_span < 0.1:  # Less than ~36 days
        # CAGR is not meaningful for such short periods - set to None
        metrics['cagr'] = None
    else:
        metrics['cagr'] = calculate_cagr(result.initial_capital, result.final_capital, time_span)
        
        # Validate CAGR makes sense relative to total return
        # For short periods, CAGR can be much higher than total return when annualized
        # But if it's unreasonably high, there's likely a calculation error
        
        # Check if CAGR is reasonable given total return and time span
        # For a 1-year period, CAGR should approximately equal total return
        # For shorter periods, CAGR will be higher when annualized
        # For longer periods, CAGR will be lower than total return
        
        # Calculate expected CAGR range based on time span
        # If time_span < 0.5 years, CAGR can be 2x+ total return
        # If time_span > 2 years, CAGR should be less than total return
        if time_span < 0.5:  # Less than 6 months
            # For very short periods, CAGR can be much higher
            # But if it's > 10x total return, something is wrong
            max_reasonable_cagr = abs(total_return_pct) * 10.0
            if abs(metrics['cagr']) > max_reasonable_cagr and max_reasonable_cagr > 0:
                import warnings
                warnings.warn(f"CAGR of {metrics['cagr']:.2f}% seems unrealistic for {time_span:.2f} years period "
                             f"with {total_return_pct:.2f}% total return. Time span may be calculated incorrectly.")
                # Use simple annualized return as fallback
                metrics['cagr'] = (total_return_pct / time_span) if time_span > 0 else total_return_pct
    
    # Validate CAGR sign matches total return sign (only if CAGR is not None)
    if metrics['cagr'] is not None:
        if total_return_pct < 0 and metrics['cagr'] > 0:
            # This is a calculation error - fix it
            import warnings
            warnings.warn(f"CAGR calculation error: Total return is {total_return_pct:.2f}% but CAGR is {metrics['cagr']:.2f}%. "
                         f"Initial: {result.initial_capital:.2f}, Final: {result.final_capital:.2f}, Time span: {time_span:.4f} years. "
                         f"Fixing CAGR sign.")
            # If return is negative, CAGR should be negative
            if result.final_capital < result.initial_capital:
                metrics['cagr'] = -abs(metrics['cagr'])
            else:
                # Use simple annualized return as fallback
                metrics['cagr'] = (total_return_pct / time_span) if time_span > 0 else total_return_pct
        
        # Validate CAGR is reasonable
        # For most strategies, CAGR > 200% is exceptional, > 500% is suspicious
        if metrics['cagr'] > 500.0:
            import warnings
            warnings.warn(f"CAGR of {metrics['cagr']:.2f}% is extremely high. Time span: {time_span:.4f} years, "
                         f"Total return: {total_return_pct:.2f}%. This may indicate: "
                         f"1) Time span calculation error (too short), 2) Position sizing issue, "
                         f"3) Unrealistic assumptions, or 4) Exceptional market conditions.")
            
            # Cap at 1000% to prevent display issues, but warn
            if metrics['cagr'] > 1000.0:
                warnings.warn(f"CAGR capped at 1000% (was {metrics['cagr']:.2f}%). "
                             f"Investigate time span calculation and position sizing.")
                metrics['cagr'] = 1000.0
        
        # Cap negative CAGR at -100% (can't lose more than 100%)
        if metrics['cagr'] < -100.0:
            metrics['cagr'] = -100.0
    
    # Recovery factor
    metrics['recovery_factor'] = calculate_recovery_factor(result.total_pnl, result.max_drawdown)
    
    # Calmar ratio (requires CAGR, so set to None if CAGR is None)
    if metrics['cagr'] is not None:
        metrics['calmar_ratio'] = calculate_calmar_ratio(metrics['cagr'], result.max_drawdown)
    else:
        metrics['calmar_ratio'] = None
    
    # Expectancy
    metrics['expectancy'] = calculate_expectancy(result.trades)
    
    # Calculate avg_win and avg_loss from trades
    winning_trades = [t for t in result.trades if t.net_pnl > 0]
    losing_trades = [t for t in result.trades if t.net_pnl <= 0]
    avg_win = np.mean([t.net_pnl for t in winning_trades]) if winning_trades else 0.0
    avg_loss = abs(np.mean([t.net_pnl for t in losing_trades])) if losing_trades else 0.0
    
    # Kelly percentage
    metrics['kelly_percent'] = calculate_kelly_percent(
        result.win_rate / 100.0 if result.win_rate > 1.0 else result.win_rate,  # Convert to decimal if percentage
        avg_win,
        avg_loss
    )
    
    # Ulcer Index
    equity_array = result.equity_curve.values
    metrics['ulcer_index'] = calculate_ulcer_index(equity_array)
    
    # Sortino/Sharpe use equity curve returns (period-based)
    # However, if equity curve is too sparse (< 3 returns), fall back to trade returns
    # This prevents issues where sparse equity curves (e.g., from few trades) produce
    # insufficient data for Sortino calculation (e.g., all positive returns = no downside)
    equity_returns = calculate_equity_returns(result.equity_curve)
    
    # Calculate actual periods per year from equity curve frequency FIRST
    # This is more accurate than assuming daily (252) periods
    # We need this for the excess returns check below
    # Handle both datetime-indexed and integer-indexed equity curves
    if len(result.equity_curve) > 1:
        idx = result.equity_curve.index
        if isinstance(idx, pd.DatetimeIndex):
            # Calculate average time between equity curve points
            time_diffs = idx.to_series().diff().dropna()
            if len(time_diffs) > 0:
                avg_period_seconds = time_diffs.mean().total_seconds()
                if avg_period_seconds > 0:
                    # Calculate periods per year based on actual frequency
                    actual_periods_per_year = (365.25 * 24 * 3600) / avg_period_seconds
                    # Cap at reasonable values (e.g., if data is sub-second, cap at daily)
                    actual_periods_per_year = min(actual_periods_per_year, 252)
                    # Use calculated value, but don't go below 12 (monthly) for very sparse data
                    actual_periods_per_year = max(actual_periods_per_year, 12)
                else:
                    actual_periods_per_year = 252  # default to daily
            else:
                actual_periods_per_year = periods_per_year if periods_per_year else 252
        else:
            # Integer index (trade-level equity curve) - use trade timestamps or default
            if result.trades and len(result.trades) > 0:
                # Calculate from trade timestamps
                trade_times = [t.exit_time for t in result.trades if hasattr(t, 'exit_time')]
                if len(trade_times) > 1:
                    time_diffs = pd.Series(trade_times).diff().dropna()
                    if len(time_diffs) > 0:
                        avg_period_seconds = time_diffs.mean().total_seconds()
                        if avg_period_seconds > 0:
                            actual_periods_per_year = (365.25 * 24 * 3600) / avg_period_seconds
                            actual_periods_per_year = min(actual_periods_per_year, 252)
                            actual_periods_per_year = max(actual_periods_per_year, 12)
                        else:
                            actual_periods_per_year = 252
                    else:
                        actual_periods_per_year = 252
                else:
                    actual_periods_per_year = 252  # default to daily
            else:
                actual_periods_per_year = 252  # default to daily
    else:
        actual_periods_per_year = periods_per_year if periods_per_year else 252
    
    # Check if equity returns have sufficient data and downside returns for Sortino
    # Need at least 3 returns and at least 3 downside returns for meaningful Sortino
    # (Need 3+ downside returns to calculate variance with ddof=1 and avoid edge cases)
    # We check excess returns (after risk-free rate adjustment) to match what calculate_sortino_ratio does
    # This ensures we don't call calculate_sortino_ratio with insufficient downside returns
    # Use the ACTUAL periods_per_year that will be used in calculate_sortino_ratio
    if len(equity_returns) >= 3:
        # Calculate excess returns to match what calculate_sortino_ratio does
        excess_returns_check = equity_returns - risk_free_rate / actual_periods_per_year
        downside_returns_check = excess_returns_check[excess_returns_check < 0]
        downside_count = len(downside_returns_check)
        # Require at least 3 downside returns AND sufficient variance to avoid warnings
        # Check variance to prevent "downside deviation near zero" warnings
        if downside_count >= 3:
            downside_std_check = np.std(downside_returns_check, ddof=1)
            # Require variance to be above threshold (same as calculate_sortino_ratio uses: 1e-10)
            use_equity_returns = downside_std_check >= 1e-10
        else:
            use_equity_returns = False
        # #region agent log
        import json
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'post-fix',
                    'hypothesisId': 'O',
                    'location': 'metrics.py:697',
                    'message': 'equity_returns check calculation',
                    'data': {
                        'equity_returns_len': len(equity_returns),
                        'downside_count': int(downside_count),
                        'use_equity_returns': use_equity_returns,
                        'excess_returns_check_sample': excess_returns_check.tolist()[:10] if len(excess_returns_check) > 0 else [],
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
    else:
        use_equity_returns = False
    
    # #region agent log
    import json
    try:
        with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
            excess_returns_check_val = equity_returns - risk_free_rate / actual_periods_per_year if len(equity_returns) >= 3 else np.array([])
            downside_count_val = int(np.sum(excess_returns_check_val < 0)) if len(excess_returns_check_val) > 0 else 0
            f.write(json.dumps({
                'sessionId': 'debug-session',
                'runId': 'post-fix',
                'hypothesisId': 'G',
                'location': 'metrics.py:680',
                'message': 'use_equity_returns decision',
                'data': {
                    'equity_returns_len': len(equity_returns),
                    'equity_returns_negative_count': int(np.sum(equity_returns < 0)) if len(equity_returns) > 0 else 0,
                    'excess_returns_downside_count': downside_count_val,
                    'actual_periods_per_year': actual_periods_per_year,
                    'use_equity_returns': use_equity_returns,
                    'n_trades': len(result.trades),
                },
                'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
            }) + '\n')
    except: pass
    # #endregion
    
    def _calc_ratios(returns, periods):
        """
        Calculate Sharpe and Sortino ratios from returns.
        
        Sharpe: mean(returns) / std(returns) - no annualization unless periods specified
        Sortino: mean(returns) / std(negative_returns)
        """
        if len(returns) == 0:
            return 0.0, 0.0
        
        # Sortino: use negative returns only
        negative_returns = returns[returns < 0]
        if len(negative_returns) > 1:
            negative_std = np.std(negative_returns, ddof=1)
            if negative_std > 0:
                sortino = np.mean(returns) / negative_std
            else:
                # No downside variance - all returns positive
                sortino = np.inf if np.mean(returns) > 0 else 0.0
        else:
            sortino = 0.0
        
        # Sharpe: mean / std (no annualization by default)
        sharpe = 0.0
        if len(returns) > 1:
            std = np.std(returns, ddof=1)
            if std > 0:
                # Basic Sharpe: mean / std
                sharpe = np.mean(returns) / std
                # Only annualize if periods_per_year is provided and > 0
                if periods and periods > 0:
                    sharpe = sharpe * np.sqrt(periods)
        elif len(returns) == 1:
            # Single return - can't calculate Sharpe properly
            sharpe = 0.0 if returns[0] <= 0 else np.inf
        
        return sortino, sharpe
    
    if use_equity_returns:
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'post-fix',
                    'hypothesisId': 'H',
                    'location': 'metrics.py:690',
                    'message': 'USING equity_returns for Sortino',
                    'data': {
                        'equity_returns_len': len(equity_returns),
                        'actual_periods_per_year': int(actual_periods_per_year),
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        # Calculate Sharpe and Sortino from equity returns
        # Sharpe: mean(returns) / std(returns) - no annualization by default
        # Sortino: mean(returns) / std(negative_returns)
        if len(equity_returns) > 1:
            # Sharpe: basic formula (no annualization)
            std = np.std(equity_returns, ddof=1)
            if std > 0:
                metrics['sharpe_ratio'] = float(np.mean(equity_returns) / std)
                # Only annualize if periods_per_year is provided
                if actual_periods_per_year and actual_periods_per_year > 0:
                    metrics['sharpe_ratio'] = metrics['sharpe_ratio'] * np.sqrt(actual_periods_per_year)
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Sortino: use negative returns only
            negative_returns = equity_returns[equity_returns < 0]
            if len(negative_returns) > 1:
                negative_std = np.std(negative_returns, ddof=1)
                if negative_std > 0:
                    metrics['sortino_ratio'] = float(np.mean(equity_returns) / negative_std)
                    if actual_periods_per_year and actual_periods_per_year > 0:
                        metrics['sortino_ratio'] = metrics['sortino_ratio'] * np.sqrt(actual_periods_per_year)
                else:
                    # No downside variance - all returns positive
                    metrics['sortino_ratio'] = float('inf') if np.mean(equity_returns) > 0 else 0.0
            else:
                metrics['sortino_ratio'] = 0.0
        else:
            metrics['sharpe_ratio'] = 0.0
            metrics['sortino_ratio'] = 0.0
        
        # Add 'sharpe' alias for compatibility
        metrics['sharpe'] = metrics['sharpe_ratio']
    else:
        # Fall back to trade returns if equity returns are insufficient
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'post-fix',
                    'hypothesisId': 'I',
                    'location': 'metrics.py:693',
                    'message': 'FALLING BACK to trade_returns for Sortino',
                    'data': {
                        'equity_returns_len': len(equity_returns),
                        'n_trades': len(result.trades),
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        trade_returns = calculate_trade_returns(result.trades, result.initial_capital)
        
        # #region agent log
        try:
            with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                f.write(json.dumps({
                    'sessionId': 'debug-session',
                    'runId': 'post-fix',
                    'hypothesisId': 'J',
                    'location': 'metrics.py:728',
                    'message': 'trade_returns calculated',
                    'data': {
                        'trade_returns_len': len(trade_returns),
                        'trade_returns_negative_count': int(np.sum(trade_returns < 0)) if len(trade_returns) > 0 else 0,
                        'n_trades': len(result.trades),
                    },
                    'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                }) + '\n')
        except: pass
        # #endregion
        
        # Check if trade returns are also sufficient for Sortino calculation
        # Need at least 3 returns and at least 3 negative returns with sufficient variance
        # (Need 3+ downside returns to calculate variance with ddof=1 and avoid edge cases)
        # Calculate trade_periods_per_year first
        if time_span > 0:
            trades_per_year = len(result.trades) / time_span
            trade_periods_per_year = min(int(trades_per_year), 252)
            trade_periods_per_year = max(trade_periods_per_year, 12)
        else:
            trade_periods_per_year = 12
        
        # Check downside returns and variance
        trade_excess_returns_check = trade_returns - risk_free_rate / trade_periods_per_year
        trade_downside_returns_check = trade_excess_returns_check[trade_excess_returns_check < 0]
        trade_downside_count = len(trade_downside_returns_check)
        if len(trade_returns) >= 3 and trade_downside_count >= 3:
            # Also check variance to prevent "downside deviation near zero" warnings
            trade_downside_std_check = np.std(trade_downside_returns_check, ddof=1)
            trade_sufficient = trade_downside_std_check >= 1e-10
        else:
            trade_sufficient = False
        
        if trade_sufficient:
            # #region agent log
            try:
                with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'post-fix',
                        'hypothesisId': 'K',
                        'location': 'metrics.py:740',
                        'message': 'USING trade_returns for Sortino (sufficient)',
                        'data': {
                            'trade_returns_len': len(trade_returns),
                            'trade_periods_per_year': trade_periods_per_year,
                        },
                        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            # Calculate Sharpe and Sortino from trade returns
            # Sharpe: mean(returns) / std(returns) - no annualization by default
            # Sortino: mean(returns) / std(negative_returns)
            if len(trade_returns) > 1:
                # Sharpe: basic formula (no annualization)
                std = np.std(trade_returns, ddof=1)
                if std > 0:
                    metrics['sharpe_ratio'] = float(np.mean(trade_returns) / std)
                    # Only annualize if periods_per_year is provided
                    if trade_periods_per_year and trade_periods_per_year > 0:
                        metrics['sharpe_ratio'] = metrics['sharpe_ratio'] * np.sqrt(trade_periods_per_year)
                else:
                    metrics['sharpe_ratio'] = 0.0
                
                # Sortino: use negative returns only
                negative_returns = trade_returns[trade_returns < 0]
                if len(negative_returns) > 1:
                    negative_std = np.std(negative_returns, ddof=1)
                    if negative_std > 0:
                        metrics['sortino_ratio'] = float(np.mean(trade_returns) / negative_std)
                        if trade_periods_per_year and trade_periods_per_year > 0:
                            metrics['sortino_ratio'] = metrics['sortino_ratio'] * np.sqrt(trade_periods_per_year)
                    else:
                        metrics['sortino_ratio'] = float('inf') if np.mean(trade_returns) > 0 else 0.0
                else:
                    metrics['sortino_ratio'] = 0.0
            else:
                metrics['sharpe_ratio'] = 0.0
                metrics['sortino_ratio'] = 0.0
            
            # Add 'sharpe' alias for compatibility
            metrics['sharpe'] = metrics['sharpe_ratio']
        else:
            # Insufficient trade returns - set Sortino to 0.0 to avoid misleading warnings
            # This happens when there are very few trades or all trades are winners
            # #region agent log
            try:
                with open('/home/amfree/Documents/CodeBank/Trading/TradingLab/.cursor/debug.log', 'a') as f:
                    f.write(json.dumps({
                        'sessionId': 'debug-session',
                        'runId': 'post-fix',
                        'hypothesisId': 'L',
                        'location': 'metrics.py:750',
                        'message': 'SETTING Sortino to 0.0 (insufficient trade_returns)',
                        'data': {
                            'trade_returns_len': len(trade_returns),
                            'trade_returns_negative_count': int(np.sum(trade_returns < 0)) if len(trade_returns) > 0 else 0,
                            'n_trades': len(result.trades),
                        },
                        'timestamp': int(pd.Timestamp.now().timestamp() * 1000)
                    }) + '\n')
            except: pass
            # #endregion
            metrics['sortino_ratio'] = 0.0
            metrics['sharpe_ratio'] = 0.0
    
    # Profit factor: gross wins / gross losses
    gross_wins = sum(t.net_pnl for t in winning_trades)
    gross_losses = abs(sum(t.net_pnl for t in losing_trades))
    if gross_losses == 0:
        metrics['profit_factor'] = float('inf') if gross_wins > 0 else 0.0
    else:
        metrics['profit_factor'] = gross_wins / gross_losses
    
    # Ensure 'sharpe' alias is present (for compatibility with permutation test)
    if 'sharpe_ratio' in metrics and 'sharpe' not in metrics:
        metrics['sharpe'] = metrics['sharpe_ratio']
    
    # Ensure 'final_pnl' is present
    if 'final_pnl' not in metrics:
        metrics['final_pnl'] = result.total_pnl
    
    # Ensure 'total_return_pct' is present
    if 'total_return_pct' not in metrics:
        metrics['total_return_pct'] = ((result.final_capital / result.initial_capital) - 1.0) * 100.0 if result.initial_capital > 0 else 0.0
    
    # Calculate Edge Latency metric
    # Extract R-multiples from trades
    if result.trades and len(result.trades) > 0:
        r_multiples = np.array([t.r_multiple for t in result.trades])
        
        # Calculate trades per year for time conversion
        trades_per_year = None
        if time_span > 0:
            trades_per_year = len(result.trades) / time_span
        
        # Compute edge latency
        edge_latency_result = compute_edge_latency(
            r_multiples=r_multiples,
            confidence=0.95,
            power=0.80,
            trades_per_year=trades_per_year
        )
        
        # Add to metrics dict
        metrics['edge_latency'] = edge_latency_result
        
        # Calculate MFE/MAE analysis (industry standard terminology)
        mfe_mae_analysis = analyze_mfe_mae(result.trades)
        metrics['mfe_mae'] = get_mfe_mae_summary(mfe_mae_analysis)
    else:
        # No trades - set edge latency to None
        metrics['edge_latency'] = {
            'mean_r': None,
            'std_r': None,
            'signal_ratio': None,
            'required_trades': None,
            'years_to_significance': None,
            'confidence': 0.95,
            'power': 0.80
        }
        # No trades - set MFE/MAE to empty
        metrics['mfe_mae'] = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'avg_mfe_r': 0.0,
            'avg_mae_r': 0.0,
            'avg_winning_mfe_r': 0.0,
            'avg_losing_mfe_r': 0.0,
            'avg_losing_mae_r': 0.0,
            'mfe_reached_0_5r': 0,
            'mfe_reached_1r': 0,
            'mfe_reached_2r': 0,
            'mfe_reached_3r_plus': 0,
            'trades_mae_ge_095r': 0,
            'losing_trades_mfe_before_stop': 0,
            'avg_mfe_efficiency': 0.0,
            'avg_mae_containment': 0.0,
            'avg_mae_containment_losing': 0.0,
            'avg_loss_recovery_ratio': None,
            'median_loss_recovery_ratio': None,
            'recovery_trades': 0,
            'mfe_distribution': {},
            'mae_distribution': {},
            'avg_time_to_mfe_bars': None,
            'avg_time_to_mae_bars': None,
            'median_time_to_mfe_bars': None,
            'median_time_to_mae_bars': None,
            'mfe_by_exit_reason': {},
            'mae_by_exit_reason': {},
            'mfe_pnl_correlation': None,
            'mae_pnl_correlation': None,
            'mfe_percentiles': {},
            'mae_percentiles': {}
        }
    
    return metrics

