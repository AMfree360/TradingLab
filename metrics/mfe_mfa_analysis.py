"""MFE/MAE (Maximum Favorable/Adverse Excursion) Analysis Module.

This module provides comprehensive analysis of trade MFE/MAE metrics,
including R-multiple grouping and statistical analysis.

Industry Standard Terminology:
- MFE: Maximum Favorable Excursion
- MAE: Maximum Adverse Excursion (replaces MFA)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from engine.backtest_engine import Trade
try:
    from scipy.stats import pearsonr
except ImportError:
    # Fallback if scipy not available
    def pearsonr(x, y):
        """Simple correlation fallback."""
        if len(x) != len(y) or len(x) < 2:
            return (0.0, 1.0)
        x_arr = np.array(x)
        y_arr = np.array(y)
        corr = np.corrcoef(x_arr, y_arr)[0, 1]
        return (corr if not np.isnan(corr) else 0.0, 1.0)


@dataclass
class MFEMAEAnalysis:
    """MFE/MAE analysis results (industry standard terminology)."""
    # Overall statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    
    # Average MFE/MAE
    avg_mfe_r: float  # Average MFE in R-multiples
    avg_mae_r: float  # Average MAE in R-multiples (industry standard)
    avg_winning_mfe_r: float  # Average MFE for winning trades
    avg_losing_mfe_r: float  # Average MFE for losing trades
    avg_losing_mae_r: float  # Average MAE for losing trades
    
    # R-multiple distribution for MFE
    mfe_reached_0_5r: int  # Trades that reached 0.5R MFE
    mfe_reached_1r: int  # Trades that reached 1R MFE
    mfe_reached_2r: int  # Trades that reached 2R MFE
    mfe_reached_3r_plus: int  # Trades that reached 3R+ MFE
    
    # Losing trade analysis
    trades_mae_ge_095r: int  # Trades with MAE ≥ 0.95R (near full stop exposure)
    losing_trades_mfe_before_stop: int  # Losing trades with MFE before hitting stop
    
    # MFE efficiency (MAE efficiency removed - invalid metric that can exceed 100%)
    avg_mfe_efficiency: float  # Average (Final P&L / MFE) for winning trades
    
    # Industry-standard MAE metrics
    avg_mae_containment: float  # Average MAE containment ratio (abs(mae_r) / 1.0) for all trades
    avg_mae_containment_losing: float  # Average MAE containment for losing trades only
    
    # Loss Recovery Ratio (losing trades only)
    avg_loss_recovery_ratio: Optional[float]  # Mean: mfe_r / abs(mae_r) for losing trades
    median_loss_recovery_ratio: Optional[float]  # Median loss recovery ratio
    
    # Recovery trades (went negative but recovered)
    recovery_trades: int  # Trades with MAE < 0 but final P&L > 0
    
    # Detailed breakdowns
    mfe_distribution: Dict[str, int]  # Distribution by R-multiple buckets
    mae_distribution: Dict[str, int]  # Distribution by R-multiple buckets (industry standard)
    
    # Time to MFE/MAE (in bars - approximate)
    avg_time_to_mfe_bars: Optional[float]  # Average bars to reach MFE
    avg_time_to_mae_bars: Optional[float]  # Average bars to reach MAE
    median_time_to_mfe_bars: Optional[float]  # Median bars to reach MFE
    median_time_to_mae_bars: Optional[float]  # Median bars to reach MAE
    
    # MFE/MAE by exit reason
    mfe_by_exit_reason: Dict[str, Dict[str, float]]  # MFE stats grouped by exit reason
    mae_by_exit_reason: Dict[str, Dict[str, float]]  # MAE stats grouped by exit reason
    
    # MFE/MAE correlation with final P&L
    mfe_pnl_correlation: Optional[float]  # Correlation between MFE and final P&L
    mae_pnl_correlation: Optional[float]  # Correlation between MAE and final P&L
    
    # MFE/MAE percentiles
    mfe_percentiles: Dict[str, float]  # 25th, 50th, 75th, 90th, 95th percentiles
    mae_percentiles: Dict[str, float]  # 25th, 50th, 75th, 90th, 95th percentiles


def analyze_mfe_mae(trades: List[Trade]) -> MFEMAEAnalysis:
    """
    Analyze MFE/MAE metrics for a list of trades.
    
    Industry standard analysis including:
    - MFE (Maximum Favorable Excursion)
    - MAE (Maximum Adverse Excursion) - replaces MFA
    - MAE Containment Ratio (bounded, interpretable)
    - Loss Recovery Ratio (for losing trades)
    
    Args:
        trades: List of Trade objects with MFE/MAE data
        
    Returns:
        MFEMAEAnalysis object with comprehensive statistics
    """
    if not trades:
        return _empty_analysis()
    
    # Filter trades with valid MFE/MAE data
    valid_trades = [t for t in trades if t.mfe_r is not None and t.mae_r is not None]
    
    if not valid_trades:
        return _empty_analysis()
    
    winning_trades = [t for t in valid_trades if t.pnl_after_costs > 0]
    losing_trades = [t for t in valid_trades if t.pnl_after_costs < 0]
    
    # Calculate averages
    avg_mfe_r = np.mean([t.mfe_r for t in valid_trades if t.mfe_r is not None])
    avg_mae_r = np.mean([t.mae_r for t in valid_trades if t.mae_r is not None])
    
    avg_winning_mfe_r = np.mean([t.mfe_r for t in winning_trades if t.mfe_r is not None]) if winning_trades else 0.0
    avg_losing_mfe_r = np.mean([t.mfe_r for t in losing_trades if t.mfe_r is not None]) if losing_trades else 0.0
    avg_losing_mae_r = np.mean([t.mae_r for t in losing_trades if t.mae_r is not None]) if losing_trades else 0.0
    
    # MFE R-multiple distribution
    mfe_reached_0_5r = sum(1 for t in valid_trades if t.mfe_r is not None and t.mfe_r >= 0.5)
    mfe_reached_1r = sum(1 for t in valid_trades if t.mfe_r is not None and t.mfe_r >= 1.0)
    mfe_reached_2r = sum(1 for t in valid_trades if t.mfe_r is not None and t.mfe_r >= 2.0)
    mfe_reached_3r_plus = sum(1 for t in valid_trades if t.mfe_r is not None and t.mfe_r >= 3.0)
    
    # Losing trade analysis
    # Trades with MAE ≥ 0.95R (near full stop exposure) - clearer than "full stop hits"
    trades_mae_ge_095r = sum(1 for t in valid_trades if t.mae_r is not None and abs(t.mae_r) >= 0.95)
    losing_trades_mfe_before_stop = sum(1 for t in losing_trades if t.mfe_r is not None and t.mfe_r > 0)
    
    # MFE efficiency (MAE efficiency removed - invalid metric that can exceed 100%)
    # MFE efficiency: how much of maximum favorable excursion was captured
    winning_with_mfe = [t for t in winning_trades if t.mfe_r is not None and t.mfe_r > 0]
    if winning_with_mfe:
        mfe_efficiencies = []
        for t in winning_with_mfe:
            if t.mfe_pnl > 0:
                efficiency = t.pnl_after_costs / t.mfe_pnl
                mfe_efficiencies.append(efficiency)
        avg_mfe_efficiency = np.mean(mfe_efficiencies) if mfe_efficiencies else 0.0
    else:
        avg_mfe_efficiency = 0.0
    
    # MAE Containment Ratio (industry standard, bounded metric)
    # For each trade: mae_containment = abs(mae_r) / 1.0 (assuming 1R = full stop distance)
    mae_containments = [abs(t.mae_r) for t in valid_trades if t.mae_r is not None]
    avg_mae_containment = np.mean(mae_containments) if mae_containments else 0.0
    
    # MAE containment for losing trades only
    losing_mae_containments = [abs(t.mae_r) for t in losing_trades if t.mae_r is not None]
    avg_mae_containment_losing = np.mean(losing_mae_containments) if losing_mae_containments else 0.0
    
    # Loss Recovery Ratio (losing trades only)
    # loss_recovery_ratio = mfe_r / abs(mae_r) - identifies "almost-winners"
    loss_recovery_ratios = []
    for t in losing_trades:
        if t.mfe_r is not None and t.mae_r is not None and t.mae_r != 0:
            ratio = t.mfe_r / abs(t.mae_r)
            loss_recovery_ratios.append(ratio)
    
    avg_loss_recovery_ratio = np.mean(loss_recovery_ratios) if loss_recovery_ratios else None
    median_loss_recovery_ratio = np.median(loss_recovery_ratios) if loss_recovery_ratios else None
    
    # Recovery trades (went negative but recovered)
    recovery_trades = sum(1 for t in valid_trades if t.mae_r is not None and t.mae_r < 0 and t.pnl_after_costs > 0)
    
    # Distribution buckets
    mfe_distribution = _calculate_distribution([t.mfe_r for t in valid_trades if t.mfe_r is not None])
    mae_distribution = _calculate_distribution([t.mae_r for t in valid_trades if t.mae_r is not None])
    
    # Time to MFE/MAE (in bars - approximate using trade duration)
    time_to_mfe_bars = []
    time_to_mae_bars = []
    for t in valid_trades:
        if t.mfe_reached_at is not None and t.entry_time is not None:
            # Approximate bars: assume 1 bar per time unit (will vary by timeframe)
            # For more accuracy, would need to track actual bar count
            duration = (t.mfe_reached_at - t.entry_time).total_seconds() / 60  # minutes
            time_to_mfe_bars.append(duration)
        if t.mae_reached_at is not None and t.entry_time is not None:
            duration = (t.mae_reached_at - t.entry_time).total_seconds() / 60  # minutes
            time_to_mae_bars.append(duration)
    
    avg_time_to_mfe_bars = np.mean(time_to_mfe_bars) if time_to_mfe_bars else None
    avg_time_to_mae_bars = np.mean(time_to_mae_bars) if time_to_mae_bars else None
    median_time_to_mfe_bars = np.median(time_to_mfe_bars) if time_to_mfe_bars else None
    median_time_to_mae_bars = np.median(time_to_mae_bars) if time_to_mae_bars else None
    
    # MFE/MAE by exit reason
    mfe_by_exit_reason = _calculate_by_exit_reason(valid_trades, 'mfe_r')
    mae_by_exit_reason = _calculate_by_exit_reason(valid_trades, 'mae_r')
    
    # MFE/MAE correlation with final P&L
    mfe_values = [t.mfe_r for t in valid_trades if t.mfe_r is not None]
    pnl_values = [t.pnl_after_costs for t in valid_trades if t.mfe_r is not None]
    mfe_pnl_correlation = None
    if len(mfe_values) > 1 and len(pnl_values) > 1:
        try:
            corr, _ = pearsonr(mfe_values, pnl_values)
            mfe_pnl_correlation = float(corr) if not np.isnan(corr) else None
        except:
            mfe_pnl_correlation = None
    
    mae_values = [t.mae_r for t in valid_trades if t.mae_r is not None]
    pnl_values_mae = [t.pnl_after_costs for t in valid_trades if t.mae_r is not None]
    mae_pnl_correlation = None
    if len(mae_values) > 1 and len(pnl_values_mae) > 1:
        try:
            corr, _ = pearsonr(mae_values, pnl_values_mae)
            mae_pnl_correlation = float(corr) if not np.isnan(corr) else None
        except:
            mae_pnl_correlation = None
    
    # MFE/MAE percentiles
    mfe_percentiles = _calculate_percentiles([t.mfe_r for t in valid_trades if t.mfe_r is not None])
    mae_percentiles = _calculate_percentiles([t.mae_r for t in valid_trades if t.mae_r is not None])
    
    return MFEMAEAnalysis(
        total_trades=len(valid_trades),
        winning_trades=len(winning_trades),
        losing_trades=len(losing_trades),
        avg_mfe_r=float(avg_mfe_r),
        avg_mae_r=float(avg_mae_r),
        avg_winning_mfe_r=float(avg_winning_mfe_r),
        avg_losing_mfe_r=float(avg_losing_mfe_r),
        avg_losing_mae_r=float(avg_losing_mae_r),
        mfe_reached_0_5r=mfe_reached_0_5r,
        mfe_reached_1r=mfe_reached_1r,
        mfe_reached_2r=mfe_reached_2r,
        mfe_reached_3r_plus=mfe_reached_3r_plus,
        trades_mae_ge_095r=trades_mae_ge_095r,
        losing_trades_mfe_before_stop=losing_trades_mfe_before_stop,
        avg_mfe_efficiency=float(avg_mfe_efficiency),
        avg_mae_containment=float(avg_mae_containment),
        avg_mae_containment_losing=float(avg_mae_containment_losing),
        avg_loss_recovery_ratio=float(avg_loss_recovery_ratio) if avg_loss_recovery_ratio is not None else None,
        median_loss_recovery_ratio=float(median_loss_recovery_ratio) if median_loss_recovery_ratio is not None else None,
        recovery_trades=recovery_trades,
        mfe_distribution=mfe_distribution,
        mae_distribution=mae_distribution,
        avg_time_to_mfe_bars=float(avg_time_to_mfe_bars) if avg_time_to_mfe_bars is not None else None,
        avg_time_to_mae_bars=float(avg_time_to_mae_bars) if avg_time_to_mae_bars is not None else None,
        median_time_to_mfe_bars=float(median_time_to_mfe_bars) if median_time_to_mfe_bars is not None else None,
        median_time_to_mae_bars=float(median_time_to_mae_bars) if median_time_to_mae_bars is not None else None,
        mfe_by_exit_reason=mfe_by_exit_reason,
        mae_by_exit_reason=mae_by_exit_reason,
        mfe_pnl_correlation=mfe_pnl_correlation,
        mae_pnl_correlation=mae_pnl_correlation,
        mfe_percentiles=mfe_percentiles,
        mae_percentiles=mae_percentiles
    )


def _empty_analysis() -> MFEMAEAnalysis:
    """Return empty analysis for no trades."""
    return MFEMAEAnalysis(
        total_trades=0,
        winning_trades=0,
        losing_trades=0,
        avg_mfe_r=0.0,
        avg_mae_r=0.0,
        avg_winning_mfe_r=0.0,
        avg_losing_mfe_r=0.0,
        avg_losing_mae_r=0.0,
        mfe_reached_0_5r=0,
        mfe_reached_1r=0,
        mfe_reached_2r=0,
        mfe_reached_3r_plus=0,
        trades_mae_ge_095r=0,
        losing_trades_mfe_before_stop=0,
        avg_mfe_efficiency=0.0,
        avg_mae_containment=0.0,
        avg_mae_containment_losing=0.0,
        avg_loss_recovery_ratio=None,
        median_loss_recovery_ratio=None,
        recovery_trades=0,
        mfe_distribution={},
        mae_distribution={},
        avg_time_to_mfe_bars=None,
        avg_time_to_mae_bars=None,
        median_time_to_mfe_bars=None,
        median_time_to_mae_bars=None,
        mfe_by_exit_reason={},
        mae_by_exit_reason={},
        mfe_pnl_correlation=None,
        mae_pnl_correlation=None,
        mfe_percentiles={},
        mae_percentiles={}
    )


def _calculate_distribution(values: List[float]) -> Dict[str, int]:
    """Calculate distribution of values into R-multiple buckets."""
    buckets = {
        '< -2R': 0,
        '-2R to -1R': 0,
        '-1R to -0.5R': 0,
        '-0.5R to 0R': 0,
        '0R to 0.5R': 0,
        '0.5R to 1R': 0,
        '1R to 2R': 0,
        '2R to 3R': 0,
        '3R+': 0
    }
    
    for val in values:
        if val < -2.0:
            buckets['< -2R'] += 1
        elif val < -1.0:
            buckets['-2R to -1R'] += 1
        elif val < -0.5:
            buckets['-1R to -0.5R'] += 1
        elif val < 0.0:
            buckets['-0.5R to 0R'] += 1
        elif val < 0.5:
            buckets['0R to 0.5R'] += 1
        elif val < 1.0:
            buckets['0.5R to 1R'] += 1
        elif val < 2.0:
            buckets['1R to 2R'] += 1
        elif val < 3.0:
            buckets['2R to 3R'] += 1
        else:
            buckets['3R+'] += 1
    
    return buckets


def _calculate_by_exit_reason(trades: List[Trade], metric: str) -> Dict[str, Dict[str, float]]:
    """Calculate MFE/MAE statistics grouped by exit reason.
    
    Args:
        trades: List of trades
        metric: 'mfe_r' or 'mae_r'
        
    Returns:
        Dictionary mapping exit reason to statistics (count, avg, median, min, max)
    """
    by_reason: Dict[str, List[float]] = {}
    
    for t in trades:
        exit_reason = t.exit_reason or 'unknown'
        value = getattr(t, metric, None)
        if value is not None:
            if exit_reason not in by_reason:
                by_reason[exit_reason] = []
            by_reason[exit_reason].append(value)
    
    result = {}
    for reason, values in by_reason.items():
        if values:
            result[reason] = {
                'count': len(values),
                'avg': float(np.mean(values)),
                'median': float(np.median(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values))
            }
    
    return result


def _calculate_percentiles(values: List[float]) -> Dict[str, float]:
    """Calculate percentiles for MFE/MAE values.
    
    Args:
        values: List of MFE or MAE values
        
    Returns:
        Dictionary with 25th, 50th, 75th, 90th, 95th percentiles
    """
    if not values:
        return {}
    
    percentiles = np.percentile(values, [25, 50, 75, 90, 95])
    return {
        'p25': float(percentiles[0]),
        'p50': float(percentiles[1]),
        'p75': float(percentiles[2]),
        'p90': float(percentiles[3]),
        'p95': float(percentiles[4])
    }


def get_mfe_mae_summary(analysis: MFEMAEAnalysis) -> Dict:
    """Get a summary dictionary of MFE/MAE analysis for reports."""
    return {
        'total_trades': analysis.total_trades,
        'winning_trades': analysis.winning_trades,
        'losing_trades': analysis.losing_trades,
        'avg_mfe_r': analysis.avg_mfe_r,
        'avg_mae_r': analysis.avg_mae_r,
        'avg_winning_mfe_r': analysis.avg_winning_mfe_r,
        'avg_losing_mfe_r': analysis.avg_losing_mfe_r,
        'avg_losing_mae_r': analysis.avg_losing_mae_r,
        'mfe_reached_0_5r': analysis.mfe_reached_0_5r,
        'mfe_reached_1r': analysis.mfe_reached_1r,
        'mfe_reached_2r': analysis.mfe_reached_2r,
        'mfe_reached_3r_plus': analysis.mfe_reached_3r_plus,
        'trades_mae_ge_095r': analysis.trades_mae_ge_095r,
        'losing_trades_mfe_before_stop': analysis.losing_trades_mfe_before_stop,
        'avg_mfe_efficiency': analysis.avg_mfe_efficiency,
        'avg_mae_containment': analysis.avg_mae_containment,
        'avg_mae_containment_losing': analysis.avg_mae_containment_losing,
        'avg_loss_recovery_ratio': analysis.avg_loss_recovery_ratio,
        'median_loss_recovery_ratio': analysis.median_loss_recovery_ratio,
        'recovery_trades': analysis.recovery_trades,
        'mfe_distribution': analysis.mfe_distribution,
        'mae_distribution': analysis.mae_distribution,
        'avg_time_to_mfe_bars': analysis.avg_time_to_mfe_bars,
        'avg_time_to_mae_bars': analysis.avg_time_to_mae_bars,
        'median_time_to_mfe_bars': analysis.median_time_to_mfe_bars,
        'median_time_to_mae_bars': analysis.median_time_to_mae_bars,
        'mfe_by_exit_reason': analysis.mfe_by_exit_reason,
        'mae_by_exit_reason': analysis.mae_by_exit_reason,
        'mfe_pnl_correlation': analysis.mfe_pnl_correlation,
        'mae_pnl_correlation': analysis.mae_pnl_correlation,
        'mfe_percentiles': analysis.mfe_percentiles,
        'mae_percentiles': analysis.mae_percentiles
    }

