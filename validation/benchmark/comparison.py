"""Trade and metric comparison logic.

Compares trades trade-by-trade and metrics between Trading Lab and external platforms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import timedelta

from engine.backtest_engine import Trade, BacktestResult
from metrics.metrics import calculate_enhanced_metrics


@dataclass
class TradeMatch:
    """Result of matching a Trading Lab trade with an external platform trade."""
    trading_lab_trade: Trade
    external_trade: Trade
    match_score: float  # 0.0 to 1.0, where 1.0 is perfect match
    differences: Dict[str, Tuple[float, float]]  # Field name -> (trading_lab_value, external_value)
    is_matched: bool  # True if match_score >= threshold


@dataclass
class TradeComparison:
    """Results of trade-by-trade comparison."""
    total_trading_lab_trades: int
    total_external_trades: int
    matched_trades: int
    unmatched_trading_lab: List[Trade]
    unmatched_external: List[Trade]
    matches: List[TradeMatch]
    match_rate: float  # Percentage of trades matched
    avg_match_score: float  # Average match score for matched trades
    
    def is_equivalent(self, threshold: float = 0.995) -> bool:
        """Check if engine equivalence is achieved (≥99.5% match rate)."""
        return self.match_rate >= threshold


@dataclass
class MetricComparison:
    """Results of metric comparison."""
    metrics: Dict[str, Dict[str, float]]  # metric_name -> {'trading_lab': val, 'external': val, 'diff': val, 'diff_pct': val}
    tolerance_checks: Dict[str, bool]  # metric_name -> passed tolerance check
    overall_match: bool  # True if all critical metrics are within tolerance
    
    def get_summary(self) -> Dict[str, any]:
        """Get summary of metric comparison."""
        return {
            'total_metrics': len(self.metrics),
            'metrics_within_tolerance': sum(1 for v in self.tolerance_checks.values() if v),
            'overall_match': self.overall_match,
        }


@dataclass
class ComparisonResult:
    """Complete comparison result between Trading Lab and external platform."""
    trade_comparison: TradeComparison
    metric_comparison: MetricComparison
    platform_name: str
    strategy_name: str
    data_period: Tuple[pd.Timestamp, pd.Timestamp]
    
    def is_equivalent(self) -> bool:
        """Check if Trading Lab is equivalent to external platform."""
        return (
            self.trade_comparison.is_equivalent() and
            self.metric_comparison.overall_match
        )


class TradeComparator:
    """Compares trades trade-by-trade between Trading Lab and external platform."""
    
    def __init__(
        self,
        time_tolerance_seconds: int = 60,  # 1 minute tolerance for timestamps
        price_tolerance_pct: float = 0.01,  # 0.01% tolerance for prices
        quantity_tolerance_pct: float = 0.01,  # 0.01% tolerance for quantity
        match_threshold: float = 0.95  # Minimum match score to consider trades matched
    ):
        """
        Initialize trade comparator.
        
        Args:
            time_tolerance_seconds: Maximum time difference (seconds) to consider trades matched
            price_tolerance_pct: Maximum price difference (percentage) to consider matched
            quantity_tolerance_pct: Maximum quantity difference (percentage) to consider matched
            match_threshold: Minimum match score (0.0-1.0) to consider trades matched
        """
        self.time_tolerance = timedelta(seconds=time_tolerance_seconds)
        self.price_tolerance_pct = price_tolerance_pct
        self.quantity_tolerance_pct = quantity_tolerance_pct
        self.match_threshold = match_threshold
    
    def compare_trades(
        self,
        trading_lab_trades: List[Trade],
        external_trades: List[Trade]
    ) -> TradeComparison:
        """Compare trades trade-by-trade.
        
        Args:
            trading_lab_trades: Trades from Trading Lab
            external_trades: Trades from external platform
            
        Returns:
            TradeComparison result
        """
        if not trading_lab_trades and not external_trades:
            return TradeComparison(
                total_trading_lab_trades=0,
                total_external_trades=0,
                matched_trades=0,
                unmatched_trading_lab=[],
                unmatched_external=[],
                matches=[],
                match_rate=1.0,
                avg_match_score=1.0,
            )
        
        # Sort trades by entry time
        trading_lab_sorted = sorted(trading_lab_trades, key=lambda t: t.entry_time)
        external_sorted = sorted(external_trades, key=lambda t: t.entry_time)
        
        matches = []
        matched_trading_lab_indices = set()
        matched_external_indices = set()
        
        # Match trades based on entry time proximity
        for i, tl_trade in enumerate(trading_lab_sorted):
            best_match = None
            best_score = 0.0
            best_external_idx = None
            
            for j, ext_trade in enumerate(external_sorted):
                if j in matched_external_indices:
                    continue
                
                match_score, differences = self._calculate_match_score(tl_trade, ext_trade)
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = (ext_trade, differences)
                    best_external_idx = j
            
            if best_match and best_score >= self.match_threshold:
                ext_trade, differences = best_match
                matches.append(TradeMatch(
                    trading_lab_trade=tl_trade,
                    external_trade=ext_trade,
                    match_score=best_score,
                    differences=differences,
                    is_matched=True,
                ))
                matched_trading_lab_indices.add(i)
                matched_external_indices.add(best_external_idx)
            else:
                # No good match found
                matches.append(TradeMatch(
                    trading_lab_trade=tl_trade,
                    external_trade=None,
                    match_score=best_score if best_match else 0.0,
                    differences={},
                    is_matched=False,
                ))
        
        # Find unmatched external trades
        unmatched_external = [
            ext_trade for j, ext_trade in enumerate(external_sorted)
            if j not in matched_external_indices
        ]
        
        unmatched_trading_lab = [
            tl_trade for i, tl_trade in enumerate(trading_lab_sorted)
            if i not in matched_trading_lab_indices
        ]
        
        matched_count = len([m for m in matches if m.is_matched])
        match_rate = matched_count / len(trading_lab_trades) if trading_lab_trades else 0.0
        avg_match_score = np.mean([m.match_score for m in matches if m.is_matched]) if matched_count > 0 else 0.0
        
        return TradeComparison(
            total_trading_lab_trades=len(trading_lab_trades),
            total_external_trades=len(external_trades),
            matched_trades=matched_count,
            unmatched_trading_lab=unmatched_trading_lab,
            unmatched_external=unmatched_external,
            matches=matches,
            match_rate=match_rate,
            avg_match_score=avg_match_score,
        )
    
    def _calculate_match_score(
        self,
        tl_trade: Trade,
        ext_trade: Trade
    ) -> Tuple[float, Dict[str, Tuple[float, float]]]:
        """Calculate match score between two trades.
        
        Returns:
            Tuple of (match_score, differences_dict)
        """
        differences = {}
        scores = []
        
        # Entry time match (weight: 0.3)
        time_diff = abs((tl_trade.entry_time - ext_trade.entry_time).total_seconds())
        if time_diff <= self.time_tolerance.total_seconds():
            time_score = 1.0 - (time_diff / self.time_tolerance.total_seconds())
        else:
            time_score = 0.0
        scores.append(('entry_time', time_score, 0.3))
        differences['entry_time'] = (time_diff, self.time_tolerance.total_seconds())
        
        # Entry price match (weight: 0.2)
        price_diff_pct = abs(tl_trade.entry_price - ext_trade.entry_price) / max(tl_trade.entry_price, ext_trade.entry_price) * 100
        if price_diff_pct <= self.price_tolerance_pct:
            price_score = 1.0 - (price_diff_pct / self.price_tolerance_pct)
        else:
            price_score = max(0.0, 1.0 - (price_diff_pct / (self.price_tolerance_pct * 10)))
        scores.append(('entry_price', price_score, 0.2))
        differences['entry_price'] = (price_diff_pct, self.price_tolerance_pct)
        
        # Direction match (weight: 0.2)
        direction_score = 1.0 if tl_trade.direction == ext_trade.direction else 0.0
        scores.append(('direction', direction_score, 0.2))
        differences['direction'] = (1.0 if tl_trade.direction == ext_trade.direction else 0.0, 1.0)
        
        # Quantity match (weight: 0.15)
        if tl_trade.quantity > 0 and ext_trade.quantity > 0:
            qty_diff_pct = abs(tl_trade.quantity - ext_trade.quantity) / max(tl_trade.quantity, ext_trade.quantity) * 100
            if qty_diff_pct <= self.quantity_tolerance_pct:
                qty_score = 1.0 - (qty_diff_pct / self.quantity_tolerance_pct)
            else:
                qty_score = max(0.0, 1.0 - (qty_diff_pct / (self.quantity_tolerance_pct * 10)))
        else:
            qty_score = 0.0
            qty_diff_pct = float('inf')
        scores.append(('quantity', qty_score, 0.15))
        differences['quantity'] = (qty_diff_pct, self.quantity_tolerance_pct)
        
        # Exit price match (weight: 0.15)
        exit_price_diff_pct = abs(tl_trade.exit_price - ext_trade.exit_price) / max(tl_trade.exit_price, ext_trade.exit_price) * 100
        if exit_price_diff_pct <= self.price_tolerance_pct:
            exit_price_score = 1.0 - (exit_price_diff_pct / self.price_tolerance_pct)
        else:
            exit_price_score = max(0.0, 1.0 - (exit_price_diff_pct / (self.price_tolerance_pct * 10)))
        scores.append(('exit_price', exit_price_score, 0.15))
        differences['exit_price'] = (exit_price_diff_pct, self.price_tolerance_pct)
        
        # Calculate weighted average
        total_weight = sum(weight for _, _, weight in scores)
        match_score = sum(score * weight for _, score, weight in scores) / total_weight if total_weight > 0 else 0.0
        
        return match_score, differences


class MetricComparator:
    """Compares metrics between Trading Lab and external platform."""
    
    def __init__(
        self,
        tolerance_pct: float = 5.0,  # 5% tolerance for most metrics
        critical_metrics: Optional[List[str]] = None
    ):
        """
        Initialize metric comparator.
        
        Args:
            tolerance_pct: Percentage tolerance for metric differences
            critical_metrics: List of critical metrics that must match (default: profit_factor, net_profit, win_rate)
        """
        self.tolerance_pct = tolerance_pct
        self.critical_metrics = critical_metrics or ['profit_factor', 'net_profit', 'win_rate']
    
    def compare_metrics(
        self,
        trading_lab_result: BacktestResult,
        external_metrics: Dict[str, float],
        risk_free_rate: float = 0.0
    ) -> MetricComparison:
        """Compare metrics between Trading Lab and external platform.
        
        Args:
            trading_lab_result: BacktestResult from Trading Lab
            external_metrics: Dictionary of metrics from external platform
            risk_free_rate: Risk-free rate for Sharpe/Sortino calculations
            
        Returns:
            MetricComparison result
        """
        # Calculate Trading Lab metrics
        tl_metrics = self._calculate_trading_lab_metrics(trading_lab_result, risk_free_rate)
        
        # Compare metrics
        comparison_metrics = {}
        tolerance_checks = {}
        
        # Standard metrics to compare
        metric_names = [
            'net_profit', 'profit_factor', 'win_rate', 'total_trades',
            'avg_win', 'avg_loss', 'max_drawdown', 'sharpe_ratio',
            'sortino_ratio', 'expectancy', 'recovery_factor'
        ]
        
        for metric_name in metric_names:
            if metric_name in tl_metrics and metric_name in external_metrics:
                tl_val = tl_metrics[metric_name]
                ext_val = external_metrics[metric_name]
                
                # Handle zero/inf cases
                if tl_val == 0 and ext_val == 0:
                    diff = 0.0
                    diff_pct = 0.0
                    passed = True
                elif tl_val == 0 or ext_val == 0:
                    # One is zero, other is not - significant difference
                    diff = abs(tl_val - ext_val)
                    diff_pct = 100.0
                    passed = False
                else:
                    diff = abs(tl_val - ext_val)
                    diff_pct = (diff / abs(ext_val)) * 100.0
                    passed = diff_pct <= self.tolerance_pct
                
                comparison_metrics[metric_name] = {
                    'trading_lab': tl_val,
                    'external': ext_val,
                    'diff': diff,
                    'diff_pct': diff_pct,
                }
                tolerance_checks[metric_name] = passed
        
        # Check overall match (all critical metrics must pass)
        overall_match = all(
            tolerance_checks.get(metric, False)
            for metric in self.critical_metrics
            if metric in comparison_metrics
        )
        
        return MetricComparison(
            metrics=comparison_metrics,
            tolerance_checks=tolerance_checks,
            overall_match=overall_match,
        )
    
    def _calculate_trading_lab_metrics(
        self,
        result: BacktestResult,
        risk_free_rate: float = 0.0
    ) -> Dict[str, float]:
        """Calculate all metrics from Trading Lab BacktestResult."""
        metrics = {}
        
        # Basic metrics from BacktestResult
        metrics['net_profit'] = result.total_pnl
        metrics['total_trades'] = result.total_trades
        metrics['winning_trades'] = result.winning_trades
        metrics['losing_trades'] = result.losing_trades
        metrics['max_drawdown'] = result.max_drawdown
        
        # Calculate win rate
        if result.total_trades > 0:
            metrics['win_rate'] = (result.winning_trades / result.total_trades) * 100.0
        else:
            metrics['win_rate'] = 0.0
        
        # Calculate profit factor
        if result.trades:
            gross_profit = sum(t.pnl_after_costs for t in result.trades if t.pnl_after_costs > 0)
            gross_loss = abs(sum(t.pnl_after_costs for t in result.trades if t.pnl_after_costs < 0))
            if gross_loss > 0:
                metrics['profit_factor'] = gross_profit / gross_loss
            else:
                metrics['profit_factor'] = float('inf') if gross_profit > 0 else 0.0
        else:
            metrics['profit_factor'] = 0.0
        
        # Calculate average win/loss
        winning_trades = [t for t in result.trades if t.pnl_after_costs > 0]
        losing_trades = [t for t in result.trades if t.pnl_after_costs < 0]
        
        if winning_trades:
            metrics['avg_win'] = np.mean([t.pnl_after_costs for t in winning_trades])
        else:
            metrics['avg_win'] = 0.0
        
        if losing_trades:
            metrics['avg_loss'] = np.mean([t.pnl_after_costs for t in losing_trades])
        else:
            metrics['avg_loss'] = 0.0
        
        # Calculate expectancy
        if result.total_trades > 0:
            metrics['expectancy'] = result.total_pnl / result.total_trades
        else:
            metrics['expectancy'] = 0.0
        
        # Enhanced metrics
        enhanced = calculate_enhanced_metrics(result, risk_free_rate=risk_free_rate)
        metrics['sharpe_ratio'] = enhanced.get('sharpe_ratio', 0.0)
        metrics['sortino_ratio'] = enhanced.get('sortino_ratio', 0.0)
        metrics['recovery_factor'] = enhanced.get('recovery_factor', 0.0)
        metrics['cagr'] = enhanced.get('cagr', 0.0)
        metrics['calmar_ratio'] = enhanced.get('calmar_ratio', 0.0)
        
        return metrics
