"""Monte Carlo permutation tests for strategy validation.

NOTE: This module is maintained for backward compatibility.
New code should use validation.monte_carlo.runner.MonteCarloSuite
for the full professional-grade Monte Carlo validation suite.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
import numpy as np
import pandas as pd
try:
    from tqdm import tqdm
except ImportError:
    # tqdm is optional
    def tqdm(iterable, desc=None):
        return iterable

from engine.backtest_engine import Trade, BacktestResult
from metrics.metrics import calculate_trade_returns

# Import new suite for convenience
try:
    from validation.monte_carlo.runner import MonteCarloSuite
    from validation.monte_carlo.permutation import MonteCarloPermutation as NewMonteCarloPermutation
except ImportError:
    # Fallback if new module not available
    pass


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo permutation test."""
    observed_metric: float
    permuted_values: np.ndarray
    p_value: float
    percentile_rank: float
    observed_metrics: Dict
    n_iterations: int
    metric_name: str


class MonteCarloPermutation:
    """Monte Carlo permutation test engine."""
    
    def __init__(self, seed: int = 42):
        """
        Initialize Monte Carlo permutation tester.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)
        self.seed = seed
    
    def _equity_curve_from_pnls(self, pnls: np.ndarray, initial_capital: float) -> np.ndarray:
        """Calculate equity curve from trade P&Ls."""
        cumulative = np.cumsum(pnls)
        return initial_capital + cumulative
    
    def _calculate_metrics_from_equity(
        self,
        equity: np.ndarray,
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate metrics from equity curve."""
        if len(equity) < 2:
            return {
                'final_pnl': 0.0,
                'sharpe': 0.0,
                'pf': 0.0,
                'total_return_pct': 0.0,
            }
        
        final_pnl = equity[-1] - initial_capital
        total_return_pct = ((equity[-1] / initial_capital) - 1.0) * 100.0
        
        # Calculate returns
        returns = np.diff(equity) / np.maximum(equity[:-1], 1e-9)
        
        # Sharpe ratio
        if len(returns) > 1:
            mean_ret = np.mean(returns)
            std_ret = np.std(returns, ddof=1)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Profit factor (from equity changes)
        positive_changes = returns[returns > 0]
        negative_changes = returns[returns < 0]
        gross_profit = np.sum(positive_changes) * initial_capital if len(positive_changes) > 0 else 0.0
        gross_loss = abs(np.sum(negative_changes) * initial_capital) if len(negative_changes) > 0 else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
        
        return {
            'final_pnl': float(final_pnl),
            'sharpe': float(sharpe),
            'pf': float(pf),
            'total_return_pct': float(total_return_pct),
        }
    
    def _calculate_metrics_from_trades(
        self,
        trades: List[Trade],
        initial_capital: float
    ) -> Dict[str, float]:
        """Calculate metrics directly from trades."""
        if not trades:
            return {
                'final_pnl': 0.0,
                'sharpe': 0.0,
                'pf': 0.0,
                'total_return_pct': 0.0,
            }
        
        # Extract P&Ls (net_pnl is P&L after costs including commission and slippage)
        pnls = np.array([t.net_pnl for t in trades])
        
        # Final P&L
        final_pnl = np.sum(pnls)
        total_return_pct = (final_pnl / initial_capital) * 100.0
        
        # Sharpe from trade returns
        trade_returns = pnls / initial_capital
        if len(trade_returns) > 1:
            mean_ret = np.mean(trade_returns)
            std_ret = np.std(trade_returns, ddof=1)
            sharpe = (mean_ret / std_ret * np.sqrt(252)) if std_ret > 0 else 0.0
        else:
            sharpe = 0.0
        
        # Profit factor
        winning = pnls[pnls > 0]
        losing = pnls[pnls < 0]
        gross_profit = np.sum(winning) if len(winning) > 0 else 0.0
        gross_loss = abs(np.sum(losing)) if len(losing) > 0 else 0.0
        pf = (gross_profit / gross_loss) if gross_loss > 0 else (np.inf if gross_profit > 0 else 0.0)
        
        return {
            'final_pnl': float(final_pnl),
            'sharpe': float(sharpe),
            'pf': float(pf),
            'total_return_pct': float(total_return_pct),
        }
    
    def permute_trades(
        self,
        trades: List[Trade],
        initial_capital: float,
        metric: str = 'final_pnl',
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo permutation test on trade P&Ls.
        
        Args:
            trades: List of Trade objects
            initial_capital: Initial capital
            metric: Metric to test ('final_pnl', 'sharpe', 'pf', 'total_return_pct')
            n_iterations: Number of permutations
            show_progress: Show progress bar
        
        Returns:
            MonteCarloResult with p-value and distribution
        """
        if not trades:
            return MonteCarloResult(
                observed_metric=0.0,
                permuted_values=np.array([]),
                p_value=1.0,
                percentile_rank=0.0,
                observed_metrics={},
                n_iterations=n_iterations,
                metric_name=metric
            )
        
        # Extract P&Ls from trades (net_pnl is P&L after costs including commission and slippage)
        pnls = np.array([t.net_pnl for t in trades])
        
        # Calculate observed metrics
        observed_metrics = self._calculate_metrics_from_trades(trades, initial_capital)
        observed_value = observed_metrics[metric]
        
        # Run permutations
        permuted_values = np.empty(n_iterations)
        
        iterator = tqdm(range(n_iterations), desc=f"Permuting {metric}") if show_progress else range(n_iterations)
        
        for i in iterator:
            # Shuffle P&Ls
            permuted_pnls = self.rng.permutation(pnls)
            
            # Calculate equity curve
            equity = self._equity_curve_from_pnls(permuted_pnls, initial_capital)
            
            # Calculate metrics
            perm_metrics = self._calculate_metrics_from_equity(equity, initial_capital)
            permuted_values[i] = perm_metrics[metric]
        
        # Calculate p-value: fraction of permutations >= observed
        p_value = float((permuted_values >= observed_value).sum() / n_iterations)
        
        # Calculate percentile rank
        percentile_rank = float((permuted_values < observed_value).sum() / n_iterations * 100.0)
        
        return MonteCarloResult(
            observed_metric=observed_value,
            permuted_values=permuted_values,
            p_value=p_value,
            percentile_rank=percentile_rank,
            observed_metrics=observed_metrics,
            n_iterations=n_iterations,
            metric_name=metric
        )
    
    def permute_result(
        self,
        result: BacktestResult,
        metric: str = 'final_pnl',
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> MonteCarloResult:
        """
        Run Monte Carlo permutation test on BacktestResult.
        
        Args:
            result: BacktestResult object
            metric: Metric to test
            n_iterations: Number of permutations
            show_progress: Show progress bar
        
        Returns:
            MonteCarloResult
        """
        return self.permute_trades(
            trades=result.trades,
            initial_capital=result.initial_capital,
            metric=metric,
            n_iterations=n_iterations,
            show_progress=show_progress
        )
    
    def run_multiple_metrics(
        self,
        result: BacktestResult,
        metrics: List[str] = None,
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> Dict[str, MonteCarloResult]:
        """
        Run permutation tests on multiple metrics.
        
        Args:
            result: BacktestResult object
            metrics: List of metrics to test (default: ['final_pnl', 'sharpe', 'pf'])
            n_iterations: Number of permutations per metric
            show_progress: Show progress bar
        
        Returns:
            Dictionary mapping metric names to MonteCarloResult
        """
        if metrics is None:
            metrics = ['final_pnl', 'sharpe', 'pf']
        
        results = {}
        for metric in metrics:
            results[metric] = self.permute_result(
                result=result,
                metric=metric,
                n_iterations=n_iterations,
                show_progress=show_progress
            )
        
        return results

