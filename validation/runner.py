"""Validation runner that orchestrates walk-forward and Monte Carlo tests."""

from typing import Dict, Optional
import numpy as np
import pandas as pd
from pathlib import Path

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestResult
from validation.walkforward import WalkForwardAnalyzer, WalkForwardResult
from validation.monte_carlo import MonteCarloPermutation
from config.schema import WalkForwardConfig
from metrics.metrics import calculate_enhanced_metrics


class ValidationRunner:
    """Orchestrates validation tests for strategies."""
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0
    ):
        """
        Initialize validation runner.
        
        Args:
            strategy_class: Strategy class to validate
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_ticks: Slippage in ticks
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
    
    def run_walk_forward(
        self,
        data: pd.DataFrame,
        strategy_config: Dict,
        wf_config: WalkForwardConfig
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.
        
        Args:
            data: Full dataset with datetime index
            strategy_config: Strategy configuration dict
            wf_config: WalkForwardConfig
        
        Returns:
            WalkForwardResult
        """
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            config=wf_config,
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_ticks=self.slippage_ticks
        )
        
        return analyzer.run(data, strategy_config)
    
    def run_monte_carlo(
        self,
        result: BacktestResult,
        metrics: Optional[list] = None,
        n_iterations: int = 1000,
        seed: int = 42
    ) -> Dict[str, any]:
        """
        Run Monte Carlo permutation tests.
        
        Args:
            result: BacktestResult to test
            metrics: List of metrics to test (default: ['final_pnl', 'sharpe', 'pf'])
            n_iterations: Number of permutations
            seed: Random seed
        
        Returns:
            Dictionary of metric results
        """
        mc = MonteCarloPermutation(seed=seed)

        # Default whitelist of metrics to test (safe, order-sensitive metrics)
        if metrics is None:
            metrics = ['final_pnl', 'sharpe', 'profit_factor']

        # Legacy compatibility: older MonteCarloPermutation exposes
        # `run_multiple_metrics(result, metrics, n_iterations, show_progress)`
        # while the newer implementation exposes `run(backtest_result, metrics, n_iterations)`
        if hasattr(mc, 'run_multiple_metrics'):
            return mc.run_multiple_metrics(
                result=result,
                metrics=metrics,
                n_iterations=n_iterations,
                show_progress=True
            )

        # New API: call `run` and adapt the PermutationResult to the
        # legacy mapping expected by callers (objects with
        # `observed_metric`, `p_value`, `percentile_rank`).
        perm_result = mc.run(
            backtest_result=result,
            metrics=metrics,
            n_iterations=n_iterations,
            show_progress=True
        )

        # Build a compatibility dict mapping metric -> simple result object
        from types import SimpleNamespace
        out = {}
        metrics_list = metrics if metrics is not None else list(perm_result.observed_metrics.keys())
        def _extract_numeric(x):
            # Try to coerce various structures to a float
            if x is None:
                return 0.0
            if isinstance(x, (int, float)):
                return float(x)
            if isinstance(x, dict):
                # Look for common numeric keys
                for k in ('value', 'val', 'score', 'final_pnl', 'pnl'):
                    if k in x and isinstance(x[k], (int, float)):
                        return float(x[k])
                # Fallback: pick the first numeric value
                for v in x.values():
                    if isinstance(v, (int, float)):
                        return float(v)
                return 0.0
            try:
                return float(x)
            except Exception:
                return 0.0

        for m in metrics_list:
            observed = _extract_numeric(perm_result.observed_metrics.get(m, 0.0))
            p = float(perm_result.p_values.get(m, 1.0)) if hasattr(perm_result, 'p_values') else 1.0
            perc = float(perm_result.percentiles.get(m, 0.0)) if hasattr(perm_result, 'percentiles') else 0.0
            ns = SimpleNamespace(
                observed_metric=observed,
                p_value=p,
                percentile_rank=perc,
                permuted_values=getattr(perm_result, 'permuted_distributions', {}).get(m, np.array([])),
                n_iterations=getattr(perm_result, 'n_iterations', n_iterations)
            )
            out[m] = ns

        return out
    
    def validate_strategy(
        self,
        data: pd.DataFrame,
        strategy_config: Dict,
        wf_config: Optional[WalkForwardConfig] = None,
        run_monte_carlo: bool = True,
        monte_carlo_iterations: int = 1000
    ) -> Dict:
        """
        Run complete validation suite.
        
        Args:
            data: Full dataset
            strategy_config: Strategy configuration
            wf_config: Optional walk-forward config (if None, skips WFA)
            run_monte_carlo: Whether to run Monte Carlo on full backtest
            monte_carlo_iterations: Number of MC iterations
        
        Returns:
            Dictionary with all validation results
        """
        results = {
            'walk_forward': None,
            'monte_carlo': None,
            'enhanced_metrics': None,
        }
        
        # Run walk-forward if config provided
        if wf_config:
            wf_result = self.run_walk_forward(data, strategy_config, wf_config)
            results['walk_forward'] = {
                'summary': wf_result.summary,
                'steps': [
                    {
                        'step': s.step_number,
                        'train_period': f"{s.train_start} to {s.train_end}",
                        'test_period': f"{s.test_start} to {s.test_end}",
                        'train_pf': s.train_metrics['pf'],
                        'test_pf': s.test_metrics['pf'],
                        'train_sharpe': s.train_metrics['sharpe'],
                        'test_sharpe': s.test_metrics['sharpe'],
                    }
                    for s in wf_result.steps
                ]
            }
        
        # Run Monte Carlo on full backtest if requested
        if run_monte_carlo:
            from engine.backtest_engine import BacktestEngine
            
            # Run full backtest
            from config.schema import validate_strategy_config
            strategy_config_obj = validate_strategy_config(strategy_config)
            strategy = self.strategy_class(strategy_config_obj)
            
            engine = BacktestEngine(
                strategy=strategy,
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage_ticks=self.slippage_ticks
            )
            backtest_result = engine.run(data)
            
            # Calculate enhanced metrics
            enhanced = calculate_enhanced_metrics(backtest_result)
            results['enhanced_metrics'] = enhanced
            
            # Run Monte Carlo
            mc_results = self.run_monte_carlo(
                result=backtest_result,
                n_iterations=monte_carlo_iterations
            )
            
            results['monte_carlo'] = {
                metric: {
                    'observed': r.observed_metric,
                    'p_value': r.p_value,
                    'percentile_rank': r.percentile_rank,
                }
                for metric, r in mc_results.items()
            }
        
        return results

