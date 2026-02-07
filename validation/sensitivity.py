"""Parameter sensitivity analysis for strategies."""

from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from itertools import product

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine, BacktestResult
from config.schema import StrategyConfig


class SensitivityAnalyzer:
    """Analyze parameter sensitivity for strategies."""
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0
    ):
        """
        Initialize sensitivity analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_ticks: Slippage in ticks
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
    
    def grid_search(
        self,
        data: pd.DataFrame,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]],
        metric: str = 'profit_factor'
    ) -> pd.DataFrame:
        """
        Perform grid search over parameter space.
        
        Args:
            data: Dataset to test on
            base_config: Base strategy configuration
            param_grid: Dictionary mapping parameter paths to value lists
                       e.g., {'risk.risk_per_trade_pct': [0.5, 1.0, 1.5]}
            metric: Metric to optimize ('profit_factor', 'sharpe_ratio', 'total_pnl')
        
        Returns:
            DataFrame with results for each parameter combination
        """
        import tempfile
        from pathlib import Path
        from config.schema import validate_strategy_config
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        combinations = list(product(*param_values))
        
        results = []
        
        for combo in combinations:
            # Create config with this combination
            test_config = base_config.copy()
            for param_name, param_value in zip(param_names, combo):
                # Handle nested parameter paths
                self._set_nested_param(test_config, param_name, param_value)
            
            # Validate config
            try:
                strategy_config = validate_strategy_config(test_config)
            except Exception as e:
                continue  # Skip invalid configs
            
            # Run backtest
            with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                data.to_csv(f.name)
                temp_path = Path(f.name)
            
            try:
                strategy = self.strategy_class(strategy_config)
                engine = BacktestEngine(
                    strategy=strategy,
                    initial_capital=self.initial_capital,
                    commission_rate=self.commission_rate,
                    slippage_ticks=self.slippage_ticks
                )
                result = engine.run(temp_path)
                
                # Extract metrics
                result_dict = {
                    'profit_factor': result.profit_factor,
                    'sharpe_ratio': result.sharpe_ratio,
                    'total_pnl': result.total_pnl,
                    'win_rate': result.win_rate,
                    'total_trades': result.total_trades,
                    'max_drawdown_pct': result.max_drawdown_pct,
                }
                
                # Add parameter values
                for param_name, param_value in zip(param_names, combo):
                    result_dict[param_name] = param_value
                
                results.append(result_dict)
            finally:
                temp_path.unlink()
        
        return pd.DataFrame(results)
    
    def _set_nested_param(self, config: Dict, path: str, value: Any):
        """Set nested parameter using dot notation path."""
        keys = path.split('.')
        current = config
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        current[keys[-1]] = value
    
    def analyze_sensitivity(
        self,
        results_df: pd.DataFrame,
        param_name: str,
        metric: str = 'profit_factor'
    ) -> Dict[str, float]:
        """
        Analyze sensitivity of a single parameter.
        
        Args:
            results_df: DataFrame from grid_search
            param_name: Parameter name to analyze
            metric: Metric to analyze
        
        Returns:
            Dictionary with sensitivity statistics
        """
        if param_name not in results_df.columns or metric not in results_df.columns:
            return {}
        
        grouped = results_df.groupby(param_name)[metric]
        
        return {
            'mean': float(grouped.mean().mean()),
            'std': float(grouped.mean().std()),
            'min': float(grouped.mean().min()),
            'max': float(grouped.mean().max()),
            'range': float(grouped.mean().max() - grouped.mean().min()),
            'coefficient_of_variation': float(
                grouped.mean().std() / grouped.mean().mean()
                if grouped.mean().mean() != 0 else 0.0
            ),
        }

