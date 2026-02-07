#!/usr/bin/env python3
"""Optimize strategy parameters using grid search on training data."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from validation.sensitivity import SensitivityAnalyzer
from config.schema import load_config, validate_strategy_config
from adapters.data.data_loader import DataLoader
from config.market_loader import apply_market_profile, create_market_override_from_cli


def main():
    parser = argparse.ArgumentParser(
        description='Optimize strategy parameters on training data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Optimize risk_per_trade_pct
  python3 scripts/optimize_strategy.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0 \\
    --start-date 2020-01-01 \\
    --end-date 2021-12-31

  # Optimize multiple parameters
  python3 scripts/optimize_strategy.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --param risk.risk_per_trade_pct 0.5 1.0 1.5 \\
    --param moving_averages.ema5.length 5 10 15 \\
    --metric sharpe_ratio
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name (must match folder in strategies/)'
    )
    parser.add_argument(
        '--data',
        type=str,
        required=True,
        help='Path to data file (CSV or Parquet)'
    )
    parser.add_argument(
        '--market',
        type=str,
        help='Market symbol (e.g., EURUSD, BTCUSDT). Applies market profile from config/market_profiles.yml'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to strategy config file (default: strategies/{strategy}/config.yml)'
    )
    parser.add_argument(
        '--param',
        type=str,
        nargs='+',
        action='append',
        required=True,
        metavar=('PARAM_PATH', 'VALUES...'),
        help='Parameter to optimize (can specify multiple times). Format: path.to.param value1 value2 value3'
    )
    parser.add_argument(
        '--metric',
        type=str,
        default='profit_factor',
        choices=['profit_factor', 'sharpe_ratio', 'total_pnl'],
        help='Metric to optimize (default: profit_factor)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for training data (YYYY-MM-DD). Filters data to this date and later.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for training data (YYYY-MM-DD). Filters data up to this date.'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file for best parameters (YAML format). If not specified, prints to stdout.'
    )
    
    args = parser.parse_args()
    
    # Load strategy
    strategy_dir = Path(__file__).parent.parent / 'strategies' / args.strategy
    if not strategy_dir.exists():
        print(f"Error: Strategy '{args.strategy}' not found in strategies/")
        sys.exit(1)
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = strategy_dir / 'config.yml'
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    strategy_config_dict = load_config(config_path)
    
    # Load market profile if specified
    if args.market:
        try:
            market_profile = apply_market_profile(strategy_config_dict, symbol=args.market)
            print(f"Applied market profile for: {args.market}")
        except Exception as e:
            print(f"Warning: Could not load market profile: {e}")
    
    # Load strategy class
    strategy_module_path = strategy_dir / 'strategy.py'
    if not strategy_module_path.exists():
        print(f"Error: Strategy file not found: {strategy_module_path}")
        sys.exit(1)
    
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        f"strategies.{args.strategy}.strategy",
        strategy_module_path
    )
    strategy_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(strategy_module)
    
    # Find strategy class
    strategy_class = None
    for name in dir(strategy_module):
        obj = getattr(strategy_module, name)
        if (isinstance(obj, type) and 
            issubclass(obj, StrategyBase) and 
            obj != StrategyBase):
            strategy_class = obj
            break
    
    if strategy_class is None:
        print(f"Error: Could not find strategy class in {strategy_module_path}")
        sys.exit(1)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    loader = DataLoader()
    df = loader.load(data_path)
    
    # Filter by date range if specified
    original_len = len(df)
    if args.start_date or args.end_date:
        if args.start_date:
            start_date = pd.to_datetime(args.start_date).normalize()
            print(f"  Filtering to start from {start_date.date()}")
            df = df[df.index >= start_date]
        
        if args.end_date:
            end_date = pd.to_datetime(args.end_date).normalize() + pd.Timedelta(days=1)
            print(f"  Filtering to end before {end_date.date()}")
            df = df[df.index < end_date]
        
        filtered_len = len(df)
        print(f"  Data filtered: {original_len} bars -> {filtered_len} bars")
        
        if filtered_len == 0:
            print("Error: No data remaining after date filtering!")
            sys.exit(1)
    
    print(f"Using {len(df)} bars for optimization")
    print(f"Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Parse parameter grid
    param_grid = {}
    for param_spec in args.param:
        if len(param_spec) < 2:
            print(f"Error: Invalid parameter specification: {param_spec}")
            print("Format: --param path.to.param value1 value2 value3")
            sys.exit(1)
        
        param_path = param_spec[0]
        try:
            # Try to convert values to appropriate types
            values = []
            for val_str in param_spec[1:]:
                # Try int first, then float, then keep as string
                try:
                    if '.' in val_str:
                        values.append(float(val_str))
                    else:
                        values.append(int(val_str))
                except ValueError:
                    values.append(val_str)
            
            param_grid[param_path] = values
            print(f"  Optimizing {param_path}: {values}")
        except Exception as e:
            print(f"Error parsing parameter {param_path}: {e}")
            sys.exit(1)
    
    # Run optimization
    print(f"\nRunning grid search optimization...")
    print(f"  Metric: {args.metric}")
    print(f"  Total combinations: {_count_combinations(param_grid)}")
    
    analyzer = SensitivityAnalyzer(
        strategy_class=strategy_class,
        initial_capital=args.capital,
        commission_rate=strategy_config_dict.get('backtest', {}).get('commissions', 0.0004),
        slippage_ticks=strategy_config_dict.get('backtest', {}).get('slippage_ticks', 0.0)
    )
    
    results_df = analyzer.grid_search(
        data=df,
        base_config=strategy_config_dict,
        param_grid=param_grid,
        metric=args.metric
    )
    
    if results_df.empty:
        print("Error: No results from optimization!")
        sys.exit(1)
    
    # Find best parameters
    best_idx = results_df[args.metric].idxmax()
    best_result = results_df.loc[best_idx]
    
    print("\n" + "="*60)
    print("OPTIMIZATION RESULTS")
    print("="*60)
    print(f"\nBest {args.metric}: {best_result[args.metric]:.4f}")
    print("\nBest Parameters:")
    for param_path in param_grid.keys():
        print(f"  {param_path}: {best_result[param_path]}")
    
    print("\nFull Metrics:")
    print(f"  Profit Factor: {best_result.get('profit_factor', 'N/A'):.4f}")
    print(f"  Sharpe Ratio: {best_result.get('sharpe_ratio', 'N/A'):.4f}")
    print(f"  Total P&L: ${best_result.get('total_pnl', 'N/A'):.2f}")
    print(f"  Win Rate: {best_result.get('win_rate', 'N/A'):.2f}%")
    print(f"  Total Trades: {int(best_result.get('total_trades', 0))}")
    print(f"  Max Drawdown: {best_result.get('max_drawdown_pct', 'N/A'):.2f}%")
    
    # Update config with best parameters
    best_config = strategy_config_dict.copy()
    for param_path in param_grid.keys():
        # Convert numpy/pandas types to native Python types for YAML serialization
        value = best_result[param_path]
        # Handle numpy/pandas types
        import numpy as np
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        elif isinstance(value, np.ndarray):
            value = value.tolist()
        elif hasattr(value, 'item'):  # numpy scalar (fallback)
            try:
                value = value.item()
            except (AttributeError, ValueError):
                pass
        _set_nested_param(best_config, param_path, value)
    
    # Output results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            yaml.dump(best_config, f, default_flow_style=False, sort_keys=False)
        print(f"\n✓ Best parameters saved to: {output_path}")
        print(f"\nTo use these parameters, update your config.yml or use:")
        print(f"  --config {output_path}")
    else:
        print("\n" + "="*60)
        print("BEST CONFIGURATION (YAML)")
        print("="*60)
        print(yaml.dump(best_config, default_flow_style=False, sort_keys=False))
    
    return 0


def _count_combinations(param_grid: dict) -> int:
    """Count total parameter combinations."""
    count = 1
    for values in param_grid.values():
        count *= len(values)
    return count


def _set_nested_param(config: dict, path: str, value):
    """Set nested parameter using dot notation path."""
    keys = path.split('.')
    current = config
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


if __name__ == '__main__':
    sys.exit(main())

