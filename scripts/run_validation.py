#!/usr/bin/env python3
"""Script to run validation tests from command line."""

import argparse
import sys
from pathlib import Path
import pandas as pd
import yaml

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from validation import ValidationRunner
from config.schema import WalkForwardConfig, load_config
from config.config_loader import load_strategy_config_with_market
from config.market_loader import apply_market_profile
from reports.report_generator import ReportGenerator
from adapters.data.data_loader import DataLoader


def main():
    parser = argparse.ArgumentParser(description='Run validation tests on a strategy')
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
        '--wf-config',
        type=str,
        help='Path to walk-forward config file (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output report name (default: auto-generated)'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    parser.add_argument(
        '--mc-iterations',
        type=int,
        default=1000,
        help='Monte Carlo iterations (default: 1000)'
    )
    parser.add_argument(
        '--no-monte-carlo',
        action='store_true',
        help='Skip Monte Carlo tests'
    )
    parser.add_argument(
        '--no-walk-forward',
        action='store_true',
        help='Skip walk-forward analysis'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for validation (YYYY-MM-DD). Filters data to this date and later.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for validation (YYYY-MM-DD). Filters data up to this date.'
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
    
    # Use hierarchical config loader if market symbol provided (like backtest script)
    market_symbol = args.market or None
    if market_symbol:
        try:
            strategy_config_dict = load_strategy_config_with_market(
                base_config_path=config_path,
                market_symbol=market_symbol
            )
            print(f"✓ Loaded market-specific config for: {market_symbol}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to base config only")
            strategy_config_dict = load_config(config_path)
    else:
        # Load base config only
        strategy_config_dict = load_config(config_path)
        print("ℹ Using base config only (no market-specific config loaded)")
    
    # Apply market profile if market specified
    if args.market:
        try:
            strategy_config_dict = apply_market_profile(strategy_config_dict, symbol=args.market)
            print(f"✓ Applied market profile for: {args.market}")
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
    
    # Load data using universal data loader
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    loader = DataLoader()
    df = loader.load(data_path)
    
    # Filter by date range if specified
    original_len = len(df)
    original_start = df.index[0] if original_len > 0 else None
    original_end = df.index[-1] if original_len > 0 else None
    
    if args.start_date or args.end_date:
        # Parse dates and normalize to start/end of day
        if args.start_date:
            start_date = pd.to_datetime(args.start_date).normalize()  # Start of day (00:00:00)
            print(f"  Original data range: {original_start} to {original_end}")
            print(f"  Filtering to start from {start_date.date()}")
            df = df[df.index >= start_date]
        
        if args.end_date:
            # End date should include the entire day, so use start of next day for comparison
            end_date = pd.to_datetime(args.end_date).normalize() + pd.Timedelta(days=1)  # Start of next day
            print(f"  Filtering to end before {end_date.date()}")
            df = df[df.index < end_date]
        
        filtered_len = len(df)
        print(f"  Data filtered: {original_len} bars -> {filtered_len} bars")
        
        if filtered_len == 0:
            print("\n❌ Error: No data remaining after date filtering!")
            print(f"\n  Data file contains: {original_start.date()} to {original_end.date()}")
            if args.start_date:
                print(f"  Requested start date: {args.start_date}")
            if args.end_date:
                print(f"  Requested end date: {args.end_date}")
            print(f"\n  The requested date range ({args.start_date if args.start_date else 'start'} to {args.end_date if args.end_date else 'end'})")
            print(f"  does not overlap with the data in the file ({original_start.date()} to {original_end.date()}).")
            print(f"\n  Solution: Use --start-date and --end-date that fall within the data range,")
            print(f"  or use a different data file that contains the requested period.")
            sys.exit(1)
        
        if filtered_len > 0:
            print(f"  Filtered data range: {df.index[0]} to {df.index[-1]}")
    
    # Load walk-forward config if provided
    wf_config = None
    if not args.no_walk_forward:
        if args.wf_config:
            wf_config_dict = load_config(Path(args.wf_config))
            wf_config = WalkForwardConfig(**wf_config_dict)
        else:
            # Use defaults
            wf_config = WalkForwardConfig(
                start_date=str(df.index[0]),
                end_date=str(df.index[-1]),
                window_type="expanding",
                training_period={"duration": "1 year"},
                test_period={"duration": "6 months"},
                min_training_period="1 year"
            )
    
    # Create runner
    runner = ValidationRunner(
        strategy_class=strategy_class,
        initial_capital=args.capital
    )
    
    # Run validation
    print("Running validation tests...")
    print(f"  - Walk-forward: {'Yes' if wf_config else 'No'}")
    print(f"  - Monte Carlo: {'Yes' if not args.no_monte_carlo else 'No'}")
    
    results = runner.validate_strategy(
        data=df,
        strategy_config=strategy_config_dict,
        wf_config=wf_config,
        run_monte_carlo=not args.no_monte_carlo,
        monte_carlo_iterations=args.mc_iterations
    )
    
    # Print summary
    print("\n" + "="*60)
    print("VALIDATION RESULTS")
    print("="*60)
    
    if results.get('walk_forward'):
        wf = results['walk_forward']
        summary = wf.get('summary', {})
        print(f"\nWalk-Forward Analysis:")
        print(f"  Steps: {summary.get('total_steps', 0)}")
        print(f"  Mean Test PF: {summary.get('mean_test_pf', 0):.2f}")
        print(f"  Consistency Score: {summary.get('consistency_score', 0):.2f}")
    
    if results.get('monte_carlo'):
        print(f"\nMonte Carlo Permutation:")
        for metric, data in results['monte_carlo'].items():
            p_value = data.get('p_value', 1.0)
            status = "✓ PASS" if p_value < 0.05 else "✗ FAIL"
            print(f"  {metric}: p-value={p_value:.4f} {status}")
    
    print("="*60)
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    report_path = generator.generate_validation_report(
        validation_results=results,
        output_name=args.output
    )
    
    print(f"\nReport saved to: {report_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

