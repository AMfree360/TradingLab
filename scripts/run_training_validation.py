#!/usr/bin/env python3
"""Run Phase 1: Training Validation on training data only."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from validation.pipeline import ValidationPipeline
from config.schema import load_config, load_validation_criteria
from config.config_loader import load_strategy_config_with_market
from adapters.data.data_loader import DataLoader
from config.market_loader import apply_market_profile
from reports.report_generator import ReportGenerator
from experiments.registry import ExperimentRegistry, RunRecord
from repro.dataset_manifest import sha256_text, canonical_json_dumps


def main():
    parser = argparse.ArgumentParser(
        description='Phase 1: Training Validation (uses training data only)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 1 validates the strategy on training data only:
- Backtest quality check
- Monte Carlo Permutation (proves edge > random)
- Parameter Sensitivity (tests robustness)

This phase MUST pass before OOS data can be used.

Examples:
  # Basic training validation
  python3 scripts/run_training_validation.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --start-date 2020-01-01 \\
    --end-date 2021-12-31

  # With parameter sensitivity
  python3 scripts/run_training_validation.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --start-date 2020-01-01 \\
    --end-date 2021-12-31 \\
    --sensitivity-param risk.risk_per_trade_pct 0.5 1.0 1.5 2.0
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
        help='Market symbol (e.g., EURUSD, BTCUSDT). Applies market profile'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to strategy config file (default: strategies/{strategy}/config.yml)'
    )
    parser.add_argument(
        '--criteria',
        type=str,
        help='Path to validation criteria config (default: config/validation_criteria.yml)'
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
        required=True,
        help='Start date for training data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date for training data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--sensitivity-param',
        type=str,
        nargs='+',
        action='append',
        metavar=('PARAM_PATH', 'VALUES...'),
        help='Parameter for sensitivity analysis (can specify multiple times)'
    )
    parser.add_argument(
        '--no-sensitivity',
        action='store_true',
        help='Skip parameter sensitivity analysis'
    )
    parser.add_argument(
        '--mc-iterations',
        type=int,
        default=1000,
        help='Monte Carlo iterations (default: 1000)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output report name (default: auto-generated)'
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
    
    # Apply market profile automatically if symbol is in market profiles (like backtest script)
    symbol = args.market or strategy_config_dict.get('market', {}).get('symbol')
    if symbol:
        try:
            strategy_config_dict = apply_market_profile(strategy_config_dict, symbol=symbol)
            print(f"✓ Applied market profile for: {symbol}")
        except Exception as e:
            print(f"Warning: Could not load market profile: {e}")
    
    # Also apply if explicitly specified (takes precedence)
    if args.market and args.market != symbol:
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
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"Loading data from {data_path}...")
    loader = DataLoader()
    df = loader.load(data_path)
    
    # Filter to training period
    start_date = pd.to_datetime(args.start_date).normalize()
    end_date = pd.to_datetime(args.end_date).normalize() + pd.Timedelta(days=1)
    
    print(f"Filtering training data: {start_date.date()} to {args.end_date}")
    training_data = df[(df.index >= start_date) & (df.index < end_date)].copy()
    
    if len(training_data) == 0:
        print("Error: No training data remaining after filtering!")
        sys.exit(1)
    
    print(f"Training data: {len(training_data)} bars")
    print(f"Date range: {training_data.index[0].date()} to {training_data.index[-1].date()}")
    
    # Parse sensitivity parameters
    sensitivity_params = None
    if not args.no_sensitivity and args.sensitivity_param:
        sensitivity_params = {}
        for param_spec in args.sensitivity_param:
            if len(param_spec) < 2:
                print(f"Warning: Invalid sensitivity parameter: {param_spec}")
                continue
            
            param_path = param_spec[0]
            try:
                values = []
                for val_str in param_spec[1:]:
                    try:
                        if '.' in val_str:
                            values.append(float(val_str))
                        else:
                            values.append(int(val_str))
                    except ValueError:
                        values.append(val_str)
                sensitivity_params[param_path] = values
            except Exception as e:
                print(f"Warning: Could not parse sensitivity parameter {param_path}: {e}")
    
    # Load validation criteria
    criteria_config = None
    if args.criteria:
        from config.schema import load_validation_criteria
        criteria_config = load_validation_criteria(Path(args.criteria))
    
    # Create pipeline
    # Don't pass commission/slippage - let engine use market profile values
    # This ensures market profile settings (leverage, contract_size, etc.) are used correctly
    pipeline = ValidationPipeline(
        strategy_class=strategy_class,
        initial_capital=args.capital,
        commission_rate=None,  # Use market profile or strategy config
        slippage_ticks=None,   # Use market profile or strategy config
        criteria_config=criteria_config,
        manifest_dir=Path(__file__).parent.parent / 'data' / 'manifests'
    )
    
    # Get state for retry info
    from validation.state import ValidationStateManager
    state_manager = ValidationStateManager()
    state = state_manager.load_state(args.strategy, str(args.data))
    
    # Run Phase 1
    result = pipeline.run_phase1_training(
        strategy_name=args.strategy,
        data_file=str(args.data),
        training_data=training_data,
        strategy_config=strategy_config_dict,
        training_start=args.start_date,
        training_end=args.end_date,
        run_sensitivity=not args.no_sensitivity,
        sensitivity_params=sensitivity_params,
        mc_iterations=args.mc_iterations
    )
    
    # Reload state to get updated failure count
    state = state_manager.load_state(args.strategy, str(args.data))
    retry_info = state.get_phase1_retry_info() if state and state.phase1_failure_count > 0 else {}
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    
    # Create simplified report data
    # Extract backtest metrics for display
    backtest_metrics = None
    if result.backtest_result:
        from metrics.metrics import calculate_enhanced_metrics
        enhanced = calculate_enhanced_metrics(result.backtest_result)
        backtest_metrics = {
            'profit_factor': enhanced.get('profit_factor', 0.0),
            'sharpe_ratio': enhanced.get('sharpe_ratio', 0.0),
            'total_trades': result.backtest_result.total_trades
        }
    
    report_data = {
        'phase': 'training',
        'strategy': args.strategy,
        'passed': result.passed,
        'date_range': f"{training_data.index[0].date()} to {training_data.index[-1].date()}",
        'retry_info': retry_info,
        'backtest': backtest_metrics,
        'backtest_result': result.backtest_result,  # Full result for visualizations
        'monte_carlo': result.monte_carlo_results,
        'sensitivity': result.sensitivity_results,
        'criteria_checks': result.criteria_checks,
        'failure_reasons': result.failure_reasons
    }
    
    report_path = generator.generate_validation_report(
        validation_results={'phase1_training': report_data},
        output_name=args.output or f"training_validation_{args.strategy}"
    )
    
    print(f"\nReport saved to: {report_path}")

    # Record run
    try:
        reg = ExperimentRegistry()
        state = state_manager.load_state(args.strategy, str(args.data))
        dataset_hash = None
        if state and state.phase1_dataset_lock:
            dataset_hash = state.phase1_dataset_lock.get('manifest_hash')
        config_hash = sha256_text(canonical_json_dumps(strategy_config_dict))

        reg.record_run(
            RunRecord(
                strategy=args.strategy,
                phase='phase1_training',
                data_file=str(args.data),
                dataset_manifest_hash=dataset_hash,
                config_hash=config_hash,
                passed=bool(result.passed),
                report_path=str(report_path),
                metrics=backtest_metrics,
                outcome={
                    'passed': bool(result.passed),
                    'failure_reasons': list(result.failure_reasons or []),
                    'criteria_checks': dict(result.criteria_checks or {}),
                    'monte_carlo_combined': dict((result.monte_carlo_results or {}).get('combined', {}) or {}),
                },
                params={
                    'start_date': args.start_date,
                    'end_date': args.end_date,
                    'mc_iterations': args.mc_iterations,
                    'no_sensitivity': bool(args.no_sensitivity),
                },
                command=' '.join(sys.argv),
            ),
            repo_dir=Path(__file__).parent.parent,
        )
    except Exception:
        # Registry should never break validations.
        pass
    
    # Exit code based on result
    return 0 if result.passed else 1


if __name__ == '__main__':
    sys.exit(main())

