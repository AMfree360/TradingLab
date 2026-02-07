#!/usr/bin/env python3
"""Run Phase 2: Out-of-Sample Validation (OOS data used only once)."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from validation.pipeline import ValidationPipeline
from config.schema import load_config, load_validation_criteria, WalkForwardConfig
from config.config_loader import load_strategy_config_with_market
from adapters.data.data_loader import DataLoader
from config.market_loader import apply_market_profile
from reports.report_generator import ReportGenerator
from experiments.registry import ExperimentRegistry, RunRecord
from repro.dataset_manifest import sha256_text, canonical_json_dumps


def main():
    parser = argparse.ArgumentParser(
        description='Phase 2: Out-of-Sample Validation (OOS data used only once)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 2 validates the strategy on unseen out-of-sample data.
⚠️  WARNING: OOS data will be marked as used after this test!

Requirements:
- Phase 1 (Training Validation) must have PASSED
- OOS data must not have been used before

Examples:
  # Basic OOS validation
  python3 scripts/run_oos_validation.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --start-date 2022-01-01 \\
    --end-date 2023-12-31 \\
    --wf-training-period 1 year \\
    --wf-test-period 6 months
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
        help='Start date for OOS data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date for OOS data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--wf-training-period',
        type=str,
        default='1 year',
        help='Walk-forward training period (default: 1 year)'
    )
    parser.add_argument(
        '--wf-test-period',
        type=str,
        default='6 months',
        help='Walk-forward test period (default: 6 months)'
    )
    parser.add_argument(
        '--wf-window-type',
        type=str,
        choices=['expanding', 'rolling'],
        default='expanding',
        help='Walk-forward window type (default: expanding)'
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
    
    # Filter to OOS period
    start_date = pd.to_datetime(args.start_date).normalize()
    end_date = pd.to_datetime(args.end_date).normalize() + pd.Timedelta(days=1)
    
    print(f"Filtering OOS data: {start_date.date()} to {args.end_date}")
    oos_data = df[(df.index >= start_date) & (df.index < end_date)].copy()
    
    if len(oos_data) == 0:
        print("Error: No OOS data remaining after filtering!")
        sys.exit(1)
    
    print(f"OOS data: {len(oos_data)} bars")
    print(f"Date range: {oos_data.index[0].date()} to {oos_data.index[-1].date()}")
    
    # Create walk-forward config
    wf_config = WalkForwardConfig(
        start_date=str(oos_data.index[0]),
        end_date=str(oos_data.index[-1]),
        window_type=args.wf_window_type,
        training_period={'duration': args.wf_training_period},
        test_period={'duration': args.wf_test_period},
        min_training_period=args.wf_training_period
    )
    
    # Load validation criteria
    criteria_config = None
    if args.criteria:
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
    
    # Run Phase 2
    # Pass full dataset (df) for warm-up periods, not just filtered oos_data
    # The walk-forward analyzer needs historical data before OOS period for indicators
    try:
        result = pipeline.run_phase2_oos(
            strategy_name=args.strategy,
            data_file=str(args.data),
            oos_data=oos_data,  # Used for validation boundaries
            strategy_config=strategy_config_dict,
            wf_config=wf_config,
            oos_start=args.start_date,
            oos_end=args.end_date,
            full_data=df  # Pass full dataset for warm-up periods
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    
    # Create report data
    wf = result.walk_forward_result
    # Get criteria config for benchmarks (convert to dict if it's a Pydantic model)
    criteria_config_dict = None
    if criteria_config:
        if hasattr(criteria_config, 'dict'):
            criteria_config_dict = criteria_config.dict()
        elif hasattr(criteria_config, 'model_dump'):
            criteria_config_dict = criteria_config.model_dump()
        elif isinstance(criteria_config, dict):
            criteria_config_dict = criteria_config
    
    report_data = {
        'phase': 'oos',
        'strategy': args.strategy,
        'passed': result.passed,
        'walk_forward': {
            'summary': wf.summary,
            'steps': [
                {
                    'step': s.step_number,
                    'train_period': f"{s.train_start} to {s.train_end}",
                    'test_period': f"{s.test_start} to {s.test_end}",
                    'train_pf': s.train_metrics.get('pf', 0),
                    'test_pf': s.test_metrics.get('pf', 0),
                    'train_trades': s.train_result.total_trades,
                    'test_trades': s.test_result.total_trades,
                    'excluded_from_stats': s.excluded_from_stats,
                    'exclusion_reason': s.exclusion_reason,
                }
                for s in wf.steps
            ],
            'wf_result': wf  # Pass full WalkForwardResult for visualizations
        },
        'criteria_checks': result.criteria_checks,
        'failure_reasons': result.failure_reasons,
        'initial_capital': args.capital,
        'criteria_config': criteria_config_dict  # Pass criteria config for benchmarks
    }
    
    report_path = generator.generate_validation_report(
        validation_results={'phase2_oos': report_data},
        output_name=args.output or f"oos_validation_{args.strategy}"
    )
    
    print(f"\nReport saved to: {report_path}")

    # Record run
    try:
        from validation.state import ValidationStateManager
        state_manager = ValidationStateManager()
        state = state_manager.load_state(args.strategy, str(args.data))
        dataset_hash = None
        if state and state.phase2_dataset_lock:
            dataset_hash = state.phase2_dataset_lock.get('manifest_hash')

        reg = ExperimentRegistry()
        config_hash = sha256_text(canonical_json_dumps(strategy_config_dict))
        reg.record_run(
            RunRecord(
                strategy=args.strategy,
                phase='phase2_oos',
                data_file=str(args.data),
                dataset_manifest_hash=dataset_hash,
                config_hash=config_hash,
                passed=bool(result.passed),
                report_path=str(report_path),
                metrics={
                    'wf_steps': len(wf.steps),
                    'wf_test_pf_median': wf.summary.get('median_test_pf') if isinstance(wf.summary, dict) else None,
                },
                outcome={
                    'passed': bool(result.passed),
                    'failure_reasons': list(result.failure_reasons or []),
                    'criteria_checks': dict(result.criteria_checks or {}),
                },
                params={
                    'oos_start': args.start_date,
                    'oos_end': args.end_date,
                    'wf_training_period': args.wf_training_period,
                    'wf_test_period': args.wf_test_period,
                    'wf_window_type': args.wf_window_type,
                },
                command=' '.join(sys.argv),
            ),
            repo_dir=Path(__file__).parent.parent,
        )
    except Exception:
        pass
    
    # Exit code based on result
    return 0 if result.passed else 1


if __name__ == '__main__':
    sys.exit(main())

