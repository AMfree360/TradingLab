#!/usr/bin/env python3
"""Run Final OOS Test: One-time test on holdout period (e.g., 2025).

⚠️  CRITICAL: This test can only be run ONCE per strategy/data combination.
The holdout period is reserved for final validation after all development
and walk-forward validation has passed.

This implements the "Final OOS Test" from the recommended workflow:
- Lock away holdout period (e.g., 2025) during development
- Run walk-forward on development data only (2020-2024)
- After walk-forward passes, run this script ONCE on holdout period
- Record results permanently - no re-runs allowed
"""

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine
from config.schema import load_config, load_validation_criteria, validate_strategy_config
from config.config_loader import load_strategy_config_with_market
from adapters.data.data_loader import DataLoader
from config.market_loader import apply_market_profile
from reports.report_generator import ReportGenerator
from metrics.metrics import calculate_enhanced_metrics
from validation.state import ValidationStateManager
from repro.dataset_manifest import build_manifest, write_manifest_json, sha256_text, canonical_json_dumps
from experiments.registry import ExperimentRegistry, RunRecord


def main():
    parser = argparse.ArgumentParser(
        description='Final OOS Test: One-time test on holdout period',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
⚠️  WARNING: This test can only be run ONCE per strategy/data combination!

This script implements the final step of the recommended validation workflow:
1. Development Phase: Optimize on 2020-2024 (excluding holdout)
2. Walk-Forward Validation: Test on 2020-2024 with holdout excluded
3. Final OOS Test (THIS SCRIPT): Run once on holdout period (e.g., 2025)

Requirements:
- Phase 1 (Training Validation) must have PASSED
- Phase 2 (OOS Validation) must have PASSED
- Holdout period must not have been tested before

Examples:
  # Run final test on 2025 holdout period
  python3 scripts/run_final_oos_test.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020-2025.parquet \\
    --holdout-start 2025-01-01 \\
    --holdout-end 2025-12-31 \\
    --market BTCUSDT
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
        '--holdout-start',
        type=str,
        required=True,
        help='Start date for holdout period (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--holdout-end',
        type=str,
        required=True,
        help='End date for holdout period (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output report name (default: auto-generated)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force run even if already tested (NOT RECOMMENDED - breaks validation integrity)'
    )
    
    args = parser.parse_args()
    
    # Load strategy
    strategy_dir = Path(__file__).parent.parent / 'strategies' / args.strategy
    if not strategy_dir.exists():
        print(f"❌ Error: Strategy '{args.strategy}' not found in strategies/")
        sys.exit(1)
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = strategy_dir / 'config.yml'
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
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
            print(f"⚠️  Warning: {e}")
            print("Falling back to base config only")
            strategy_config_dict = load_config(config_path)
    else:
        # Load base config only
        strategy_config_dict = load_config(config_path)
        print("ℹ Using base config only (no market-specific config loaded)")
    
    # Load market profile if specified
    if args.market:
        try:
            strategy_config_dict = apply_market_profile(strategy_config_dict, symbol=args.market)
            print(f"✓ Applied market profile for: {args.market}")
        except Exception as e:
            print(f"⚠️  Warning: Could not load market profile: {e}")
    
    # Load strategy class
    strategy_module_path = strategy_dir / 'strategy.py'
    if not strategy_module_path.exists():
        print(f"❌ Error: Strategy file not found: {strategy_module_path}")
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
        print(f"❌ Error: Could not find strategy class in {strategy_module_path}")
        sys.exit(1)
    
    # Check validation state
    state_manager = ValidationStateManager()
    state = state_manager.load_state(args.strategy, args.data)
    
    if state is None:
        print("❌ Error: Phase 1 (Training Validation) must be completed first!")
        print("   Run Phase 1 validation before using holdout data.")
        sys.exit(1)
    
    if not state.phase1_passed:
        print("❌ Error: Phase 1 validation FAILED. Cannot proceed to final OOS test.")
        print(f"   Failure reasons: {', '.join(state.phase1_results.get('failure_reasons', []))}")
        sys.exit(1)
    
    if not state.phase2_passed:
        print("❌ Error: Phase 2 (OOS Validation) must have PASSED before final OOS test.")
        print("   Complete walk-forward validation on development data first.")
        sys.exit(1)
    
    # Check if holdout already tested
    holdout_key = f"{args.holdout_start}_{args.holdout_end}"
    if not args.force:
        if hasattr(state, 'holdout_tests') and state.holdout_tests is not None and holdout_key in state.holdout_tests:
            print("=" * 70)
            print("⚠️  WARNING: Holdout period already tested!")
            print("=" * 70)
            print(f"Holdout period {args.holdout_start} to {args.holdout_end} has already been tested.")
            print("This violates validation integrity - holdout should only be used ONCE.")
            print("\nPrevious test results:")
            prev_result = state.holdout_tests[holdout_key]
            print(f"  Date: {prev_result.get('test_date', 'Unknown')}")
            print(f"  PF: {prev_result.get('pf', 'N/A')}")
            print(f"  Passed: {prev_result.get('passed', 'N/A')}")
            print("\nTo re-run anyway (NOT RECOMMENDED), use --force flag.")
            sys.exit(1)
    
    # Load data
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"❌ Error: Data file not found: {data_path}")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("FINAL OOS TEST - HOLDOUT PERIOD")
    print(f"{'='*70}")
    print(f"Strategy: {args.strategy}")
    print(f"Holdout Period: {args.holdout_start} to {args.holdout_end}")
    print(f"{'='*70}\n")
    
    print(f"Loading data from {data_path}...")
    loader = DataLoader()
    df = loader.load(data_path)
    
    # Filter to holdout period
    holdout_start = pd.to_datetime(args.holdout_start).normalize()
    holdout_end = pd.to_datetime(args.holdout_end).normalize() + pd.Timedelta(days=1)
    
    print(f"Filtering holdout data: {holdout_start.date()} to {args.holdout_end}")
    holdout_data = df[(df.index >= holdout_start) & (df.index < holdout_end)].copy()
    
    if len(holdout_data) == 0:
        print("❌ Error: No holdout data remaining after filtering!")
        sys.exit(1)
    
    print(f"Holdout data: {len(holdout_data)} bars")
    print(f"Date range: {holdout_data.index[0].date()} to {holdout_data.index[-1].date()}\n")

    # Dataset lock + consume-on-attempt: reserve this holdout slice as soon as we have data.
    try:
        manifest = build_manifest(
            file_path=data_path,
            df=holdout_data,
            purpose='holdout',
            slice_start=args.holdout_start,
            slice_end=args.holdout_end,
        )

        try:
            manifest_path = write_manifest_json(manifest, Path(__file__).parent.parent / 'data' / 'manifests')
            print(f"🧾 Wrote dataset manifest: {manifest_path}")
        except Exception:
            pass

        state.verify_or_set_phase_lock(
            f"holdout:{holdout_key}",
            {
                'manifest_hash': manifest.manifest_hash(),
                'manifest': manifest.to_dict(),
            },
        )

        # Mark consumed even if backtest later fails.
        if state.holdout_tests is None:
            state.holdout_tests = {}
        if holdout_key not in state.holdout_tests:
            state.holdout_tests[holdout_key] = {
                'test_date': datetime.now().isoformat(),
                'holdout_start': args.holdout_start,
                'holdout_end': args.holdout_end,
                'status': 'attempted',
                'forced': args.force,
            }
        state_manager.save_state(state)
    except Exception as e:
        print(f"⚠️  Warning: Could not create/verify holdout dataset lock: {e}")
    
    # Validate strategy config
    strategy_config_obj = validate_strategy_config(strategy_config_dict)
    
    # Create strategy instance
    strategy = strategy_class(strategy_config_obj)
    
    # Create backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        initial_capital=args.capital,
        commission_rate=None,  # Use market profile
        slippage_ticks=None    # Use market profile
    )
    
    # Run backtest on holdout period
    print("Running backtest on holdout period...")
    print("-" * 70)
    
    # Need warm-up data before holdout period
    # Get data before holdout for indicator warm-up
    warmup_bars = 500  # Standard warm-up
    
    # Find the index position of holdout_start (or nearest after it if exact match not found)
    try:
        holdout_idx = df.index.get_loc(holdout_start)
        # Handle different return types from get_loc
        if isinstance(holdout_idx, slice):
            holdout_idx = holdout_idx.start if holdout_idx.start is not None else 0
        elif isinstance(holdout_idx, np.ndarray):
            holdout_idx = int(holdout_idx[0]) if len(holdout_idx) > 0 else 0
        else:
            holdout_idx = int(holdout_idx)
    except (KeyError, TypeError):
        # If exact timestamp not found, find the first index >= holdout_start
        mask = df.index >= holdout_start
        if mask.any():
            first_idx = df.index[mask][0]
            holdout_idx = int(df.index.get_loc(first_idx))
        else:
            # If no data >= holdout_start, use the last index
            holdout_idx = len(df) - 1
    
    warmup_start_idx = max(0, holdout_idx - warmup_bars)
    warmup_start = df.index[warmup_start_idx]
    
    # Get full data including warm-up
    full_data = df[df.index >= warmup_start].copy()
    
    result = engine.run(
        full_data,
        start_date=holdout_start,
        end_date=holdout_end
    )
    
    # Calculate enhanced metrics
    enhanced_metrics = calculate_enhanced_metrics(result)
    pf = enhanced_metrics.get('profit_factor', 0.0)
    sharpe = enhanced_metrics.get('sharpe_ratio', enhanced_metrics.get('sharpe', 0.0))
    
    # Load validation criteria
    criteria_config = None
    if args.criteria:
        criteria_config = load_validation_criteria(Path(args.criteria))
    
    # Check against criteria
    min_pf = 1.5
    if criteria_config and criteria_config.oos:
        min_pf = criteria_config.oos.wf_min_test_pf
    
    passed = pf >= min_pf
    
    # Print results
    print("\n" + "=" * 70)
    print("FINAL OOS TEST RESULTS")
    print("=" * 70)
    print(f"Holdout Period: {args.holdout_start} to {args.holdout_end}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Profit Factor: {pf:.2f}")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}")
    print(f"Total PnL: {result.total_pnl:.2f}")
    print(f"\nThreshold: PF >= {min_pf}")
    print(f"Result: {'✅ PASSED' if passed else '❌ FAILED'}")
    print("=" * 70)
    
    # Save results to state
    holdout_result = {
        'test_date': datetime.now().isoformat(),
        'holdout_start': args.holdout_start,
        'holdout_end': args.holdout_end,
        'total_trades': result.total_trades,
        'win_rate': result.win_rate,
        'pf': pf,
        'sharpe': sharpe,
        'max_drawdown': result.max_drawdown,
        'total_pnl': result.total_pnl,
        'passed': passed,
        'forced': args.force
    }
    
    state.holdout_tests[holdout_key] = holdout_result
    state_manager.save_state(state)
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    
    report_data = {
        'phase': 'final_oos',
        'strategy': args.strategy,
        'passed': passed,
        'holdout_period': {
            'start': args.holdout_start,
            'end': args.holdout_end
        },
        'metrics': {
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'profit_factor': pf,
            'sharpe_ratio': sharpe,
            'max_drawdown': result.max_drawdown,
            'total_pnl': result.total_pnl,
        },
        'enhanced_metrics': enhanced_metrics,
        'criteria': {
            'min_pf': min_pf,
            'passed': passed
        }
    }
    
    report_name = args.output or f"final_oos_{args.strategy}_{args.holdout_start}_{args.holdout_end}"
    # Use generate_backtest_report since we have a BacktestResult
    report_path = generator.generate_backtest_report(
        result=result,
        enhanced_metrics=enhanced_metrics,
        output_name=report_name,
        report_title="Final Out of Sample Report"
    )
    
    print(f"\n📊 Report saved to: {report_path}")

    # Record run
    try:
        reg = ExperimentRegistry()
        dataset_hash = None
        if state.holdout_dataset_locks and holdout_key in state.holdout_dataset_locks:
            dataset_hash = state.holdout_dataset_locks[holdout_key].get('manifest_hash')
        config_hash = sha256_text(canonical_json_dumps(strategy_config_dict))
        reg.record_run(
            RunRecord(
                strategy=args.strategy,
                phase='final_holdout',
                data_file=str(args.data),
                dataset_manifest_hash=dataset_hash,
                config_hash=config_hash,
                passed=bool(passed),
                report_path=str(report_path),
                metrics={
                    'profit_factor': pf,
                    'sharpe_ratio': sharpe,
                    'max_drawdown': result.max_drawdown,
                    'total_pnl': result.total_pnl,
                    'total_trades': result.total_trades,
                },
                outcome={
                    'passed': bool(passed),
                    'criteria': {
                        'min_pf': min_pf,
                        'passed': bool(passed),
                    },
                },
                params={
                    'holdout_start': args.holdout_start,
                    'holdout_end': args.holdout_end,
                    'force': bool(args.force),
                },
                command=' '.join(sys.argv),
            ),
            repo_dir=Path(__file__).parent.parent,
        )
    except Exception:
        pass
    
    if not passed:
        print("\n⚠️  Final OOS test FAILED. Strategy does not meet criteria on holdout period.")
        sys.exit(1)
    else:
        print("\n✅ Final OOS test PASSED. Strategy validated on holdout period.")
        print("⚠️  Remember: This holdout period can never be used again for validation.")


if __name__ == '__main__':
    main()
