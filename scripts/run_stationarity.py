#!/usr/bin/env python3
"""Run Phase 3: Stationarity Analysis (post-live, determines retrain frequency)."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from validation.pipeline import ValidationPipeline
from config.schema import load_config, load_validation_criteria
from adapters.data.data_loader import DataLoader
from config.market_loader import apply_market_profile
from reports.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Phase 3: Stationarity Analysis (determines retrain frequency)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Phase 3 analyzes strategy stationarity to determine optimal retraining frequency.
This should be run after the strategy has been live trading.

Requirements:
- Phase 1 (Training Validation) must have PASSED
- Phase 2 (OOS Validation) must have PASSED

Examples:
  # Analyze stationarity on live data
  python3 scripts/run_stationarity.py \\
    --strategy ma_alignment \\
    --data data/raw/BTCUSDT-15m-2020.parquet \\
    --start-date 2024-01-01 \\
    --end-date 2024-12-31 \\
    --training-period-days 365 \\
    --max-days 30
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
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        required=True,
        help='Start date for live data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        required=True,
        help='End date for live data (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--training-period-days',
        type=int,
        default=365,
        help='Training period in days before OOS testing (default: 365)'
    )
    parser.add_argument(
        '--max-days',
        type=int,
        default=30,
        help='Maximum days to test (default: 30)'
    )
    parser.add_argument(
        '--step-days',
        type=int,
        default=1,
        help='Step size for days (default: 1)'
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
    
    strategy_config_dict = load_config(config_path)
    
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
    
    # Filter to live period
    start_date = pd.to_datetime(args.start_date).normalize()
    end_date = pd.to_datetime(args.end_date).normalize() + pd.Timedelta(days=1)
    
    print(f"Filtering live data: {start_date.date()} to {args.end_date}")
    live_data = df[(df.index >= start_date) & (df.index < end_date)].copy()
    
    if len(live_data) == 0:
        print("Error: No live data remaining after filtering!")
        sys.exit(1)
    
    print(f"Live data: {len(live_data)} bars")
    print(f"Date range: {live_data.index[0].date()} to {live_data.index[-1].date()}")
    
    # Create pipeline
    # Don't pass commission/slippage - let engine use market profile values
    # This ensures market profile settings (leverage, contract_size, etc.) are used correctly
    pipeline = ValidationPipeline(
        strategy_class=strategy_class,
        initial_capital=args.capital,
        commission_rate=None,  # Use market profile or strategy config
        slippage_ticks=None,   # Use market profile or strategy config
        criteria_config=None
    )
    
    # Run Phase 3
    try:
        result = pipeline.run_phase3_stationarity(
            strategy_name=args.strategy,
            data_file=str(args.data),
            live_data=live_data,
            strategy_config=strategy_config_dict,
            max_days=args.max_days,
            step_days=args.step_days,
            training_period_days=args.training_period_days
        )
    except ValueError as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    
    # Serialize enhanced results
    report_data = {
        'phase': 'stationarity',
        'strategy': args.strategy,
        'recommended_retrain_days': result.recommended_retrain_days,
        'days_vs_pf': result.days_vs_pf,
        'days_vs_sharpe': result.days_vs_sharpe,
        'analysis_periods': len(result.analysis_periods),
        # Enhanced metrics
        'retrain_reason': getattr(result, 'retrain_reason', ''),
        'min_acceptable_pf': getattr(result, 'min_acceptable_pf', 1.5),
        'min_pf_observed': getattr(result, 'min_pf_observed', 0.0),
        'max_pf_observed': getattr(result, 'max_pf_observed', 0.0),
        'conditional_metrics_stable': getattr(result, 'conditional_metrics_stable', True),
        'enabled_regime_filters': getattr(result, 'enabled_regime_filters', []),
        # Conditional metrics
        'days_vs_conditional_win_rate': getattr(result, 'days_vs_conditional_win_rate', {}),
        'days_vs_signal_frequency': getattr(result, 'days_vs_signal_frequency', {}),
        'days_vs_payoff_ratio': getattr(result, 'days_vs_payoff_ratio', {}),
        'days_vs_big_win_frequency': getattr(result, 'days_vs_big_win_frequency', {}),
        'days_vs_consecutive_losses': getattr(result, 'days_vs_consecutive_losses', {}),
        # Regime performance
        'regime_performance': getattr(result, 'regime_performance', {}),
        'regime_stability': getattr(result, 'regime_stability', {})
    }
    
    report_path = generator.generate_validation_report(
        validation_results={'phase3_stationarity': report_data},
        output_name=args.output or f"stationarity_{args.strategy}"
    )
    
    print(f"\nReport saved to: {report_path}")
    print(f"\n✅ Recommended retraining frequency: {result.recommended_retrain_days} days")
    
    # Print enhanced summary
    if hasattr(result, 'retrain_reason') and result.retrain_reason:
        print(f"\n📊 Retraining Reason: {result.retrain_reason}")
    
    if hasattr(result, 'min_pf_observed') and hasattr(result, 'max_pf_observed'):
        print(f"📈 PF Range: {result.min_pf_observed:.2f} - {result.max_pf_observed:.2f}")
        min_acceptable = getattr(result, 'min_acceptable_pf', 1.5)
        if result.min_pf_observed >= min_acceptable:
            print(f"✅ All PF values above {min_acceptable} - Strategy remains profitable")
    
    if hasattr(result, 'enabled_regime_filters') and result.enabled_regime_filters:
        print(f"🔍 Enabled Regime Filters: {', '.join(result.enabled_regime_filters)}")
    
    if hasattr(result, 'conditional_metrics_stable'):
        status = "Stable ✓" if result.conditional_metrics_stable else "Unstable ✗"
        print(f"📊 Conditional Metrics: {status}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

