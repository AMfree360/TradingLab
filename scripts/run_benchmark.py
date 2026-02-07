#!/usr/bin/env python3
"""Run benchmark comparison against external platforms (MT5, NinjaTrader, TradingView).

This script compares Trading Lab's backtest engine against external platforms by:
1. Running the same strategy in Trading Lab
2. Importing trades from external platform export
3. Comparing trades trade-by-trade (engine equivalence)
4. Comparing metrics (metric parity)

Focus: Engine correctness and metric accuracy, not performance comparison.
"""

import argparse
import sys
from pathlib import Path
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from validation.benchmark import BenchmarkRunner
from reports.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(
        description='Benchmark Trading Lab against external platforms',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Benchmark Trading Lab's backtest engine against external platforms (MT5, NinjaTrader, TradingView).

This tool verifies engine equivalence (same trades) and metric parity (same metrics).

Requirements:
- Strategy must be implemented in Trading Lab
- External platform must export trades to CSV/Excel
- Same data source and date range for fair comparison

Examples:
  # Benchmark against MT5
  python3 scripts/run_benchmark.py \\
        --strategy ema_crossover \\
    --data data/raw/EURUSD_M15_2021_2025.csv \\
    --external-trades mt5_trades.csv \\
    --platform mt5 \\
    --start-date 2021-01-01 \\
    --end-date 2023-12-31

  # Benchmark against NinjaTrader
  python3 scripts/run_benchmark.py \\
        --strategy ema_crossover \\
    --data data/raw/ES_M15_2021_2025.csv \\
    --external-trades ninja_trades.csv \\
    --platform ninjatrader \\
    --start-date 2021-01-01 \\
    --end-date 2023-12-31
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
        help='Path to market data file (CSV or Parquet)'
    )
    parser.add_argument(
        '--external-trades',
        type=str,
        required=True,
        help='Path to external platform trade export file (CSV)'
    )
    parser.add_argument(
        '--platform',
        type=str,
        required=True,
        choices=['mt5', 'ninjatrader', 'tradingview'],
        help='External platform name'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD). If not provided, uses all data.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD). If not provided, uses all data.'
    )
    parser.add_argument(
        '--capital',
        type=float,
        default=10000.0,
        help='Initial capital (default: 10000.0)'
    )
    parser.add_argument(
        '--risk-free-rate',
        type=float,
        default=0.0,
        help='Risk-free rate for metrics (default: 0.0)'
    )
    parser.add_argument(
        '--time-tolerance',
        type=int,
        default=60,
        help='Time tolerance for trade matching in seconds (default: 60)'
    )
    parser.add_argument(
        '--price-tolerance-pct',
        type=float,
        default=0.01,
        help='Price tolerance for trade matching in percentage (default: 0.01)'
    )
    parser.add_argument(
        '--metric-tolerance-pct',
        type=float,
        default=5.0,
        help='Metric comparison tolerance in percentage (default: 5.0)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output directory for benchmark report (default: reports/)'
    )
    parser.add_argument(
        '--json',
        action='store_true',
        help='Output results as JSON instead of HTML report'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BENCHMARK: Trading Lab vs External Platform")
    print("=" * 60)
    print(f"Strategy: {args.strategy}")
    print(f"Platform: {args.platform.upper()}")
    print(f"Data: {args.data}")
    print(f"External Trades: {args.external_trades}")
    if args.start_date:
        print(f"Start Date: {args.start_date}")
    if args.end_date:
        print(f"End Date: {args.end_date}")
    print()
    
    # Initialize benchmark runner
    runner = BenchmarkRunner(
        time_tolerance_seconds=args.time_tolerance,
        price_tolerance_pct=args.price_tolerance_pct,
        metric_tolerance_pct=args.metric_tolerance_pct
    )
    
    # Run benchmark
    print("Running benchmark...")
    try:
        result = runner.run_benchmark(
            strategy_name=args.strategy,
            data_file=args.data,
            external_trades_file=args.external_trades,
            platform=args.platform,
            start_date=args.start_date,
            end_date=args.end_date,
            initial_capital=args.capital,
            risk_free_rate=args.risk_free_rate
        )
    except Exception as e:
        print(f"❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Print summary
    summary = result.get_summary()
    print()
    print("=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Platform: {summary['platform'].upper()}")
    print(f"Strategy: {summary['strategy']}")
    print()
    print("Trade Comparison:")
    print(f"  Trading Lab Trades: {summary['trading_lab_trades']}")
    print(f"  External Trades: {summary['external_trades']}")
    print(f"  Matched Trades: {summary['matched_trades']}")
    print(f"  Match Rate: {summary['trade_match_rate']:.2%}")
    print()
    print("Metric Comparison:")
    print(f"  Metrics Match: {'✓' if summary['metric_match'] else '✗'}")
    print()
    print("Overall Result:")
    if result.is_equivalent:
        print("  ✅ EQUIVALENT - Trading Lab matches external platform")
    else:
        print("  ❌ NOT EQUIVALENT - Differences detected")
        print()
        print("Trade Differences:")
        if result.comparison_result.trade_comparison.unmatched_trading_lab:
            print(f"  Unmatched Trading Lab trades: {len(result.comparison_result.trade_comparison.unmatched_trading_lab)}")
        if result.comparison_result.trade_comparison.unmatched_external:
            print(f"  Unmatched External trades: {len(result.comparison_result.trade_comparison.unmatched_external)}")
        
        print()
        print("Metric Differences:")
        for metric_name, metric_data in result.comparison_result.metric_comparison.metrics.items():
            if not result.comparison_result.metric_comparison.tolerance_checks.get(metric_name, False):
                diff_pct = metric_data['diff_pct']
                print(f"  {metric_name}: {diff_pct:.2f}% difference")
                print(f"    Trading Lab: {metric_data['trading_lab']:.4f}")
                print(f"    External: {metric_data['external']:.4f}")
    
    # Output results
    if args.json:
        # Output as JSON
        output_path = Path(args.output) if args.output else Path('reports')
        output_path.mkdir(parents=True, exist_ok=True)
        
        json_file = output_path / f"benchmark_{args.strategy}_{args.platform}.json"
        
        # Serialize results
        json_data = {
            'summary': summary,
            'trade_comparison': {
                'total_trading_lab_trades': result.comparison_result.trade_comparison.total_trading_lab_trades,
                'total_external_trades': result.comparison_result.trade_comparison.total_external_trades,
                'matched_trades': result.comparison_result.trade_comparison.matched_trades,
                'match_rate': result.comparison_result.trade_comparison.match_rate,
                'avg_match_score': result.comparison_result.trade_comparison.avg_match_score,
                'unmatched_trading_lab_count': len(result.comparison_result.trade_comparison.unmatched_trading_lab),
                'unmatched_external_count': len(result.comparison_result.trade_comparison.unmatched_external),
            },
            'metric_comparison': {
                'metrics': result.comparison_result.metric_comparison.metrics,
                'tolerance_checks': result.comparison_result.metric_comparison.tolerance_checks,
                'overall_match': result.comparison_result.metric_comparison.overall_match,
            },
            'is_equivalent': result.is_equivalent,
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print()
        print(f"Results saved to: {json_file}")
    else:
        # Generate HTML report (if report generator supports benchmarks)
        output_path = Path(args.output) if args.output else Path('reports')
        output_path.mkdir(parents=True, exist_ok=True)
        
        # For now, just save JSON (HTML report can be added later)
        json_file = output_path / f"benchmark_{args.strategy}_{args.platform}.json"
        
        json_data = {
            'summary': summary,
            'trade_comparison': {
                'total_trading_lab_trades': result.comparison_result.trade_comparison.total_trading_lab_trades,
                'total_external_trades': result.comparison_result.trade_comparison.total_external_trades,
                'matched_trades': result.comparison_result.trade_comparison.matched_trades,
                'match_rate': result.comparison_result.trade_comparison.match_rate,
                'avg_match_score': result.comparison_result.trade_comparison.avg_match_score,
            },
            'metric_comparison': {
                'metrics': result.comparison_result.metric_comparison.metrics,
                'tolerance_checks': result.comparison_result.metric_comparison.tolerance_checks,
                'overall_match': result.comparison_result.metric_comparison.overall_match,
            },
            'is_equivalent': result.is_equivalent,
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2, default=str)
        
        print()
        print(f"Results saved to: {json_file}")
        print("Note: HTML report generation for benchmarks coming soon.")


if __name__ == '__main__':
    sys.exit(main())
