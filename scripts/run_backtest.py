#!/usr/bin/env python3
"""Script to run backtests from command line."""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine
from engine.market import MarketSpec
from config.schema import load_and_validate_strategy_config, load_config, validate_strategy_config
from config.market_loader import apply_market_profile, create_market_override_from_cli
from config.config_loader import load_strategy_config_with_market
from metrics.metrics import calculate_enhanced_metrics
from reports.report_generator import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description='Run a backtest on a strategy')
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name (must match folder in strategies/)'
    )
    parser.add_argument(
        '--data',
        type=str,
        help='Path to data file (CSV or Parquet). If not provided, will download test data.'
    )
    parser.add_argument(
        '--download-data',
        action='store_true',
        help='Download test data automatically if data file not found'
    )
    parser.add_argument(
        '--download-symbol',
        type=str,
        default='BTCUSDT',
        help='Symbol to download (default: BTCUSDT)'
    )
    parser.add_argument(
        '--download-interval',
        type=str,
        default='15m',
        help='Timeframe to download: 1m, 5m, 15m, 30m, 1h, 4h, 1d (default: 15m)'
    )
    parser.add_argument(
        '--download-start',
        type=str,
        help='Start date for download (YYYY-MM-DD). Default: 1 year ago'
    )
    parser.add_argument(
        '--download-end',
        type=str,
        help='End date for download (YYYY-MM-DD). Default: today'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to strategy config file (default: strategies/{strategy}/config.yml)'
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
        '--commission',
        type=float,
        default=0.0004,
        help='Commission rate (default: 0.0004)'
    )
    parser.add_argument(
        '--slippage',
        type=float,
        default=0.0,
        help='Slippage in ticks (default: 0.0)'
    )
    parser.add_argument(
        '--market',
        type=str,
        help='Override market symbol (e.g., EURUSD, BTCUSDT). Loads market profile automatically.'
    )
    parser.add_argument(
        '--exchange',
        type=str,
        help='Override exchange name'
    )
    parser.add_argument(
        '--market-type',
        type=str,
        choices=['spot', 'futures'],
        help='Override market type (spot or futures)'
    )
    parser.add_argument(
        '--leverage',
        type=float,
        help='Override leverage (default: 1.0)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date for backtest (YYYY-MM-DD). Filters data to this date and later. Useful for reserving later data for OOS testing.'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date for backtest (YYYY-MM-DD). Filters data up to this date. Useful for reserving later data for OOS testing.'
    )
    parser.add_argument(
        '--config-profile',
        type=str,
        help='Config profile name to load (e.g., optimized_eurusd_2024). Loads from strategies/{strategy}/configs/profiles/{profile}.yml'
    )
    parser.add_argument(
        '--auto-resample',
        action='store_true',
        help='Automatically resample loaded data to the requested timeframe without prompting (non-interactive)'
    )
    
    args = parser.parse_args()
    
    # Load strategy
    strategy_dir = Path(__file__).parent.parent / 'strategies' / args.strategy
    if not strategy_dir.exists():
        print(f"Error: Strategy '{args.strategy}' not found in strategies/")
        sys.exit(1)
    
    # Load config with market-specific overrides
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = strategy_dir / 'config.yml'
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    # Use new hierarchical config loader if market symbol or profile provided
    market_symbol = args.market or None
    if market_symbol or args.config_profile:
        try:
            config_dict = load_strategy_config_with_market(
                base_config_path=config_path,
                market_symbol=market_symbol,
                config_profile=args.config_profile
            )
            if market_symbol:
                print(f"✓ Loaded market-specific config for: {market_symbol}")
                # Debug: Show key config values to verify market config was loaded
                if 'timeframes' in config_dict:
                    print(f"  Timeframes: signal={config_dict['timeframes'].get('signal_tf')}, entry={config_dict['timeframes'].get('entry_tf')}")
                if 'stop_loss' in config_dict:
                    sl = config_dict['stop_loss']
                    print(f"  Stop Loss: type={sl.get('type')}, length={sl.get('length')}, buffer={sl.get('buffer_pips')} {sl.get('buffer_unit', 'units')}")
            if args.config_profile:
                print(f"✓ Loaded config profile: {args.config_profile}")
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            print("Falling back to base config only")
            config_dict = load_config(config_path)
    else:
        # Load base config only (backward compatibility)
        config_dict = load_config(config_path)
        print("ℹ Using base config only (no market-specific config loaded)")
    
    # Apply market profile if market override provided
    if args.market or args.exchange or args.market_type or args.leverage is not None:
        market_override = create_market_override_from_cli(
            symbol=args.market,
            exchange=args.exchange,
            market_type=args.market_type,
            leverage=args.leverage,
            commission=args.commission if args.commission != 0.0004 else None,
            slippage=args.slippage if args.slippage != 0.0 else None
        )
        config_dict = apply_market_profile(
            config_dict,
            symbol=args.market or config_dict.get('market', {}).get('symbol'),
            market_override=market_override if market_override else None
        )
        print(f"Applied market profile for: {config_dict.get('market', {}).get('symbol', 'unknown')}")
    
    # Also apply market profile automatically if symbol is in market profiles
    # This allows seamless switching between markets
    symbol = args.market or config_dict.get('market', {}).get('symbol')
    if symbol:
        config_dict = apply_market_profile(config_dict, symbol=symbol)
    
    # Override with CLI args (CLI takes precedence)
    if args.commission != 0.0004:
        if 'backtest' not in config_dict:
            config_dict['backtest'] = {}
        config_dict['backtest']['commissions'] = args.commission
    if args.slippage != 0.0:
        if 'backtest' not in config_dict:
            config_dict['backtest'] = {}
        config_dict['backtest']['slippage_ticks'] = args.slippage
    
    # Validate and create config object from updated dict
    strategy_config = validate_strategy_config(config_dict)
    
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
    
    # Find strategy class (assume it's the only class that inherits from StrategyBase)
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
    
    # Create strategy instance
    strategy = strategy_class(strategy_config)
    
    # Create MarketSpec if market override provided
    market_spec = None
    if args.market or args.leverage is not None:
        # Try to load from market profiles first
        symbol = args.market or getattr(strategy.config.market, 'symbol', 'EURUSD')
        try:
            market_spec = MarketSpec.load_from_profiles(symbol)
            # Override leverage if provided via CLI
            if args.leverage is not None:
                market_spec.leverage = args.leverage
        except ValueError:
            # Fallback: create from CLI args
            market_spec = MarketSpec(
                symbol=symbol,
                exchange=args.exchange or 'unknown',
                asset_class='forex' if 'USD' in symbol and len(symbol) == 6 else 'crypto',
                market_type=args.market_type or 'spot',
                leverage=args.leverage or 1.0,
                commission_rate=args.commission,
                slippage_ticks=args.slippage
            )
    
    # Read enforce_margin_checks from config (for benchmarking to match NT default)
    exec_cfg = getattr(strategy_config, 'execution', None)
    if exec_cfg:
        enforce_margin = exec_cfg.enforce_margin_checks
    else:
        enforce_margin = True  # Default to True if no execution config
    
    # Get commission and slippage from config if not overridden by CLI
    commission_rate = args.commission if args.commission != 0.0004 else None
    slippage_ticks = args.slippage if args.slippage != 0.0 else None
    
    # If not provided via CLI, try to get from config.backtest
    if commission_rate is None:
        backtest_cfg = getattr(strategy_config, 'backtest', None)
        if backtest_cfg:
            # The config uses 'commissions' which we interpret as commission_rate
            # For futures, the engine will also set commission_per_contract=0 if commission_rate=0
            config_commissions = getattr(backtest_cfg, 'commissions', None)
            if config_commissions is not None:
                commission_rate = config_commissions
    
    if slippage_ticks is None:
        backtest_cfg = getattr(strategy_config, 'backtest', None)
        if backtest_cfg:
            slippage_ticks = getattr(backtest_cfg, 'slippage_ticks', None)
    
    # Create engine (MarketSpec will be auto-loaded if not provided)
    engine = BacktestEngine(
        strategy=strategy,
        market_spec=market_spec,
        initial_capital=args.capital,
        commission_rate=commission_rate,  # Override market profile if provided
        slippage_ticks=slippage_ticks,  # Override market profile if provided
        enforce_margin_checks=enforce_margin
    )
    
    # Handle data file
    if args.data:
        data_path = Path(args.data)
        if not data_path.exists():
            if args.download_data:
                print(f"Data file not found: {data_path}")
                print("Downloading test data...")
                from scripts.download_data import download_binance_data
                from datetime import datetime, timedelta
                
                # Set default dates if not provided
                if args.download_end:
                    end_date = args.download_end
                else:
                    end_date = datetime.now().strftime('%Y-%m-%d')
                
                if args.download_start:
                    start_date = args.download_start
                else:
                    # Default: 1 year ago
                    start_dt = datetime.now() - timedelta(days=365)
                    start_date = start_dt.strftime('%Y-%m-%d')
                
                # Download data
                download_binance_data(
                    symbol=args.download_symbol,
                    interval=args.download_interval,
                    start_date=start_date,
                    end_date=end_date,
                    output_path=data_path
                )
            else:
                print(f"Error: Data file not found: {data_path}")
                print("Tip: Use --download-data to automatically download test data")
                sys.exit(1)
    else:
        # No data provided, download test data
        print("No data file provided. Downloading test data...")
        from scripts.download_data import download_binance_data
        from datetime import datetime, timedelta
        
        # Set default dates
        end_date = args.download_end or datetime.now().strftime('%Y-%m-%d')
        if args.download_start:
            start_date = args.download_start
        else:
            # Default: 1 year ago
            start_dt = datetime.now() - timedelta(days=365)
            start_date = start_dt.strftime('%Y-%m-%d')
        
        # Generate output path
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{args.download_symbol}-{args.download_interval}-{start_date}.parquet"
        data_path = data_dir / filename
        
        # Download data
        download_binance_data(
            symbol=args.download_symbol,
            interval=args.download_interval,
            start_date=start_date,
            end_date=end_date,
            output_path=data_path
        )
    
    # Load and filter data if date range specified
    from adapters.data.data_loader import DataLoader
    
    print(f"\nLoading data from {data_path}...")
    loader = DataLoader()
    df = loader.load(data_path)

    # If the loaded data timeframe doesn't match required entry timeframe,
    # ask the user whether to resample the data to the requested timeframe.
    def _infer_minutes_from_index(idx):
        try:
            # Try pandas infer_freq first
            freq = pd.infer_freq(idx)
            if freq:
                # Normalize e.g. '15T' -> minutes
                if freq.endswith('T') or freq.endswith('min'):
                    # strip non-digits
                    num = ''.join([c for c in freq if c.isdigit()])
                    return int(num)
                if freq.lower().endswith('h'):
                    num = ''.join([c for c in freq if c.isdigit()])
                    return int(num) * 60
                if freq.lower().endswith('d'):
                    num = ''.join([c for c in freq if c.isdigit()])
                    return int(num) * 1440
        except Exception:
            pass
        # Fallback: take median delta in minutes
        try:
            diffs = idx.to_series().diff().dropna().dt.total_seconds() / 60.0
            med = int(diffs.median())
            return med
        except Exception:
            return None

    def _normalize_tf(tf_str: str) -> str:
        s = str(tf_str).strip().lower()
        s = s.replace('min', 'm')
        return s

    # Determine loaded and requested timeframes
    loaded_mins = _infer_minutes_from_index(df.index)
    requested_tf = getattr(strategy_config, 'timeframes', None)
    requested_entry_tf = None
    if requested_tf is not None:
        requested_entry_tf = getattr(requested_tf, 'entry_tf', None)

    if loaded_mins is not None and requested_entry_tf:
        # Map minutes to normalized tf string for comparison
        def _mins_to_tf(m):
            if m % 1440 == 0:
                days = m // 1440
                return f"{days}d" if days > 1 else '1d'
            if m % 60 == 0:
                hours = m // 60
                return f"{hours}h" if hours > 1 else '1h'
            return f"{int(m)}m"

        loaded_tf = _mins_to_tf(loaded_mins)
        if _normalize_tf(loaded_tf) != _normalize_tf(requested_entry_tf):
            print(f"\n⚠ Data timeframe mismatch: loaded data appears to be '{loaded_tf}',"
                  f" but strategy entry timeframe is '{requested_entry_tf}'.")

            do_resample = False
            if getattr(args, 'auto_resample', False):
                do_resample = True
            else:
                resp = input("Resample loaded data to the requested timeframe? [y/N]: ")
                do_resample = resp.strip().lower() == 'y'

            if do_resample:
                # Perform resampling and save to data/raw with new filename
                rule = requested_entry_tf.replace('m', 'T').replace('h', 'H').replace('d', 'D')
                # Build output path by replacing timeframe token if present in filename
                data_path_obj = Path(data_path)
                dst_name = data_path_obj.name
                # Try replace common tokens like '15m' or '15min'
                for token in ['15m', '5m', '1m', '30m', '1h', '4h', '1d', '1D', '4H']:
                    if token in dst_name:
                        dst_name = dst_name.replace(token, requested_entry_tf)
                        break
                else:
                    # Prepend timeframe if not present
                    dst_name = f"{data_path_obj.stem.rsplit('.',1)[0]}-{requested_entry_tf}.parquet"

                dst_path = data_path_obj.parent / dst_name

                print(f"Resampling to {requested_entry_tf} (rule={rule}) and saving to {dst_path}...")
                out = pd.DataFrame()
                out['open'] = df['open'].resample(rule).first()
                out['high'] = df['high'].resample(rule).max()
                out['low'] = df['low'].resample(rule).min()
                out['close'] = df['close'].resample(rule).last()
                out['volume'] = df['volume'].resample(rule).sum()
                out = out.dropna()
                out.to_parquet(dst_path)
                print(f"Saved resampled file: {dst_path} ({len(out)} bars)")
                # Replace df and data_path to use resampled file
                df = out
                data_path = dst_path
    
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
    
    # Run backtest
    print(f"\nRunning backtest on {data_path}...")
    if args.start_date or args.end_date:
        print(f"  Date range: {df.index[0].date()} to {df.index[-1].date()}")
    
    # Parse dates for engine.run()
    start_date_ts = pd.to_datetime(args.start_date) if args.start_date else None
    end_date_ts = pd.to_datetime(args.end_date) if args.end_date else None
    
    result = engine.run(df, start_date=start_date_ts, end_date=end_date_ts)
    
    # Calculate enhanced metrics
    enhanced_metrics = calculate_enhanced_metrics(result)
    
    # Print summary
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)
    print(f"Strategy: {result.strategy_name}")
    print(f"Total Trades: {result.total_trades}")
    print(f"Winning Trades: {result.winning_trades}")
    print(f"Losing Trades: {result.losing_trades}")
    print(f"Win Rate: {result.win_rate:.2f}%")
    print(f"Total P&L: ${result.total_pnl:,.2f}")
    print(f"Max Drawdown: {result.max_drawdown:.2f}%")
    # Helper function to safely format metrics that might be None
    def safe_format(value, default=0.0, format_str='.2f'):
        """Safely format a metric value, handling None."""
        if value is None:
            value = default
        return f"{value:{format_str}}"
    
    print(f"Profit Factor: {safe_format(enhanced_metrics.get('profit_factor'))}")
    print(f"Sharpe Ratio: {safe_format(enhanced_metrics.get('sharpe_ratio'))}")
    print(f"Sortino Ratio: {safe_format(enhanced_metrics.get('sortino_ratio'))}")
    print(f"CAGR: {safe_format(enhanced_metrics.get('cagr'))}%")
    print("="*60)
    
    # Generate report
    reports_dir = Path(__file__).parent.parent / 'reports'
    generator = ReportGenerator(reports_dir)
    report_path = generator.generate_backtest_report(
        result=result,
        enhanced_metrics=enhanced_metrics,
        output_name=args.output
    )
    
    print(f"\nReport saved to: {report_path}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

