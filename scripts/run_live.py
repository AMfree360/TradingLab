#!/usr/bin/env python3
"""Script to run live trading (paper or real) with Binance Futures."""

import argparse
import sys
import os
import time
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd

# Load .env file before importing other modules
try:
    from dotenv import load_dotenv
    # Load .env from project root
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence
    else:
        # Try current directory
        load_dotenv(override=True)
except ImportError:
    pass  # python-dotenv not installed, will use environment variables directly
except Exception:
    pass  # Could not load .env, will use environment variables directly

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import StrategyBase
from engine.live_trading_engine import LiveTradingEngine
from engine.market import MarketSpec
from adapters.execution.binance_futures import BinanceFuturesAdapter
from config.schema import load_config, validate_strategy_config
from config.market_loader import apply_market_profile
from config.config_loader import load_strategy_config_with_market

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'data/logs/live_trading_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log .env loading status after logging is configured
try:
    project_root = Path(__file__).parent.parent
    env_path = project_root / '.env'
    if env_path.exists():
        logger.info(f"✓ Found .env file at {env_path}")
        # Verify credentials are loaded (without exposing them)
        api_key = os.getenv('BINANCE_API_KEY')
        api_secret = os.getenv('BINANCE_API_SECRET')
        if api_key:
            logger.info(f"✓ BINANCE_API_KEY loaded (length: {len(api_key)})")
        else:
            logger.warning("✗ BINANCE_API_KEY not found in environment")
        if api_secret:
            logger.info(f"✓ BINANCE_API_SECRET loaded (length: {len(api_secret)})")
        else:
            logger.warning("✗ BINANCE_API_SECRET not found in environment")
    else:
        logger.warning(f"✗ .env file not found at {env_path}")
except Exception as e:
    logger.warning(f"Could not check .env file: {e}")


def fetch_live_data(adapter: BinanceFuturesAdapter, symbol: str, timeframe: str = '15m', limit: int = 500) -> pd.DataFrame:
    """Fetch recent historical data for strategy initialization.
    
    Args:
        adapter: Exchange adapter
        symbol: Trading symbol
        timeframe: Timeframe (e.g., '15m', '1h')
        limit: Number of bars to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    try:
        from binance.client import Client
        
        # Use appropriate client for data
        if adapter.paper_mode:
            # Paper mode: use public client (no auth needed)
            client = Client()
        else:
            # Testnet or live: use the adapter's client (already configured)
            client = adapter.client
        
        # Map timeframe to Binance interval
        interval_map = {
            '1m': Client.KLINE_INTERVAL_1MINUTE,
            '5m': Client.KLINE_INTERVAL_5MINUTE,
            '15m': Client.KLINE_INTERVAL_15MINUTE,
            '30m': Client.KLINE_INTERVAL_30MINUTE,
            '1h': Client.KLINE_INTERVAL_1HOUR,
            '4h': Client.KLINE_INTERVAL_4HOUR,
            '1d': Client.KLINE_INTERVAL_1DAY
        }
        
        interval = interval_map.get(timeframe, Client.KLINE_INTERVAL_15MINUTE)
        
        # Fetch klines
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        # Convert to DataFrame
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_volume', 'trades', 'taker_buy_base',
            'taker_buy_quote', 'ignore'
        ])
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Convert price columns to float
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Keep only OHLCV
        df = df[['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Fetched {len(df)} bars of {timeframe} data for {symbol}")
        return df
        
    except Exception as e:
        logger.error(f"Failed to fetch live data: {e}")
        raise


def main():
    parser = argparse.ArgumentParser(
        description='Run live trading (paper or real) with Binance Futures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper trading (simulated)
  python3 scripts/run_live.py \\
        --strategy ema_crossover \\
    --market BTCUSDT_FUTURES \\
    --paper-mode \\
    --timeframe 15m

  # Live trading (real money - use with caution!)
  python3 scripts/run_live.py \\
        --strategy ema_crossover \\
    --market BTCUSDT_FUTURES \\
    --timeframe 15m \\
    --capital 1000.0
        """
    )
    
    parser.add_argument(
        '--strategy',
        type=str,
        required=True,
        help='Strategy name (must match folder in strategies/)'
    )
    parser.add_argument(
        '--market',
        type=str,
        required=True,
        help='Market symbol (e.g., BTCUSDT_FUTURES, BTCUSDT)'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        default='15m',
        help='Trading timeframe (default: 15m)'
    )
    parser.add_argument(
        '--paper-mode',
        action='store_true',
        help='Enable paper trading mode (simulated, no real orders)'
    )
    parser.add_argument(
        '--testnet',
        action='store_true',
        help='Use Binance testnet (places REAL orders on testnet, not paper mode)'
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
        '--max-daily-loss-pct',
        type=float,
        default=5.0,
        help='Maximum daily loss percentage before stopping (default: 5.0)'
    )
    parser.add_argument(
        '--max-drawdown-pct',
        type=float,
        default=15.0,
        help='Maximum drawdown percentage before stopping (default: 15.0)'
    )
    parser.add_argument(
        '--update-interval',
        type=int,
        default=60,
        help='Update interval in seconds (default: 60)'
    )
    
    args = parser.parse_args()
    
    # Validate mode selection
    if args.paper_mode and args.testnet:
        logger.error("Cannot use both --paper-mode and --testnet. Choose one:")
        logger.error("  --paper-mode: Simulated trading (no API calls)")
        logger.error("  --testnet: Real orders on Binance testnet")
        sys.exit(1)
    
    # Determine actual mode
    # If testnet is specified, disable paper mode (testnet uses real API)
    if args.testnet:
        actual_paper_mode = False
        logger.info("Testnet mode: Will place REAL orders on Binance testnet")
    else:
        actual_paper_mode = args.paper_mode
    
    # Warn for live trading (not testnet, not paper)
    if not actual_paper_mode and not args.testnet:
        response = input(
            "⚠️  WARNING: You are about to trade with REAL MONEY!\n"
            "This will place actual orders on Binance LIVE exchange.\n"
            "Type 'YES' to continue: "
        )
        if response != 'YES':
            logger.info("Live trading cancelled by user")
            sys.exit(0)
    
    # Load strategy
    strategy_dir = Path(__file__).parent.parent / 'strategies' / args.strategy
    if not strategy_dir.exists():
        logger.error(f"Strategy '{args.strategy}' not found in strategies/")
        sys.exit(1)
    
    # Load config
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = strategy_dir / 'config.yml'
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Load market-specific config
    try:
        strategy_config_dict = load_strategy_config_with_market(
            base_config_path=config_path,
            market_symbol=args.market
        )
        logger.info(f"✓ Loaded market-specific config for: {args.market}")
    except FileNotFoundError:
        strategy_config_dict = load_config(config_path)
        logger.info("ℹ Using base config only")
    
    # Apply market profile
    try:
        strategy_config_dict = apply_market_profile(strategy_config_dict, symbol=args.market)
        logger.info(f"✓ Applied market profile for: {args.market}")
    except Exception as e:
        logger.warning(f"Could not load market profile: {e}")
    
    # Validate config
    strategy_config = validate_strategy_config(strategy_config_dict)
    
    # Load strategy class
    strategy_module_path = strategy_dir / 'strategy.py'
    if not strategy_module_path.exists():
        logger.error(f"Strategy file not found: {strategy_module_path}")
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
        logger.error(f"Could not find strategy class in {strategy_module_path}")
        sys.exit(1)
    
    # Create strategy instance
    strategy = strategy_class(strategy_config)
    
    # Load market spec
    try:
        market_spec = MarketSpec.load_from_profiles(args.market)
        logger.info(f"✓ Loaded market spec for: {args.market}")
    except ValueError as e:
        logger.error(f"Failed to load market spec: {e}")
        sys.exit(1)
    
    # Create exchange adapter
    try:
        adapter = BinanceFuturesAdapter(
            paper_mode=actual_paper_mode,
            testnet=args.testnet
        )
        mode_str = "PAPER" if actual_paper_mode else ("TESTNET" if args.testnet else "LIVE")
        logger.info(f"✓ Initialized Binance Futures adapter ({mode_str} mode)")
    except Exception as e:
        logger.error(f"Failed to initialize adapter: {e}")
        sys.exit(1)
    
    # Create live trading engine
    engine = LiveTradingEngine(
        strategy=strategy,
        adapter=adapter,
        market_spec=market_spec,
        initial_capital=args.capital,
        max_daily_loss_pct=args.max_daily_loss_pct,
        max_drawdown_pct=args.max_drawdown_pct
    )
    
    logger.info("="*60)
    logger.info("LIVE TRADING ENGINE STARTED")
    logger.info("="*60)
    logger.info(f"Strategy: {args.strategy}")
    logger.info(f"Market: {args.market}")
    logger.info(f"Timeframe: {args.timeframe}")
    logger.info(f"Mode: {mode_str}")
    logger.info(f"Capital: ${args.capital:,.2f}")
    logger.info("="*60)
    
    # Fetch initial data for strategy warm-up
    logger.info("Fetching initial data for strategy warm-up...")
    try:
        initial_data = fetch_live_data(adapter, market_spec.symbol, args.timeframe, limit=500)
        logger.info(f"✓ Fetched {len(initial_data)} bars of historical data")
    except Exception as e:
        logger.error(f"Failed to fetch initial data: {e}")
        sys.exit(1)

    # Provide market data to engine for ATR-based stop fallback (if enabled)
    try:
        engine.update_recent_bars(initial_data)
    except Exception as e:
        logger.warning(f"Failed to seed engine recent bars (ATR fallback may be unavailable): {e}")
    
    # Initialize strategy with historical data
    logger.info("Initializing strategy with historical data...")
    # The strategy will process the historical data to build indicators
    
    # Main trading loop
    logger.info("Starting main trading loop...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        last_update = time.time()
        iteration = 0
        
        while True:
            current_time = time.time()
            
            # Update at specified interval
            if current_time - last_update >= args.update_interval:
                iteration += 1
                logger.info(f"\n--- Update #{iteration} ---")
                
                # Get current price
                try:
                    current_price = adapter.get_current_price(market_spec.symbol)
                    logger.info(f"Current price: ${current_price:,.2f}")
                except Exception as e:
                    logger.error(f"Failed to get current price: {e}")
                    time.sleep(5)
                    continue
                
                # Update positions
                engine.update_positions()
                
                # Get status
                status = engine.get_status()
                logger.info(f"Balance: ${status['balance']:,.2f} | "
                          f"Available: ${status['available']:,.2f} | "
                          f"Positions: {status['positions']} | "
                          f"Daily P&L: ${status['daily_pnl']:,.2f}")
                
                # Check for signals (this would need to be implemented based on strategy)
                # For now, we'll just monitor
                # In a full implementation, you would:
                # 1. Fetch latest bar data
                # 2. Update strategy with new bar
                # 3. Check for signals
                # 4. Process signals through engine
                
                last_update = current_time
            
            # Sleep briefly to avoid tight loop
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("\n\nTrading stopped by user")
    except Exception as e:
        logger.error(f"Error in trading loop: {e}", exc_info=True)
    finally:
        # Close all positions on exit
        logger.info("Closing all positions...")
        engine._close_all_positions()
        
        # Final status
        status = engine.get_status()
        logger.info("\n" + "="*60)
        logger.info("FINAL STATUS")
        logger.info("="*60)
        logger.info(f"Balance: ${status['balance']:,.2f}")
        logger.info(f"Daily P&L: ${status['daily_pnl']:,.2f}")
        logger.info(f"Trades Executed: {status['trades_executed']}")
        logger.info(f"Trades Skipped: {status['trades_skipped']}")
        logger.info("="*60)


if __name__ == '__main__':
    main()

