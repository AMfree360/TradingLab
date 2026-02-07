#!/usr/bin/env python3
"""Download historical data for backtesting."""

import argparse
import sys
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def download_binance_data(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_path: Path
) -> Path:
    """
    Download data from Binance public API.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT')
        interval: Timeframe (e.g., '15m', '1h', '1d')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_path: Where to save the data
    
    Returns:
        Path to saved file
    """
    try:
        import requests
    except ImportError:
        print("Error: 'requests' library not installed.")
        print("Install it with: pip install requests")
        sys.exit(1)
    
    # Map interval to Binance format
    interval_map = {
        '1m': '1m',
        '5m': '5m',
        '15m': '15m',
        '30m': '30m',
        '1h': '1h',
        '4h': '4h',
        '1d': '1d',
    }
    
    if interval not in interval_map:
        raise ValueError(f"Unsupported interval: {interval}")
    
    binance_interval = interval_map[interval]
    
    # Convert dates to timestamps
    start_dt = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_ts = int(start_dt.timestamp() * 1000)
    end_ts = int(end_dt.timestamp() * 1000)
    
    print(f"Downloading {symbol} {interval} data from {start_date} to {end_date}...")
    
    all_data = []
    current_start = start_ts
    limit = 1000  # Binance limit per request
    
    while current_start < end_ts:
        # Binance public API endpoint
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': binance_interval,
            'startTime': current_start,
            'limit': limit
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
            
            all_data.extend(data)
            
            # Update start time for next batch
            current_start = data[-1][0] + 1  # Next millisecond after last candle
            
            # Progress indicator
            last_time = datetime.fromtimestamp(data[-1][0] / 1000)
            print(f"  Downloaded up to {last_time.strftime('%Y-%m-%d %H:%M')}... ({len(all_data)} candles)")
            
            # Check if we've reached the end
            if data[-1][0] >= end_ts:
                break
            
            # Rate limiting - be nice to the API
            import time
            time.sleep(0.2)
            
        except requests.exceptions.RequestException as e:
            print(f"Error downloading data: {e}")
            sys.exit(1)
    
    if not all_data:
        print("No data downloaded!")
        sys.exit(1)
    
    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    
    # Select and convert OHLCV columns
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
    # Sort by index
    df = df.sort_index()
    
    # Remove duplicates
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"\nDownloaded {len(df)} candles")
    print(f"Date range: {df.index[0]} to {df.index[-1]}")
    
    # Save to file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.parquet':
        df.to_parquet(output_path)
        print(f"Saved to: {output_path}")
    else:
        df.to_csv(output_path)
        print(f"Saved to: {output_path}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Download historical data for backtesting')
    parser.add_argument(
        '--symbol',
        type=str,
        default='BTCUSDT',
        help='Trading pair (default: BTCUSDT)'
    )
    parser.add_argument(
        '--interval',
        type=str,
        default='15m',
        help='Timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d (default: 15m)'
    )
    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: data/raw/{SYMBOL}-{INTERVAL}-{START}.parquet)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output format (default: parquet)'
    )
    
    args = parser.parse_args()
    
    # Generate output path if not provided
    if args.output:
        output_path = Path(args.output)
    else:
        data_dir = Path(__file__).parent.parent / 'data' / 'raw'
        data_dir.mkdir(parents=True, exist_ok=True)
        ext = '.parquet' if args.format == 'parquet' else '.csv'
        filename = f"{args.symbol}-{args.interval}-{args.start}{ext}"
        output_path = data_dir / filename
    
    # Download data
    try:
        download_binance_data(
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            output_path=output_path
        )
        print(f"\n✓ Data downloaded successfully!")
        print(f"\nYou can now run backtest with:")
        print(f"  python3 scripts/run_backtest.py --strategy ma_alignment --data {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

