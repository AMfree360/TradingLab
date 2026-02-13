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


def download_market_data(
    *,
    provider: str,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    output_path: Path,
    market_type: str = 'spot',
    yf_auto_adjust: bool = False,
    csv_url: str | None = None,
    csv_datetime_col: str = 'timestamp',
    csv_open_col: str = 'open',
    csv_high_col: str = 'high',
    csv_low_col: str = 'low',
    csv_close_col: str = 'close',
    csv_volume_col: str = 'volume',
    csv_date_format: str | None = None,
    csv_unit: str | None = None,
    csv_tz: str | None = None,
) -> tuple[Path, pd.DataFrame | None]:
    """Download OHLCV from the selected provider and save to output_path.

    Returns:
        (path, df_or_none)
        df_or_none is provided for providers that return data in-memory;
        for binance_api, data is saved directly and df is loaded only when needed by caller.
    """
    provider = provider.lower().strip()
    if provider == 'binance_api':
        return (
            download_binance_data(
                symbol=symbol,
                interval=interval,
                start_date=start_date,
                end_date=end_date,
                output_path=output_path,
            ),
            None,
        )

    if provider == 'binance_vision':
        from adapters.data.binance_vision import download_klines_range, save_ohlcv

        df = download_klines_range(
            symbol=symbol,
            interval=interval,
            start_date=start_date,
            end_date=end_date,
            market=market_type,
        )
        save_ohlcv(df, output_path)
        return output_path, df

    if provider == 'yfinance':
        from adapters.data.providers.base import DownloadRequest
        from adapters.data.providers.yfinance_provider import YFinanceProvider
        from adapters.data.binance_vision import save_ohlcv

        req = DownloadRequest(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date)
        df = YFinanceProvider().download(req, auto_adjust=yf_auto_adjust)
        save_ohlcv(df, output_path)
        return output_path, df

    if provider == 'stooq':
        from adapters.data.providers.base import DownloadRequest
        from adapters.data.providers.stooq import StooqProvider
        from adapters.data.binance_vision import save_ohlcv

        req = DownloadRequest(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date)
        df = StooqProvider().download(req)
        save_ohlcv(df, output_path)
        return output_path, df

    if provider == 'csv_url':
        from adapters.data.providers.base import DownloadRequest
        from adapters.data.providers.csv_url import CsvUrlProvider
        from adapters.data.binance_vision import save_ohlcv

        req = DownloadRequest(symbol=symbol, interval=interval, start_date=start_date, end_date=end_date)
        df = CsvUrlProvider().download(
            req,
            url=csv_url,
            datetime_col=csv_datetime_col,
            open_col=csv_open_col,
            high_col=csv_high_col,
            low_col=csv_low_col,
            close_col=csv_close_col,
            volume_col=csv_volume_col,
            date_format=csv_date_format,
            unit=csv_unit,
            tz=csv_tz,
        )
        save_ohlcv(df, output_path)
        return output_path, df

    raise ValueError(
        f"Unsupported provider: {provider}. Supported: binance_api, binance_vision, yfinance, stooq, csv_url"
    )


def _standardize_output_path(
    *,
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
    fmt: str,
    output: str | None,
    folder: str = 'data/raw',
) -> Path:
    if output:
        return Path(output)
    ext = '.parquet' if fmt == 'parquet' else '.csv'
    data_dir = Path(__file__).parent.parent / folder
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir / f"{symbol}-{interval}-{start_date}_to_{end_date}{ext}"


def _maybe_resample_to_processed(
    *,
    df: pd.DataFrame,
    target_tf: str | None,
    processed_output: str | None,
    base_tf: str | None,
) -> Path | None:
    if not target_tf:
        return None

    from engine.resampler import resample_ohlcv

    out_df = resample_ohlcv(
        df,
        target_tf=target_tf,
        base_tf=base_tf,
        label='left',
        closed='left',
    )
    if processed_output:
        out_path = Path(processed_output)
    else:
        out_path = Path(__file__).parent.parent / 'data' / 'processed' / f"AUTO-{target_tf}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path)
    print(f"Saved resampled data to: {out_path} ({len(out_df)} bars)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Download historical data for backtesting')
    parser.add_argument(
        '--provider',
        type=str,
        choices=['binance_api', 'binance_vision', 'yfinance', 'stooq', 'csv_url'],
        help='Data provider (preferred). If omitted, --source/--market-type provide Binance compatibility.'
    )
    parser.add_argument(
        '--source',
        type=str,
        choices=['api', 'vision'],
        default='api',
        help='Download source: api (Binance REST) or vision (Binance Data Vision bulk files). (default: api)'
    )
    parser.add_argument(
        '--market-type',
        type=str,
        choices=['spot', 'futures_um'],
        default='spot',
        help='Market type for vision downloads (default: spot)'
    )
    parser.add_argument(
        '--yf-auto-adjust',
        action='store_true',
        help='Yahoo Finance: return auto-adjusted OHLC (splits/dividends)'
    )
    parser.add_argument(
        '--csv-url',
        type=str,
        help='CSV URL for provider=csv_url'
    )
    parser.add_argument('--csv-datetime-col', type=str, default='timestamp', help='CSV datetime column (default: timestamp)')
    parser.add_argument('--csv-open-col', type=str, default='open', help='CSV open column (default: open)')
    parser.add_argument('--csv-high-col', type=str, default='high', help='CSV high column (default: high)')
    parser.add_argument('--csv-low-col', type=str, default='low', help='CSV low column (default: low)')
    parser.add_argument('--csv-close-col', type=str, default='close', help='CSV close column (default: close)')
    parser.add_argument('--csv-volume-col', type=str, default='volume', help='CSV volume column (default: volume)')
    parser.add_argument('--csv-date-format', type=str, help='Optional: strftime date format for CSV datetime parsing')
    parser.add_argument('--csv-unit', type=str, choices=['s', 'ms'], help='Optional: epoch unit for CSV datetime parsing')
    parser.add_argument('--csv-tz', type=str, help='Optional: timezone name to localize CSV datetimes (e.g., UTC)')
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
        help='Output file path (default: data/raw/{SYMBOL}-{INTERVAL}-{START}_to_{END}.parquet)'
    )
    parser.add_argument(
        '--format',
        type=str,
        choices=['parquet', 'csv'],
        default='parquet',
        help='Output format (default: parquet)'
    )

    parser.add_argument(
        '--resample-to',
        type=str,
        help='Optional: resample downloaded data to timeframe (e.g., 4h) and write to data/processed'
    )
    parser.add_argument(
        '--processed-output',
        type=str,
        help='Optional: output path for resampled data (default: data/processed/AUTO-{TF}.parquet)'
    )
    
    args = parser.parse_args()

    provider = args.provider
    if not provider:
        # Back-compat: --source api|vision
        provider = 'binance_api' if args.source == 'api' else 'binance_vision'

    output_path = _standardize_output_path(
        symbol=args.symbol,
        interval=args.interval,
        start_date=args.start,
        end_date=args.end,
        fmt=args.format,
        output=args.output,
        folder='data/raw',
    )
    
    try:
        df_path, df = download_market_data(
            provider=provider,
            symbol=args.symbol,
            interval=args.interval,
            start_date=args.start,
            end_date=args.end,
            output_path=output_path,
            market_type=args.market_type,
            yf_auto_adjust=bool(args.yf_auto_adjust),
            csv_url=args.csv_url,
            csv_datetime_col=args.csv_datetime_col,
            csv_open_col=args.csv_open_col,
            csv_high_col=args.csv_high_col,
            csv_low_col=args.csv_low_col,
            csv_close_col=args.csv_close_col,
            csv_volume_col=args.csv_volume_col,
            csv_date_format=args.csv_date_format,
            csv_unit=args.csv_unit,
            csv_tz=args.csv_tz,
        )
        if df is None:
            df = pd.read_parquet(df_path) if df_path.suffix == '.parquet' else pd.read_csv(df_path, index_col=0, parse_dates=True)
        print(f"Saved to: {df_path}")

        _maybe_resample_to_processed(
            df=df,
            target_tf=args.resample_to,
            processed_output=args.processed_output,
            base_tf=args.interval,
        )

        print(f"\n✓ Data downloaded successfully!")
        print(f"\nYou can now run backtest with:")
        print(f"  python3 scripts/run_backtest.py --strategy ma_alignment --data {output_path}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

