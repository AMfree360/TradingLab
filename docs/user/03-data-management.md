# Data Management Guide

This guide explains how to get, manage, and use historical data for backtesting in Trading Lab.

## Data Requirements

Trading Lab needs historical market data in a specific format:

### Required Format

**File format**: CSV or Parquet
**Required columns**: `open`, `high`, `low`, `close`, `volume`
**Index**: Datetime (or `timestamp` column)

### Example CSV Format

```csv
timestamp,open,high,low,close,volume
2023-01-01 00:00:00,16500.0,16550.0,16480.0,16520.0,1234.56
2023-01-01 00:01:00,16520.0,16530.0,16510.0,16525.0,987.65
...
```

### Example with Datetime Index

```csv
,open,high,low,close,volume
2023-01-01 00:00:00,16500.0,16550.0,16480.0,16520.0,1234.56
2023-01-01 00:01:00,16520.0,16530.0,16510.0,16525.0,987.65
...
```

## Getting Data

### Method 1: Manual Download

#### From Binance

1. **Visit**: [Binance Historical Data](https://www.binance.com/en/support/faq/how-to-download-historical-data-on-binance-360003492232)
2. **Select**:
   - Symbol (e.g., BTCUSDT)
   - Timeframe (e.g., 1m, 5m, 1h)
   - Date range
3. **Download**: CSV file
4. **Save**: To `data/raw/` folder

#### From Other Sources

- **CryptoCompare**: Free historical data
- **Yahoo Finance**: For stocks
- **TradingView**: Export data
- **Your exchange**: Most exchanges provide historical data

### Method 2: Using APIs (Automated)

#### Binance API

Create a script to download data:

```python
import pandas as pd
from binance.client import Client
from datetime import datetime, timedelta

# Initialize client (no API key needed for public data)
client = Client()

# Get historical data
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
start_date = '2023-01-01'
end_date = '2023-12-31'

# Fetch data
klines = client.get_historical_klines(
    symbol,
    interval,
    start_date,
    end_date
)

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_volume', 'trades', 'taker_buy_base',
    'taker_buy_quote', 'ignore'
])

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Select required columns
df = df[['open', 'high', 'low', 'close', 'volume']]

# Convert to float
for col in df.columns:
    df[col] = df[col].astype(float)

# Save
df.to_csv(f'data/raw/{symbol}_1m.csv')
```

#### Other Exchanges

Similar process for other exchanges:
- **Coinbase**: Use their API
- **Kraken**: Use their API
- **Custom scripts**: Write your own downloader

### Method 3: Using Data Providers

**Paid services**:
- **Alpha Vantage**: Stock and crypto data
- **Quandl**: Financial data
- **Polygon.io**: Real-time and historical

**Free services**:
- **Yahoo Finance API**: Free stock data
- **CryptoCompare**: Free crypto data
- **FRED**: Economic data

## Data Organization

### Recommended Structure

```
data/
├── raw/                    # Original downloaded data
│   ├── BTCUSDT_1m.csv
│   ├── BTCUSDT_1h.csv
│   └── ETHUSDT_1m.csv
├── processed/              # Processed/cleaned data (auto-generated)
└── manifests/              # Data metadata (auto-generated)

### Dataset manifests (auto-generated)

Trading Lab writes dataset manifest JSON artifacts under `data/manifests/` during backtests and validation runs.

- Manifests provide a deterministic identity (stable hashing) for a particular dataset slice.
- Phase 1 / Phase 2 / Holdout runs can **lock** to a manifest identity to prevent accidental data leakage or silent dataset mutation.

**Practical guidance**:
- Avoid overwriting a dataset file in-place once you start Phase 1/2/holdout. If you need to update data, create a new file name (treat it as a new dataset version).
- If you see a **dataset lock mismatch** error, stop and decide whether you intended to change the dataset. Do not “force reset” state as a convenience unless you are explicitly in dev/testing mode.
```

### Naming Convention

**Recommended format**: `{SYMBOL}_{TIMEFRAME}.csv`

Examples:
- `BTCUSDT_1m.csv` - Bitcoin 1-minute data
- `ETHUSDT_1h.csv` - Ethereum 1-hour data
- `SPY_1d.csv` - S&P 500 ETF daily data

## Data Quality

### Important Checks

Before using data, verify:

1. **Completeness**: No missing periods
2. **Accuracy**: Prices make sense
3. **Format**: Correct columns and types
4. **Timeframe**: Matches expected interval

### Common Issues

#### Issue 1: Missing Data

**Problem**: Gaps in data (missing candles)

**Solutions**:
- Fill gaps with previous close
- Interpolate missing values
- Remove incomplete periods
- Use data from reliable source

#### Issue 2: Incorrect Format

**Problem**: Wrong column names or types

**Solutions**:
- Rename columns to match requirements
- Convert types (strings to floats)
- Fix datetime format

#### Issue 3: Bad Data

**Problem**: Unrealistic prices (e.g., $0 or $1,000,000,000)

**Solutions**:
- Filter outliers
- Check data source
- Validate against known prices

### Data Validation Script

Create a script to validate data:

```python
import pandas as pd

def validate_data(file_path):
    """Validate data file."""
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    
    # Check columns
    required = ['open', 'high', 'low', 'close', 'volume']
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"Missing columns: {missing}")
        return False
    
    # Check for NaN values
    if df.isnull().any().any():
        print("Found NaN values!")
        return False
    
    # Check price logic (high >= low, etc.)
    if (df['high'] < df['low']).any():
        print("Found high < low!")
        return False
    
    if (df['high'] < df['close']).any() or (df['high'] < df['open']).any():
        print("Found high < close or open!")
        return False
    
    if (df['low'] > df['close']).any() or (df['low'] > df['open']).any():
        print("Found low > close or open!")
        return False
    
    # Check for zeros
    if (df[['open', 'high', 'low', 'close']] == 0).any().any():
        print("Found zero prices!")
        return False
    
    print("Data validation passed!")
    return True

# Use it
validate_data('data/raw/BTCUSDT_1m.csv')
```

## Data Preparation

### Consolidating Multiple Data Files

For validation (walk-forward analysis), you need a single consolidated file containing all your historical data. If you download data in yearly chunks (e.g., `BTCUSDT-15m-2020.parquet`, `BTCUSDT-15m-2021.parquet`, etc.), you'll need to combine them.

#### Using the Consolidation Script

Trading Lab includes a script to consolidate multiple files:

```bash
# Consolidate multiple yearly files
python3 scripts/consolidate_data.py \
  --input data/raw/BTCUSDT-15m-2020.parquet \
           data/raw/BTCUSDT-15m-2021.parquet \
           data/raw/BTCUSDT-15m-2022.parquet \
  --output data/raw/BTCUSDT-15m-2020-2022.parquet

# Or use wildcards to match multiple files
python3 scripts/consolidate_data.py \
  --input "data/raw/BTCUSDT-15m-*.parquet" \
  --output data/raw/BTCUSDT-15m-consolidated.parquet
```

**What the script does**:
- Loads all input files using the universal data loader
- Combines them chronologically
- Removes duplicate timestamps (keeps first occurrence)
- Sorts by timestamp
- Saves as a single file ready for validation

**Options**:
- `--input`: One or more input files (supports wildcards)
- `--output`: Output consolidated file path
- `--no-sort`: Don't sort by timestamp (default: sorted)
- `--keep-duplicates`: Keep duplicate timestamps (default: removed)
- `--dedupe-keep {first,last}`: When removing duplicates, which record to keep (default: first)
- `--allow-mixed-frequency`: Allow consolidating files that appear to have different bar intervals (default: off)

#### Important: Mixed-frequency downloads (daily + hourly + monthly)

If you manually download data in mixed “chunks” (e.g., some files are daily bars, others are hourly), **do not merge them directly**.

- Trading Lab expects a single consistent bar interval per dataset.
- Mixing intervals can silently create incorrect backtests (duplicated timestamps, inconsistent spacing, distorted indicators).

**Recommended fix**:

1) Pick the timeframe you want to trade/test (e.g., `4h`).
2) Resample each source file to that timeframe.
3) Consolidate the resampled files.

Example:

```bash
# Resample each chunk to the same timeframe
python3 scripts/resample_ohlcv.py --src data/raw/BTCUSDT-1h-2021.parquet --dst data/raw/BTCUSDT-4h-2021.parquet --rule 4h
python3 scripts/resample_ohlcv.py --src data/raw/BTCUSDT-1h-2022.parquet --dst data/raw/BTCUSDT-4h-2022.parquet --rule 4h

# Then consolidate
python3 scripts/consolidate_data.py \
    --input "data/raw/BTCUSDT-4h-*.parquet" \
    --output data/processed/BTCUSDT-4h-2021-2022.parquet
```

If you truly intend to merge mixed intervals (rare), you can pass `--allow-mixed-frequency`, but you should treat the output as “research only” unless you fully understand the consequences.

**Example workflow**:

```bash
# Step 1: Download yearly data (or use existing files)
# You have: BTCUSDT-15m-2020.parquet, BTCUSDT-15m-2021.parquet, BTCUSDT-15m-2022.parquet

# Step 2: Consolidate into one file
python3 scripts/consolidate_data.py \
  --input data/raw/BTCUSDT-15m-2020.parquet \
           data/raw/BTCUSDT-15m-2021.parquet \
           data/raw/BTCUSDT-15m-2022.parquet \
  --output data/raw/BTCUSDT-15m-2020-2022.parquet

# Step 3: Run validation on consolidated file
python3 scripts/run_validation.py \
    --strategy ema_crossover \
  --data data/raw/BTCUSDT-15m-2020-2022.parquet
```

**Why consolidate?**
- Walk-forward analysis needs the full date range in one file
- Easier to manage than multiple files
- Better performance (single file read vs. multiple)
- Ensures chronological order across years

### Converting Formats

#### From Exchange Export

Many exchanges export in their own format. Convert to required format:

```python
import pandas as pd

# Read exchange format
df = pd.read_csv('exchange_export.csv')

# Rename columns if needed
df.rename(columns={
    'Open': 'open',
    'High': 'high',
    'Low': 'low',
    'Close': 'close',
    'Volume': 'volume',
    'Date': 'timestamp'
}, inplace=True)

# Set index
df['timestamp'] = pd.to_datetime(df['timestamp'])
df.set_index('timestamp', inplace=True)

# Select required columns
df = df[['open', 'high', 'low', 'close', 'volume']]

# Save
df.to_csv('data/raw/BTCUSDT_1m.csv')
```

### Resampling Data

Convert between timeframes:

```python
import pandas as pd

# Read 1-minute data
df_1m = pd.read_csv('data/raw/BTCUSDT_1m.csv', index_col=0, parse_dates=True)

# Resample to 1-hour
df_1h = df_1m.resample('1H').agg({
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum'
})

# Save
df_1h.to_csv('data/raw/BTCUSDT_1h.csv')
```

## Using Data in Backtests

### Loading Data

Trading Lab automatically loads data when you run backtests:

```bash
python scripts/run_backtest.py --strategy my_strategy --data data/raw/BTCUSDT_1m.csv
```

### Using chunked downloads in backtests

Backtests normally expect a **single** data file. If your `--data` resolves to multiple files (a directory or a glob), you have two options:

1) **Explicitly consolidate first** (recommended for reproducibility):

```bash
python3 scripts/consolidate_data.py --input "data/raw/BTCUSDT-15m-*.parquet" --output data/processed/BTCUSDT-15m.parquet
python3 scripts/run_backtest.py --strategy my_strategy --data data/processed/BTCUSDT-15m.parquet
```

2) **Let the backtest runner consolidate for you**:

```bash
python3 scripts/run_backtest.py --strategy my_strategy --data "data/raw/BTCUSDT-15m-*.parquet" --auto-consolidate
```

Tip: the runner caches the consolidated parquet under `data/processed/consolidated/` so you don’t pay the consolidation cost every run.

### Data Requirements by Strategy

Different strategies need different data:

**Short-term strategies**:
- Need: 1m or 5m data
- Duration: Few weeks to months

**Medium-term strategies**:
- Need: 15m or 1h data
- Duration: Months to 1 year

**Long-term strategies**:
- Need: 4h or 1d data
- Duration: 1+ years

### Minimum Data Requirements

**For backtesting**:
- Minimum: 1-3 months
- Recommended: 6-12 months
- Ideal: 2+ years

**For validation**:
- Minimum: 2 years (for walk-forward)
- Recommended: 3+ years
- Ideal: 5+ years

## Data Storage

### File Sizes

**Approximate sizes**:
- 1-minute data: ~500MB per year per symbol
- 1-hour data: ~10MB per year per symbol
- Daily data: ~1MB per year per symbol

### Storage Tips

1. **Use Parquet**: More efficient than CSV (smaller, faster)
2. **Compress**: Use gzip compression
3. **Archive old data**: Move to separate storage
4. **Keep backups**: Don't lose your data!

### Converting to Parquet

```python
import pandas as pd

# Read CSV
df = pd.read_csv('data/raw/BTCUSDT_1m.csv', index_col=0, parse_dates=True)

# Save as Parquet (much smaller!)
df.to_parquet('data/raw/BTCUSDT_1m.parquet')
```

## Best Practices

### 1. Use Reliable Sources

- Prefer official exchange data
- Verify data quality
- Cross-check with multiple sources

### 2. Keep Raw Data

- Never modify original files
- Keep backups
- Version control if possible

### 3. Document Your Data

- Note data source
- Record download date
- Document any processing

### 4. Validate Before Use

- Always validate data
- Check for issues
- Fix problems before backtesting

### 5. Organize Well

- Use consistent naming
- Organize by symbol/timeframe
- Keep structure clean

## Troubleshooting

### "Data file not found"

**Solutions**:
- Check file path is correct
- Verify file exists
- Use absolute path if needed

### "Missing required columns"

**Solutions**:
- Check column names match requirements
- Rename columns if needed
- Verify data format

### "Invalid datetime index"

**Solutions**:
- Check datetime format
- Convert to datetime if needed
- Set as index properly

### "Data too small"

**Solutions**:
- Download more data
- Check date range
- Verify data covers needed period

## Next Steps

After getting data:

1. **Validate**: Make sure data is correct
2. **Organize**: Put in right folders
3. **Test**: Run a simple backtest
4. **Document**: Note data source and details

## Summary

Good data is essential for good backtests:
- Get data from reliable sources
- Validate before using
- Organize properly
- Keep backups

Remember: Garbage in, garbage out. Quality data leads to quality results!

