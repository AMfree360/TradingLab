"""Universal data loader for OHLCV market data.

Supports multiple formats:
- CSV files (comma, tab, semicolon separated)
- Parquet files
- Various column naming conventions (OPEN, open, <OPEN>, etc.)
- DATE/TIME column combinations
- Numeric timestamps (Unix seconds/milliseconds)
- Missing columns (derives from available data)
"""

from pathlib import Path
import pandas as pd
from typing import Optional, Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """Universal loader for OHLCV market data from various sources."""
    
    # Common column name variations (case-insensitive, with/without angle brackets)
    COLUMN_ALIASES = {
        'open': ['open', 'o', '<open>', 'open_price', 'openprice'],
        'high': ['high', 'h', '<high>', 'high_price', 'highprice'],
        'low': ['low', 'l', '<low>', 'low_price', 'lowprice'],
        'close': ['close', 'c', 'closed', '<close>', 'close_price', 'closeprice'],
        'volume': ['volume', 'vol', 'v', '<vol>', '<volume>', 'tickvol', '<tickvol>'],
        'timestamp': [
            'timestamp', 'datetime', 'date', 'time',
            'open_time', 'close_time', 'opentime', 'closetime',
            'timestamp_ms', 'open_time_ms', 'close_time_ms',
            'Timestamp', 'DateTime', 'Date', 'Time',
            'Open Time', 'Close Time'
        ],
        'date': ['date', '<date>', 'DATE'],
        'time': ['time', '<time>', 'TIME']
    }
    
    def load(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load OHLCV data from CSV or Parquet file.
        
        Args:
            file_path: Path to data file
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            DataFrame with datetime index and standardized OHLCV columns:
            ['open', 'high', 'low', 'close', 'volume']
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        # Load raw data
        df = self._load_raw_data(file_path, **kwargs)
        
        # Process timestamp/index
        df = self._process_timestamp(df, file_path)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Derive missing columns if needed
        df = self._derive_missing_columns(df)
        
        # Validate and select final columns
        df = self._finalize_dataframe(df, file_path)
        
        return df
    
    def _load_raw_data(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load raw data from file."""
        if file_path.suffix.lower() == '.parquet':
            return self._load_parquet(file_path, **kwargs)
        else:
            return self._load_csv(file_path, **kwargs)
    
    def _load_csv(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load CSV file with automatic separator detection."""
        # First, check if file has the special format: timestamp;open;high;low;close;volume
        # (common in some NinjaTrader exports)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
            
            # Check if the entire line is in format: "YYYYMMDD HHMMSS;O;H;L;C;V"
            if ';' in first_line and len(first_line.split(';')) >= 6:
                # Check if first part looks like timestamp (YYYYMMDD HHMMSS)
                parts = first_line.split(';')
                timestamp_str = parts[0].strip()
                if len(timestamp_str) >= 14 and timestamp_str.replace(' ', '').isdigit():
                    # Parse this special format
                    return self._parse_semicolon_ohlcv_format(file_path)
        except Exception as e:
            logger.debug(f"Could not check for special format: {e}")
        
        # Check for Binance klines format (no header, comma-separated)
        # Format: Open time, Open, High, Low, Close, Volume, Close time, Quote volume, Trades, Taker buy base, Taker buy quote, Ignore
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline().strip()
                parts = first_line.split(',')
                # Check if first column is a numeric timestamp (milliseconds)
                if len(parts) >= 6:
                    try:
                        first_val = float(parts[0])
                        # Binance timestamps are in milliseconds (typically > 1e12)
                        if first_val > 1e12:
                            # Likely Binance klines format
                            return self._parse_binance_klines_format(file_path)
                    except ValueError:
                        pass
        except Exception as e:
            logger.debug(f"Could not check for Binance format: {e}")
        
        # Detect separator
        sep = self._detect_separator(file_path)
        
        # Use detected separator unless overridden
        if 'sep' not in kwargs and 'delimiter' not in kwargs:
            kwargs['sep'] = sep
        
        logger.debug(f"Loading CSV with separator: '{sep}'")
        df = pd.read_csv(file_path, **kwargs)
        
        logger.debug(f"CSV loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def _parse_semicolon_ohlcv_format(self, file_path: Path) -> pd.DataFrame:
        """Parse CSV with format: timestamp;open;high;low;close;volume per line."""
        def parse_row(row_str):
            """Parse a row like '20250914 004800;6645.25;6645.25;6645.25;6645.25;1'"""
            parts = str(row_str).strip().split(';')
            if len(parts) >= 6:
                timestamp_str = parts[0].strip()
                # Parse YYYYMMDD HHMMSS format
                if len(timestamp_str) >= 14:
                    date_part = timestamp_str[:8]  # YYYYMMDD
                    time_part = timestamp_str[9:15] if len(timestamp_str) > 9 else "000000"  # HHMMSS
                    datetime_str = f"{date_part} {time_part}"
                    try:
                        dt = pd.to_datetime(datetime_str, format='%Y%m%d %H%M%S')
                    except:
                        dt = pd.to_datetime(datetime_str)
                    return {
                        'timestamp': dt,
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]) if len(parts) > 5 else 0
                    }
            return None
        
        # Read and parse all lines
        parsed_rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parsed = parse_row(line)
                    if parsed:
                        parsed_rows.append(parsed)
        
        if not parsed_rows:
            raise ValueError(f"Could not parse any rows from {file_path}")
        
        df = pd.DataFrame(parsed_rows)
        df.set_index('timestamp', inplace=True)
        
        logger.debug(f"Parsed semicolon format: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def _parse_binance_klines_format(self, file_path: Path) -> pd.DataFrame:
        """Parse Binance klines CSV format (no header).
        
        Format: Open time, Open, High, Low, Close, Volume, Close time, Quote volume, Trades, Taker buy base, Taker buy quote, Ignore
        """
        def parse_row(row_str):
            """Parse a Binance klines row."""
            parts = str(row_str).strip().split(',')
            if len(parts) >= 6:
                try:
                    # Column 0: Open time (timestamp in milliseconds)
                    timestamp_ms = int(float(parts[0]))
                    dt = pd.to_datetime(timestamp_ms, unit='ms')
                    
                    return {
                        'timestamp': dt,
                        'open': float(parts[1]),
                        'high': float(parts[2]),
                        'low': float(parts[3]),
                        'close': float(parts[4]),
                        'volume': float(parts[5]) if len(parts) > 5 else 0.0
                    }
                except (ValueError, IndexError) as e:
                    logger.debug(f"Failed to parse row: {e}")
                    return None
            return None
        
        # Read and parse all lines
        parsed_rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:  # Skip empty lines
                    parsed = parse_row(line)
                    if parsed:
                        parsed_rows.append(parsed)
        
        if not parsed_rows:
            raise ValueError(f"Could not parse any rows from {file_path}")
        
        df = pd.DataFrame(parsed_rows)
        df.set_index('timestamp', inplace=True)
        
        logger.debug(f"Parsed Binance klines format: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        return df
    
    def _load_parquet(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """Load Parquet file, handling transposed formats."""
        df = pd.read_parquet(file_path, **kwargs)
        
        logger.debug(f"Parquet loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        
        # Check for transposed format (timestamps as column names)
        if self._is_transposed(df):
            logger.info("Detected transposed Parquet format, transposing...")
            df = df.T
            logger.debug(f"After transpose: {df.shape[0]} rows, {df.shape[1]} columns")
        
        return df
    
    def _detect_separator(self, file_path: Path) -> str:
        """Detect CSV separator (comma, tab, semicolon)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                first_line = f.readline()
            
            # Count occurrences of each separator
            counts = {
                '\t': first_line.count('\t'),
                ';': first_line.count(';'),
                ',': first_line.count(',')
            }
            
            # Return separator with most occurrences
            sep = max(counts, key=counts.get)
            
            # If no clear winner, default to comma
            if counts[sep] == 0:
                sep = ','
            
            logger.debug(f"Detected separator: '{sep}' (counts: {counts})")
            return sep
        except Exception as e:
            logger.warning(f"Could not detect separator: {e}, using comma")
            return ','
    
    def _is_transposed(self, df: pd.DataFrame) -> bool:
        """Check if Parquet file is transposed (timestamps as column names)."""
        if len(df.columns) == 0:
            return False
        
        # Check if first column name looks like a timestamp
        try:
            first_col = str(df.columns[0])
            ts_val = float(first_col)
            # If it's a timestamp and we have many columns, likely transposed
            if ts_val > 946684800 and len(df.columns) > 100:
                # Check if index has OHLCV-like names
                ohlcv_names = ['open', 'high', 'low', 'close', 'volume']
                index_lower = [str(idx).lower() for idx in df.index[:10]]
                has_ohlcv = any(name in ' '.join(index_lower) for name in ohlcv_names)
                return has_ohlcv
        except (ValueError, TypeError):
            pass
        
        return False
    
    def _process_timestamp(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Process timestamp/index to create datetime index."""
        # If already datetime index, we're done
        if isinstance(df.index, pd.DatetimeIndex):
            logger.debug("Index is already datetime")
            return df
        
        # Try DATE + TIME column combination (common in forex data)
        df, combined = self._combine_date_time(df)
        if combined:
            logger.debug("Successfully combined DATE and TIME columns")
            return df
        
        # Try to find timestamp column
        timestamp_col = self._find_timestamp_column(df)
        
        if timestamp_col:
            df = self._set_index_from_column(df, timestamp_col)
            return df
        
        # Try first column if it looks like a timestamp
        if len(df.columns) > 0:
            first_col = df.columns[0]
            if self._is_timestamp_column(df, first_col):
                df = self._set_index_from_column(df, first_col)
                return df
        
        # Try to parse existing index
        if self._is_timestamp_index(df.index):
            df.index = self._parse_timestamp_series(df.index)
            return df
        
        # If all else fails, raise error
        raise ValueError(
            f"Could not determine timestamp/index for {file_path}. "
            f"Expected: datetime index, timestamp column, DATE+TIME columns, "
            f"or numeric timestamp index. "
            f"Current index type: {type(df.index)}, columns: {df.columns.tolist()}"
        )
    
    def _combine_date_time(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, bool]:
        """Combine DATE and TIME columns into datetime index."""
        date_col = None
        time_col = None
        
        # Find DATE and TIME columns (case-insensitive, with/without brackets)
        for col in df.columns:
            col_clean = self._clean_column_name(col)
            if col_clean == 'date' and date_col is None:
                date_col = col
            elif col_clean == 'time' and time_col is None:
                time_col = col
        
        if not (date_col and time_col):
            return df, False
        
        logger.debug(f"Found DATE column: '{date_col}', TIME column: '{time_col}'")
        
        try:
            # Combine date and time strings
            datetime_str = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
            datetime_series = pd.to_datetime(datetime_str, errors='coerce')
            
            # Drop DATE and TIME columns, set combined datetime as index
            df = df.drop(columns=[date_col, time_col])
            df.index = datetime_series
            
            logger.debug(f"Combined DATE and TIME into datetime index")
            return df, True
        except Exception as e:
            logger.warning(f"Failed to combine DATE and TIME: {e}")
            return df, False
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column by checking common names."""
        for alias in self.COLUMN_ALIASES['timestamp']:
            # Check exact match (case-sensitive)
            if alias in df.columns:
                return alias
            # Check case-insensitive
            for col in df.columns:
                if col.lower() == alias.lower():
                    return col
        
        return None
    
    def _is_timestamp_column(self, df: pd.DataFrame, col: str) -> bool:
        """Check if a column contains timestamp data."""
        if col not in df.columns:
            return False
        
        # Check if numeric (Unix timestamp)
        if pd.api.types.is_numeric_dtype(df[col]):
            non_null = df[col].dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[0]
                # Reasonable timestamp range (after 2000-01-01)
                if sample > 946684800:
                    return True
        
        # Check if string that can be parsed as datetime
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                sample = df[col].dropna().iloc[0] if len(df[col].dropna()) > 0 else None
                if sample:
                    pd.to_datetime(sample, errors='raise')
                    return True
            except:
                pass
        
        return False
    
    def _is_timestamp_index(self, index: pd.Index) -> bool:
        """Check if index contains timestamp data."""
        if pd.api.types.is_numeric_dtype(index):
            non_null = index.dropna()
            if len(non_null) > 0:
                sample = non_null[0]
                if sample > 946684800:
                    return True
        return False
    
    def _set_index_from_column(self, df: pd.DataFrame, col: str) -> pd.DataFrame:
        """Set datetime index from a column."""
        # Parse timestamp values
        timestamp_series = self._parse_timestamp_series(df[col])
        
        # Set as index and drop column
        df = df.drop(columns=[col])
        df.index = timestamp_series
        
        logger.debug(f"Set datetime index from column '{col}'")
        return df
    
    def _parse_timestamp_series(self, series: pd.Series) -> pd.Series:
        """Parse timestamp series (handles Unix seconds/milliseconds)."""
        # If already datetime, return as-is
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        
        # If numeric, determine unit (seconds vs milliseconds)
        if pd.api.types.is_numeric_dtype(series):
            non_null = series.dropna()
            if len(non_null) > 0:
                sample = non_null.iloc[0]
                if sample > 1e12:  # Likely milliseconds
                    return pd.to_datetime(series, unit='ms', errors='coerce')
                elif sample > 946684800:  # Likely seconds (after 2000-01-01)
                    return pd.to_datetime(series, unit='s', errors='coerce')
                else:
                    # Try milliseconds first, fallback to seconds
                    try:
                        return pd.to_datetime(series, unit='ms', errors='coerce')
                    except:
                        return pd.to_datetime(series, unit='s', errors='coerce')
        
        # Try to parse as datetime string
        return pd.to_datetime(series, errors='coerce')
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase OHLCV."""
        column_map = {}
        
        # Map each standard column
        for standard_name, aliases in self.COLUMN_ALIASES.items():
            if standard_name in ['timestamp', 'date', 'time']:
                continue  # Skip these, already handled
            
            # Clean aliases for comparison (strip brackets, lowercase)
            cleaned_aliases = [self._clean_column_name(a) for a in aliases]
            
            # Find matching column
            for col in df.columns:
                col_clean = self._clean_column_name(col)
                
                # Check if this column matches any alias for this standard name
                if col_clean in cleaned_aliases:
                    if standard_name not in column_map.values():
                        column_map[col] = standard_name
                        logger.debug(f"Mapped '{col}' -> '{standard_name}'")
                        break
        
        # Apply mapping
        if column_map:
            df = df.rename(columns=column_map)
            logger.debug(f"Column mapping applied: {column_map}")
        
        logger.debug(f"Columns after standardization: {df.columns.tolist()}")
        return df
    
    def _clean_column_name(self, col: str) -> str:
        """Clean column name for comparison (strip brackets, lowercase)."""
        return str(col).strip().strip('<>').lower()
    
    def _derive_missing_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Derive missing OHLCV columns from available data."""
        # Derive CLOSE if missing
        if 'close' not in df.columns:
            logger.warning("CLOSE column missing, deriving from other columns...")
            if 'open' in df.columns and len(df) > 1:
                # Use next bar's OPEN as current bar's CLOSE (common in forex)
                df['close'] = df['open'].shift(-1)
                # For last bar, use (HIGH+LOW)/2 or HIGH
                if len(df) > 0:
                    last_idx = df.index[-1]
                    if 'high' in df.columns and 'low' in df.columns:
                        df.loc[last_idx, 'close'] = (df.loc[last_idx, 'high'] + df.loc[last_idx, 'low']) / 2
                    elif 'high' in df.columns:
                        df.loc[last_idx, 'close'] = df.loc[last_idx, 'high']
                    else:
                        df.loc[last_idx, 'close'] = df.loc[last_idx, 'open']
            elif 'high' in df.columns and 'low' in df.columns:
                df['close'] = (df['high'] + df['low']) / 2
            elif 'high' in df.columns:
                df['close'] = df['high']
            elif 'open' in df.columns:
                df['close'] = df['open']
            else:
                raise ValueError("Cannot derive CLOSE: missing OPEN, HIGH, and LOW")
        
        # Derive VOLUME if missing (set to 1.0)
        if 'volume' not in df.columns:
            logger.warning("VOLUME column missing, using default value of 1.0")
            df['volume'] = 1.0
        
        return df
    
    def _finalize_dataframe(self, df: pd.DataFrame, file_path: Path) -> pd.DataFrame:
        """Finalize dataframe: validate, clean, and select columns."""
        # Validate required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        
        if missing:
            file_type = "Parquet" if file_path.suffix.lower() == '.parquet' else "CSV"
            raise ValueError(
                f"{file_type} file missing required columns: {missing}. "
                f"Available columns: {df.columns.tolist()}. "
                f"Please ensure your data file contains OPEN, HIGH, LOW, CLOSE, and VOLUME columns."
            )
        
        # Select only required columns
        df = df[required]
        
        # Remove rows with invalid timestamps (NaT)
        if isinstance(df.index, pd.DatetimeIndex):
            initial_len = len(df)
            df = df[df.index.notna()]
            removed = initial_len - len(df)
            if removed > 0:
                logger.warning(f"Removed {removed} rows ({removed/initial_len*100:.1f}%) with invalid timestamps")
                if removed / initial_len > 0.5:
                    raise ValueError(
                        f"More than 50% of data was removed due to invalid timestamps! "
                        f"This suggests a data format issue. Please check your data file."
                    )
        
        # Sort by index
        df = df.sort_index()
        
        # Final validation
        if len(df) == 0:
            raise ValueError("No valid data remaining after processing")
        
        logger.info(f"Successfully loaded {len(df)} bars from {file_path.name}")
        logger.debug(f"Date range: {df.index[0]} to {df.index[-1]}")
        
        return df

