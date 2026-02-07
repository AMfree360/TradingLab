"""Data loader adapter for CSV and Parquet files."""

from pathlib import Path
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class CSVDataLoader:
    """Loads OHLCV data from CSV or Parquet files."""
    
    def load(self, file_path: Path, **kwargs) -> pd.DataFrame:
        """
        Load data from CSV or Parquet file.
        
        Expected format:
        - timestamp or datetime column (will be used as index)
        - open, high, low, close, volume columns
        
        Args:
            file_path: Path to CSV or Parquet file
            **kwargs: Additional arguments passed to pandas read function
        
        Returns:
            DataFrame with datetime index and OHLCV columns
        """
        file_path = Path(file_path)
        
        # Determine file type and load accordingly
        if file_path.suffix.lower() == '.parquet':
            # Load Parquet file
            df = pd.read_parquet(file_path, **kwargs)
            
            # Check if Parquet file has timestamps as column names (unusual format)
            # This happens when data is transposed or saved incorrectly
            if len(df.columns) > 0:
                first_col = str(df.columns[0])
                try:
                    first_col_as_ts = float(first_col)
                    # If first column name is a timestamp and we have many columns,
                    # the file might be transposed or in an unusual format
                    if first_col_as_ts > 946684800 and len(df.columns) > 100:
                        logger.warning(f"Detected unusual Parquet format: timestamps as column names. "
                                     f"Shape: {df.shape}, attempting to fix...")
                        
                        # Check if we have OHLCV-like row names
                        ohlcv_in_index = any(str(idx).lower() in ['open', 'high', 'low', 'close', 'volume'] 
                                           for idx in df.index[:20])
                        
                        if ohlcv_in_index:
                            # Data is transposed: rows are fields, columns are timestamps
                            logger.info("Transposing data (rows->columns, columns->rows)")
                            df = df.T
                            # Now column names (old index) should be field names
                            # And index (old column names) should be timestamps
                            
                            # Convert index (which are timestamps) to datetime
                            try:
                                if first_col_as_ts > 1e12:
                                    df.index = pd.to_datetime(df.index.astype(float), unit='ms', errors='coerce')
                                else:
                                    df.index = pd.to_datetime(df.index.astype(float), unit='s', errors='coerce')
                                logger.info(f"Converted {df.index.name} index to datetime")
                            except Exception as e:
                                logger.warning(f"Failed to convert index to datetime: {e}")
                except (ValueError, TypeError):
                    pass  # First column is not a numeric timestamp
        else:
            # Load CSV file
            # Try to detect separator - common formats use comma, tab, or semicolon
            # Read first line to detect separator
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    first_line = f.readline()
                    # Check for tab separator (common in forex data)
                    if '\t' in first_line:
                        sep = '\t'
                    elif ';' in first_line:
                        sep = ';'
                    else:
                        sep = ','  # Default to comma
            except:
                sep = ','  # Fallback to comma
            
            # Use detected separator, but allow override via kwargs
            if 'sep' not in kwargs and 'delimiter' not in kwargs:
                kwargs['sep'] = sep
            df = pd.read_csv(file_path, **kwargs)
        
        # Debug: Log initial state
        logger.debug(f"Loaded DataFrame shape: {df.shape}")
        logger.debug(f"Index type: {type(df.index)}, dtype: {df.index.dtype}")
        logger.debug(f"Index name: {df.index.name}")
        logger.debug(f"All columns after CSV load: {df.columns.tolist()}")
        if len(df) > 0:
            logger.debug(f"First index value: {df.index[0]}, type: {type(df.index[0])}")
        
        # Check if OPEN column exists
        open_cols = [col for col in df.columns if 'open' in col.lower() or col.lower().strip('<>') == 'open']
        if not open_cols:
            logger.warning(f"WARNING: No OPEN column found immediately after CSV load! Columns: {df.columns.tolist()}")
        else:
            logger.debug(f"Found OPEN column(s): {open_cols}")
        
        # Check if index is already datetime
        # Also check if index name suggests it should be datetime
        index_is_datetime = isinstance(df.index, pd.DatetimeIndex)
        index_name_suggests_datetime = df.index.name and df.index.name.lower() in [
            'timestamp', 'datetime', 'date', 'time', 'open_time', 'close_time'
        ]
        
        # Initialize timestamp_col for scope
        timestamp_col = None
        
        if not index_is_datetime:
            # Need to set datetime index
            # Check for DATE and TIME columns (common in forex data)
            # Handle both <DATE> and DATE formats
            date_col = None
            time_col = None
            for col in df.columns:
                col_clean = col.strip().strip('<>').lower()
                if col_clean == 'date' and date_col is None:
                    date_col = col
                elif col_clean == 'time' and time_col is None:
                    time_col = col
            
            if date_col and time_col:
                # Combine DATE and TIME columns into datetime
                logger.debug(f"Found DATE column: '{date_col}', TIME column: '{time_col}'")
                logger.debug(f"All columns before DATE/TIME processing: {df.columns.tolist()}")
                
                try:
                    # Combine date and time strings
                    datetime_str = df[date_col].astype(str) + ' ' + df[time_col].astype(str)
                    datetime_series = pd.to_datetime(datetime_str, errors='coerce')
                    # Only drop DATE and TIME columns, keep all others
                    columns_to_drop = [date_col, time_col]
                    columns_before_drop = df.columns.tolist()
                    df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])
                    df.index = datetime_series
                    logger.debug(f"Combined DATE and TIME columns into datetime index")
                    logger.debug(f"Columns before drop: {columns_before_drop}")
                    logger.debug(f"Columns after DATE/TIME removal: {df.columns.tolist()}")
                    # Verify we didn't accidentally drop other columns
                    expected_columns = len(columns_before_drop) - 2
                    if len(df.columns) < expected_columns:
                        dropped = set(columns_before_drop) - set(df.columns)
                        logger.warning(f"Warning: Dropped {len(columns_before_drop) - len(df.columns)} columns: {dropped}, expected only DATE and TIME")
                    # Check if OPEN is still there
                    open_cols = [col for col in df.columns if col.strip().strip('<>').lower() in ['open', 'o']]
                    if not open_cols:
                        logger.warning(f"WARNING: No OPEN column found after DATE/TIME removal! Available: {df.columns.tolist()}")
                    # Skip the rest of timestamp processing since we've set the index
                    # Update the flag to reflect we now have a datetime index
                    index_is_datetime = True
                except Exception as e:
                    logger.debug(f"Failed to combine DATE/TIME: {e}")
            
            # Continue with other timestamp detection if DATE/TIME combination didn't work
            if not isinstance(df.index, pd.DatetimeIndex):
                # Common timestamp column names (including Binance-specific)
                timestamp_cols = [
                    'timestamp', 'datetime', 'date', 'time', 
                    'Timestamp', 'DateTime', 'Date', 'Time',
                    'open_time', 'close_time', 'Open Time', 'Close Time',
                    'openTime', 'closeTime', 'open_time_ms', 'timestamp_ms'
                ]
                
                # Try to find timestamp column
                for col in timestamp_cols:
                    if col in df.columns:
                        timestamp_col = col
                        break
            
            # If no named timestamp column found, check all columns for timestamp-like data
            if not timestamp_col and len(df.columns) > 0:
                # First, check if any column name is a timestamp (unusual format)
                for col in df.columns[:20]:  # Check first 20 columns
                    try:
                        col_name_as_ts = float(str(col))
                        if col_name_as_ts > 946684800:  # Looks like a timestamp
                            # Column name is a timestamp - this suggests transposed data
                            # But check if column values might also be timestamps
                            if pd.api.types.is_numeric_dtype(df[col]):
                                # Check if values are also timestamps (not just the name)
                                non_null_vals = df[col].dropna()
                                if len(non_null_vals) > 0:
                                    sample_val = non_null_vals.iloc[0]
                                    if sample_val > 946684800:
                                        timestamp_col = col
                                        break
                    except (ValueError, TypeError):
                        pass
                
                # If still not found, check first column values
                if not timestamp_col:
                    first_col = df.columns[0]
                    if pd.api.types.is_numeric_dtype(df[first_col]):
                        # Get a valid sample value
                        non_null_vals = df[first_col].dropna()
                        if len(non_null_vals) > 0:
                            sample_val = non_null_vals.iloc[0]
                            # Check if it looks like a Unix timestamp (reasonable date range)
                            if sample_val > 946684800:  # After 2000-01-01 in seconds
                                timestamp_col = first_col
            
            if timestamp_col:
                # Check if it's numeric (Unix timestamp)
                if pd.api.types.is_numeric_dtype(df[timestamp_col]):
                    # Determine if milliseconds or seconds
                    sample_val = df[timestamp_col].iloc[0] if len(df) > 0 else 0
                    if sample_val > 1e12:  # Likely milliseconds (after year 2001)
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
                    elif sample_val > 946684800:  # Likely seconds (after 2000-01-01)
                        df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
                    else:
                        # Try as milliseconds anyway (might be pre-2000 data)
                        try:
                            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='ms')
                        except:
                            df[timestamp_col] = pd.to_datetime(df[timestamp_col], unit='s')
                else:
                    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.set_index(timestamp_col)
                logger.debug(f"Set index from column '{timestamp_col}'")
            elif len(df.columns) > 0 and df.columns[0] not in ['open', 'high', 'low', 'close', 'volume']:
                # Try first column if it's not OHLCV
                try:
                    first_col = df.columns[0]
                    # Check if numeric (Unix timestamp)
                    if pd.api.types.is_numeric_dtype(df[first_col]):
                        sample_val = df[first_col].iloc[0] if len(df) > 0 else 0
                        if sample_val > 1e12:  # Likely milliseconds
                            datetime_series = pd.to_datetime(df[first_col], unit='ms')
                        else:  # Likely seconds
                            datetime_series = pd.to_datetime(df[first_col], unit='s')
                    else:
                        datetime_series = pd.to_datetime(df[first_col])
                    df = df.drop(columns=[first_col])
                    df.index = datetime_series
                    logger.debug(f"Set index from first column '{first_col}'")
                except Exception as e:
                    logger.debug(f"Failed to set index from first column: {e}")
                    pass
            
            # If still not datetime index, try to parse existing index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    # Check if index is numeric (Unix timestamp)
                    if pd.api.types.is_numeric_dtype(df.index):
                        # Get a valid sample value (skip NaT/NaN)
                        sample_val = None
                        for val in df.index[:100]:  # Check more values
                            if pd.notna(val):
                                sample_val = val
                                break
                        
                        if sample_val is not None:
                            if sample_val > 1e12:  # Likely milliseconds
                                df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                            elif sample_val > 946684800:  # Likely seconds (after 2000-01-01)
                                df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
                            else:
                                # Try as milliseconds anyway (might be pre-2000 data in ms)
                                df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                        else:
                            # Index is all NaN - this is unusual, but don't fail yet
                            logger.warning("Index is all NaN - will try to find timestamp in columns")
                    else:
                        # Try to parse as datetime string
                        df.index = pd.to_datetime(df.index, errors='coerce')
                except Exception as e:
                    logger.warning(f"Failed to parse datetime index: {e}")
                    # Don't raise yet - we might find timestamp in columns
        
        # Also handle case where index name suggests it's a timestamp but index is not datetime
        elif index_name_suggests_datetime and not index_is_datetime:
            try:
                if pd.api.types.is_numeric_dtype(df.index):
                    sample_val = df.index[0] if len(df) > 0 and pd.notna(df.index[0]) else None
                    if sample_val and sample_val > 1e12:
                        df.index = pd.to_datetime(df.index, unit='ms', errors='coerce')
                    elif sample_val and sample_val > 946684800:
                        df.index = pd.to_datetime(df.index, unit='s', errors='coerce')
            except:
                pass
        
        # Standardize column names (case-insensitive)
        column_map = {}
        logger.debug(f"Columns before mapping: {df.columns.tolist()}")
        
        # Track which standard columns we've mapped
        mapped_standard_cols = set()
        
        # First pass: map OHLCV columns
        for col in df.columns:
            # Strip angle brackets, whitespace, and convert to lowercase
            # Handles formats like: <OPEN>, OPEN, open, etc.
            col_clean = col.strip().strip('<>').lower()
            logger.debug(f"Processing column: '{col}' -> cleaned: '{col_clean}'")
            
            if col_clean in ['open', 'o'] and 'open' not in mapped_standard_cols:
                column_map[col] = 'open'
                mapped_standard_cols.add('open')
                logger.debug(f"Mapped '{col}' -> 'open'")
            elif col_clean in ['high', 'h'] and 'high' not in mapped_standard_cols:
                column_map[col] = 'high'
                mapped_standard_cols.add('high')
            elif col_clean in ['low', 'l'] and 'low' not in mapped_standard_cols:
                column_map[col] = 'low'
                mapped_standard_cols.add('low')
            elif col_clean in ['close', 'c', 'closed'] and 'close' not in mapped_standard_cols:
                column_map[col] = 'close'
                mapped_standard_cols.add('close')
        
        # Second pass: map volume columns (after OHLC to avoid conflicts)
        for col in df.columns:
            col_clean = col.strip().strip('<>').lower()
            if col_clean in ['volume', 'vol', 'v'] and 'volume' not in mapped_standard_cols:
                # Map volume columns (prefer VOL over TICKVOL)
                column_map[col] = 'volume'
                mapped_standard_cols.add('volume')
                break  # Only map first volume column
        
        # Third pass: TICKVOL as fallback
        if 'volume' not in mapped_standard_cols:
            for col in df.columns:
                col_clean = col.strip().strip('<>').lower()
                if col_clean == 'tickvol':
                    column_map[col] = 'volume'
                    mapped_standard_cols.add('volume')
                    break
            # Ignore other columns like SPREAD, DATE, TIME (already handled)
        
        logger.debug(f"Column mapping: {column_map}")
        logger.debug(f"Mapped standard columns: {mapped_standard_cols}")
        if column_map:
            df = df.rename(columns=column_map)
        logger.debug(f"Columns after mapping: {df.columns.tolist()}")
        
        # Check what we have before trying to derive missing columns
        available_after_mapping = set(df.columns)
        logger.debug(f"Available columns after mapping: {available_after_mapping}")
        
        # Handle missing CLOSE column (common in some forex data formats)
        # For forex, CLOSE is typically the last price of the bar
        # We can derive it from the next bar's OPEN, or use (HIGH+LOW)/2 as estimate
        # Note: This check happens AFTER column mapping, so we check for 'close' (lowercase)
        if 'close' not in df.columns:
            logger.warning(f"CLOSE column not found after mapping. Available columns: {df.columns.tolist()}")
            logger.warning("Deriving CLOSE from other columns.")
            if 'open' in df.columns and len(df) > 1:
                # Use next bar's OPEN as current bar's CLOSE (common in forex)
                df['close'] = df['open'].shift(-1)
                # For last bar, use (HIGH+LOW)/2 as estimate, or HIGH if LOW missing
                if 'high' in df.columns and 'low' in df.columns:
                    df.loc[df.index[-1], 'close'] = (df.loc[df.index[-1], 'high'] + df.loc[df.index[-1], 'low']) / 2
                elif 'high' in df.columns:
                    df.loc[df.index[-1], 'close'] = df.loc[df.index[-1], 'high']
                else:
                    df.loc[df.index[-1], 'close'] = df.loc[df.index[-1], 'open']
            elif 'high' in df.columns and 'low' in df.columns:
                # Fallback: use (HIGH+LOW)/2 as estimate
                df['close'] = (df['high'] + df['low']) / 2
            elif 'high' in df.columns:
                # Use HIGH as CLOSE
                df['close'] = df['high']
            elif 'open' in df.columns:
                # Last resort: use OPEN
                df['close'] = df['open']
            else:
                raise ValueError("Cannot derive CLOSE column: missing OPEN, HIGH, and LOW columns")
        
        # Handle missing volume column (some data sources don't have volume)
        if 'volume' not in df.columns:
            logger.warning("VOLUME column not found. Using default volume of 1.0.")
            df['volume'] = 1.0
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        missing = [col for col in required if col not in df.columns]
        if missing:
            file_type = "Parquet" if file_path.suffix.lower() == '.parquet' else "CSV"
            # Show what we tried to map
            logger.error(f"Missing required columns: {missing}")
            logger.error(f"Available columns after all processing: {list(df.columns)}")
            logger.error(f"Column mapping that was applied: {column_map if 'column_map' in locals() else 'N/A'}")
            raise ValueError(f"{file_type} file missing required columns: {missing}. "
                           f"Available columns: {list(df.columns)}. "
                           f"Please check that your data file has OPEN, HIGH, LOW, CLOSE, and VOLUME columns.")
        
        # Select only required columns
        df = df[required]
        
        # Remove rows with NaT (Not a Time) in index
        if isinstance(df.index, pd.DatetimeIndex):
            initial_len = len(df)
            df = df[df.index.notna()]
            removed = initial_len - len(df)
            if removed > 0:
                removal_pct = (removed / initial_len) * 100
                logger.warning(f"Removed {removed} rows ({removal_pct:.1f}%) with invalid timestamps (NaT)")
                if removal_pct > 50:
                    logger.error(f"WARNING: More than 50% of data was removed! This suggests a data format issue.")
                    logger.error(f"Original shape: {initial_len} rows, Final shape: {len(df)} rows")
                    logger.error(f"Please check your data file format. Expected: timestamp/index + OHLCV columns")
        
        # Sort by index
        df = df.sort_index()
        
        # Final debug check
        if len(df) > 0:
            logger.debug(f"Final date range: {df.index[0]} to {df.index[-1]}")
        else:
            logger.warning("No valid data remaining after processing")
        
        return df


