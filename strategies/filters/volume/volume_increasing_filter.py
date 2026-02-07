"""Volume Increasing filter.

This filter checks if volume is increasing on both middle and higher timeframes.
Supports two modes: raw volume or volume oscillator.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
import pandas as pd
import numpy as np


class VolumeIncreasingFilter(FilterBase):
    """Volume Increasing filter for volume confirmation.
    
    This filter checks if volume is increasing on both middle and higher timeframes.
    Supports two modes:
    
    1. Raw Volume Mode: Checks if raw volume[0] > volume[1] on both timeframes
    2. Volume Oscillator Mode: Calculates VolumeOscillator = EMA(vol, fast) - EMA(vol, slow),
       then checks if volOsc[0] > volOsc[1] on both timeframes
    
    Both modes require volume to be increasing on both middle and higher timeframes.
    """
    
    def __init__(self, config):
        """
        Initialize Volume Increasing filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - mode: str ("raw_volume" or "volume_oscillator", default "volume_oscillator")
                - middle_tf: str (middle timeframe label, default "5m")
                - higher_tf: str (higher timeframe label, default "15m")
                - oscillator_fast: int (fast EMA period for oscillator, default 15)
                - oscillator_slow: int (slow EMA period for oscillator, default 30)
        """
        super().__init__(config)
        self.mode = getattr(config, 'mode', None) or config.get('mode', 'volume_oscillator')
        self.middle_tf = getattr(config, 'middle_tf', None) or config.get('middle_tf', '5m')
        self.higher_tf = getattr(config, 'higher_tf', None) or config.get('higher_tf', '15m')
        self.oscillator_fast = getattr(config, 'oscillator_fast', None) or config.get('oscillator_fast', 15)
        self.oscillator_slow = getattr(config, 'oscillator_slow', None) or config.get('oscillator_slow', 30)
        
        if self.mode not in ['raw_volume', 'volume_oscillator']:
            raise ValueError(f"Invalid volume mode: {self.mode}. Must be 'raw_volume' or 'volume_oscillator'")
    
    def _calculate_volume_oscillator(self, volume_series: pd.Series) -> pd.Series:
        """
        Calculate Volume Oscillator = EMA(volume, fast) - EMA(volume, slow).
        
        Args:
            volume_series: Series of volume values
            
        Returns:
            Series of volume oscillator values
        """
        ema_fast = volume_series.ewm(span=self.oscillator_fast, adjust=False).mean()
        ema_slow = volume_series.ewm(span=self.oscillator_slow, adjust=False).mean()
        return ema_fast - ema_slow
    
    def _check_volume_increasing(self, df: pd.DataFrame, tf_label: str, timestamp: pd.Timestamp = None) -> tuple[bool, str]:
        """
        Check if volume is increasing for a timeframe.
        
        Args:
            df: DataFrame with volume data
            tf_label: Timeframe label for error messages
            timestamp: Optional timestamp to check at (if None, uses last bar)
            
        Returns:
            Tuple of (is_increasing, error_message)
        """
        # Filter to bars up to and including the timestamp (if provided)
        if timestamp is not None:
            df = df[df.index <= timestamp]
            if len(df) < 2:
                return False, f"Not enough data for {tf_label} at timestamp {timestamp}"
        if self.mode == 'raw_volume':
            # Check raw volume
            if 'volume' not in df.columns:
                return False, f"Volume column not found for {tf_label}"
            
            volume_series = df['volume'].dropna()
            if len(volume_series) < 2:
                return False, f"Not enough volume data for {tf_label}"
            
            # Check if current > previous
            current = float(volume_series.iloc[-1])
            previous = float(volume_series.iloc[-2])
            is_increasing = current > previous
            
            return is_increasing, None if is_increasing else f"Raw volume not increasing: {current:.2f} <= {previous:.2f}"
        
        else:  # volume_oscillator
            # Calculate volume oscillator
            if 'volume' not in df.columns:
                return False, f"Volume column not found for {tf_label}"
            
            volume_series = df['volume'].dropna()
            if len(volume_series) < self.oscillator_slow + 1:
                return False, f"Not enough volume data for oscillator calculation on {tf_label} (need {self.oscillator_slow + 1} bars)"
            
            # Calculate volume oscillator
            vol_osc = self._calculate_volume_oscillator(volume_series)
            vol_osc_clean = vol_osc.dropna()
            
            if len(vol_osc_clean) < 2:
                return False, f"Not enough volume oscillator data for {tf_label}"
            
            # Check if current > previous
            # For volume oscillator, "increasing" means the value is becoming more positive
            # (or less negative). So we check if current > previous regardless of sign.
            current = float(vol_osc_clean.iloc[-1])
            previous = float(vol_osc_clean.iloc[-2])
            is_increasing = current > previous
            
            if is_increasing:
                return True, None
            else:
                # Check if it's actually increasing in magnitude (for negative values)
                # If both are negative, check if current is less negative (closer to zero)
                if current < 0 and previous < 0:
                    # Both negative - check if current is less negative (more positive)
                    is_increasing = current > previous  # This is already checked above
                    if not is_increasing:
                        return False, f"Volume oscillator not increasing: {current:.4f} <= {previous:.4f} (both negative)"
                return False, f"Volume oscillator not increasing: {current:.4f} <= {previous:.4f}"
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if volume is increasing on both middle and higher timeframes.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if volume is increasing
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get dataframes for middle and higher timeframes
        middle_df = context.df_by_tf.get(self.middle_tf)
        higher_df = context.df_by_tf.get(self.higher_tf)
        
        if middle_df is None or higher_df is None:
            return self._create_fail_result(
                reason=f"Missing timeframe data: middle={self.middle_tf}, higher={self.higher_tf}",
                metadata={
                    'filter': 'volume_increasing',
                    'mode': self.mode,
                    'available_tfs': list(context.df_by_tf.keys()),
                    'symbol': context.symbol
                }
            )
        
        # Check volume increasing on middle timeframe (at signal timestamp)
        middle_ok, middle_error = self._check_volume_increasing(middle_df, self.middle_tf, context.timestamp)
        if not middle_ok:
            return self._create_fail_result(
                reason=f"Middle TF ({self.middle_tf}) volume not increasing: {middle_error}",
                metadata={
                    'filter': 'volume_increasing',
                    'mode': self.mode,
                    'timeframe': self.middle_tf,
                    'error': middle_error
                }
            )
        
        # Check volume increasing on higher timeframe (at signal timestamp)
        higher_ok, higher_error = self._check_volume_increasing(higher_df, self.higher_tf, context.timestamp)
        if not higher_ok:
            return self._create_fail_result(
                reason=f"Higher TF ({self.higher_tf}) volume not increasing: {higher_error}",
                metadata={
                    'filter': 'volume_increasing',
                    'mode': self.mode,
                    'timeframe': self.higher_tf,
                    'error': higher_error
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'volume_increasing',
                'mode': self.mode,
                'middle_tf': self.middle_tf,
                'higher_tf': self.higher_tf
            }
        )

