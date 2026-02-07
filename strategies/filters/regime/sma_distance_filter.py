"""SMA Distance Quality filter.

This filter checks if price is meaningfully above/below SMA50 on both
middle and higher timeframes, measured relative to ATR.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
import pandas as pd
import numpy as np


class SMADistanceFilter(FilterBase):
    """SMA Distance Quality filter for trend confirmation.
    
    This filter checks if price is meaningfully above (for LONG) or below (for SHORT)
    the SMA50 on both middle and higher timeframes. The distance is measured relative
    to ATR to ensure it's meaningful regardless of asset class.
    
    For LONG signals:
    - Price must be above SMA50 on both timeframes
    - Distance from SMA50 must be >= min_distance_atr (in ATR units)
    
    For SHORT signals:
    - Price must be below SMA50 on both timeframes
    - Distance from SMA50 must be >= min_distance_atr (in ATR units)
    """
    
    def __init__(self, config):
        """
        Initialize SMA Distance filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - min_distance_atr: float (minimum distance in ATR units, default 0.5)
                - middle_tf: str (middle timeframe label, default "5m")
                - higher_tf: str (higher timeframe label, default "15m")
                - sma_period: int (SMA period, default 50)
        """
        super().__init__(config)
        self.min_distance_atr = getattr(config, 'min_distance_atr', None) or config.get('min_distance_atr', 0.5)
        self.middle_tf = getattr(config, 'middle_tf', None) or config.get('middle_tf', '5m')
        self.higher_tf = getattr(config, 'higher_tf', None) or config.get('higher_tf', '15m')
        self.sma_period = getattr(config, 'sma_period', None) or config.get('sma_period', 50)
    
    def _get_sma_key(self, tf: str) -> str:
        """Get SMA column key for timeframe."""
        return f'sma{self.sma_period}_{tf}'
    
    def _get_atr_value(self, context: FilterContext) -> float:
        """Get ATR value from signal data or calculate from entry timeframe."""
        # Try to get ATR from signal data
        atr_keys = ['atr', 'atr_14', 'atr14']
        for key in atr_keys:
            if key in context.signal_data and not pd.isna(context.signal_data[key]):
                return float(context.signal_data[key])
        
        # Try to get from entry timeframe data
        # Find entry timeframe (smallest timeframe)
        tfs = list(context.df_by_tf.keys())
        if not tfs:
            return None
        
        entry_tf = min(tfs, key=lambda x: self._parse_tf_minutes(x))
        
        if entry_tf in context.df_by_tf:
            df = context.df_by_tf[entry_tf]
            if 'atr' in df.columns:
                # Get most recent ATR value
                atr_series = df['atr'].dropna()
                if len(atr_series) > 0:
                    return float(atr_series.iloc[-1])
        
        return None
    
    def _parse_tf_minutes(self, tf: str) -> int:
        """Parse timeframe string to minutes for comparison."""
        tf_lower = tf.lower()
        if tf_lower.endswith('m'):
            return int(tf_lower[:-1])
        elif tf_lower.endswith('h'):
            return int(tf_lower[:-1]) * 60
        elif tf_lower.endswith('d'):
            return int(tf_lower[:-1]) * 1440
        return 999999  # Default to large value for unknown
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if price distance from SMA50 meets quality requirements.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if distance quality is met
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ATR value
        atr_value = self._get_atr_value(context)
        if atr_value is None or atr_value <= 0:
            return self._create_fail_result(
                reason="ATR value not available for distance calculation",
                metadata={
                    'filter': 'sma_distance',
                    'symbol': context.symbol
                }
            )
        
        # Get dataframes for middle and higher timeframes
        middle_df = context.df_by_tf.get(self.middle_tf)
        higher_df = context.df_by_tf.get(self.higher_tf)
        
        if middle_df is None or higher_df is None:
            return self._create_fail_result(
                reason=f"Missing timeframe data: middle={self.middle_tf}, higher={self.higher_tf}",
                metadata={
                    'filter': 'sma_distance',
                    'available_tfs': list(context.df_by_tf.keys()),
                    'symbol': context.symbol
                }
            )
        
        # Get current price (from signal data or most recent close)
        if 'close' in context.signal_data and not pd.isna(context.signal_data['close']):
            current_price = float(context.signal_data['close'])
        else:
            # Get from entry timeframe (smallest)
            entry_tf = min(context.df_by_tf.keys(), key=lambda x: self._parse_tf_minutes(x))
            entry_df = context.df_by_tf[entry_tf]
            if len(entry_df) == 0 or 'close' not in entry_df.columns:
                return self._create_fail_result(
                    reason="Cannot determine current price",
                    metadata={'filter': 'sma_distance'}
                )
            current_price = float(entry_df['close'].iloc[-1])
        
        # Get SMA values for both timeframes
        # Try to get from dataframe columns first
        middle_sma_key = self._get_sma_key(self.middle_tf)
        higher_sma_key = self._get_sma_key(self.higher_tf)
        
        # Check if SMA columns exist in dataframes
        if middle_sma_key in middle_df.columns:
            middle_sma = float(middle_df[middle_sma_key].iloc[-1])
        elif f'sma{self.sma_period}' in middle_df.columns:
            middle_sma = float(middle_df[f'sma{self.sma_period}'].iloc[-1])
        else:
            return self._create_fail_result(
                reason=f"SMA{self.sma_period} not found for middle timeframe {self.middle_tf}",
                metadata={
                    'filter': 'sma_distance',
                    'available_columns': list(middle_df.columns)
                }
            )
        
        if higher_sma_key in higher_df.columns:
            higher_sma = float(higher_df[higher_sma_key].iloc[-1])
        elif f'sma{self.sma_period}' in higher_df.columns:
            higher_sma = float(higher_df[f'sma{self.sma_period}'].iloc[-1])
        else:
            return self._create_fail_result(
                reason=f"SMA{self.sma_period} not found for higher timeframe {self.higher_tf}",
                metadata={
                    'filter': 'sma_distance',
                    'available_columns': list(higher_df.columns)
                }
            )
        
        # Calculate distances in ATR units
        distance_middle = abs(current_price - middle_sma) / atr_value
        distance_higher = abs(current_price - higher_sma) / atr_value
        
        # Check conditions based on direction
        direction = context.signal_direction  # 1=long, -1=short
        
        if direction == 1:  # LONG
            # Price must be above both SMAs and distance >= min
            price_above_middle = current_price > middle_sma
            price_above_higher = current_price > higher_sma
            distance_ok_middle = distance_middle >= self.min_distance_atr
            distance_ok_higher = distance_higher >= self.min_distance_atr
            
            if not (price_above_middle and price_above_higher):
                return self._create_fail_result(
                    reason=f"Price not above SMA50: middle={price_above_middle}, higher={price_above_higher}",
                    metadata={
                        'filter': 'sma_distance',
                        'price': current_price,
                        'middle_sma': middle_sma,
                        'higher_sma': higher_sma,
                        'direction': 'LONG'
                    }
                )
            
            if not (distance_ok_middle and distance_ok_higher):
                return self._create_fail_result(
                    reason=f"Distance from SMA50 too small: middle={distance_middle:.2f}ATR, higher={distance_higher:.2f}ATR (need {self.min_distance_atr:.2f}ATR)",
                    metadata={
                        'filter': 'sma_distance',
                        'distance_middle_atr': distance_middle,
                        'distance_higher_atr': distance_higher,
                        'min_distance_atr': self.min_distance_atr,
                        'direction': 'LONG'
                    }
                )
        
        else:  # SHORT
            # Price must be below both SMAs and distance >= min
            price_below_middle = current_price < middle_sma
            price_below_higher = current_price < higher_sma
            distance_ok_middle = distance_middle >= self.min_distance_atr
            distance_ok_higher = distance_higher >= self.min_distance_atr
            
            if not (price_below_middle and price_below_higher):
                return self._create_fail_result(
                    reason=f"Price not below SMA50: middle={price_below_middle}, higher={price_below_higher}",
                    metadata={
                        'filter': 'sma_distance',
                        'price': current_price,
                        'middle_sma': middle_sma,
                        'higher_sma': higher_sma,
                        'direction': 'SHORT'
                    }
                )
            
            if not (distance_ok_middle and distance_ok_higher):
                return self._create_fail_result(
                    reason=f"Distance from SMA50 too small: middle={distance_middle:.2f}ATR, higher={distance_higher:.2f}ATR (need {self.min_distance_atr:.2f}ATR)",
                    metadata={
                        'filter': 'sma_distance',
                        'distance_middle_atr': distance_middle,
                        'distance_higher_atr': distance_higher,
                        'min_distance_atr': self.min_distance_atr,
                        'direction': 'SHORT'
                    }
                )
        
        return self._create_pass_result(
            metadata={
                'filter': 'sma_distance',
                'distance_middle_atr': distance_middle,
                'distance_higher_atr': distance_higher,
                'min_distance_atr': self.min_distance_atr,
                'direction': 'LONG' if direction == 1 else 'SHORT'
            }
        )

