"""SMA Slope filter.

This filter checks if SMA50 is trending in the correct direction on both
middle and higher timeframes by measuring the slope (points per bar).
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
import pandas as pd
import numpy as np


class SMASlopeFilter(FilterBase):
    """SMA Slope filter for trend direction confirmation.
    
    This filter checks if the SMA50 is trending in the correct direction
    on both middle and higher timeframes by measuring the slope (points per bar).
    
    For LONG signals:
    - Both SMAs must be trending up (slope >= min_slope)
    
    For SHORT signals:
    - Both SMAs must be trending down (slope <= -min_slope)
    
    The slope is calculated over a configurable lookback period.
    """
    
    def __init__(self, config):
        """
        Initialize SMA Slope filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - min_slope: float (minimum slope in points per bar, default 0.1)
                - middle_tf: str (middle timeframe label, default "5m")
                - higher_tf: str (higher timeframe label, default "15m")
                - middle_lookback: int (lookback bars for middle TF, default 3)
                - higher_lookback: int (lookback bars for higher TF, default 2)
                - sma_period: int (SMA period, default 50)
        """
        super().__init__(config)
        self.min_slope = getattr(config, 'min_slope', None) or config.get('min_slope', 0.1)
        self.middle_tf = getattr(config, 'middle_tf', None) or config.get('middle_tf', '5m')
        self.higher_tf = getattr(config, 'higher_tf', None) or config.get('higher_tf', '15m')
        self.middle_lookback = getattr(config, 'middle_lookback', None) or config.get('middle_lookback', 3)
        self.higher_lookback = getattr(config, 'higher_lookback', None) or config.get('higher_lookback', 2)
        self.sma_period = getattr(config, 'sma_period', None) or config.get('sma_period', 50)
    
    def _get_sma_key(self, tf: str) -> str:
        """Get SMA column key for timeframe."""
        return f'sma{self.sma_period}_{tf}'
    
    def _calculate_slope(self, sma_series: pd.Series, lookback: int) -> float:
        """
        Calculate SMA slope (points per bar) over lookback period.
        
        Args:
            sma_series: Series of SMA values
            lookback: Number of bars to look back
            
        Returns:
            Slope in points per bar (positive = up, negative = down)
        """
        if len(sma_series) < lookback + 1:
            return 0.0
        
        # Get current and past values
        current = float(sma_series.iloc[-1])
        past = float(sma_series.iloc[-(lookback + 1)])
        
        # Calculate slope: (current - past) / lookback
        slope = (current - past) / lookback
        return slope
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if SMA50 slope meets trend direction requirements.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if slope quality is met
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
                    'filter': 'sma_slope',
                    'available_tfs': list(context.df_by_tf.keys()),
                    'symbol': context.symbol
                }
            )
        
        # Get SMA series for both timeframes
        middle_sma_key = self._get_sma_key(self.middle_tf)
        higher_sma_key = self._get_sma_key(self.higher_tf)
        
        # Check if SMA columns exist in dataframes
        if middle_sma_key in middle_df.columns:
            middle_sma_series = middle_df[middle_sma_key].dropna()
        elif f'sma{self.sma_period}' in middle_df.columns:
            middle_sma_series = middle_df[f'sma{self.sma_period}'].dropna()
        else:
            return self._create_fail_result(
                reason=f"SMA{self.sma_period} not found for middle timeframe {self.middle_tf}",
                metadata={
                    'filter': 'sma_slope',
                    'available_columns': list(middle_df.columns)
                }
            )
        
        if higher_sma_key in higher_df.columns:
            higher_sma_series = higher_df[higher_sma_key].dropna()
        elif f'sma{self.sma_period}' in higher_df.columns:
            higher_sma_series = higher_df[f'sma{self.sma_period}'].dropna()
        else:
            return self._create_fail_result(
                reason=f"SMA{self.sma_period} not found for higher timeframe {self.higher_tf}",
                metadata={
                    'filter': 'sma_slope',
                    'available_columns': list(higher_df.columns)
                }
            )
        
        # Check if we have enough data
        if len(middle_sma_series) < self.middle_lookback + 1:
            return self._create_fail_result(
                reason=f"Not enough data for middle TF slope calculation: need {self.middle_lookback + 1}, have {len(middle_sma_series)}",
                metadata={'filter': 'sma_slope'}
            )
        
        if len(higher_sma_series) < self.higher_lookback + 1:
            return self._create_fail_result(
                reason=f"Not enough data for higher TF slope calculation: need {self.higher_lookback + 1}, have {len(higher_sma_series)}",
                metadata={'filter': 'sma_slope'}
            )
        
        # Calculate slopes
        slope_middle = self._calculate_slope(middle_sma_series, self.middle_lookback)
        slope_higher = self._calculate_slope(higher_sma_series, self.higher_lookback)
        
        # Check conditions based on direction
        direction = context.signal_direction  # 1=long, -1=short
        
        if direction == 1:  # LONG
            # Both SMAs must be trending up (slope >= min_slope)
            middle_ok = slope_middle >= self.min_slope
            higher_ok = slope_higher >= self.min_slope
            
            if not (middle_ok and higher_ok):
                return self._create_fail_result(
                    reason=f"SMA slope too weak for LONG: middle={slope_middle:.4f}, higher={slope_higher:.4f} (need >= {self.min_slope:.4f})",
                    metadata={
                        'filter': 'sma_slope',
                        'slope_middle': slope_middle,
                        'slope_higher': slope_higher,
                        'min_slope': self.min_slope,
                        'direction': 'LONG'
                    }
                )
        
        else:  # SHORT
            # Both SMAs must be trending down (slope <= -min_slope)
            middle_ok = slope_middle <= -self.min_slope
            higher_ok = slope_higher <= -self.min_slope
            
            if not (middle_ok and higher_ok):
                return self._create_fail_result(
                    reason=f"SMA slope too weak for SHORT: middle={slope_middle:.4f}, higher={slope_higher:.4f} (need <= {-self.min_slope:.4f})",
                    metadata={
                        'filter': 'sma_slope',
                        'slope_middle': slope_middle,
                        'slope_higher': slope_higher,
                        'min_slope': self.min_slope,
                        'direction': 'SHORT'
                    }
                )
        
        return self._create_pass_result(
            metadata={
                'filter': 'sma_slope',
                'slope_middle': slope_middle,
                'slope_higher': slope_higher,
                'min_slope': self.min_slope,
                'direction': 'LONG' if direction == 1 else 'SHORT'
            }
        )

