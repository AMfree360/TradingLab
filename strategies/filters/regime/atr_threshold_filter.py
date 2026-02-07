"""ATR (Average True Range) threshold filter.

This filter checks if ATR is within configured min/max thresholds.
Supports multiple unit types (pips, points, direct price units) via MarketSpec.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
from strategies.filters.utils import UnitConverter
from typing import Optional
import pandas as pd


class ATRThresholdFilter(FilterBase):
    """ATR threshold filter with min/max limits.
    
    This filter checks if the ATR value is within configured thresholds.
    Supports multiple unit types:
    - Direct price units: min_atr, max_atr
    - Pips (forex): min_atr_pips, max_atr_pips
    - Points (stocks/indices/futures): min_atr_points, max_atr_points
    
    The filter uses MarketSpec to determine the appropriate unit conversion.
    """
    
    def __init__(self, config):
        """
        Initialize ATR threshold filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - atr_period: int (default: 14)
                - min_atr: Optional[float] (direct price units)
                - max_atr: Optional[float] (direct price units)
                - min_atr_pips: Optional[float] (for forex)
                - max_atr_pips: Optional[float] (for forex)
                - min_atr_points: Optional[float] (for stocks/indices/futures)
                - max_atr_points: Optional[float] (for stocks/indices/futures)
        """
        super().__init__(config)
        self.atr_period = getattr(config, 'atr_period', None) or config.get('atr_period', 14)
        
        # Direct price units
        self.min_atr = getattr(config, 'min_atr', None) or config.get('min_atr', None)
        self.max_atr = getattr(config, 'max_atr', None) or config.get('max_atr', None)
        
        # Pips (forex)
        self.min_atr_pips = getattr(config, 'min_atr_pips', None) or config.get('min_atr_pips', None)
        self.max_atr_pips = getattr(config, 'max_atr_pips', None) or config.get('max_atr_pips', None)
        
        # Points (stocks/indices/futures)
        self.min_atr_points = getattr(config, 'min_atr_points', None) or config.get('min_atr_points', None)
        self.max_atr_points = getattr(config, 'max_atr_points', None) or config.get('max_atr_points', None)
    
    def _get_thresholds_in_price_units(self, context: FilterContext) -> tuple[Optional[float], Optional[float]]:
        """
        Get min/max thresholds in price units, converting from pips/points if needed.
        
        Args:
            context: FilterContext with MarketSpec
            
        Returns:
            Tuple of (min_threshold, max_threshold) in price units
        """
        min_threshold = None
        max_threshold = None
        
        # Start with direct price units
        if self.min_atr is not None:
            min_threshold = self.min_atr
        if self.max_atr is not None:
            max_threshold = self.max_atr
        
        # Convert pips/points if MarketSpec is available
        if context.market_spec:
            converter = UnitConverter(context.market_spec)
            
            # Convert pips (forex)
            if context.market_spec.asset_class == 'forex':
                if self.min_atr_pips is not None:
                    min_threshold = converter.pips_to_price(self.min_atr_pips)
                if self.max_atr_pips is not None:
                    max_threshold = converter.pips_to_price(self.max_atr_pips)
            
            # Convert points (stocks/indices/futures)
            elif context.market_spec.asset_class in ['stock', 'futures']:
                if self.min_atr_points is not None:
                    min_threshold = converter.points_to_price(self.min_atr_points)
                if self.max_atr_points is not None:
                    max_threshold = converter.points_to_price(self.max_atr_points)
            
            # For crypto, use direct price units or points if specified
            elif context.market_spec.asset_class == 'crypto':
                if self.min_atr_points is not None:
                    min_threshold = converter.points_to_price(self.min_atr_points)
                if self.max_atr_points is not None:
                    max_threshold = converter.points_to_price(self.max_atr_points)
        
        return min_threshold, max_threshold
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if ATR is within configured thresholds.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if ATR is within thresholds
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ATR value from signal data
        atr_key = f'atr_{self.atr_period}' if f'atr_{self.atr_period}' in context.signal_data else 'atr'
        
        if atr_key not in context.signal_data or pd.isna(context.signal_data[atr_key]):
            return self._create_fail_result(
                reason=f"ATR value not available (key: {atr_key})",
                metadata={
                    'filter': 'atr_threshold',
                    'value': None,
                    'symbol': context.symbol
                }
            )
        
        atr_value = float(context.signal_data[atr_key])
        
        # Get thresholds in price units
        min_threshold, max_threshold = self._get_thresholds_in_price_units(context)
        
        # Check minimum threshold
        if min_threshold is not None and atr_value < min_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} below minimum threshold {min_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': min_threshold,
                    'type': 'min'
                }
            )
        
        # Check maximum threshold
        if max_threshold is not None and atr_value > max_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} above maximum threshold {max_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': max_threshold,
                    'type': 'max'
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'atr_threshold',
                'value': atr_value,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold
            }
        )
