"""Utility classes for filter system, including unit conversion."""

from typing import Union, Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from engine.market import MarketSpec


class UnitConverter:
    """Helper class for unit conversions across asset classes.
    
    This class provides methods to convert between different unit types
    (pips, points, ticks, price units) based on the asset class defined
    in MarketSpec. This ensures filters work correctly for all asset classes.
    
    Example:
        converter = UnitConverter(market_spec)
        price_diff = converter.pips_to_price(10.0)  # 10 pips to price units
        pips = converter.price_to_pips(0.0010)  # Price units to pips
    """
    
    def __init__(self, market_spec: 'MarketSpec'):
        """
        Initialize unit converter with market specification.
        
        Args:
            market_spec: MarketSpec with unit definitions
        """
        self.market_spec = market_spec
    
    def pips_to_price(self, pips: float) -> float:
        """
        Convert pips to price units (forex only).
        
        Args:
            pips: Number of pips
            
        Returns:
            Price difference in price units
            
        Raises:
            ValueError: If asset class is not forex
        """
        if self.market_spec.asset_class != 'forex':
            raise ValueError(
                f"Pips not applicable for asset class: {self.market_spec.asset_class}. "
                f"Use points or price units instead."
            )
        
        pip_value = self.market_spec.pip_value or 0.0001
        return pips * pip_value
    
    def points_to_price(self, points: float) -> float:
        """
        Convert points to price units (stocks, indices, futures).
        
        Args:
            points: Number of points
            
        Returns:
            Price difference in price units
        """
        if self.market_spec.asset_class == 'stock':
            # Stocks: 1 point = 0.01
            return points * 0.01
        elif self.market_spec.asset_class == 'futures':
            # Futures: Use tick_value if available
            tick_value = getattr(self.market_spec, 'tick_value', None)
            if tick_value:
                return points * tick_value
            else:
                # Default: 1 point = 1.0 for futures
                return points * 1.0
        elif self.market_spec.asset_class == 'crypto':
            # For crypto, points might mean direct price units
            # Check if tick_value is defined
            tick_value = getattr(self.market_spec, 'tick_value', None)
            if tick_value:
                return points * tick_value
            else:
                # Default: 1 point = 1.0 price unit
                return points * 1.0
        else:
            # For forex, points don't apply (use pips instead)
            raise ValueError(
                f"Points not applicable for asset class: {self.market_spec.asset_class}. "
                f"For forex, use pips instead."
            )
    
    def price_to_pips(self, price_diff: float) -> float:
        """
        Convert price difference to pips (forex only).
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of pips
            
        Raises:
            ValueError: If asset class is not forex
        """
        if self.market_spec.asset_class != 'forex':
            raise ValueError(
                f"Pips not applicable for asset class: {self.market_spec.asset_class}"
            )
        
        pip_value = self.market_spec.pip_value or 0.0001
        if pip_value == 0:
            return 0.0
        return price_diff / pip_value
    
    def price_to_points(self, price_diff: float) -> float:
        """
        Convert price difference to points (stocks, indices, futures).
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of points
        """
        if self.market_spec.asset_class == 'stock':
            return price_diff / 0.01
        elif self.market_spec.asset_class == 'futures':
            tick_value = getattr(self.market_spec, 'tick_value', None) or 1.0
            if tick_value == 0:
                return 0.0
            return price_diff / tick_value
        else:
            tick_value = getattr(self.market_spec, 'tick_value', None)
            if tick_value:
                if tick_value == 0:
                    return 0.0
                return price_diff / tick_value
            else:
                return price_diff / 1.0
    
    def get_minimum_price_move(self) -> float:
        """
        Get minimum price move for this asset class.
        
        Returns:
            Minimum price increment (pip_value, tick_value, or 0.01 for stocks)
        """
        if self.market_spec.asset_class == 'forex':
            return self.market_spec.pip_value or 0.0001
        elif self.market_spec.asset_class == 'stock':
            return 0.01
        elif self.market_spec.asset_class == 'futures':
            tick_value = getattr(self.market_spec, 'tick_value', None)
            return tick_value or 1.0
        else:  # crypto
            # Crypto: Use price precision to determine minimum move
            precision = self.market_spec.price_precision
            return 10 ** (-precision)
    
    def normalize_threshold(self, threshold: Union[float, Dict]) -> float:
        """
        Normalize threshold to price units based on asset class.
        
        Accepts thresholds in multiple formats:
        - Direct price units: 0.001
        - Pips (forex): {"pips": 10}
        - Points (stocks/indices): {"points": 5}
        - Ticks (futures): {"ticks": 4}
        
        Args:
            threshold: Threshold value (float or dict with unit type)
            
        Returns:
            Threshold in price units
        """
        if isinstance(threshold, (int, float)):
            # Direct price units
            return float(threshold)
        
        if isinstance(threshold, dict):
            if 'pips' in threshold:
                return self.pips_to_price(threshold['pips'])
            elif 'points' in threshold:
                return self.points_to_price(threshold['points'])
            elif 'ticks' in threshold:
                return self.points_to_price(threshold['ticks'])  # Ticks same as points for futures
            else:
                # Default: assume price units
                return float(threshold.get('value', 0.0))
        
        return float(threshold)
