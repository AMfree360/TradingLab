"""ADX (Average Directional Index) filter for trend strength.

This filter checks if ADX meets minimum threshold for the asset class.
Works for all asset classes by using MarketSpec to determine appropriate thresholds.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
import pandas as pd


class ADXFilter(FilterBase):
    """ADX filter for trend strength detection.
    
    This filter checks if the ADX (Average Directional Index) value meets
    the minimum threshold for the asset class. Different asset classes have
    different minimum ADX requirements:
    
    - Forex: Typically 20-25
    - Indices: Typically 18-22
    - Metals: Typically 25-30
    - Commodities: Typically 26-30
    - Crypto: Typically 28-35
    
    The filter uses MarketSpec to determine the appropriate threshold based on
    the symbol's asset class.
    """
    
    def __init__(self, config):
        """
        Initialize ADX filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - period: int (ADX calculation period, default 14)
                - min_forex: float (minimum ADX for forex, default 23.0)
                - min_indices: float (minimum ADX for indices, default 20.0)
                - min_metals: float (minimum ADX for metals, default 25.0)
                - min_commodities: float (minimum ADX for commodities, default 26.0)
                - min_crypto: float (minimum ADX for crypto, default 28.0)
        """
        super().__init__(config)
        self.period = getattr(config, 'period', None) or config.get('period', 14)
        self.min_forex = getattr(config, 'min_forex', None) or config.get('min_forex', 23.0)
        self.min_indices = getattr(config, 'min_indices', None) or config.get('min_indices', 20.0)
        self.min_metals = getattr(config, 'min_metals', None) or config.get('min_metals', 25.0)
        self.min_commodities = getattr(config, 'min_commodities', None) or config.get('min_commodities', 26.0)
        self.min_crypto = getattr(config, 'min_crypto', None) or config.get('min_crypto', 28.0)
    
    def _get_min_adx_for_symbol(self, symbol: str) -> float:
        """
        Get minimum ADX threshold for symbol's asset class.
        
        This method determines the appropriate minimum ADX threshold based on
        the symbol name and asset class. It checks for common symbol patterns
        to identify asset classes.
        
        Args:
            symbol: Trading symbol (e.g., 'EURUSD', 'US500', 'XAUUSD')
            
        Returns:
            Minimum ADX threshold for this asset class
        """
        symbol_upper = symbol.upper()
        
        # Indices
        if any(x in symbol_upper for x in ['US500', 'US30', 'NAS100', 'SPX', 'DOW', 'NDX']):
            return self.min_indices
        
        # Metals
        elif any(x in symbol_upper for x in ['XAU', 'XAG', 'GOLD', 'SILVER']):
            return self.min_metals
        
        # Commodities
        elif any(x in symbol_upper for x in ['OIL', 'CRUDE', 'WTI', 'BRENT', 'NGAS', 'NATURAL']):
            return self.min_commodities
        
        # Crypto
        elif any(x in symbol_upper for x in ['BTC', 'ETH', 'LTC', 'CRYPTO', 'USDT', 'USDC']):
            return self.min_crypto
        
        # Default: Forex
        return self.min_forex
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if ADX meets minimum threshold for asset class.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult with pass/fail and reason
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ADX value from signal data
        if 'adx' not in context.signal_data or pd.isna(context.signal_data['adx']):
            return self._create_fail_result(
                reason="ADX value not available",
                metadata={
                    'filter': 'adx',
                    'value': None,
                    'symbol': context.symbol
                }
            )
        
        adx_value = float(context.signal_data['adx'])
        
        # Determine minimum ADX based on asset class
        min_adx = self._get_min_adx_for_symbol(context.symbol)
        
        # Also check MarketSpec asset_class if available (more reliable)
        if context.market_spec:
            if context.market_spec.asset_class == 'forex':
                min_adx = self.min_forex
            elif context.market_spec.asset_class == 'crypto':
                min_adx = self.min_crypto
            # For other asset classes, use symbol-based detection
        
        if adx_value < min_adx:
            return self._create_fail_result(
                reason=f"ADX {adx_value:.1f} below minimum {min_adx:.1f} for {context.symbol}",
                metadata={
                    'filter': 'adx',
                    'value': adx_value,
                    'threshold': min_adx,
                    'symbol': context.symbol,
                    'asset_class': context.market_spec.asset_class if context.market_spec else 'unknown'
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'adx',
                'value': adx_value,
                'threshold': min_adx,
                'symbol': context.symbol
            }
        )
