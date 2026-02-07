"""Market profile loader for market-specific settings."""

from pathlib import Path
from typing import Dict, Any, Optional
import yaml
from config.schema import MarketConfig, BacktestConfig


def load_market_profiles() -> Dict[str, Any]:
    """Load market profiles from YAML file."""
    profiles_path = Path(__file__).parent / "market_profiles.yml"
    if not profiles_path.exists():
        return {}
    
    with open(profiles_path, 'r') as f:
        return yaml.safe_load(f)


def get_market_profile(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Get market profile for a specific symbol.
    
    Supports futures variants (e.g., BTCUSDT_FUTURES) and will:
    1. Try exact symbol match first
    2. If symbol ends with _FUTURES, try to find it directly
    3. If not found and symbol doesn't end with _FUTURES, try the futures variant
    4. If not found and symbol ends with _FUTURES, try the base symbol (without _FUTURES)
    
    Args:
        symbol: Trading symbol (e.g., 'BTCUSDT', 'EURUSD', 'BTCUSDT_FUTURES')
    
    Returns:
        Market profile dict or None if not found
    """
    profiles = load_market_profiles()
    markets = profiles.get('markets', {})
    
    # First try exact symbol match
    market_profile = markets.get(symbol)
    
    # If user passed a futures variant (e.g., BTCUSDT_FUTURES), try to find it
    if market_profile is None and symbol.endswith('_FUTURES'):
        # Already tried exact match above, so try base symbol
        base_symbol = symbol.replace('_FUTURES', '')
        market_profile = markets.get(base_symbol)
    
    # If not found and symbol doesn't end with "_FUTURES", try futures variant
    if market_profile is None and not symbol.endswith('_FUTURES'):
        futures_symbol = f"{symbol}_FUTURES"
        market_profile = markets.get(futures_symbol)
    
    return market_profile


def get_asset_class_defaults(asset_class: str) -> Optional[Dict[str, Any]]:
    """
    Get default settings for an asset class.
    
    Args:
        asset_class: Asset class ('crypto', 'forex', 'stock')
    
    Returns:
        Default settings dict or None if not found
    """
    profiles = load_market_profiles()
    defaults = profiles.get('asset_class_defaults', {})
    return defaults.get(asset_class)


def apply_market_profile(
    strategy_config: Dict[str, Any],
    symbol: Optional[str] = None,
    market_override: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Apply market profile to strategy config.
    
    This allows the same strategy to work across different markets by:
    1. Loading market-specific settings from profiles
    2. Overriding strategy config with market settings
    3. Using asset class defaults if market not found
    
    Args:
        strategy_config: Strategy configuration dict
        symbol: Trading symbol to look up (if None, uses config's symbol)
        market_override: Optional dict to override market settings
    
    Returns:
        Updated strategy config with market settings applied
    """
    config = strategy_config.copy()
    
    # Determine symbol
    if symbol is None:
        symbol = config.get('market', {}).get('symbol')
    
    if symbol is None:
        return config  # Can't apply market profile without symbol
    
    # Get market profile
    market_profile = get_market_profile(symbol)
    
    # If market override provided, use it (takes precedence)
    if market_override:
        market_profile = market_override
    
    # If no profile found, try asset class defaults
    if market_profile is None:
        # Provide helpful error message for futures variants
        if symbol.endswith('_FUTURES'):
            base_symbol = symbol.replace('_FUTURES', '')
            # Check if base symbol exists (might be a typo)
            base_profile = get_market_profile(base_symbol)
            if base_profile is None:
                # Check if there's a similar futures profile (e.g., BTCUSDT_FUTURES vs BTCUSD_FUTURES)
                profiles = load_market_profiles()
                markets = profiles.get('markets', {})
                # Find futures profiles that might be what the user meant
                # Check if base symbol is a substring of any futures profile base symbol
                similar_futures = []
                for futures_key in markets.keys():
                    if futures_key.endswith('_FUTURES'):
                        futures_base = futures_key.replace('_FUTURES', '')
                        # Check if they're similar (one contains the other or vice versa)
                        if base_symbol in futures_base or futures_base in base_symbol:
                            similar_futures.append(futures_key)
                if similar_futures:
                    import warnings
                    warnings.warn(
                        f"Market profile not found for '{symbol}'. "
                        f"Did you mean one of: {', '.join(similar_futures)}?",
                        UserWarning
                    )
        # Try to infer asset class from symbol
        asset_class = None
        if 'USD' in symbol and len(symbol) == 6:  # Forex pairs like EURUSD
            asset_class = 'forex'
        elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'USDT']):
            asset_class = 'crypto'
        
        if asset_class:
            market_profile = get_asset_class_defaults(asset_class)
    
    # Apply market profile to config
    if market_profile:
        # Update market config
        if 'market' not in config:
            config['market'] = {}
        
        # Update market settings (don't override if already set)
        for key, value in market_profile.items():
            if key in ['exchange', 'symbol', 'market_type', 'leverage']:
                # These go in market section
                if key not in config['market']:
                    config['market'][key] = value
            elif key in ['commission_rate', 'slippage_ticks']:
                # These go in backtest section
                if 'backtest' not in config:
                    config['backtest'] = {}
                if key == 'commission_rate':
                    if 'commissions' not in config['backtest']:
                        config['backtest']['commissions'] = value
                elif key == 'slippage_ticks':
                    if 'slippage_ticks' not in config['backtest']:
                        config['backtest']['slippage_ticks'] = value
    
    return config


def create_market_override_from_cli(
    symbol: Optional[str] = None,
    exchange: Optional[str] = None,
    market_type: Optional[str] = None,
    leverage: Optional[float] = None,
    commission: Optional[float] = None,
    slippage: Optional[float] = None
) -> Dict[str, Any]:
    """
    Create market override dict from CLI arguments.
    
    Args:
        symbol: Trading symbol
        exchange: Exchange name
        market_type: 'spot' or 'futures'
        leverage: Leverage multiplier
        commission: Commission rate
        slippage: Slippage in ticks
    
    Returns:
        Market override dict
    """
    override = {}
    
    if symbol:
        override['symbol'] = symbol
    if exchange:
        override['exchange'] = exchange
    if market_type:
        override['market_type'] = market_type
    if leverage is not None:
        override['leverage'] = leverage
    if commission is not None:
        override['commission_rate'] = commission
    if slippage is not None:
        override['slippage_ticks'] = slippage
    
    return override

