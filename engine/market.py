"""Market specification for instrument-specific trading rules.

This module defines MarketSpec, which encapsulates all market-specific parameters
that affect position sizing, margin requirements, and P&L calculations.

Key principle: Leverage only affects margin, NOT P&L.
P&L is always calculated as: (price_change) * quantity
"""

from dataclasses import dataclass
from typing import Optional, Literal
from pathlib import Path
import yaml


@dataclass
class MarketSpec:
    """Market specification defining market-specific trading rules.
    
    This class encapsulates all market-specific parameters that affect
    position sizing, margin requirements, and P&L calculations.
    
    Key principle: Leverage only affects margin, NOT P&L.
    P&L is always calculated as: (price_change) * quantity
    
    Attributes:
        symbol: Trading symbol (e.g., 'EURUSD', 'BTCUSDT')
        exchange: Exchange name (e.g., 'oanda', 'binance')
        asset_class: Asset class ('forex', 'crypto', 'stock', 'futures')
        market_type: 'spot' or 'futures'
        leverage: Leverage multiplier (1.0 = no leverage, 100.0 = 100:1)
        contract_size: Standard contract size (e.g., 100000 for forex lots)
        pip_value: Price value of one pip (e.g., 0.0001 for most FX pairs)
        tick_value: Price value of one tick (for futures)
        min_trade_size: Minimum trade size in base currency
        lot_step: Lot size increment (e.g., 0.01 for forex)
        price_precision: Decimal places for price display
        quantity_precision: Decimal places for quantity display
        commission_rate: Commission rate per trade (e.g., 0.0004 = 0.04%)
        slippage_ticks: Expected slippage in price units
    """
    symbol: str
    exchange: str
    asset_class: Literal["forex", "crypto", "stock", "futures"]
    market_type: Literal["spot", "futures"] = "spot"
    leverage: float = 1.0
    contract_size: Optional[float] = None  # For forex/futures
    pip_value: Optional[float] = None  # For forex (e.g., 0.0001)
    tick_value: Optional[float] = None  # For futures
    min_trade_size: float = 0.01
    lot_step: float = 0.01  # Lot size increment
    price_precision: int = 5
    quantity_precision: int = 2
    commission_rate: float = 0.0004
    commission_per_contract: Optional[float] = None  # Fixed commission per contract (for futures)
    slippage_ticks: float = 0.0
    initial_margin_per_contract: Optional[float] = None  # Initial margin per contract (for futures, overnight)
    intraday_margin_per_contract: Optional[float] = None  # Day trading margin per contract (for futures, intraday)
    maintenance_margin_rate: Optional[float] = None  # Maintenance margin rate (e.g., 0.005 for 0.5%) for Binance-style futures
    margin_mode: Literal["isolated", "cross"] = "cross"  # Margin mode: isolated (per position) or cross (account-wide)
    
    def calculate_margin(self, entry_price: float, quantity: float, is_intraday: bool = True) -> float:
        """Calculate margin required for a position.
        
        Supports two margin systems:
        1. Traditional futures: Fixed margin per contract (CME-style)
        2. Binance-style futures: Leverage-based margin (notional / leverage)
        
        For other assets: Uses (entry_price * quantity) / leverage
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            is_intraday: True for intraday/day trading (uses lower margin for traditional futures)
            
        Returns:
            Margin required in cash units
        """
        # For futures, check if using fixed margin (traditional) or leverage-based (Binance-style)
        if self.asset_class == 'futures':
            # Traditional futures: Fixed margin per contract (CME, ICE, etc.)
            if self.initial_margin_per_contract is not None:
                if is_intraday and self.intraday_margin_per_contract is not None:
                    # Use day trading margin (lower)
                    return abs(quantity) * self.intraday_margin_per_contract
                else:
                    # Use initial margin (overnight)
                    return abs(quantity) * self.initial_margin_per_contract
            else:
                # Binance-style futures: Leverage-based margin
                # Margin = Notional Value / Leverage
                notional_value = entry_price * abs(quantity)
                margin = notional_value / self.leverage
                return margin
        
        # For other asset classes (forex, crypto spot, stocks), use leverage-based calculation
        notional_value = entry_price * abs(quantity)
        margin = notional_value / self.leverage
        return margin
    
    def calculate_maintenance_margin(
        self,
        entry_price: float,
        quantity: float,
        current_price: Optional[float] = None
    ) -> float:
        """Calculate maintenance margin required.
        
        Maintenance margin is the minimum equity required to keep a position open.
        If equity falls below maintenance margin, the position may be liquidated.
        
        For Binance-style futures: Uses maintenance_margin_rate (percentage of notional)
        For traditional futures: Typically same as initial margin (or could be lower)
        For other assets: Same as initial margin
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            current_price: Current price (for mark-to-market, uses entry if None)
            
        Returns:
            Maintenance margin required
        """
        price = current_price or entry_price
        notional_value = price * abs(quantity)
        
        if self.asset_class == 'futures':
            if self.maintenance_margin_rate is not None:
                # Binance-style: Percentage of notional value
                return notional_value * self.maintenance_margin_rate
            elif self.initial_margin_per_contract is not None:
                # Traditional futures: Use initial margin (maintenance typically same or lower)
                # For simplicity, we use initial margin as maintenance margin
                # In reality, maintenance might be lower, but this is conservative
                return abs(quantity) * self.initial_margin_per_contract
            else:
                # Fallback: Use initial margin calculation
                return self.calculate_margin(entry_price, quantity)
        
        # For other assets, maintenance margin = initial margin
        return self.calculate_margin(entry_price, quantity)
    
    def calculate_unrealized_pnl(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        direction: str
    ) -> float:
        """Calculate unrealized P&L for an open position.
        
        CRITICAL: P&L is independent of leverage.
        Leverage only affects margin, not profits or losses.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            quantity: Position quantity
            direction: 'long' or 'short'
            
        Returns:
            Unrealized P&L (positive = profit, negative = loss)
        """
        if direction == 'long':
            return (current_price - entry_price) * quantity
        else:  # short
            return (entry_price - current_price) * quantity
    
    def calculate_realized_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str
    ) -> float:
        """Calculate realized P&L for a closed position.
        
        CRITICAL: P&L is independent of leverage.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            direction: 'long' or 'short'
            
        Returns:
            Realized P&L (positive = profit, negative = loss)
        """
        return self.calculate_unrealized_pnl(entry_price, exit_price, quantity, direction)
    
    def calculate_pip_value_per_lot(self, entry_price: float) -> float:
        """Calculate pip value per standard lot for forex pairs.
        
        For major forex pairs where USD is the quote currency (EURUSD, GBPUSD):
        - 1 standard lot = 100,000 units
        - Pip value = $10 per standard lot
        
        For pairs where USD is base currency (USDJPY):
        - Pip value = (100,000 × 0.01) / entry_price = 1,000 / entry_price
        
        Args:
            entry_price: Current entry price
            
        Returns:
            Pip value per standard lot in USD
        """
        if self.asset_class != 'forex' or self.contract_size is None:
            return 0.0
        
        # For major pairs where USD is quote (EURUSD, GBPUSD, AUDUSD, etc.)
        if 'USD' in self.symbol[-3:]:  # USD is quote currency
            return 10.0  # $10 per pip per standard lot
        
        # For pairs where USD is base (USDJPY, USDCHF, etc.)
        if self.symbol.startswith('USD'):
            pip_size = self.pip_value or 0.0001
            if 'JPY' in self.symbol:  # JPY pairs use 0.01 for pip
                pip_size = 0.01
            return (self.contract_size * pip_size) / entry_price
        
        # Default: assume USD is quote
        return 10.0
    
    def calculate_pip_value_per_unit(self, entry_price: float) -> float:
        """Calculate pip value per unit (not per lot) for forex pairs.
        
        This is the fundamental value needed for position sizing.
        
        Formula: pip_value_per_unit = pip_value_per_lot / contract_size
        
        For EURUSD:
        - pip_value_per_lot = $10 per standard lot
        - contract_size = 100,000 units
        - pip_value_per_unit = $10 / 100,000 = $0.0001 per pip per unit
        
        Args:
            entry_price: Entry price (needed for USD base pairs)
            
        Returns:
            Pip value per unit in USD
        """
        if self.asset_class != 'forex' or self.contract_size is None:
            return 0.0
        
        pip_value_per_lot = self.calculate_pip_value_per_lot(entry_price)
        if pip_value_per_lot <= 0:
            return 0.0
        
        return pip_value_per_lot / self.contract_size
    
    def calculate_lot_size_from_risk(
        self,
        risk_amount: float,
        stop_loss_pips: float,
        entry_price: float
    ) -> float:
        """Calculate lot size based on risk amount and stop loss in pips.
        
        Formula: Lot Size = Risk Amount / (Stop Loss in Pips × Pip Value per Lot)
        
        Equivalent formula using units:
        Quantity = Risk Amount / (Stop Loss in Pips × Pip Value per Unit)
        Lot Size = Quantity / Contract Size
        
        This is the industry-standard way to calculate forex position sizes.
        Leverage does NOT affect this calculation - it only affects margin.
        
        Args:
            risk_amount: Amount to risk in USD (e.g., $50)
            stop_loss_pips: Stop loss distance in pips (e.g., 25)
            entry_price: Entry price (needed for USD base pairs)
            
        Returns:
            Lot size (e.g., 0.2 for 0.2 standard lots)
        """
        if self.asset_class != 'forex':
            return 0.0
        
        pip_value_per_lot = self.calculate_pip_value_per_lot(entry_price)
        if pip_value_per_lot <= 0 or stop_loss_pips <= 0:
            return 0.0
        
        # Formula: lot_size = risk_amount / (stop_loss_pips × pip_value_per_lot)
        lot_size = risk_amount / (stop_loss_pips * pip_value_per_lot)
        return lot_size
    
    def calculate_quantity_from_risk(
        self,
        risk_amount: float,
        stop_loss_pips: float,
        entry_price: float
    ) -> float:
        """Calculate position quantity (units) directly from risk amount and stop loss.
        
        This is an alternative to calculate_lot_size_from_risk that returns units directly.
        
        Formula: Quantity = Risk Amount / (Stop Loss in Pips × Pip Value per Unit)
        
        Where: pip_value_per_unit = pip_value_per_lot / contract_size
        
        Args:
            risk_amount: Amount to risk in USD (e.g., $50)
            stop_loss_pips: Stop loss distance in pips (e.g., 25)
            entry_price: Entry price (needed for USD base pairs)
            
        Returns:
            Position quantity in units
        """
        if self.asset_class != 'forex':
            return 0.0
        
        pip_value_per_unit = self.calculate_pip_value_per_unit(entry_price)
        if pip_value_per_unit <= 0 or stop_loss_pips <= 0:
            return 0.0
        
        # Formula: quantity = risk_amount / (stop_loss_pips × pip_value_per_unit)
        quantity = risk_amount / (stop_loss_pips * pip_value_per_unit)
        return quantity
    
    def lot_size_to_units(self, lot_size: float) -> float:
        """Convert lot size to units (quantity).
        
        For forex: 1 standard lot = contract_size units (typically 100,000)
        
        Args:
            lot_size: Lot size (e.g., 0.2)
            
        Returns:
            Quantity in units (e.g., 20,000 for 0.2 lots)
        """
        if self.asset_class == 'forex' and self.contract_size is not None:
            return lot_size * self.contract_size
        # For non-forex, lot_size is already in units
        return lot_size
    
    @classmethod
    def load_from_profiles(
        cls,
        symbol: str,
        profiles_path: Optional[Path] = None
    ) -> 'MarketSpec':
        """Load MarketSpec from market_profiles.yml.
        
        Args:
            symbol: Trading symbol
            profiles_path: Path to market_profiles.yml (defaults to config/market_profiles.yml)
            
        Returns:
            MarketSpec instance
            
        Raises:
            ValueError: If symbol not found in profiles
        """
        if profiles_path is None:
            profiles_path = Path(__file__).parent.parent / "config" / "market_profiles.yml"
        
        if not profiles_path.exists():
            raise ValueError(f"Market profiles file not found: {profiles_path}")
        
        with open(profiles_path, 'r') as f:
            profiles = yaml.safe_load(f)
        
        markets = profiles.get('markets', {})
        asset_class_defaults = profiles.get('asset_class_defaults', {})
        
        # Get market profile
        # First try exact symbol match
        market_profile = markets.get(symbol)
        actual_symbol = symbol  # Track the actual symbol to use
        
        # If user passed a futures variant (e.g., BTCUSDT_FUTURES), use the profile's symbol field
        if market_profile and symbol.endswith('_FUTURES'):
            # User passed BTCUSDT_FUTURES and found the profile
            # Use the symbol from the profile (should be BTCUSDT, not BTCUSDT_FUTURES)
            actual_symbol = market_profile.get('symbol', symbol.replace('_FUTURES', ''))
        
        # If not found and symbol doesn't end with "_FUTURES", try futures variant
        if market_profile is None and not symbol.endswith('_FUTURES'):
            futures_symbol = f"{symbol}_FUTURES"
            if futures_symbol in markets:
                market_profile = markets[futures_symbol]
                # Keep the original symbol (e.g., BTCUSDT)
                actual_symbol = symbol
        
        if market_profile is None:
            # Try to infer asset class and use defaults
            asset_class = None
            if 'USD' in symbol and len(symbol) == 6:  # Forex pairs
                asset_class = 'forex'
            elif any(crypto in symbol for crypto in ['BTC', 'ETH', 'USDT']):
                asset_class = 'crypto'
            
            if asset_class and asset_class in asset_class_defaults:
                defaults = asset_class_defaults[asset_class]
                market_profile = defaults.copy()
                market_profile['asset_class'] = asset_class
            else:
                raise ValueError(
                    f"Market profile not found for symbol '{symbol}' and no asset class defaults available"
                )
        
        # Get asset class defaults for merging
        asset_class = market_profile.get('asset_class', 'crypto')
        defaults = asset_class_defaults.get(asset_class, {})
        
        # Merge with defaults
        merged = {}
        if defaults:
            merged.update(defaults)
        merged.update(market_profile)
        
        # Extract fields (use actual_symbol which may differ from lookup key)
        return cls(
            symbol=actual_symbol,
            exchange=merged.get('exchange', 'unknown'),
            asset_class=merged.get('asset_class', 'crypto'),
            market_type=merged.get('market_type', 'spot'),
            leverage=merged.get('leverage', 1.0),
            contract_size=merged.get('contract_size'),
            pip_value=merged.get('pip_value'),
            tick_value=merged.get('tick_value'),
            min_trade_size=merged.get('min_trade_size', 0.01),
            lot_step=merged.get('lot_step', 0.01),
            price_precision=merged.get('price_precision', 5),
            quantity_precision=merged.get('quantity_precision', 2),
            commission_rate=merged.get('commission_rate', 0.0004),
            commission_per_contract=merged.get('commission_per_contract'),
            slippage_ticks=merged.get('slippage_ticks', 0.0),
            initial_margin_per_contract=merged.get('initial_margin_per_contract'),
            intraday_margin_per_contract=merged.get('intraday_margin_per_contract'),
            maintenance_margin_rate=merged.get('maintenance_margin_rate'),
            margin_mode=merged.get('margin_mode', 'cross')
        )

