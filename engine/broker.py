"""Broker abstraction layer for margin and position validation.

The BrokerModel handles:
- Margin calculations (using MarketSpec)
- Position validation (can we afford this trade?)
- Commission & slippage application
- Realized / unrealized P&L calculations
- Margin calls & forced liquidation logic

Key principle: The broker enforces constraints, the strategy doesn't know about them.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from engine.market import MarketSpec


@dataclass
class BrokerModel:
    """Broker abstraction layer for margin and position validation.
    
    The BrokerModel handles:
    - Margin calculations (using MarketSpec)
    - Position validation (can we afford this trade?)
    - Commission & slippage application
    - Realized / unrealized P&L calculations
    - Margin calls & forced liquidation logic
    
    Key principle: The broker enforces constraints, the strategy doesn't know about them.
    
    Attributes:
        market_spec: Market specification for this instrument
        margin_call_level: Margin call threshold (equity < margin_used * level)
    """
    market_spec: MarketSpec
    margin_call_level: float = 1.0  # Margin call when equity < margin_used * margin_call_level
    
    def calculate_margin_required(self, entry_price: float, quantity: float, is_intraday: bool = True) -> float:
        """Calculate margin required for a position.
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            is_intraday: True for intraday/day trading (uses lower margin for futures)
            
        Returns:
            Margin required
        """
        return self.market_spec.calculate_margin(entry_price, quantity, is_intraday=is_intraday)
    
    def can_afford_position(
        self,
        entry_price: float,
        quantity: float,
        available_cash: float,
        commission_rate: Optional[float] = None,
        available_margin: Optional[float] = None
    ) -> Tuple[bool, float]:
        """Check if account can afford a position.
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            available_cash: Available cash in account (for commission)
            commission_rate: Commission rate (uses market_spec if None)
            available_margin: Available margin (for futures, uses available_cash if None)
            
        Returns:
            Tuple of (can_afford, required_cash)
            required_cash = margin + commission
        """
        margin = self.calculate_margin_required(entry_price, quantity, is_intraday=True)
        # Calculate commission (use fixed per-contract for futures, percentage for others)
        commission = self.calculate_commission(entry_price, quantity, commission_rate)
        required_cash = margin + commission
        
        # For futures, check margin against free_margin and commission against cash separately
        if self.market_spec.asset_class == 'futures' and available_margin is not None:
            can_afford = margin <= available_margin and commission <= available_cash
        else:
            can_afford = required_cash <= available_cash
        
        return can_afford, required_cash
    
    def check_margin_call(
        self,
        equity: float,
        margin_used: float,
        maintenance_margin: Optional[float] = None
    ) -> bool:
        """Check if margin call should be triggered.
        
        For Binance-style futures: Uses maintenance margin (equity < maintenance_margin)
        For traditional futures/other: Uses margin_call_level (equity < margin_used * margin_call_level)
        
        CRITICAL: Also trigger if equity <= 0 to prevent negative equity.
        With leverage, unrealized P&L can exceed margin, causing equity to go negative
        even if margin_level > 1.0. We must prevent this.
        
        Args:
            equity: Current account equity (cash + unrealized_pnl)
            margin_used: Total margin used by all open positions
            maintenance_margin: Optional maintenance margin (for Binance-style futures)
            
        Returns:
            True if margin call should be triggered
        """
        if margin_used <= 0:
            return False
        
        # CRITICAL: Prevent negative equity
        if equity <= 0:
            return True
        
        # For Binance-style futures with maintenance margin rate
        if maintenance_margin is not None and maintenance_margin > 0:
            # Liquidation occurs when equity < maintenance margin
            return equity < maintenance_margin
        
        # Standard margin call: equity < margin_used * margin_call_level
        margin_level = equity / margin_used if margin_used > 0 else float('inf')
        return margin_level < self.margin_call_level
    
    def calculate_maintenance_margin_required(
        self,
        entry_price: float,
        quantity: float,
        current_price: Optional[float] = None
    ) -> float:
        """Calculate maintenance margin required for a position.
        
        This is a convenience method that delegates to MarketSpec.
        
        Args:
            entry_price: Entry price
            quantity: Position quantity
            current_price: Current price (for mark-to-market, uses entry if None)
            
        Returns:
            Maintenance margin required
        """
        return self.market_spec.calculate_maintenance_margin(
            entry_price, quantity, current_price
        )
    
    def adjust_quantity_for_cash(
        self,
        entry_price: float,
        desired_quantity: float,
        available_cash: float,
        commission_rate: Optional[float] = None,
        available_margin: Optional[float] = None
    ) -> float:
        """Adjust quantity to fit available cash and margin.
        
        For futures: checks margin against free_margin and commission against cash separately
        For others: checks total cost (margin + commission) against cash
        
        Args:
            entry_price: Entry price
            desired_quantity: Desired position quantity
            available_cash: Available cash (for commission)
            commission_rate: Commission rate (uses market_spec if None, ignored for futures)
            available_margin: Available margin (for futures, uses available_cash if None)
            
        Returns:
            Adjusted quantity that fits available cash/margin
        """
        # For futures, check margin and commission separately
        if self.market_spec.asset_class == 'futures':
            if available_margin is None:
                available_margin = available_cash
            
            # Calculate max quantity based on margin constraint
            margin_per_contract = self.calculate_margin_required(entry_price, 1.0, is_intraday=True)
            if margin_per_contract > 0:
                max_qty_by_margin = available_margin / margin_per_contract
            else:
                max_qty_by_margin = float('inf')
            
            # Calculate max quantity based on commission constraint
            if self.market_spec.commission_per_contract is not None:
                commission_per_contract = self.market_spec.commission_per_contract
                if commission_per_contract > 0:
                    max_qty_by_commission = available_cash / commission_per_contract
                else:
                    max_qty_by_commission = float('inf')
            else:
                max_qty_by_commission = float('inf')
            
            # Take the minimum of both constraints
            max_affordable_qty = min(max_qty_by_margin, max_qty_by_commission)
        else:
            # For other asset classes with percentage commission
            commission_rate = commission_rate or self.market_spec.commission_rate
            # Solve: margin + commission = cash
            # (price * qty) / leverage + price * qty * commission_rate = cash
            # qty * price * (1/leverage + commission_rate) = cash
            # qty = cash / (price * (1/leverage + commission_rate))
            max_affordable_qty = available_cash / (
                entry_price * (1.0 / self.market_spec.leverage + commission_rate)
            )
        
        return min(desired_quantity, max_affordable_qty)
    
    def apply_slippage(self, price: float, quantity: float, is_entry: bool = True) -> float:
        """Apply slippage to a price.
        
        Args:
            price: Base price
            quantity: Position quantity
            is_entry: True for entry, False for exit
            
        Returns:
            Price adjusted for slippage
        """
        # For FX, slippage_ticks is specified in pips, so convert to price units.
        # For other asset classes, treat slippage_ticks as absolute price ticks.
        slippage = self._slippage_price_units()
        if is_entry:
            # Entry: buy at higher price, sell at lower price
            return price + slippage if quantity > 0 else price - slippage
        else:
            # Exit: sell at lower price, buy back at higher price
            return price - slippage if quantity > 0 else price + slippage

    def _slippage_price_units(self) -> float:
        """Convert configured slippage_ticks into price units."""
        if self.market_spec.asset_class == 'forex':
            pip_size = self.market_spec.pip_value or 0.0001
            return self.market_spec.slippage_ticks * pip_size
        return self.market_spec.slippage_ticks
    
    def calculate_commission(
        self,
        price: float,
        quantity: float,
        commission_rate: Optional[float] = None
    ) -> float:
        """Calculate commission for a trade.
        
        For futures with fixed per-contract commission, uses commission_per_contract.
        For other asset classes, uses percentage-based commission_rate.
        
        Args:
            price: Trade price
            quantity: Trade quantity
            commission_rate: Commission rate (uses market_spec if None)
            
        Returns:
            Commission amount
        """
        # For futures, use fixed per-contract commission if available
        if self.market_spec.asset_class == 'futures' and self.market_spec.commission_per_contract is not None:
            return abs(quantity) * self.market_spec.commission_per_contract
        
        # For other asset classes, use percentage-based commission
        commission_rate = commission_rate or self.market_spec.commission_rate
        return price * quantity * commission_rate
    
    def calculate_unrealized_pnl(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        direction: str
    ) -> float:
        """Calculate unrealized P&L for an open position.
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            quantity: Position quantity
            direction: 'long' or 'short'
            
        Returns:
            Unrealized P&L
        """
        return self.market_spec.calculate_unrealized_pnl(
            entry_price, current_price, quantity, direction
        )
    
    def calculate_realized_pnl(
        self,
        entry_price: float,
        exit_price: float,
        quantity: float,
        direction: str
    ) -> float:
        """Calculate realized P&L for a closed position.
        
        Args:
            entry_price: Entry price
            exit_price: Exit price
            quantity: Position quantity
            direction: 'long' or 'short'
            
        Returns:
            Realized P&L
        """
        return self.market_spec.calculate_realized_pnl(
            entry_price, exit_price, quantity, direction
        )

