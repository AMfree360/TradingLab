"""Account state management for backtesting.

This module defines AccountState, which tracks:
- cash: Available cash balance
- equity: Total account value (cash + unrealized_pnl)
- used_margin: Total margin locked in open positions
- free_margin: Available margin (equity - used_margin)
- open_positions: List of open positions
- equity_curve: Timestamped equity history
"""

from dataclasses import dataclass, field
from typing import List, Optional
import pandas as pd
from engine.broker import BrokerModel


@dataclass
class AccountState:
    """Account state tracking for backtesting.
    
    This class maintains the account balance and tracks all open positions.
    It enforces the fundamental accounting invariant:
        equity == cash + unrealized_pnl
    
    Attributes:
        cash: Available cash balance
        used_margin: Total margin locked in open positions
        unrealized_pnl: Total unrealized P&L from all open positions
        commission_paid: Total commission paid
        slippage_paid: Total slippage paid
        open_positions: List of open Position objects
        equity_curve: Timestamped equity history (Series with datetime index)
    """
    cash: float
    used_margin: float = 0.0
    unrealized_pnl: float = 0.0
    commission_paid: float = 0.0
    slippage_paid: float = 0.0
    open_positions: List = field(default_factory=list)
    equity_curve: Optional[pd.Series] = None
    
    @property
    def equity(self) -> float:
        """Calculate current equity.
        
        Equity = cash + unrealized_pnl
        
        This is the fundamental accounting invariant.
        
        Returns:
            Current account equity
        """
        return self.cash + self.unrealized_pnl
    
    @property
    def free_margin(self) -> float:
        """Calculate free margin.
        
        Free margin = equity - used_margin
        
        This is the amount of margin available for new positions.
        
        Returns:
            Free margin
        """
        return self.equity - self.used_margin
    
    def update_unrealized_pnl(
        self,
        broker: BrokerModel,
        current_price: float
    ) -> None:
        """Update unrealized P&L for all open positions.
        
        This method recalculates unrealized P&L for all open positions
        and updates the account state accordingly.
        
        Args:
            broker: Broker model for P&L calculations
            current_price: Current market price
        """
        if not self.open_positions:
            self.unrealized_pnl = 0.0
            self.used_margin = 0.0
            return
        
        # Calculate total unrealized P&L and margin used
        total_unrealized = 0.0
        total_margin = 0.0
        
        for position in self.open_positions:
            # Calculate unrealized P&L for this position
            unrealized = broker.calculate_unrealized_pnl(
                position.entry_price,
                current_price,
                position.size,
                position.direction
            )
            total_unrealized += unrealized
            
            # Sum margin used
            total_margin += position.margin_used
        
        self.unrealized_pnl = total_unrealized
        self.used_margin = total_margin
    
    def validate_invariant(
        self,
        broker: BrokerModel,
        current_price: float
    ) -> bool:
        """Validate the accounting invariant.
        
        The invariant is: equity == cash + unrealized_pnl
        
        This must hold at all times. If it doesn't, there's a bug.
        
        Args:
            broker: Broker model for P&L calculations
            current_price: Current market price
            
        Returns:
            True if invariant holds, False otherwise
        """
        self.update_unrealized_pnl(broker, current_price)
        expected_equity = self.cash + self.unrealized_pnl
        actual_equity = self.equity
        
        # Allow small floating point errors
        return abs(expected_equity - actual_equity) < 1e-6
    
    def record_equity(self, timestamp: pd.Timestamp) -> None:
        """Record current equity in equity curve.
        
        Args:
            timestamp: Current timestamp
        """
        equity = self.equity
        
        if self.equity_curve is None:
            self.equity_curve = pd.Series([equity], index=[timestamp])
        else:
            # Append to existing series
            self.equity_curve = pd.concat([
                self.equity_curve,
                pd.Series([equity], index=[timestamp])
            ])
    
    def get_equity_curve(self) -> pd.Series:
        """Get equity curve as a pandas Series.
        
        Returns:
            Series with datetime index and equity values
        """
        if self.equity_curve is None:
            return pd.Series(dtype=float)
        return self.equity_curve.copy()

