"""Exit condition resolution logic for handling simultaneous exits."""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ExitType(Enum):
    """Types of exit conditions."""
    HARD_STOP = "hard_stop"
    TRAILING_STOP = "trailing_stop"
    TAKE_PROFIT = "take_profit"
    PARTIAL_EXIT = "partial_exit"
    TIME_BASED = "time_based"
    EMA_ALIGNMENT = "ema_alignment"


@dataclass
class ExitCondition:
    """Represents a potential exit condition.
    
    Attributes:
        exit_type: Type of exit (hard_stop, trailing_stop, take_profit, etc.)
        exit_price: Price at which exit would occur
        pnl: P&L if this exit is taken (can be negative for losses)
        r_multiple: R multiple for this exit (positive for profits, negative for losses)
        exit_pct: Percentage of position to exit (for partials)
        priority: Priority level (lower = higher priority, hard stops = 0)
    """
    exit_type: ExitType
    exit_price: float
    pnl: float
    r_multiple: float
    exit_pct: float = 100.0  # 100 = full exit
    priority: int = 10  # Default priority
    
    def __lt__(self, other):
        """Compare for sorting: most profitable for gains, smallest loss for losses."""
        # Hard stops always have highest priority (lowest number)
        if self.priority != other.priority:
            return self.priority < other.priority
        
        # For losses: smallest loss (most negative = worst, but we want smallest absolute loss)
        if self.pnl < 0 and other.pnl < 0:
            return self.pnl > other.pnl  # More negative = worse, so reverse
        
        # For profits: biggest gain
        if self.pnl > 0 and other.pnl > 0:
            return self.pnl > other.pnl
        
        # Mixed: prefer profit over loss
        if self.pnl > 0 and other.pnl <= 0:
            return False  # Profit is better
        if self.pnl <= 0 and other.pnl > 0:
            return True  # Loss is worse
        
        return False


class ExitResolver:
    """Resolves multiple simultaneous exit conditions.
    
    Industry standard priority:
    1. Hard stops (safety) - but use trailing if it provides better exit
    2. For profitable exits: Most profitable (highest R)
    3. For loss exits: Smallest loss (smallest absolute R)
    """
    
    @staticmethod
    def resolve(exits: List[ExitCondition]) -> Optional[ExitCondition]:
        """Resolve multiple simultaneous exit conditions.
        
        Args:
            exits: List of potential exit conditions
            
        Returns:
            Best exit condition to execute, or None if no valid exits
        """
        if not exits:
            return None
        
        # Filter out disabled exits
        valid_exits = [e for e in exits if e.exit_price > 0]
        if not valid_exits:
            return None
        
        # Separate by profit/loss
        profitable = [e for e in valid_exits if e.pnl > 0]
        losses = [e for e in valid_exits if e.pnl <= 0]
        
        # Industry standard: For losses, take smallest loss (best trailing stop if better than hard stop)
        if losses:
            # Sort by P&L (ascending = smallest loss first)
            losses_sorted = sorted(losses, key=lambda e: e.pnl, reverse=True)  # Most negative = worst
            # But we want smallest absolute loss, so reverse
            best_loss = losses_sorted[0]  # Smallest loss (least negative)
            
            # Check if there's a hard stop - it has priority unless trailing is better
            hard_stops = [e for e in losses if e.exit_type == ExitType.HARD_STOP]
            if hard_stops:
                hard_stop = hard_stops[0]
                # Use trailing if it's better (smaller loss)
                if best_loss.pnl > hard_stop.pnl:  # More positive = better (less negative)
                    return best_loss
                return hard_stop
            
            return best_loss
        
        # For profits: take most profitable
        if profitable:
            return max(profitable, key=lambda e: e.pnl)
        
        # Fallback: return first valid exit
        return valid_exits[0]
    
    @staticmethod
    def resolve_multiple_partials(exits: List[ExitCondition]) -> List[ExitCondition]:
        """Resolve multiple partial exit conditions that can execute simultaneously.
        
        Unlike full exits, partials can execute together (e.g., 30% at 1R, 50% at 2R).
        This method returns all partials that should execute, sorted by priority.
        
        Args:
            exits: List of potential partial exit conditions
            
        Returns:
            List of partial exits to execute, sorted by priority (highest first)
        """
        partials = [e for e in exits if e.exit_type == ExitType.PARTIAL_EXIT and e.exit_pct < 100.0]
        if not partials:
            return []
        
        # Sort by R multiple (ascending = execute lower R first)
        return sorted(partials, key=lambda e: e.r_multiple)

