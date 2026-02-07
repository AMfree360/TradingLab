"""Time blackout filter for blocking signal generation during specific time windows.

This filter blocks signal generation during configured time windows (UTC+00).
Unlike news filters which block trading around events, this filter directly
blocks signal generation during the specified time ranges.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
from typing import List, Dict, Optional
import pandas as pd
from datetime import time


class TimeBlackoutSlot:
    """Individual time blackout slot configuration."""
    
    def __init__(self, config: Dict):
        """
        Initialize time blackout slot.
        
        Args:
            config: Slot configuration dict with:
                - enabled: bool
                - name: str (optional)
                - start_time: str (format: "HH:MM" in UTC+00)
                - end_time: str (format: "HH:MM" in UTC+00)
                - days: List[str] (e.g., ["Mon", "Tue", "Wed", "Thu", "Fri"])
                - symbols: List[str] (optional, empty = all symbols)
        """
        self.enabled = config.get('enabled', False)
        self.name = config.get('name', '')
        self.start_time_str = config.get('start_time', '')
        self.end_time_str = config.get('end_time', '')
        self.days = config.get('days', [])
        self.symbols = config.get('symbols', [])
        
        # Parse times (UTC+00)
        if self.start_time_str:
            self.start_time = self._parse_time(self.start_time_str)
        else:
            self.start_time = None
            
        if self.end_time_str:
            self.end_time = self._parse_time(self.end_time_str)
        else:
            self.end_time = None
    
    def _parse_time(self, time_str: str) -> time:
        """
        Parse time string 'HH:MM' to time object.
        
        Args:
            time_str: Time string in format "HH:MM"
            
        Returns:
            time object
        """
        if not time_str:
            return None
        
        parts = time_str.split(':')
        if len(parts) != 2:
            return None
        
        try:
            return time(int(parts[0]), int(parts[1]))
        except (ValueError, IndexError):
            return None
    
    def _day_name_to_int(self, day_name: str) -> Optional[int]:
        """
        Convert day name to integer (0=Monday, 6=Sunday).
        
        Args:
            day_name: Day name (e.g., "Mon", "Monday")
            
        Returns:
            Day integer or None if invalid
        """
        day_map = {
            'mon': 0, 'monday': 0,
            'tue': 1, 'tuesday': 1,
            'wed': 2, 'wednesday': 2,
            'thu': 3, 'thursday': 3,
            'fri': 4, 'friday': 4,
            'sat': 5, 'saturday': 5,
            'sun': 6, 'sunday': 6
        }
        return day_map.get(day_name.lower())
    
    def is_active(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if time blackout slot is active for given timestamp and symbol.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if blackout is active
        """
        if not self.enabled:
            return False
        
        if self.start_time is None or self.end_time is None:
            return False
        
        # Check symbol filtering
        if self.symbols and symbol not in self.symbols:
            return False
        
        # Check day filtering
        if self.days:
            day_int = timestamp.weekday()  # 0=Monday, 6=Sunday
            day_names = [self._day_name_to_int(d) for d in self.days if self._day_name_to_int(d) is not None]
            if day_int not in day_names:
                return False
        
        # Get time component (UTC+00)
        current_time = timestamp.time()
        
        # Handle time range that might span midnight
        if self.start_time <= self.end_time:
            # Normal range (e.g., 13:00 to 13:45)
            return self.start_time <= current_time <= self.end_time
        else:
            # Spans midnight (e.g., 23:00 to 08:00)
            return current_time >= self.start_time or current_time <= self.end_time
    
    def should_block_signal_generation(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if signal generation should be blocked.
        
        This is the same as is_active() but with a more descriptive name
        for use in signal generation loops.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if signal generation should be blocked
        """
        return self.is_active(timestamp, symbol)


class TimeBlackoutFilter(FilterBase):
    """Time blackout filter with 5 configurable slots.
    
    This filter blocks signal generation during configured time windows.
    Unlike news filters, this directly blocks signal generation, not just
    trading around events.
    
    All times are in UTC+00.
    """
    
    def __init__(self, config):
        """
        Initialize time blackout filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - slots: Dict with slot_1 through slot_5
        """
        super().__init__(config)
        
        # Initialize 5 time blackout slots
        slots_config = config.get('slots', {})
        self.slots: List[TimeBlackoutSlot] = []
        for i in range(1, 6):
            slot_key = f'slot_{i}'
            slot_config = slots_config.get(slot_key, {})
            self.slots.append(TimeBlackoutSlot(slot_config))
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if any time blackout slot is active.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if signal should be blocked
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Check each slot
        for slot in self.slots:
            if slot.is_active(context.timestamp, context.symbol):
                return self._create_fail_result(
                    reason=f"Time blackout active: {slot.name or 'Unnamed'} ({slot.start_time_str} - {slot.end_time_str})",
                    metadata={
                        'filter': 'time_blackout',
                        'slot_name': slot.name,
                        'start_time': slot.start_time_str,
                        'end_time': slot.end_time_str
                    }
                )
        
        return self._create_pass_result()
    
    def should_block_signal_generation(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if signal generation should be blocked at this timestamp.
        
        This method is used by StrategyBase to check BEFORE generating signals,
        allowing early exit from signal generation loops.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if signal generation should be blocked
        """
        if not self.enabled:
            return False
        
        for slot in self.slots:
            if slot.should_block_signal_generation(timestamp, symbol):
                return True
        
        return False
