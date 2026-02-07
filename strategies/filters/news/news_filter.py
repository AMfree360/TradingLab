"""News filter for blocking trading around economic events.

This filter blocks signals during configured news event windows with
configurable buffers before and after the event time.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
from typing import List, Dict, Optional
import pandas as pd
from datetime import time, timedelta


class NewsSlot:
    """Individual news slot configuration."""
    
    def __init__(self, config: Dict, default_buffer: int = 15):
        """
        Initialize news slot.
        
        Args:
            config: Slot configuration dict with:
                - enabled: bool
                - name: str (event name, e.g., "NFP")
                - start_time: str (format: "HH:MM" in UTC+00)
                - end_time: str (format: "HH:MM" in UTC+00)
                - buffer_before: int (minutes before event, default from parent)
                - buffer_after: int (minutes after event, default from parent)
                - symbols: List[str] (optional, empty = all symbols)
            default_buffer: Default buffer in minutes if not specified
        """
        self.enabled = config.get('enabled', False)
        self.name = config.get('name', '')
        self.start_time_str = config.get('start_time', '')
        self.end_time_str = config.get('end_time', '')
        self.buffer_before = config.get('buffer_before', default_buffer)
        self.buffer_after = config.get('buffer_after', default_buffer)
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
        
        # Calculate effective window (start_time - buffer_before to end_time + buffer_after)
        self.effective_start_time = None
        self.effective_end_time = None
        
        if self.start_time and self.end_time:
            # Calculate effective start (start_time - buffer_before)
            start_dt = pd.Timestamp.combine(pd.Timestamp.now().date(), self.start_time)
            start_dt = start_dt - timedelta(minutes=self.buffer_before)
            self.effective_start_time = start_dt.time()
            
            # Calculate effective end (end_time + buffer_after)
            end_dt = pd.Timestamp.combine(pd.Timestamp.now().date(), self.end_time)
            end_dt = end_dt + timedelta(minutes=self.buffer_after)
            self.effective_end_time = end_dt.time()
    
    def _parse_time(self, time_str: str) -> Optional[time]:
        """
        Parse time string 'HH:MM' to time object.
        
        Args:
            time_str: Time string in format "HH:MM"
            
        Returns:
            time object or None if invalid
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
    
    def is_active(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if news slot is active for given timestamp and symbol.
        
        The slot is active if the timestamp falls within the effective window
        (start_time - buffer_before to end_time + buffer_after).
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if news slot is active
        """
        if not self.enabled:
            return False
        
        if self.effective_start_time is None or self.effective_end_time is None:
            return False
        
        # Check symbol filtering
        if self.symbols and symbol not in self.symbols:
            return False
        
        # Get time component (UTC+00)
        current_time = timestamp.time()
        
        # Handle time range that might span midnight
        if self.effective_start_time <= self.effective_end_time:
            # Normal range (e.g., 13:00 to 14:00)
            return self.effective_start_time <= current_time <= self.effective_end_time
        else:
            # Spans midnight (e.g., 23:00 to 08:00)
            return current_time >= self.effective_start_time or current_time <= self.effective_end_time


class NewsFilter(FilterBase):
    """News filter with 5 configurable slots.
    
    This filter blocks signals during configured news event windows with
    configurable buffers. Pre-configured with popular high-impact news times:
    - NFP (Non-Farm Payrolls): 13:15-13:45 UTC
    - FOMC (Federal Open Market Committee): 18:45-19:15 UTC
    - ECB (European Central Bank): 13:30-14:00 UTC
    - BOE (Bank of England): 11:45-12:15 UTC
    - US CPI (US Consumer Price Index): 13:15-13:45 UTC
    
    All times are in UTC+00.
    """
    
    def __init__(self, config):
        """
        Initialize news filter.
        
        Args:
            config: Filter configuration with:
                - enabled: bool
                - default_buffer_minutes: int (default: 15)
                - slots: Dict with slot_1 through slot_5
        """
        super().__init__(config)
        self.default_buffer = config.get('default_buffer_minutes', 15)
        
        # Initialize 5 news slots
        slots_config = config.get('slots', {})
        self.slots: List[NewsSlot] = []
        for i in range(1, 6):
            slot_key = f'slot_{i}'
            slot_config = slots_config.get(slot_key, {})
            self.slots.append(NewsSlot(slot_config, self.default_buffer))
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if any news slot is active.
        
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
                    reason=f"News filter active: {slot.name} ({slot.start_time_str} - {slot.end_time_str}, "
                           f"buffer: {slot.buffer_before}m before, {slot.buffer_after}m after)",
                    metadata={
                        'filter': 'news',
                        'slot_name': slot.name,
                        'start_time': slot.start_time_str,
                        'end_time': slot.end_time_str,
                        'buffer_before': slot.buffer_before,
                        'buffer_after': slot.buffer_after
                    }
                )
        
        return self._create_pass_result()
