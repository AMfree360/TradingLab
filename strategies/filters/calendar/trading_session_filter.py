"""Trading session filter for allowing trades only during specific time windows.

This filter checks if the current time falls within any enabled trading session.
Sessions can span midnight (e.g., 23:00-08:00).

All times are in UTC+00.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
from typing import Dict, Optional
import pandas as pd
from datetime import time


class TradingSession:
    """Individual trading session configuration."""
    
    def __init__(self, config: Dict):
        """
        Initialize trading session.
        
        Args:
            config: Session configuration dict with:
                - enabled: bool
                - start: str (format: "HH:MM" in UTC+00)
                - end: str (format: "HH:MM" in UTC+00)
        """
        self.enabled = config.get('enabled', False)
        self.start_str = config.get('start', '')
        self.end_str = config.get('end', '')
        
        # Parse times (UTC+00)
        if self.start_str:
            self.start_time = self._parse_time(self.start_str)
        else:
            self.start_time = None
            
        if self.end_str:
            self.end_time = self._parse_time(self.end_str)
        else:
            self.end_time = None
    
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
    
    def is_active(self, timestamp: pd.Timestamp) -> bool:
        """
        Check if trading session is active for given timestamp.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            
        Returns:
            True if session is active
        """
        if not self.enabled:
            return False
        
        if self.start_time is None or self.end_time is None:
            return False
        
        # Special case: 24/7 session (start == end == 00:00:00)
        # This represents a session that spans all 24 hours
        if self.start_time == self.end_time == time(0, 0):
            return True
        
        # Get time component (UTC+00)
        current_time = timestamp.time()
        
        # Handle time range that might span midnight
        if self.start_time <= self.end_time:
            # Normal range (e.g., 13:00 to 21:00)
            # But exclude the case where start == end (handled above)
            return self.start_time <= current_time < self.end_time
        else:
            # Spans midnight (e.g., 23:00 to 08:00)
            return current_time >= self.start_time or current_time < self.end_time


class TradingSessionFilter(FilterBase):
    """Trading session filter with configurable sessions.
    
    This filter allows trades only during enabled trading sessions.
    Sessions are checked in order, and if ANY enabled session matches,
    the signal passes.
    
    All times are in UTC+00.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize trading session filter.
        
        Args:
            config: Filter configuration with:
                - trading_sessions: Dict with Session1, Session2, Session3, etc.
        """
        super().__init__(config)
        
        # Initialize sessions from config
        sessions_config = config.get('trading_sessions', {})
        self.sessions: Dict[str, TradingSession] = {}
        
        # Parse all session configs (Session1, Session2, Session3, etc.)
        for session_key, session_config in sessions_config.items():
            if isinstance(session_config, dict):
                self.sessions[session_key] = TradingSession(session_config)
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if current time falls within any enabled trading session.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating if signal should pass
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # If no sessions configured, allow all trades
        if not self.sessions:
            return self._create_pass_result()
        
        # Check if ANY sessions are enabled
        enabled_sessions = [name for name, sess in self.sessions.items() if sess.enabled]
        if not enabled_sessions:
            # All sessions are disabled - reject all signals
            # IMPORTANT: This is the critical check - if all sessions are disabled,
            # we MUST reject all signals, not pass them
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"TradingSessionFilter: Rejecting signal - all {len(self.sessions)} sessions disabled")
            return self._create_fail_result(
                reason="All trading sessions are disabled",
                metadata={
                    'filter': 'trading_session',
                    'timestamp': str(context.timestamp),
                    'active_sessions': [],
                    'total_sessions': len(self.sessions),
                    'enabled_count': 0
                }
            )
        
        # Check each enabled session - if ANY enabled session matches, pass
        for session_name, session in self.sessions.items():
            if session.enabled and session.is_active(context.timestamp):
                return self._create_pass_result()
        
        # No enabled session matches - reject signal
        return self._create_fail_result(
            reason=f"Outside trading sessions. Active sessions: {', '.join(enabled_sessions)}",
            metadata={
                'filter': 'trading_session',
                'timestamp': str(context.timestamp),
                'active_sessions': enabled_sessions
            }
        )

