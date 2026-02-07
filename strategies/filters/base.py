"""Base classes for filter system."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional, TYPE_CHECKING
import pandas as pd

if TYPE_CHECKING:
    from engine.market import MarketSpec


@dataclass
class FilterContext:
    """Context passed to filters for decision making.
    
    This context provides all information needed for filters to make decisions,
    including market data, signal information, and market specification for
    unit conversions.
    """
    timestamp: pd.Timestamp
    symbol: str
    signal_direction: int  # 1=long, -1=short
    signal_data: pd.Series  # Current signal bar data
    df_by_tf: Dict[str, pd.DataFrame]  # All timeframe data
    indicators: Dict[str, pd.Series] = field(default_factory=dict)  # Pre-calculated indicators
    market_spec: Optional['MarketSpec'] = None  # Market specification for unit conversions


@dataclass
class FilterResult:
    """Result from filter check.
    
    Attributes:
        passed: Whether the filter passed (True) or failed (False)
        reason: Optional reason for failure (human-readable)
        metadata: Optional dictionary with additional information
    """
    passed: bool
    reason: Optional[str] = None
    metadata: Dict = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure metadata is always a dict."""
        if self.metadata is None:
            self.metadata = {}


class FilterBase(ABC):
    """Base class for all filters.
    
    All filters must inherit from this class and implement the `check` method.
    Filters are designed to be asset-class agnostic by using MarketSpec for
    unit conversions.
    
    Example:
        class MyFilter(FilterBase):
            def check(self, context: FilterContext) -> FilterResult:
                # Filter logic here
                if condition:
                    return self._create_pass_result()
                else:
                    return self._create_fail_result("Reason for failure")
    """
    
    def __init__(self, config):
        """
        Initialize filter with configuration.
        
        Args:
            config: Filter configuration (dict or config object)
        """
        self.config = config
        self.enabled = getattr(config, 'enabled', True) if hasattr(config, 'enabled') else config.get('enabled', True)
        self.name = self.__class__.__name__
    
    @abstractmethod
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if filter passes.
        
        This is the main method that all filters must implement. It receives
        a FilterContext with all necessary information and returns a FilterResult
        indicating whether the filter passed or failed.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating pass/fail and reason
        """
        pass
    
    def is_enabled(self) -> bool:
        """
        Check if filter is enabled.
        
        Returns:
            True if filter is enabled, False otherwise
        """
        return self.enabled
    
    def _create_pass_result(self, metadata: Optional[Dict] = None) -> FilterResult:
        """
        Helper to create a pass result.
        
        Args:
            metadata: Optional metadata to include in result
            
        Returns:
            FilterResult with passed=True
        """
        return FilterResult(passed=True, metadata=metadata or {})
    
    def _create_fail_result(self, reason: str, metadata: Optional[Dict] = None) -> FilterResult:
        """
        Helper to create a fail result.
        
        Args:
            reason: Human-readable reason for failure
            metadata: Optional metadata to include in result
            
        Returns:
            FilterResult with passed=False
        """
        return FilterResult(
            passed=False,
            reason=reason,
            metadata=metadata or {}
        )
