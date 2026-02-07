"""Filter system for trading strategies.

This module provides a reusable filter system that works across all asset classes
(forex, stocks, futures, crypto, CFDs) by using MarketSpec for unit conversions.
"""

from strategies.filters.base import FilterBase, FilterContext, FilterResult
from strategies.filters.manager import FilterManager

# Import regime filters
from strategies.filters.regime.adx_filter import ADXFilter
from strategies.filters.regime.atr_threshold_filter import ATRThresholdFilter
from strategies.filters.regime.sma_distance_filter import SMADistanceFilter
from strategies.filters.regime.sma_slope_filter import SMASlopeFilter

# Import calendar filters
from strategies.filters.calendar.day_of_week_filter import DayOfWeekFilter
from strategies.filters.calendar.time_blackout_filter import TimeBlackoutFilter
from strategies.filters.calendar.trading_session_filter import TradingSessionFilter

# Import news filters
from strategies.filters.news.news_filter import NewsFilter

# Import volume filters
from strategies.filters.volume.volume_increasing_filter import VolumeIncreasingFilter

__all__ = [
    'FilterBase',
    'FilterContext',
    'FilterResult',
    'FilterManager',
    'ADXFilter',
    'ATRThresholdFilter',
    'SMADistanceFilter',
    'SMASlopeFilter',
    'DayOfWeekFilter',
    'TimeBlackoutFilter',
    'TradingSessionFilter',
    'NewsFilter',
    'VolumeIncreasingFilter',
]

# UnitConverter is not imported at module level to avoid circular import
# Import it directly when needed: from strategies.filters.utils import UnitConverter
