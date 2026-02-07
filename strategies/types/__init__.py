"""Strategy type templates for common trading styles.

These templates provide pre-configured base classes for different trading styles:
- TrendStrategy: For trend-following strategies
- MeanReversionStrategy: For mean-reversion strategies
- VolatilityStrategy: For volatility-based strategies
"""

from strategies.types.trend_strategy import TrendStrategy
from strategies.types.mean_reversion_strategy import MeanReversionStrategy
from strategies.types.volatility_strategy import VolatilityStrategy

__all__ = [
    'TrendStrategy',
    'MeanReversionStrategy',
    'VolatilityStrategy',
]
