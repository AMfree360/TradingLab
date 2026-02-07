# Strategy Filter Abstraction Architecture

## Table of Contents

1. [Overview](#overview)
2. [Current State Analysis](#current-state-analysis)
3. [Design Goals](#design-goals)
4. [Architecture Design](#architecture-design)
5. [Unit Handling and Asset Class Support](#unit-handling-and-asset-class-support) ⭐ **IMPORTANT**
6. [Filter System Details](#filter-system-details)
7. [Configuration System](#configuration-system)
8. [Implementation Details](#implementation-details)
9. [Code Examples](#code-examples)
10. [Migration Strategy](#migration-strategy)
11. [Testing Strategy](#testing-strategy)

## Overview

This document describes the architecture for abstracting common strategy features (filters, position management, etc.) into reusable components. This system eliminates code duplication, improves maintainability, and follows industry-standard design patterns.

### Key Benefits

- **Code Reuse**: Filters implemented once, used by all strategies
- **Consistency**: Same filter logic across all strategies
- **Maintainability**: Fix bugs once, all strategies benefit
- **Flexibility**: Easy to add new filters or strategy types
- **Testing**: Test filters independently
- **Configuration**: Master defaults with strategy-level overrides

## Current State Analysis

### What We Have

✅ **Comprehensive Filter Configurations**:
- Calendar filters (day of week, month, sessions, blackouts)
- Regime filters (ADX, ATR percentile, EMA expansion, swing)
- News filters (basic structure)
- All defined in `config/schema.py`

✅ **Position Management Features**:
- Trailing stops (EMA-based, stepped R-based)
- Partial exits
- Multiple stop loss types
- Position flattening (basic)
- Implemented in `engine/backtest_engine.py`

✅ **Trade Management Abstraction** (NEW):
- `TradeManagementManager` - Merges master and strategy configs
- `ExitResolver` - Handles simultaneous exit conditions
- Master config pattern (`config/master_trade_management.yml`)
- Support for multiple TP/partial exit levels
- Multiple trailing stop types (EMA, SMA, ATR, percentage, fixed distance)
- Minimum stop distance (ATR-based or fixed)
- Strategy-level overrides supported

❌ **Code Duplication**:
- Filter logic duplicated in each strategy (e.g., legacy strategy-specific checks)
- Same filter checks repeated across strategies
- Hard to maintain and update

❌ **No Strategy Type Templates**:
- Each strategy implements filters from scratch
- No pre-configured strategy types (Trend, Mean Reversion, etc.)

❌ **No Master Configuration**:
- Each strategy configures filters independently
- No way to set defaults across all strategies

## Design Goals

1. **Eliminate Code Duplication**: Common filters implemented once
2. **Strategy Type Templates**: Pre-configured types (Trend, Mean Reversion, Volatility)
3. **Flexible Configuration**: Master-level defaults with strategy-level overrides
4. **Composition over Inheritance**: Filters as composable components
5. **Industry Standards**: Follow established patterns (Filter Chain, Strategy Pattern)

## Architecture Design

### System Architecture Diagram

```
┌─────────────────────────────────────────────────────────┐
│                  Strategy Layer                          │
│                                                          │
│  StrategyBase (Abstract)                                 │
│    │                                                      │
│    ├── TrendStrategy (Pre-configured)                   │
│    ├── MeanReversionStrategy (Pre-configured)          │
│    ├── VolatilityStrategy (Pre-configured)             │
│    └── CustomStrategy (Fully configurable)              │
│                                                          │
│  Each strategy has:                                      │
│    - FilterManager (applies filter chain)              │
│    - PositionManager (handles stops, trailing, partials)│
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┼──────────────┐
        ▼              ▼              ▼
┌─────────────┐ ┌─────────────┐ ┌─────────────┐
│   Filter    │ │  Position   │ │   Config   │
│   System    │ │  Management │ │   System   │
└─────────────┘ └─────────────┘ └─────────────┘
```

### Filter System Architecture

```
FilterBase (Abstract)
    │
    ├── CalendarFilter
    │   ├── DayOfWeekFilter
    │   ├── MonthOfYearFilter
    │   ├── TradingSessionFilter
    │   └── TimeBlackoutFilter (NEW - 5 slots)
    │
    ├── RegimeFilter
    │   ├── ADXFilter
    │   ├── ATRPercentileFilter
    │   ├── ATRThresholdFilter (NEW - min/max)
    │   ├── EMAExpansionFilter
    │   ├── SwingFilter
    │   ├── ChoppyMarketFilter
    │   └── TrendingMarketFilter
    │
    ├── NewsFilter (ENHANCED - 5 slots)
    │   ├── NewsSlot1 (start-end time, buffer before/after)
    │   ├── NewsSlot2
    │   ├── NewsSlot3
    │   ├── NewsSlot4
    │   └── NewsSlot5
    │
    └── VolumeFilter
        ├── VolumeOscillatorFilter
        └── VolumeThresholdFilter
```

### Strategy Type Hierarchy

```
StrategyBase (Abstract)
    │
    ├── TrendStrategy
    │   └── Pre-configured:
    │       - ADX filter enabled (min 23.0)
    │       - EMA Expansion enabled
    │       - Swing filter enabled
    │       - Composite regime enabled
    │       - Trailing stops enabled
    │
    ├── MeanReversionStrategy
    │   └── Pre-configured:
    │       - ATR Percentile filter enabled
    │       - Range/Choppy market filters
    │       - Fixed TP (no trailing)
    │
    ├── VolatilityStrategy
    │   └── Pre-configured:
    │       - ATR min/max thresholds
    │       - Volatility bands
    │       - Dynamic position sizing
    │
    └── CustomStrategy
        └── Fully configurable, all filters available
```

## Unit Handling and Asset Class Support

### Critical Design Decision: Market Profile vs Filter Architecture

**Answer: Both, with clear separation of concerns**

#### Market Profile (config/market_profiles.yml + engine/market.py)
**Responsibility**: Defines units and conversion factors for each asset class

- **Pip Value**: Price value of one pip (forex: 0.0001, JPY pairs: 0.01)
- **Point Value**: Price value of one point (stocks: 0.01, indices: varies)
- **Tick Value**: Price value of one tick (futures: varies by contract)
- **Contract Size**: Standard contract size (forex: 100,000, futures: varies)
- **Asset Class**: Identifies market type (forex, crypto, stock, futures)

#### Filter Architecture (strategies/filters/)
**Responsibility**: Uses MarketSpec to work with correct units across all asset classes

- Filters receive MarketSpec in context
- Filters convert units using MarketSpec methods
- Filters are asset-class agnostic (work for all asset classes)

### Unit Conversion Strategy

**Key Principle**: All filters must work with normalized units or use MarketSpec for conversions.

#### Supported Units

1. **Pips** (Forex):
   - Most pairs: 1 pip = 0.0001 (e.g., EURUSD: 1.12345 → 1.12346 = +1 pip)
   - JPY pairs: 1 pip = 0.01 (e.g., USDJPY: 110.50 → 110.51 = +1 pip)
   - Stored in: `MarketSpec.pip_value`

2. **Points** (Stocks, Indices):
   - Stocks: 1 point = 0.01 (e.g., AAPL: 150.00 → 150.01 = +1 point)
   - Indices: Varies (e.g., SPX: 1 point = 1.0, NAS100: 1 point = 1.0)
   - Stored in: `MarketSpec.tick_value` or calculated from price_precision

3. **Price Units** (Crypto, Direct):
   - Crypto: Direct price units (e.g., BTCUSDT: 50000.00 → 50001.00 = +1.00)
   - No conversion needed for crypto
   - Use direct price differences

4. **Ticks** (Futures):
   - Varies by contract (e.g., ES: 1 tick = 0.25, NQ: 1 tick = 0.25)
   - Stored in: `MarketSpec.tick_value`

### Filter Context Enhancement

**Location**: `strategies/filters/base.py` (update FilterContext)

```python
@dataclass
class FilterContext:
    """Context passed to filters for decision making."""
    timestamp: pd.Timestamp
    symbol: str
    signal_direction: int  # 1=long, -1=short
    signal_data: pd.Series  # Current signal bar data
    df_by_tf: Dict[str, pd.DataFrame]  # All timeframe data
    indicators: Dict[str, pd.Series]  # Pre-calculated indicators
    market_spec: MarketSpec  # NEW: Market specification for unit conversions
```

### Unit Conversion Helper

**Location**: `strategies/filters/utils.py` (new file)

```python
from engine.market import MarketSpec
from typing import Union

class UnitConverter:
    """Helper class for unit conversions across asset classes."""
    
    def __init__(self, market_spec: MarketSpec):
        self.market_spec = market_spec
    
    def pips_to_price(self, pips: float) -> float:
        """
        Convert pips to price units.
        
        Args:
            pips: Number of pips
            
        Returns:
            Price difference in price units
        """
        if self.market_spec.asset_class == 'forex':
            pip_value = self.market_spec.pip_value or 0.0001
            return pips * pip_value
        else:
            # For non-forex, pips don't apply
            raise ValueError(f"Pips not applicable for asset class: {self.market_spec.asset_class}")
    
    def points_to_price(self, points: float) -> float:
        """
        Convert points to price units.
        
        Args:
            points: Number of points
            
        Returns:
            Price difference in price units
        """
        if self.market_spec.asset_class == 'stock':
            # Stocks: 1 point = 0.01
            return points * 0.01
        elif self.market_spec.asset_class in ['futures', 'indices']:
            # Indices/Futures: Use tick_value if available
            if self.market_spec.tick_value:
                return points * self.market_spec.tick_value
            else:
                # Default: 1 point = 1.0 for indices
                return points * 1.0
        else:
            # For forex/crypto, points don't apply
            raise ValueError(f"Points not applicable for asset class: {self.market_spec.asset_class}")
    
    def price_to_pips(self, price_diff: float) -> float:
        """
        Convert price difference to pips.
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of pips
        """
        if self.market_spec.asset_class == 'forex':
            pip_value = self.market_spec.pip_value or 0.0001
            return price_diff / pip_value
        else:
            raise ValueError(f"Pips not applicable for asset class: {self.market_spec.asset_class}")
    
    def price_to_points(self, price_diff: float) -> float:
        """
        Convert price difference to points.
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of points
        """
        if self.market_spec.asset_class == 'stock':
            return price_diff / 0.01
        elif self.market_spec.asset_class in ['futures', 'indices']:
            if self.market_spec.tick_value:
                return price_diff / self.market_spec.tick_value
            else:
                return price_diff / 1.0
        else:
            raise ValueError(f"Points not applicable for asset class: {self.market_spec.asset_class}")
    
    def get_minimum_price_move(self) -> float:
        """
        Get minimum price move for this asset class.
        
        Returns:
            Minimum price increment (pip_value, tick_value, or 0.01 for stocks)
        """
        if self.market_spec.asset_class == 'forex':
            return self.market_spec.pip_value or 0.0001
        elif self.market_spec.asset_class == 'stock':
            return 0.01
        elif self.market_spec.asset_class in ['futures', 'indices']:
            return self.market_spec.tick_value or 1.0
        else:  # crypto
            # Crypto: Use price precision to determine minimum move
            precision = self.market_spec.price_precision
            return 10 ** (-precision)
```

### Updated Filter Examples with Unit Handling

#### ATR Threshold Filter (Asset-Class Agnostic)

**Location**: `strategies/filters/regime/atr_threshold_filter.py` (updated)

```python
from strategies.filters.base import FilterBase, FilterContext, FilterResult
from strategies.filters.utils import UnitConverter
import pandas as pd

class ATRThresholdFilter(FilterBase):
    """ATR threshold filter (min/max ATR values) - works for all asset classes."""
    
    def __init__(self, config):
        super().__init__(config)
        self.atr_period = config.get('atr_period', 14)
        
        # Accept thresholds in multiple formats
        self.min_atr = config.get('min_atr', None)  # Direct price units
        self.max_atr = config.get('max_atr', None)  # Direct price units
        self.min_atr_pips = config.get('min_atr_pips', None)  # Pips (forex only)
        self.max_atr_pips = config.get('max_atr_pips', None)  # Pips (forex only)
        self.min_atr_points = config.get('min_atr_points', None)  # Points (stocks/indices)
        self.max_atr_points = config.get('max_atr_points', None)  # Points (stocks/indices)
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if ATR is within configured thresholds.
        Works for all asset classes by using MarketSpec for unit conversion.
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ATR value from signal data
        if 'atr' not in context.signal_data or pd.isna(context.signal_data['atr']):
            return self._create_fail_result(
                reason="ATR value not available",
                metadata={'filter': 'atr_threshold', 'value': None}
            )
        
        atr_value = float(context.signal_data['atr'])
        converter = UnitConverter(context.market_spec)
        
        # Determine thresholds based on asset class and config
        min_threshold = self.min_atr
        max_threshold = self.max_atr
        
        # Convert pip thresholds to price units (forex only)
        if context.market_spec.asset_class == 'forex':
            if self.min_atr_pips is not None:
                min_threshold = converter.pips_to_price(self.min_atr_pips)
            if self.max_atr_pips is not None:
                max_threshold = converter.pips_to_price(self.max_atr_pips)
        
        # Convert point thresholds to price units (stocks/indices/futures)
        elif context.market_spec.asset_class in ['stock', 'futures']:
            if self.min_atr_points is not None:
                min_threshold = converter.points_to_price(self.min_atr_points)
            if self.max_atr_points is not None:
                max_threshold = converter.points_to_price(self.max_atr_points)
        
        # Check minimum threshold
        if min_threshold is not None and atr_value < min_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} below minimum threshold {min_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': min_threshold,
                    'type': 'min',
                    'asset_class': context.market_spec.asset_class
                }
            )
        
        # Check maximum threshold
        if max_threshold is not None and atr_value > max_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} above maximum threshold {max_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': max_threshold,
                    'type': 'max',
                    'asset_class': context.market_spec.asset_class
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'atr_threshold',
                'value': atr_value,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold,
                'asset_class': context.market_spec.asset_class
            }
        )
```

### Market Profile Enhancement

**Location**: `config/market_profiles.yml` (add missing fields)

```yaml
markets:
  # Forex markets
  EURUSD:
    exchange: oanda
    symbol: EURUSD
    asset_class: forex
    market_type: spot
    leverage: 30.0
    commission_rate: 0.000035
    slippage_ticks: 2  # In pips
    min_trade_size: 0.01
    price_precision: 5
    quantity_precision: 2
    pip_value: 0.0001  # 1 pip = 0.0001 price units
    contract_size: 100000  # Standard lot size
    point_value: null  # Not applicable for forex
    
  # JPY pairs (different pip size)
  USDJPY:
    exchange: oanda
    symbol: USDJPY
    asset_class: forex
    market_type: spot
    leverage: 30.0
    commission_rate: 0.000035
    slippage_ticks: 2
    min_trade_size: 0.01
    price_precision: 3  # JPY pairs have 3 decimals
    quantity_precision: 2
    pip_value: 0.01  # 1 pip = 0.01 for JPY pairs
    contract_size: 100000
    point_value: null
    
  # Stock markets
  AAPL:
    exchange: nasdaq
    symbol: AAPL
    asset_class: stock
    market_type: spot
    leverage: 1.0
    commission_rate: 0.001
    slippage_ticks: 0.01  # $0.01 for stocks
    min_trade_size: 1.0  # Minimum 1 share
    price_precision: 2
    quantity_precision: 0  # Whole shares only
    pip_value: null  # Not applicable
    contract_size: null  # Not applicable
    point_value: 0.01  # 1 point = $0.01
    tick_value: 0.01  # Minimum price move
    
  # Indices (CFDs)
  US500:
    exchange: oanda
    symbol: US500
    asset_class: futures  # Or 'indices' if separate
    market_type: spot  # CFD
    leverage: 20.0
    commission_rate: 0.0
    slippage_ticks: 1  # In points
    min_trade_size: 1.0
    price_precision: 2
    quantity_precision: 0
    pip_value: null
    contract_size: null
    point_value: 1.0  # 1 point = 1.0 price units
    tick_value: 1.0
    
  # Futures
  ES:
    exchange: cme
    symbol: ES
    asset_class: futures
    market_type: futures
    leverage: 20.0
    commission_rate: 0.0
    slippage_ticks: 1
    min_trade_size: 1.0
    price_precision: 2
    quantity_precision: 0
    pip_value: null
    contract_size: 50  # ES contract multiplier
    point_value: 1.0
    tick_value: 0.25  # ES ticks in 0.25 increments
    
  # Crypto
  BTCUSDT:
    exchange: binance
    symbol: BTCUSDT
    asset_class: crypto
    market_type: spot
    leverage: 1.0
    commission_rate: 0.0004
    slippage_ticks: 0.0
    min_trade_size: 10.0
    price_precision: 2
    quantity_precision: 6
    pip_value: null  # Not applicable
    contract_size: null  # Not applicable
    point_value: null  # Use direct price units
    tick_value: null  # Use price_precision
```

### Filter Manager Update

**Location**: `strategies/filters/manager.py` (update to pass MarketSpec)

```python
def apply_filters(
    self, 
    signal: pd.Series, 
    context: FilterContext
) -> FilterResult:
    """
    Apply all enabled filters to signal.
    
    Args:
        signal: Signal Series with direction, entry_price, etc.
        context: FilterContext with market data and MarketSpec
        
    Returns:
        FilterResult indicating if signal passed all filters
    """
    # Update context with signal data
    context.signal_data = signal
    context.signal_direction = signal.get('direction', 1)
    
    # Ensure MarketSpec is available in context
    if not hasattr(context, 'market_spec') or context.market_spec is None:
        # Load MarketSpec from symbol
        from engine.market import MarketSpec
        try:
            context.market_spec = MarketSpec.load_from_profiles(context.symbol)
        except ValueError:
            # Fallback: create basic MarketSpec
            context.market_spec = MarketSpec(
                symbol=context.symbol,
                exchange='unknown',
                asset_class='crypto'  # Default fallback
            )
    
    # Apply filters in sequence (short-circuit on first failure)
    for filter_obj in self.filters:
        if not filter_obj.is_enabled():
            continue
        
        result = filter_obj.check(context)
        if not result.passed:
            return result  # Short-circuit on first failure
    
    # All filters passed
    return FilterResult(passed=True)
```

### Unit Handling Summary

**Design Decision Summary**:

| Component | Responsibility | Example |
|-----------|---------------|---------|
| **Market Profile** | Defines unit values | `pip_value: 0.0001` for EURUSD |
| **MarketSpec** | Provides unit conversion methods | `pips_to_price()`, `points_to_price()` |
| **UnitConverter** | Helper for filters | Converts config values to price units |
| **Filters** | Uses MarketSpec/UnitConverter | Never hardcodes unit values |

**Example Flow**:
```
1. Config: min_atr_pips = 10.0 (for EURUSD)
2. Market Profile: EURUSD has pip_value = 0.0001
3. Filter receives: MarketSpec with pip_value = 0.0001
4. UnitConverter converts: 10.0 pips × 0.0001 = 0.0010 price units
5. Filter compares: ATR value (in price units) vs 0.0010
6. Result: Works for EURUSD, USDJPY, AAPL, US500, BTCUSDT - all asset classes!
```

**Key Benefits**:
- ✅ **One filter implementation** works for all asset classes
- ✅ **Unit definitions centralized** in Market Profile
- ✅ **Easy to add new asset classes** (just update Market Profile)
- ✅ **No hardcoded values** in filters
- ✅ **Consistent behavior** across all strategies

## Filter System Details

### Critical Design Decision: Market Profile vs Filter Architecture

**Answer: Both, with clear separation of concerns**

#### Market Profile (config/market_profiles.yml + engine/market.py)
**Responsibility**: Defines units and conversion factors for each asset class

- **Pip Value**: Price value of one pip (forex: 0.0001, JPY pairs: 0.01)
- **Point Value**: Price value of one point (stocks: 0.01, indices: varies)
- **Tick Value**: Price value of one tick (futures: varies by contract)
- **Contract Size**: Standard contract size (forex: 100,000, futures: varies)
- **Asset Class**: Identifies market type (forex, crypto, stock, futures)

#### Filter Architecture (strategies/filters/)
**Responsibility**: Uses MarketSpec to work with correct units across all asset classes

- Filters receive MarketSpec in context
- Filters convert units using MarketSpec methods
- Filters are asset-class agnostic (work for all asset classes)

### Unit Conversion Strategy

**Key Principle**: All filters must work with normalized units or use MarketSpec for conversions.

#### Supported Units by Asset Class

| Asset Class | Primary Unit | Conversion Factor | Example |
|------------|--------------|-------------------|---------|
| **Forex** | Pips | pip_value (0.0001 or 0.01) | EURUSD: 1 pip = 0.0001 |
| **Stocks** | Points | 0.01 (fixed) | AAPL: 1 point = $0.01 |
| **Indices (CFDs)** | Points | 1.0 (typically) | US500: 1 point = 1.0 |
| **Futures** | Ticks | tick_value (varies) | ES: 1 tick = 0.25 |
| **Crypto** | Price Units | Direct (no conversion) | BTCUSDT: 1.00 = 1.00 |

### Filter Context Enhancement

**Location**: `strategies/filters/base.py` (update FilterContext)

```python
@dataclass
class FilterContext:
    """Context passed to filters for decision making."""
    timestamp: pd.Timestamp
    symbol: str
    signal_direction: int  # 1=long, -1=short
    signal_data: pd.Series  # Current signal bar data
    df_by_tf: Dict[str, pd.DataFrame]  # All timeframe data
    indicators: Dict[str, pd.Series]  # Pre-calculated indicators
    market_spec: MarketSpec  # NEW: Market specification for unit conversions
```

### Unit Conversion Helper

**Location**: `strategies/filters/utils.py` (new file)

```python
from engine.market import MarketSpec
from typing import Union

class UnitConverter:
    """Helper class for unit conversions across asset classes."""
    
    def __init__(self, market_spec: MarketSpec):
        self.market_spec = market_spec
    
    def pips_to_price(self, pips: float) -> float:
        """
        Convert pips to price units (forex only).
        
        Args:
            pips: Number of pips
            
        Returns:
            Price difference in price units
            
        Raises:
            ValueError: If asset class is not forex
        """
        if self.market_spec.asset_class != 'forex':
            raise ValueError(f"Pips not applicable for asset class: {self.market_spec.asset_class}")
        
        pip_value = self.market_spec.pip_value or 0.0001
        return pips * pip_value
    
    def points_to_price(self, points: float) -> float:
        """
        Convert points to price units (stocks, indices, futures).
        
        Args:
            points: Number of points
            
        Returns:
            Price difference in price units
        """
        if self.market_spec.asset_class == 'stock':
            # Stocks: 1 point = 0.01
            return points * 0.01
        elif self.market_spec.asset_class in ['futures']:
            # Futures: Use tick_value if available, else default to 1.0
            tick_value = getattr(self.market_spec, 'tick_value', None) or 1.0
            return points * tick_value
        elif self.market_spec.asset_class in ['crypto']:
            # For crypto/indices, points might mean direct price units
            # Check if tick_value is defined
            tick_value = getattr(self.market_spec, 'tick_value', None)
            if tick_value:
                return points * tick_value
            else:
                # Default: 1 point = 1.0 price unit
                return points * 1.0
        else:
            raise ValueError(f"Points not applicable for asset class: {self.market_spec.asset_class}")
    
    def price_to_pips(self, price_diff: float) -> float:
        """
        Convert price difference to pips (forex only).
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of pips
        """
        if self.market_spec.asset_class != 'forex':
            raise ValueError(f"Pips not applicable for asset class: {self.market_spec.asset_class}")
        
        pip_value = self.market_spec.pip_value or 0.0001
        return price_diff / pip_value
    
    def price_to_points(self, price_diff: float) -> float:
        """
        Convert price difference to points (stocks, indices, futures).
        
        Args:
            price_diff: Price difference in price units
            
        Returns:
            Number of points
        """
        if self.market_spec.asset_class == 'stock':
            return price_diff / 0.01
        elif self.market_spec.asset_class in ['futures']:
            tick_value = getattr(self.market_spec, 'tick_value', None) or 1.0
            return price_diff / tick_value
        else:
            tick_value = getattr(self.market_spec, 'tick_value', None)
            if tick_value:
                return price_diff / tick_value
            else:
                return price_diff / 1.0
    
    def get_minimum_price_move(self) -> float:
        """
        Get minimum price move for this asset class.
        
        Returns:
            Minimum price increment (pip_value, tick_value, or 0.01 for stocks)
        """
        if self.market_spec.asset_class == 'forex':
            return self.market_spec.pip_value or 0.0001
        elif self.market_spec.asset_class == 'stock':
            return 0.01
        elif self.market_spec.asset_class in ['futures']:
            tick_value = getattr(self.market_spec, 'tick_value', None)
            return tick_value or 1.0
        else:  # crypto
            # Crypto: Use price precision to determine minimum move
            precision = self.market_spec.price_precision
            return 10 ** (-precision)
    
    def normalize_threshold(self, threshold: Union[float, Dict]) -> float:
        """
        Normalize threshold to price units based on asset class.
        
        Accepts thresholds in multiple formats:
        - Direct price units: 0.001
        - Pips (forex): {"pips": 10}
        - Points (stocks/indices): {"points": 5}
        - Ticks (futures): {"ticks": 4}
        
        Args:
            threshold: Threshold value (float or dict with unit type)
            
        Returns:
            Threshold in price units
        """
        if isinstance(threshold, (int, float)):
            # Direct price units
            return float(threshold)
        
        if isinstance(threshold, dict):
            if 'pips' in threshold:
                return self.pips_to_price(threshold['pips'])
            elif 'points' in threshold:
                return self.points_to_price(threshold['points'])
            elif 'ticks' in threshold:
                return self.points_to_price(threshold['ticks'])  # Ticks same as points for futures
            else:
                # Default: assume price units
                return float(threshold.get('value', 0.0))
        
        return float(threshold)
```

### Updated ATR Threshold Filter (Asset-Class Agnostic)

**Location**: `strategies/filters/regime/atr_threshold_filter.py` (updated)

```python
from strategies.filters.base import FilterBase, FilterContext, FilterResult
from strategies.filters.utils import UnitConverter
import pandas as pd

class ATRThresholdFilter(FilterBase):
    """ATR threshold filter (min/max ATR values) - works for all asset classes."""
    
    def __init__(self, config):
        super().__init__(config)
        self.atr_period = config.get('atr_period', 14)
        
        # Accept thresholds in multiple formats (normalized in check method)
        self.min_atr = config.get('min_atr', None)  # Direct price units
        self.max_atr = config.get('max_atr', None)  # Direct price units
        self.min_atr_pips = config.get('min_atr_pips', None)  # Pips (forex only)
        self.max_atr_pips = config.get('max_atr_pips', None)  # Pips (forex only)
        self.min_atr_points = config.get('min_atr_points', None)  # Points (stocks/indices/futures)
        self.max_atr_points = config.get('max_atr_points', None)  # Points (stocks/indices/futures)
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if ATR is within configured thresholds.
        Works for all asset classes by using MarketSpec for unit conversion.
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ATR value from signal data
        if 'atr' not in context.signal_data or pd.isna(context.signal_data['atr']):
            return self._create_fail_result(
                reason="ATR value not available",
                metadata={'filter': 'atr_threshold', 'value': None}
            )
        
        atr_value = float(context.signal_data['atr'])
        converter = UnitConverter(context.market_spec)
        
        # Determine thresholds based on asset class and config
        min_threshold = self.min_atr
        max_threshold = self.max_atr
        
        # Convert pip thresholds to price units (forex only)
        if context.market_spec.asset_class == 'forex':
            if self.min_atr_pips is not None:
                try:
                    min_threshold = converter.pips_to_price(self.min_atr_pips)
                except ValueError:
                    # If pips not applicable, ignore
                    pass
            if self.max_atr_pips is not None:
                try:
                    max_threshold = converter.pips_to_price(self.max_atr_pips)
                except ValueError:
                    pass
        
        # Convert point thresholds to price units (stocks/indices/futures)
        elif context.market_spec.asset_class in ['stock', 'futures']:
            if self.min_atr_points is not None:
                try:
                    min_threshold = converter.points_to_price(self.min_atr_points)
                except ValueError:
                    pass
            if self.max_atr_points is not None:
                try:
                    max_threshold = converter.points_to_price(self.max_atr_points)
                except ValueError:
                    pass
        
        # Check minimum threshold
        if min_threshold is not None and atr_value < min_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} below minimum threshold {min_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': min_threshold,
                    'type': 'min',
                    'asset_class': context.market_spec.asset_class
                }
            )
        
        # Check maximum threshold
        if max_threshold is not None and atr_value > max_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} above maximum threshold {max_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': max_threshold,
                    'type': 'max',
                    'asset_class': context.market_spec.asset_class
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'atr_threshold',
                'value': atr_value,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold,
                'asset_class': context.market_spec.asset_class
            }
        )
```

### Configuration Examples by Asset Class

#### Forex (EURUSD) - Using Pips

```yaml
regime_filters:
  atr_threshold:
    enabled: true
    min_atr_pips: 10.0  # Minimum 10 pips ATR
    max_atr_pips: 50.0  # Maximum 50 pips ATR
    # OR use direct price units:
    # min_atr: 0.0010  # 10 pips = 0.0010 for EURUSD
    # max_atr: 0.0050  # 50 pips = 0.0050 for EURUSD
```

#### Stocks (AAPL) - Using Points

```yaml
regime_filters:
  atr_threshold:
    enabled: true
    min_atr_points: 0.50  # Minimum $0.50 ATR (50 points)
    max_atr_points: 5.00   # Maximum $5.00 ATR (500 points)
    # OR use direct price units:
    # min_atr: 0.50  # $0.50
    # max_atr: 5.00  # $5.00
```

#### Indices (US500) - Using Points

```yaml
regime_filters:
  atr_threshold:
    enabled: true
    min_atr_points: 10.0  # Minimum 10 points ATR
    max_atr_points: 100.0  # Maximum 100 points ATR
    # OR use direct price units:
    # min_atr: 10.0  # 10 points = 10.0 price units
    # max_atr: 100.0  # 100 points = 100.0 price units
```

#### Futures (ES) - Using Ticks

```yaml
regime_filters:
  atr_threshold:
    enabled: true
    min_atr_points: 4.0  # Minimum 4 ticks ATR (ES: 1 tick = 0.25, so 4 ticks = 1.0)
    max_atr_points: 40.0  # Maximum 40 ticks ATR (40 ticks = 10.0)
    # OR use direct price units:
    # min_atr: 1.0  # 1.0 price units
    # max_atr: 10.0  # 10.0 price units
```

#### Crypto (BTCUSDT) - Using Price Units

```yaml
regime_filters:
  atr_threshold:
    enabled: true
    min_atr: 100.0  # Minimum $100 ATR (direct price units)
    max_atr: 1000.0  # Maximum $1000 ATR (direct price units)
    # Crypto doesn't use pips/points, use direct price units
```

### Market Profile Enhancement

**Location**: `config/market_profiles.yml` (add point_value and tick_value)

```yaml
markets:
  # Forex markets
  EURUSD:
    exchange: oanda
    symbol: EURUSD
    asset_class: forex
    market_type: spot
    leverage: 30.0
    commission_rate: 0.000035
    slippage_ticks: 2  # In pips
    min_trade_size: 0.01
    price_precision: 5
    quantity_precision: 2
    pip_value: 0.0001  # 1 pip = 0.0001 price units
    contract_size: 100000  # Standard lot size
    point_value: null  # Not applicable for forex
    tick_value: null  # Not applicable for forex
    
  # JPY pairs (different pip size)
  USDJPY:
    exchange: oanda
    symbol: USDJPY
    asset_class: forex
    market_type: spot
    leverage: 30.0
    commission_rate: 0.000035
    slippage_ticks: 2
    min_trade_size: 0.01
    price_precision: 3  # JPY pairs have 3 decimals
    quantity_precision: 2
    pip_value: 0.01  # 1 pip = 0.01 for JPY pairs
    contract_size: 100000
    point_value: null
    tick_value: null
    
  # Stock markets
  AAPL:
    exchange: nasdaq
    symbol: AAPL
    asset_class: stock
    market_type: spot
    leverage: 1.0
    commission_rate: 0.001
    slippage_ticks: 0.01  # $0.01 for stocks
    min_trade_size: 1.0  # Minimum 1 share
    price_precision: 2
    quantity_precision: 0  # Whole shares only
    pip_value: null  # Not applicable
    contract_size: null  # Not applicable
    point_value: 0.01  # 1 point = $0.01
    tick_value: 0.01  # Minimum price move
    
  # Indices (CFDs)
  US500:
    exchange: oanda
    symbol: US500
    asset_class: futures  # Or 'indices' if separate
    market_type: spot  # CFD
    leverage: 20.0
    commission_rate: 0.0
    slippage_ticks: 1  # In points
    min_trade_size: 1.0
    price_precision: 2
    quantity_precision: 0
    pip_value: null
    contract_size: null
    point_value: 1.0  # 1 point = 1.0 price units
    tick_value: 1.0
    
  # Futures
  ES:
    exchange: cme
    symbol: ES
    asset_class: futures
    market_type: futures
    leverage: 20.0
    commission_rate: 0.0
    slippage_ticks: 1
    min_trade_size: 1.0
    price_precision: 2
    quantity_precision: 0
    pip_value: null
    contract_size: 50  # ES contract multiplier
    point_value: 1.0
    tick_value: 0.25  # ES ticks in 0.25 increments
    
  # Crypto
  BTCUSDT:
    exchange: binance
    symbol: BTCUSDT
    asset_class: crypto
    market_type: spot
    leverage: 1.0
    commission_rate: 0.0004
    slippage_ticks: 0.0
    min_trade_size: 10.0
    price_precision: 2
    quantity_precision: 6
    pip_value: null  # Not applicable
    contract_size: null  # Not applicable
    point_value: null  # Use direct price units
    tick_value: null  # Use price_precision
```

### Summary: Unit Handling Design

**Question**: Do unit conversions belong in Market Profile or Filter Architecture?

**Answer**: **Both, with clear separation of concerns**

1. **Market Profile (`config/market_profiles.yml` + `engine/market.py`)**:
   - **Defines** unit values (pip_value, point_value, tick_value)
   - **Stores** asset class information
   - **Provides** conversion methods (e.g., `calculate_pip_value_per_lot()`)
   - **Single source of truth** for market-specific constants

2. **Filter Architecture (`strategies/filters/`)**:
   - **Uses** MarketSpec from context
   - **Converts** units using MarketSpec/UnitConverter
   - **Works** across all asset classes (agnostic)
   - **Never hardcodes** unit values

**Key Principle**: 
- Market Profile = **WHAT** (defines units)
- Filter Architecture = **HOW** (uses units correctly)

**Example Flow**:
```
1. Market Profile defines: EURUSD has pip_value = 0.0001
2. Filter receives: MarketSpec with pip_value = 0.0001
3. Filter config says: min_atr_pips = 10.0
4. Filter converts: 10.0 pips × 0.0001 = 0.0010 price units
5. Filter compares: ATR value vs 0.0010
```

**Benefits**:
- ✅ Filters work for all asset classes (forex, stocks, futures, crypto)
- ✅ Unit definitions centralized in Market Profile
- ✅ Easy to add new asset classes (just update Market Profile)
- ✅ No hardcoded unit values in filters
- ✅ Consistent unit handling across all filters

### MarketSpec Enhancement

**Location**: `engine/market.py` (add point_value and tick_value fields)

```python
@dataclass
class MarketSpec:
    """Market specification defining market-specific trading rules."""
    
    symbol: str
    exchange: str
    asset_class: Literal["forex", "crypto", "stock", "futures"]
    market_type: Literal["spot", "futures"] = "spot"
    leverage: float = 1.0
    contract_size: Optional[float] = None  # For forex/futures
    pip_value: Optional[float] = None  # For forex (e.g., 0.0001)
    point_value: Optional[float] = None  # For stocks/indices (e.g., 0.01 for stocks, 1.0 for indices)
    tick_value: Optional[float] = None  # For futures (e.g., 0.25 for ES)
    min_trade_size: float = 0.01
    price_precision: int = 5
    quantity_precision: int = 2
    commission_rate: float = 0.0004
    slippage_ticks: float = 0.0
    
    # ... existing methods ...
    
    def get_price_unit(self) -> str:
        """
        Get the primary price unit name for this asset class.
        
        Returns:
            Unit name: 'pips', 'points', 'ticks', or 'price_units'
        """
        if self.asset_class == 'forex':
            return 'pips'
        elif self.asset_class == 'stock':
            return 'points'
        elif self.asset_class == 'futures':
            return 'ticks'
        else:
            return 'price_units'
    
    def convert_to_price_units(self, value: float, unit_type: str) -> float:
        """
        Convert value from specified unit type to price units.
        
        Args:
            value: Value in specified unit
            unit_type: 'pips', 'points', 'ticks', or 'price_units'
            
        Returns:
            Value in price units
        """
        if unit_type == 'pips':
            if self.asset_class != 'forex':
                raise ValueError(f"Pips not applicable for {self.asset_class}")
            return value * (self.pip_value or 0.0001)
        elif unit_type == 'points':
            if self.asset_class == 'stock':
                return value * 0.01
            elif self.point_value:
                return value * self.point_value
            elif self.tick_value:
                return value * self.tick_value
            else:
                return value * 1.0
        elif unit_type == 'ticks':
            if self.tick_value:
                return value * self.tick_value
            else:
                return value * 1.0
        else:  # price_units
            return value
```

## Filter System Details

### 1. News Filter (Enhanced)

**Features**:
- **5 News Slots**: Each slot can be independently configured
- **Time-Based**: Each slot has start time and end time (UTC+00)
- **Buffer System**: Configurable minutes before/after news (default: 15 minutes)
- **Enable/Disable**: Each slot can be turned on/off independently
- **Symbol Filtering**: Optional symbol-specific filtering per slot
- **Default High-Impact News**: Pre-configured with 5 most popular high-impact news times

**Default News Slots (High-Impact Economic Events)**:

1. **NFP (Non-Farm Payrolls)**: First Friday of month, 13:30 GMT
   - Affects: USD pairs, indices
   - Default buffer: 15 min before/after

2. **FOMC (Federal Reserve)**: Various dates, 19:00 GMT
   - Affects: USD pairs, indices
   - Default buffer: 15 min before/after

3. **ECB (European Central Bank)**: Various dates, 13:45 GMT
   - Affects: EUR pairs
   - Default buffer: 15 min before/after

4. **BOE (Bank of England)**: Various dates, 12:00 GMT
   - Affects: GBP pairs
   - Default buffer: 15 min before/after

5. **US CPI (Consumer Price Index)**: Monthly (~13th), 13:30 GMT
   - Affects: USD pairs, indices
   - Default buffer: 15 min before/after

**Configuration Structure**:
```yaml
news_filter:
  enabled: true
  default_buffer_minutes: 15  # Default buffer before/after news
  slots:
    slot_1:
      enabled: true
      name: "NFP"
      start_time: "13:15"  # 15 min before news (13:30)
      end_time: "13:45"    # 15 min after news (13:30)
      buffer_before: 15
      buffer_after: 15
      symbols: []  # Empty = all symbols
    slot_2:
      enabled: true
      name: "FOMC"
      start_time: "18:45"
      end_time: "19:15"
      buffer_before: 15
      buffer_after: 15
      symbols: []
    slot_3:
      enabled: true
      name: "ECB"
      start_time: "13:30"
      end_time: "14:00"
      buffer_before: 15
      buffer_after: 15
      symbols: []
    slot_4:
      enabled: true
      name: "BOE"
      start_time: "11:45"
      end_time: "12:15"
      buffer_before: 15
      buffer_after: 15
      symbols: []
    slot_5:
      enabled: true
      name: "US CPI"
      start_time: "13:15"
      end_time: "13:45"
      buffer_before: 15
      buffer_after: 15
      symbols: []
```

### 2. Time Blackout Filter (New)

**Features**:
- **5 Blackout Slots**: Each slot can be independently configured
- **Direct Signal Blocking**: Blocks signal generation during configured time windows
- **Time-Based**: Each slot has start time and end time (UTC+00)
- **Enable/Disable**: Each slot can be turned on/off independently
- **Day Filtering**: Optional day-of-week filtering per slot
- **Symbol Filtering**: Optional symbol-specific filtering per slot

**Configuration Structure**:
```yaml
time_blackouts:
  enabled: true
  slots:
    slot_1:
      enabled: true
      name: "Lunch Break"
      start_time: "13:00"
      end_time: "13:45"
      days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
      symbols: []  # Empty = all symbols
    slot_2:
      enabled: false
      name: "Custom Block 1"
      start_time: "14:00"
      end_time: "14:30"
      days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
      symbols: []
    slot_3:
      enabled: false
      name: "Custom Block 2"
      start_time: ""
      end_time: ""
      days: []
      symbols: []
    slot_4:
      enabled: false
      name: "Custom Block 3"
      start_time: ""
      end_time: ""
      days: []
      symbols: []
    slot_5:
      enabled: false
      name: "Custom Block 4"
      start_time: ""
      end_time: ""
      days: []
      symbols: []
```

**Behavior**:
- When a time blackout is active, **NO signals are generated** during that time window
- This is different from news filter which blocks trades but allows signals
- Useful for avoiding known low-liquidity or high-volatility periods

### 3. Position Flattening (Enhanced)

**Features**:
- **Flatten Time**: Configurable time to flatten all positions (default: 21:30 GMT)
- **Stop Signal Search**: Optional minutes before flatten time to stop searching for new signals
- **All Times UTC+00**: Consistent timezone handling

**Configuration Structure**:
```yaml
execution:
  flatten_enabled: true
  flatten_time: "21:30"  # UTC+00, format: "HH:MM"
  stop_signal_search_minutes_before: 30  # Optional: stop signal search 30 min before flatten
```

**Behavior**:
- At `flatten_time`, all open positions are closed
- If `stop_signal_search_minutes_before` is set, signal generation stops that many minutes before flatten time
- Example: If flatten_time is 21:30 and stop_signal_search_minutes_before is 30, signal search stops at 21:00

### 4. ATR Threshold Filters (New)

**Features**:
- **Min ATR Threshold**: Optional minimum ATR value (filters out low volatility periods)
- **Max ATR Threshold**: Optional maximum ATR value (filters out extreme volatility periods)
- **ATR Period**: Configurable ATR calculation period (default: 14)

**Configuration Structure**:
```yaml
regime_filters:
  atr_threshold:
    enabled: true
    atr_period: 14
    min_atr: null  # Optional: minimum ATR value (null = no minimum)
    max_atr: null  # Optional: maximum ATR value (null = no maximum)
    min_atr_pips: null  # Optional: minimum ATR in pips (alternative to min_atr)
    max_atr_pips: null  # Optional: maximum ATR in pips (alternative to max_atr)
```

**Behavior**:
- If `min_atr` is set and current ATR < min_atr: Filter fails (signal rejected)
- If `max_atr` is set and current ATR > max_atr: Filter fails (signal rejected)
- Useful for filtering out periods with unsuitable volatility levels

## Configuration System

### Master Configuration

**Location**: `config/master_filters.yml`

```yaml
# Master Filter Configuration
# These are defaults applied to all strategies
# Can be overridden at strategy level

master_filters:
  calendar_filters:
    master_filters_enabled: true
    day_of_week:
      enabled: true
      allowed_days: [0, 1, 2, 3, 4]  # Mon-Fri
    month_of_year:
      enabled: false
      allowed_months: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    trading_sessions:
      Asia:
        enabled: true
        start: "23:00"
        end: "08:00"
      London:
        enabled: true
        start: "07:00"
        end: "16:00"
      NewYork:
        enabled: true
        start: "13:00"
        end: "21:00"
    time_blackouts_enabled: true
    time_blackouts:
      slots:
        slot_1:
          enabled: false
          name: "Lunch Break"
          start_time: "13:00"
          end_time: "13:45"
          days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
          symbols: []
        slot_2:
          enabled: false
          start_time: ""
          end_time: ""
          days: []
          symbols: []
        slot_3:
          enabled: false
          start_time: ""
          end_time: ""
          days: []
          symbols: []
        slot_4:
          enabled: false
          start_time: ""
          end_time: ""
          days: []
          symbols: []
        slot_5:
          enabled: false
          start_time: ""
          end_time: ""
          days: []
          symbols: []
  
  news_filter:
    enabled: true
    default_buffer_minutes: 15
    slots:
      slot_1:
        enabled: true
        name: "NFP"
        start_time: "13:15"
        end_time: "13:45"
        buffer_before: 15
        buffer_after: 15
        symbols: []
      slot_2:
        enabled: true
        name: "FOMC"
        start_time: "18:45"
        end_time: "19:15"
        buffer_before: 15
        buffer_after: 15
        symbols: []
      slot_3:
        enabled: true
        name: "ECB"
        start_time: "13:30"
        end_time: "14:00"
        buffer_before: 15
        buffer_after: 15
        symbols: []
      slot_4:
        enabled: true
        name: "BOE"
        start_time: "11:45"
        end_time: "12:15"
        buffer_before: 15
        buffer_after: 15
        symbols: []
      slot_5:
        enabled: true
        name: "US CPI"
        start_time: "13:15"
        end_time: "13:45"
        buffer_before: 15
        buffer_after: 15
        symbols: []
  
  regime_filters:
    adx:
      enabled: true
      period: 14
      min_forex: 23.0
      min_indices: 20.0
      min_metals: 25.0
      min_commodities: 26.0
      min_crypto: 28.0
    atr_percentile:
      enabled: true
      lookback: 100
      min_percentile: 30.0
    atr_threshold:
      enabled: false
      atr_period: 14
      min_atr: null
      max_atr: null
      min_atr_pips: null
      max_atr_pips: null
    ema_expansion:
      enabled: true
      lookback: 5
    swing:
      enabled: false
      lookback: 50
      min_trend_score: 0.60
    composite:
      enabled: true
      min_score: 0.55
      use_adx: true
      use_atr_percentile: true
      use_ema_expansion: true
      use_swing: false
```

### Strategy Configuration

**Location**: `strategies/my_trend_strategy/config.yml`

```yaml
strategy_name: my_trend_strategy
strategy_type: trend  # Uses TrendStrategy base class

# Strategy-specific settings
market:
  symbol: EURUSD
  exchange: oanda

timeframes:
  signal_tf: "1h"
  entry_tf: "15m"

# Execution settings (including flatten)
execution:
  flatten_enabled: true
  flatten_time: "21:30"  # UTC+00
  stop_signal_search_minutes_before: 30  # Stop signal search at 21:00

# Filter overrides (merges with master config)
filters:
  calendar_filters:
    day_of_week:
      enabled: false  # Override: Trade all days
    time_blackouts:
      slots:
        slot_1:
          enabled: true  # Override: Enable lunch break blackout
          start_time: "13:00"
          end_time: "13:45"
  news_filter:
    slots:
      slot_1:
        enabled: false  # Override: Disable NFP filter for this strategy
      slot_2:
        buffer_before: 30  # Override: Increase FOMC buffer to 30 minutes
        buffer_after: 30
  regime_filters:
    adx:
      min_forex: 25.0  # Override: Stricter requirement
    atr_threshold:
      enabled: true  # Override: Enable ATR thresholds
      min_atr_pips: 10.0  # Minimum 10 pips ATR
      max_atr_pips: 50.0  # Maximum 50 pips ATR
    ema_expansion:
      enabled: false  # Override: Disable for this strategy

# Position management
position_management:
  stop_loss:
    type: "ma_buffer"
    ma_type: "SMA"
    ma_period: 50
    buffer_pips: 3.0
  trailing_stop:
    enabled: true
    type: "ema"
    ema_period: 21
    activation_r: 0.5
  partial_exit:
    enabled: true
    level_r: 2.0
    exit_pct: 80.0
```

## Trade Management Abstraction

Similar to the filter abstraction, trade management has been abstracted into a reusable system.

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│              Trade Management System                      │
│                                                          │
│  TradeManagementManager                                   │
│    - Merges master config + strategy config              │
│    - Provides unified config to backtest engine          │
│                                                          │
│  ExitResolver                                            │
│    - Resolves simultaneous exit conditions                │
│    - Prioritizes most profitable/smallest loss          │
│                                                          │
│  Master Config: config/master_trade_management.yml       │
│  Strategy Config: strategies/*/config.yml                │
└──────────────────────────────────────────────────────────┘
```

### Components

#### 1. TradeManagementManager

**Location**: `engine/trade_management/manager.py`

**Purpose**: Merges master trade management config with strategy-specific overrides.

**Features**:
- Loads master config from `config/master_trade_management.yml`
- Merges with strategy config (strategy config overrides master)
- Provides unified configuration to backtest engine
- Supports multiple TP/partial exit levels
- Supports multiple trailing stop types

**Example**:
```python
from engine.trade_management import TradeManagementManager

# In backtest engine
tm_manager = TradeManagementManager(
    master_config_path="config/master_trade_management.yml",
    strategy_config=strategy.config
)

# Get unified config
trailing_config = tm_manager.get_trailing_stop_config()
tp_levels = tm_manager.get_take_profit_levels()
partial_levels = tm_manager.get_partial_exit_levels()
```

#### 2. ExitResolver

**Location**: `engine/trade_management/exit_resolver.py`

**Purpose**: Resolves simultaneous exit conditions using industry-standard rules.

**Rules**:
- **Hard Stop Loss**: Always honored (safety first)
- **Positive Exits**: Choose most profitable (highest price for longs, lowest for shorts)
- **Negative Exits**: Choose smallest loss (closest to entry)

**Example**:
```python
from engine.trade_management import ExitResolver, ExitCondition, ExitType

# Collect all exit conditions
exits = [
    ExitCondition(ExitType.HARD_STOP, price=1.0650, r_multiple=-1.0),
    ExitCondition(ExitType.TRAILING_STOP, price=1.0680, r_multiple=0.5),
    ExitCondition(ExitType.TAKE_PROFIT, price=1.0700, r_multiple=2.0),
]

# Resolve
best_exit = ExitResolver.resolve_simultaneous_exits(
    exits=exits,
    direction=1,  # Long
    entry_price=1.0660
)
# Returns: ExitCondition(ExitType.TAKE_PROFIT, price=1.0700, r_multiple=2.0)
```

### Configuration Structure

#### Master Config (`config/master_trade_management.yml`)

```yaml
min_stop_distance:
  enabled: true
  type: "fixed"  # "atr" or "fixed"
  fixed_distance_pips: 6.0
  atr_multiplier: 1.5
  atr_period: 14
  mode: "skip"  # "skip" or "use_fixed"/"use_atr"

trailing_stop:
  enabled: false
  type: "EMA"
  length: 21
  activation_type: "r_based"
  activation_r: 1.5
  stepped: false

take_profit:
  enabled: false
  levels:
    - enabled: true
      type: "r_based"
      target_r: 3.0
      exit_pct: 100.0

partial_exit:
  enabled: false
  levels:
    - enabled: true
      type: "r_based"
      level_r: 1.5
      exit_pct: 80.0
```

#### Strategy Config Override

```yaml
# In strategies/ema_crossover/config.yml
stop_loss:
  min_stop_distance:
    enabled: true
    type: "fixed"
    fixed_distance_pips: 6.0
    mode: "use_fixed"

trailing_stop:
  enabled: true
  activation_r: 1.5
  stepped: true

partial_exit:
  enabled: true
  levels:
    - enabled: true
      type: "r_based"
      level_r: 1.5
      exit_pct: 80.0
```

### Supported Features

#### Stop Loss
- **Types**: SMA, EMA
- **Minimum Stop Distance**: ATR-based or fixed distance
- **Buffer**: Configurable in pips/points/price

#### Trailing Stop
- **Types**: EMA, SMA, ATR, percentage, fixed distance
- **Activation**: R-based, ATR-based, price-based, time-based
- **Stepped**: R-based stepped trailing (example: ema_crossover)

#### Take Profit
- **Multiple Levels**: Support for multiple TP levels
- **Types**: R-based, ATR-based, price-based, percentage-based, time-based
- **Partial Exits**: Can exit partial position at each level

#### Partial Exit
- **Multiple Levels**: Support for multiple partial exit levels
- **Types**: R-based, ATR-based, price-based, percentage-based, time-based
- **Sequential Execution**: Levels execute in order

### Benefits

✅ **Consistency**: Same trade management logic across all strategies
✅ **Flexibility**: Strategy-level overrides for customization
✅ **Maintainability**: Fix bugs once, all strategies benefit
✅ **Extensibility**: Easy to add new exit types or levels
✅ **Industry Standards**: Follows best practices for exit resolution

## Implementation Details

### 1. Filter Base Classes

**Location**: `strategies/filters/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict
import pandas as pd

@dataclass
class FilterContext:
    """Context passed to filters for decision making."""
    timestamp: pd.Timestamp
    symbol: str
    signal_direction: int  # 1=long, -1=short
    signal_data: pd.Series  # Current signal bar data
    df_by_tf: Dict[str, pd.DataFrame]  # All timeframe data
    indicators: Dict[str, pd.Series]  # Pre-calculated indicators

@dataclass
class FilterResult:
    """Result from filter check."""
    passed: bool
    reason: Optional[str] = None
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class FilterBase(ABC):
    """Base class for all filters."""
    
    def __init__(self, config):
        self.config = config
        self.enabled = getattr(config, 'enabled', True)
        self.name = self.__class__.__name__
    
    @abstractmethod
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if filter passes.
        
        Args:
            context: FilterContext with signal and market data
            
        Returns:
            FilterResult indicating pass/fail and reason
        """
        pass
    
    def is_enabled(self) -> bool:
        """Check if filter is enabled."""
        return self.enabled
    
    def _create_pass_result(self, metadata: Dict = None) -> FilterResult:
        """Helper to create pass result."""
        return FilterResult(passed=True, metadata=metadata or {})
    
    def _create_fail_result(self, reason: str, metadata: Dict = None) -> FilterResult:
        """Helper to create fail result."""
        return FilterResult(
            passed=False, 
            reason=reason,
            metadata=metadata or {}
        )
```

### 2. News Filter Implementation

**Location**: `strategies/filters/news/news_filter.py`

```python
from strategies.filters.base import FilterBase, FilterContext, FilterResult
from typing import List, Dict
import pandas as pd
from datetime import time, timedelta

class NewsSlot:
    """Individual news slot configuration."""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', False)
        self.name = config.get('name', '')
        self.start_time_str = config.get('start_time', '')
        self.end_time_str = config.get('end_time', '')
        self.buffer_before = config.get('buffer_before', 15)
        self.buffer_after = config.get('buffer_after', 15)
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
        """Parse time string 'HH:MM' to time object."""
        parts = time_str.split(':')
        return time(int(parts[0]), int(parts[1]))
    
    def is_active(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if news slot is active for given timestamp and symbol.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if news slot is active
        """
        if not self.enabled:
            return False
        
        if self.start_time is None or self.end_time is None:
            return False
        
        # Check symbol filtering
        if self.symbols and symbol not in self.symbols:
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

class NewsFilter(FilterBase):
    """News filter with 5 configurable slots."""
    
    def __init__(self, config):
        super().__init__(config)
        self.default_buffer = config.get('default_buffer_minutes', 15)
        
        # Initialize 5 news slots
        slots_config = config.get('slots', {})
        self.slots = []
        for i in range(1, 6):
            slot_key = f'slot_{i}'
            slot_config = slots_config.get(slot_key, {})
            self.slots.append(NewsSlot(slot_config))
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if any news slot is active.
        
        Returns:
            FilterResult indicating if signal should be blocked
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Check each slot
        for slot in self.slots:
            if slot.is_active(context.timestamp, context.symbol):
                return self._create_fail_result(
                    reason=f"News filter active: {slot.name} ({slot.start_time_str} - {slot.end_time_str})",
                    metadata={
                        'filter': 'news',
                        'slot_name': slot.name,
                        'start_time': slot.start_time_str,
                        'end_time': slot.end_time_str
                    }
                )
        
        return self._create_pass_result()
```

### 3. Time Blackout Filter Implementation

**Location**: `strategies/filters/calendar/time_blackout_filter.py`

```python
from strategies.filters.base import FilterBase, FilterContext, FilterResult
from typing import List, Dict
import pandas as pd
from datetime import time

class TimeBlackoutSlot:
    """Individual time blackout slot configuration."""
    
    def __init__(self, config: Dict):
        self.enabled = config.get('enabled', False)
        self.name = config.get('name', '')
        self.start_time_str = config.get('start_time', '')
        self.end_time_str = config.get('end_time', '')
        self.days = config.get('days', [])  # List of day names: ["Mon", "Tue", etc.]
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
        """Parse time string 'HH:MM' to time object."""
        if not time_str:
            return None
        parts = time_str.split(':')
        return time(int(parts[0]), int(parts[1]))
    
    def _day_name_to_weekday(self, day_name: str) -> int:
        """Convert day name to weekday (0=Monday, 6=Sunday)."""
        day_map = {
            'Mon': 0, 'Monday': 0,
            'Tue': 1, 'Tuesday': 1,
            'Wed': 2, 'Wednesday': 2,
            'Thu': 3, 'Thursday': 3,
            'Fri': 4, 'Friday': 4,
            'Sat': 5, 'Saturday': 5,
            'Sun': 6, 'Sunday': 6
        }
        return day_map.get(day_name, -1)
    
    def is_active(self, timestamp: pd.Timestamp, symbol: str) -> bool:
        """
        Check if time blackout slot is active.
        
        Args:
            timestamp: Timestamp to check (assumed UTC+00)
            symbol: Symbol to check
            
        Returns:
            True if blackout is active (signal should be blocked)
        """
        if not self.enabled:
            return False
        
        if self.start_time is None or self.end_time is None:
            return False
        
        # Check day filtering
        if self.days:
            weekday = timestamp.weekday()  # 0=Monday, 6=Sunday
            day_names = [self._day_name_to_weekday(d) for d in self.days]
            if weekday not in day_names:
                return False
        
        # Check symbol filtering
        if self.symbols and symbol not in self.symbols:
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

class TimeBlackoutFilter(FilterBase):
    """Time blackout filter with 5 configurable slots."""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize 5 blackout slots
        slots_config = config.get('slots', {})
        self.slots = []
        for i in range(1, 6):
            slot_key = f'slot_{i}'
            slot_config = slots_config.get(slot_key, {})
            self.slots.append(TimeBlackoutSlot(slot_config))
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if any time blackout slot is active.
        
        Returns:
            FilterResult indicating if signal should be blocked
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Check each slot
        for slot in self.slots:
            if slot.is_active(context.timestamp, context.symbol):
                return self._create_fail_result(
                    reason=f"Time blackout active: {slot.name} ({slot.start_time_str} - {slot.end_time_str})",
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
        Check if signal generation should be blocked at this time.
        
        This is called BEFORE signal generation to prevent signals during blackout.
        """
        if not self.enabled:
            return False
        
        for slot in self.slots:
            if slot.is_active(timestamp, symbol):
                return True
        
        return False
```

### 4. ATR Threshold Filter Implementation

**Location**: `strategies/filters/regime/atr_threshold_filter.py`

```python
from strategies.filters.base import FilterBase, FilterContext, FilterResult
import pandas as pd
import numpy as np

class ATRThresholdFilter(FilterBase):
    """ATR threshold filter (min/max ATR values)."""
    
    def __init__(self, config):
        super().__init__(config)
        self.atr_period = config.get('atr_period', 14)
        self.min_atr = config.get('min_atr', None)
        self.max_atr = config.get('max_atr', None)
        self.min_atr_pips = config.get('min_atr_pips', None)
        self.max_atr_pips = config.get('max_atr_pips', None)
        self.pip_size = config.get('pip_size', 0.0001)  # Default for forex
    
    def check(self, context: FilterContext) -> FilterResult:
        """
        Check if ATR is within configured thresholds.
        
        Returns:
            FilterResult indicating if ATR threshold passes
        """
        if not self.enabled:
            return self._create_pass_result()
        
        # Get ATR value from signal data
        if 'atr' not in context.signal_data or pd.isna(context.signal_data['atr']):
            return self._create_fail_result(
                reason="ATR value not available",
                metadata={'filter': 'atr_threshold', 'value': None}
            )
        
        atr_value = float(context.signal_data['atr'])
        
        # Convert pip thresholds to price thresholds if needed
        min_threshold = self.min_atr
        max_threshold = self.max_atr
        
        if self.min_atr_pips is not None:
            min_threshold = self.min_atr_pips * self.pip_size
        
        if self.max_atr_pips is not None:
            max_threshold = self.max_atr_pips * self.pip_size
        
        # Check minimum threshold
        if min_threshold is not None and atr_value < min_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} below minimum threshold {min_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': min_threshold,
                    'type': 'min'
                }
            )
        
        # Check maximum threshold
        if max_threshold is not None and atr_value > max_threshold:
            return self._create_fail_result(
                reason=f"ATR {atr_value:.4f} above maximum threshold {max_threshold:.4f}",
                metadata={
                    'filter': 'atr_threshold',
                    'value': atr_value,
                    'threshold': max_threshold,
                    'type': 'max'
                }
            )
        
        return self._create_pass_result(
            metadata={
                'filter': 'atr_threshold',
                'value': atr_value,
                'min_threshold': min_threshold,
                'max_threshold': max_threshold
            }
        )
```

### 5. Filter Manager Enhancement

**Location**: `strategies/filters/manager.py` (update to include new filters)

```python
# Add to _build_calendar_filters method:
def _build_calendar_filters(self, config: Dict) -> List[FilterBase]:
    """Build calendar filter chain."""
    filters = []
    calendar_cfg = config.get('calendar_filters', {})
    
    # ... existing filters ...
    
    # Time blackout filter
    if calendar_cfg.get('time_blackouts_enabled', False):
        from strategies.filters.calendar import TimeBlackoutFilter
        filters.append(TimeBlackoutFilter(calendar_cfg.get('time_blackouts', {})))
    
    return filters

# Add to _build_regime_filters method:
def _build_regime_filters(self, config: Dict) -> List[FilterBase]:
    """Build regime filter chain."""
    filters = []
    regime_cfg = config.get('regime_filters', {})
    
    # ... existing filters ...
    
    # ATR threshold filter
    if regime_cfg.get('atr_threshold', {}).get('enabled', False):
        from strategies.filters.regime import ATRThresholdFilter
        filters.append(ATRThresholdFilter(regime_cfg['atr_threshold']))
    
    return filters

# Add to _build_news_filters method:
def _build_news_filters(self, config: Dict) -> List[FilterBase]:
    """Build news filter chain."""
    filters = []
    news_cfg = config.get('news_filter', {})
    
    if news_cfg.get('enabled', False):
        from strategies.filters.news import NewsFilter
        filters.append(NewsFilter(news_cfg))
    
    return filters
```

### 6. Signal Generation with Time Blackout

**Location**: `strategies/base/strategy_base.py` (enhancement)

```python
def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Generate trading signals with time blackout checking.
    """
    signals = []
    
    # Get filter manager if available
    filter_manager = getattr(self, 'filter_manager', None)
    
    # Check for time blackout filter
    time_blackout_filter = None
    if filter_manager:
        for filter_obj in filter_manager.filters:
            if hasattr(filter_obj, 'should_block_signal_generation'):
                time_blackout_filter = filter_obj
                break
    
    # Iterate through data
    for idx, row in df_by_tf[self.config.timeframes.signal_tf].iterrows():
        # Check time blackout BEFORE generating signal
        if time_blackout_filter:
            if time_blackout_filter.should_block_signal_generation(idx, self.config.market.symbol):
                continue  # Skip signal generation during blackout
        
        # Generate signal (existing logic)
        signal = self._generate_signal(row, df_by_tf)
        if signal is not None:
            signals.append(signal)
    
    return pd.DataFrame(signals)
```

### 7. Execution Config Enhancement

**Location**: `config/schema.py` (update ExecutionConfig)

```python
class ExecutionConfig(BaseModel):
    """Execution configuration."""
    max_positions: int = Field(default=1, ge=1)
    flatten_enabled: bool = Field(
        default=True,
        description="Enable end-of-day flattening of all open positions"
    )
    flatten_time: Optional[str] = Field(
        default="21:30",
        description="Time to flatten all positions (format: 'HH:MM' in UTC+00). Example: '21:30' for 9:30 PM GMT"
    )
    stop_signal_search_minutes_before: Optional[int] = Field(
        default=None,
        ge=0,
        description="Stop searching for new signals X minutes before flatten time. "
                   "Example: If flatten_time is 21:30 and this is 30, signal search stops at 21:00. "
                   "None = no early stop (default)"
    )
    max_wait_bars: Optional[int] = Field(
        default=None,
        ge=1,
        description="Maximum bars to wait for entry condition after signal. If entry condition not met within this time, signal is skipped. Default: None (no limit)"
    )
```

## Code Examples

### Example 1: Strategy with All New Features

```python
# strategies/my_strategy/strategy.py
from strategies.types import TrendStrategy
from typing import Dict
import pandas as pd

class MyStrategy(TrendStrategy):
    """Strategy using all new filter features."""
    
    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Time blackout is checked automatically in base class
        # before signal generation
        
        # Generate base signals
        signals = self._generate_my_signals(df_by_tf)
        
        # Apply other filters (news, ATR threshold, etc.)
        filtered_signals = self.apply_filters(signals, df_by_tf)
        
        return filtered_signals
```

### Example 2: Configuration with All Features

```yaml
# strategies/my_strategy/config.yml
strategy_name: my_strategy
strategy_type: trend

execution:
  flatten_enabled: true
  flatten_time: "21:30"  # UTC+00
  stop_signal_search_minutes_before: 30  # Stop at 21:00

filters:
  calendar_filters:
    time_blackouts:
      slots:
        slot_1:
          enabled: true
          name: "Lunch Break"
          start_time: "13:00"
          end_time: "13:45"
          days: ["Mon", "Tue", "Wed", "Thu", "Fri"]
  
  news_filter:
    enabled: true
    slots:
      slot_1:
        enabled: true
        name: "NFP"
        start_time: "13:15"
        end_time: "13:45"
      slot_2:
        enabled: true
        name: "FOMC"
        start_time: "18:45"
        end_time: "19:15"
  
  regime_filters:
    atr_threshold:
      enabled: true
      min_atr_pips: 10.0
      max_atr_pips: 50.0
```

## Migration Strategy

### Phase 1: Foundation (Non-Breaking)

1. Create filter base classes and manager
2. Implement new filters (News, Time Blackout, ATR Threshold)
3. Update ExecutionConfig schema
4. Keep existing strategies unchanged
5. Test filter system independently

### Phase 2: Integration

1. Integrate time blackout check into signal generation
2. Integrate flatten time logic into engine
3. Update filter manager to include new filters
4. Test with one strategy

### Phase 3: Gradual Migration

1. Migrate strategies one at a time
2. Maintain backward compatibility
3. Update documentation

## Testing Strategy

### Unit Tests

```python
# tests/filters/test_news_filter.py
def test_news_filter_blocks_during_slot():
    """Test news filter blocks signals during active slot."""
    config = {
        'enabled': True,
        'slots': {
            'slot_1': {
                'enabled': True,
                'name': 'NFP',
                'start_time': '13:15',
                'end_time': '13:45'
            }
        }
    }
    
    filter_obj = NewsFilter(config)
    
    # Test during news slot
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-05 13:30:00'),  # During NFP
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series(),
        df_by_tf={},
        indicators={}
    )
    
    result = filter_obj.check(context)
    assert result.passed is False
    assert 'NFP' in result.reason

# tests/filters/test_time_blackout_filter.py
def test_time_blackout_blocks_signal_generation():
    """Test time blackout blocks signal generation."""
    # Similar test structure
    pass

# tests/filters/test_atr_threshold_filter.py
def test_atr_threshold_min():
    """Test ATR threshold minimum check."""
    # Test implementation
    pass
```

## Industry Standards Alignment

This architecture follows established industry patterns:

1. **Filter Chain Pattern**: Sequential filter application with short-circuiting
2. **Strategy Pattern**: Different strategy types with common interface
3. **Template Method Pattern**: Common workflow with customizable steps
4. **Composition over Inheritance**: Filters as composable components
5. **Configuration over Code**: Filters configured, not hardcoded
6. **Single Responsibility**: Each filter has one clear purpose
7. **Open/Closed Principle**: Open for extension, closed for modification

## Summary

This architecture provides:

- **Enhanced News Filter**: 5 slots with configurable times and buffers
- **Time Blackout Filter**: 5 slots for blocking signal generation
- **Position Flattening**: Configurable time with optional early stop
- **ATR Thresholds**: Min/max ATR filtering
- **All Times UTC+00**: Consistent timezone handling
- **Reusable Components**: Filters implemented once, used everywhere
- **Strategy Templates**: Pre-configured types for common patterns
- **Flexible Configuration**: Master defaults with strategy overrides

The system is designed to be:
- **Non-breaking**: Existing strategies continue to work
- **Gradual**: Can be migrated incrementally
- **Testable**: Each component can be tested independently
- **Documented**: Clear architecture and examples
