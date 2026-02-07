"""Unit tests for ADX filter."""

import pytest
import pandas as pd
from strategies.filters.regime.adx_filter import ADXFilter
from strategies.filters.base import FilterContext
from engine.market import MarketSpec


def test_adx_filter_pass():
    """Test ADX filter passes when ADX is above threshold."""
    config = {
        'enabled': True,
        'min_forex': 23.0,
        'min_indices': 20.0,
        'min_metals': 25.0,
        'min_commodities': 26.0,
        'min_crypto': 28.0
    }
    
    filter_obj = ADXFilter(config)
    
    # Create market spec for EURUSD (forex)
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex',
        pip_value=0.0001,
        contract_size=100000
    )
    
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 25.0}),
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context)
    assert result.passed is True
    assert result.reason is None
    assert result.metadata['value'] == 25.0
    assert result.metadata['threshold'] == 23.0


def test_adx_filter_fail():
    """Test ADX filter fails when ADX is below threshold."""
    config = {
        'enabled': True,
        'min_forex': 23.0,
        'min_indices': 20.0,
        'min_metals': 25.0,
        'min_commodities': 26.0,
        'min_crypto': 28.0
    }
    
    filter_obj = ADXFilter(config)
    
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex',
        pip_value=0.0001,
        contract_size=100000
    )
    
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 20.0}),  # Below threshold
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context)
    assert result.passed is False
    assert 'below minimum' in result.reason.lower()
    assert result.metadata['value'] == 20.0
    assert result.metadata['threshold'] == 23.0


def test_adx_filter_disabled():
    """Test ADX filter passes when disabled."""
    config = {
        'enabled': False,
        'min_forex': 23.0
    }
    
    filter_obj = ADXFilter(config)
    
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex'
    )
    
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 10.0}),  # Would fail if enabled
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context)
    assert result.passed is True  # Passes because filter is disabled


def test_adx_filter_missing_value():
    """Test ADX filter fails when ADX value is missing."""
    config = {
        'enabled': True,
        'min_forex': 23.0
    }
    
    filter_obj = ADXFilter(config)
    
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex'
    )
    
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({}),  # No ADX value
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context)
    assert result.passed is False
    assert 'not available' in result.reason.lower()


def test_adx_filter_different_asset_classes():
    """Test ADX filter uses correct threshold for different asset classes."""
    config = {
        'enabled': True,
        'min_forex': 23.0,
        'min_indices': 20.0,
        'min_metals': 25.0,
        'min_commodities': 26.0,
        'min_crypto': 28.0
    }
    
    filter_obj = ADXFilter(config)
    
    # Test forex (EURUSD)
    market_spec_forex = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex'
    )
    context_forex = FilterContext(
        timestamp=pd.Timestamp('2024-01-01'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 24.0}),
        df_by_tf={},
        market_spec=market_spec_forex
    )
    result_forex = filter_obj.check(context_forex)
    assert result_forex.passed is True  # 24.0 > 23.0 (forex threshold)
    
    # Test indices (US500)
    market_spec_indices = MarketSpec(
        symbol='US500',
        exchange='oanda',
        asset_class='futures'  # Indices often classified as futures
    )
    context_indices = FilterContext(
        timestamp=pd.Timestamp('2024-01-01'),
        symbol='US500',
        signal_direction=1,
        signal_data=pd.Series({'adx': 21.0}),
        df_by_tf={},
        market_spec=market_spec_indices
    )
    result_indices = filter_obj.check(context_indices)
    assert result_indices.passed is True  # 21.0 > 20.0 (indices threshold, detected by symbol)
    
    # Test metals (XAUUSD)
    market_spec_metals = MarketSpec(
        symbol='XAUUSD',
        exchange='oanda',
        asset_class='forex'  # Gold often classified as forex
    )
    context_metals = FilterContext(
        timestamp=pd.Timestamp('2024-01-01'),
        symbol='XAUUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 26.0}),
        df_by_tf={},
        market_spec=market_spec_metals
    )
    result_metals = filter_obj.check(context_metals)
    assert result_metals.passed is True  # 26.0 > 25.0 (metals threshold, detected by symbol)
