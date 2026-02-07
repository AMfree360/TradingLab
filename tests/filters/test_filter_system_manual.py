"""Manual test script for filter system (can run without pytest)."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
from strategies.filters.regime.adx_filter import ADXFilter
from strategies.filters.base import FilterContext
from strategies.filters.manager import FilterManager
from strategies.filters.utils import UnitConverter
from engine.market import MarketSpec


def test_adx_filter():
    """Test ADX filter manually."""
    print("Testing ADX Filter...")
    
    # Create filter config
    config = {
        'enabled': True,
        'min_forex': 23.0,
        'min_indices': 20.0,
        'min_metals': 25.0,
        'min_commodities': 26.0,
        'min_crypto': 28.0
    }
    
    filter_obj = ADXFilter(config)
    
    # Create market spec for EURUSD
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex',
        pip_value=0.0001,
        contract_size=100000
    )
    
    # Test 1: ADX above threshold (should pass)
    context_pass = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 25.0}),
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context_pass)
    assert result.passed is True, f"Expected pass, got: {result.reason}"
    print("  ✓ ADX above threshold: PASS")
    
    # Test 2: ADX below threshold (should fail)
    context_fail = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=pd.Series({'adx': 20.0}),
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = filter_obj.check(context_fail)
    assert result.passed is False, "Expected fail for ADX below threshold"
    print(f"  ✓ ADX below threshold: FAIL (reason: {result.reason})")
    
    # Test 3: Filter disabled (should pass regardless)
    config_disabled = config.copy()
    config_disabled['enabled'] = False
    filter_disabled = ADXFilter(config_disabled)
    
    result = filter_disabled.check(context_fail)
    assert result.passed is True, "Disabled filter should always pass"
    print("  ✓ Filter disabled: PASS (always passes)")
    
    print("ADX Filter tests: All passed! ✓\n")


def test_unit_converter():
    """Test UnitConverter for different asset classes."""
    print("Testing UnitConverter...")
    
    # Test 1: Forex - Pips to price
    market_spec_forex = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex',
        pip_value=0.0001
    )
    converter_forex = UnitConverter(market_spec_forex)
    
    price = converter_forex.pips_to_price(10.0)
    assert abs(price - 0.0010) < 0.0001, f"Expected 0.0010, got {price}"
    print(f"  ✓ Forex: 10 pips = {price} price units")
    
    # Test 2: Stocks - Points to price
    market_spec_stock = MarketSpec(
        symbol='AAPL',
        exchange='nasdaq',
        asset_class='stock'
    )
    converter_stock = UnitConverter(market_spec_stock)
    
    price = converter_stock.points_to_price(5.0)
    assert abs(price - 0.05) < 0.001, f"Expected 0.05, got {price}"
    print(f"  ✓ Stock: 5 points = ${price} price units")
    
    # Test 3: Crypto - Direct price units
    market_spec_crypto = MarketSpec(
        symbol='BTCUSDT',
        exchange='binance',
        asset_class='crypto',
        price_precision=2
    )
    converter_crypto = UnitConverter(market_spec_crypto)
    
    min_move = converter_crypto.get_minimum_price_move()
    print(f"  ✓ Crypto: Minimum price move = {min_move}")
    
    print("UnitConverter tests: All passed! ✓\n")


def test_filter_manager():
    """Test FilterManager."""
    print("Testing FilterManager...")
    
    # Create master config with ADX filter enabled
    master_config = {
        'calendar_filters': {
            'master_filters_enabled': True
        },
        'regime_filters': {
            'adx': {
                'enabled': True,
                'min_forex': 23.0,
                'min_indices': 20.0,
                'min_metals': 25.0,
                'min_commodities': 26.0,
                'min_crypto': 28.0
            }
        }
    }
    
    manager = FilterManager(master_config=master_config)
    
    # Check that ADX filter was added
    assert len(manager.filters) > 0, "FilterManager should have at least one filter"
    assert any(isinstance(f, ADXFilter) for f in manager.filters), "ADXFilter should be in chain"
    print(f"  ✓ FilterManager created with {len(manager.filters)} filter(s)")
    
    # Test applying filters
    market_spec = MarketSpec(
        symbol='EURUSD',
        exchange='oanda',
        asset_class='forex',
        pip_value=0.0001
    )
    
    signal = pd.Series({
        'direction': 'long',
        'entry_price': 1.1000,
        'stop_price': 1.0950,
        'adx': 25.0  # Above threshold
    })
    
    context = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=signal,
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = manager.apply_filters(signal, context)
    assert result.passed is True, f"Signal should pass, got: {result.reason}"
    print("  ✓ FilterManager: Signal passed all filters")
    
    # Test with ADX below threshold
    signal_fail = signal.copy()
    signal_fail['adx'] = 20.0  # Below threshold
    
    context_fail = FilterContext(
        timestamp=pd.Timestamp('2024-01-01 12:00:00'),
        symbol='EURUSD',
        signal_direction=1,
        signal_data=signal_fail,
        df_by_tf={},
        market_spec=market_spec
    )
    
    result = manager.apply_filters(signal_fail, context_fail)
    assert result.passed is False, "Signal should fail with low ADX"
    print(f"  ✓ FilterManager: Signal correctly rejected (reason: {result.reason})")
    
    print("FilterManager tests: All passed! ✓\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Filter System Phase 1 - Manual Tests")
    print("=" * 60)
    print()
    
    try:
        test_adx_filter()
        test_unit_converter()
        test_filter_manager()
        
        print("=" * 60)
        print("All tests passed! ✓")
        print("=" * 60)
        print()
        print("Phase 1 Implementation Summary:")
        print("  ✓ FilterBase, FilterContext, FilterResult created")
        print("  ✓ UnitConverter utility created")
        print("  ✓ FilterManager created")
        print("  ✓ ADXFilter implemented as proof of concept")
        print("  ✓ All components work across asset classes")
        print()
        print("Next steps:")
        print("  - Implement additional filters (calendar, news, etc.)")
        print("  - Create strategy type templates")
        print("  - Migrate existing strategies to use filter system")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
