from engine.market import MarketSpec
from engine.market_units import native_distance_unit, native_to_price_distance, price_to_native_distance


def test_forex_native_unit_is_pips():
    ms = MarketSpec(
        symbol="EURUSD",
        exchange="oanda",
        asset_class="forex",
        market_type="spot",
        pip_value=0.0001,
    )
    assert native_distance_unit(ms) == "pips"
    assert native_to_price_distance(ms, 10) == 0.001
    assert price_to_native_distance(ms, 0.001) == 10


def test_futures_native_unit_is_ticks_when_tick_size_present():
    ms = MarketSpec(
        symbol="ES",
        exchange="cme",
        asset_class="futures",
        market_type="futures",
        tick_size=0.25,
    )
    assert native_distance_unit(ms) == "ticks"
    assert native_to_price_distance(ms, 4) == 1.0
    assert price_to_native_distance(ms, 1.0) == 4


def test_price_native_unit_fallback():
    ms = MarketSpec(
        symbol="BTCUSDT",
        exchange="binance",
        asset_class="crypto",
        market_type="spot",
    )
    assert native_distance_unit(ms) == "price"
    assert native_to_price_distance(ms, 12.34) == 12.34
    assert price_to_native_distance(ms, 12.34) == 12.34
