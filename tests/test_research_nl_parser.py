from __future__ import annotations

from research.nl_parser import parse_english_strategy
from research.spec import StrategySpec


def test_parse_english_strategy_requires_symbol_and_tf_when_missing():
    txt = """
    Go long when EMA(20) crosses above EMA(50)
    Stop ATR(14) * 3
    """.strip()

    res = parse_english_strategy(txt, name="nl_test")
    keys = {c.key for c in res.clarifications}
    assert "market.symbol" in keys
    assert "entry_tf" in keys
    assert "condition_tf" in keys


def test_parse_english_strategy_happy_path_validates_after_defaults():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when EMA(20) crosses above EMA(50) on 4h
    Enter short when close < SMA(200) on 4h
    Stop ATR(14) * 3
    Take profit 2R 50%, 3R 50%
    Partial exit 1.5R 80%
    Trailing EMA 21 after 0.5R
    """.strip()

    res = parse_english_strategy(txt, name="nl_test", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.market.symbol == "BTCUSDT"
    assert spec.entry_tf == "4h"
    assert spec.long is not None and spec.long.conditions_all
    assert spec.short is not None and spec.short.conditions_all
    assert spec.trade_management.trailing_stop is not None
    assert spec.trade_management.take_profit is not None
    assert spec.trade_management.partial_exit is not None


def test_parse_english_strategy_donchian_breakout_templates():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when breakout above Donchian high 20 on 4h
    Stop ATR(20) * 2.5
    """.strip()

    res = parse_english_strategy(txt, name="nl_donchian", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None
    assert any("donchian_high" in c.expr for c in spec.long.conditions_all)
    assert any("shift" in c.expr for c in spec.long.conditions_all)


def test_parse_english_strategy_donchian_breakdown_templates():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go short when breakdown below 20-bar low on 4h
    Stop ATR(20) * 2.5
    """.strip()

    res = parse_english_strategy(txt, name="nl_donchian_short", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.short is not None
    assert any("donchian_low" in c.expr for c in spec.short.conditions_all)
    assert any("crosses_below" in c.expr for c in spec.short.conditions_all)


def test_parse_english_strategy_global_confirm_applies_to_both_sides_and_sets_extra_tfs():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when EMA(20) crosses above EMA(50) on 4h
    Enter short when close crosses below EMA(50) on 4h
    Only trade when EMA(50) > EMA(200) on 1d
    Stop 2%
    """.strip()

    res = parse_english_strategy(txt, name="nl_confirm", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None and spec.short is not None
    assert any(c.tf == "1d" for c in spec.long.conditions_all)
    assert any(c.tf == "1d" for c in spec.short.conditions_all)
    assert "1d" in spec.extra_tfs


def test_parse_english_strategy_calendar_session_filter_parses():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when EMA(20) crosses above EMA(50) on 4h
    Only trade during London session
    Stop 2%
    """.strip()

    res = parse_english_strategy(txt, name="nl_session", default_entry_tf="4h")
    spec = StrategySpec.model_validate(res.spec_dict)
    cal = spec.filters.calendar_filters
    assert cal.get("master_filters_enabled") is True
    assert cal.get("trading_sessions_enabled") is True
    sessions = cal.get("trading_sessions") or {}
    assert "London" in sessions
    assert sessions["London"].get("enabled") is True


def test_parse_english_strategy_exit_when_parses_to_exit_conditions():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when breakout above Donchian high 20 on 4h
    Exit long when breakdown below 20-bar low on 4h
    Stop ATR(20) * 2.5
    """.strip()

    res = parse_english_strategy(txt, name="nl_exit_when", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None
    assert spec.long.exit_conditions_all
    assert any("donchian_low" in c.expr for c in spec.long.exit_conditions_all)


def test_parse_english_strategy_exit_on_opposite_breakout_maps_opposite_side():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when breakout above Donchian high 20 on 4h
    Exit on opposite breakout on 4h
    Stop ATR(20) * 2.5
    """.strip()

    res = parse_english_strategy(txt, name="nl_exit_opposite", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None
    assert spec.long.exit_conditions_all
    # Opposite of long breakout is breakdown below Donchian low
    assert any("crosses_below" in c.expr and "donchian_low" in c.expr for c in spec.long.exit_conditions_all)


def test_parse_english_strategy_supports_buy_sell_and_if_synonyms():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Buy if EMA(20) crosses above EMA(50) on 4h
    Sell when close below SMA(200) on 4h
    Stop 2%
    """.strip()

    res = parse_english_strategy(txt, name="nl_buy_sell", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None and spec.long.conditions_all
    assert spec.short is not None and spec.short.conditions_all


def test_parse_english_strategy_supports_implicit_above_below_comparators():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when close is above EMA(50) on 4h
    Go short when EMA(20) is below EMA(50) on 4h
    Stop ATR(14) * 3
    """.strip()

    res = parse_english_strategy(txt, name="nl_implicit_comp", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert any(">" in c.expr and "ema(close, 50)" in c.expr for c in (spec.long.conditions_all if spec.long else []))
    assert any("<" in c.expr and "ema(close, 20)" in c.expr and "ema(close, 50)" in c.expr for c in (spec.short.conditions_all if spec.short else []))


def test_parse_english_strategy_supports_cross_up_down_synonyms():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when EMA(20) crosses up EMA(50) on 4h
    Exit long when EMA(20) crosses down EMA(50) on 4h
    Stop 2%
    """.strip()

    res = parse_english_strategy(txt, name="nl_cross_up_down", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None and spec.long.conditions_all
    assert spec.long.exit_conditions_all


def test_parse_english_strategy_lenient_mode_extracts_side_tagged_expr_lines():
    # This line contains a valid condition + timeframe, and mentions the side,
    # but does not use the strict entry templates ("go/enter/open ... long/short ...").
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    EMA(20) crosses above EMA(50) on 4h long
    Stop 2%
    """.strip()

    strict = parse_english_strategy(txt, name="nl_lenient_strict", default_entry_tf="4h", mode="strict")
    assert any(c.key == "entries" for c in strict.clarifications)

    lenient = parse_english_strategy(txt, name="nl_lenient", default_entry_tf="4h", mode="lenient")
    assert not lenient.clarifications
    spec = StrategySpec.model_validate(lenient.spec_dict)
    assert spec.long is not None and spec.long.conditions_all


def test_parse_english_strategy_supports_compound_long_short_and_rsi_above_below_without_length():
    txt = """
    Symbol: BTCUSDT
    Exchange: binance
    Go long when rsi is above 90 , go short when rsi is below 10.
    Stop 2%
    """.strip()

    res = parse_english_strategy(txt, name="nl_rsi_compound", default_entry_tf="4h")
    assert not res.clarifications

    spec = StrategySpec.model_validate(res.spec_dict)
    assert spec.long is not None and spec.long.conditions_all
    assert spec.short is not None and spec.short.conditions_all
    assert any("rsi(close, 14)" in c.expr and ">" in c.expr for c in spec.long.conditions_all)
    assert any("rsi(close, 14)" in c.expr and "<" in c.expr for c in spec.short.conditions_all)
