import pandas as pd
import pytest

from research.dsl import DSLCompileError, EvalContext, compile_condition, extract_indicator_requests
from research.indicators import ensure_indicators


def _sample_df(n: int = 20) -> pd.DataFrame:
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    df = pd.DataFrame(
        {
            "open": pd.Series(range(n), index=idx, dtype=float),
            "high": pd.Series(range(n), index=idx, dtype=float) + 1.0,
            "low": pd.Series(range(n), index=idx, dtype=float) - 1.0,
            "close": pd.Series(range(n), index=idx, dtype=float) + 0.5,
            "volume": pd.Series([100.0] * n, index=idx, dtype=float),
        }
    )
    return df


def test_extract_indicator_requests_accepts_derived_series() -> None:
    reqs = extract_indicator_requests("ema(hl2, 20) > sma(close, 5)", tf="1h")
    assert ("ema", "1h", ("hl2", 20)) in [(r.name, r.tf, r.args) for r in reqs]
    assert ("sma", "1h", ("close", 5)) in [(r.name, r.tf, r.args) for r in reqs]


def test_compile_condition_with_derived_series_and_crosses() -> None:
    expr = "crosses_above(ema(hl2, 3), ema(close, 5))"
    df = _sample_df(30)
    reqs = extract_indicator_requests(expr, tf="1h")
    df2 = ensure_indicators(df, reqs)

    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df2}, tf="1h", target_index=df2.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df2)


def test_highest_lowest_shift_and_nz() -> None:
    df = _sample_df(10)

    fn = compile_condition("highest(high, 3) >= high")
    out = fn(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert out.dtype == bool

    fn2 = compile_condition("lowest(low, 3) <= low")
    out2 = fn2(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert out2.dtype == bool

    fn3 = compile_condition("close > shift(close, 1)")
    out3 = fn3(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert out3.dtype == bool

    fn4 = compile_condition("nz(shift(close, 1), 0) >= 0")
    out4 = fn4(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert out4.dtype == bool


def test_multitimeframe_at_aligns_to_current_tf() -> None:
    df_1h = _sample_df(30)
    # 2h data: take every other bar
    df_2h = df_1h.iloc[::2].copy()

    expr = 'close > at("2h", close)'
    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df_1h, "2h": df_2h}, tf="1h", target_index=df_1h.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df_1h)


def test_multitimeframe_shorthand_series_at_tf() -> None:
    df_1h = _sample_df(30)
    df_2h = df_1h.iloc[::2].copy()

    expr = "close > close@2h"
    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df_1h, "2h": df_2h}, tf="1h", target_index=df_1h.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df_1h)


def test_multitimeframe_shorthand_atom_at_tf_function_call() -> None:
    df_1h = _sample_df(50)
    df_2h = df_1h.iloc[::2].copy()

    expr = "ema(close, 5)@2h > ema(close, 3)"
    reqs = extract_indicator_requests(expr, tf="1h")
    df_1h2 = ensure_indicators(df_1h, [r for r in reqs if r.tf == "1h"])
    df_2h2 = ensure_indicators(df_2h, [r for r in reqs if r.tf == "2h"])

    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df_1h2, "2h": df_2h2}, tf="1h", target_index=df_1h2.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df_1h2)


def test_multitimeframe_shorthand_series_inside_indicator_call() -> None:
    df_1h = _sample_df(60)
    df_4h = df_1h.iloc[::4].copy()

    expr = "ema(close@4h, 5) > ema(close, 3)"
    reqs = extract_indicator_requests(expr, tf="1h")

    df_1h2 = ensure_indicators(df_1h, [r for r in reqs if r.tf == "1h"])
    df_4h2 = ensure_indicators(df_4h, [r for r in reqs if r.tf == "4h"])

    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df_1h2, "4h": df_4h2}, tf="1h", target_index=df_1h2.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df_1h2)


def test_multitimeframe_shorthand_atom_at_tf_parenthesized_expression() -> None:
    df_1h = _sample_df(50)
    df_4h = df_1h.iloc[::4].copy()

    expr = "(close > ema(close, 5))@4h"
    reqs = extract_indicator_requests(expr, tf="1h")
    df_1h2 = ensure_indicators(df_1h, [r for r in reqs if r.tf == "1h"])
    df_4h2 = ensure_indicators(df_4h, [r for r in reqs if r.tf == "4h"])

    fn = compile_condition(expr)
    out = fn(EvalContext(df_by_tf={"1h": df_1h2, "4h": df_4h2}, tf="1h", target_index=df_1h2.index))
    assert isinstance(out, pd.Series)
    assert len(out) == len(df_1h2)


def test_rejects_unsafe_syntax() -> None:
    with pytest.raises(DSLCompileError):
        compile_condition("__import__('os').system('echo nope')")


def test_inside_bar_compiles_and_returns_bool_series() -> None:
    df = _sample_df(20)
    fn = compile_condition("inside_bar(high, low)")
    out = fn(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert isinstance(out, pd.Series)
    assert out.dtype == bool
    assert len(out) == len(df)


def test_engulfing_compiles_and_returns_bool_series() -> None:
    df = _sample_df(20)
    fn = compile_condition('engulfing(open, close, "bull")')
    out = fn(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert isinstance(out, pd.Series)
    assert out.dtype == bool
    assert len(out) == len(df)


def test_pin_bar_compiles_and_returns_bool_series() -> None:
    df = _sample_df(20)
    fn = compile_condition('pin_bar(open, high, low, close, "bull", 2.0, 1.0, 0.2)')
    out = fn(EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index))
    assert isinstance(out, pd.Series)
    assert out.dtype == bool
    assert len(out) == len(df)
