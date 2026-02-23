import pandas as pd
from research.dsl import compile_condition, EvalContext
from gui_launcher.app import _builder_v2_preview


def _make_sample_df(n=20, freq="H"):
    idx = pd.date_range("2020-01-01", periods=n, freq=freq)
    df = pd.DataFrame(index=idx)
    # simple monotonic series to keep operations well-defined
    vals = list(range(n))
    df["open"] = [v + 0.0 for v in vals]
    df["high"] = [v + 1.0 for v in vals]
    df["low"] = [v - 1.0 for v in vals]
    df["close"] = [v + 0.5 for v in vals]
    df["volume"] = [100 + v for v in vals]
    return df


def test_preview_exprs_include_tf_annotations():
    draft = {
        "entry_tf": "1h",
        "signal_rules": [
            {"type": "z_score", "tf": "4h", "length": 10, "op": ">", "threshold": 1.5, "valid_for_bars": 2}
        ],
        "trigger_rules": [
            {"type": "volume_rejection", "tf": "30m", "vol_len": 10, "mult": 2, "valid_for_bars": 1}
        ],
        "context_rules": [
            {"type": "bollinger_context", "tf": "1d", "length": 20, "mult": 2}
        ],
        "long_enabled": True,
        "short_enabled": False,
    }

    preview = _builder_v2_preview(draft)

    # Collect expressions from preview
    exprs = []
    for it in preview["long"]["signals"]:
        exprs.append(it.get("expr", ""))
    for it in preview["long"]["triggers"]:
        exprs.append(it.get("expr", ""))
    for it in preview["long"]["context"]:
        exprs.append(it.get("expr", ""))

    # Ensure each non-empty expression includes a timeframe annotation (@...)
    for e in exprs:
        if not e or e in {"(empty)", "(unknown)"}:
            continue
        assert ("@" in e) or ("at(" in e), f"Expression missing TF annotation: {e}"


def test_compile_preview_expression_evaluates_and_aligns():
    # Use entry_tf so compilation uses a single timeframe path
    draft = {
        "entry_tf": "1h",
        "signal_rules": [
            {"type": "z_score", "tf": "", "length": 10, "op": ">", "threshold": 1.0},
            {"type": "ma_cross", "ma_type": "ema", "fast": 3, "slow": 8, "tf": ""},
        ],
        "long_enabled": True,
        "short_enabled": False,
    }

    preview = _builder_v2_preview(draft)
    sigs = preview["long"]["signals"]
    assert sigs, "No signals produced by preview"
    # Choose the first preview expression that the DSL can compile
    compiled_fn = None
    for s in sigs:
        e = s.get("expr")
        try:
            compiled_fn = compile_condition(e)
            expr = e
            break
        except Exception:
            compiled_fn = None
            continue

    assert compiled_fn is not None, "No preview signal could be compiled by the DSL"

    # Compile and evaluate against a simple 1h DataFrame
    df = _make_sample_df(n=30, freq="H")

    # Precompute common indicator columns the DSL expects (ema/sma)
    import re

    for m in re.finditer(r"ema\(\s*close\s*,\s*(\d+)\s*\)", expr):
        L = int(m.group(1))
        col = f"ema_close_{L}"
        df[col] = df["close"].ewm(span=L, adjust=False).mean()
    for m in re.finditer(r"sma\(\s*close\s*,\s*(\d+)\s*\)", expr):
        L = int(m.group(1))
        col = f"sma_close_{L}"
        df[col] = df["close"].rolling(window=L, min_periods=1).mean()

    ctx = EvalContext(df_by_tf={"1h": df}, tf="1h", target_index=df.index)
    out = compiled_fn(ctx)
    assert isinstance(out, pd.Series)
    assert len(out) == len(df.index)
