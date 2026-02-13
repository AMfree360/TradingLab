from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Callable

import pandas as pd


ALLOWED_BASE_SERIES = {"open", "high", "low", "close", "volume"}
DERIVED_SERIES = {"hl2", "hlc3", "ohlc4"}
ALLOWED_SERIES = ALLOWED_BASE_SERIES | DERIVED_SERIES


@dataclass(frozen=True)
class IndicatorRequest:
    name: str
    tf: str
    args: tuple


class DSLCompileError(ValueError):
    pass


@dataclass(frozen=True)
class EvalContext:
    df_by_tf: dict[str, pd.DataFrame]
    tf: str
    target_index: pd.DatetimeIndex | None = None


_SERIES_AT_TF_RE = re.compile(
    r"\b(?P<series>open|high|low|close|volume|hl2|hlc3|ohlc4)\s*@\s*(?P<tf>[0-9]+[mhdw])\b",
    flags=re.IGNORECASE,
)


def _rewrite_atom_at_tf(expr: str) -> str:
    """Rewrite `<atom>@<tf>` into `at("<tf>", <atom>)`.

    Supports atoms that are:
      - identifiers (e.g. close@1h)
      - function calls (e.g. ema(close, 20)@4h)
      - parenthesized expressions (e.g. (ema(close,20) > ema(close,50))@4h)

    Notes:
      - This runs before AST parsing because `@` is the matrix-multiply operator in Python.
      - Only rewrites when the right-hand side looks like a timeframe token: digits + [mhdw].
    """

    def _is_ident_char(ch: str) -> bool:
        return ch.isalnum() or ch == "_"

    i = 0
    out: list[str] = []
    n = len(expr)

    while i < n:
        at_pos = expr.find("@", i)
        if at_pos == -1:
            out.append(expr[i:])
            break

        # Attempt parse timeframe after '@'
        j = at_pos + 1
        while j < n and expr[j].isspace():
            j += 1

        k = j
        while k < n and expr[k].isdigit():
            k += 1
        if k == j or k >= n:
            out.append(expr[i : at_pos + 1])
            i = at_pos + 1
            continue

        unit = expr[k]
        if unit.lower() not in {"m", "h", "d", "w"}:
            out.append(expr[i : at_pos + 1])
            i = at_pos + 1
            continue

        tf = expr[j : k + 1]
        tf_end = k + 1

        # Find the start of the atom on the left.
        left_end = at_pos - 1
        while left_end >= 0 and expr[left_end].isspace():
            left_end -= 1
        if left_end < 0:
            out.append(expr[i : at_pos + 1])
            i = at_pos + 1
            continue

        atom_start = None
        ch = expr[left_end]

        if ch == ")":
            depth = 0
            p = left_end
            while p >= 0:
                if expr[p] == ")":
                    depth += 1
                elif expr[p] == "(":
                    depth -= 1
                    if depth == 0:
                        # Include optional function name immediately before '('
                        q = p - 1
                        while q >= 0 and expr[q].isspace():
                            q -= 1
                        r = q
                        while r >= 0 and _is_ident_char(expr[r]):
                            r -= 1
                        # Only include name if it looks like an identifier
                        if r != q and (r < 0 or not _is_ident_char(expr[r])):
                            atom_start = r + 1
                        else:
                            atom_start = p
                        break
                p -= 1

        elif _is_ident_char(ch):
            p = left_end
            while p >= 0 and _is_ident_char(expr[p]):
                p -= 1
            atom_start = p + 1

        elif ch == "]":
            # Not currently supported; fall through.
            atom_start = None

        if atom_start is None:
            out.append(expr[i : at_pos + 1])
            i = at_pos + 1
            continue

        # Emit everything before the atom, then rewritten at() call.
        out.append(expr[i:atom_start])
        atom = expr[atom_start : left_end + 1]
        out.append(f'at("{tf}", {atom})')
        i = tf_end

    return "".join(out)


def preprocess_expression(expr: str) -> str:
    """Normalize shorthand syntax into canonical DSL.

    Currently supported rewrites:
      - `close@1h` -> `at("1h", close)`
      - `hl2@4h` -> `at("4h", hl2)`
    """

    # 1) Handle general atom@tf patterns (covers identifiers, calls, parenthesized expressions)
    expr = _rewrite_atom_at_tf(expr)
    # 2) Also handle series@tf when it appears in contexts missed by the atom rewriter.
    def _repl(m: re.Match) -> str:
        series = m.group("series")
        tf = m.group("tf")
        return f'at("{tf}", {series})'

    return _SERIES_AT_TF_RE.sub(_repl, expr)


def _is_series_expr(node: ast.AST) -> bool:
    if isinstance(node, ast.Name) and node.id in ALLOWED_SERIES:
        return True
    if isinstance(node, ast.Call):
        fn = _name_of_call(node.func)
        if fn in DERIVED_SERIES and len(node.args) == 0:
            return True
    return False


def normalize_expression_ast(tree: ast.Expression) -> ast.Expression:
    """Normalize expression AST into canonical multi-timeframe form.

    Currently:
      - `ema(at("4h", close), 20)` -> `at("4h", ema(close, 20))`
      - same for sma()/rsi()

    This enables shorthand user syntax like `ema(close@4h, 20)`.
    """

    class T(ast.NodeTransformer):
        def visit_Call(self, node: ast.Call):
            node = self.generic_visit(node)
            fn = _name_of_call(node.func)

            if fn in {"ema", "sma", "rsi"} and len(node.args) >= 2:
                first = node.args[0]
                if isinstance(first, ast.Call) and _name_of_call(first.func) == "at":
                    if len(first.args) != 2:
                        raise DSLCompileError('at() expects 2 args: at("1h", close)')
                    tf_node = first.args[0]
                    if not (isinstance(tf_node, ast.Constant) and isinstance(tf_node.value, str)):
                        raise DSLCompileError(
                            'at() first arg must be a literal timeframe string, e.g. "1h"'
                        )
                    series_expr = first.args[1]
                    if not _is_series_expr(series_expr):
                        raise DSLCompileError(
                            f"{fn}() only supports at(tf, <series>) as first arg; "
                            "use at(tf, ema(series, n)) for more complex expressions"
                        )

                    inner = ast.Call(
                        func=node.func,
                        args=[series_expr, node.args[1]],
                        keywords=node.keywords,
                    )
                    rewritten = ast.Call(
                        func=ast.Name(id="at", ctx=ast.Load()),
                        args=[tf_node, inner],
                        keywords=[],
                    )
                    return ast.copy_location(rewritten, node)

            return node

    new_tree = T().visit(tree)
    ast.fix_missing_locations(new_tree)
    return new_tree


def extract_timeframe_refs(expr: str, default_tf: str) -> set[str]:
    """Return all timeframe strings referenced by an expression.

    Includes:
      - default_tf
      - any `at("1h", ...)` calls
      - any `close@1h` shorthand
    """

    expr = preprocess_expression(expr)
    out = {default_tf}
    tree = normalize_expression_ast(_parse(expr))

    class V(ast.NodeVisitor):
        def visit_Call(self, node: ast.Call):
            fn = _name_of_call(node.func)
            if fn == "at":
                if len(node.args) != 2:
                    raise DSLCompileError('at() expects 2 args: at("1h", close)')
                tf_node = node.args[0]
                if not (isinstance(tf_node, ast.Constant) and isinstance(tf_node.value, str)):
                    raise DSLCompileError('at() first arg must be a literal timeframe string, e.g. "1h"')
                out.add(tf_node.value)
            self.generic_visit(node)

    V().visit(tree)
    return out


def extract_indicator_requests(expr: str, tf: str) -> list[IndicatorRequest]:
    """Extract required indicator computations from an expression.

        Supported:
      - ema(close, 20)
      - sma(close, 50)
      - rsi(close, 14)
      - atr(14)
      - donchian_high(20)
      - donchian_low(20)
            - ema(hl2, 20)
            - ema(hl2(), 20)

    Notes:
            - First arg of ema/sma/rsi must be a supported series: open/high/low/close/volume/hl2/hlc3/ohlc4.
      - All numeric args must be literal ints/floats.
    """

    expr = preprocess_expression(expr)
    tree = normalize_expression_ast(_parse(expr))
    reqs: list[IndicatorRequest] = []

    def _visit(node: ast.AST, active_tf: str) -> None:
        if isinstance(node, ast.Call):
            fn = _name_of_call(node.func)
            if fn == "at":
                if len(node.args) != 2:
                    raise DSLCompileError('at() expects 2 args: at("1h", close)')
                tf_node = node.args[0]
                if not (isinstance(tf_node, ast.Constant) and isinstance(tf_node.value, str)):
                    raise DSLCompileError('at() first arg must be a literal timeframe string, e.g. "1h"')
                inner_tf = tf_node.value
                _visit(node.args[1], inner_tf)
                return

            if fn in {"ema", "sma", "rsi"}:
                if len(node.args) != 2:
                    raise DSLCompileError(f"{fn}() expects 2 args: {fn}(close, 20)")
                series_name = _literal_series_key(node.args[0])
                length = _literal_number(node.args[1])
                reqs.append(IndicatorRequest(fn, active_tf, (series_name, int(length))))
            elif fn == "atr":
                if len(node.args) != 1:
                    raise DSLCompileError("atr() expects 1 arg: atr(14)")
                length = _literal_number(node.args[0])
                reqs.append(IndicatorRequest("atr", active_tf, (int(length),)))
            elif fn in {"donchian_high", "donchian_low"}:
                if len(node.args) != 1:
                    raise DSLCompileError(f"{fn}() expects 1 arg: {fn}(20)")
                length = _literal_number(node.args[0])
                reqs.append(IndicatorRequest(fn, active_tf, (int(length),)))

        for child in ast.iter_child_nodes(node):
            _visit(child, active_tf)

    _visit(tree, tf)

    # Deduplicate while preserving order
    seen = set()
    out: list[IndicatorRequest] = []
    for r in reqs:
        key = (r.name, r.tf, r.args)
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def compile_condition(expr: str) -> Callable[[EvalContext], pd.Series]:
    """Compile an expression into a function that returns a boolean Series.

    This is a safe subset of Python expressions over pandas Series.

    Allowed:
      - comparisons: >, >=, <, <=, ==, !=
      - boolean ops: and/or/not (rewritten to &, |, ~)
      - arithmetic: +, -, *, /
      - functions: ema/sma/rsi/atr/donchian_high/donchian_low, crosses_above/crosses_below
      - base series names: open/high/low/close/volume

        The resulting callable expects an EvalContext with:
            - df_by_tf: mapping timeframe -> DataFrame
            - tf: the "current" timeframe for expression evaluation
            - target_index (optional): if provided, the result is aligned using ffill
                to this index (typically the entry timeframe index).
    """

    expr = preprocess_expression(expr)
    tree = normalize_expression_ast(_parse(expr))
    _validate_ast(tree)

    def _eval(ctx: EvalContext) -> pd.Series:
        if ctx.tf not in ctx.df_by_tf:
            raise DSLCompileError(f"Missing timeframe in df_by_tf: {ctx.tf}")
        df = ctx.df_by_tf[ctx.tf]
        env = _build_env(df)
        out = _eval_node(tree.body, env, ctx)
        if ctx.target_index is None:
            return out
        if isinstance(out, pd.Series):
            return out.reindex(ctx.target_index, method="ffill")
        return pd.Series([out] * len(ctx.target_index), index=ctx.target_index)

    return _eval


# -----------------
# internals
# -----------------

def _parse(expr: str) -> ast.Expression:
    try:
        return ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise DSLCompileError(f"Invalid expression syntax: {e}")


def _validate_ast(tree: ast.AST) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BoolOp,
        ast.BinOp,
        ast.UnaryOp,
        ast.Compare,
        ast.Call,
        ast.Name,
        ast.Load,
        ast.Constant,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Mod,
        ast.USub,
        ast.UAdd,
        ast.Gt,
        ast.GtE,
        ast.Lt,
        ast.LtE,
        ast.Eq,
        ast.NotEq,
    )

    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise DSLCompileError(f"Unsupported syntax in expression: {type(node).__name__}")

        if isinstance(node, ast.Call):
            fn = _name_of_call(node.func)
            if fn is None:
                raise DSLCompileError("Only simple function calls are allowed")
            allowed_fns = {
                "ema",
                "sma",
                "rsi",
                "atr",
                "donchian_high",
                "donchian_low",
                "crosses_above",
                "crosses_below",
                "at",
                "hl2",
                "hlc3",
                "ohlc4",
                "highest",
                "lowest",
                "shift",
                "nz",
                # Candle pattern primitives (no indicator precompute required)
                "pin_bar",
                "inside_bar",
                "engulfing",
            }
            if fn not in allowed_fns:
                raise DSLCompileError(f"Unsupported function: {fn}")

            if fn == "at":
                if len(node.args) != 2:
                    raise DSLCompileError('at() expects 2 args: at("1h", close)')
                tf_node = node.args[0]
                if not (isinstance(tf_node, ast.Constant) and isinstance(tf_node.value, str)):
                    raise DSLCompileError('at() first arg must be a literal timeframe string, e.g. "1h"')


def _name_of_call(fn: ast.AST) -> str | None:
    if isinstance(fn, ast.Name):
        return fn.id
    return None


def _literal_series_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name) and node.id in ALLOWED_BASE_SERIES:
        return node.id
    raise DSLCompileError("First arg must be a base series name: open/high/low/close/volume")


def _literal_series_key(node: ast.AST) -> str:
    if isinstance(node, ast.Name) and node.id in ALLOWED_SERIES:
        return node.id

    if isinstance(node, ast.Call):
        fn = _name_of_call(node.func)
        if fn in DERIVED_SERIES and len(node.args) == 0:
            return fn

    raise DSLCompileError(
        "First arg must be a supported series name (open/high/low/close/volume/hl2/hlc3/ohlc4) "
        "or a derived series call like hl2()"
    )


def _literal_number(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    raise DSLCompileError("Numeric arguments must be literal numbers")


def _build_env(df: pd.DataFrame) -> dict[str, object]:
    env: dict[str, object] = {}

    def _series(name: str) -> pd.Series:
        if name in df.columns:
            return df[name]
        if name == "hl2":
            s = (df["high"] + df["low"]) / 2
            s.name = "hl2"
            return s
        if name == "hlc3":
            s = (df["high"] + df["low"] + df["close"]) / 3
            s.name = "hlc3"
            return s
        if name == "ohlc4":
            s = (df["open"] + df["high"] + df["low"] + df["close"]) / 4
            s.name = "ohlc4"
            return s
        raise DSLCompileError(f"Unknown series: {name}")

    for name in ALLOWED_SERIES:
        try:
            env[name] = _series(name)
        except Exception:
            # Some series may be unavailable if df is missing required columns
            pass

    # indicator column accessors
    def ema(series_name: pd.Series, length: int) -> pd.Series:
        col = f"ema_{_series_label(series_name)}_{int(length)}"
        return _require_col(df, col)

    def sma(series_name: pd.Series, length: int) -> pd.Series:
        col = f"sma_{_series_label(series_name)}_{int(length)}"
        return _require_col(df, col)

    def rsi(series_name: pd.Series, length: int) -> pd.Series:
        col = f"rsi_{_series_label(series_name)}_{int(length)}"
        return _require_col(df, col)

    def atr(length: int) -> pd.Series:
        col = f"atr_{int(length)}"
        return _require_col(df, col)

    def donchian_high(length: int) -> pd.Series:
        col = f"donchian_high_{int(length)}"
        return _require_col(df, col)

    def donchian_low(length: int) -> pd.Series:
        col = f"donchian_low_{int(length)}"
        return _require_col(df, col)

    def crosses_above(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a > b) & (a.shift(1) <= b.shift(1))

    def crosses_below(a: pd.Series, b: pd.Series) -> pd.Series:
        return (a < b) & (a.shift(1) >= b.shift(1))

    def highest(a: pd.Series, length: int) -> pd.Series:
        return a.rolling(window=int(length), min_periods=1).max()

    def lowest(a: pd.Series, length: int) -> pd.Series:
        return a.rolling(window=int(length), min_periods=1).min()

    def shift(a: pd.Series, periods: int) -> pd.Series:
        return a.shift(int(periods))

    def nz(a: pd.Series, value: float = 0.0) -> pd.Series:
        return a.fillna(value)

    def inside_bar(high: pd.Series, low: pd.Series) -> pd.Series:
        """True when the current bar is inside the previous bar."""
        return (high < high.shift(1)) & (low > low.shift(1))

    def engulfing(open_: pd.Series, close_: pd.Series, direction: str) -> pd.Series:
        """Candle body engulfing pattern.

        direction: "bull" or "bear".
        - bull: current candle bullish and its body engulfs previous body; previous candle bearish.
        - bear: current candle bearish and its body engulfs previous body; previous candle bullish.
        """

        d = str(direction).strip().lower()
        prev_open = open_.shift(1)
        prev_close = close_.shift(1)

        prev_bear = prev_close < prev_open
        prev_bull = prev_close > prev_open
        cur_bull = close_ > open_
        cur_bear = close_ < open_

        # Body bounds
        prev_lo = prev_open.where(prev_open < prev_close, prev_close)
        prev_hi = prev_open.where(prev_open > prev_close, prev_close)
        cur_lo = open_.where(open_ < close_, close_)
        cur_hi = open_.where(open_ > close_, close_)

        if d in {"bull", "bullish", "long"}:
            return prev_bear & cur_bull & (cur_lo <= prev_lo) & (cur_hi >= prev_hi)
        if d in {"bear", "bearish", "short"}:
            return prev_bull & cur_bear & (cur_lo <= prev_lo) & (cur_hi >= prev_hi)

        raise DSLCompileError('engulfing() direction must be "bull" or "bear"')

    def pin_bar(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close_: pd.Series,
        direction: str,
        wick_body_ratio: float,
        opp_wick_body_max: float,
        min_body_pct_of_range: float,
    ) -> pd.Series:
        """Pin bar pattern.

        Parameters (all positional; keyword args are not supported by the DSL):
        - direction: "bull" or "bear".
        - wick_body_ratio: long wick / body must be >= this.
        - opp_wick_body_max: opposite wick / body must be <= this.
        - min_body_pct_of_range: body must be >= this fraction of candle range.
        """

        d = str(direction).strip().lower()
        wbr = float(wick_body_ratio)
        owmax = float(opp_wick_body_max)
        min_body_pct = float(min_body_pct_of_range)

        # Avoid pd.NA here: it can force object/nullable dtypes and trigger
        # FutureWarnings on fillna() downcasting.
        rng = (high - low).where((high - low) != 0, float("nan"))
        body = (close_ - open_).abs().where((close_ - open_).abs() != 0, float("nan"))

        upper_wick = high - open_.where(open_ > close_, close_)
        lower_wick = open_.where(open_ < close_, close_) - low

        # basic doji avoidance (optional)
        body_ok = (body / rng).fillna(0.0) >= min_body_pct

        if d in {"bull", "bullish", "long"}:
            long_wick_ok = (lower_wick / body).fillna(0.0) >= wbr
            opp_wick_ok = (upper_wick / body).fillna(0.0) <= owmax
            return body_ok & long_wick_ok & opp_wick_ok
        if d in {"bear", "bearish", "short"}:
            long_wick_ok = (upper_wick / body).fillna(0.0) >= wbr
            opp_wick_ok = (lower_wick / body).fillna(0.0) <= owmax
            return body_ok & long_wick_ok & opp_wick_ok

        raise DSLCompileError('pin_bar() direction must be "bull" or "bear"')

    env.update(
        {
            "ema": ema,
            "sma": sma,
            "rsi": rsi,
            "atr": atr,
            "donchian_high": donchian_high,
            "donchian_low": donchian_low,
            "crosses_above": crosses_above,
            "crosses_below": crosses_below,
            "highest": highest,
            "lowest": lowest,
            "shift": shift,
            "nz": nz,
            "inside_bar": inside_bar,
            "engulfing": engulfing,
            "pin_bar": pin_bar,
        }
    )

    return env


def _series_label(series: pd.Series) -> str:
    # We only allow known series names (base + derived). Use series.name.
    name = getattr(series, "name", None)
    if not name or str(name) not in ALLOWED_SERIES:
        raise DSLCompileError(
            "Indicator inputs must be a supported series (open/high/low/close/volume/hl2/hlc3/ohlc4)"
        )
    return str(name)


def _require_col(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        raise DSLCompileError(
            f"Required indicator column missing: {col}. "
            "This usually means the compiler didn't generate indicators, or the expression uses unsupported inputs."
        )
    return df[col]


def _eval_node(node: ast.AST, env: dict[str, object], ctx: EvalContext):
    # Evaluate using Python operators over pandas Series.
    # BoolOp: rewrite to bitwise &, |
    if isinstance(node, ast.BoolOp):
        values = [_eval_node(v, env, ctx) for v in node.values]
        if isinstance(node.op, ast.And):
            out = values[0]
            for v in values[1:]:
                out = out & v
            return out
        if isinstance(node.op, ast.Or):
            out = values[0]
            for v in values[1:]:
                out = out | v
            return out
        raise DSLCompileError("Unsupported boolean operator")

    if isinstance(node, ast.UnaryOp):
        v = _eval_node(node.operand, env, ctx)
        if isinstance(node.op, ast.Not):
            return ~v
        if isinstance(node.op, ast.USub):
            return -v
        if isinstance(node.op, ast.UAdd):
            return +v
        raise DSLCompileError("Unsupported unary operator")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, env, ctx)
        right = _eval_node(node.right, env, ctx)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Mod):
            return left % right
        raise DSLCompileError("Unsupported binary operator")

    if isinstance(node, ast.Compare):
        if len(node.ops) != 1 or len(node.comparators) != 1:
            raise DSLCompileError("Chained comparisons are not supported")
        left = _eval_node(node.left, env, ctx)
        right = _eval_node(node.comparators[0], env, ctx)
        op = node.ops[0]
        if isinstance(op, ast.Gt):
            return left > right
        if isinstance(op, ast.GtE):
            return left >= right
        if isinstance(op, ast.Lt):
            return left < right
        if isinstance(op, ast.LtE):
            return left <= right
        if isinstance(op, ast.Eq):
            return left == right
        if isinstance(op, ast.NotEq):
            return left != right
        raise DSLCompileError("Unsupported comparison operator")

    if isinstance(node, ast.Call):
        fn_name = _name_of_call(node.func)
        if fn_name is None:
            raise DSLCompileError("Unknown function")

        # Derived series support: allow `hl2()`, `hlc3()`, `ohlc4()` without
        # colliding with series variables of the same name.
        if fn_name in DERIVED_SERIES:
            if node.args:
                raise DSLCompileError(f"{fn_name}() takes no arguments")
            if fn_name not in env:
                raise DSLCompileError(f"Unknown series: {fn_name}")
            return env[fn_name]

        if fn_name == "at":
            if len(node.args) != 2:
                raise DSLCompileError('at() expects 2 args: at("1h", close)')
            tf_node = node.args[0]
            if not (isinstance(tf_node, ast.Constant) and isinstance(tf_node.value, str)):
                raise DSLCompileError('at() first arg must be a literal timeframe string, e.g. "1h"')
            other_tf = tf_node.value
            if other_tf not in ctx.df_by_tf:
                raise DSLCompileError(f"Missing timeframe in df_by_tf: {other_tf}")

            other_df = ctx.df_by_tf[other_tf]
            other_env = _build_env(other_df)
            other_ctx = EvalContext(df_by_tf=ctx.df_by_tf, tf=other_tf, target_index=None)
            inner = _eval_node(node.args[1], other_env, other_ctx)
            current_index = ctx.df_by_tf[ctx.tf].index
            if isinstance(inner, pd.Series):
                return inner.reindex(current_index, method="ffill")
            return pd.Series([inner] * len(current_index), index=current_index)

        if fn_name not in env:
            raise DSLCompileError("Unknown function")
        fn = env[fn_name]
        args = [_eval_node(a, env, ctx) for a in node.args]
        return fn(*args)

    if isinstance(node, ast.Name):
        if node.id not in env:
            raise DSLCompileError(f"Unknown name: {node.id}")
        return env[node.id]

    if isinstance(node, ast.Constant):
        return node.value

    raise DSLCompileError(f"Unsupported node: {type(node).__name__}")
