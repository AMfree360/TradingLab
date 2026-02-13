from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional


TimeframeRef = str


@dataclass(frozen=True)
class Clarification:
    """A single clarification question the CLI can ask the user."""

    key: str
    question: str
    options: list[str] | None = None
    default: str | None = None


@dataclass(frozen=True)
class ParseResult:
    spec_dict: dict[str, Any]
    clarifications: list[Clarification]
    warnings: list[str]


_TF_RE = re.compile(
    r"\b(?P<tf>\d+\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days|w|wk|wks|week|weeks))\b",
    re.IGNORECASE,
)


def _norm_tf(tf: str) -> str:
    t = tf.strip().lower().replace(" ", "")
    t = (
        t.replace("minutes", "m")
        .replace("minute", "m")
        .replace("mins", "m")
        .replace("min", "m")
        .replace("hours", "h")
        .replace("hour", "h")
        .replace("hrs", "h")
        .replace("hr", "h")
        .replace("days", "d")
        .replace("day", "d")
        .replace("weeks", "w")
        .replace("week", "w")
        .replace("wks", "w")
        .replace("wk", "w")
    )
    return t


_TIME_HHMM_RE = re.compile(r"\b(?P<h>\d{1,2}):(?P<m>\d{2})\b")


def _is_time_hhmm(s: str) -> bool:
    m = _TIME_HHMM_RE.search(s)
    if not m:
        return False
    h = int(m.group("h"))
    mi = int(m.group("m"))
    return 0 <= h <= 23 and 0 <= mi <= 59


def _parse_calendar_filters(text: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    calendar: dict[str, Any] = {}

    t = text.lower()

    # Day-of-week shortcuts
    if re.search(r"\b(no\s+weekends|weekdays\s+only|mon\s*[-–]\s*fri|monday\s+to\s+friday)\b", t):
        calendar["master_filters_enabled"] = True
        calendar["day_of_week"] = {"enabled": True, "allowed_days": [0, 1, 2, 3, 4]}

    # Trading sessions
    # Supported: Asia/London/NewYork (UTC times follow schema defaults)
    session_map: dict[str, tuple[str, str, str]] = {
        "asia": ("Asia", "23:00", "08:00"),
        "london": ("London", "07:00", "16:00"),
        "newyork": ("NewYork", "13:00", "21:00"),
        "new york": ("NewYork", "13:00", "21:00"),
        "ny": ("NewYork", "13:00", "21:00"),
    }

    chosen_session: tuple[str, str, str] | None = None
    if re.search(r"\b(session)\b", t) or re.search(r"\bonly\s+trade\b", t):
        for key, val in session_map.items():
            if re.search(rf"\b{re.escape(key)}\b", t):
                chosen_session = val
                break

    # Explicit time window: "between 13:00 and 21:00 UTC" or "13:00-21:00 UTC"
    m = re.search(
        r"\b(?:between\s+)?(?P<start>\d{1,2}:\d{2})\s*(?:to|\-|–)\s*(?P<end>\d{1,2}:\d{2})\b",
        t,
    )
    if m and _is_time_hhmm(m.group("start")) and _is_time_hhmm(m.group("end")):
        chosen_session = ("Custom", m.group("start"), m.group("end"))
        if not re.search(r"\butc\b", t):
            warnings.append(
                "Parsed a trading time window but did not see 'UTC' specified; TradingLab sessions are interpreted as UTC+00."
            )

    if chosen_session is not None:
        name, start, end = chosen_session
        calendar["master_filters_enabled"] = True
        calendar["trading_sessions_enabled"] = True
        calendar["trading_sessions"] = {
            name: {"enabled": True, "start": start, "end": end},
        }

    return calendar, warnings


def _parse_condition_expr(ln: str) -> Optional[str]:
    """Parse a single boolean rule line into a DSL expression.

    This intentionally stays small and deterministic.
    """

    # Normalize some common phrasing.
    s = ln.strip()
    s_l = s.lower()

    # MA crossover (EMA/SMA vs EMA/SMA)
    m = re.search(
        r"\b(ema|sma)\s*\(?\s*(\d+)\s*\)?\s*"
        r"(?:cross(?:es)?\s*(?:over|above|under|below)?|crossover|cross\s*over|cross\s*under|cross\s*up|cross\s*down|crosses\s*up|crosses\s*down)\s*"
        r"(above|below|over|under|up|down)\s*"
        r"(ema|sma)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        fast_type = m.group(1).lower()
        fast_len = int(m.group(2))
        direction = m.group(3).lower()
        slow_type = m.group(4).lower()
        slow_len = int(m.group(5))
        fn = "crosses_above" if direction in {"above", "over", "up"} else "crosses_below"
        return f"{fn}({fast_type}(close, {fast_len}), {slow_type}(close, {slow_len}))"

    # Price/close crosses above/below a MA
    m = re.search(
        r"\b(close|price)\b\s*(?:cross(?:es)?\s*(?:over|under|above|below)?|crossover|cross\s*over|cross\s*under)\s*"
        r"(above|below|over|under)\s*(ema|sma)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        direction = m.group(2).lower()
        ma = m.group(3).lower()
        length = int(m.group(4))
        fn = "crosses_above" if direction in {"above", "over"} else "crosses_below"
        return f"{fn}(close, {ma}(close, {length}))"

    # Price/close above/below a MA (implicit comparator)
    m = re.search(
        r"\b(close|price)\b\s*(?:is\s*)?(above|below|over|under)\s*(ema|sma)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        direction = m.group(2).lower()
        ma = m.group(3).lower()
        length = int(m.group(4))
        op = ">" if direction in {"above", "over"} else "<"
        return f"close {op} {ma}(close, {length})"

    # MA comparisons: close > sma(200)
    m = re.search(
        r"\b(close|price)\b\s*(>=|<=|>|<)\s*(sma|ema)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        op = m.group(2)
        ma = m.group(3).lower()
        length = int(m.group(4))
        return f"close {op} {ma}(close, {length})"

    # MA vs MA comparisons: EMA(20) > EMA(50)
    m = re.search(
        r"\b(ema|sma)\s*\(?\s*(\d+)\s*\)?\s*(>=|<=|>|<)\s*(ema|sma)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        left = m.group(1).lower()
        left_len = int(m.group(2))
        op = m.group(3)
        right = m.group(4).lower()
        right_len = int(m.group(5))
        return f"{left}(close, {left_len}) {op} {right}(close, {right_len})"

    # MA above/below MA (implicit comparator)
    m = re.search(
        r"\b(ema|sma)\s*\(?\s*(\d+)\s*\)?\s*(?:is\s*)?(above|below|over|under)\s*(ema|sma)\s*\(?\s*(\d+)\s*\)?\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        left = m.group(1).lower()
        left_len = int(m.group(2))
        direction = m.group(3).lower()
        right = m.group(4).lower()
        right_len = int(m.group(5))
        op = ">" if direction in {"above", "over"} else "<"
        return f"{left}(close, {left_len}) {op} {right}(close, {right_len})"

    # RSI threshold (length optional):
    # - "RSI(14) < 30"
    # - "RSI < 30" (defaults to 14)
    # - "RSI is above 70" / "RSI 14 is below 10"
    m = re.search(
        r"\brsi\b\s*(?:\(\s*(?P<len1>\d+)\s*\)|(?P<len2>\d+)\b)?\s*(?:is\s*)?"
        r"(?P<op>>=|<=|>|<|above|below|over|under)\s*(?P<lvl>\d+(?:\.\d+)?)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        length_s = m.group("len1") or m.group("len2")
        length = int(length_s) if length_s else 14
        op_raw = m.group("op").lower()
        if op_raw in {"above", "over"}:
            op = ">"
        elif op_raw in {"below", "under"}:
            op = "<"
        else:
            op = op_raw
        level = float(m.group("lvl"))
        return f"rsi(close, {length}) {op} {level}"

    # Donchian breakout/breakdown: "breakout above Donchian high 20" / "breakdown below 20-bar low"
    m = re.search(
        r"\b(?:breakout|breaks?)\b\s*(?:above|over|through)\s*(?:donchian\s+)?(?:high|channel\s+high|upper\s+channel)\s*(\d+)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        length = int(m.group(1))
        # Use previous channel to stay causal
        return f"crosses_above(close, shift(donchian_high({length}), 1))"

    m = re.search(
        r"\b(?:breakout|breakdown|breaks?)\b\s*(?:below|under|through)\s*(?:donchian\s+)?(?:low|channel\s+low|lower\s+channel)\s*(\d+)\b",
        s,
        flags=re.IGNORECASE,
    )
    if m:
        length = int(m.group(1))
        return f"crosses_below(close, shift(donchian_low({length}), 1))"

    # Alternate phrasing: "above 20-bar high" / "below 20-bar low"
    m = re.search(r"\b(?:breakout|breaks?)\b\s*(?:above|over|through)\s*(\d+)\s*[- ]?bar\s+high\b", s_l)
    if m:
        length = int(m.group(1))
        return f"crosses_above(close, shift(donchian_high({length}), 1))"

    m = re.search(r"\b(?:breakout|breakdown|breaks?)\b\s*(?:below|under|through)\s*(\d+)\s*[- ]?bar\s+low\b", s_l)
    if m:
        length = int(m.group(1))
        return f"crosses_below(close, shift(donchian_low({length}), 1))"

    return None


def _find_timeframe(text: str) -> Optional[str]:
    m = _TF_RE.search(text)
    if not m:
        return None
    return _norm_tf(m.group("tf"))


def _strip_code_fences(text: str) -> str:
    # Allow users to paste notes with ``` blocks.
    return re.sub(r"```.*?```", "", text, flags=re.DOTALL)


def _clean(text: str) -> str:
    text = text.replace("\r\n", "\n")
    text = _strip_code_fences(text)
    return text.strip()


def _parse_symbol(text: str) -> Optional[str]:
    # Very lightweight heuristic.
    # Accept: "Symbol BTCUSDT", "Market: EURUSD", "Instrument=NQ".
    m = re.search(r"\b(symbol|market|instrument)\s*[:=]\s*([A-Za-z0-9_\-\.]{3,20})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(2).upper()

    m = re.search(r"\btrade\s+([A-Za-z0-9_\-\.]{3,20})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()

    return None


def _parse_exchange(text: str) -> Optional[str]:
    m = re.search(r"\bexchange\s*[:=]\s*([A-Za-z0-9_\-]{2,30})\b", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower()
    return None


def _parse_market_type(text: str) -> Optional[str]:
    if re.search(r"\bfutures\b", text, flags=re.IGNORECASE):
        return "futures"
    if re.search(r"\bspot\b", text, flags=re.IGNORECASE):
        return "spot"
    return None


def _parse_entry_rule(text: str, side: Literal["long", "short"]) -> tuple[list[dict[str, str]], list[str]]:
    """Parse entry conditions.

    Controlled English supported (case-insensitive):
    - "Go long when EMA(20) crosses above EMA(50) on 4h"
    - "Enter short when close < SMA(200) on 1h"
    - "Long when RSI(14) < 30 on 15m"

    Returns list of ConditionSpec-like dicts: {tf, expr}.
    """

    warnings: list[str] = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    conds: list[dict[str, str]] = []

    side_words = [side]
    if side == "long":
        side_words += ["buy"]
    else:
        side_words += ["sell", "short"]

    entry_verbs = ["go", "enter", "entry", "open", "take", "initiate"]

    # Pick candidate lines that look like entry rules.
    candidates = []
    for ln in lines:
        if re.search(r"\b(" + "|".join(entry_verbs) + r")\b", ln, flags=re.IGNORECASE) and any(
            re.search(rf"\b{w}\b", ln, flags=re.IGNORECASE) for w in side_words
        ):
            candidates.append(ln)
        elif re.match(rf"^{side}\s+(?:when|if)\b", ln, flags=re.IGNORECASE):
            candidates.append(ln)
        elif any(re.match(rf"^{w}\s+(?:when|if)\b", ln, flags=re.IGNORECASE) for w in side_words):
            candidates.append(ln)
        elif any(re.search(rf"\b{w}\b", ln, flags=re.IGNORECASE) for w in side_words) and re.search(
            r"\b(breakout|breakdown|breaks?)\b", ln, flags=re.IGNORECASE
        ):
            candidates.append(ln)

    # Some users write multiple actions in one sentence, e.g.
    # "Go long when ..., go short when ...". Split into per-action clauses.
    def _split_compound_action_line(ln: str) -> list[str]:
        parts = re.split(
            r"\s*(?:,|;|\band\b)\s*(?=(?:go|enter|open|take|initiate|buy|sell)\b)",
            ln,
            flags=re.IGNORECASE,
        )
        out = [p.strip() for p in parts if p and p.strip()]
        return out or [ln]

    for ln in candidates:
        sub_lines = _split_compound_action_line(ln)
        parsed_any = False
        for sub in sub_lines:
            if not any(re.search(rf"\b{w}\b", sub, flags=re.IGNORECASE) for w in side_words):
                continue
            tf = _find_timeframe(sub)
            expr = _parse_condition_expr(sub)
            if expr is not None:
                conds.append({"tf": tf or "__MISSING_TF__", "expr": expr})
                parsed_any = True
        if not parsed_any:
            warnings.append(f"Unparsed {side} entry rule: {ln}")

    return conds, warnings


def _parse_exit_rule(text: str, side: Literal["long", "short"]) -> tuple[list[dict[str, str]], list[str]]:
    """Parse exit conditions.

    Controlled English supported (case-insensitive):
    - "Exit long when EMA(20) crosses below EMA(50) on 4h"
    - "Exit when close crosses below EMA(50) on 4h" (applies to both sides later)
    - "Exit on opposite breakout" (Donchian breakout/breakdown inferred)

    Returns list of ConditionSpec-like dicts: {tf, expr}.
    """

    warnings: list[str] = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    conds: list[dict[str, str]] = []

    side_words = [side]
    if side == "long":
        side_words += ["buy"]
    else:
        side_words += ["sell", "short"]

    candidates: list[str] = []
    for ln in lines:
        # Side-specific exit lines
        if re.search(r"\b(exit|close)\b", ln, flags=re.IGNORECASE) and any(
            re.search(rf"\b{w}\b", ln, flags=re.IGNORECASE) for w in side_words
        ):
            candidates.append(ln)
            continue

        # Side-specific close shorthand (e.g. "Close long if ...")
        if re.match(r"^(exit|close)\s+(?:the\s+)?(" + "|".join(side_words) + r")\s+(?:when|if)\b", ln, flags=re.IGNORECASE):
            candidates.append(ln)
            continue

        # Generic exit lines (handled by caller; here we allow them for both sides)
        if re.match(r"^exit\s+when\b", ln, flags=re.IGNORECASE) or re.match(r"^exit\b", ln, flags=re.IGNORECASE):
            candidates.append(ln)
            continue

        # "Exit on opposite breakout" shorthand
        if re.search(r"\b(exit|close)\b", ln, flags=re.IGNORECASE) and re.search(
            r"\b(opposite|reverse)\b", ln, flags=re.IGNORECASE
        ) and re.search(r"\b(breakout|breakdown|breaks?)\b", ln, flags=re.IGNORECASE):
            candidates.append(ln)
            continue

    for ln in candidates:
        tf = _find_timeframe(ln)
        ln_l = ln.lower()

        # Special-case: "opposite breakout" -> map to opposite Donchian trigger.
        if re.search(r"\b(opposite|reverse)\b", ln_l) and re.search(r"\b(breakout|breakdown|breaks?)\b", ln_l):
            # Try to infer Donchian length from the same line.
            m = re.search(r"\b(\d+)\s*[- ]?bar\b", ln_l) or re.search(r"\b(\d+)\b", ln_l)
            length = int(m.group(1)) if m else 20
            if side == "long":
                expr = f"crosses_below(close, shift(donchian_low({length}), 1))"
            else:
                expr = f"crosses_above(close, shift(donchian_high({length}), 1))"
            conds.append({"tf": tf or "__MISSING_TF__", "expr": expr})
            continue

        expr = _parse_condition_expr(ln)
        if expr is not None:
            # If line says "exit ..." but parses to an entry-style expr, it's still fine.
            conds.append({"tf": tf or "__MISSING_TF__", "expr": expr})
            continue
        # If it didn't parse and it looked like an exit line, warn.
        if re.search(r"\b(exit|close)\b", ln, flags=re.IGNORECASE):
            warnings.append(f"Unparsed {side} exit rule: {ln}")

    return conds, warnings


def _parse_stop(text: str, side: Literal["long", "short"]) -> tuple[dict[str, Any], list[Clarification], list[str]]:
    warnings: list[str] = []
    clarifications: list[Clarification] = []

    # Percent stop: "stop 2%" or "stop = 2%" or "SL 1.5%"
    m = re.search(
        r"\b(stop|sl|stop loss)\b\s*(?:[:=]|is|=)?\s*(\d+(?:\.\d+)?)\s*%(?:\s|$)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        pct = float(m.group(2))
        # store as fraction for spec
        return {"type": "percent", "percent": pct / 100.0}, clarifications, warnings

    # ATR stop: "Stop ATR(14) * 3" or "SL = ATR 14 x 3"
    m = re.search(
        r"\b(stop|sl|stop loss)\b.*?atr\s*\(?\s*(\d+)\s*\)?\s*(?:\*|x)\s*(\d+(?:\.\d+)?)",
        text,
        flags=re.IGNORECASE,
    )
    if m:
        length = int(m.group(2))
        mult = float(m.group(3))
        return {"type": "atr", "atr_length": length, "atr_multiplier": mult}, clarifications, warnings

    # If user mentions stop but we can't parse it, ask.
    if re.search(r"\b(stop|sl|stop loss)\b", text, flags=re.IGNORECASE):
        clarifications.append(
            Clarification(
                key=f"{side}.stop",
                question=(
                    f"I saw you mention a stop loss for {side}. Which format did you mean?"
                ),
                options=[
                    "Percent (example: stop 2%)",
                    "ATR (example: stop ATR(14) * 3)",
                    "Skip (use default fallback)",
                ],
                default="ATR (example: stop ATR(14) * 3)",
            )
        )

    return {}, clarifications, warnings


def _parse_r_levels(text: str, kind: Literal["take_profit", "partial_exit"]) -> list[dict[str, Any]]:
    """Parse sequences like: "2R 50%, 3R 50%" or "1.5R 80%"."""
    levels: list[dict[str, Any]] = []
    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*r\b\s*(\d+(?:\.\d+)?)\s*%", text, flags=re.IGNORECASE):
        r_mult = float(m.group(1))
        exit_pct = float(m.group(2))
        if kind == "take_profit":
            levels.append({"target_r": r_mult, "exit_pct": exit_pct, "enabled": True})
        else:
            levels.append({"level_r": r_mult, "exit_pct": exit_pct, "enabled": True})
    return levels


def _parse_trade_management(text: str) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    tm: dict[str, Any] = {}

    # Take profit
    if re.search(r"\b(take profit|tp)\b", text, flags=re.IGNORECASE):
        levels = _parse_r_levels(text, kind="take_profit")
        if levels:
            tm["take_profit"] = {"enabled": True, "levels": levels}
        else:
            warnings.append("Mentioned take profit but no R-levels parsed (expected e.g. '2R 50%, 3R 50%').")

    # Partial exit
    if re.search(r"\b(partial|scale out|partial exit)\b", text, flags=re.IGNORECASE):
        levels = _parse_r_levels(text, kind="partial_exit")
        if levels:
            tm["partial_exit"] = {"enabled": True, "levels": levels}
        else:
            warnings.append("Mentioned partial exits but no R-levels parsed (expected e.g. '1.5R 80%').")

    # Trailing stop
    if re.search(r"\btrailing\b", text, flags=re.IGNORECASE):
        m = re.search(r"trailing\s+(ema|sma)\s*(\d+)", text, flags=re.IGNORECASE)
        if m:
            t = m.group(1).upper()
            length = int(m.group(2))
            activation_r = 0.5
            m2 = re.search(r"after\s*(\d+(?:\.\d+)?)\s*r\b", text, flags=re.IGNORECASE)
            if m2:
                activation_r = float(m2.group(1))
            tm["trailing_stop"] = {
                "enabled": True,
                "type": t,
                "length": length,
                "activation_type": "r_based",
                "activation_r": activation_r,
            }
        else:
            # Basic trailing mention but not parseable
            warnings.append("Mentioned trailing stop but could not parse type/length (expected e.g. 'trailing EMA 21').")

    return tm, warnings


def parse_english_strategy(
    text: str,
    *,
    name: str,
    default_entry_tf: Optional[TimeframeRef] = None,
    mode: Literal["strict", "lenient"] = "strict",
) -> ParseResult:
    """Parse a plain-English description into a StrategySpec-compatible dict.

    This is intentionally *not* free-form NLP. It's a controlled-English parser
    with explicit templates, designed to work offline (no ML required).

    If anything is missing or ambiguous, `clarifications` will be non-empty and
    the caller should ask the user targeted follow-up questions.
    """

    text = _clean(text)
    warnings: list[str] = []
    clarifications: list[Clarification] = []

    symbol = _parse_symbol(text)
    exchange = _parse_exchange(text)
    market_type = _parse_market_type(text)

    entry_tf = default_entry_tf or _find_timeframe(text)
    if not entry_tf:
        clarifications.append(
            Clarification(
                key="entry_tf",
                question="Which entry timeframe should I use?",
                options=["5m", "15m", "1h", "4h", "1d"],
                default="1h",
            )
        )
        entry_tf = "__MISSING_TF__"

    if not symbol:
        clarifications.append(
            Clarification(
                key="market.symbol",
                question="Which symbol/instrument should I trade? (example: BTCUSDT, EURUSD)",
            )
        )
        symbol = "UNKNOWN"

    long_conds, w_long = _parse_entry_rule(text, side="long")
    short_conds, w_short = _parse_entry_rule(text, side="short")
    warnings.extend(w_long)
    warnings.extend(w_short)

    # Lenient fallback: if strict heuristics didn't find any entry candidates, try to
    # interpret lines that mention the side and contain a parseable expression.
    # This still does not guess missing values; it only broadens line selection.
    if mode == "lenient" and not long_conds and not short_conds:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        def _lenient_side_extract(side_key: Literal["long", "short"]) -> list[dict[str, str]]:
            side_words = [side_key]
            if side_key == "long":
                side_words += ["buy"]
            else:
                side_words += ["sell", "short"]

            out: list[dict[str, str]] = []
            for ln in lines:
                if not any(re.search(rf"\b{w}\b", ln, flags=re.IGNORECASE) for w in side_words):
                    continue
                expr = _parse_condition_expr(ln)
                if expr is None:
                    continue
                tf = _find_timeframe(ln)
                out.append({"tf": tf or "__MISSING_TF__", "expr": expr})
            return out

        long_conds = _lenient_side_extract("long")
        short_conds = _lenient_side_extract("short")

    long_exit_conds, w = _parse_exit_rule(text, side="long")
    warnings.extend(w)
    short_exit_conds, w = _parse_exit_rule(text, side="short")
    warnings.extend(w)

    # Global/gating confirmations: apply to both sides when present.
    global_conds: list[dict[str, str]] = []
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    for ln in lines:
        if not re.search(r"\b(only\s+trade\s+when|confirm|filter|gate)\b", ln, flags=re.IGNORECASE):
            continue
        # Avoid double-counting explicit long/short lines.
        if re.search(r"\b(go|enter|entry)\b", ln, flags=re.IGNORECASE) and re.search(
            r"\b(long|short|buy|sell)\b", ln, flags=re.IGNORECASE
        ):
            continue

        tf = _find_timeframe(ln)
        expr = _parse_condition_expr(ln)
        if expr is None:
            # Don't warn noisily for pure English gating lines that don't map cleanly.
            continue
        global_conds.append({"tf": tf or "__MISSING_TF__", "expr": expr})

    if global_conds:
        if long_conds:
            long_conds = long_conds + global_conds
        if short_conds:
            short_conds = short_conds + global_conds

    # If the user provided a default entry TF (or we inferred one) and individual rules
    # omit a timeframe, fill it deterministically to avoid unnecessary clarifications.
    if entry_tf and entry_tf != "__MISSING_TF__":
        def _fill_tf(conds: list[dict[str, str]]) -> list[dict[str, str]]:
            out: list[dict[str, str]] = []
            for c in conds:
                if c.get("tf") == "__MISSING_TF__":
                    out.append({**c, "tf": str(entry_tf)})
                else:
                    out.append(c)
            return out

        long_conds = _fill_tf(long_conds)
        short_conds = _fill_tf(short_conds)
        long_exit_conds = _fill_tf(long_exit_conds)
        short_exit_conds = _fill_tf(short_exit_conds)

    if not long_conds and not short_conds:
        clarifications.append(
            Clarification(
                key="entries",
                question=(
                    "I couldn't find a LONG or SHORT entry rule. "
                    "Write one like: 'Go long when EMA(20) crosses above EMA(50) on 4h' or 'Buy when close < SMA(200) on 1h'."
                ),
            )
        )

    # If we extracted conditions but TF is missing, ask once.
    if any(c.get("tf") == "__MISSING_TF__" for c in (long_conds + short_conds + long_exit_conds + short_exit_conds)):
        clarifications.append(
            Clarification(
                key="condition_tf",
                question=(
                    "Your entry rule didn't specify a timeframe (like 'on 4h'). Which timeframe should the entry conditions run on?"
                ),
                options=["5m", "15m", "1h", "4h", "1d"],
                default=str(entry_tf if entry_tf != "__MISSING_TF__" else "1h"),
            )
        )

    long_stop, long_stop_cl, w = _parse_stop(text, side="long")
    clarifications.extend(long_stop_cl)
    warnings.extend(w)
    short_stop, short_stop_cl, w = _parse_stop(text, side="short")
    clarifications.extend(short_stop_cl)
    warnings.extend(w)

    tm, tm_w = _parse_trade_management(text)
    warnings.extend(tm_w)

    calendar_filters, cal_w = _parse_calendar_filters(text)
    warnings.extend(cal_w)

    # Build a StrategySpec-like dict; compiler will validate via pydantic.
    spec: dict[str, Any] = {
        "name": name,
        "description": "(from plain English)",
        "market": {
            "symbol": symbol,
            "exchange": exchange,
            "market_type": market_type,
        },
        "entry_tf": entry_tf,
        "extra_tfs": [],
        "filters": {"calendar_filters": calendar_filters or {"master_filters_enabled": False}},
        "risk_per_trade_pct": 1.0,
    }

    # Compute extra_tfs for readability.
    used_tfs: set[str] = set()
    for cond in (long_conds + short_conds + long_exit_conds + short_exit_conds):
        tf = cond.get("tf")
        if tf and tf != "__MISSING_TF__":
            used_tfs.add(str(tf))
    if entry_tf and entry_tf != "__MISSING_TF__":
        used_tfs.discard(str(entry_tf))
    spec["extra_tfs"] = sorted(used_tfs)

    if long_conds:
        spec["long"] = {
            "enabled": True,
            "conditions_all": long_conds,
            "exit_conditions_all": long_exit_conds,
            "stop_type": long_stop.get("type"),
            "stop_percent": long_stop.get("percent"),
            "atr_length": long_stop.get("atr_length"),
            "atr_multiplier": long_stop.get("atr_multiplier"),
        }
    elif long_exit_conds:
        # Allow exit-only definition to still serialize cleanly.
        spec["long"] = {
            "enabled": True,
            "conditions_all": [],
            "exit_conditions_all": long_exit_conds,
            "stop_type": long_stop.get("type"),
            "stop_percent": long_stop.get("percent"),
            "atr_length": long_stop.get("atr_length"),
            "atr_multiplier": long_stop.get("atr_multiplier"),
        }

    if short_conds:
        spec["short"] = {
            "enabled": True,
            "conditions_all": short_conds,
            "exit_conditions_all": short_exit_conds,
            "stop_type": short_stop.get("type"),
            "stop_percent": short_stop.get("percent"),
            "atr_length": short_stop.get("atr_length"),
            "atr_multiplier": short_stop.get("atr_multiplier"),
        }
    elif short_exit_conds:
        spec["short"] = {
            "enabled": True,
            "conditions_all": [],
            "exit_conditions_all": short_exit_conds,
            "stop_type": short_stop.get("type"),
            "stop_percent": short_stop.get("percent"),
            "atr_length": short_stop.get("atr_length"),
            "atr_multiplier": short_stop.get("atr_multiplier"),
        }

    if tm:
        spec["trade_management"] = tm

    return ParseResult(spec_dict=spec, clarifications=clarifications, warnings=warnings)


def apply_clarifications(spec_dict: dict[str, Any], answers: dict[str, str]) -> dict[str, Any]:
    """Apply CLI answers to a partially-parsed spec dict."""
    out = {**spec_dict}

    # entry_tf
    if "entry_tf" in answers:
        out["entry_tf"] = _norm_tf(answers["entry_tf"])

    # symbol
    if "market.symbol" in answers:
        market = dict(out.get("market") or {})
        market["symbol"] = answers["market.symbol"].upper().strip()
        out["market"] = market

    # condition timeframe fixups
    if "condition_tf" in answers:
        tf = _norm_tf(answers["condition_tf"])
        for side_key in ("long", "short"):
            side = out.get(side_key)
            if not side:
                continue
            conds = []
            for c in side.get("conditions_all") or []:
                if c.get("tf") == "__MISSING_TF__":
                    c = {**c, "tf": tf}
                conds.append(c)
            side["conditions_all"] = conds

            exit_conds = []
            for c in side.get("exit_conditions_all") or []:
                if c.get("tf") == "__MISSING_TF__":
                    c = {**c, "tf": tf}
                exit_conds.append(c)
            side["exit_conditions_all"] = exit_conds
            out[side_key] = side

    return out
