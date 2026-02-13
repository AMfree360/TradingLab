Rewrite the following messy trading notes into **controlled-English strategy lines** that a deterministic parser can understand.

Requirements:
- Output ONLY the lines. No explanations, no bullets, no markdown.
- Prefer explicit timeframes on each rule when possible.
- If the notes omit a timeframe, you may use the default entry timeframe: {default_entry_tf}
- Use simple, consistent phrasing.

Supported examples (use these patterns):
- Go long when EMA(20) crosses above EMA(50) on 4h
- Buy if EMA(20) crosses above EMA(50) on 4h
- Enter short when close < SMA(200) on 1h
- Sell when close is below SMA(200) on 1h
- Go long when close is above EMA(50) on 4h
- Exit long when EMA(20) crosses down EMA(50) on 4h
- Only trade when session is London
- Only trade when day_of_week in Mon,Tue,Wed,Thu,Fri
- Stop ATR(14) * 3
- Stop 1.0%
- Take profit 2R 50%, 3R 50%
- Partial exit 1.5R 80%
- Trailing EMA 21 after 0.5R
- Exit when opposite breakout happens on 1h

Strategy name: {strategy_name}

Raw notes:
{notes}
