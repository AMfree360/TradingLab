You will be given:
1) The original messy notes
2) Your previous controlled-English output
3) A list of deterministic clarification questions produced by a strict parser

Task:
- Rewrite ONLY the controlled-English lines to reduce/eliminate clarifications by making missing details explicit.
- Preserve the user’s intent.
- Use clarification defaults when provided.
- If a clarification has no default, do NOT invent values; leave it unresolved.
- Do NOT guess or fabricate symbols/markets/exchanges.
- Output ONLY the corrected lines. No explanations, no bullets, no markdown.

Important:
- Prefer explicit timeframes on rule lines when asked (e.g., "on 1h", "on 4h").
- Keep each rule on its own line.

Supported examples:
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
Default entry timeframe (if missing): {default_entry_tf}

Original notes:
{notes}

Previous controlled-English:
{previous}

Clarifications to address:
{clarifications}
