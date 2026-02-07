Donchian / Turtle-style breakout strategy
----------------------------------------

Simple breakout strategy used as a canonical passing example.

Defaults:
- Timeframe: `4h` (configurable)
- Entry: Breakout above prior `N`-bar high (default `20`)
- Exit: `M`-bar low (default `10`) or ATR-based stop (default `2.5x ATR`)

Use this strategy to demonstrate a robust trend-following approach that
typically performs well on higher timeframes with volatility-scaled sizing.
