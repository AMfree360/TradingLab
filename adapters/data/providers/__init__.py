"""Market data provider adapters.

Providers normalize data into TradingLab's OHLCV schema:
- DatetimeIndex (tz-naive)
- columns: open, high, low, close, volume
"""
