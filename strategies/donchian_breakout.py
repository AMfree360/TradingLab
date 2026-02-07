"""Donchian / Turtle-style breakout strategy.

Configurable timeframe (default: 4h). Entry: breakout above N-bar high.
Exit: N2-bar low or ATR-based stop. Volatility-scaled position sizing
is expected to be handled by higher-level risk config; this strategy
provides `stop_price` and basic `weight` metadata.
"""
from __future__ import annotations

from typing import Dict, Optional
import pandas as pd
import numpy as np

from strategies.base.strategy_base import StrategyBase


class DonchianBreakout(StrategyBase):
    """Simple Donchian/Turtle-style breakout implementation.

    Config expectations (best-effort; defaults applied if missing):
      - config.timeframes.signal_tf (str)
      - config.timeframes.entry_tf (str)
      - config.params.entry_lookback (int) default 20
      - config.params.exit_lookback (int) default 10
      - config.stop_loss.atr_multiplier (float) default 2.5
      - config.stop_loss.atr_length (int) default 20
    """

    def __init__(self, config):
        super().__init__(config)
        # Read values from config with sensible fallbacks (do NOT mutate StrategyConfig)
        tf = getattr(self.config, 'timeframes', None)
        self.signal_tf = getattr(tf, 'signal_tf', '4h') if tf is not None else '4h'
        self.entry_tf = getattr(tf, 'entry_tf', '4h') if tf is not None else '4h'

        params = getattr(self.config, 'params', None)

        def _param(name: str, default):
            if params is None:
                return default
            if isinstance(params, dict):
                return params.get(name, default)
            return getattr(params, name, default)

        self.entry_lookback = int(_param('entry_lookback', 20))
        self.exit_lookback = int(_param('exit_lookback', 10))
        self.breakout_buffer_atr = float(_param('breakout_buffer_atr', 0.0))
        self.trend_filter_enabled = bool(_param('trend_filter_enabled', False))
        self.trend_ema_length = int(_param('trend_ema_length', 200))

        sl = getattr(self.config, 'stop_loss', None)
        self.atr_multiplier = float(getattr(sl, 'atr_multiplier', 2.5))
        self.atr_length = int(getattr(sl, 'atr_length', 20))

        regime = getattr(self.config, 'regime_filters', None)
        adx_cfg = getattr(regime, 'adx', None) if regime is not None else None
        self.adx_enabled = bool(getattr(adx_cfg, 'enabled', False)) if adx_cfg is not None else False
        self.adx_period = int(getattr(adx_cfg, 'period', 14)) if adx_cfg is not None else 14

    def _compute_adx(self, df: pd.DataFrame, period: int) -> pd.Series:
        """Compute a simple ADX series (sufficient for regime filtering)."""
        high = df['high']
        low = df['low']
        close = df['close']

        up_move = high.diff()
        down_move = (-low.diff())

        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

        high_low = high - low
        high_close = (high - close.shift(1)).abs()
        low_close = (low - close.shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

        atr = tr.rolling(period, min_periods=period).mean()
        plus_di = 100.0 * pd.Series(plus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr
        minus_di = 100.0 * pd.Series(minus_dm, index=df.index).rolling(period, min_periods=period).mean() / atr

        denom = (plus_di + minus_di)
        dx = 100.0 * (plus_di - minus_di).abs() / denom.replace(0.0, np.nan)
        return dx.rolling(period, min_periods=period).mean()

    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        # ensure columns exist
        self._validate_dataframe(df)

        entry_lookback = int(getattr(self, 'entry_lookback', 20))
        exit_lookback = int(getattr(self, 'exit_lookback', 10))
        atr_len = int(getattr(self, 'atr_length', 20))
        trend_ema_len = int(getattr(self, 'trend_ema_length', 200))
        adx_period = int(getattr(self, 'adx_period', 14))

        df['donchian_high'] = df['high'].rolling(entry_lookback, min_periods=1).max()
        df['donchian_low'] = df['low'].rolling(entry_lookback, min_periods=1).min()
        df['donchian_exit_low'] = df['low'].rolling(exit_lookback, min_periods=1).min()

        # ATR
        high_low = df['high'] - df['low']
        high_close = (df['high'] - df['close'].shift(1)).abs()
        low_close = (df['low'] - df['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(atr_len, min_periods=1).mean()

        # Optional trend context
        if trend_ema_len > 1:
            df['trend_ema'] = df['close'].ewm(span=trend_ema_len, adjust=False).mean()

        # Optional regime context for ADX filter
        if getattr(self, 'adx_enabled', False):
            df['adx'] = self._compute_adx(df, adx_period)

        return df

    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # use entry timeframe
        entry_tf = getattr(self.config.timeframes, 'entry_tf', '4h')
        if entry_tf not in df_by_tf:
            raise ValueError(f"Entry timeframe {entry_tf} not provided in data")

        df = df_by_tf[entry_tf].copy()
        df = self.get_indicators(df, tf=entry_tf)

        sigs = []

        atr_mult = float(getattr(self, 'atr_multiplier', 2.5))
        breakout_buffer_atr = float(getattr(self, 'breakout_buffer_atr', 0.0))

        symbol = getattr(getattr(self.config, 'market', None), 'symbol', None) or 'UNKNOWN'

        prev_high_series = df['donchian_high'].shift(1)
        min_bars = int(getattr(self, 'entry_lookback', 20))

        for i, (idx, row) in enumerate(df.iterrows()):
            # Early block for blackout windows, etc.
            if self._should_block_signal_generation(idx, symbol):
                continue

            # Ensure stable channel history
            if i < min_bars:
                continue

            prev_high = prev_high_series.loc[idx]
            if not np.isfinite(prev_high):
                continue

            close = float(row['close'])
            atr = float(row.get('atr', np.nan))

            # Optional trend filter
            if getattr(self, 'trend_filter_enabled', False):
                trend_ema = float(row.get('trend_ema', np.nan))
                if not np.isfinite(trend_ema) or close <= trend_ema:
                    continue

            required = float(prev_high)
            if np.isfinite(atr) and breakout_buffer_atr > 0:
                required += breakout_buffer_atr * atr

            if close <= required:
                continue

            entry_price = close
            stop_price = float(row.get('donchian_exit_low', np.nan))
            if not np.isfinite(stop_price) or stop_price <= 0:
                if np.isfinite(atr):
                    stop_price = entry_price - atr * atr_mult
                else:
                    stop_price = entry_price * 0.98

            sig = {
                'timestamp': idx,
                'direction': 'long',
                'entry_price': entry_price,
                'stop_price': stop_price,
                'weight': 1.0,
                'metadata': {
                    'type': 'donchian_breakout',
                    'atr': atr,
                    'donchian_high_prev': float(prev_high),
                    'breakout_required': float(required),
                }
            }

            # Provide ADX to the regime filter if enabled
            if getattr(self, 'adx_enabled', False):
                sig['adx'] = row.get('adx', np.nan)

            # Apply filter chain
            if not self.apply_filters(pd.Series(sig), idx, symbol, df_by_tf):
                continue

            sigs.append(sig)

        sig_df = pd.DataFrame(sigs).set_index('timestamp') if sigs else self._create_signal_dataframe()
        return sig_df


__all__ = ['DonchianBreakout']
