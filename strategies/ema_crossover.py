"""EMA Crossover Strategy (5 EMA vs 100 EMA)
Default example strategy for Trading Lab

This implementation follows the project's `StrategyBase` patterns:
- Accepts a validated `StrategyConfig` or a dict-like config
- Uses the configured `signal_tf` timeframe
- Applies master/strategy filters via the base `apply_filters` method
"""

from typing import Dict, Optional, Any
import pandas as pd
from strategies.base.strategy_base import StrategyBase
from config.schema import StrategyConfig


class EMACrossoverStrategy(StrategyBase):
    def __init__(self, config: StrategyConfig | Dict[str, Any]):
        super().__init__(config)

        # Helper to read values from either a pydantic model or a dict
        def _get(path, default=None):
            # path: tuple of keys/attrs to traverse
            cur = self.config
            try:
                for p in path:
                    if isinstance(cur, dict):
                        cur = cur.get(p)
                    else:
                        cur = getattr(cur, p)
                return cur if cur is not None else default
            except Exception:
                return default

        # EMA lengths (look into moving_averages if present)
        self.ema_fast = _get(('moving_averages', 'ema5', 'length'), _get(('moving_averages', 'ema5',), 5) or 5)
        self.ema_slow = _get(('moving_averages', 'ema100', 'length'), _get(('moving_averages', 'ema100',), 100) or 100)

        # Trade management defaults (use conservative defaults if not provided)
        self.stoploss_pct = _get(('stop_loss', 'percent'), 0.01)
        # Try trailing_stop percentage field
        self.trailing_pct = _get(('trailing_stop', 'percentage'), 0.01)
        # Partial/take-profit config handled by engine's trade management; read simple defaults
        self.partial_levels = _get(('partial_exit', 'levels'), [])
        # Entry filters: confirmation and minimum EMA separation (fraction of price)
        self.entry_confirmation = _get(('entry', 'confirmation_required'), False)
        # Minimum separation between fast and slow EMA as fraction of price (e.g., 0.002 = 0.2%)
        self.ema_separation_min_pct = float(_get(('entry', 'ema_separation_pct'), 0.0) or 0.0)
        # Trend confirmation: require slow EMA slope over N bars
        self.trend_slope_bars = int(_get(('entry', 'trend_slope_bars'), 3) or 3)
        # ATR stop parameters (defaults)
        self.atr_length = int(_get(('stop_loss', 'atr_length'), 14) or 14)
        self.atr_multiplier = float(_get(('stop_loss', 'atr_multiplier'), 3.0) or 3.0)

    def get_indicators(self, df: pd.DataFrame, tf: Optional[str] = None) -> pd.DataFrame:
        df = df.copy()
        df[f'ema_{self.ema_fast}'] = df['close'].ewm(span=int(self.ema_fast), adjust=False).mean()
        df[f'ema_{self.ema_slow}'] = df['close'].ewm(span=int(self.ema_slow), adjust=False).mean()
        # ATR calculation (True Range then rolling mean)
        try:
            tr1 = df['high'] - df['low']
            tr2 = (df['high'] - df['close'].shift(1)).abs()
            tr3 = (df['low'] - df['close'].shift(1)).abs()
            tr = tr1.combine(tr2, max).combine(tr3, max)
            atr_len = int(getattr(self, 'atr_length', 14))
            df['atr'] = tr.rolling(window=atr_len, min_periods=1).mean()
        except Exception:
            df['atr'] = 0.0
        return df

    def generate_signals(self, df_by_tf: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        # Determine signal timeframe
        signal_tf = None
        try:
            signal_tf = self.config.timeframes.signal_tf  # type: ignore[attr-defined]
        except Exception:
            # Fallback to 'main' or first available
            signal_tf = 'main' if 'main' in df_by_tf else next(iter(df_by_tf.keys()))

        if signal_tf not in df_by_tf:
            return self._create_signal_dataframe()

        df_signal = df_by_tf[signal_tf].copy()
        df_signal = self.get_indicators(df_signal, tf=signal_tf)

        # Build raw cross signals on boolean crossing
        df_signal['ma_fast_above'] = df_signal[f'ema_{self.ema_fast}'] > df_signal[f'ema_{self.ema_slow}']
        df_signal['cross'] = df_signal['ma_fast_above'].astype(int).diff().fillna(0)

        # Causal confirmation (anti-lookahead):
        # - If confirmation_required=False: signal on the cross bar.
        # - If confirmation_required=True: wait one additional CLOSED bar and only signal
        #   if the post-cross state still holds.
        if self.entry_confirmation:
            df_signal['confirmed_long'] = (df_signal['cross'].shift(1) == 1) & (df_signal['ma_fast_above'] == True)
            df_signal['confirmed_short'] = (df_signal['cross'].shift(1) == -1) & (df_signal['ma_fast_above'] == False)
        else:
            df_signal['confirmed_long'] = df_signal['cross'] == 1
            df_signal['confirmed_short'] = df_signal['cross'] == -1

        if self.ema_separation_min_pct and self.ema_separation_min_pct > 0:
            # separation relative to price (use slow EMA as base)
            df_signal['ema_sep_pct'] = (df_signal[f'ema_{self.ema_fast}'] - df_signal[f'ema_{self.ema_slow}']) / df_signal['close']
        else:
            df_signal['ema_sep_pct'] = 0.0
        # Slow EMA slope
        df_signal['ema_slow_slope'] = df_signal[f'ema_{self.ema_slow}'].diff(self.trend_slope_bars)

        signals = []
        symbol = None
        try:
            symbol = self.config.market.symbol  # type: ignore[attr-defined]
        except Exception:
            # Will be provided by engine when applying filters
            symbol = None

        for idx, row in df_signal.iterrows():
            timestamp = idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx)

            # Check time blackout filters before generating signal
            if symbol and self._should_block_signal_generation(timestamp, symbol):
                continue

            if bool(row.get('confirmed_long', False)):
                if self.ema_separation_min_pct and row.get('ema_sep_pct', 0.0) < self.ema_separation_min_pct:
                    continue
                # Trend slope: require slow EMA rising
                if row.get('ema_slow_slope', 0.0) <= 0:
                    continue

                # Engine owns execution/fill timing; keep entry fields informational
                entry_price = row['close']
                entry_timestamp = timestamp

                # Determine stop price: ATR-based if configured, else percent
                stop_price = None
                try:
                    sl_type = None
                    try:
                        sl_type = self.config.stop_loss.type  # type: ignore[attr-defined]
                    except Exception:
                        sl_type = None

                    # Allow ATR stop if explicitly requested via stop_loss.type == 'ATR'
                    # or if a developer/test config enables ATR via stop_loss.atr_enabled
                    atr_enabled_cfg = False
                    try:
                        atr_enabled_cfg = bool(self.config.stop_loss.atr_enabled)  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            atr_enabled_cfg = bool(getattr(self.config.stop_loss, 'atr_enabled', False))
                        except Exception:
                            atr_enabled_cfg = False

                    use_atr = (str(sl_type).upper() == 'ATR') or atr_enabled_cfg

                    if use_atr and 'atr' in df_signal.columns:
                        atr_val = None
                        try:
                            atr_val = df_signal.at[entry_timestamp, 'atr']
                        except Exception:
                            atr_val = row.get('atr', None)
                        if atr_val and atr_val > 0:
                            stop_price = entry_price - (self.atr_multiplier * atr_val)
                except Exception:
                    stop_price = None

                if stop_price is None:
                    stop_price = entry_price * (1 - float(self.stoploss_pct))

                signal = {
                    'timestamp': entry_timestamp,
                    'direction': 'long',
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'weight': 1.0,
                    'metadata': {'type': 'ema_cross_long'}
                }
            elif bool(row.get('confirmed_short', False)):
                if self.ema_separation_min_pct and abs(row.get('ema_sep_pct', 0.0)) < self.ema_separation_min_pct:
                    continue
                # Trend slope: require slow EMA falling for shorts
                if row.get('ema_slow_slope', 0.0) >= 0:
                    continue

                # Engine owns execution/fill timing; keep entry fields informational
                entry_price = row['close']
                entry_timestamp = timestamp

                # Determine ATR-based stop for short
                stop_price = None
                try:
                    sl_type = None
                    try:
                        sl_type = self.config.stop_loss.type  # type: ignore[attr-defined]
                    except Exception:
                        sl_type = None

                    atr_enabled_cfg = False
                    try:
                        atr_enabled_cfg = bool(self.config.stop_loss.atr_enabled)  # type: ignore[attr-defined]
                    except Exception:
                        try:
                            atr_enabled_cfg = bool(getattr(self.config.stop_loss, 'atr_enabled', False))
                        except Exception:
                            atr_enabled_cfg = False

                    use_atr = (str(sl_type).upper() == 'ATR') or atr_enabled_cfg

                    if use_atr and 'atr' in df_signal.columns:
                        atr_val = None
                        try:
                            atr_val = df_signal.at[entry_timestamp, 'atr']
                        except Exception:
                            atr_val = row.get('atr', None)
                        if atr_val and atr_val > 0:
                            stop_price = entry_price + (self.atr_multiplier * atr_val)
                except Exception:
                    stop_price = None

                if stop_price is None:
                    stop_price = entry_price * (1 + float(self.stoploss_pct))

                signal = {
                    'timestamp': entry_timestamp,
                    'direction': 'short',
                    'entry_price': entry_price,
                    'stop_price': stop_price,
                    'weight': 1.0,
                    'metadata': {'type': 'ema_cross_short'}
                }
            else:
                continue

            # Apply configured filters using base class method
            if not self.apply_filters(
                signal=pd.Series(signal),
                timestamp=timestamp,
                symbol=symbol or signal.get('metadata', {}).get('symbol', ''),
                df_by_tf=df_by_tf
            ):
                continue

            signals.append(signal)

        if not signals:
            return self._create_signal_dataframe()

        signals_df = pd.DataFrame(signals)
        signals_df.set_index('timestamp', inplace=True)
        return signals_df


__all__ = ["EMACrossoverStrategy"]
