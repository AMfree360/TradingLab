import pandas as pd
import numpy as np
from types import SimpleNamespace

from strategies.donchian_breakout import DonchianBreakout


def _build_price_df(n=50, start='2023-01-01', freq='4H'):
    idx = pd.date_range(start=start, periods=n, freq=freq)
    # build a gentle uptrend with noise
    price = np.linspace(100.0, 120.0, n) + np.random.normal(0, 0.5, n)
    df = pd.DataFrame(index=idx)
    df['open'] = price + np.random.normal(0, 0.2, n)
    df['high'] = df['open'] + np.abs(np.random.normal(0.5, 0.2, n))
    df['low'] = df['open'] - np.abs(np.random.normal(0.5, 0.2, n))
    df['close'] = price + np.random.normal(0, 0.2, n)
    df['volume'] = np.random.randint(1, 100, n)
    return df


def test_donchian_import_and_signals():
    cfg = SimpleNamespace()
    cfg.strategy_name = 'donchian_test'
    tf = SimpleNamespace()
    tf.signal_tf = '4h'
    tf.entry_tf = '4h'
    cfg.timeframes = tf
    params = SimpleNamespace(entry_lookback=5, exit_lookback=3)
    cfg.params = params
    sl = SimpleNamespace(atr_multiplier=2.5, atr_length=5)
    cfg.stop_loss = sl

    strat = DonchianBreakout(cfg)
    df = _build_price_df(n=60, freq='4H')
    df_by_tf = {'4h': df}

    inds = strat.get_indicators(df.copy(), tf='4h')
    assert 'donchian_high' in inds.columns
    sigs = strat.generate_signals(df_by_tf)
    # signals may be empty on small synthetic data, but should be a DataFrame
    assert hasattr(sigs, 'index')
