import numpy as np
import pandas as pd
from validation.monte_carlo.permutation import MonteCarloPermutation
from validation.monte_carlo.block_bootstrap import MonteCarloBlockBootstrap
from validation.monte_carlo.randomized_entry import MonteCarloRandomizedEntry, RandomizedEntryConfig
from engine.backtest_engine import BacktestResult, Trade


def _build_simple_backtest():
    # Simple synthetic backtest with deterministic returns
    initial = 1000.0
    trade_returns = np.array([0.01, -0.005, 0.02, -0.01])
    equity = [initial]
    trades = []
    timestamps = pd.date_range(start="2023-01-01", periods=len(trade_returns) + 1, freq='min')
    for i, r in enumerate(trade_returns):
        before = equity[-1]
        pnl = before * r
        equity.append(before + pnl)
        t = Trade(
            entry_time=timestamps[i],
            exit_time=timestamps[i + 1],
            direction='long' if r >= 0 else 'short',
            entry_price=100.0 + i,
            exit_price=100.0 + i + r * 10,
            quantity=1.0,
            pnl_raw=pnl,
            pnl_after_costs=pnl,
            commission=0.0,
            slippage=0.0,
            stop_price=None,
            exit_reason='test'
        )
        trades.append(t)

    equity_series = pd.Series(equity, index=range(len(equity)))
    bt = BacktestResult(
        initial_capital=initial,
        final_capital=equity[-1],
        equity_curve=equity_series,
        trade_returns=trade_returns,
        trades=trades,
        total_pnl=equity[-1] - initial,
        total_trades=len(trades)
    )
    return bt


def _build_price_series(n=200):
    # simple deterministic price series
    idx = pd.date_range(start="2023-01-01", periods=n, freq='min')
    prices = pd.Series(100.0 + np.cumsum(np.ones(n) * 0.01), index=idx)
    return prices


def test_monte_carlo_deterministic_seeding():
    bt = _build_simple_backtest()
    prices = _build_price_series()

    seed = 12345
    n_iter = 50

    # Permutation
    perm_a = MonteCarloPermutation(seed=seed)
    res_a = perm_a.run(backtest_result=bt, n_iterations=n_iter, show_progress=False)
    perm_b = MonteCarloPermutation(seed=seed)
    res_b = perm_b.run(backtest_result=bt, n_iterations=n_iter, show_progress=False)

    # Ensure permuted distributions identical
    for k in res_a.permuted_distributions:
        assert np.array_equal(res_a.permuted_distributions[k], res_b.permuted_distributions[k])

    # Block bootstrap
    boot_a = MonteCarloBlockBootstrap(seed=seed)
    boot_b = MonteCarloBlockBootstrap(seed=seed)
    res_boot_a = boot_a.run(backtest_result=bt, price_series=prices, n_iterations=n_iter, show_progress=False)
    res_boot_b = boot_b.run(backtest_result=bt, price_series=prices, n_iterations=n_iter, show_progress=False)

    for k in res_boot_a.bootstrap_distributions:
        assert np.array_equal(res_boot_a.bootstrap_distributions[k], res_boot_b.bootstrap_distributions[k])

    # Randomized entry: use simple config and dummy strategy-like object
    cfg = RandomizedEntryConfig(enabled=True, mode='price', rng_seed=None)
    rand_a = MonteCarloRandomizedEntry(seed=seed, config=cfg)
    rand_b = MonteCarloRandomizedEntry(seed=seed, config=cfg)

    # Minimal dummy strategy object with necessary attributes
    class Dummy:
        pass

    strategy = Dummy()
    # Provide minimal config used by randomized entry fallback logic
    strategy.config = Dummy()
    strategy.config.stop_loss = Dummy()
    strategy.config.stop_loss.type = 'PERCENT'
    strategy.config.stop_loss.buffer_pips = 0
    strategy.config.risk = Dummy()
    strategy.config.risk.sizing_mode = 'account_size'
    strategy.config.risk.risk_per_trade_pct = 1.0
    strategy.config.risk.account_size = 1000.0
    strategy.config.take_profit = Dummy()
    strategy.config.take_profit.enabled = False

    # Build OHLCV DataFrame for randomized entry
    # give each bar a small spread so price randomization is possible
    spread = 0.05
    price_df = pd.DataFrame({
        'open': prices.values,
        'high': prices.values + spread,
        'low': prices.values - spread,
        'close': prices.values,
        'volume': np.zeros(len(prices))
    }, index=prices.index)

    res_rand_a = rand_a.run(backtest_result=bt, price_data=price_df, strategy=strategy, n_iterations=20, show_progress=False)
    res_rand_b = rand_b.run(backtest_result=bt, price_data=price_df, strategy=strategy, n_iterations=20, show_progress=False)

    for k in res_rand_a.random_distributions:
        assert np.array_equal(res_rand_a.random_distributions[k], res_rand_b.random_distributions[k])

    # Different seed should produce different distributions for at least one engine
    perm_c = MonteCarloPermutation(seed=seed + 1)
    res_c = perm_c.run(backtest_result=bt, n_iterations=n_iter, show_progress=False)
    different = False
    for k in res_a.permuted_distributions:
        if not np.array_equal(res_a.permuted_distributions[k], res_c.permuted_distributions[k]):
            different = True
            break
    assert different
