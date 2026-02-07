# validation/monte_carlo/permutation.py
"""
Corrected Monte Carlo Permutation test.

Key points:
 - Permute trade-level *returns* computed as:
       return_i = trade_pnl_i / equity_at_trade_entry_i
   where equity_at_trade_entry_i is taken from the original backtest equity curve.
 - Apply permuted returns sequentially with compounding:
       equity_{t+1} = equity_t * (1 + return_perm[t])
 - Recompute metrics from the synthetic (trade-level) equity curve.
 - Do NOT permute profit factor (PF) as PF is order-insensitive; include PF only
   as an informational check (will usually be invariant).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import warnings
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, desc=None): return it

from engine.backtest_engine import BacktestResult, Trade
from metrics.metrics import calculate_enhanced_metrics

# Utility: safe extraction of trade pnl
def _trade_pnl(trade: Trade) -> float:
    # try common attribute names
    for attr in ('net_pnl', 'pnl_after_costs', 'pnl', 'profit'):
        if hasattr(trade, attr):
            val = getattr(trade, attr)
            try:
                return float(val)
            except Exception:
                continue
    # fallback: try entry/exit prices * size if available
    if hasattr(trade, 'entry_price') and hasattr(trade, 'exit_price') and hasattr(trade, 'quantity'):
        try:
            return float((trade.exit_price - trade.entry_price) * trade.quantity) if trade.quantity else 0.0
        except Exception:
            pass
    # Also try 'size' if 'quantity' not available
    if hasattr(trade, 'entry_price') and hasattr(trade, 'exit_price') and hasattr(trade, 'size'):
        try:
            direction_mult = 1.0 if getattr(trade, 'direction', 'long') == 'long' else -1.0
            return float((trade.exit_price - trade.entry_price) * trade.size * direction_mult)
        except Exception:
            pass
    raise AttributeError("Trade object does not expose a known PnL attribute")


def _equity_at_timestamp(equity_series: pd.Series, ts: pd.Timestamp, default: float) -> float:
    """Return equity value at or immediately before timestamp ts from equity_series."""
    if equity_series is None or len(equity_series) == 0:
        return default
    # Ensure index is DatetimeIndex
    idx = equity_series.index
    try:
        # if exact timestamp present
        if ts in idx:
            return float(equity_series.loc[ts])
        # else find last earlier index
        earlier = idx[idx <= ts]
        if len(earlier) == 0:
            return float(equity_series.iloc[0])
        return float(equity_series.loc[earlier[-1]])
    except Exception:
        # fallback to first element
        return float(equity_series.iloc[0])


@dataclass
class PermutationResult:
    observed_metrics: Dict[str, float]
    permuted_distributions: Dict[str, np.ndarray]
    p_values: Dict[str, float]
    percentiles: Dict[str, float]
    n_iterations: int
    equity_curves: Optional[List[pd.Series]] = None


class MonteCarloPermutation:
    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.seed = seed

    def _compute_entry_equities_and_returns(
        self,
        backtest_result: BacktestResult
    ) -> Tuple[np.ndarray, np.ndarray, float, List[pd.Timestamp]]:
        """
        Compute:
          - trade_pnls: array of raw PnL per trade (floats)
          - trade_returns: array of return_i = pnl_i / equity_at_entry_i
          - initial_capital
          - entry_timestamps list (for possible diagnostics)
        
        PREFERRED: Use precomputed trade_returns from BacktestResult if available.
        FALLBACK: Compute from trades and equity_curve if not available.
        """
        trades = backtest_result.trades
        initial_capital = float(getattr(backtest_result, 'initial_capital', 0.0) or 0.0)
        
        # PREFERRED: Use precomputed trade_returns from engine (Monte Carlo compatible)
        if hasattr(backtest_result, 'trade_returns') and len(backtest_result.trade_returns) == len(trades):
            trade_returns = backtest_result.trade_returns
            # Extract P&Ls for reference
            trade_pnls = np.array([_trade_pnl(t) for t in trades], dtype=float)
            entry_times = [getattr(t, 'entry_time', None) or getattr(t, 'exit_time', None) for t in trades]
            return trade_pnls, trade_returns, initial_capital, entry_times
        
        # FALLBACK: Compute from trades and equity_curve
        equity_series = getattr(backtest_result, 'equity_curve', None)
        trade_pnls = []
        trade_returns = []
        entry_times = []
        for t in trades:
            pnl = _trade_pnl(t)
            trade_pnls.append(pnl)
            entry_ts = getattr(t, 'entry_time', None) or getattr(t, 'exit_time', None)
            entry_times.append(entry_ts)
            eq_at_entry = _equity_at_timestamp(equity_series, pd.Timestamp(entry_ts), initial_capital) if entry_ts is not None else initial_capital
            # avoid division by zero
            base = max(eq_at_entry, 1e-9)
            ret = pnl / base
            trade_returns.append(ret)

        return np.array(trade_pnls, dtype=float), np.array(trade_returns, dtype=float), initial_capital, entry_times

    def _simulate_from_returns(self, returns: np.ndarray, initial_capital: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply sequential compounding of returns.
        Returns:
          - equity_path: array length = len(returns) + 1 (initial + after each trade)
          - pnl_path: array length = len(returns) (pnl amounts per step = return * equity_before)
        """
        n = len(returns)
        equity = np.empty(n + 1, dtype=float)
        pnl_path = np.empty(n, dtype=float)
        equity[0] = initial_capital
        for i in range(n):
            pnl = returns[i] * equity[i]
            pnl_path[i] = pnl
            equity[i + 1] = equity[i] + pnl  # equivalent to equity[i] * (1 + returns[i])
        return equity, pnl_path

    def _validate_distribution(self, arr: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Basic checks for a valid numeric distribution with non-zero variance."""
        if arr is None or len(arr) == 0:
            return False, "empty distribution"
        if not np.isfinite(arr).all():
            return False, "non-finite values present"
        std = float(np.std(arr, ddof=1))
        mean = float(np.mean(arr))
        # relative std threshold
        rel = (std / abs(mean)) if abs(mean) > 0 else std
        if std == 0 or rel < 1e-9:
            return False, f"Distribution variance too low (std={std:.3e}, mean={mean:.3e}, std/mean={rel:.3e})"
        return True, None

    def _empirical_p_and_percentile(self, simulated: np.ndarray, observed: float) -> Tuple[float, float]:
        """Empirical p-value and percentile (one-sided p = fraction >= observed)."""
        n = len(simulated)
        if n == 0:
            return 1.0, 0.0
        ge = np.sum(simulated >= observed)
        lt = np.sum(simulated < observed)
        p = float(ge / n)
        perc = float((lt / n) * 100.0)
        return p, perc

    def run(
        self,
        backtest_result: BacktestResult,
        metrics: Optional[List[str]] = None,
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> PermutationResult:
        """
        Run permutation MC.
        metrics: choose among ['final_pnl', 'sharpe', 'profit_factor', 'total_return_pct']
        NOTES:
          - PF is invariant to order; permutation test is not meaningful for PF.
          - We compute PF distribution as informational but will warn if invariant.
        """
        if metrics is None:
            metrics = ['final_pnl', 'sharpe', 'profit_factor']

        # Extract trade pnls and returns
        pnls, returns, initial_capital, entry_times = self._compute_entry_equities_and_returns(backtest_result)

        if len(returns) == 0:
            # empty result
            observed = calculate_enhanced_metrics(backtest_result)
            return PermutationResult(
                observed_metrics=observed,
                permuted_distributions={m: np.array([]) for m in metrics},
                p_values={m: 1.0 for m in metrics},
                percentiles={m: 0.0 for m in metrics},
                n_iterations=n_iterations,
                equity_curves=None
            )

        # Observed metrics (use metrics pipeline)
        observed_metrics = calculate_enhanced_metrics(backtest_result)
        # Ensure final_pnl present
        if 'final_pnl' in metrics and 'final_pnl' not in observed_metrics:
            observed_metrics['final_pnl'] = backtest_result.final_capital - backtest_result.initial_capital if hasattr(backtest_result, 'final_capital') else backtest_result.total_pnl

        # Prepare storage
        permuted_distributions = {m: np.empty(n_iterations, dtype=float) for m in metrics}
        equity_curves = [] if n_iterations <= 1000 else None

        iterator = tqdm(range(n_iterations), desc="Permutation MC") if show_progress else range(n_iterations)
        for i in iterator:
            # Shuffle the trade *returns* (not PnLs)
            permuted_returns = self.rng.permutation(returns)
            # Simulate compounding
            equity_path, pnl_path = self._simulate_from_returns(permuted_returns, initial_capital)
            # Build a minimal BacktestResult-like object for metrics calculation
            # We'll create a shallow copy of the original and override trades & equity
            # Instead of attempting to rebuild full Trade objects with mismatched fields,
            # feed metrics a BacktestResult stub with trades replaced by synthetic trades.
            # Construct synthetic trades with pnl amounts from pnl_path and timestamps copied from original trades order.
            synthetic_trades = []
            for orig_trade, pnl_amount in zip(backtest_result.trades, pnl_path):
                # Create Trade object with new structure (Monte Carlo compatible)
                # Use quantity (not size), pnl_after_costs (not net_pnl)
                t = Trade(
                    entry_time=orig_trade.entry_time,
                    exit_time=orig_trade.exit_time,
                    direction=orig_trade.direction,
                    entry_price=getattr(orig_trade, 'entry_price', 0.0),
                    exit_price=getattr(orig_trade, 'exit_price', 0.0),
                    quantity=getattr(orig_trade, 'quantity', getattr(orig_trade, 'size', 0.0)),  # Use quantity
                    pnl_raw=float(pnl_amount),  # Raw P&L
                    pnl_after_costs=float(pnl_amount),  # P&L after costs (assume costs already included in returns calc)
                    commission=getattr(orig_trade, 'commission', 0.0),
                    slippage=getattr(orig_trade, 'slippage', 0.0),
                    stop_price=getattr(orig_trade, 'stop_price', None),
                    exit_reason=getattr(orig_trade, 'exit_reason', None),
                    partial_exits=getattr(orig_trade, 'partial_exits', [])
                )
                synthetic_trades.append(t)

            # Build a simple BacktestResult-like object (dataclass in your engine can accept)
            # Use keys from your BacktestResult signature — we construct using kwargs to be robust
            permuted_bt = BacktestResult(
                trades=synthetic_trades,
                equity_curve=pd.Series(equity_path),  # index-less series (trade-level)
                initial_capital=initial_capital,
                final_capital=float(equity_path[-1]),
                total_trades=len(synthetic_trades),
                winning_trades=sum(1 for x in synthetic_trades if getattr(x, 'net_pnl', getattr(x, 'gross_pnl', 0.0)) > 0),
                losing_trades=sum(1 for x in synthetic_trades if getattr(x, 'net_pnl', getattr(x, 'gross_pnl', 0.0)) <= 0),
                total_pnl=float(equity_path[-1] - initial_capital),
                total_commission=sum(getattr(x, 'commission', 0.0) for x in synthetic_trades),
                total_slippage=sum(getattr(x, 'slippage', 0.0) for x in synthetic_trades),
                max_drawdown=0.0,  # will be computed by metrics
                strategy_name=getattr(backtest_result, 'strategy_name', 'PERMUTED'),
                symbol=getattr(backtest_result, 'symbol', 'PERMUTED'),
                win_rate=(sum(1 for x in synthetic_trades if getattr(x, 'net_pnl', getattr(x, 'gross_pnl', 0.0)) > 0) / len(synthetic_trades) * 100.0) if synthetic_trades else 0.0
            )

            if equity_curves is not None:
                # store a pandas Series with simple integer index for visualization
                equity_curves.append(pd.Series(equity_path))

            # Compute metrics from permuted result
            perm_metrics = calculate_enhanced_metrics(permuted_bt)

            # Ensure final_pnl present
            if 'final_pnl' in metrics and 'final_pnl' not in perm_metrics:
                perm_metrics['final_pnl'] = permuted_bt.total_pnl

            for m in metrics:
                # For metrics that are intrinsically order-insensitive (profit_factor),
                # we still store what was computed, but permutation test is not meaningful.
                permuted_distributions[m][i] = float(perm_metrics.get(m, 0.0))

        # Calculate p-values and percentiles
        p_values = {}
        percentiles = {}
        for m in metrics:
            observed = float(observed_metrics.get(m, 0.0))
            sim = permuted_distributions[m]

            # Validate distribution
            valid, err = self._validate_distribution(sim)
            if not valid:
                # If distribution invalid (likely constant), return conservative p-value
                warnings.warn(f"Permutation test {m}: {err}. Using conservative p-value=1.0")
                p_values[m] = 1.0
                percentiles[m] = 0.0
                continue

            p, perc = self._empirical_p_and_percentile(sim, observed)
            p_values[m] = p
            percentiles[m] = perc

        return PermutationResult(
            observed_metrics=observed_metrics,
            permuted_distributions=permuted_distributions,
            p_values=p_values,
            percentiles=percentiles,
            n_iterations=n_iterations,
            equity_curves=equity_curves
        )
