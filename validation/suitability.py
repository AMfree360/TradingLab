"""Strategy Suitability Assessment for Validation Tests.

This module determines which validation tests are appropriate for a given strategy
based on its characteristics (return variance, exit uniformity, sample size, etc.).

Industry Best Practice: Not all tests are suitable for all strategies. Running
inappropriate tests leads to:
- False negatives (rejecting good strategies)
- Wasted compute
- Confusion about test results

This module implements adaptive test selection based on strategy characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
from collections import Counter

from engine.backtest_engine import BacktestResult


@dataclass
class TestSuitability:
    """Suitability information for a validation test.
    
    Attributes:
        suitable: Whether the test is suitable for this strategy
        reason: Explanation of why the test is/isn't suitable
        priority: Weight in combined score (0.0-1.0)
        alternatives: List of alternative test names if not suitable
        category: 'universal' or 'conditional'. Universal tests may be used for pass/fail
                  gating when suitable; conditional tests are informational unless explicitly
                  elevated elsewhere.
    """
    suitable: bool
    reason: str
    priority: float = 0.0  # Weight in combined score
    alternatives: List[str] = field(default_factory=list)
    category: str = 'universal'


@dataclass
class StrategyProfile:
    """Characterizes strategy for test suitability assessment.
    
    Attributes:
        return_cv: Coefficient of variation of trade returns
        exit_uniformity: Percentage of trades with same exit reason
        n_trades: Number of completed trades
        n_bars: Number of bars in backtest
        avg_hold_time_cv: Coefficient of variation of holding times
        strategy_type: Classification ('systematic_mechanical', 'dynamic_discretionary', 'hybrid')
        final_equity_cv: Coefficient of variation of final equity when returns are permuted
    """
    return_cv: float  # Coefficient of variation of trade returns
    exit_uniformity: float  # % of trades with same exit reason
    n_trades: int
    n_bars: int
    avg_hold_time_cv: float
    strategy_type: str  # 'systematic_mechanical', 'dynamic_discretionary', 'hybrid'
    final_equity_cv: float = 0.0  # CV of final equity from quick permutation test


class ValidationSuitabilityAssessor:
    """Determines which validation tests are appropriate for a strategy.
    
    This class implements industry-standard test suitability assessment:
    - Permutation test: Requires sufficient return variance (CV > 1%), sample size (n >= 30),
                        AND returns large enough that order matters (final_equity_cv > 0.1%)
    - Bootstrap test: Requires sufficient bars (n >= 100)
    - Randomized Entry: Almost always suitable (tests true edge)
    
    Usage:
        assessor = ValidationSuitabilityAssessor()
        profile = assessor.assess_strategy(backtest_result)
        suitability = assessor.get_test_suitability(profile)
    """
    
    def _quick_permutation_test(
        self, 
        returns: np.ndarray, 
        initial_capital: float,
        n_quick_tests: int = 20
    ) -> float:
        """Quick test: permute returns multiple times and check if final equity varies.
        
        This tests whether the ORDER of returns matters for final equity.
        If returns are very small, compounding effects are minimal and permuting
        returns produces nearly identical final equity, making permutation test useless.
        
        Args:
            returns: Array of trade returns (percentage returns)
            initial_capital: Initial capital
            n_quick_tests: Number of quick permutations to test (default: 20)
            
        Returns:
            Coefficient of variation of final equity from permutations
            If CV is very low (< 0.001), permutation test won't have statistical power
        """
        if len(returns) == 0 or initial_capital <= 0:
            return 0.0
        
        final_equities = []
        rng = np.random.default_rng(42)  # Seeded for reproducibility
        
        for _ in range(n_quick_tests):
            # Permute returns
            permuted = rng.permutation(returns)
            
            # Apply compounding
            equity = initial_capital
            for ret in permuted:
                equity = equity * (1 + ret)
            
            final_equities.append(equity)
        
        if len(final_equities) > 1:
            final_equities_arr = np.array(final_equities)
            mean_final = np.mean(final_equities_arr)
            
            if abs(mean_final) > 1e-6:
                final_cv = float(np.std(final_equities_arr, ddof=1) / abs(mean_final))
            else:
                final_cv = 0.0
            
            return final_cv
        
        return 0.0
    
    def assess_strategy(self, backtest_result: BacktestResult, n_bars: Optional[int] = None) -> StrategyProfile:
        """Assess strategy characteristics from backtest results.
        
        Args:
            backtest_result: BacktestResult from training data
            
        Returns:
            StrategyProfile with strategy characteristics
        """
        trades = backtest_result.trades
        
        # Get trade returns (preferred: use precomputed from engine)
        if hasattr(backtest_result, 'trade_returns') and len(backtest_result.trade_returns) > 0:
            returns = backtest_result.trade_returns
        else:
            # Fallback: compute from trades
            if backtest_result.initial_capital > 0:
                returns = np.array([
                    t.pnl_after_costs / backtest_result.initial_capital 
                    for t in trades
                ])
            else:
                returns = np.array([t.pnl_after_costs for t in trades])
        
        # Calculate return uniformity (Coefficient of Variation)
        if len(returns) > 0 and abs(np.mean(returns)) > 1e-6:
            return_cv = float(np.std(returns) / abs(np.mean(returns)))
        else:
            return_cv = np.inf  # No returns or zero mean
        
        # Calculate exit uniformity
        exit_reasons = Counter([t.exit_reason for t in trades if t.exit_reason])
        exit_uniformity = max(exit_reasons.values()) / len(trades) if trades else 0.0
        
        # Calculate holding time CV
        hold_times = []
        for t in trades:
            if hasattr(t, 'entry_time') and hasattr(t, 'exit_time'):
                try:
                    delta = (t.exit_time - t.entry_time).total_seconds()
                    if delta > 0:
                        hold_times.append(delta)
                except (AttributeError, TypeError):
                    pass
        
        avg_hold_time_cv = 0.0
        if len(hold_times) > 1:
            mean_hold = np.mean(hold_times)
            if mean_hold > 0:
                avg_hold_time_cv = float(np.std(hold_times) / mean_hold)
        
        # Classify strategy type
        if exit_uniformity > 0.8 and return_cv < 0.01:
            strategy_type = 'systematic_mechanical'
        elif return_cv > 0.1:
            strategy_type = 'dynamic_discretionary'
        else:
            strategy_type = 'hybrid'
        
        # Get number of bars.
        # IMPORTANT: equity_curve length is often trade-based (not bar-based), which can
        # incorrectly mark bar-dependent tests (e.g., block bootstrap) as "not suitable".
        if n_bars is None:
            if backtest_result.equity_curve is not None:
                n_bars = len(backtest_result.equity_curve)
            else:
                # Estimate from trades (rough approximation)
                n_bars = len(trades) * 10  # Rough estimate
        
        # CRITICAL: Quick permutation test to check if order matters
        # This tests whether permuting returns actually changes final equity
        # If returns are too small, compounding effects are minimal and order doesn't matter
        final_equity_cv = self._quick_permutation_test(
            returns, 
            backtest_result.initial_capital
        )
        
        return StrategyProfile(
            return_cv=return_cv,
            exit_uniformity=exit_uniformity,
            n_trades=len(trades),
            n_bars=n_bars,
            avg_hold_time_cv=avg_hold_time_cv,
            strategy_type=strategy_type,
            final_equity_cv=final_equity_cv
        )
    
    def get_test_suitability(
        self, 
        profile: StrategyProfile
    ) -> Dict[str, TestSuitability]:
        """Determine which tests are suitable based on strategy profile.
        
        Args:
            profile: StrategyProfile from assess_strategy()
            
        Returns:
            Dictionary mapping test names to TestSuitability objects
        """
        suitability = {}
        
        # === Permutation Test Suitability ===
        # Requires THREE conditions:
        # 1. Return variance: CV > 1% (sufficient return variance)
        # 2. Sample size: n_trades >= 30
        # 3. Order matters: final_equity_cv > 0.1% (permuting returns changes final equity)
        
        if profile.return_cv < 0.01:
            suitability['permutation'] = TestSuitability(
                suitable=False,
                reason=f"Trade returns too uniform (CV={profile.return_cv:.4f} < 1%). "
                       "Permutation test lacks statistical power for mechanical strategies. "
                       "Strategy has uniform exits that homogenize outcomes.",
                alternatives=['r_multiple_bootstrap', 'runs_test', 'bootstrap']
            )
        elif profile.n_trades < 30:
            suitability['permutation'] = TestSuitability(
                suitable=False,
                reason=f"Insufficient trades ({profile.n_trades} < 30) for reliable permutation test. "
                       "Sample size too small for meaningful statistical inference.",
                alternatives=['bootstrap', 'randomized_entry']
            )
        elif profile.final_equity_cv < 0.001:
            # CRITICAL CHECK: Does permuting returns actually change final equity?
            # If final_equity_cv < 0.1%, returns are too small for order to matter
            suitability['permutation'] = TestSuitability(
                suitable=False,
                reason=f"Permuting returns produces near-identical final equity "
                       f"(final_equity_cv={profile.final_equity_cv:.6f} < 0.1%). "
                       "Returns are too small relative to equity for compounding effects to matter. "
                       "Permutation test lacks statistical power - order of trades doesn't affect outcome.",
                alternatives=['bootstrap', 'randomized_entry', 'r_multiple_bootstrap']
            )
        else:
            suitability['permutation'] = TestSuitability(
                suitable=True,
                reason=f"Trade returns show sufficient variance (CV={profile.return_cv:.4f}), "
                       f"sample size ({profile.n_trades} trades), and order matters "
                       f"(final_equity_cv={profile.final_equity_cv:.4f}) for permutation test.",
                priority=0.2,
                category='universal'
            )
        
        # === Bootstrap Test Suitability ===
        # Requires: n_bars >= 100 (sufficient data for block bootstrap)
        if profile.n_bars < 100:
            suitability['bootstrap'] = TestSuitability(
                suitable=False,
                reason=f"Insufficient bars ({profile.n_bars} < 100) for block bootstrap. "
                       "Need sufficient data to create meaningful blocks.",
                alternatives=['permutation', 'randomized_entry'],
                category='universal'
            )
        else:
            suitability['bootstrap'] = TestSuitability(
                suitable=True,
                reason=f"Sufficient data ({profile.n_bars} bars) for block bootstrap test.",
                priority=0.3,
                category='universal'
            )
        
        # === Randomized Entry Test Suitability ===
        # This test is often informative, but not universally fair as a hard gate.
        # Treat it as CONDITIONAL: run/report when suitable, but don't use it for
        # universal pass/fail gating by default.
        suitability['randomized_entry'] = TestSuitability(
            suitable=True,
            reason="Tests true edge regardless of strategy type. "
                   "Compares strategy performance to random entries with identical risk management.",
            priority=0.5,
            category='conditional'
        )
        
        return suitability

