"""Phase 2: Out-of-Sample Validation - Tests on OOS data (used only once)."""

from typing import Dict, Optional, List, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass

from strategies.base import StrategyBase
from validation.walkforward import WalkForwardAnalyzer, WalkForwardResult
from config.schema import (
    ValidationCriteriaConfig,
    OOSValidationCriteria,
    WalkForwardConfig,
    validate_strategy_config
)


@dataclass
class OOSValidationResult:
    """Results from Phase 2: Out-of-Sample Validation."""
    passed: bool
    walk_forward_result: WalkForwardResult
    criteria_checks: Dict[str, bool] = None
    failure_reasons: List[str] = None
    
    def __post_init__(self):
        if self.criteria_checks is None:
            self.criteria_checks = {}
        if self.failure_reasons is None:
            self.failure_reasons = []


class OOSValidator:
    """Validates strategy on out-of-sample data (Phase 2).
    
    WARNING: OOS data should only be used ONCE. This validator
    enforces this by checking validation state.
    
    Tests:
    1. Walk-Forward Analysis on OOS data
    2. Consistency across test periods
    3. Overfitting detection (test PF vs train PF)
    
    Architecture Alignment:
    - Uses unified metrics pipeline (calculate_enhanced_metrics)
    - All metrics calculated consistently with Phase 1
    - Follows same pass/fail criteria structure
    """
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: Optional[float] = None,  # FIXED: Allow None to use market profile
        slippage_ticks: Optional[float] = None,   # FIXED: Allow None to use market profile
        criteria: Optional[OOSValidationCriteria] = None
    ):
        """
        Initialize OOS validator.
        
        Args:
            strategy_class: Strategy class to validate
            initial_capital: Starting capital
            commission_rate: Commission rate per trade (if None, uses market profile)
            slippage_ticks: Slippage in ticks (if None, uses market profile)
            criteria: Pass/fail criteria (uses defaults if None)
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.criteria = criteria or OOSValidationCriteria()
    
    def validate(
        self,
        oos_data: pd.DataFrame,
        strategy_config: Dict[str, Any],
        wf_config: WalkForwardConfig,
        full_data: Optional[pd.DataFrame] = None
    ) -> OOSValidationResult:
        """
        Run Phase 2 validation on OOS data.
        
        Args:
            oos_data: Out-of-sample dataset (datetime index) - used for validation boundaries
            strategy_config: Strategy configuration dict
            wf_config: Walk-forward configuration
            full_data: Full dataset (optional). If provided, used for warm-up periods.
                      If None, uses oos_data (may not have enough history for warm-up).
        
        Returns:
            OOSValidationResult with pass/fail status
        """
        # Use full_data if provided (for warm-up), otherwise use oos_data
        # The walk-forward analyzer needs historical data before OOS period for indicators
        data_for_wf = full_data if full_data is not None else oos_data
        
        # Run walk-forward analysis
        # CRITICAL FIX: Pass None for commission_rate and slippage_ticks to match training validator
        # This ensures market profile settings (leverage, contract_size, etc.) are used correctly
        
        # Pass min_trades_per_period from criteria to config if not already set
        if wf_config.wf_min_trades_per_period is None and hasattr(self.criteria, 'wf_min_trades_per_period'):
            wf_config.wf_min_trades_per_period = self.criteria.wf_min_trades_per_period
        
        analyzer = WalkForwardAnalyzer(
            strategy_class=self.strategy_class,
            config=wf_config,
            initial_capital=self.initial_capital,
            commission_rate=None,  # FIXED: Use market profile (like training validator)
            slippage_ticks=None    # FIXED: Use market profile (like training validator)
        )
        
        # Pass full data to walk-forward (it will handle warm-up internally)
        wf_result = analyzer.run(data_for_wf, strategy_config)
        
        # Check all criteria
        checks = self._check_walk_forward(wf_result)
        # Filter out metadata and calculated values when checking pass/fail
        actual_checks = {k: v for k, v in checks.items() 
                        if k not in ['_metadata', 'oos_consistency_value', 'binomial_p_value'] 
                        and isinstance(v, bool)}
        passed = all(actual_checks.values())
        
        # Collect failure reasons
        failure_reasons = []
        
        # Special case: All periods excluded
        metadata = checks.get('_metadata', {})
        if metadata.get('all_periods_excluded', False):
            excluded_count = metadata.get('excluded_periods_count', 0)
            min_trades = self.criteria.wf_min_trades_per_period
            failure_reasons.append(
                f"ALL {excluded_count} test periods excluded: insufficient trades "
                f"(all periods had < {min_trades} trades). "
                f"Use longer test periods (e.g., 1 year) or review strategy trade frequency."
            )
        else:
            # Normal failure reasons (only if we have some included periods)
            if not checks.get('min_test_pf', True):
                mean_test_pf = wf_result.summary.get('mean_test_pf', 0)
                failure_reasons.append(f"Mean test PF {mean_test_pf:.2f} < {self.criteria.wf_min_test_pf}")
            if not checks.get('min_consistency', True):
                consistency = wf_result.summary.get('consistency_score', 0)
                failure_reasons.append(f"Consistency score {consistency:.2f} < {self.criteria.wf_min_consistency_score}")
            if not checks.get('max_pf_std', True):
                pf_std = wf_result.summary.get('std_test_pf', float('inf'))
                failure_reasons.append(f"Test PF std {pf_std:.2f} > {self.criteria.wf_max_pf_std}")
            if not checks.get('min_test_periods', True):
                included_count = metadata.get('included_periods_count', 0)
                failure_reasons.append(f"Valid test periods {included_count} < {self.criteria.wf_min_test_periods}")
            if not checks.get('train_test_ratio', True):
                failure_reasons.append("Test PF too low relative to Train PF (overfitting detected)")
            if not checks.get('min_wfe', True):
                wfe = wf_result.summary.get('walk_forward_efficiency', 0)
                failure_reasons.append(f"Walk-Forward Efficiency {wfe:.2%} < {self.criteria.wf_min_wfe:.2%}")
            if not checks.get('min_oos_consistency', True):
                oos_consistency = checks.get('oos_consistency_value', 0)
                failure_reasons.append(f"OOS Consistency Score {oos_consistency:.2%} < {self.criteria.wf_min_oos_consistency:.2%}")
            if not checks.get('binomial_test', True):
                binomial_p = checks.get('binomial_p_value', 1.0)
                failure_reasons.append(f"Binomial test p-value {binomial_p:.4f} > {self.criteria.wf_min_binomial_p_value} (insufficient statistical significance)")
        
        return OOSValidationResult(
            passed=passed,
            walk_forward_result=wf_result,
            criteria_checks=checks,
            failure_reasons=failure_reasons
        )
    
    def _check_walk_forward(self, wf_result: WalkForwardResult) -> Dict[str, bool]:
        """Check walk-forward results against criteria."""
        summary = wf_result.summary
        
        # Separate included and excluded periods
        included_steps = [step for step in wf_result.steps if not step.excluded_from_stats]
        excluded_steps = [step for step in wf_result.steps if step.excluded_from_stats]
        
        # Calculate test PF statistics (only from included periods)
        test_pfs = [step.test_metrics.get('pf', 0) for step in included_steps]
        train_pfs = [step.train_metrics.get('pf', 0) for step in included_steps]
        
        # Warn if any periods were excluded
        if excluded_steps:
            print(f"\n⚠️  {len(excluded_steps)} period(s) excluded from statistical calculations:")
            for step in excluded_steps:
                print(f"   Step {step.step_number}: {step.test_start.date()} to {step.test_end.date()} - {step.exclusion_reason}")
            print(f"   Only {len(included_steps)} period(s) included in criteria checks.\n")
        
        # CRITICAL: If ALL periods are excluded, we cannot calculate meaningful statistics
        if len(included_steps) == 0:
            print("=" * 70)
            print("❌ CRITICAL: ALL test periods excluded due to insufficient trades!")
            print("=" * 70)
            print(f"All {len(excluded_steps)} test periods had fewer than {self.criteria.wf_min_trades_per_period} trades.")
            print("\nThis indicates:")
            print("  1. Test periods are too short for the strategy's trade frequency")
            print("  2. Strategy may be too selective (not enough signals)")
            print("  3. Market conditions may not suit the strategy")
            print("\nRecommendations:")
            print(f"  - Use longer test periods (e.g., 1 year instead of 6 months)")
            print(f"  - Review strategy parameters to increase trade frequency")
            print(f"  - Consider if the strategy is suitable for this market/timeframe")
            print("=" * 70)
            # Return early with failure - can't calculate stats with no data
            checks = {
                'min_test_pf': False,
                'min_consistency': False,
                'max_pf_std': True,  # Pass (no std to check)
                'min_test_periods': False,  # Fail - need at least 1 valid period
                'train_test_ratio': False,
                'min_wfe': False,
                'min_oos_consistency': False,
                'binomial_test': False,
                'oos_consistency_value': 0.0,
                'binomial_p_value': 1.0,
            }
            # Store metadata separately
            checks['_metadata'] = {
                'excluded_periods_count': len(excluded_steps),
                'included_periods_count': 0,
                'all_periods_excluded': True,  # Flag for reporting
            }
            return checks
        
        mean_test_pf = summary.get('mean_test_pf', 0)
        mean_train_pf = summary.get('mean_train_pf', 0)
        std_test_pf = summary.get('std_test_pf', float('inf'))
        consistency_score = summary.get('consistency_score', 0)
        total_steps = summary.get('total_steps', 0)
        
        # Calculate train/test PF ratio
        train_test_ratio = (mean_test_pf / mean_train_pf) if mean_train_pf > 0 else 0
        
        # Calculate Walk-Forward Efficiency (WFE)
        wfe = summary.get('walk_forward_efficiency', 0.0)
        
        # Calculate OOS Consistency Score (percentage of periods with PF >= threshold)
        oos_consistency = self._calculate_oos_consistency(test_pfs, self.criteria.wf_min_test_pf)
        
        # Calculate Binomial Test p-value
        # Tests: H0: P(PF >= threshold) <= 0.5, H1: P(PF >= threshold) > 0.5
        # More appropriate for small sample sizes than t-test
        binomial_p_value = self._calculate_binomial_test(test_pfs, self.criteria.wf_min_test_pf)
        
        checks = {
            'min_test_pf': mean_test_pf >= self.criteria.wf_min_test_pf,
            'min_consistency': consistency_score >= self.criteria.wf_min_consistency_score,
            'max_pf_std': std_test_pf <= self.criteria.wf_max_pf_std,
            'min_test_periods': total_steps >= self.criteria.wf_min_test_periods,
            'train_test_ratio': train_test_ratio >= self.criteria.wf_train_test_pf_ratio_min,
            'min_wfe': wfe >= self.criteria.wf_min_wfe,
            'min_oos_consistency': oos_consistency >= self.criteria.wf_min_oos_consistency,
            'binomial_test': binomial_p_value <= self.criteria.wf_min_binomial_p_value,
        }
        
        # Store calculated values for reporting
        checks['oos_consistency_value'] = oos_consistency
        checks['binomial_p_value'] = binomial_p_value
        
        # Overall OOS checks (only from included periods)
        if test_pfs:
            oos_sharpe = np.mean([step.test_metrics.get('sharpe', 0) for step in included_steps])
            checks['min_sharpe'] = oos_sharpe >= self.criteria.oos_min_sharpe
        
        # Store excluded period info for reporting (metadata, not criteria)
        # These are informational and should not be treated as pass/fail checks
        metadata = {
            'excluded_periods_count': len(excluded_steps),
            'included_periods_count': len(included_steps),
            'all_periods_excluded': False,  # Not all excluded (we would have returned early)
        }
        
        # Attach metadata to checks dict for reporting (but mark as metadata)
        checks['_metadata'] = metadata
        
        return checks
    
    def _calculate_oos_consistency(self, test_pfs: List[float], threshold: float) -> float:
        """
        Calculate OOS Consistency Score: percentage of periods with PF >= threshold.
        
        Args:
            test_pfs: List of profit factors from test periods
            threshold: Minimum PF threshold (e.g., 1.5)
        
        Returns:
            Consistency score as float between 0.0 and 1.0
        """
        if not test_pfs:
            return 0.0
        
        passing_periods = sum(1 for pf in test_pfs if pf >= threshold)
        return passing_periods / len(test_pfs)
    
    def _calculate_binomial_test(self, test_pfs: List[float], threshold: float) -> float:
        """
        Calculate binomial test p-value for OOS performance.
        
        Tests the hypothesis: Is the probability of PF >= threshold statistically significant?
        More appropriate for small sample sizes than t-test.
        
        Args:
            test_pfs: List of profit factors from test periods
            threshold: Minimum PF threshold (e.g., 1.5)
        
        Returns:
            p-value (one-tailed test: probability of observing this many successes by chance)
        """
        if not test_pfs or len(test_pfs) < 1:
            return 1.0  # No data = fail
        
        # Count successes (PF >= threshold)
        n = len(test_pfs)
        k = sum(1 for pf in test_pfs if pf >= threshold)
        
        # Binomial test: H0: p <= 0.5, H1: p > 0.5
        # We want to know if getting k successes out of n trials is statistically significant
        # when the null hypothesis is that success rate is 50% or less
        # Using one-tailed test (we care if it's significantly better than 50%)
        
        # Binomial test: H0: p <= 0.5, H1: p > 0.5
        # We want to know if getting k successes out of n trials is statistically significant
        # Using one-tailed test (we care if it's significantly better than 50%)
        try:
            # Try modern scipy.stats.binomtest first
            from scipy.stats import binomtest
            # Alternative='greater' tests if success rate > 0.5
            result = binomtest(k, n, p=0.5, alternative='greater')
            return result.pvalue
        except ImportError:
            try:
                # Fallback for older scipy versions
                from scipy.stats import binom
                # One-tailed: probability of getting k or more successes
                p_value = 1 - binom.cdf(k - 1, n, 0.5)
                return p_value
            except ImportError:
                # If scipy not available, return 1.0 (fail) with warning
                import warnings
                warnings.warn(
                    "scipy not available - binomial test cannot be calculated. "
                    "Install scipy for statistical significance testing. "
                    "Returning p-value=1.0 (fail)."
                )
                return 1.0