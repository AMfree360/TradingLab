"""Phase 1: Training Validation - Tests on training data only."""

from typing import Dict, Optional, List, Any
import pandas as pd
from dataclasses import dataclass

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine, BacktestResult
from validation.monte_carlo.runner import MonteCarloSuite
from validation.suitability import ValidationSuitabilityAssessor
from validation.sensitivity import SensitivityAnalyzer
from config.schema import (
    ValidationCriteriaConfig,
    TrainingValidationCriteria,
    validate_strategy_config
)
from metrics.metrics import calculate_enhanced_metrics


@dataclass
class TrainingValidationResult:
    """Results from Phase 1: Training Validation."""
    passed: bool
    backtest_result: BacktestResult
    monte_carlo_results: Dict[str, Any]
    sensitivity_results: Optional[Dict[str, Any]] = None
    criteria_checks: Dict[str, bool] = None
    failure_reasons: List[str] = None
    
    def __post_init__(self):
        if self.criteria_checks is None:
            self.criteria_checks = {}
        if self.failure_reasons is None:
            self.failure_reasons = []


class TrainingValidator:
    """Validates strategy on training data only (Phase 1).
    
    This phase must pass before OOS data can be used.
    Tests:
    1. Backtest quality on training data
    2. Monte Carlo Permutation (proves edge > random)
    3. Parameter Sensitivity (tests robustness)
    """
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: Optional[float] = None,
        slippage_ticks: Optional[float] = None,
        criteria: Optional[TrainingValidationCriteria] = None
    ):
        """
        Initialize training validator.
        
        Args:
            strategy_class: Strategy class to validate
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_ticks: Slippage in ticks
            criteria: Pass/fail criteria (uses defaults if None)
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.criteria = criteria or TrainingValidationCriteria()
        self.suitability_assessor = ValidationSuitabilityAssessor()  # NEW: Suitability assessment
    
    def validate(
        self,
        training_data: pd.DataFrame,
        strategy_config: Dict[str, Any],
        run_sensitivity: bool = True,
        sensitivity_params: Optional[Dict[str, List[Any]]] = None,
        mc_iterations: int = 1000
    ) -> TrainingValidationResult:
        """
        Run complete Phase 1 validation on training data.
        
        Args:
            training_data: Training dataset (datetime index)
            strategy_config: Strategy configuration dict
            run_sensitivity: Whether to run parameter sensitivity analysis
            sensitivity_params: Parameters to test for sensitivity (if None, skips)
            mc_iterations: Number of Monte Carlo iterations
        
        Returns:
            TrainingValidationResult with pass/fail status
        """
        # Validate config
        strategy_config_obj = validate_strategy_config(strategy_config)
        strategy = self.strategy_class(strategy_config_obj)
        
        # 1. Run backtest on training data
        # Don't pass commission_rate/slippage_ticks explicitly - let engine load from market profile
        # This ensures market profile settings (leverage, contract_size, etc.) are used correctly
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission_rate=None,  # Use market profile or strategy config
            slippage_ticks=None    # Use market profile or strategy config
        )
        backtest_result = engine.run(training_data)
        
        # Calculate enhanced metrics (profit_factor, sharpe_ratio, etc.)
        enhanced_metrics = calculate_enhanced_metrics(backtest_result)
        
        # 2. Check backtest quality criteria
        quality_checks = self._check_backtest_quality(backtest_result, enhanced_metrics)
        
        # 3. Assess strategy suitability for MC tests
        print("\n" + "="*60)
        print("STRATEGY SUITABILITY ASSESSMENT")
        print("="*60)
        profile = self.suitability_assessor.assess_strategy(backtest_result, n_bars=len(training_data))
        test_suitability = self.suitability_assessor.get_test_suitability(profile)

        suitability_payload = {
            test_name: {
                'suitable': bool(suit.suitable),
                'reason': suit.reason,
                'alternatives': list(suit.alternatives) if suit.alternatives else [],
                'priority': float(getattr(suit, 'priority', 0.0)),
                'category': str(getattr(suit, 'category', 'universal')),
            }
            for test_name, suit in test_suitability.items()
        }
        
        print(f"Strategy Type: {profile.strategy_type}")
        print(f"Return CV: {profile.return_cv:.4f}")
        print(f"Exit Uniformity: {profile.exit_uniformity:.1%}")
        print(f"Number of Trades: {profile.n_trades}")
        print(f"Number of Bars: {profile.n_bars}")
        print(f"Final Equity CV (from quick permutation): {profile.final_equity_cv:.6f}")
        print("\nTest Suitability:")
        for test_name, suit in test_suitability.items():
            status = "✅ SUITABLE" if suit.suitable else "❌ NOT SUITABLE"
            category = getattr(suit, 'category', 'universal')
            print(f"  {test_name.upper()} ({category.upper()}): {status}")
            print(f"    Reason: {suit.reason}")
            if suit.alternatives:
                print(f"    Alternatives: {', '.join(suit.alternatives)}")
        
        # 4. Run Monte Carlo Suite (conditionally based on suitability)
        print("\n" + "="*60)
        print("MONTE CARLO VALIDATION")
        print("="*60)
        mc_suite = MonteCarloSuite(seed=42)
        # Pass full OHLCV DataFrame (not just close prices) so randomized entry test
        # can properly simulate intrabar stop losses and take profits
        mc_results = mc_suite.run_conditional(
            backtest_result=backtest_result,
            price_series=training_data,  # Full DataFrame with OHLCV columns
            strategy=strategy,
            test_suitability=test_suitability,
            metrics=['final_pnl', 'sharpe_ratio', 'profit_factor'],
            n_iterations=mc_iterations,
            show_progress=True
        )
        
        mc_checks = self._check_monte_carlo_suite_conditional(mc_results, test_suitability)
        
        # 4. Run Parameter Sensitivity (if requested and params provided)
        sensitivity_results = None
        sensitivity_checks = {}
        if run_sensitivity and sensitivity_params:
            sensitivity_results = self._run_sensitivity(
                training_data,
                strategy_config,
                sensitivity_params
            )
            sensitivity_checks = self._check_sensitivity(sensitivity_results)
        
        # Combine all checks
        all_checks = {**quality_checks, **mc_checks, **sensitivity_checks}
        
        # REDEFINED: Pass/fail logic based on suitable tests
        # - If 3 suitable tests: require at least 2 to pass
        # - If 2 suitable tests: require at least 1 to pass (majority)
        mc_individual_tests_pass = all_checks.get('mc_individual_tests', False)
        mc_score_pass = all_checks.get('mc_score', False)
        mc_percentile_pass = all_checks.get('mc_percentile', False)
        quality_pass = all(quality_checks.values())
        
        # Pass if: quality checks pass AND (MC score/percentile pass) AND (suitable MC tests pass)
        passed = quality_pass and (mc_score_pass and mc_percentile_pass) and mc_individual_tests_pass
        
        # Collect failure reasons
        failure_reasons = []
        profit_factor = enhanced_metrics.get('profit_factor', 0.0)
        sharpe_ratio = enhanced_metrics.get('sharpe_ratio', 0.0)
        if not all_checks.get('min_profit_factor', True):
            failure_reasons.append(f"Training PF {profit_factor:.2f} < {self.criteria.training_min_profit_factor}")
        if not all_checks.get('min_sharpe', True):
            failure_reasons.append(f"Training Sharpe {sharpe_ratio:.2f} < {self.criteria.training_min_sharpe}")
        if not all_checks.get('min_trades', True):
            failure_reasons.append(f"Training trades {backtest_result.total_trades} < {self.criteria.training_min_trades}")
        if not all_checks.get('max_drawdown', True):
            failure_reasons.append(
                f"Training Max DD {backtest_result.max_drawdown:.2f}% > {self.criteria.training_max_drawdown_pct:.1f}%"
            )
        if not all_checks.get('mc_score', True):
            combined_score = mc_results.combined.get('score', 0.0)
            failure_reasons.append(f"Universal Monte Carlo robustness score {combined_score:.2f} < {self.criteria.min_mc_score}")
        if not all_checks.get('mc_percentile', True):
            combined_percentile = mc_results.combined.get('percentile', 0.0)
            failure_reasons.append(f"Universal Monte Carlo percentile {combined_percentile:.1f} < {self.criteria.min_mc_percentile}")
        if not all_checks.get('mc_individual_tests', True):
            # Get suitable test count for better error message
            suitable_count = len([
                name for name, suit in test_suitability.items()
                if suit.suitable and getattr(suit, 'category', 'universal') == 'universal'
            ])
            if suitable_count == 2:
                failure_reasons.append("Universal Monte Carlo individual tests: Required at least 1/2 suitable tests to pass (majority rule)")
            elif suitable_count == 3:
                failure_reasons.append("Universal Monte Carlo individual tests: Required at least 2/3 suitable tests to pass")
            else:
                failure_reasons.append("Universal Monte Carlo individual tests: Required suitable test(s) to pass")
        if sensitivity_checks and not all_checks.get('sensitivity_cv', True):
            failure_reasons.append("Parameter sensitivity too high (strategy not robust)")
        
        return TrainingValidationResult(
            passed=passed,
            backtest_result=backtest_result,
            monte_carlo_results={
                'permutation': mc_results.permutation,
                'bootstrap': mc_results.bootstrap,
                'randomized_entry': mc_results.randomized_entry,
                'combined': mc_results.combined,
                'suitability': suitability_payload,
                'skipped_tests': mc_results.skipped_tests,
            },
            sensitivity_results=sensitivity_results,
            criteria_checks=all_checks,
            failure_reasons=failure_reasons
        )
    
    def _check_backtest_quality(self, result: BacktestResult, enhanced_metrics: Dict[str, float]) -> Dict[str, bool]:
        """Check if backtest meets quality criteria."""
        profit_factor = enhanced_metrics.get('profit_factor', 0.0)
        sharpe_ratio = enhanced_metrics.get('sharpe_ratio', 0.0)
        return {
            'min_profit_factor': profit_factor >= self.criteria.training_min_profit_factor,
            'min_sharpe': sharpe_ratio >= self.criteria.training_min_sharpe,
            'min_trades': result.total_trades >= self.criteria.training_min_trades,
            'max_drawdown': result.max_drawdown <= self.criteria.training_max_drawdown_pct,
        }

    def _check_monte_carlo_suite_conditional(
        self,
        mc_results,
        test_suitability: Dict[str, Any]
    ) -> Dict[str, bool]:
        """Check Monte Carlo suite results accounting for skipped tests.

        Policy:
        - Only SUITABLE + UNIVERSAL tests contribute to pass/fail.
        - CONDITIONAL tests (e.g., randomized entry baseline) are reported but not gated.
        """
        checks: Dict[str, bool] = {}

        combined = mc_results.combined

        # Combined robustness score/percentile are computed from UNIVERSAL tests only
        checks['mc_score'] = combined.get('score', 0.0) >= self.criteria.min_mc_score
        checks['mc_percentile'] = combined.get('percentile', 0.0) >= self.criteria.min_mc_percentile

        suitable_universal_tests = [
            name for name, suit in test_suitability.items()
            if suit.suitable and getattr(suit, 'category', 'universal') == 'universal'
        ]

        def _passes_majority(p_values: Dict[str, float]) -> bool:
            if not p_values:
                return False
            passing = sum(1 for p in p_values.values() if p <= self.criteria.monte_carlo_p_value_max)
            return passing >= (len(p_values) / 2)

        perm_pass = False
        boot_pass = False
        rand_pass = False

        if 'permutation' in suitable_universal_tests and not mc_results.permutation.get('skipped', False):
            perm_pass = _passes_majority(mc_results.permutation.get('p_values', {}))

        if 'bootstrap' in suitable_universal_tests and not mc_results.bootstrap.get('skipped', False):
            boot_pass = _passes_majority(mc_results.bootstrap.get('p_values', {}))

        # Only relevant if randomized_entry is marked universal (e.g., legacy mode)
        if 'randomized_entry' in suitable_universal_tests and not mc_results.randomized_entry.get('skipped', False):
            rand_pass = _passes_majority(mc_results.randomized_entry.get('p_values', {}))

        n_suitable = len(suitable_universal_tests)
        if n_suitable == 0:
            checks['mc_individual_tests'] = False
        elif n_suitable == 1:
            if 'permutation' in suitable_universal_tests:
                checks['mc_individual_tests'] = perm_pass
            elif 'bootstrap' in suitable_universal_tests:
                checks['mc_individual_tests'] = boot_pass
            else:
                checks['mc_individual_tests'] = rand_pass
        elif n_suitable == 2:
            passes = 0
            if 'permutation' in suitable_universal_tests:
                passes += 1 if perm_pass else 0
            if 'bootstrap' in suitable_universal_tests:
                passes += 1 if boot_pass else 0
            if 'randomized_entry' in suitable_universal_tests:
                passes += 1 if rand_pass else 0
            checks['mc_individual_tests'] = passes >= 1
        else:
            checks['mc_individual_tests'] = sum([perm_pass, boot_pass, rand_pass]) >= 2

        return checks
    
    def _check_monte_carlo_suite(self, mc_results) -> Dict[str, bool]:
        """Legacy method for backward compatibility.
        
        This method is kept for compatibility but should use
        _check_monte_carlo_suite_conditional() instead.
        """
        # Create dummy suitability (all tests suitable) for legacy compatibility
        from validation.suitability import TestSuitability
        test_suitability = {
            'permutation': TestSuitability(suitable=True, reason="Legacy mode", priority=0.2),
            'bootstrap': TestSuitability(suitable=True, reason="Legacy mode", priority=0.3),
            'randomized_entry': TestSuitability(suitable=True, reason="Legacy mode", priority=0.5)
        }
        return self._check_monte_carlo_suite_conditional(mc_results, test_suitability)
    
    def _run_sensitivity(
        self,
        data: pd.DataFrame,
        base_config: Dict[str, Any],
        param_grid: Dict[str, List[Any]]
    ) -> Dict[str, Any]:
        """Run parameter sensitivity analysis."""
        analyzer = SensitivityAnalyzer(
            strategy_class=self.strategy_class,
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_ticks=self.slippage_ticks
        )
        
        results_df = analyzer.grid_search(
            data=data,
            base_config=base_config,
            param_grid=param_grid,
            metric='profit_factor'
        )
        
        # Analyze sensitivity for each parameter
        sensitivity_analysis = {}
        for param_name in param_grid.keys():
            if param_name in results_df.columns:
                sensitivity_analysis[param_name] = analyzer.analyze_sensitivity(
                    results_df,
                    param_name,
                    metric='profit_factor'
                )
        
        return {
            'results_df': results_df,
            'sensitivity_analysis': sensitivity_analysis
        }
    
    def _check_sensitivity(self, sensitivity_results: Dict[str, Any]) -> Dict[str, bool]:
        """Check parameter sensitivity results."""
        if not sensitivity_results:
            return {}
        
        checks = {}
        sensitivity_analysis = sensitivity_results.get('sensitivity_analysis', {})
        
        for param_name, analysis in sensitivity_analysis.items():
            cv = analysis.get('coefficient_of_variation', float('inf'))
            checks[f'sensitivity_{param_name}_cv'] = cv <= self.criteria.sensitivity_max_cv
        
        # Overall sensitivity check (all params must pass)
        checks['sensitivity_cv'] = all(
            analysis.get('coefficient_of_variation', float('inf')) <= self.criteria.sensitivity_max_cv
            for analysis in sensitivity_analysis.values()
        )
        
        return checks

