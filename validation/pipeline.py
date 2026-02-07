"""Validation Pipeline - Orchestrates phased validation workflow."""

from typing import Dict, Optional, List
import pandas as pd
from pathlib import Path

from strategies.base import StrategyBase
from validation.training_validator import TrainingValidator, TrainingValidationResult
from validation.oos_validator import OOSValidator, OOSValidationResult
from validation.stationarity import StationarityAnalyzer, StationarityResult
from validation.state import ValidationState, ValidationStateManager
from repro.dataset_manifest import build_manifest, write_manifest_json
from config.schema import (
    ValidationCriteriaConfig,
    WalkForwardConfig,
    load_validation_criteria
)


class ValidationPipeline:
    """Orchestrates the complete validation workflow.
    
    Enforces the methodology:
    1. Phase 1: Training Validation (training data only)
    2. Phase 2: OOS Validation (OOS data, only if Phase 1 passes)
    3. Phase 3: Stationarity (post-live, determines retrain frequency)
    """
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: Optional[float] = None,
        slippage_ticks: Optional[float] = None,
        criteria_config: Optional[ValidationCriteriaConfig] = None,
        state_dir: Optional[Path] = None,
        manifest_dir: Optional[Path] = None
    ):
        """
        Initialize validation pipeline.
        
        Args:
            strategy_class: Strategy class to validate
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_ticks: Slippage in ticks
            criteria_config: Validation criteria (uses defaults if None)
            state_dir: Directory for validation state files
        """
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.criteria = criteria_config or load_validation_criteria()
        self.state_manager = ValidationStateManager(state_dir)
        self.manifest_dir = manifest_dir
        
        # Initialize validators
        self.training_validator = TrainingValidator(
            strategy_class=strategy_class,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_ticks=slippage_ticks,
            criteria=self.criteria.training
        )
        
        self.oos_validator = OOSValidator(
            strategy_class=strategy_class,
            initial_capital=initial_capital,
            commission_rate=commission_rate,
            slippage_ticks=slippage_ticks,
            criteria=self.criteria.oos
        )
    
    def run_phase1_training(
        self,
        strategy_name: str,
        data_file: str,
        training_data: pd.DataFrame,
        strategy_config: Dict,
        training_start: Optional[str] = None,
        training_end: Optional[str] = None,
        run_sensitivity: bool = True,
        sensitivity_params: Optional[Dict[str, List]] = None,
        mc_iterations: int = 1000
    ) -> TrainingValidationResult:
        """
        Run Phase 1: Training Validation.
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
            training_data: Training dataset
            strategy_config: Strategy configuration
            training_start: Start date of training period
            training_end: End date of training period
            run_sensitivity: Whether to run sensitivity analysis
            sensitivity_params: Parameters for sensitivity analysis
            mc_iterations: Monte Carlo iterations
        
        Returns:
            TrainingValidationResult
        """
        # Load or create state
        state = self.state_manager.create_or_load_state(
            strategy_name=strategy_name,
            data_file=data_file,
            training_start=training_start,
            training_end=training_end
        )

        # Phase lock: prevent silent dataset changes mid-phase.
        try:
            phase1_manifest = build_manifest(
                file_path=Path(data_file),
                df=training_data,
                purpose='phase1',
                slice_start=training_start,
                slice_end=training_end,
            )

            if self.manifest_dir is not None:
                try:
                    manifest_path = write_manifest_json(phase1_manifest, self.manifest_dir)
                    print(f"🧾 Wrote dataset manifest: {manifest_path}")
                except Exception:
                    pass

            state.verify_or_set_phase_lock(
                'phase1',
                {
                    'manifest_hash': phase1_manifest.manifest_hash(),
                    'identity': phase1_manifest.identity_dict(),
                },
            )
            self.state_manager.save_state(state)
        except FileNotFoundError:
            # Some tests may pass a non-file data_file. Skip locking in that case.
            pass
        
        # Check if already completed
        # Best Practice: Allow re-running Phase 1 if it failed (reoptimization is part of development)
        # But prevent re-running if it passed (to avoid data mining)
        if state.phase1_training_completed:
            if state.phase1_passed:
                # Phase 1 passed - don't allow re-running (move to Phase 2)
                print(f"⚠️  Phase 1 already completed and PASSED (completed: {state.phase1_completed_at})")
                print("Phase 1 cannot be re-run once it passes. Proceed to Phase 2 (OOS Validation).")
                if state.phase1_results:
                    # Reconstruct result from state (simplified)
                    print("Using cached results.")
                    return TrainingValidationResult(
                        passed=state.phase1_passed,
                        backtest_result=None,  # Not stored in state
                        monte_carlo_results=state.phase1_results.get('monte_carlo', {}),
                        sensitivity_results=state.phase1_results.get('sensitivity'),
                        criteria_checks=state.phase1_results.get('criteria_checks', {}),
                        failure_reasons=state.phase1_results.get('failure_reasons', [])
                    )
            else:
                # Phase 1 failed - check if we can retry
                can_retry, reason = state.can_retry_phase1()
                retry_info = state.get_phase1_retry_info()
                
                if not can_retry:
                    print(f"❌ Phase 1 retry blocked: {reason}")
                    raise ValueError(f"Cannot retry Phase 1: {reason}")
                
                # Show retry information
                print(f"⚠️  Phase 1 previously completed but FAILED (completed: {state.phase1_completed_at})")
                print(f"   Failure count: {retry_info['failure_count']}/3")
                print(f"   Remaining attempts: {retry_info['remaining_attempts']}")
                print("Re-running Phase 1 validation (reoptimization is part of development process)...")
                print("Previous failure reasons:")
                if retry_info['previous_failure_reasons']:
                    for reason in retry_info['previous_failure_reasons']:
                        print(f"  - {reason}")
                else:
                    print("  - (No previous failure reasons recorded)")
                print()
                
                # Save state in case failure count was reset (after 30 days)
                self.state_manager.save_state(state)
                
                # Reset Phase 1 state to allow re-running (but keep failure count)
                state.phase1_training_completed = False
                state.phase1_passed = False
                state.phase1_completed_at = None
                # Keep phase1_results for display, but we'll update it after new validation
        
        # Run validation
        print("="*60)
        print("PHASE 1: TRAINING VALIDATION")
        print("="*60)
        print(f"Date range: {training_data.index[0].date()} to {training_data.index[-1].date()}")
        print("Testing on training data only...")
        
        result = self.training_validator.validate(
            training_data=training_data,
            strategy_config=strategy_config,
            run_sensitivity=run_sensitivity,
            sensitivity_params=sensitivity_params,
            mc_iterations=mc_iterations
        )
        
        # Save state
        state.mark_phase1_complete(
            passed=result.passed,
            results={
                'monte_carlo': result.monte_carlo_results,
                'sensitivity': result.sensitivity_results,
                'criteria_checks': result.criteria_checks,
                'failure_reasons': result.failure_reasons
            }
        )
        self.state_manager.save_state(state)
        
        # Print results
        self._print_phase1_results(result)
        
        return result
    
    def run_phase2_oos(
        self,
        strategy_name: str,
        data_file: str,
        oos_data: pd.DataFrame,
        strategy_config: Dict,
        wf_config: WalkForwardConfig,
        oos_start: Optional[str] = None,
        oos_end: Optional[str] = None,
        full_data: Optional[pd.DataFrame] = None
    ) -> OOSValidationResult:
        """
        Run Phase 2: Out-of-Sample Validation.
        
        WARNING: OOS data should only be used ONCE. This method enforces this.
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
            oos_data: Out-of-sample dataset (for validation boundaries)
            strategy_config: Strategy configuration
            wf_config: Walk-forward configuration
            oos_start: Start date of OOS period
            oos_end: End date of OOS period
            full_data: Full dataset (optional, for warm-up periods). If None, uses oos_data.
        
        Returns:
            OOSValidationResult
        
        Raises:
            ValueError: If Phase 1 hasn't passed or OOS data already used
        """
        # Load state
        state = self.state_manager.load_state(strategy_name, data_file)
        
        if state is None:
            raise ValueError(
                "Phase 1 (Training Validation) must be completed first!\n"
                "Run Phase 1 validation before using OOS data."
            )
        
        if not state.phase1_passed:
            raise ValueError(
                f"Phase 1 validation FAILED. Cannot proceed to Phase 2.\n"
                f"Failure reasons: {', '.join(state.phase1_results.get('failure_reasons', []))}"
            )
        
        if not state.can_use_oos_data():
            raise ValueError(
                "⚠️  OOS data has already been used!\n"
                "Out-of-sample data should only be used ONCE for validation.\n"
                f"OOS validation was completed on: {state.phase2_completed_at}\n"
                "To test on new OOS data, use a different data file or strategy."
            )
        
        # Update OOS dates
        if oos_start:
            state.oos_start_date = oos_start
        if oos_end:
            state.oos_end_date = oos_end

        # OOS consumed on attempt (not on success)
        state.mark_phase2_consumed()

        # Phase lock: bind Phase 2 to this dataset slice.
        try:
            phase2_manifest = build_manifest(
                file_path=Path(data_file),
                df=oos_data,
                purpose='phase2',
                slice_start=oos_start,
                slice_end=oos_end,
            )

            if self.manifest_dir is not None:
                try:
                    manifest_path = write_manifest_json(phase2_manifest, self.manifest_dir)
                    print(f"🧾 Wrote dataset manifest: {manifest_path}")
                except Exception:
                    pass

            state.verify_or_set_phase_lock(
                'phase2',
                {
                    'manifest_hash': phase2_manifest.manifest_hash(),
                    'manifest': phase2_manifest.to_dict(),
                },
            )
        except FileNotFoundError:
            pass

        # Persist consumption + lock before running anything expensive.
        self.state_manager.save_state(state)
        
        # Run validation
        print("="*60)
        print("PHASE 2: OUT-OF-SAMPLE VALIDATION")
        print("="*60)
        print("⚠️  WARNING: OOS data will be marked as used after this test!")
        print("Testing on out-of-sample data...")
        
        # Use full_data if provided (for warm-up periods), otherwise use oos_data.
        # Bound the accessible window to train-start -> oos_end to prevent holdout leakage.
        data_for_wf = full_data if full_data is not None else oos_data

        try:
            if state.training_start_date:
                train_start_dt = pd.to_datetime(state.training_start_date).normalize()
                data_for_wf = data_for_wf[data_for_wf.index >= train_start_dt]
        except Exception:
            pass

        try:
            if state.oos_end_date:
                oos_end_dt = pd.to_datetime(state.oos_end_date).normalize() + pd.Timedelta(days=1)
                data_for_wf = data_for_wf[data_for_wf.index < oos_end_dt]
        except Exception:
            pass
        
        result = self.oos_validator.validate(
            oos_data=oos_data,  # Still pass oos_data for validation boundaries
            strategy_config=strategy_config,
            wf_config=wf_config,
            full_data=data_for_wf  # Pass full data for warm-up
        )
        
        # Save state (marks OOS as used)
        state.mark_phase2_complete(
            passed=result.passed,
            results={
                'walk_forward_summary': result.walk_forward_result.summary,
                'criteria_checks': result.criteria_checks,
                'failure_reasons': result.failure_reasons
            }
        )
        self.state_manager.save_state(state)
        
        # Print results
        self._print_phase2_results(result)
        
        if result.passed:
            print("\n✅ Strategy QUALIFIED for live trading!")
        else:
            print("\n❌ Strategy did NOT qualify for live trading.")
        
        return result
    
    def run_phase3_stationarity(
        self,
        strategy_name: str,
        data_file: str,
        live_data: pd.DataFrame,
        strategy_config: Dict,
        max_days: int = 30,
        step_days: int = 1,
        training_period_days: int = 365
    ) -> StationarityResult:
        """
        Run Phase 3: Stationarity Analysis (post-live).
        
        Determines optimal retraining frequency to combat edge erosion.
        
        Args:
            strategy_name: Name of strategy
            data_file: Path to data file
            live_data: Live trading data
            strategy_config: Strategy configuration
            max_days: Maximum days to test
            step_days: Step size for days
            training_period_days: Training period before OOS testing
        
        Returns:
            StationarityResult
        """
        # Load state
        state = self.state_manager.load_state(strategy_name, data_file)
        
        if state is None or not state.is_qualified_for_live():
            raise ValueError(
                "Strategy must pass Phase 1 and Phase 2 before running stationarity analysis."
            )
        
        # Run analysis
        print("="*60)
        print("PHASE 3: STATIONARITY ANALYSIS")
        print("="*60)
        print("Determining optimal retraining frequency...")
        
        from config.schema import validate_strategy_config
        strategy_config_obj = validate_strategy_config(strategy_config)
        
        # Pass stationarity criteria to analyzer
        analyzer = StationarityAnalyzer(
            strategy_class=self.strategy_class,
            initial_capital=self.initial_capital,
            commission_rate=self.commission_rate,
            slippage_ticks=self.slippage_ticks,
            criteria=self.criteria.stationarity if self.criteria else None
        )
        
        result = analyzer.analyze_days_vs_performance(
            data=live_data,
            strategy_config=strategy_config_obj,
            max_days=max_days,
            step_days=step_days,
            training_period_days=training_period_days
        )
        
        # Save state
        state.mark_phase3_complete(
            passed=True,  # Stationarity is informational, not pass/fail
            results={
                'recommended_retrain_days': result.recommended_retrain_days,
                'days_vs_pf': result.days_vs_pf,
                'days_vs_sharpe': result.days_vs_sharpe
            }
        )
        self.state_manager.save_state(state)
        
        # Print results
        print(f"\nRecommended retraining frequency: {result.recommended_retrain_days} days")
        print(f"Analysis periods: {len(result.analysis_periods)}")
        
        return result
    
    def _print_phase1_results(self, result: TrainingValidationResult):
        """Print Phase 1 validation results."""
        print("\n" + "="*60)
        print("PHASE 1 RESULTS")
        print("="*60)
        
        if result.backtest_result:
            # Calculate enhanced metrics (profit_factor, sharpe_ratio, etc.)
            from metrics.metrics import calculate_enhanced_metrics
            enhanced_metrics = calculate_enhanced_metrics(result.backtest_result)
            
            print(f"\nTraining Backtest:")
            print(f"  Profit Factor: {enhanced_metrics.get('profit_factor', 0):.2f}")
            print(f"  Sharpe Ratio: {enhanced_metrics.get('sharpe_ratio', 0):.2f}")
            print(f"  Total Trades: {result.backtest_result.total_trades}")
        
        print(f"\nMonte Carlo Suite Results:")
        
        # Check if new format (MonteCarloSuite) or old format (MonteCarloPermutation)
        if 'permutation' in result.monte_carlo_results:
            # New format: MonteCarloSuite results
            mc = result.monte_carlo_results
            
            # Permutation results
            if 'permutation' in mc:
                perm = mc['permutation']
                if perm.get('skipped', False):
                    print(f"\n  1. PERMUTATION TEST (Order Independence)")
                    print(f"     Question: Does trade order matter?")
                    print(f"     Status: ❌ SKIPPED")
                    print(f"     Reason: {perm.get('reason', 'Not assessed')}")
                    if perm.get('alternatives'):
                        print(f"     Alternatives: {', '.join(perm.get('alternatives', []))}")
                else:
                    print(f"\n  1. PERMUTATION TEST (Order Independence)")
                    print(f"     Question: Does trade order matter?")
                    print(f"     Status: ✅ COMPLETED")
                    if 'p_values' in perm and 'percentiles' in perm:
                        # Get average percentile for interpretation
                        percentiles = perm.get('percentiles', {})
                        avg_percentile = sum(percentiles.values()) / len(percentiles) if percentiles else 0.0
                        p_values = perm.get('p_values', {})
                        avg_p_value = sum(p_values.values()) / len(p_values) if p_values else 1.0
                        
                        for metric in perm.get('p_values', {}).keys():
                            p_val = perm['p_values'].get(metric, 1.0)
                            percentile = perm['percentiles'].get(metric, 0.0)
                            status = "✓" if p_val <= 0.05 else "✗"
                            print(f"       {metric}: p-value={p_val:.4f}, percentile={percentile:.1f}% {status}")
                        
                        # Interpretation
                        if avg_percentile > 95:
                            print(f"     Interpretation: Strategy's sequence is significantly better than random ordering.")
                            print(f"     Action: Compounding effects or trade sequencing is contributing to edge.")
                        elif avg_percentile < 5:
                            print(f"     Interpretation: Strategy's sequence is WORSE than random ordering.")
                            print(f"     Action: Investigate trade sequencing and risk management.")
                        elif avg_p_value <= 0.05:
                            print(f"     Interpretation: Strategy's sequence is better than random (statistically significant).")
                        else:
                            print(f"     Interpretation: Strategy's performance is order-independent.")
                            print(f"     Action: Returns are too small relative to equity for compounding to matter.")
            
            # Bootstrap results
            if 'bootstrap' in mc:
                boot = mc['bootstrap']
                if boot.get('skipped', False):
                    print(f"\n  2. BLOCK BOOTSTRAP TEST (Market Structure Robustness)")
                    print(f"     Question: Does strategy work under resampled market conditions?")
                    print(f"     Status: ❌ SKIPPED")
                    print(f"     Reason: {boot.get('reason', 'Not assessed')}")
                    if boot.get('alternatives'):
                        print(f"     Alternatives: {', '.join(boot.get('alternatives', []))}")
                else:
                    print(f"\n  2. BLOCK BOOTSTRAP TEST (Market Structure Robustness)")
                    print(f"     Question: Does strategy work under resampled market conditions?")
                    print(f"     Status: ✅ COMPLETED")
                    if 'p_values' in boot and 'percentiles' in boot:
                        percentiles = boot.get('percentiles', {})
                        avg_percentile = sum(percentiles.values()) / len(percentiles) if percentiles else 0.0
                        p_values = boot.get('p_values', {})
                        avg_p_value = sum(p_values.values()) / len(p_values) if p_values else 1.0
                        
                        for metric in boot.get('p_values', {}).keys():
                            p_val = boot['p_values'].get(metric, 1.0)
                            percentile = boot['percentiles'].get(metric, 0.0)
                            status = "✓" if p_val <= 0.05 else "✗"
                            print(f"       {metric}: p-value={p_val:.4f}, percentile={percentile:.1f}% {status}")
                        
                        # Interpretation
                        if avg_percentile > 95:
                            print(f"     Interpretation: Strategy performs well even when market structure is resampled.")
                            print(f"     Action: Edge is robust to market conditions. Exits and risk management work.")
                        elif avg_percentile < 5:
                            print(f"     Interpretation: Strategy performs poorly under resampled conditions.")
                            print(f"     Action: Strategy may be overfitted to specific market patterns.")
                        elif avg_p_value <= 0.05:
                            print(f"     Interpretation: Strategy significantly outperforms resampled markets.")
                        else:
                            print(f"     Interpretation: Strategy's performance is similar to resampled markets.")
                            print(f"     Action: Edge may be marginal or non-existent.")
            
            # Randomized Entry results (always runs)
            if 'randomized_entry' in mc:
                rand = mc['randomized_entry']
                print(f"\n  3. RANDOMIZED ENTRY TEST (Entry Contribution)")
                print(f"     Question: Does entry timing/location contribute to edge?")
                print(f"     Status: ✅ COMPLETED (Conditional Test)")
                if 'p_values' in rand and 'percentiles' in rand:
                    percentiles = rand.get('percentiles', {})
                    avg_percentile = sum(percentiles.values()) / len(percentiles) if percentiles else 0.0
                    p_values = rand.get('p_values', {})
                    avg_p_value = sum(p_values.values()) / len(p_values) if p_values else 1.0
                    
                    for metric in rand.get('p_values', {}).keys():
                        p_val = rand['p_values'].get(metric, 1.0)
                        percentile = rand['percentiles'].get(metric, 0.0)
                        status = "✓" if p_val <= 0.05 else "✗"
                        print(f"       {metric}: p-value={p_val:.4f}, percentile={percentile:.1f}% {status}")
                    
                    # Interpretation (most important test)
                    if avg_percentile > 95:
                        print(f"     Interpretation: Strategy significantly outperforms random entries.")
                        print(f"     Action: Entry timing/location IS contributing to edge.")
                        print(f"     ⚠️  Warning: Strategy may be fragile to execution errors.")
                    elif avg_percentile < 5:
                        print(f"     Interpretation: Strategy performs WORSE than random entries.")
                        print(f"     ⚠️  Critical: Entry logic is HURTING performance.")
                        print(f"     Action: Consider:")
                        print(f"       - Fixing entry signals/logic")
                        print(f"       - Using random entries (if exits/risk management are the edge)")
                        print(f"       - Investigating why entry selection is suboptimal")
                    elif 45 <= avg_percentile <= 55:
                        print(f"     Interpretation: Strategy performs similarly to random entries.")
                        print(f"     ✅ Excellent: Entry has little/no edge, but strategy is still profitable.")
                        print(f"     Action: Edge is in exits, risk management, trade structure (robust).")
                    elif avg_p_value <= 0.05:
                        print(f"     Interpretation: Strategy significantly outperforms random entries.")
                    else:
                        print(f"     Interpretation: Strategy's performance is similar to random entries.")
                        print(f"     Action: Entry contribution is marginal. Edge is primarily structural.")
                
                # Display diagnostic information if available
                if 'randomized_entry_diagnostics' in rand and rand['randomized_entry_diagnostics']:
                    diag = rand['randomized_entry_diagnostics']
                    print(f"\n     Diagnostics:")
                    print(f"       Randomness Effective: {'✅ YES' if diag.get('randomness_effective') else '❌ NO'}")
                    print(f"       Unique Entry Bars: {diag.get('unique_entry_bars', 0)}")
                    print(f"       Unique Entry Prices: {diag.get('unique_entry_prices', 0)}")
                    print(f"       Classification: {diag.get('classification', 'UNKNOWN')}")
                
                # Show distribution statistics to clarify p-value = 1.0 (only when percentile is very low)
                if avg_percentile < 5 and 'random_distributions' in rand and 'observed_metrics' in rand:
                    import numpy as np
                    print(f"\n     Distribution Analysis (Why p-value = 1.0):")
                    print(f"       ⚠️  Key Insight: Variance in entry bars/prices ≠ Variance in outcomes")
                    print(f"       ⚠️  Randomization is working (26 unique entry bars/prices)")
                    print(f"       ⚠️  But all random entries performed better than original strategy")
                    for metric in ['final_pnl', 'sharpe_ratio', 'profit_factor']:
                        if metric in rand.get('random_distributions', {}) and metric in rand.get('observed_metrics', {}):
                            random_vals = rand['random_distributions'][metric]
                            observed = rand['observed_metrics'][metric]
                            if len(random_vals) > 0:
                                finite_vals = random_vals[np.isfinite(random_vals)]
                                if len(finite_vals) > 0:
                                    min_random = np.min(finite_vals)
                                    max_random = np.max(finite_vals)
                                    mean_random = np.mean(finite_vals)
                                    n_better = np.sum(finite_vals >= observed)
                                    n_total = len(finite_vals)
                                    print(f"       {metric}:")
                                    print(f"         Observed: {observed:.4f}")
                                    print(f"         Random: min={min_random:.4f}, max={max_random:.4f}, mean={mean_random:.4f}")
                                    print(f"         Random entries ≥ observed: {n_better}/{n_total} ({100.0*n_better/n_total:.1f}%)")
                                    if n_better == n_total:
                                        print(f"         → All random entries performed better - entry logic is hurting performance")
            
            # Combined results (with warning about interpretation)
            if 'combined' in mc:
                combined = mc['combined']
                print(f"\n  COMBINED ROBUSTNESS SCORE (UNIVERSAL - Pass/Fail Only):")
                print(f"     ⚠️  WARNING: Tests measure different things. This score masks critical insights.")
                print(f"     ⚠️  Always interpret individual test results above, not this combined score.")
                print(f"     ⚠️  Use UNIVERSAL combined score ONLY for automated pass/fail threshold.")
                score = combined.get('score', 0.0)
                percentile = combined.get('percentile', 0.0)
                p_value = combined.get('p_value', 1.0)
                robust = combined.get('robust', False)
                n_suitable = combined.get('n_suitable_tests', 3)
                status = "✓ ROBUST" if robust else "✗ NOT ROBUST"
                print(f"     Score: {score:.3f}, Percentile: {percentile:.1f}%, P-value: {p_value:.4f} {status}")
                print(f"     (Based on {n_suitable} suitable test(s))")

                # Optional: show informational combined score over all suitable tests
                if 'score_all' in combined or 'percentile_all' in combined:
                    score_all = combined.get('score_all', score)
                    percentile_all = combined.get('percentile_all', percentile)
                    p_value_all = combined.get('p_value_all', p_value)
                    robust_all = combined.get('robust_all', robust)
                    n_all = combined.get('n_suitable_tests_all', combined.get('n_suitable_tests', 3))
                    status_all = "✓ ROBUST" if robust_all else "✗ NOT ROBUST"
                    print(f"\n  COMBINED ROBUSTNESS SCORE (ALL - Informational):")
                    print(f"     Score: {score_all:.3f}, Percentile: {percentile_all:.1f}%, P-value: {p_value_all:.4f} {status_all}")
                    print(f"     (Based on {n_all} suitable test(s))")
                
                # Show breakdown of what's being combined
                print(f"\n     Score Breakdown:")
                if 'permutation' in mc and not mc['permutation'].get('skipped', False):
                    perm_score = combined.get('permutation_score', 0.0)
                    perm_weight = combined.get('normalized_weights', {}).get('permutation', 0.0)
                    print(f"       Permutation: {perm_score:.3f} (weight: {perm_weight:.1%})")
                if 'bootstrap' in mc and not mc['bootstrap'].get('skipped', False):
                    boot_score = combined.get('bootstrap_score', 0.0)
                    boot_weight = combined.get('normalized_weights', {}).get('bootstrap', 0.0)
                    print(f"       Bootstrap: {boot_score:.3f} (weight: {boot_weight:.1%})")
                if 'randomized_entry' in mc and not mc['randomized_entry'].get('skipped', False):
                    rand_score = combined.get('randomized_entry_score', 0.0)
                    rand_weight = combined.get('normalized_weights', {}).get('randomized_entry', 0.0)
                    if rand_weight > 0:
                        print(f"       Randomized Entry: {rand_score:.3f} (weight: {rand_weight:.1%})")
                    else:
                        print(f"       Randomized Entry: {rand_score:.3f} (CONDITIONAL - not used for pass/fail)")
                
                # Show metrics used
                print(f"\n     Metrics Combined:")
                print(f"       - final_pnl, sharpe_ratio, profit_factor")
                print(f"       Formula: metric_score = 0.5*(1-p_value) + 0.5*(percentile/100)")
                print(f"       Test score = average(metric_scores)")
                print(f"       Combined = weighted_average(test_scores)")
                
                if combined.get('warnings'):
                    for warning in combined['warnings']:
                        print(f"     ⚠️  {warning}")
                
                # Generate overall assessment summary
                print(f"\n  OVERALL ASSESSMENT:")
                strengths = []
                weaknesses = []
                recommendations = []
                
                # Analyze each test
                if 'bootstrap' in mc and not mc['bootstrap'].get('skipped', False):
                    boot_percentiles = mc['bootstrap'].get('percentiles', {})
                    if boot_percentiles:
                        avg_boot = sum(boot_percentiles.values()) / len(boot_percentiles)
                        if avg_boot > 90:
                            strengths.append("Exits and risk management work (Bootstrap: high percentile)")
                        elif avg_boot < 10:
                            weaknesses.append("Strategy may be overfitted to specific market patterns (Bootstrap: low percentile)")
                
                if 'randomized_entry' in mc:
                    rand_percentiles = mc['randomized_entry'].get('percentiles', {})
                    if rand_percentiles:
                        avg_rand = sum(rand_percentiles.values()) / len(rand_percentiles)
                        if avg_rand > 95:
                            strengths.append("Entry timing/location contributes to edge (Randomized Entry: high percentile)")
                            recommendations.append("Monitor execution quality - strategy may be fragile to entry errors")
                        elif avg_rand < 5:
                            weaknesses.append("Entry logic is hurting performance (Randomized Entry: low percentile)")
                            recommendations.append("Investigate entry signals - consider fixing or using random entries")
                        elif 45 <= avg_rand <= 55:
                            strengths.append("Entry is irrelevant - edge is structural (Randomized Entry: ~50%)")
                            strengths.append("Strategy is robust and not dependent on perfect entry execution")
                
                if 'permutation' in mc and not mc['permutation'].get('skipped', False):
                    perm_percentiles = mc['permutation'].get('percentiles', {})
                    if perm_percentiles:
                        avg_perm = sum(perm_percentiles.values()) / len(perm_percentiles)
                        if avg_perm > 90:
                            strengths.append("Trade sequencing contributes to edge (Permutation: high percentile)")
                        elif avg_perm < 10:
                            weaknesses.append("Trade sequencing is hurting performance (Permutation: low percentile)")
                
                if strengths:
                    print(f"     ✅ Strengths:")
                    for strength in strengths:
                        print(f"        - {strength}")
                
                if weaknesses:
                    print(f"     ⚠️  Weaknesses:")
                    for weakness in weaknesses:
                        print(f"        - {weakness}")
                
                if recommendations:
                    print(f"     🎯 Recommendations:")
                    for i, rec in enumerate(recommendations, 1):
                        print(f"        {i}. {rec}")
                
                if not strengths and not weaknesses:
                    print(f"     (Review individual test results above for detailed interpretation)")
        else:
            # Old format: MonteCarloPermutation results
            print(f"\n  Permutation Test (Legacy):")
            for metric, data in result.monte_carlo_results.items():
                p_value = data.get('p_value', 1.0)
                percentile = data.get('percentile_rank', 0.0)
                status = "✓" if p_value <= 0.05 else "✗"
                print(f"    {metric}: p-value={p_value:.4f}, percentile={percentile:.1f}% {status}")
        
        if result.sensitivity_results:
            print(f"\nParameter Sensitivity:")
            for param, analysis in result.sensitivity_results.get('sensitivity_analysis', {}).items():
                cv = analysis.get('coefficient_of_variation', 0)
                print(f"  {param}: CV={cv:.3f}")
        
        print(f"\nCriteria Checks:")
        for check, passed in result.criteria_checks.items():
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"  {check}: {status}")
        
        if result.passed:
            print("\n✅ Phase 1 PASSED - Strategy has edge on training data")
        else:
            print("\n❌ Phase 1 FAILED")
            print("Failure reasons:")
            for reason in result.failure_reasons:
                print(f"  - {reason}")
    
    def _print_phase2_results(self, result: OOSValidationResult):
        """Print Phase 2 validation results."""
        print("\n" + "="*60)
        print("PHASE 2 RESULTS")
        print("="*60)
        
        wf = result.walk_forward_result
        summary = wf.summary
        
        print(f"\nWalk-Forward Analysis:")
        print(f"  Total Steps: {summary.get('total_steps', 0)}")
        
        # Handle None values (when all periods are excluded)
        mean_pf = summary.get('mean_test_pf')
        if mean_pf is not None:
            print(f"  Mean Test PF: {mean_pf:.2f}")
        else:
            print(f"  Mean Test PF: N/A (all periods excluded)")
        
        std_pf = summary.get('std_test_pf')
        if std_pf is not None:
            print(f"  Std Test PF: {std_pf:.2f}")
        else:
            print(f"  Std Test PF: N/A (all periods excluded)")
        
        consistency = summary.get('consistency_score')
        if consistency is not None:
            print(f"  Consistency Score: {consistency:.2f}")
        else:
            print(f"  Consistency Score: N/A (all periods excluded)")
        
        print(f"\nCriteria Checks:")
        # Filter out metadata and calculated values
        for check, value in result.criteria_checks.items():
            # Skip metadata
            if check == '_metadata':
                continue
            # Skip calculated values (not boolean checks)
            if check in ['oos_consistency_value', 'binomial_p_value']:
                continue
            # Only show boolean checks
            if isinstance(value, bool):
                status = "✓ PASS" if value else "✗ FAIL"
                print(f"  {check}: {status}")
        
        if result.passed:
            print("\n✅ Phase 2 PASSED - Strategy works on unseen data")
        else:
            print("\n❌ Phase 2 FAILED")
            print("Failure reasons:")
            for reason in result.failure_reasons:
                print(f"  - {reason}")

