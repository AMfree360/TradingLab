"""Master Monte Carlo Suite - Orchestrates all MC validation engines.

This module provides a unified interface to run all three Monte Carlo engines:
1. Permutation Test
2. Block Bootstrap
3. Randomized Entry Baseline

It combines results and computes a robustness score.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd

from .permutation import MonteCarloPermutation, PermutationResult
from .block_bootstrap import MonteCarloBlockBootstrap, BootstrapResult
from .randomized_entry import MonteCarloRandomizedEntry, RandomizedEntryResult, RandomizedEntryConfig
from .utils import normalize_to_z_scores, normalize_to_ranks
from engine.backtest_engine import BacktestResult
from strategies.base import StrategyBase


@dataclass
class MonteCarloSuiteResult:
    """Complete results from Monte Carlo suite."""
    permutation: Dict[str, Any]
    bootstrap: Dict[str, Any]
    randomized_entry: Dict[str, Any]
    combined: Dict[str, Any]
    skipped_tests: Optional[Dict[str, Any]] = None  # Tests that were skipped with reasons


class MonteCarloSuite:
    """Master orchestrator for all Monte Carlo validation engines.
    
    This class runs all three independent Monte Carlo tests and combines
    their results into a unified robustness assessment.
    
    The combined score is a weighted average of all tests, providing a
    comprehensive answer to: "Is this strategy better than randomness?"
    """
    
    def __init__(
        self,
        seed: int = 42,
        weights: Optional[Dict[str, float]] = None,
        randomized_entry_config: Optional[RandomizedEntryConfig] = None
    ):
        """
        Initialize Monte Carlo suite.
        
        Args:
            seed: Random seed for reproducibility
            weights: Weights for each test in combined score (default: equal weights)
                     Keys: 'permutation', 'bootstrap', 'randomized_entry'
            randomized_entry_config: Configuration for randomized entry test (uses defaults if None)
        """
        self.seed = seed
        self.permutation_engine = MonteCarloPermutation(seed=seed)
        self.bootstrap_engine = MonteCarloBlockBootstrap(seed=seed)
        self.randomized_entry_engine = MonteCarloRandomizedEntry(seed=seed, config=randomized_entry_config)
        
        # Default weights per specification:
        # random-entry: 50%, bootstrap: 30%, permutation: 20%
        if weights is None:
            weights = {
                'permutation': 0.20,
                'bootstrap': 0.30,
                'randomized_entry': 0.50
            }
        self.weights = weights
    
    def _normalize_percentile(self, percentile: float) -> float:
        """Normalize percentile to 0-1 scale for scoring."""
        # Percentile is 0-100, normalize to 0-1
        return percentile / 100.0
    
    def _normalize_p_value(self, p_value: float) -> float:
        """Normalize p-value to 0-1 scale for scoring (inverted)."""
        # Lower p-value = better, so invert: 1 - p_value
        # But cap at reasonable range
        return max(0.0, min(1.0, 1.0 - p_value))
    
    def _calculate_metric_score(
        self,
        percentile: float,
        p_value: float
    ) -> float:
        """Calculate metric score per specification.
        
        Formula: metric_score = 0.5*(1 - p_value) + 0.5*(percentile/100)
        
        Args:
            percentile: Percentile (0-100)
            p_value: P-value (0-1)
        
        Returns:
            Metric score (0-1)
        """
        # Per specification: metric_score = 0.5*(1 - p_value) + 0.5*(percentile/100)
        score = 0.5 * (1.0 - p_value) + 0.5 * (percentile / 100.0)
        return float(max(0.0, min(1.0, score)))  # Clamp to [0, 1]
    
    def _calculate_test_score(
        self,
        percentiles: Dict[str, float],
        p_values: Dict[str, float],
        metrics: List[str]
    ) -> float:
        """Calculate a single test's score by averaging metric scores.
        
        Per specification: metric_score = 0.5*(1 - p_value) + 0.5*(percentile/100)
        Test score = average of metric scores across all metrics.
        """
        if not metrics:
            return 0.0
        
        # Calculate metric score for each metric, then average
        metric_scores = []
        for metric in metrics:
            percentile = percentiles.get(metric, 0.0)
            p_value = p_values.get(metric, 1.0)
            metric_score = self._calculate_metric_score(percentile, p_value)
            metric_scores.append(metric_score)
        
        # Average across metrics
        return float(np.mean(metric_scores))
    
    def _normalize_percentiles_for_combination(
        self,
        percentiles: Dict[str, float],
        metrics: List[str]
    ) -> float:
        """Normalize percentiles to 0-1 scale for combination.
        
        Since different tests have different null hypotheses,
        we normalize percentiles before combining to make them comparable.
        """
        if not metrics:
            return 0.0
        
        # Get percentiles for all metrics
        values = [percentiles.get(m, 0.0) for m in metrics]
        
        # Normalize to 0-1 (percentiles are already 0-100)
        normalized = np.array(values) / 100.0
        
        # Return mean normalized percentile
        return float(np.mean(normalized))
    
    def _normalize_p_values_for_combination(
        self,
        p_values: Dict[str, float],
        metrics: List[str]
    ) -> float:
        """Normalize p-values for combination.
        
        Lower p-value = better, so we invert: 1 - p_value
        Then average across metrics.
        """
        if not metrics:
            return 0.0
        
        # Get p-values for all metrics
        values = [p_values.get(m, 1.0) for m in metrics]
        
        # Invert: lower p-value = better, so 1 - p_value gives higher score
        inverted = [max(0.0, min(1.0, 1.0 - p)) for p in values]
        
        # Return mean inverted p-value
        return float(np.mean(inverted))
    
    def _calculate_combined_score(
        self,
        permutation_result: PermutationResult,
        bootstrap_result: BootstrapResult,
        randomized_result: RandomizedEntryResult,
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Calculate combined robustness score from all tests.
        
        CORRECT IMPLEMENTATION (per specification):
        - metric_score = 0.5*(1 - p_value) + 0.5*(percentile/100)
        - Combined score = Σ(weight_i * method_score_i)
        - Weights: random-entry: 50%, bootstrap: 30%, permutation: 20%
        
        IMPORTANT: Different tests have different null hypotheses:
        - Permutation: "Trade outcomes in random order"
        - Bootstrap: "Resampled market returns"
        - Randomized Entry: "Random entries with same risk management"
        """
        # Calculate individual test scores (average of metric scores per test)
        perm_score = self._calculate_test_score(
            permutation_result.percentiles,
            permutation_result.p_values,
            metrics
        )
        
        bootstrap_score = self._calculate_test_score(
            bootstrap_result.percentiles,
            bootstrap_result.p_values,
            metrics
        )
        
        randomized_score = self._calculate_test_score(
            randomized_result.percentiles,
            randomized_result.p_values,
            metrics
        )
        
        # Combined score: weighted average of test scores
        # Weights: random-entry: 50%, bootstrap: 30%, permutation: 20%
        combined_score = (
            self.weights['permutation'] * perm_score +
            self.weights['bootstrap'] * bootstrap_score +
            self.weights['randomized_entry'] * randomized_score
        )
        
        # Calculate combined percentile (weighted average)
        perm_percentile_avg = np.mean([permutation_result.percentiles.get(m, 0.0) for m in metrics])
        bootstrap_percentile_avg = np.mean([bootstrap_result.percentiles.get(m, 0.0) for m in metrics])
        randomized_percentile_avg = np.mean([randomized_result.percentiles.get(m, 0.0) for m in metrics])
        
        combined_percentile = (
            self.weights['permutation'] * perm_percentile_avg +
            self.weights['bootstrap'] * bootstrap_percentile_avg +
            self.weights['randomized_entry'] * randomized_percentile_avg
        )
        
        # Calculate combined p-value (weighted average)
        perm_p_value_avg = np.mean([permutation_result.p_values.get(m, 1.0) for m in metrics])
        bootstrap_p_value_avg = np.mean([bootstrap_result.p_values.get(m, 1.0) for m in metrics])
        randomized_p_value_avg = np.mean([randomized_result.p_values.get(m, 1.0) for m in metrics])
        
        combined_p_value = (
            self.weights['permutation'] * perm_p_value_avg +
            self.weights['bootstrap'] * bootstrap_p_value_avg +
            self.weights['randomized_entry'] * randomized_p_value_avg
        )
        
        # Robustness threshold: score >= 0.6 and percentile >= 70
        robust = combined_score >= 0.60 and combined_percentile >= 70.0
        
        return {
            'robust': robust,
            'score': float(combined_score),
            'percentile': float(combined_percentile),
            'p_value': float(combined_p_value),
            'permutation_score': float(perm_score),
            'bootstrap_score': float(bootstrap_score),
            'randomized_entry_score': float(randomized_score),
            'warnings': self._collect_warnings(
                permutation_result, bootstrap_result, randomized_result, metrics
            )
        }
    
    def _collect_warnings(
        self,
        permutation_result: PermutationResult,
        bootstrap_result: BootstrapResult,
        randomized_result: RandomizedEntryResult,
        metrics: List[str]
    ) -> List[str]:
        """Collect warnings about test validity."""
        warnings = []
        
        # Check for flat distributions
        for metric in metrics:
            perm_dist = permutation_result.permuted_distributions.get(metric, np.array([]))
            boot_dist = bootstrap_result.bootstrap_distributions.get(metric, np.array([]))
            rand_dist = randomized_result.random_distributions.get(metric, np.array([]))
            
            if len(perm_dist) > 1:
                perm_std = np.std(perm_dist, ddof=1)
                if perm_std < 1e-6:
                    warnings.append(f"Permutation {metric}: near-zero variance (distribution may be invalid)")
            
            if len(boot_dist) > 1:
                boot_std = np.std(boot_dist, ddof=1)
                if boot_std < 1e-6:
                    warnings.append(f"Bootstrap {metric}: near-zero variance (distribution may be invalid)")
            
            if len(rand_dist) > 1:
                rand_std = np.std(rand_dist, ddof=1)
                if rand_std < 1e-6:
                    warnings.append(f"Randomized entry {metric}: near-zero variance (distribution may be invalid)")
        
        # Check randomized entry trade count
        if randomized_result.avg_random_trades < 1.0:
            warnings.append(f"Randomized entry test generated very few trades (avg: {randomized_result.avg_random_trades:.1f})")
        
        return warnings
    
    def run_all(
        self,
        backtest_result: BacktestResult,
        price_series: pd.Series,
        strategy: StrategyBase,
        metrics: Optional[List[str]] = None,
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> MonteCarloSuiteResult:
        """Run all three Monte Carlo engines and combine results.

        Args:
            backtest_result: Original BacktestResult
            price_series: Price series (close prices) used in backtest
            strategy: Strategy instance (needed for randomized entry test)
            metrics: List of metrics to test (default: ['final_pnl', 'sharpe', 'profit_factor'])
            n_iterations: Number of iterations per test
            show_progress: Show progress bars

        Returns:
            MonteCarloSuiteResult with all test results and combined metrics
        """
        if metrics is None:
            # NOTE: Profit factor is not sensitive to trade order.
            # Permutation p-values for PF are informational only.
            metrics = ['final_pnl', 'sharpe', 'profit_factor']
        
        # Run permutation test
        permutation_result = self.permutation_engine.run(
            backtest_result=backtest_result,
            metrics=metrics,
            n_iterations=n_iterations,
            show_progress=show_progress
        )
        
        # Run block bootstrap
        bootstrap_result = self.bootstrap_engine.run(
            backtest_result=backtest_result,
            price_series=price_series,
            metrics=metrics,
            n_iterations=n_iterations,
            show_progress=show_progress
        )
        
        # Prepare price data for randomized entry
        # Convert Series to DataFrame with OHLCV columns
        if isinstance(price_series, pd.Series):
            price_data = pd.DataFrame({
                'open': price_series,
                'high': price_series,
                'low': price_series,
                'close': price_series,
                'volume': 0.0
            })
        else:
            price_data = price_series
        
        # Run randomized entry test
        randomized_result = self.randomized_entry_engine.run(
            backtest_result=backtest_result,
            price_data=price_data,
            strategy=strategy,
            metrics=metrics,
            n_iterations=n_iterations,
            show_progress=show_progress
        )
        
        # Calculate combined results (canonical single dict)
        combined = self._calculate_combined_score(
            permutation_result,
            bootstrap_result,
            randomized_result,
            metrics
        )
        
        # Format results
        return MonteCarloSuiteResult(
            permutation={
                'observed_metrics': permutation_result.observed_metrics,
                'p_values': permutation_result.p_values,
                'percentiles': permutation_result.percentiles,
                'n_iterations': permutation_result.n_iterations,
                'permuted_distributions': permutation_result.permuted_distributions,
                'equity_curves': permutation_result.equity_curves,  # Store for visualization
                    'permutation': permutation_result.permuted_distributions,
            },
            bootstrap={
                'observed_metrics': bootstrap_result.observed_metrics,
                'p_values': bootstrap_result.p_values,
                'percentiles': bootstrap_result.percentiles,
                'n_iterations': bootstrap_result.n_iterations,
                'block_length': bootstrap_result.block_length,
                'bootstrap_distributions': bootstrap_result.bootstrap_distributions,
                'bootstrap': bootstrap_result.bootstrap_distributions,
            },
            randomized_entry={
                'observed_metrics': randomized_result.observed_metrics,
                'p_values': randomized_result.p_values,
                'percentiles': randomized_result.percentiles,
                'n_iterations': randomized_result.n_iterations,
                'avg_random_trades': randomized_result.avg_random_trades,
                'random_distributions': randomized_result.random_distributions,
                'randomized_entry': randomized_result.random_distributions,
            },
            combined=combined,
            skipped_tests=None  # Only populated in run_conditional()
        )
    
    def run_conditional(
        self,
        backtest_result: BacktestResult,
        price_series: pd.Series,
        strategy: StrategyBase,
        test_suitability: Dict[str, Any],  # Dict[str, TestSuitability]
        metrics: Optional[List[str]] = None,
        n_iterations: int = 1000,
        show_progress: bool = True
    ) -> MonteCarloSuiteResult:
        """
        Run Monte Carlo tests conditionally based on suitability assessment.
        
        This method runs only tests that are suitable for the strategy, skipping
        tests that lack statistical power. This prevents false negatives and wasted compute.
        
        Args:
            backtest_result: Original BacktestResult
            price_series: Price series (close prices) used in backtest
            strategy: Strategy instance (needed for randomized entry test)
            test_suitability: Dictionary mapping test names to TestSuitability objects
            metrics: List of metrics to test (default: ['final_pnl', 'sharpe_ratio', 'profit_factor'])
            n_iterations: Number of iterations per test
            show_progress: Show progress bars
            
        Returns:
            MonteCarloSuiteResult with results from suitable tests and skipped test info
        """
        if metrics is None:
            metrics = ['final_pnl', 'sharpe', 'profit_factor']
        
        results = {}
        skipped = {}
        
        # === Permutation Test (Conditional) ===
        perm_suit = test_suitability.get('permutation')
        if perm_suit and perm_suit.suitable:
            if show_progress:
                print("Running Permutation Test (suitable)...")
            permutation_result = self.permutation_engine.run(
                backtest_result=backtest_result,
                metrics=metrics,
                n_iterations=n_iterations,
                show_progress=show_progress
            )
            results['permutation'] = {
                'observed_metrics': permutation_result.observed_metrics,
                'p_values': permutation_result.p_values,
                'percentiles': permutation_result.percentiles,
                'n_iterations': permutation_result.n_iterations,
                'permuted_distributions': permutation_result.permuted_distributions,
                'equity_curves': permutation_result.equity_curves,
                'skipped': False
            }
        else:
            skipped['permutation'] = {
                'skipped': True,
                'reason': perm_suit.reason if perm_suit else "Not assessed",
                'alternatives': perm_suit.alternatives if perm_suit else []
            }
            results['permutation'] = {
                'skipped': True,
                'reason': perm_suit.reason if perm_suit else "Not assessed",
                'alternatives': perm_suit.alternatives if perm_suit else []
            }
        
        # === Bootstrap Test (Conditional) ===
        bootstrap_suit = test_suitability.get('bootstrap')
        if bootstrap_suit and bootstrap_suit.suitable:
            if show_progress:
                print("Running Block Bootstrap Test (suitable)...")
            # Bootstrap test needs only close prices (Series), not full DataFrame
            if isinstance(price_series, pd.DataFrame):
                bootstrap_price_series = price_series['close'] if 'close' in price_series.columns else price_series.iloc[:, 0]
            else:
                bootstrap_price_series = price_series
            bootstrap_result = self.bootstrap_engine.run(
                backtest_result=backtest_result,
                price_series=bootstrap_price_series,
                metrics=metrics,
                n_iterations=n_iterations,
                show_progress=show_progress
            )
            results['bootstrap'] = {
                'observed_metrics': bootstrap_result.observed_metrics,
                'p_values': bootstrap_result.p_values,
                'percentiles': bootstrap_result.percentiles,
                'n_iterations': bootstrap_result.n_iterations,
                'block_length': bootstrap_result.block_length,
                'bootstrap_distributions': bootstrap_result.bootstrap_distributions,
                'skipped': False
            }
        else:
            skipped['bootstrap'] = {
                'skipped': True,
                'reason': bootstrap_suit.reason if bootstrap_suit else "Not assessed",
                'alternatives': bootstrap_suit.alternatives if bootstrap_suit else []
            }
            results['bootstrap'] = {
                'skipped': True,
                'reason': bootstrap_suit.reason if bootstrap_suit else "Not assessed",
                'alternatives': bootstrap_suit.alternatives if bootstrap_suit else []
            }

        # === Randomized Entry Test (Conditional, but commonly run) ===
        randomized_suit = test_suitability.get('randomized_entry')
        if randomized_suit and randomized_suit.suitable:
            if show_progress:
                print("Running Randomized Entry Test (conditional)...")

            # Prepare price data for randomized entry
            # CRITICAL: Randomized entry test needs full OHLCV data to properly simulate
            # intrabar stop losses and take profits. Passing only close prices creates
            # identical bars where stops/targets never trigger, resulting in zero variance.
            if isinstance(price_series, pd.Series):
                # If only close prices provided, warn and create minimal OHLCV
                # This shouldn't happen in normal usage, but handle gracefully
                import warnings
                warnings.warn(
                    "Randomized entry test received only close prices (Series). "
                    "This will create identical OHLC bars where stops/targets never trigger. "
                    "Pass full DataFrame with OHLCV columns for proper simulation."
                )
                price_data = pd.DataFrame({
                    'open': price_series,
                    'high': price_series,
                    'low': price_series,
                    'close': price_series,
                    'volume': 0.0
                })
            elif isinstance(price_series, pd.DataFrame):
                # Full DataFrame provided - use as-is but ensure required columns exist
                price_data = price_series.copy()
                required_cols = ['open', 'high', 'low', 'close', 'volume']
                missing_cols = [col for col in required_cols if col not in price_data.columns]

                if missing_cols:
                    # Try to derive missing columns
                    if 'close' in price_data.columns:
                        close_col = price_data['close']
                        for col in missing_cols:
                            if col == 'volume':
                                price_data[col] = 0.0
                            elif col in ['open', 'high', 'low']:
                                # Use close as fallback for missing OHLC
                                price_data[col] = close_col
                    else:
                        # No close column - use first numeric column as fallback
                        numeric_cols = price_data.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            fallback_col = price_data[numeric_cols[0]]
                            for col in required_cols:
                                if col not in price_data.columns:
                                    if col == 'volume':
                                        price_data[col] = 0.0
                                    else:
                                        price_data[col] = fallback_col
                        else:
                            raise ValueError(
                                f"Cannot prepare price data: missing required columns {missing_cols} "
                                f"and no numeric columns found to use as fallback. "
                                f"Available columns: {price_data.columns.tolist()}"
                            )
            else:
                raise ValueError(
                    f"price_series must be pd.Series or pd.DataFrame, got {type(price_series)}"
                )

            randomized_result = self.randomized_entry_engine.run(
                backtest_result=backtest_result,
                price_data=price_data,
                strategy=strategy,
                metrics=metrics,
                n_iterations=n_iterations,
                show_progress=show_progress
            )
            results['randomized_entry'] = {
                'observed_metrics': randomized_result.observed_metrics,
                'p_values': randomized_result.p_values,
                'percentiles': randomized_result.percentiles,
                'n_iterations': randomized_result.n_iterations,
                'avg_random_trades': randomized_result.avg_random_trades,
                'random_distributions': randomized_result.random_distributions,
                'randomized_entry_diagnostics': randomized_result.randomized_entry_diagnostics,  # Include diagnostics
                'skipped': False
            }
        else:
            skipped['randomized_entry'] = {
                'skipped': True,
                'reason': randomized_suit.reason if randomized_suit else "Not assessed",
                'alternatives': randomized_suit.alternatives if randomized_suit else []
            }
            results['randomized_entry'] = {
                'skipped': True,
                'reason': randomized_suit.reason if randomized_suit else "Not assessed",
                'alternatives': randomized_suit.alternatives if randomized_suit else []
            }
        
        # Calculate combined score (only from suitable tests)
        combined = self._calculate_combined_score_conditional(
            results, test_suitability, metrics
        )
        
        return MonteCarloSuiteResult(
            permutation=results['permutation'],
            bootstrap=results['bootstrap'],
            randomized_entry=results['randomized_entry'],
            combined=combined,
            skipped_tests=skipped if skipped else None
        )
    
    def _calculate_combined_score_conditional(
        self,
        results: Dict[str, Dict[str, Any]],
        test_suitability: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """Calculate combined score only from suitable tests that actually ran.
        
        Args:
            results: Dictionary of test results (may include skipped tests)
            test_suitability: Dictionary mapping test names to TestSuitability objects
            metrics: List of metrics to test
            
        Returns:
            Combined score dictionary with normalized weights
        """
        # Collect results from tests that actually ran (not skipped)
        actual_results = {}
        actual_suitability = {}
        
        for test_name in ['permutation', 'bootstrap', 'randomized_entry']:
            result = results.get(test_name, {})
            if not result.get('skipped', False):
                actual_results[test_name] = result
                suit = test_suitability.get(test_name)
                if suit and suit.suitable:
                    actual_suitability[test_name] = suit

        # Split suitable tests into universal vs conditional
        universal_tests = {
            name: suit for name, suit in actual_suitability.items()
            if getattr(suit, 'category', 'universal') == 'universal'
        }
        all_tests = actual_suitability
        
        def _normalized_weights(suitability_map: Dict[str, Any]) -> Dict[str, float]:
            total = sum(
                suit.priority
                for test_name, suit in suitability_map.items()
                if test_name in actual_results
            )
            if total <= 0:
                return {}
            return {
                test_name: suit.priority / total
                for test_name, suit in suitability_map.items()
                if test_name in actual_results
            }

        normalized_weights_universal = _normalized_weights(universal_tests)
        normalized_weights_all = _normalized_weights(all_tests)
        
        # Calculate individual test scores
        perm_score = 0.0
        bootstrap_score = 0.0
        randomized_score = 0.0
        
        perm_percentile_avg = 0.0
        bootstrap_percentile_avg = 0.0
        randomized_percentile_avg = 0.0
        
        perm_p_value_avg = 1.0
        bootstrap_p_value_avg = 1.0
        randomized_p_value_avg = 1.0
        
        if 'permutation' in actual_results:
            perm_p_values = actual_results['permutation'].get('p_values', {})
            perm_percentiles = actual_results['permutation'].get('percentiles', {})
            perm_score = self._calculate_test_score(perm_percentiles, perm_p_values, metrics)
            perm_percentile_avg = np.mean([perm_percentiles.get(m, 0.0) for m in metrics]) if perm_percentiles else 0.0
            perm_p_value_avg = np.mean([perm_p_values.get(m, 1.0) for m in metrics]) if perm_p_values else 1.0
        
        if 'bootstrap' in actual_results:
            bootstrap_p_values = actual_results['bootstrap'].get('p_values', {})
            bootstrap_percentiles = actual_results['bootstrap'].get('percentiles', {})
            bootstrap_score = self._calculate_test_score(bootstrap_percentiles, bootstrap_p_values, metrics)
            bootstrap_percentile_avg = np.mean([bootstrap_percentiles.get(m, 0.0) for m in metrics]) if bootstrap_percentiles else 0.0
            bootstrap_p_value_avg = np.mean([bootstrap_p_values.get(m, 1.0) for m in metrics]) if bootstrap_p_values else 1.0
        
        if 'randomized_entry' in actual_results:
            randomized_p_values = actual_results['randomized_entry'].get('p_values', {})
            randomized_percentiles = actual_results['randomized_entry'].get('percentiles', {})
            randomized_score = self._calculate_test_score(randomized_percentiles, randomized_p_values, metrics)
            randomized_percentile_avg = np.mean([randomized_percentiles.get(m, 0.0) for m in metrics]) if randomized_percentiles else 0.0
            randomized_p_value_avg = np.mean([randomized_p_values.get(m, 1.0) for m in metrics]) if randomized_p_values else 1.0
        
        def _weighted_combo(weights: Dict[str, float]) -> Dict[str, float]:
            score = (
                weights.get('permutation', 0.0) * perm_score +
                weights.get('bootstrap', 0.0) * bootstrap_score +
                weights.get('randomized_entry', 0.0) * randomized_score
            )
            percentile = (
                weights.get('permutation', 0.0) * perm_percentile_avg +
                weights.get('bootstrap', 0.0) * bootstrap_percentile_avg +
                weights.get('randomized_entry', 0.0) * randomized_percentile_avg
            )
            p_value = (
                weights.get('permutation', 0.0) * perm_p_value_avg +
                weights.get('bootstrap', 0.0) * bootstrap_p_value_avg +
                weights.get('randomized_entry', 0.0) * randomized_p_value_avg
            )
            return {
                'score': float(score),
                'percentile': float(percentile),
                'p_value': float(p_value),
                'robust': bool(score >= 0.60 and percentile >= 70.0),
            }

        universal_combo = _weighted_combo(normalized_weights_universal)
        all_combo = _weighted_combo(normalized_weights_all)
        
        # Collect warnings
        warnings = []
        if len(actual_results) < 3:
            skipped_count = 3 - len(actual_results)
            warnings.append(f"{skipped_count} test(s) skipped due to unsuitability")

        if not normalized_weights_universal:
            warnings.append("No suitable universal tests ran - pass/fail gating cannot be computed")
        
        return {
            # Pass/fail keys: UNIVERSAL ONLY
            'robust': universal_combo['robust'],
            'score': universal_combo['score'],
            'percentile': universal_combo['percentile'],
            'p_value': universal_combo['p_value'],

            # Informational: ALL suitable tests (universal + conditional)
            'robust_all': all_combo['robust'],
            'score_all': all_combo['score'],
            'percentile_all': all_combo['percentile'],
            'p_value_all': all_combo['p_value'],
            'permutation_score': float(perm_score),
            'bootstrap_score': float(bootstrap_score),
            'randomized_entry_score': float(randomized_score),
            'warnings': warnings,
            'normalized_weights': normalized_weights_universal,
            'normalized_weights_all': normalized_weights_all,
            'n_suitable_tests': len(universal_tests),
            'n_suitable_tests_all': len(actual_results)
        }

