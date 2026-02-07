"""Walk-forward analysis for strategy validation - FIXED VERSION."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine, BacktestResult
from config.schema import WalkForwardConfig
from metrics.metrics import calculate_enhanced_metrics


@dataclass
class WalkForwardStep:
    """Results from a single walk-forward step."""
    step_number: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_result: BacktestResult
    test_result: BacktestResult
    train_metrics: Dict
    test_metrics: Dict
    excluded_from_stats: bool = False  # True if period has insufficient trades
    exclusion_reason: Optional[str] = None  # Reason for exclusion (e.g., "Insufficient trades: 4 < 30")


@dataclass
class WalkForwardResult:
    """Complete walk-forward analysis results."""
    steps: List[WalkForwardStep]
    summary: Dict
    config: WalkForwardConfig


class WalkForwardAnalyzer:
    """Walk-forward analysis engine - FIXED VERSION."""
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        config: WalkForwardConfig,
        initial_capital: float = 10000.0,
        commission_rate: Optional[float] = None,  # FIXED: Allow None to use market profile
        slippage_ticks: Optional[float] = None    # FIXED: Allow None to use market profile
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            strategy_class: Strategy class to test
            config: WalkForwardConfig with analysis parameters
            initial_capital: Starting capital for each period
            commission_rate: Commission rate per trade (if None, uses market profile)
            slippage_ticks: Slippage in ticks (if None, uses market profile)
        """
        self.strategy_class = strategy_class
        self.config = config
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
    
    def _parse_period(self, period_str: str) -> timedelta:
        """Parse period string like '1 year', '6 months', '30 days'."""
        parts = period_str.lower().strip().split()
        if len(parts) != 2:
            raise ValueError(f"Invalid period format: {period_str}")
        
        value = int(parts[0])
        unit = parts[1]
        
        if unit in ['year', 'years', 'y']:
            return timedelta(days=value * 365)
        elif unit in ['month', 'months', 'm']:
            return timedelta(days=value * 30)
        elif unit in ['week', 'weeks', 'w']:
            return timedelta(weeks=value)
        elif unit in ['day', 'days', 'd']:
            return timedelta(days=value)
        else:
            raise ValueError(f"Unknown time unit: {unit}")
    
    def _build_expanding_windows(
        self,
        data: pd.DataFrame,
        train_period: timedelta,
        test_period: timedelta,
        analysis_start: Optional[pd.Timestamp] = None,
        analysis_end: Optional[pd.Timestamp] = None,
        holdout_end: Optional[pd.Timestamp] = None
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Build expanding window train/test pairs.
        
        Args:
            data: Full dataset with datetime index (may include data before analysis_start for warm-up)
            train_period: Training period duration
            test_period: Test period duration
            analysis_start: Start of analysis period (if None, uses data.index[0])
            analysis_end: End of analysis period (if None, uses data.index[-1])
            holdout_end: End date of holdout period to exclude (if provided, windows stop before this date)
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        # Use config dates if provided, otherwise use data range
        # This allows us to pass full dataset for warm-up but constrain windows to OOS period
        data_start = analysis_start if analysis_start is not None else data.index[0]
        data_end = analysis_end if analysis_end is not None else data.index[-1]
        
        # CRITICAL: Exclude holdout period from walk-forward
        # If holdout_end is provided, ensure we stop before it
        if holdout_end is not None:
            holdout_end_ts = pd.to_datetime(holdout_end)
            if data_end > holdout_end_ts:
                data_end = holdout_end_ts
                print(f"⚠️  Holdout period excluded: Walk-forward stops at {holdout_end_ts.date()} (before holdout)")
        
        # Ensure dates are within data range
        if data_start < data.index[0]:
            data_start = data.index[0]
        if data_end > data.index[-1]:
            data_end = data.index[-1]
        
        # Start with initial training window
        train_start = data_start
        train_end = train_start + train_period
        
        # Ensure minimum training period
        min_train = self._parse_period(self.config.min_training_period)
        if train_end - train_start < min_train:
            train_end = train_start + min_train
        
        while train_end < data_end:
            # Test window starts right after training ends
            test_start = train_end
            test_end = test_start + test_period
            
            # Don't exceed data range
            if test_end > data_end:
                test_end = data_end
            
            # Only add if we have valid windows
            if test_start < test_end and train_start < train_end:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Expand training window for next iteration
            train_end = test_end
            
            # Check if we can continue
            if train_end >= data_end:
                break
        
        return windows
    
    def _build_rolling_windows(
        self,
        data: pd.DataFrame,
        train_period: timedelta,
        test_period: timedelta,
        analysis_start: Optional[pd.Timestamp] = None,
        analysis_end: Optional[pd.Timestamp] = None,
        holdout_end: Optional[pd.Timestamp] = None
    ) -> List[Tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
        """
        Build rolling window train/test pairs.
        
        Args:
            data: Full dataset with datetime index (may include data before analysis_start for warm-up)
            train_period: Training period duration
            test_period: Test period duration
            analysis_start: Start of analysis period (if None, uses data.index[0])
            analysis_end: End of analysis period (if None, uses data.index[-1])
            holdout_end: End date of holdout period to exclude (if provided, windows stop before this date)
        
        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        windows = []
        # Use config dates if provided, otherwise use data range
        # This allows us to pass full dataset for warm-up but constrain windows to OOS period
        data_start = analysis_start if analysis_start is not None else data.index[0]
        data_end = analysis_end if analysis_end is not None else data.index[-1]
        
        # CRITICAL: Exclude holdout period from walk-forward
        # If holdout_end is provided, ensure we stop before it
        if holdout_end is not None:
            holdout_end_ts = pd.to_datetime(holdout_end)
            if data_end > holdout_end_ts:
                data_end = holdout_end_ts
                print(f"⚠️  Holdout period excluded: Walk-forward stops at {holdout_end_ts.date()} (before holdout)")
        
        # Ensure dates are within data range
        if data_start < data.index[0]:
            data_start = data.index[0]
        if data_end > data.index[-1]:
            data_end = data.index[-1]
        
        # Start with initial training window
        train_start = data_start
        train_end = train_start + train_period
        
        # Ensure minimum training period
        min_train = self._parse_period(self.config.min_training_period)
        if train_end - train_start < min_train:
            train_end = train_start + min_train
        
        while train_end < data_end:
            # Test window starts right after training ends
            test_start = train_end
            test_end = test_start + test_period
            
            # Don't exceed data range
            if test_end > data_end:
                test_end = data_end
            
            # Only add if we have valid windows
            if test_start < test_end and train_start < train_end:
                windows.append((train_start, train_end, test_start, test_end))
            
            # Roll forward: move both windows by test period
            train_start = test_start
            train_end = test_end
            
            # Check if we can continue
            if train_end >= data_end:
                break
        
        return windows
    
    def _run_backtest_on_slice(
        self,
        strategy: StrategyBase,
        data: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        warmup_bars: int = 500
    ) -> BacktestResult:
        """Run backtest on a data slice with warm-up for indicators.
        
        CRITICAL FIX: The original version had issues with:
        1. Warm-up data not being included properly
        2. Trade filtering being too aggressive
        3. Engine running on wrong date ranges
        
        NEW APPROACH:
        - Get warm-up data BEFORE start
        - Run engine with start_date=start, end_date=end (let engine handle filtering)
        - NO post-filtering of trades (engine already handles it correctly)
        
        Args:
            strategy: Strategy instance
            data: Full dataset
            start: Start of trading period (inclusive)
            end: End of trading period (exclusive)
            warmup_bars: Number of bars before start for indicator warm-up
        """
        # CRITICAL FIX: Find warm-up start by going back warmup_bars from start
        try:
            start_loc = data.index.get_loc(start)
            warmup_start_idx = max(0, start_loc - warmup_bars)
            warmup_start = data.index[warmup_start_idx]
        except (KeyError, IndexError):
            # If start not in index, find closest
            valid_dates = data.index[data.index <= start]
            if len(valid_dates) > 0:
                warmup_start = valid_dates[0]
            else:
                warmup_start = data.index[0]
        
        # CRITICAL FIX: Get data including warm-up
        # Use iloc to ensure we get all data from warmup_start onwards
        warmup_start_idx = data.index.get_loc(warmup_start)
        try:
            end_idx = data.index.get_loc(end)
        except KeyError:
            # If end not in index, find closest
            valid_dates = data.index[data.index < end]
            if len(valid_dates) > 0:
                end_idx = data.index.get_loc(valid_dates[-1]) + 1
            else:
                end_idx = len(data)
        
        full_data = data.iloc[warmup_start_idx:end_idx].copy()
        
        if len(full_data) == 0:
            warnings.warn(f"No data found for period {start} to {end}")
            # Return empty result
            from engine.backtest_engine import BacktestResult
            return BacktestResult(
                strategy_name=strategy.name if hasattr(strategy, 'name') else 'unknown',
                symbol=strategy.config.market.symbol if hasattr(strategy, 'config') else 'unknown',
                initial_capital=self.initial_capital,
                final_capital=self.initial_capital,
                total_trades=0,
                winning_trades=0,
                losing_trades=0,
                win_rate=0.0,
                total_pnl=0.0,
                total_commission=0.0,
                total_slippage=0.0,
                max_drawdown=0.0,
                trades=[],
                equity_curve=pd.Series([self.initial_capital], index=[0]),
                trade_returns=np.array([], dtype=float),
                drawdown_curve=pd.Series([0.0], index=[0]),
                exposure_stats={},
                leverage_stats={},
                margin_utilization=None
            )
        
        # Validate data quality - check for missing OHLC values
        required_cols = ['open', 'high', 'low', 'close']
        missing_cols = [col for col in required_cols if col not in full_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in data: {missing_cols}")
        
        # Check for NaN values in OHLC data (critical for backtesting)
        ohlc_data = full_data[required_cols]
        nan_counts = ohlc_data.isna().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            # Missing OHLC values are a serious issue - they can cause:
            # 1. Indicator calculation errors (NaN propagation)
            # 2. Signal generation failures
            # 3. Incorrect trade execution prices
            warnings.warn(
                f"⚠️  WARNING: Found {total_nans} missing OHLC values in data:\n"
                f"   {nan_counts.to_dict()}\n"
                f"   This may cause backtest errors. Consider:\n"
                f"   1. Cleaning data source\n"
                f"   2. Forward-filling missing values\n"
                f"   3. Dropping rows with missing OHLC\n"
                f"   Data range: {full_data.index[0]} to {full_data.index[-1]}"
            )
            # Drop rows with any missing OHLC values (safer than forward-fill for backtesting)
            rows_before = len(full_data)
            full_data = full_data.dropna(subset=required_cols)
            rows_after = len(full_data)
            if rows_before != rows_after:
                warnings.warn(
                    f"   Dropped {rows_before - rows_after} rows with missing OHLC values. "
                    f"Remaining: {rows_after} bars"
                )
        
        # Debug: Log data slice info (only if issues found)
        if total_nans > 0:
            print(f"    Data quality: {len(full_data)} bars, Missing OHLC: {total_nans} (dropped)")
            print(f"    Running backtest on data: {full_data.index[0]} to {full_data.index[-1]}")
            print(f"    Trading period: {start} to {end}")
        
        # Create engine
        engine = BacktestEngine(
            strategy=strategy,
            initial_capital=self.initial_capital,
            commission_rate=None,  # CRITICAL FIX: Let engine use market profile
            slippage_ticks=None    # CRITICAL FIX: Let engine use market profile
        )
        
        # CRITICAL FIX: Let engine handle date filtering
        # Pass full_data (including warm-up) but specify start_date and end_date
        # Engine will use warm-up for indicators but only trade during [start, end)
        result = engine.run(full_data, start_date=start, end_date=end)
        
        # DEBUG: Check how many signals were generated
        # This helps diagnose if the problem is signal generation or trade execution
        try:
            # Try to access signals from engine (may need to modify engine to expose this)
            # For now, just report trade count
            print(f"    Result: {result.total_trades} trades executed")
        except Exception:
            pass
        
        # NO POST-FILTERING - engine already did it correctly
        
        return result
    
    def run(
        self,
        data: pd.DataFrame,
        strategy_config: Dict
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis.
        
        Args:
            data: Full dataset with datetime index
            strategy_config: Strategy configuration dict
        
        Returns:
            WalkForwardResult with all steps and summary
        """
        # Parse periods
        # Handle both dict and string formats
        if isinstance(self.config.training_period, dict):
            train_period_str = self.config.training_period.get('duration', '1 year')
        else:
            train_period_str = str(self.config.training_period)
        
        if isinstance(self.config.test_period, dict):
            test_period_str = self.config.test_period.get('duration', '1 year')
        else:
            test_period_str = str(self.config.test_period)
        
        train_period = self._parse_period(train_period_str)
        test_period = self._parse_period(test_period_str)
        
        # Extract analysis boundaries from config (if provided)
        # This allows constraining windows to OOS period while using full data for warm-up
        analysis_start = None
        analysis_end = None
        holdout_end = None
        if hasattr(self.config, 'start_date') and self.config.start_date:
            try:
                analysis_start = pd.to_datetime(self.config.start_date)
            except Exception:
                pass
        if hasattr(self.config, 'end_date') and self.config.end_date:
            try:
                analysis_end = pd.to_datetime(self.config.end_date)
            except Exception:
                pass
        if hasattr(self.config, 'holdout_end_date') and self.config.holdout_end_date:
            try:
                holdout_end = pd.to_datetime(self.config.holdout_end_date)
            except Exception:
                pass
        
        # Build windows (constrained to analysis_start/end if provided, excluding holdout)
        if self.config.window_type == "expanding":
            windows = self._build_expanding_windows(data, train_period, test_period, analysis_start, analysis_end, holdout_end)
        else:  # rolling
            windows = self._build_rolling_windows(data, train_period, test_period, analysis_start, analysis_end, holdout_end)
        
        if not windows:
            raise ValueError("No valid windows found for walk-forward analysis")
        
        print(f"\nGenerated {len(windows)} walk-forward windows:")
        for i, (ts, te, vs, ve) in enumerate(windows):
            print(f"  Window {i+1}: Train [{ts.date()} to {te.date()}], Test [{vs.date()} to {ve.date()}]")
        
        steps = []
        
        # Run each step
        for step_num, (train_start, train_end, test_start, test_end) in enumerate(windows):
            print(f"\n{'='*60}")
            print(f"Step {step_num + 1}/{len(windows)}")
            print(f"{'='*60}")
            
            # Create strategy instance
            from config.schema import validate_strategy_config
            strategy_config_obj = validate_strategy_config(strategy_config)
            
            # Print strategy config for first step
            if step_num == 0:
                print("\n" + "="*60)
                print("Strategy Configuration")
                print("="*60)
                print(f"Strategy class: {self.strategy_class.__name__}")
                print(f"Market: {strategy_config_obj.market.symbol}")
                print(f"Timeframes: signal={strategy_config_obj.timeframes.signal_tf}, entry={strategy_config_obj.timeframes.entry_tf}")
                print(f"Risk per trade: {strategy_config_obj.risk.risk_per_trade_pct}%")
                print(f"MA Settings: {strategy_config_obj.ma_settings}")
                print(f"MACD Settings: {strategy_config_obj.macd_settings}")
                if hasattr(strategy_config_obj, 'execution'):
                    print(f"Execution: {strategy_config_obj.execution}")
                print("="*60 + "\n")
            
            strategy = self.strategy_class(strategy_config_obj)
            
            # Calculate warm-up needed based on strategy configuration
            # Industry standard: Use 3x the longest indicator period for stability
            # For multi-timeframe strategies, ensure we have enough bars for the longest timeframe
            try:
                ma_settings = getattr(strategy.config, 'ma_settings', {})
                longest_period = 0
                
                # Check both signal and entry timeframes
                for tf_key in ['signal', 'entry']:
                    if tf_key in ma_settings:
                        ma_cfg = ma_settings[tf_key]
                        periods = [
                            getattr(ma_cfg, 'fast', 0),
                            getattr(ma_cfg, 'medium', 0),
                            getattr(ma_cfg, 'slow', 0),
                            getattr(ma_cfg, 'slowest', 0),
                            getattr(ma_cfg, 'optional', 0),
                        ]
                        longest_period = max(longest_period, max(periods) if periods else 0)
                
                # For multi-timeframe strategies, account for the signal timeframe
                # If signal timeframe is daily or higher, we need more warm-up bars
                signal_tf = getattr(strategy.config.timeframes, 'signal_tf', '1h')
                entry_tf = getattr(strategy.config.timeframes, 'entry_tf', '1h')
                
                # Calculate minimum warm-up based on longest period
                base_warmup = max(200, longest_period * 3)
                
                # For daily or higher timeframes, ensure we have enough bars
                # This is a conservative estimate - actual needs depend on data frequency
                if 'd' in signal_tf.lower() or 'day' in signal_tf.lower():
                    # For daily signals, ensure at least 100 days worth of base bars
                    # This is conservative but ensures indicators stabilize
                    # Assuming base timeframe is at least hourly (24 bars/day)
                    # For 15m data: 100 days * 96 bars/day = 9600 bars (too much)
                    # Instead, use a more reasonable multiplier
                    warmup_bars = max(base_warmup, 1000)  # Conservative for daily signals
                else:
                    warmup_bars = base_warmup
                    
            except Exception:
                # Fallback to defaults if config inspection fails
                warmup_bars = 500
            
            print(f"Using {warmup_bars} bars for warm-up")
            
            # Run training backtest
            print(f"\nTraining period: {train_start.date()} to {train_end.date()}")
            train_result = self._run_backtest_on_slice(strategy, data, train_start, train_end, warmup_bars=warmup_bars)
            
            # Run test backtest
            print(f"\nTest period: {test_start.date()} to {test_end.date()}")
            test_result = self._run_backtest_on_slice(strategy, data, test_start, test_end, warmup_bars=warmup_bars)
            
            # Calculate enhanced metrics using unified pipeline
            train_enhanced = calculate_enhanced_metrics(train_result)
            test_enhanced = calculate_enhanced_metrics(test_result)
            
            # Extract key metrics from enhanced metrics
            train_metrics = {
                'pf': train_enhanced.get('profit_factor', 0.0),
                'sharpe': train_enhanced.get('sharpe_ratio', train_enhanced.get('sharpe', 0.0)),
                'total_trades': train_result.total_trades,
                'win_rate': train_result.win_rate,
            }
            
            test_metrics = {
                'pf': test_enhanced.get('profit_factor', 0.0),
                'sharpe': test_enhanced.get('sharpe_ratio', test_enhanced.get('sharpe', 0.0)),
                'total_trades': test_result.total_trades,
                'win_rate': test_result.win_rate,
            }
            
            # Check for insufficient trades (industry standard: minimum 30 trades for statistical significance)
            # Get min_trades threshold from config if available, otherwise use default 30
            min_trades = 30  # Default industry standard
            if hasattr(self.config, 'wf_min_trades_per_period') and self.config.wf_min_trades_per_period is not None:
                min_trades = self.config.wf_min_trades_per_period
            elif hasattr(self.config, 'min_trades_per_period') and self.config.min_trades_per_period is not None:
                min_trades = self.config.min_trades_per_period
            
            excluded_from_stats = False
            exclusion_reason = None
            
            if test_result.total_trades < min_trades:
                excluded_from_stats = True
                exclusion_reason = f"Insufficient trades: {test_result.total_trades} < {min_trades} (statistical significance requires minimum {min_trades} trades)"
                print(f"  ⚠️  WARNING: Test period excluded from statistics - {exclusion_reason}")
            
            # Print results with correct PF from metrics pipeline
            # Note: win_rate from BacktestResult is already in percentage (0-100), not decimal (0-1)
            status_marker = " [EXCLUDED]" if excluded_from_stats else ""
            train_win_rate_pct = train_metrics['win_rate']  # Already in percentage
            test_win_rate_pct = test_metrics['win_rate']    # Already in percentage
            print(f"  Training result: {train_result.total_trades} trades, PF={train_metrics['pf']:.2f}, Sharpe={train_metrics['sharpe']:.2f}, Win Rate={train_win_rate_pct:.2f}%")
            print(f"  Test result: {test_result.total_trades} trades, PF={test_metrics['pf']:.2f}, Sharpe={test_metrics['sharpe']:.2f}, Win Rate={test_win_rate_pct:.2f}%{status_marker}")
            
            step = WalkForwardStep(
                step_number=step_num + 1,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
                train_result=train_result,
                test_result=test_result,
                train_metrics=train_metrics,
                test_metrics=test_metrics,
                excluded_from_stats=excluded_from_stats,
                exclusion_reason=exclusion_reason
            )
            steps.append(step)
        
        # Separate included and excluded periods
        included_steps = [step for step in steps if not step.excluded_from_stats]
        excluded_steps = [step for step in steps if step.excluded_from_stats]
        
        # Collect metrics from included steps only (for summary calculation)
        test_pfs = [step.test_metrics.get('pf', 0) for step in included_steps]
        test_sharpes = [step.test_metrics.get('sharpe', 0) for step in included_steps]
        
        # Calculate train PFs for WFE calculation (only from included periods)
        train_pfs = [step.train_metrics.get('pf', 0) for step in included_steps]
        
        # Calculate Walk-Forward Efficiency (WFE) - per-period efficiency, then averaged (only included periods)
        wfe_values = []
        for step in included_steps:
            train_pf = step.train_metrics.get('pf', 0)
            test_pf = step.test_metrics.get('pf', 0)
            if train_pf > 0:
                wfe = test_pf / train_pf
                wfe_values.append(wfe)
        
        mean_wfe = float(np.mean(wfe_values)) if wfe_values else 0.0
        
        # Handle case where all periods are excluded
        if len(included_steps) == 0:
            # All periods excluded - set summary to indicate this
            summary = {
                'total_steps': len(steps),
                'included_steps': 0,
                'excluded_steps': len(excluded_steps),
                'mean_test_pf': None,  # Use None instead of 0.0 to indicate N/A
                'std_test_pf': None,
                'min_test_pf': None,
                'max_test_pf': None,
                'mean_train_pf': None,
                'mean_test_sharpe': None,
                'std_test_sharpe': None,
                'consistency_score': None,
                'walk_forward_efficiency': None,
                'wfe_values': [],
                'all_periods_excluded': True,
                'excluded_periods': [
                    {
                        'step': step.step_number,
                        'period': f"{step.test_start.date()} to {step.test_end.date()}",
                        'trades': step.test_result.total_trades,
                        'reason': step.exclusion_reason
                    }
                    for step in excluded_steps
                ]
            }
        else:
            # Normal case - calculate statistics from included periods
            summary = {
                'total_steps': len(steps),
                'included_steps': len(included_steps),
                'excluded_steps': len(excluded_steps),
                'mean_test_pf': float(np.mean(test_pfs)) if test_pfs else 0.0,
                'std_test_pf': float(np.std(test_pfs)) if len(test_pfs) > 1 else 0.0,
                'min_test_pf': float(np.min(test_pfs)) if test_pfs else 0.0,
                'max_test_pf': float(np.max(test_pfs)) if test_pfs else 0.0,
                'mean_train_pf': float(np.mean(train_pfs)) if train_pfs else 0.0,
                'mean_test_sharpe': float(np.mean(test_sharpes)) if test_sharpes else 0.0,
                'std_test_sharpe': float(np.std(test_sharpes)) if len(test_sharpes) > 1 else 0.0,
                'consistency_score': self._calculate_consistency(test_pfs),
                'walk_forward_efficiency': mean_wfe,  # Mean WFE across all periods
                'wfe_values': wfe_values,  # Individual WFE values for each period
                'all_periods_excluded': False,
                'excluded_periods': [
                    {
                        'step': step.step_number,
                        'period': f"{step.test_start.date()} to {step.test_end.date()}",
                        'trades': step.test_result.total_trades,
                        'reason': step.exclusion_reason
                    }
                    for step in excluded_steps
                ]
            }
        
        return WalkForwardResult(
            steps=steps,
            summary=summary,
            config=self.config
        )
    
    def _calculate_consistency(self, values: List[float]) -> float:
        """
        Calculate consistency score (coefficient of variation).
        Lower is better (more consistent).
        """
        if not values or len(values) < 2:
            return 0.0
        
        mean_val = np.mean(values)
        if mean_val == 0:
            return 0.0
        
        std_val = np.std(values)
        cv = std_val / abs(mean_val) if mean_val != 0 else 0.0
        
        # Return as consistency score (inverse of CV, normalized)
        consistency = 1.0 / (1.0 + cv) if cv > 0 else 1.0
        return float(consistency)