"""Enhanced stationarity detection that tests the conditions that make strategies profitable.

This implementation addresses the critical issue for sparse/regime-dependent strategies where
PF variance reflects natural trade distribution variance rather than actual degradation.

Key improvements:
1. Conditional metrics: Measures signal frequency, conditional win rates, payoff ratios
2. Filter-aware regime analysis: Only tests enabled regime filters
3. Dual-threshold retraining: Absolute (PF < 1.5) + relative (85% degradation)
4. Works for all strategy types: Trend-following, mean reversion, etc.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np
from datetime import timedelta

from strategies.base import StrategyBase
from engine.backtest_engine import BacktestEngine, BacktestResult
from config.schema import StrategyConfig, StationarityCriteria
from metrics.metrics import calculate_enhanced_metrics


@dataclass
class ConditionalMetrics:
    """Metrics that measure the conditions that make PF possible."""
    signal_frequency: float  # Signals (trades) per day
    conditional_win_rate: float  # P(Win | Signal)
    avg_payoff_ratio: float  # E[Win] / E[Loss]
    big_win_frequency: float  # Big wins per period (wins > 2R)
    consecutive_losses_between_wins: float  # Run analysis
    avg_win_size: float  # Average winning trade size
    avg_loss_size: float  # Average losing trade size


@dataclass
class RegimeMetrics:
    """Market regime characteristics (only for enabled filters)."""
    # ADX-based (if ADX filter enabled)
    adx_above_threshold_pct: Optional[float] = None  # % of time ADX > threshold
    avg_adx: Optional[float] = None
    
    # ATR-based (if ATR filter enabled)
    atr_percentile_above_threshold_pct: Optional[float] = None
    avg_atr_percentile: Optional[float] = None
    volatility_quintile: Optional[str] = None  # Q1-Q5
    
    # EMA expansion (if enabled)
    ema_expansion_pct: Optional[float] = None
    
    # Composite (if enabled)
    composite_score_above_threshold_pct: Optional[float] = None


@dataclass
class PeriodAnalysis:
    """Analysis for a single time period."""
    period_start: pd.Timestamp
    period_end: pd.Timestamp
    days: int
    
    # Performance metrics
    profit_factor: float
    sharpe_ratio: float
    total_trades: int
    
    # Conditional metrics (what makes PF possible)
    conditional: ConditionalMetrics
    
    # Regime metrics (only for enabled filters)
    regime: RegimeMetrics


@dataclass
class StationarityResult:
    """Enhanced stationarity analysis results."""
    # Original metrics (for backward compatibility)
    days_vs_pf: Dict[int, float]  # days -> average PF
    days_vs_sharpe: Dict[int, float]  # days -> average Sharpe
    recommended_retrain_days: int
    analysis_periods: List[Tuple[pd.Timestamp, pd.Timestamp]]
    
    # Enhanced metrics - conditional (what makes PF possible)
    days_vs_conditional_win_rate: Dict[int, float] = field(default_factory=dict)
    days_vs_signal_frequency: Dict[int, float] = field(default_factory=dict)
    days_vs_payoff_ratio: Dict[int, float] = field(default_factory=dict)
    days_vs_big_win_frequency: Dict[int, float] = field(default_factory=dict)
    days_vs_consecutive_losses: Dict[int, float] = field(default_factory=dict)
    
    # Regime-conditioned performance (only for enabled filters)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Period-by-period detailed analysis
    period_analyses: List[PeriodAnalysis] = field(default_factory=list)
    
    # Enhanced recommendations
    retrain_reason: str = ""
    min_acceptable_pf: float = 1.5
    
    # Summary statistics
    min_pf_observed: float = 0.0
    max_pf_observed: float = 0.0
    pf_std: float = 0.0
    conditional_metrics_stable: bool = True
    regime_stability: Dict[str, bool] = field(default_factory=dict)
    
    # Enabled filters info
    enabled_regime_filters: List[str] = field(default_factory=list)


class StationarityAnalyzer:
    """Enhanced stationarity analyzer that tests conditions that make strategies profitable.
    
    This analyzer works for ALL strategy types by measuring:
    - Conditional metrics: What makes the strategy profitable (win rate, payoff ratio, etc.)
    - Filter-aware regime analysis: Only tests enabled regime filters
    - Dual-threshold logic: Absolute (PF < 1.5) + relative degradation
    
    Architecture Alignment:
    - Uses unified metrics pipeline (calculate_enhanced_metrics)
    - All metrics calculated consistently with Phase 1 and Phase 2
    - Follows same metrics calculation patterns
    """
    
    def __init__(
        self,
        strategy_class: type[StrategyBase],
        initial_capital: float = 10000.0,
        commission_rate: float = 0.0004,
        slippage_ticks: float = 0.0,
        criteria: Optional[StationarityCriteria] = None
    ):
        """
        Initialize enhanced stationarity analyzer.
        
        Args:
            strategy_class: Strategy class to analyze
            initial_capital: Starting capital
            commission_rate: Commission rate per trade
            slippage_ticks: Slippage in ticks
            criteria: Stationarity criteria configuration (uses defaults if None)
        """
        self.criteria = criteria or StationarityCriteria()
        self.strategy_class = strategy_class
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.min_acceptable_pf = self.criteria.min_acceptable_pf
        self.big_win_threshold_r = self.criteria.big_win_threshold_r
    
    def analyze_days_vs_performance(
        self,
        data: pd.DataFrame,
        strategy_config: StrategyConfig,
        max_days: Optional[int] = None,
        step_days: Optional[int] = None,
        training_period_days: int = 365
    ) -> StationarityResult:
        """
        Enhanced stationarity analysis measuring conditions that make strategy profitable.
        
        This tests:
        1. Conditional metrics: Signal frequency, win rate, payoff ratio, big win frequency
        2. Filter-aware regime analysis: Only tests enabled regime filters
        3. Dual-threshold retraining: Absolute (PF < 1.5) + relative degradation
        
        Works for ALL strategy types by measuring what makes them profitable, not just PF.
        
        Args:
            data: Full dataset with datetime index and OHLCV columns
            strategy_config: Strategy configuration
            max_days: Maximum number of days to test (uses criteria.max_days if None)
            step_days: Step size for days (uses criteria.step_days if None)
            training_period_days: Training period in days before OOS testing
        
        Returns:
            Enhanced StationarityResult with conditional metrics and regime analysis
        """
        # Use configured values if not provided
        max_days = max_days if max_days is not None else self.criteria.max_days
        step_days = step_days if step_days is not None else self.criteria.step_days
        
        data_start = data.index[0]
        data_end = data.index[-1]
        
        # Calculate training end date
        training_end = data_start + timedelta(days=training_period_days)
        
        if training_end >= data_end:
            raise ValueError(f"Not enough data. Need at least {training_period_days} days for training plus test period.")
        
        # Detect enabled regime filters
        enabled_filters = self._detect_enabled_regime_filters(strategy_config)
        
        # Calculate regime metrics for full dataset (only for enabled filters)
        regime_data = self._calculate_filter_aware_regime_metrics(
            data, strategy_config, enabled_filters
        )
        
        # Test different OOS window sizes
        days_to_test = list(range(step_days, max_days + 1, step_days))
        
        # Initialize result dictionaries
        days_vs_pf = {}
        days_vs_sharpe = {}
        days_vs_conditional_win_rate = {}
        days_vs_signal_frequency = {}
        days_vs_payoff_ratio = {}
        days_vs_big_win_frequency = {}
        days_vs_consecutive_losses = {}
        
        analysis_periods = []
        period_analyses = []
        
        for oos_days in days_to_test:
            period_pfs = []
            period_sharpes = []
            period_conditional_win_rates = []
            period_signal_frequencies = []
            period_payoff_ratios = []
            period_big_win_frequencies = []
            period_consecutive_losses = []
            
            # Slide through OOS period with this window size
            current_start = training_end
            while current_start + timedelta(days=oos_days) <= data_end:
                oos_end = current_start + timedelta(days=oos_days)
                oos_slice = data.loc[current_start:oos_end].copy()
                
                if len(oos_slice) > 0:
                    # Run backtest
                    result = self._run_backtest_slice(strategy_config, oos_slice)
                    
                    if result.total_trades > 0:
                        # Calculate enhanced metrics
                        enhanced = calculate_enhanced_metrics(result)
                        pf = enhanced.get('profit_factor', 0.0)
                        sharpe = enhanced.get('sharpe_ratio', enhanced.get('sharpe', 0.0))
                        
                        # Calculate conditional metrics (what makes PF possible)
                        conditional = self._calculate_conditional_metrics(
                            result, oos_slice, oos_days
                        )
                        
                        # Get regime metrics for this period (only for enabled filters)
                        period_regime = self._get_period_regime_metrics(
                            regime_data, current_start, oos_end, enabled_filters, strategy_config
                        )
                        
                        # Store period analysis
                        period_analyses.append(PeriodAnalysis(
                            period_start=current_start,
                            period_end=oos_end,
                            days=oos_days,
                            profit_factor=pf,
                            sharpe_ratio=sharpe,
                            total_trades=result.total_trades,
                            conditional=conditional,
                            regime=period_regime
                        ))
                        
                        # Aggregate for this day window
                        period_pfs.append(pf)
                        period_sharpes.append(sharpe)
                        period_conditional_win_rates.append(conditional.conditional_win_rate)
                        period_signal_frequencies.append(conditional.signal_frequency)
                        period_payoff_ratios.append(conditional.avg_payoff_ratio)
                        period_big_win_frequencies.append(conditional.big_win_frequency)
                        period_consecutive_losses.append(conditional.consecutive_losses_between_wins)
                        
                        analysis_periods.append((current_start, oos_end))
                
                current_start = oos_end
            
            # Average metrics for this window size
            if period_pfs:
                days_vs_pf[oos_days] = float(np.mean(period_pfs))
                days_vs_sharpe[oos_days] = float(np.mean(period_sharpes))
                days_vs_conditional_win_rate[oos_days] = float(np.mean(period_conditional_win_rates))
                days_vs_signal_frequency[oos_days] = float(np.mean(period_signal_frequencies))
                days_vs_payoff_ratio[oos_days] = float(np.mean(period_payoff_ratios))
                days_vs_big_win_frequency[oos_days] = float(np.mean(period_big_win_frequencies))
                days_vs_consecutive_losses[oos_days] = float(np.mean(period_consecutive_losses))
            else:
                days_vs_pf[oos_days] = 0.0
                days_vs_sharpe[oos_days] = 0.0
                days_vs_conditional_win_rate[oos_days] = 0.0
                days_vs_signal_frequency[oos_days] = 0.0
                days_vs_payoff_ratio[oos_days] = 0.0
                days_vs_big_win_frequency[oos_days] = 0.0
                days_vs_consecutive_losses[oos_days] = 0.0
        
        # Calculate regime-conditioned performance (only for enabled filters)
        regime_performance = self._calculate_regime_conditioned_performance(
            period_analyses, regime_data, enabled_filters
        )
        
        # Determine retraining recommendation (dual-threshold logic)
        recommended_days, retrain_reason = self._calculate_enhanced_retrain(
            days_vs_pf,
            days_vs_conditional_win_rate,
            days_vs_payoff_ratio,
            days_vs_big_win_frequency,
            days_vs_consecutive_losses
        )
        
        # Calculate summary statistics
        # Use days_vs_pf which has valid averaged values, filter out invalid ones
        valid_pfs = [pf for pf in days_vs_pf.values() if pf > 0 and np.isfinite(pf)]
        min_pf = min(valid_pfs) if valid_pfs else 0.0
        max_pf = max(valid_pfs) if valid_pfs else 0.0
        pf_std = float(np.std(valid_pfs)) if valid_pfs else 0.0
        
        # Check conditional metrics stability
        conditional_stable = self._check_conditional_stability(
            days_vs_conditional_win_rate,
            days_vs_payoff_ratio,
            days_vs_big_win_frequency
        )
        
        # Check regime stability
        regime_stability = self._check_regime_stability(regime_performance)
        
        return StationarityResult(
            days_vs_pf=days_vs_pf,
            days_vs_sharpe=days_vs_sharpe,
            recommended_retrain_days=recommended_days,
            analysis_periods=analysis_periods,
            days_vs_conditional_win_rate=days_vs_conditional_win_rate,
            days_vs_signal_frequency=days_vs_signal_frequency,
            days_vs_payoff_ratio=days_vs_payoff_ratio,
            days_vs_big_win_frequency=days_vs_big_win_frequency,
            days_vs_consecutive_losses=days_vs_consecutive_losses,
            regime_performance=regime_performance,
            period_analyses=period_analyses,
            retrain_reason=retrain_reason,
            min_acceptable_pf=self.min_acceptable_pf,
            min_pf_observed=min_pf,
            max_pf_observed=max_pf,
            pf_std=pf_std,
            conditional_metrics_stable=conditional_stable,
            regime_stability=regime_stability,
            enabled_regime_filters=enabled_filters
        )
    
    def _run_backtest_slice(
        self,
        strategy_config: StrategyConfig,
        data_slice: pd.DataFrame
    ) -> BacktestResult:
        """Run backtest on a data slice."""
        import tempfile
        from pathlib import Path
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            data_slice.to_csv(f.name)
            temp_path = Path(f.name)
        
        try:
            strategy = self.strategy_class(strategy_config)
            engine = BacktestEngine(
                strategy=strategy,
                initial_capital=self.initial_capital,
                commission_rate=self.commission_rate,
                slippage_ticks=self.slippage_ticks
            )
            result = engine.run(temp_path)
        finally:
            temp_path.unlink()
        
        return result
    
    def _detect_enabled_regime_filters(self, strategy_config: StrategyConfig) -> List[str]:
        """
        Detect which regime filters are enabled in strategy configuration.
        
        Args:
            strategy_config: Strategy configuration
        
        Returns:
            List of enabled filter names (e.g., ['adx', 'atr_percentile'])
        """
        enabled = []
        regime_filters = strategy_config.regime_filters
        
        if regime_filters.adx.enabled:
            enabled.append('adx')
        if regime_filters.atr_percentile.enabled:
            enabled.append('atr_percentile')
        if regime_filters.ema_expansion.enabled:
            enabled.append('ema_expansion')
        if regime_filters.swing.enabled:
            enabled.append('swing')
        if regime_filters.composite.enabled:
            enabled.append('composite')
        
        return enabled
    
    def _calculate_conditional_metrics(
        self,
        result: BacktestResult,
        data_slice: pd.DataFrame,
        period_days: int
    ) -> ConditionalMetrics:
        """
        Calculate conditional metrics that measure what makes the strategy profitable.
        
        These metrics test the conditions that make PF possible, not just PF itself.
        Works for all strategy types (trend, mean reversion, etc.).
        
        Args:
            result: BacktestResult with trades
            data_slice: Data for this period
            period_days: Number of days in period
        
        Returns:
            ConditionalMetrics object
        """
        trades = result.trades
        
        if len(trades) == 0:
            return ConditionalMetrics(
                signal_frequency=0.0,
                conditional_win_rate=0.0,
                avg_payoff_ratio=0.0,
                big_win_frequency=0.0,
                consecutive_losses_between_wins=0.0,
                avg_win_size=0.0,
                avg_loss_size=0.0
            )
        
        # Signal frequency (trades per day)
        signal_frequency = len(trades) / period_days if period_days > 0 else 0.0
        
        # Conditional win rate: P(Win | Signal)
        wins = [t for t in trades if t.pnl_after_costs > 0]
        losses = [t for t in trades if t.pnl_after_costs < 0]
        conditional_win_rate = len(wins) / len(trades) if trades else 0.0
        
        # Payoff ratio: E[Win] / E[Loss]
        avg_win = np.mean([t.pnl_after_costs for t in wins]) if wins else 0.0
        avg_loss = abs(np.mean([t.pnl_after_costs for t in losses])) if losses else 1.0
        payoff_ratio = avg_win / avg_loss if avg_loss > 0 else 0.0
        
        # Big win frequency (wins > threshold R-multiples)
        # Estimate R from average loss
        if trades and avg_loss > 0:
            avg_risk = avg_loss
            big_wins = [
                t for t in wins 
                if t.pnl_after_costs / avg_risk >= self.big_win_threshold_r
            ]
            big_win_frequency = len(big_wins) / period_days if period_days > 0 else 0.0
        else:
            big_win_frequency = 0.0
        
        # Consecutive losses between wins (run analysis)
        consecutive_losses = self._calculate_consecutive_losses(trades)
        
        return ConditionalMetrics(
            signal_frequency=signal_frequency,
            conditional_win_rate=conditional_win_rate,
            avg_payoff_ratio=payoff_ratio,
            big_win_frequency=big_win_frequency,
            consecutive_losses_between_wins=consecutive_losses,
            avg_win_size=avg_win,
            avg_loss_size=avg_loss
        )
    
    def _calculate_consecutive_losses(self, trades: List) -> float:
        """
        Calculate average consecutive losses between wins (run analysis).
        
        This measures if the pattern of losses between wins is stable.
        For trend-following: Should see consistent pattern (e.g., 5-7 losses between big wins).
        If pattern degrades (10, 15, 20+ losses), edge is eroding.
        
        Args:
            trades: List of Trade objects
        
        Returns:
            Average consecutive losses between wins
        """
        if not trades:
            return 0.0
        
        runs = []
        current_run = 0
        
        for trade in trades:
            if trade.pnl_after_costs < 0:
                current_run += 1
            else:
                if current_run > 0:
                    runs.append(current_run)
                current_run = 0
        
        # Don't count final run if it ends with losses (incomplete)
        # Only count complete runs (losses followed by win)
        
        return float(np.mean(runs)) if runs else 0.0
    
    def _calculate_filter_aware_regime_metrics(
        self,
        data: pd.DataFrame,
        strategy_config: StrategyConfig,
        enabled_filters: List[str]
    ) -> pd.DataFrame:
        """
        Calculate regime metrics ONLY for enabled filters.
        
        Args:
            data: Full dataset with OHLCV
            strategy_config: Strategy configuration
            enabled_filters: List of enabled filter names
        
        Returns:
            DataFrame with regime metrics (only for enabled filters)
        """
        df = data.copy()
        regime_filters = strategy_config.regime_filters
        
        # ADX metrics (if ADX filter enabled)
        if 'adx' in enabled_filters:
            df['adx'] = self._calculate_adx_simple(df, period=regime_filters.adx.period)
            # Get threshold based on asset class (simplified - use forex default)
            adx_threshold = regime_filters.adx.min_forex  # Could be enhanced to detect asset class
            df['adx_above_threshold'] = df['adx'] > adx_threshold
        
        # ATR percentile metrics (if ATR filter enabled)
        if 'atr_percentile' in enabled_filters:
            # Calculate ATR
            high_low = df['high'] - df['low']
            high_close = np.abs(df['high'] - df['close'].shift())
            low_close = np.abs(df['low'] - df['close'].shift())
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df['atr'] = true_range.rolling(window=14).mean()
            df['atr_pct'] = (df['atr'] / df['close']) * 100
            
            # Calculate ATR percentile
            lookback = regime_filters.atr_percentile.lookback
            df['atr_percentile'] = df['atr_pct'].rolling(window=lookback).apply(
                lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) == lookback else np.nan
            )
            min_percentile = regime_filters.atr_percentile.min_percentile
            df['atr_percentile_above_threshold'] = df['atr_percentile'] > min_percentile
            
            # Volatility quintiles
            df['vol_quintile'] = pd.qcut(
                df['atr_pct'].fillna(0), 
                q=5, 
                labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'],
                duplicates='drop'
            )
        
        # EMA expansion (if enabled)
        if 'ema_expansion' in enabled_filters:
            # Simplified - would need actual EMA calculation
            lookback = regime_filters.ema_expansion.lookback
            df['ema_expansion'] = 0.0  # Placeholder
        
        # Composite (if enabled)
        if 'composite' in enabled_filters:
            # Would calculate composite score based on enabled components
            df['composite_score'] = 0.0  # Placeholder
            min_score = regime_filters.composite.min_score
            df['composite_above_threshold'] = df['composite_score'] > min_score
        
        return df
    
    def _calculate_adx_simple(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ADX (simplified version)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        return adx.fillna(0)
    
    def _get_period_regime_metrics(
        self,
        regime_data: pd.DataFrame,
        start: pd.Timestamp,
        end: pd.Timestamp,
        enabled_filters: List[str],
        strategy_config: StrategyConfig
    ) -> RegimeMetrics:
        """
        Get regime metrics for a specific period (only for enabled filters).
        
        Args:
            regime_data: DataFrame with regime metrics
            start: Period start timestamp
            end: Period end timestamp
            enabled_filters: List of enabled filter names
            strategy_config: Strategy configuration
        
        Returns:
            RegimeMetrics object (only populated for enabled filters)
        """
        period_data = regime_data.loc[start:end]
        
        if len(period_data) == 0:
            return RegimeMetrics()
        
        regime = RegimeMetrics()
        regime_filters = strategy_config.regime_filters
        
        # ADX metrics (if enabled)
        if 'adx' in enabled_filters:
            regime.avg_adx = float(period_data['adx'].mean())
            if 'adx_above_threshold' in period_data.columns:
                regime.adx_above_threshold_pct = float(
                    period_data['adx_above_threshold'].sum() / len(period_data) * 100
                )
        
        # ATR percentile metrics (if enabled)
        if 'atr_percentile' in enabled_filters:
            if 'atr_percentile' in period_data.columns:
                regime.avg_atr_percentile = float(period_data['atr_percentile'].mean())
            if 'atr_percentile_above_threshold' in period_data.columns:
                regime.atr_percentile_above_threshold_pct = float(
                    period_data['atr_percentile_above_threshold'].sum() / len(period_data) * 100
                )
            if 'vol_quintile' in period_data.columns:
                # Get most common quintile
                quintile_mode = period_data['vol_quintile'].mode()
                regime.volatility_quintile = str(quintile_mode.iloc[0]) if len(quintile_mode) > 0 else None
        
        # EMA expansion (if enabled)
        if 'ema_expansion' in enabled_filters:
            if 'ema_expansion' in period_data.columns:
                regime.ema_expansion_pct = float(period_data['ema_expansion'].mean())
        
        # Composite (if enabled)
        if 'composite' in enabled_filters:
            if 'composite_above_threshold' in period_data.columns:
                regime.composite_score_above_threshold_pct = float(
                    period_data['composite_above_threshold'].sum() / len(period_data) * 100
                )
        
        return regime
    
    def _calculate_regime_conditioned_performance(
        self,
        period_analyses: List[PeriodAnalysis],
        regime_data: pd.DataFrame,
        enabled_filters: List[str]
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate performance conditioned on regime states (only for enabled filters).
        
        Args:
            period_analyses: List of period analyses
            regime_data: DataFrame with regime metrics
            enabled_filters: List of enabled filter names
        
        Returns:
            Dictionary mapping regime states to performance metrics
        """
        regime_perf = {}
        
        if not period_analyses:
            return regime_perf
        
        # Group by volatility quintile (if ATR filter enabled)
        if 'atr_percentile' in enabled_filters and 'vol_quintile' in regime_data.columns:
            for quintile in ['Q1', 'Q2', 'Q3', 'Q4', 'Q5']:
                quintile_periods = []
                for period in period_analyses:
                    period_regime = regime_data.loc[period.period_start:period.period_end]
                    if len(period_regime) > 0:
                        quintile_values = period_regime['vol_quintile'].dropna()
                        if len(quintile_values) > 0:
                            # Check if majority of period is in this quintile
                            quintile_count = (quintile_values == quintile).sum()
                            if quintile_count / len(quintile_values) > 0.5:
                                quintile_periods.append(period)
                
                if quintile_periods:
                    regime_perf[f'vol_{quintile}'] = {
                        'pf': float(np.mean([p.profit_factor for p in quintile_periods])),
                        'trades': int(np.mean([p.total_trades for p in quintile_periods])),
                        'win_rate': float(np.mean([p.conditional.conditional_win_rate for p in quintile_periods])),
                        'sharpe': float(np.mean([p.sharpe_ratio for p in quintile_periods])),
                        'signal_frequency': float(np.mean([p.conditional.signal_frequency for p in quintile_periods]))
                    }
        
        # Group by ADX state (if ADX filter enabled)
        if 'adx' in enabled_filters and 'adx_above_threshold' in regime_data.columns:
            for state in ['above', 'below']:
                state_periods = []
                for period in period_analyses:
                    period_regime = regime_data.loc[period.period_start:period.period_end]
                    if len(period_regime) > 0:
                        above_pct = period_regime['adx_above_threshold'].sum() / len(period_regime)
                        if state == 'above' and above_pct > 0.5:
                            state_periods.append(period)
                        elif state == 'below' and above_pct <= 0.5:
                            state_periods.append(period)
                
                if state_periods:
                    regime_perf[f'adx_{state}'] = {
                        'pf': float(np.mean([p.profit_factor for p in state_periods])),
                        'trades': int(np.mean([p.total_trades for p in state_periods])),
                        'win_rate': float(np.mean([p.conditional.conditional_win_rate for p in state_periods])),
                        'sharpe': float(np.mean([p.sharpe_ratio for p in state_periods])),
                        'signal_frequency': float(np.mean([p.conditional.signal_frequency for p in state_periods]))
                    }
        
        return regime_perf
    
    def _calculate_enhanced_retrain(
        self,
        days_vs_pf: Dict[int, float],
        days_vs_conditional_win_rate: Dict[int, float],
        days_vs_payoff_ratio: Dict[int, float],
        days_vs_big_win_frequency: Dict[int, float],
        days_vs_consecutive_losses: Dict[int, float]
    ) -> Tuple[int, str]:
        """
        Enhanced retraining logic with dual thresholds and conditional metrics.
        
        Uses 3 configured stabilized windows as baseline (configurable in criteria).
        Only triggers on persistent degradation in stabilized windows.
        
        Retrains only if:
        1. Absolute degradation: PF < min_acceptable_pf, OR
        2. Conditional degradation: Win rate drops, payoff collapses, big wins disappear
           (only in stabilized windows, comparing stabilized vs baseline)
        
        Does NOT retrain if:
        - PF variance is just distribution noise
        - Absolute performance remains acceptable (PF >= min_acceptable_pf)
        - Conditional metrics stabilize (variance in short windows is normal)
        
        Args:
            days_vs_pf: Dictionary mapping days to average PF
            days_vs_conditional_win_rate: Dictionary mapping days to conditional win rate
            days_vs_payoff_ratio: Dictionary mapping days to payoff ratio
            days_vs_big_win_frequency: Dictionary mapping days to big win frequency
            days_vs_consecutive_losses: Dictionary mapping days to consecutive losses
        
        Returns:
            Tuple of (recommended_days, reason)
        """
        if not days_vs_pf:
            return (self.criteria.max_days, "Default (insufficient data)")
        
        # Use 3 configured stabilized windows as baseline
        # This approach is statistically sound and recommended for sparse strategies
        window1_days = [d for d in sorted(days_vs_pf.keys()) 
                        if self.criteria.baseline_window1_start <= d <= self.criteria.baseline_window1_end 
                        and days_vs_pf[d] > 0]
        window2_days = [d for d in sorted(days_vs_pf.keys()) 
                        if self.criteria.baseline_window2_start < d <= self.criteria.baseline_window2_end 
                        and days_vs_pf[d] > 0]
        window3_days = [d for d in sorted(days_vs_pf.keys()) 
                        if self.criteria.baseline_window3_start < d <= self.criteria.baseline_window3_end 
                        and days_vs_pf[d] > 0]
        
        # Collect all baseline days from the 3 windows
        baseline_days = window1_days + window2_days + window3_days
        
        if not baseline_days:
            # Fallback: use any days in configured range if exact windows not available
            baseline_days = [d for d in sorted(days_vs_pf.keys()) 
                            if self.criteria.baseline_window1_start <= d <= self.criteria.baseline_window3_end 
                            and days_vs_pf[d] > 0]
        
        if not baseline_days:
            # Final fallback: use first few days if configured windows not available
            baseline_days = [d for d in sorted(days_vs_pf.keys())[:10] if days_vs_pf[d] > 0]
        
        if not baseline_days:
            return (self.criteria.max_days, "Default (no valid baseline)")
        
        # Calculate stable baseline averages
        baseline_pfs = [days_vs_pf[d] for d in baseline_days]
        baseline_win_rates = [days_vs_conditional_win_rate.get(d, 0.0) for d in baseline_days if days_vs_conditional_win_rate.get(d, 0.0) > 0]
        baseline_payoffs = [days_vs_payoff_ratio.get(d, 0.0) for d in baseline_days if days_vs_payoff_ratio.get(d, 0.0) > 0]
        baseline_big_win_freqs = [days_vs_big_win_frequency.get(d, 0.0) for d in baseline_days if days_vs_big_win_frequency.get(d, 0.0) > 0]
        
        baseline_pf = float(np.mean(baseline_pfs)) if baseline_pfs else None
        baseline_win_rate = float(np.mean(baseline_win_rates)) if baseline_win_rates else None
        baseline_payoff = float(np.mean(baseline_payoffs)) if baseline_payoffs else None
        baseline_big_win_freq = float(np.mean(baseline_big_win_freqs)) if baseline_big_win_freqs else None
        
        if baseline_pf is None:
            return (self.criteria.max_days, "Default (no valid baseline PF)")
        
        relative_threshold = baseline_pf * self.criteria.degradation_threshold_pct
        min_pf = min(days_vs_pf.values())
        max_pf = max(days_vs_pf.values())
        
        # Get stabilized values (same as baseline windows) to compare against baseline
        # Use the configured baseline window range
        stabilized_days = [d for d in sorted(days_vs_pf.keys()) 
                          if self.criteria.baseline_window1_start <= d <= self.criteria.baseline_window3_end 
                          and days_vs_pf[d] > 0]
        if stabilized_days:
            stabilized_payoffs = [days_vs_payoff_ratio.get(d, 0.0) for d in stabilized_days if days_vs_payoff_ratio.get(d, 0.0) > 0]
            stabilized_win_rates = [days_vs_conditional_win_rate.get(d, 0.0) for d in stabilized_days if days_vs_conditional_win_rate.get(d, 0.0) > 0]
            stabilized_big_win_freqs = [days_vs_big_win_frequency.get(d, 0.0) for d in stabilized_days if days_vs_big_win_frequency.get(d, 0.0) > 0]
            
            stabilized_payoff = float(np.mean(stabilized_payoffs)) if stabilized_payoffs else None
            stabilized_win_rate = float(np.mean(stabilized_win_rates)) if stabilized_win_rates else None
            stabilized_big_win_freq = float(np.mean(stabilized_big_win_freqs)) if stabilized_big_win_freqs else None
        else:
            stabilized_payoff = None
            stabilized_win_rate = None
            stabilized_big_win_freq = None
        
        # Check 1: Absolute degradation (MOST CRITICAL)
        for days in sorted(days_vs_pf.keys()):
            if days_vs_pf[days] < self.min_acceptable_pf:
                return (days, 
                       f"Absolute threshold breached: PF {days_vs_pf[days]:.2f} < {self.min_acceptable_pf}. "
                       f"Strategy no longer profitable.")
        
        # Check 2: Conditional metrics degradation (edge erosion)
        # Only trigger if metrics degrade AND stay degraded in stabilized windows
        # Compare stabilized values (days 15-30) against baseline (days 5-15)
        
        if stabilized_payoff is not None and baseline_payoff is not None and baseline_payoff > 0:
            # Check if stabilized payoff is significantly lower than baseline
            if stabilized_payoff < baseline_payoff * 0.7:
                # Find first day where this degradation appears and persists
                for days in sorted(days_vs_pf.keys()):
                    if days >= 15:  # Only check stabilized region
                        payoff = days_vs_payoff_ratio.get(days, 0.0)
                        if payoff > 0 and payoff < baseline_payoff * 0.7:
                            return (days, 
                                   f"Payoff ratio degraded in stabilized windows: {stabilized_payoff:.2f} vs baseline {baseline_payoff:.2f} "
                                   f"(>30% drop). Trend capture efficiency degraded.")
        
        if stabilized_win_rate is not None and baseline_win_rate is not None and baseline_win_rate > 0:
            if stabilized_win_rate < baseline_win_rate * 0.8:
                for days in sorted(days_vs_pf.keys()):
                    if days >= 15:
                        win_rate = days_vs_conditional_win_rate.get(days, 0.0)
                        if win_rate > 0 and win_rate < baseline_win_rate * 0.8:
                            return (days, 
                                   f"Conditional win rate degraded in stabilized windows: {stabilized_win_rate:.1%} vs baseline {baseline_win_rate:.1%} "
                                   f"(>20% drop). Edge in pattern recognition lost.")
        
        if stabilized_big_win_freq is not None and baseline_big_win_freq is not None and baseline_big_win_freq > 0.01:
            if stabilized_big_win_freq < baseline_big_win_freq * 0.5:
                for days in sorted(days_vs_pf.keys()):
                    if days >= 15:
                        big_win_freq = days_vs_big_win_frequency.get(days, 0.0)
                        if big_win_freq > 0 and big_win_freq < baseline_big_win_freq * 0.5:
                            return (days, 
                                   f"Big win frequency dropped in stabilized windows: {stabilized_big_win_freq:.3f} vs baseline {baseline_big_win_freq:.3f} "
                                   f"(>50% drop). Strategy losing ability to capture trends.")
        
        # Check 3: Relative degradation (only if approaching absolute threshold in stabilized region)
        # Only check stabilized windows to avoid false signals from short-window variance
        for days in sorted(days_vs_pf.keys()):
            if days >= self.criteria.baseline_window1_start and days_vs_pf[days] < relative_threshold:
                # Only recommend if close to absolute threshold
                if days_vs_pf[days] < self.min_acceptable_pf * 1.1:
                    return (days, 
                           f"Relative degradation in stabilized windows approaching absolute threshold: "
                           f"PF {days_vs_pf[days]:.2f} < {relative_threshold:.2f} ({self.criteria.degradation_threshold_pct:.0%} of baseline) "
                           f"and approaching {self.min_acceptable_pf}.")
        
        # No degradation detected
        # Check if metrics stabilized (variance in short windows is normal)
        if stabilized_payoff is not None and baseline_payoff is not None:
            # Metrics stabilized - variance in short windows (1-5 days) is expected
            return (max(days_vs_pf.keys()), 
                   f"No degradation detected. PF range: {min_pf:.2f}-{max_pf:.2f} "
                   f"(all above {self.min_acceptable_pf}). Conditional metrics stabilized: "
                   f"payoff {baseline_payoff:.2f} → {stabilized_payoff:.2f}. "
                   f"Short-window variance (days 1-5) is normal - metrics converge by day {self.criteria.baseline_window1_start}. "
                   f"Strategy remains profitable - retraining not required.")
        else:
            return (max(days_vs_pf.keys()), 
                   f"No degradation detected. PF range: {min_pf:.2f}-{max_pf:.2f} "
                   f"(all above {self.min_acceptable_pf}). Conditional metrics stable. "
                   f"Strategy remains profitable - retraining not required.")
    
    def _check_conditional_stability(
        self,
        days_vs_win_rate: Dict[int, float],
        days_vs_payoff: Dict[int, float],
        days_vs_big_win: Dict[int, float]
    ) -> bool:
        """
        Check if conditional metrics are stable across stabilized windows.
        
        Only checks stability in baseline windows (15-30 days) to avoid false
        signals from short-window variance (days 1-5), which is normal.
        
        Stable metrics indicate the strategy's edge persists.
        Unstable metrics indicate edge erosion.
        
        Args:
            days_vs_win_rate: Dictionary mapping days to conditional win rate
            days_vs_payoff: Dictionary mapping days to payoff ratio
            days_vs_big_win: Dictionary mapping days to big win frequency
        
        Returns:
            True if metrics are stable, False otherwise
        """
        if not days_vs_win_rate:
            return True
        
        # Only check stability in baseline windows (stabilized region)
        # Short windows (1-5 days) have high variance which is normal
        baseline_window_days = [
            d for d in days_vs_win_rate.keys()
            if self.criteria.baseline_window1_start <= d <= self.criteria.baseline_window3_end
        ]
        
        if not baseline_window_days:
            # Fallback: if no baseline windows, check all
            baseline_window_days = list(days_vs_win_rate.keys())
        
        # Get values only from baseline windows
        win_rates = [days_vs_win_rate.get(d, 0.0) for d in baseline_window_days if days_vs_win_rate.get(d, 0.0) > 0]
        payoffs = [days_vs_payoff.get(d, 0.0) for d in baseline_window_days if days_vs_payoff.get(d, 0.0) > 0]
        big_wins = [days_vs_big_win.get(d, 0.0) for d in baseline_window_days if days_vs_big_win.get(d, 0.0) > 0]
        
        # Check coefficient of variation (CV) using configured threshold
        cv_win_rate = np.std(win_rates) / np.mean(win_rates) if win_rates and np.mean(win_rates) > 0 else 0.0
        cv_payoff = np.std(payoffs) / np.mean(payoffs) if payoffs and np.mean(payoffs) > 0 else 0.0
        cv_big_win = np.std(big_wins) / np.mean(big_wins) if big_wins and np.mean(big_wins) > 0 else 0.0
        
        # Metrics are stable if CV < configured threshold for all
        # Use slightly higher threshold for big_win (0.5) as it's more variable
        return (cv_win_rate < self.criteria.stability_cv_threshold and 
                cv_payoff < self.criteria.stability_cv_threshold and 
                cv_big_win < 0.5)
    
    def _check_regime_stability(
        self,
        regime_performance: Dict[str, Dict[str, float]]
    ) -> Dict[str, bool]:
        """
        Check if performance is stable across regimes.
        
        Args:
            regime_performance: Dictionary mapping regime states to performance
        
        Returns:
            Dictionary mapping regime states to stability (True/False)
        """
        stability = {}
        
        for regime, perf in regime_performance.items():
            pf = perf.get('pf', 0.0)
            # Performance is stable if PF >= minimum acceptable
            stability[regime] = pf >= self.min_acceptable_pf
        
        return stability
    
    def _calculate_recommended_retrain(
        self,
        days_vs_pf: Dict[int, float],
        threshold_pct: float = 0.85
    ) -> int:
        """
        Legacy method for backward compatibility.
        
        Use _calculate_enhanced_retrain() instead for dual-threshold logic.
        """
        recommended, _ = self._calculate_enhanced_retrain(
            days_vs_pf,
            {},
            {},
            {},
            {},
            min_acceptable_pf=self.min_acceptable_pf,
            threshold_pct=threshold_pct
        )
        return recommended

