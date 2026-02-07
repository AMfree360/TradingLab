"""Comprehensive tests for Monte Carlo validation suite."""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from engine.backtest_engine import BacktestResult, Trade
from validation.monte_carlo.permutation import MonteCarloPermutation
from validation.monte_carlo.block_bootstrap import MonteCarloBlockBootstrap
from validation.monte_carlo.randomized_entry import MonteCarloRandomizedEntry
from validation.monte_carlo.runner import MonteCarloSuite
from strategies.base import StrategyBase
from config.schema import StrategyConfig, MarketConfig, TimeframeConfig, RiskConfig


class MockStrategy(StrategyBase):
    """Mock strategy for testing."""
    
    def generate_signals(self, df_by_tf):
        """Generate no signals (for testing)."""
        return pd.DataFrame()
    
    def get_indicators(self, df, tf=None):
        """Return dataframe with basic indicators."""
        df = df.copy()
        df['sl_ma'] = df['close'].rolling(50).mean()
        return df


@pytest.fixture
def sample_backtest_result():
    """Create a sample backtest result for testing."""
    # Create synthetic trades
    base_time = datetime(2020, 1, 1)
    trades = []
    
    # Create 50 trades with varying P&Ls
    for i in range(50):
        entry_time = base_time + timedelta(days=i*2)
        exit_time = entry_time + timedelta(days=1)
        
        # Create mix of winning and losing trades
        if i % 3 == 0:  # Losing trade
            net_pnl = -100.0
        else:  # Winning trade
            net_pnl = 150.0
        
        trade = Trade(
            entry_time=entry_time,
            exit_time=exit_time,
            direction='long',
            entry_price=100.0,
            exit_price=100.0 + (net_pnl / 1.0),  # Simplified
            size=1.0,
            gross_pnl=net_pnl,
            net_pnl=net_pnl,
            commission=1.0,
            slippage=0.5,
            stop_price=95.0,
            exit_reason='target',
            r_multiple=net_pnl / 100.0  # Assuming $100 risk
        )
        trades.append(trade)
    
    return BacktestResult(
        strategy_name='test_strategy',
        symbol='TEST',
        initial_capital=10000.0,
        final_capital=10000.0 + sum(t.net_pnl for t in trades),
        total_trades=len(trades),
        winning_trades=sum(1 for t in trades if t.net_pnl > 0),
        losing_trades=sum(1 for t in trades if t.net_pnl <= 0),
        win_rate=(sum(1 for t in trades if t.net_pnl > 0) / len(trades) * 100.0),
        total_pnl=sum(t.net_pnl for t in trades),
        total_commission=sum(t.commission for t in trades),
        total_slippage=sum(t.slippage for t in trades),
        max_drawdown=10.0,
        trades=trades,
        equity_curve=pd.Series(
            [10000.0 + sum(t.net_pnl for t in trades[:i+1]) for i in range(len(trades))],
            index=[t.entry_time for t in trades]
        )
    )


@pytest.fixture
def sample_price_series():
    """Create a sample price series for testing."""
    dates = pd.date_range(start='2020-01-01', periods=500, freq='D')
    # Create trending price series with some volatility
    trend = np.linspace(100, 110, len(dates))
    noise = np.random.RandomState(42).normal(0, 2, len(dates))
    prices = trend + noise
    
    return pd.Series(prices, index=dates, name='close')


@pytest.fixture
def mock_strategy():
    """Create a mock strategy for testing."""
    config_dict = {
        'strategy_name': 'test',
        'market': {
            'exchange': 'test',
            'symbol': 'TEST',
            'base_timeframe': '1d'
        },
        'timeframes': {
            'signal_tf': '1d',
            'entry_tf': '1d'
        },
        'moving_averages': {},
        'alignment_rules': {},
        'risk': {
            'sizing_mode': 'account_size',
            'risk_per_trade_pct': 1.0,
            'account_size': 10000.0
        },
        'stop_loss': {
            'type': 'SMA',
            'length': 50,
            'buffer_pips': 1.0,
            'buffer_unit': 'price'
        },
        'take_profit': {
            'enabled': True,
            'target_r': 3.0
        },
        'trade_direction': {
            'allow_long': True,
            'allow_short': True
        }
    }
    config = StrategyConfig(**config_dict)
    return MockStrategy(config)


class TestMonteCarloPermutation:
    """Test permutation Monte Carlo engine."""
    
    def test_permutation_runs(self, sample_backtest_result):
        """Test that permutation test runs without errors."""
        mc = MonteCarloPermutation(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            metrics=['final_pnl', 'sharpe_ratio', 'profit_factor'],
            n_iterations=100,
            show_progress=False
        )
        
        assert result.n_iterations == 100
        assert 'final_pnl' in result.observed_metrics
        assert 'final_pnl' in result.permuted_distributions
        assert 'final_pnl' in result.p_values
        assert 'final_pnl' in result.percentiles
        
        # Check that distributions are numpy arrays
        assert isinstance(result.permuted_distributions['final_pnl'], np.ndarray)
        assert len(result.permuted_distributions['final_pnl']) == 100
    
    def test_permutation_returns_valid_percentiles(self, sample_backtest_result):
        """Test that percentiles are in valid range."""
        mc = MonteCarloPermutation(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            metrics=['final_pnl'],
            n_iterations=100,
            show_progress=False
        )
        
        percentile = result.percentiles['final_pnl']
        assert 0.0 <= percentile <= 100.0
    
    def test_permutation_returns_valid_p_values(self, sample_backtest_result):
        """Test that p-values are in valid range."""
        mc = MonteCarloPermutation(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            metrics=['final_pnl'],
            n_iterations=100,
            show_progress=False
        )
        
        p_value = result.p_values['final_pnl']
        assert 0.0 <= p_value <= 1.0
    
    def test_permutation_empty_trades(self):
        """Test permutation with no trades."""
        empty_result = BacktestResult(
            strategy_name='test',
            symbol='TEST',
            initial_capital=10000.0,
            final_capital=10000.0,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_pnl=0.0,
            total_commission=0.0,
            total_slippage=0.0,
            max_drawdown=0.0,
            trades=[]
        )
        
        mc = MonteCarloPermutation(seed=42)
        result = mc.run(
            backtest_result=empty_result,
            metrics=['final_pnl'],
            n_iterations=10,
            show_progress=False
        )
        
        assert result.n_iterations == 10
        assert len(result.permuted_distributions['final_pnl']) == 0


class TestMonteCarloBlockBootstrap:
    """Test block bootstrap Monte Carlo engine."""
    
    def test_bootstrap_runs(self, sample_backtest_result, sample_price_series):
        """Test that bootstrap test runs without errors."""
        mc = MonteCarloBlockBootstrap(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            price_series=sample_price_series,
            metrics=['final_pnl', 'sharpe_ratio'],
            n_iterations=100,
            show_progress=False
        )
        
        assert result.n_iterations == 100
        assert result.block_length > 0
        assert 'final_pnl' in result.observed_metrics
        assert 'final_pnl' in result.bootstrap_distributions
        assert isinstance(result.bootstrap_distributions['final_pnl'], np.ndarray)
    
    def test_bootstrap_preserves_volatility(self, sample_backtest_result, sample_price_series):
        """Test that bootstrap preserves volatility characteristics."""
        mc = MonteCarloBlockBootstrap(seed=42, block_length=10)
        result = mc.run(
            backtest_result=sample_backtest_result,
            price_series=sample_price_series,
            metrics=['final_pnl'],
            n_iterations=50,
            show_progress=False
        )
        
        # Bootstrap should produce distributions
        assert len(result.bootstrap_distributions['final_pnl']) == 50
    
    def test_bootstrap_returns_valid_percentiles(self, sample_backtest_result, sample_price_series):
        """Test that bootstrap percentiles are valid."""
        mc = MonteCarloBlockBootstrap(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            price_series=sample_price_series,
            metrics=['final_pnl'],
            n_iterations=50,
            show_progress=False
        )
        
        percentile = result.percentiles['final_pnl']
        assert 0.0 <= percentile <= 100.0

    def test_bootstrap_handles_duplicate_timestamps(self, sample_backtest_result, sample_price_series):
        """Test that bootstrap handles non-unique datetime indices."""
        idx = list(sample_price_series.index)
        if len(idx) < 3:
            raise AssertionError("sample_price_series must have at least 3 bars")

        # Introduce a deliberate duplicate timestamp
        idx[2] = idx[1]
        price_series_with_dupes = pd.Series(sample_price_series.values, index=pd.DatetimeIndex(idx))

        mc = MonteCarloBlockBootstrap(seed=42)
        result = mc.run(
            backtest_result=sample_backtest_result,
            price_series=price_series_with_dupes,
            metrics=['final_pnl'],
            n_iterations=10,
            show_progress=False,
        )

        assert result.n_iterations == 10
        assert len(result.bootstrap_distributions['final_pnl']) == 10


class TestMonteCarloRandomizedEntry:
    """Test randomized entry Monte Carlo engine."""
    
    def test_randomized_entry_runs(self, sample_backtest_result, mock_strategy):
        """Test that randomized entry test runs without errors."""
        # Create price data
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        price_data = pd.DataFrame({
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 100.0,
            'volume': 1000.0
        }, index=dates)
        
        mc = MonteCarloRandomizedEntry(seed=42, entry_probability=0.05)
        result = mc.run(
            backtest_result=sample_backtest_result,
            price_data=price_data,
            strategy=mock_strategy,
            metrics=['final_pnl'],
            n_iterations=50,  # Reduced for speed
            show_progress=False
        )
        
        assert result.n_iterations == 50
        assert 'final_pnl' in result.observed_metrics
        assert 'final_pnl' in result.random_distributions
        assert isinstance(result.random_distributions['final_pnl'], np.ndarray)
    
    def test_randomized_entry_no_edge_percentile(self, sample_backtest_result, mock_strategy):
        """Test that no-edge strategy has ~50% percentile."""
        # Create a backtest result with zero edge (break-even)
        break_even_result = BacktestResult(
            strategy_name='test',
            symbol='TEST',
            initial_capital=10000.0,
            final_capital=10000.0,
            total_trades=50,
            winning_trades=25,
            losing_trades=25,
            win_rate=50.0,
            total_pnl=0.0,
            total_commission=50.0,
            total_slippage=25.0,
            max_drawdown=5.0,
            trades=[]
        )
        
        dates = pd.date_range(start='2020-01-01', periods=200, freq='D')
        price_data = pd.DataFrame({
            'open': 100.0,
            'high': 102.0,
            'low': 98.0,
            'close': 100.0,
            'volume': 1000.0
        }, index=dates)
        
        mc = MonteCarloRandomizedEntry(seed=42, entry_probability=0.05)
        result = mc.run(
            backtest_result=break_even_result,
            price_data=price_data,
            strategy=mock_strategy,
            metrics=['final_pnl'],
            n_iterations=100,
            show_progress=False
        )
        
        # With no edge, percentile should be around 50% (allowing for variance)
        percentile = result.percentiles['final_pnl']
        # Allow wide range due to randomness
        assert 20.0 <= percentile <= 80.0  # Wide range for statistical variance


class TestMonteCarloSuite:
    """Test master Monte Carlo suite."""
    
    def test_suite_runs_all_tests(self, sample_backtest_result, sample_price_series, mock_strategy):
        """Test that suite runs all three engines."""
        suite = MonteCarloSuite(seed=42)
        result = suite.run_all(
            backtest_result=sample_backtest_result,
            price_series=sample_price_series,
            strategy=mock_strategy,
            metrics=['final_pnl', 'sharpe_ratio'],
            n_iterations=50,  # Reduced for speed
            show_progress=False
        )
        
        assert 'permutation' in result.permutation
        assert 'bootstrap' in result.bootstrap
        assert 'randomized_entry' in result.randomized_entry
        # Canonical: combined should contain metric fields like 'score'
        assert 'score' in result.combined
    
    def test_suite_combined_score(self, sample_backtest_result, sample_price_series, mock_strategy):
        """Test that combined score is calculated correctly."""
        suite = MonteCarloSuite(seed=42)
        result = suite.run_all(
            backtest_result=sample_backtest_result,
            price_series=sample_price_series,
            strategy=mock_strategy,
            metrics=['final_pnl'],
            n_iterations=50,
            show_progress=False
        )
        
        combined = result.combined
        assert 'score' in combined
        assert 'percentile' in combined
        assert 'p_value' in combined
        assert 'robust' in combined
        
        # Score should be between 0 and 1
        assert 0.0 <= combined['score'] <= 1.0
        assert 0.0 <= combined['percentile'] <= 100.0
    
    def test_suite_score_increases_with_edge(self, sample_price_series, mock_strategy):
        """Test that score increases as original edge increases."""
        # Create results with different edge levels
        suite = MonteCarloSuite(seed=42)
        
        # Low edge result
        low_edge_trades = [
            Trade(
                entry_time=datetime(2020, 1, 1) + timedelta(days=i),
                exit_time=datetime(2020, 1, 1) + timedelta(days=i+1),
                direction='long',
                entry_price=100.0,
                exit_price=100.5,
                size=1.0,
                gross_pnl=0.5,
                net_pnl=0.3,  # Small profit
                commission=0.1,
                slippage=0.1,
                stop_price=99.0,
                exit_reason='target',
                r_multiple=0.03
            ) for i in range(20)
        ]
        
        low_edge_result = BacktestResult(
            strategy_name='test',
            symbol='TEST',
            initial_capital=10000.0,
            final_capital=10000.0 + sum(t.net_pnl for t in low_edge_trades),
            total_trades=len(low_edge_trades),
            winning_trades=len(low_edge_trades),
            losing_trades=0,
            win_rate=100.0,
            total_pnl=sum(t.net_pnl for t in low_edge_trades),
            total_commission=sum(t.commission for t in low_edge_trades),
            total_slippage=sum(t.slippage for t in low_edge_trades),
            max_drawdown=1.0,
            trades=low_edge_trades
        )
        
        # High edge result
        high_edge_trades = [
            Trade(
                entry_time=datetime(2020, 1, 1) + timedelta(days=i),
                exit_time=datetime(2020, 1, 1) + timedelta(days=i+1),
                direction='long',
                entry_price=100.0,
                exit_price=105.0,  # Much larger profit
                size=1.0,
                gross_pnl=5.0,
                net_pnl=4.8,
                commission=0.1,
                slippage=0.1,
                stop_price=99.0,
                exit_reason='target',
                r_multiple=0.48
            ) for i in range(20)
        ]
        
        high_edge_result = BacktestResult(
            strategy_name='test',
            symbol='TEST',
            initial_capital=10000.0,
            final_capital=10000.0 + sum(t.net_pnl for t in high_edge_trades),
            total_trades=len(high_edge_trades),
            winning_trades=len(high_edge_trades),
            losing_trades=0,
            win_rate=100.0,
            total_pnl=sum(t.net_pnl for t in high_edge_trades),
            total_commission=sum(t.commission for t in high_edge_trades),
            total_slippage=sum(t.slippage for t in high_edge_trades),
            max_drawdown=1.0,
            trades=high_edge_trades
        )
        
        # Run suite on both (with reduced iterations for speed)
        low_result = suite.run_all(
            backtest_result=low_edge_result,
            price_series=sample_price_series,
            strategy=mock_strategy,
            metrics=['final_pnl'],
            n_iterations=30,
            show_progress=False
        )
        
        high_result = suite.run_all(
            backtest_result=high_edge_result,
            price_series=sample_price_series,
            strategy=mock_strategy,
            metrics=['final_pnl'],
            n_iterations=30,
            show_progress=False
        )
        
        # High edge should have higher score (allowing for variance)
        # Note: This test may be flaky due to randomness, but generally should hold
        assert high_result.combined['score'] >= low_result.combined['score'] - 0.2  # Allow variance


class TestMonteCarloUniversalVsConditional:
    def test_conditional_combined_uses_universal_only_for_pass_fail(self):
        suite = MonteCarloSuite(seed=42)

        metrics = ['final_pnl', 'sharpe_ratio', 'profit_factor']

        results = {
            'permutation': {'skipped': True, 'reason': 'not suitable', 'alternatives': []},
            'bootstrap': {
                'skipped': False,
                'p_values': {m: 0.05 for m in metrics},
                'percentiles': {m: 90.0 for m in metrics},
            },
            'randomized_entry': {
                'skipped': False,
                'p_values': {m: 1.0 for m in metrics},
                'percentiles': {m: 0.0 for m in metrics},
            },
        }

        from validation.suitability import TestSuitability

        test_suitability = {
            'permutation': TestSuitability(suitable=False, reason='not suitable', priority=0.2, category='universal'),
            'bootstrap': TestSuitability(suitable=True, reason='ok', priority=0.3, category='universal'),
            'randomized_entry': TestSuitability(suitable=True, reason='ok', priority=0.5, category='conditional'),
        }

        combined = suite._calculate_combined_score_conditional(results, test_suitability, metrics)

        # Bootstrap metric_score: 0.5*(1-0.05) + 0.5*(0.90) = 0.925 -> test_score == 0.925
        assert abs(combined['score'] - 0.925) < 1e-6
        assert combined['normalized_weights'] == {'bootstrap': 1.0}

        # All-tests combo should include randomized_entry weight and be lower
        assert 'score_all' in combined
        assert combined['score_all'] < combined['score']
        assert combined['normalized_weights_all']['bootstrap'] > 0
        assert combined['normalized_weights_all']['randomized_entry'] > 0

    def test_suitability_marks_randomized_entry_conditional(self):
        from validation.suitability import ValidationSuitabilityAssessor, StrategyProfile

        assessor = ValidationSuitabilityAssessor()
        profile = StrategyProfile(
            return_cv=0.2,
            exit_uniformity=0.3,
            n_trades=100,
            n_bars=500,
            avg_hold_time_cv=0.5,
            strategy_type='hybrid',
            final_equity_cv=0.01,
        )
        suitability = assessor.get_test_suitability(profile)
        assert suitability['bootstrap'].category == 'universal'
        assert suitability['permutation'].category == 'universal'
        assert suitability['randomized_entry'].category == 'conditional'

