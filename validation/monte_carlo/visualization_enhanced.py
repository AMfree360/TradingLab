"""Enhanced Monte Carlo visualization with interactive plots and advanced features.

Features:
1. Equity curve overlays with probability cones (percentile bands)
2. Interactive Plotly plots for zoom/pan
3. Risk-return scatter plots
4. Cumulative distribution functions (CDFs)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import gaussian_kde
import json

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import plotly.express as px
    import plotly
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Install with: pip install plotly")

from engine.backtest_engine import BacktestResult, Trade
from .runner import MonteCarloSuiteResult


class EnhancedMonteCarloVisualizer:
    """Enhanced visualizer with interactive plots and advanced features."""
    
    def __init__(self):
        """Initialize enhanced visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for enhanced visualizations. Install with: pip install plotly")
        
        self.colors = {
            'observed': '#2E86AB',      # Blue
            'median': '#A23B72',        # Purple
            'percentile_5': '#F18F01',  # Orange
            'percentile_25': '#C73E1D', # Red-orange
            'percentile_50': '#A23B72', # Purple
            'percentile_75': '#6A994E', # Green
            'percentile_95': '#BC4749', # Red
            'background': '#E8E8E8',
        }
    
    def _reconstruct_equity_curves_from_pnls(
        self,
        pnls: np.ndarray,
        initial_capital: float,
        timestamps: pd.DatetimeIndex
    ) -> pd.Series:
        """Reconstruct equity curve from PnL array.
        
        Args:
            pnls: Array of trade P&Ls
            initial_capital: Starting capital
            timestamps: Timestamps for each trade
        
        Returns:
            Equity curve as pandas Series
        """
        if len(pnls) == 0:
            return pd.Series([initial_capital], index=timestamps[:1] if len(timestamps) > 0 else pd.DatetimeIndex([]))
        
        cumulative = np.cumsum(pnls)
        equity_values = initial_capital + cumulative
        
        # Use trade timestamps, or create evenly spaced if not available
        if len(timestamps) == len(equity_values):
            return pd.Series(equity_values, index=timestamps)
        else:
            # Create synthetic timestamps
            if len(timestamps) > 0:
                start_time = timestamps[0]
                end_time = timestamps[-1]
                synthetic_times = pd.date_range(start=start_time, end=end_time, periods=len(equity_values))
            else:
                synthetic_times = pd.date_range(start='2020-01-01', periods=len(equity_values), freq='D')
            return pd.Series(equity_values, index=synthetic_times)
    
    def plot_equity_curve_with_probability_cone(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        test_name: str = 'permutation',
        percentiles: List[float] = [5, 25, 50, 75, 95],
        n_sample_paths: int = 20
    ) -> go.Figure:
        """Plot equity curve with probability cone (percentile bands).
        
        Industry standard: Shows observed equity curve with MC distribution
        as percentile bands over time (fan chart/probability cone).
        
        Args:
            backtest_result: Original backtest result
            mc_result: Monte Carlo suite results
            test_name: Which MC test to visualize
            percentiles: Which percentiles to show
            n_sample_paths: Number of individual MC paths to show (for transparency)
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Get original equity curve
        original_equity = backtest_result.equity_curve
        if len(original_equity) == 0:
            return fig
        
        original_times = original_equity.index
        original_values = original_equity.values
        
        # Get MC test results
        if test_name == 'permutation':
            test_data = mc_result.permutation
            # Check if we have stored equity curves
            if isinstance(test_data, dict) and 'equity_curves' in test_data:
                equity_curves = test_data['equity_curves']
                if equity_curves and len(equity_curves) > 0:
                    return self._plot_equity_with_curves(
                        original_equity, equity_curves, percentiles, n_sample_paths, test_name
                    )
            
            # Fallback to end-of-period comparison
            return self._plot_equity_with_end_comparison(
                backtest_result, mc_result, test_name, percentiles
            )
        elif test_name == 'bootstrap':
            test_data = mc_result.bootstrap
            # Bootstrap resamples market returns, not strategy trades
            return self._plot_equity_with_end_comparison(
                backtest_result, mc_result, test_name, percentiles
            )
        elif test_name == 'randomized_entry':
            test_data = mc_result.randomized_entry
            # Randomized entry generates new trades
            return self._plot_equity_with_end_comparison(
                backtest_result, mc_result, test_name, percentiles
            )
        else:
            return fig
    
    def _plot_equity_with_curves(
        self,
        original_equity: pd.Series,
        equity_curves: List[pd.Series],
        percentiles: List[float],
        n_sample_paths: int,
        test_name: str
    ) -> go.Figure:
        """Plot equity curve with full probability cone from stored curves."""
        fig = go.Figure()
        
        original_times = original_equity.index
        original_values = original_equity.values
        
        # Remove duplicate labels from original_times if any
        if original_times.duplicated().any():
            # Keep first occurrence of each duplicate
            original_times = original_times[~original_times.duplicated(keep='first')]
            # Reindex original_equity to remove duplicates
            original_equity = original_equity.loc[original_times]
            original_values = original_equity.values
        
        # Align all equity curves to same time index
        # Use original times as reference
        aligned_curves = []
        for curve in equity_curves:
            # Remove duplicates from curve index if any
            if curve.index.duplicated().any():
                curve = curve[~curve.index.duplicated(keep='first')]
            
            # Reindex to original times, forward fill
            try:
                aligned = curve.reindex(original_times, method='ffill')
            except ValueError:
                # If reindex still fails, try without method (will use NaN for missing)
                aligned = curve.reindex(original_times)
                aligned = aligned.ffill().bfill()
            
            aligned_curves.append(aligned.values)
        
        if len(aligned_curves) == 0:
            return fig
        
        aligned_array = np.array(aligned_curves)
        
        # Calculate percentile bands at each time point
        percentile_bands = {}
        for pct in percentiles:
            percentile_bands[pct] = np.percentile(aligned_array, pct, axis=0)
        
        # Plot percentile bands as filled areas (probability cone)
        colors_map = {
            5: 'rgba(241, 143, 1, 0.1)',
            25: 'rgba(199, 62, 29, 0.15)',
            50: 'rgba(162, 59, 114, 0.2)',
            75: 'rgba(106, 153, 78, 0.15)',
            95: 'rgba(188, 71, 73, 0.1)'
        }
        
        # Plot bands from outer to inner
        for pct in reversed(percentiles):
            if pct == 50:
                continue  # Skip median, plot separately
            band_values = percentile_bands[pct]
            fig.add_trace(go.Scatter(
                x=original_times,
                y=band_values,
                mode='lines',
                name=f'{pct}th percentile',
                line=dict(width=0),
                fillcolor=colors_map.get(pct, 'rgba(128, 128, 128, 0.1)'),
                fill='tonexty' if pct != percentiles[-1] else 'tozeroy',
                showlegend=bool(pct in [5, 95]),
                hovertemplate=f'{pct}th percentile: $%{{y:,.2f}}<extra></extra>'
            ))
        
        # Plot median
        median_values = percentile_bands[50]
        fig.add_trace(go.Scatter(
            x=original_times,
            y=median_values,
            mode='lines',
            name='Median (50th percentile)',
            line=dict(color=self.colors['median'], width=2, dash='dash'),
            hovertemplate='Median: $%{y:,.2f}<extra></extra>'
        ))
        
        # Plot sample paths (transparent)
        n_samples = min(n_sample_paths, len(equity_curves))
        sample_indices = np.linspace(0, len(equity_curves) - 1, n_samples, dtype=int)
        for idx in sample_indices:
            curve = equity_curves[idx]
            # Remove duplicates from curve index if any
            if curve.index.duplicated().any():
                curve = curve[~curve.index.duplicated(keep='first')]
            
            # Reindex to original times, forward fill
            try:
                aligned = curve.reindex(original_times, method='ffill')
            except ValueError:
                # If reindex still fails, try without method (will use NaN for missing)
                aligned = curve.reindex(original_times)
                aligned = aligned.ffill().bfill()
            fig.add_trace(go.Scatter(
                x=original_times,
                y=aligned.values,
                mode='lines',
                name='Sample Path' if idx == sample_indices[0] else '',
                line=dict(color='rgba(128, 128, 128, 0.2)', width=1),
                showlegend=bool(idx == sample_indices[0]),
                hovertemplate='Sample Path: $%{y:,.2f}<extra></extra>'
            ))
        
        # Plot observed equity curve (bold, on top)
        fig.add_trace(go.Scatter(
            x=original_times,
            y=original_values,
            mode='lines',
            name='Observed Strategy',
            line=dict(color=self.colors['observed'], width=3),
            hovertemplate='Observed: $%{y:,.2f}<extra></extra>'
        ))
        
        test_display = test_name.replace('_', ' ').title()
        fig.update_layout(
            title=f'Equity Curve with Probability Cone: {test_display} Test',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def _plot_equity_with_end_comparison(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        test_name: str,
        percentiles: List[float]
    ) -> go.Figure:
        """Plot equity curve with end-of-period MC comparison.
        
        Since we don't have full equity curves from MC iterations yet,
        we show the observed curve with end-of-period distribution.
        """
        fig = go.Figure()
        
        original_equity = backtest_result.equity_curve
        original_times = original_equity.index
        original_values = original_equity.values
        
        # Plot observed equity curve
        fig.add_trace(go.Scatter(
            x=original_times,
            y=original_values,
            mode='lines',
            name='Observed Strategy',
            line=dict(color=self.colors['observed'], width=3),
            hovertemplate='Date: %{x}<br>Equity: $%{y:,.2f}<extra></extra>'
        ))
        
        # Get final PnL distribution
        test_data = getattr(mc_result, test_name, {})
        if isinstance(test_data, dict):
            final_pnl_dist = test_data.get('permuted_distributions', {}).get('final_pnl', np.array([]))
            if len(final_pnl_dist) == 0:
                final_pnl_dist = test_data.get('bootstrap_distributions', {}).get('final_pnl', np.array([]))
            if len(final_pnl_dist) == 0:
                final_pnl_dist = test_data.get('random_distributions', {}).get('final_pnl', np.array([]))
            
            if len(final_pnl_dist) > 0:
                finite_dist = final_pnl_dist[np.isfinite(final_pnl_dist)]
                if len(finite_dist) > 0:
                    # Calculate percentile final values
                    initial_capital = backtest_result.initial_capital
                    percentile_finals = [np.percentile(finite_dist, p) + initial_capital for p in percentiles]
                    
                    # Add horizontal lines at end showing percentile ranges
                    final_time = original_times[-1]
                    for pct, val in zip(percentiles, percentile_finals):
                        fig.add_hline(
                            y=val,
                            line_dash='dash',
                            line_color=self.colors.get(f'percentile_{int(pct)}', '#666'),
                            annotation_text=f'{pct}th percentile',
                            annotation_position='right'
                        )
        
        fig.update_layout(
            title=f'Equity Curve: Observed vs {test_name.replace("_", " ").title()} Monte Carlo',
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            hovermode='x unified',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def plot_cumulative_distribution_function(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str
    ) -> go.Figure:
        """Plot cumulative distribution function (CDF) with observed value.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric
            test_name: Name of MC test
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            return fig
        
        # Sort for CDF
        sorted_dist = np.sort(finite_dist)
        n = len(sorted_dist)
        
        # Calculate CDF
        cdf_values = np.arange(1, n + 1) / n
        
        # Plot CDF
        fig.add_trace(go.Scatter(
            x=sorted_dist,
            y=cdf_values * 100,  # Convert to percentage
            mode='lines',
            name='CDF',
            line=dict(color=self.colors['median'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(162, 59, 114, 0.2)",
            hovertemplate='Value: %{x:.2f}<br>CDF: %{y:.1f}%<extra></extra>'
        ))
        
        # Find CDF value at observed
        cdf_at_observed = (finite_dist <= observed_value).sum() / len(finite_dist) * 100.0
        
        # Plot observed value
        fig.add_trace(go.Scatter(
            x=[observed_value, observed_value],
            y=[0, cdf_at_observed],
            mode='lines',
            name=f'Observed ({cdf_at_observed:.1f}th percentile)',
            line=dict(color=self.colors['observed'], width=3, dash='dash'),
            hovertemplate=f'Observed: {observed_value:.2f}<br>Percentile: {cdf_at_observed:.1f}%<extra></extra>'
        ))
        
        # Add marker at observed value
        fig.add_trace(go.Scatter(
            x=[observed_value],
            y=[cdf_at_observed],
            mode='markers',
            name='Observed',
            marker=dict(
                color=self.colors['observed'],
                size=12,
                symbol='diamond',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f'Observed: {observed_value:.2f}<br>Percentile: {cdf_at_observed:.1f}%<extra></extra>'
        ))
        
        metric_display = metric_name.replace('_', ' ').title()
        test_display = test_name.replace('_', ' ').title()
        
        fig.update_layout(
            title=f'Cumulative Distribution Function: {metric_display} ({test_display})',
            xaxis_title=metric_display,
            yaxis_title='Cumulative Probability (%)',
            hovermode='x unified',
            template='plotly_white',
            height=500,
            showlegend=True
        )
        
        return fig
    
    def plot_risk_return_scatter(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        metrics: List[str] = ['final_pnl', 'sharpe_ratio']
    ) -> go.Figure:
        """Plot risk-return scatter for all MC iterations.
        
        Shows risk (e.g., max drawdown or volatility) vs return (e.g., final PnL or Sharpe)
        for all MC iterations, with observed strategy marked.
        
        Args:
            backtest_result: Original backtest result
            mc_result: Monte Carlo suite results
            metrics: Metrics to use for risk and return
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # For now, use final_pnl as return and sharpe_ratio as risk-adjusted return
        # In a full implementation, we'd calculate max drawdown for each MC iteration
        
        test_names = ['permutation', 'bootstrap', 'randomized_entry']
        colors_map = {
            'permutation': '#2E86AB',
            'bootstrap': '#A23B72',
            'randomized_entry': '#6A994E'
        }
        
        for test_name in test_names:
            test_data = getattr(mc_result, test_name, {})
            if isinstance(test_data, dict):
                # Get distributions
                if test_name == 'permutation':
                    dist_key = 'permuted_distributions'
                elif test_name == 'bootstrap':
                    dist_key = 'bootstrap_distributions'
                else:
                    dist_key = 'random_distributions'
                
                distributions = test_data.get(dist_key, {})
                
                if 'final_pnl' in distributions and 'sharpe_ratio' in distributions:
                    final_pnl_dist = np.array(distributions['final_pnl'])
                    sharpe_dist = np.array(distributions['sharpe_ratio'])
                    
                    # Remove invalid values
                    mask = np.isfinite(final_pnl_dist) & np.isfinite(sharpe_dist)
                    final_pnl_clean = final_pnl_dist[mask]
                    sharpe_clean = sharpe_dist[mask]
                    
                    if len(final_pnl_clean) > 0:
                        test_display = test_name.replace('_', ' ').title()
                        fig.add_trace(go.Scatter(
                            x=sharpe_clean,
                            y=final_pnl_clean,
                            mode='markers',
                            name=test_display,
                            marker=dict(
                                color=colors_map.get(test_name, '#666'),
                                size=4,
                                opacity=0.6
                            ),
                            hovertemplate=f'{test_display}<br>Sharpe: %{{x:.2f}}<br>Final PnL: $%{{y:,.2f}}<extra></extra>'
                        ))
        
        # Add observed strategy
        from metrics.metrics import calculate_enhanced_metrics
        enhanced = calculate_enhanced_metrics(backtest_result)
        observed_sharpe = enhanced.get('sharpe_ratio', 0.0)
        observed_pnl = backtest_result.total_pnl
        
        fig.add_trace(go.Scatter(
            x=[observed_sharpe],
            y=[observed_pnl],
            mode='markers',
            name='Observed Strategy',
            marker=dict(
                color=self.colors['observed'],
                size=15,
                symbol='star',
                line=dict(width=2, color='white')
            ),
            hovertemplate=f'Observed Strategy<br>Sharpe: {observed_sharpe:.2f}<br>Final PnL: ${observed_pnl:,.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Risk-Return Scatter: All Monte Carlo Iterations',
            xaxis_title='Sharpe Ratio (Risk-Adjusted Return)',
            yaxis_title='Final PnL ($)',
            hovermode='closest',
            template='plotly_white',
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_interactive_distribution(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str,
        show_cdf: bool = True
    ) -> go.Figure:
        """Plot interactive distribution histogram with optional CDF overlay.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric
            test_name: Name of MC test
            show_cdf: Whether to overlay CDF
        
        Returns:
            Plotly Figure with subplots
        """
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            return go.Figure()
        
        # Create subplots
        if show_cdf:
            fig = make_subplots(
                rows=2, cols=1,
                row_heights=[0.7, 0.3],
                subplot_titles=('Distribution Histogram', 'Cumulative Distribution Function'),
                vertical_spacing=0.1
            )
        else:
            fig = go.Figure()
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(finite_dist, percentiles)
        
        # Plot histogram
        fig.add_trace(go.Histogram(
            x=finite_dist,
            nbinsx=30,
            name='MC Distribution',
            marker_color=self.colors['background'],
            marker_line_color='black',
            marker_line_width=1,
            opacity=0.7,
            hovertemplate='Value: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ), row=1 if show_cdf else None, col=1 if show_cdf else None)
        
        # Add percentile lines
        for pct, val, color_key in zip(
            percentiles,
            percentile_values,
            ['percentile_5', 'percentile_25', 'percentile_50', 'percentile_75', 'percentile_95']
        ):
            fig.add_vline(
                x=val,
                line_dash='dash',
                line_color=self.colors[color_key],
                annotation_text=f'{pct}th',
                annotation_position='top',
                row=1 if show_cdf else None,
                col=1 if show_cdf else None
            )
        
        # Add observed value
        fig.add_vline(
            x=observed_value,
            line_color=self.colors['observed'],
            line_width=3,
            annotation_text='Observed',
            annotation_position='top',
            row=1 if show_cdf else None,
            col=1 if show_cdf else None
        )
        
        # Add CDF if requested
        if show_cdf:
            sorted_dist = np.sort(finite_dist)
            n = len(sorted_dist)
            cdf_values = np.arange(1, n + 1) / n * 100
            
            fig.add_trace(go.Scatter(
                x=sorted_dist,
                y=cdf_values,
                mode='lines',
                name='CDF',
                line=dict(color=self.colors['median'], width=2),
                fill='tozeroy',
                fillcolor=f"rgba(162, 59, 114, 0.2)",
                hovertemplate='Value: %{x:.2f}<br>CDF: %{y:.1f}%<extra></extra>'
            ), row=2, col=1)
            
            # Mark observed on CDF
            cdf_at_observed = (finite_dist <= observed_value).sum() / len(finite_dist) * 100.0
            fig.add_trace(go.Scatter(
                x=[observed_value],
                y=[cdf_at_observed],
                mode='markers',
                name='Observed',
                marker=dict(
                    color=self.colors['observed'],
                    size=12,
                    symbol='diamond'
                ),
                hovertemplate=f'Observed: {observed_value:.2f}<br>Percentile: {cdf_at_observed:.1f}%<extra></extra>'
            ), row=2, col=1)
        
        metric_display = metric_name.replace('_', ' ').title()
        test_display = test_name.replace('_', ' ').title()
        percentile_rank = (finite_dist < observed_value).sum() / len(finite_dist) * 100.0
        
        fig.update_layout(
            title=f'{metric_display} Distribution: {test_display} Test<br><sub>Observed: {observed_value:.2f} | Percentile Rank: {percentile_rank:.1f}%</sub>',
            template='plotly_white',
            height=700 if show_cdf else 500,
            showlegend=False
        )
        
        if show_cdf:
            fig.update_xaxes(title_text=metric_display, row=1, col=1)
            fig.update_yaxes(title_text='Frequency', row=1, col=1)
            fig.update_xaxes(title_text=metric_display, row=2, col=1)
            fig.update_yaxes(title_text='CDF (%)', row=2, col=1)
        else:
            fig.update_xaxes(title_text=metric_display)
            fig.update_yaxes(title_text='Frequency')
        
        return fig
    
    def figure_to_html(self, fig: go.Figure) -> str:
        """Convert Plotly figure to HTML string.
        
        Args:
            fig: Plotly figure
        
        Returns:
            HTML string with embedded plot
        """
        return fig.to_html(include_plotlyjs='cdn', div_id=None)
    
    def figure_to_json(self, fig: go.Figure) -> str:
        """Convert Plotly figure to JSON for embedding.
        
        Args:
            fig: Plotly figure
        
        Returns:
            JSON string
        """
        try:
            from plotly.utils import PlotlyJSONEncoder
            return json.dumps(fig.to_dict(), cls=PlotlyJSONEncoder)
        except ImportError:
            return json.dumps(fig.to_dict())

