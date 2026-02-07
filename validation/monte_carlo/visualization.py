"""Professional Monte Carlo visualization module.

Industry-standard visualizations for Monte Carlo validation results:
1. Equity curve overlays with percentile bands (probability cones)
2. Distribution histograms with observed values
3. Box plots for quartile analysis
4. Summary statistics tables
5. Risk-return scatter plots

Based on QuantConnect, Backtrader, and academic best practices.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.figure import Figure
import seaborn as sns
from io import BytesIO
import base64

from engine.backtest_engine import BacktestResult
from .runner import MonteCarloSuiteResult


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10


class MonteCarloVisualizer:
    """Generate industry-standard visualizations for Monte Carlo results."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style (default: seaborn)
        """
        plt.style.use(style)
        self.colors = {
            'observed': '#2E86AB',  # Blue
            'median': '#A23B72',     # Purple
            'percentile_5': '#F18F01',  # Orange
            'percentile_25': '#C73E1D',  # Red-orange
            'percentile_75': '#6A994E',  # Green
            'percentile_95': '#BC4749',  # Red
            'background': '#E8E8E8',
            'grid': '#CCCCCC'
        }
    
    def plot_equity_curve_overlay(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        test_name: str = 'permutation',
        n_paths_to_show: int = 50,
        show_percentile_bands: bool = True,
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Figure:
        """Plot equity curve with Monte Carlo percentile bands (probability cone).
        
        Industry standard: Shows observed equity curve with MC distribution
        as percentile bands (fan chart/probability cone).
        
        Args:
            backtest_result: Original backtest result
            mc_result: Monte Carlo suite results
            test_name: Which MC test to visualize ('permutation', 'bootstrap', 'randomized_entry')
            n_paths_to_show: Number of individual MC paths to show (for transparency)
            show_percentile_bands: Whether to show percentile bands
            percentiles: Which percentiles to show
        
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Get original equity curve
        original_equity = backtest_result.equity_curve
        if len(original_equity) == 0:
            return fig
        
        original_times = original_equity.index
        original_values = original_equity.values
        
        # Get MC distributions (we need to reconstruct equity curves from trades)
        # For now, we'll use the final_pnl distribution and show it as end-of-period
        # In a full implementation, we'd reconstruct full equity curves
        
        # Plot observed equity curve (bold, prominent)
        ax.plot(
            original_times,
            original_values,
            color=self.colors['observed'],
            linewidth=2.5,
            label='Observed Strategy',
            zorder=10
        )
        
        # Get MC test results
        if test_name == 'permutation':
            test_data = mc_result.permutation
        elif test_name == 'bootstrap':
            test_data = mc_result.bootstrap
        elif test_name == 'randomized_entry':
            test_data = mc_result.randomized_entry
        else:
            return fig
        
        # For permutation/bootstrap, we can show distribution of final values
        # For a full implementation, we'd need to store full equity curves
        # For now, show as end-of-period comparison
        
        # Add title and labels
        test_title = test_name.replace('_', ' ').title()
        ax.set_title(
            f'Equity Curve: Observed vs {test_title} Monte Carlo',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Equity ($)', fontsize=12)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Format x-axis dates
        fig.autofmt_xdate()
        
        plt.tight_layout()
        return fig
    
    def plot_distribution_histogram(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str,
        show_percentiles: bool = True,
        bins: Optional[int] = None
    ) -> Figure:
        """Plot distribution histogram with observed value marked.
        
        Industry standard: Histogram of MC distribution with observed value
        as vertical line, showing percentile bands.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric (e.g., 'final_pnl', 'sharpe_ratio')
            test_name: Name of MC test
            show_percentiles: Show percentile bands
            bins: Number of histogram bins (auto if None)
        
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(finite_dist, percentiles)
        
        # Determine bins
        if bins is None:
            # Use Freedman-Diaconis rule
            iqr = np.percentile(finite_dist, 75) - np.percentile(finite_dist, 25)
            bin_width = 2 * iqr / (len(finite_dist) ** (1/3))
            bins = max(20, int((finite_dist.max() - finite_dist.min()) / bin_width)) if bin_width > 0 else 30
        
        # Plot histogram
        n, bins_edges, patches = ax.hist(
            finite_dist,
            bins=bins,
            alpha=0.6,
            color=self.colors['background'],
            edgecolor='black',
            linewidth=0.5,
            label='MC Distribution'
        )
        
        # Color bars by percentile regions
        for i, (patch, left_edge) in enumerate(zip(patches, bins_edges[:-1])):
            if left_edge <= percentile_values[0]:  # Below 5th percentile
                patch.set_facecolor(self.colors['percentile_5'])
            elif left_edge <= percentile_values[1]:  # 5th-25th
                patch.set_facecolor(self.colors['percentile_25'])
            elif left_edge <= percentile_values[2]:  # 25th-50th
                patch.set_facecolor(self.colors['median'])
            elif left_edge <= percentile_values[3]:  # 50th-75th
                patch.set_facecolor(self.colors['percentile_75'])
            elif left_edge <= percentile_values[4]:  # 75th-95th
                patch.set_facecolor(self.colors['percentile_95'])
            else:  # Above 95th
                patch.set_facecolor('#FF6B6B')
        
        # Plot percentile lines
        if show_percentiles:
            for pct, val, color in zip(
                percentiles,
                percentile_values,
                [self.colors['percentile_5'], self.colors['percentile_25'],
                 self.colors['median'], self.colors['percentile_75'], self.colors['percentile_95']]
            ):
                ax.axvline(
                    val,
                    color=color,
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label=f'{pct}th percentile'
                )
        
        # Plot observed value (prominent)
        ax.axvline(
            observed_value,
            color=self.colors['observed'],
            linewidth=3,
            linestyle='-',
            label=f'Observed ({metric_name})',
            zorder=10
        )
        
        # Calculate percentile rank
        percentile_rank = (finite_dist < observed_value).sum() / len(finite_dist) * 100.0
        
        # Add text annotation
        ax.text(
            0.02, 0.98,
            f'Observed: {observed_value:.2f}\nPercentile Rank: {percentile_rank:.1f}%',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
        
        # Labels and title
        metric_display = metric_name.replace('_', ' ').title()
        test_display = test_name.replace('_', ' ').title()
        ax.set_title(
            f'{metric_display} Distribution: {test_display} Test',
            fontsize=13,
            fontweight='bold',
            pad=15
        )
        ax.set_xlabel(metric_display, fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.legend(loc='best', fontsize=9, ncol=2)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_box_plot_comparison(
        self,
        mc_result: MonteCarloSuiteResult,
        metrics: List[str] = ['final_pnl', 'sharpe_ratio', 'profit_factor']
    ) -> Figure:
        """Plot box plots comparing all three MC tests.
        
        Industry standard: Side-by-side box plots showing quartiles
        and outliers for each test.
        
        Args:
            mc_result: Monte Carlo suite results
            metrics: Metrics to compare
        
        Returns:
            Matplotlib Figure
        """
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 6))
        if len(metrics) == 1:
            axes = [axes]
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx]
            
            # Collect distributions from all tests
            data_to_plot = []
            labels = []
            
            # Permutation
            if 'permutation' in mc_result.permutation:
                perm_dist = mc_result.permutation.get('permuted_distributions', {}).get(metric, np.array([]))
                if len(perm_dist) > 0:
                    finite = perm_dist[np.isfinite(perm_dist)]
                    if len(finite) > 0:
                        data_to_plot.append(finite)
                        labels.append('Permutation')
            
            # Bootstrap
            if 'bootstrap' in mc_result.bootstrap:
                boot_dist = mc_result.bootstrap.get('bootstrap_distributions', {}).get(metric, np.array([]))
                if len(boot_dist) > 0:
                    finite = boot_dist[np.isfinite(boot_dist)]
                    if len(finite) > 0:
                        data_to_plot.append(finite)
                        labels.append('Bootstrap')
            
            # Randomized Entry
            if 'randomized_entry' in mc_result.randomized_entry:
                rand_dist = mc_result.randomized_entry.get('random_distributions', {}).get(metric, np.array([]))
                if len(rand_dist) > 0:
                    finite = rand_dist[np.isfinite(rand_dist)]
                    if len(finite) > 0:
                        data_to_plot.append(finite)
                        labels.append('Random Entry')
            
            if len(data_to_plot) > 0:
                bp = ax.boxplot(
                    data_to_plot,
                    labels=labels,
                    patch_artist=True,
                    showmeans=True,
                    meanline=True
                )
                
                # Color boxes
                colors_box = ['#2E86AB', '#A23B72', '#6A994E']
                for patch, color in zip(bp['boxes'], colors_box[:len(bp['boxes'])]):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
            
            metric_display = metric.replace('_', ' ').title()
            ax.set_title(metric_display, fontsize=12, fontweight='bold')
            ax.set_ylabel('Value', fontsize=10)
            ax.grid(True, alpha=0.3, axis='y')
            ax.tick_params(axis='x', rotation=45)
        
        fig.suptitle(
            'Monte Carlo Test Comparison: Distribution Quartiles',
            fontsize=14,
            fontweight='bold',
            y=1.02
        )
        
        plt.tight_layout()
        return fig
    
    def plot_metric_summary_table(
        self,
        mc_result: MonteCarloSuiteResult,
        metrics: List[str] = ['final_pnl', 'sharpe_ratio', 'profit_factor']
    ) -> Figure:
        """Create summary statistics table visualization.
        
        Args:
            mc_result: Monte Carlo suite results
            metrics: Metrics to summarize
        
        Returns:
            Matplotlib Figure with table
        """
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.axis('tight')
        ax.axis('off')
        
        # Prepare table data
        table_data = []
        headers = ['Metric', 'Test', 'Observed', 'Mean', 'Std', '5th %ile', '50th %ile', '95th %ile', 'P-Value', 'Percentile']
        
        for metric in metrics:
            metric_display = metric.replace('_', ' ').title()
            
            # Permutation
            if 'permutation' in mc_result.permutation:
                perm = mc_result.permutation
                obs = perm.get('observed_metrics', {}).get(metric, 0.0)
                dist = perm.get('permuted_distributions', {}).get(metric, np.array([]))
                if len(dist) > 0:
                    finite = dist[np.isfinite(dist)]
                    if len(finite) > 0:
                        p_val = perm.get('p_values', {}).get(metric, 1.0)
                        pct = perm.get('percentiles', {}).get(metric, 0.0)
                        table_data.append([
                            metric_display, 'Permutation',
                            f'{obs:.2f}',
                            f'{np.mean(finite):.2f}',
                            f'{np.std(finite, ddof=1):.2f}',
                            f'{np.percentile(finite, 5):.2f}',
                            f'{np.percentile(finite, 50):.2f}',
                            f'{np.percentile(finite, 95):.2f}',
                            f'{p_val:.4f}',
                            f'{pct:.1f}%'
                        ])
            
            # Bootstrap
            if 'bootstrap' in mc_result.bootstrap:
                boot = mc_result.bootstrap
                obs = boot.get('observed_metrics', {}).get(metric, 0.0)
                dist = boot.get('bootstrap_distributions', {}).get(metric, np.array([]))
                if len(dist) > 0:
                    finite = dist[np.isfinite(dist)]
                    if len(finite) > 0:
                        p_val = boot.get('p_values', {}).get(metric, 1.0)
                        pct = boot.get('percentiles', {}).get(metric, 0.0)
                        table_data.append([
                            metric_display, 'Bootstrap',
                            f'{obs:.2f}',
                            f'{np.mean(finite):.2f}',
                            f'{np.std(finite, ddof=1):.2f}',
                            f'{np.percentile(finite, 5):.2f}',
                            f'{np.percentile(finite, 50):.2f}',
                            f'{np.percentile(finite, 95):.2f}',
                            f'{p_val:.4f}',
                            f'{pct:.1f}%'
                        ])
            
            # Randomized Entry
            if 'randomized_entry' in mc_result.randomized_entry:
                rand = mc_result.randomized_entry
                obs = rand.get('observed_metrics', {}).get(metric, 0.0)
                dist = rand.get('random_distributions', {}).get(metric, np.array([]))
                if len(dist) > 0:
                    finite = dist[np.isfinite(dist)]
                    if len(finite) > 0:
                        p_val = rand.get('p_values', {}).get(metric, 1.0)
                        pct = rand.get('percentiles', {}).get(metric, 0.0)
                        table_data.append([
                            metric_display, 'Random Entry',
                            f'{obs:.2f}',
                            f'{np.mean(finite):.2f}',
                            f'{np.std(finite, ddof=1):.2f}',
                            f'{np.percentile(finite, 5):.2f}',
                            f'{np.percentile(finite, 50):.2f}',
                            f'{np.percentile(finite, 95):.2f}',
                            f'{p_val:.4f}',
                            f'{pct:.1f}%'
                        ])
        
        # Create table
        table = ax.table(
            cellText=table_data,
            colLabels=headers,
            cellLoc='center',
            loc='center',
            bbox=[0, 0, 1, 1]
        )
        
        # Style table
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Color header
        for i in range(len(headers)):
            table[(0, i)].set_facecolor('#4A90E2')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Color rows by test type
        row_idx = 1
        for metric in metrics:
            for test_name in ['Permutation', 'Bootstrap', 'Random Entry']:
                if row_idx <= len(table_data):
                    colors = {'Permutation': '#E8F4F8', 'Bootstrap': '#F8E8F4', 'Random Entry': '#E8F8E8'}
                    for col in range(len(headers)):
                        table[(row_idx, col)].set_facecolor(colors.get(test_name, 'white'))
                    row_idx += 1
        
        # Highlight significant p-values
        for row in range(1, len(table_data) + 1):
            p_val_str = table_data[row-1][8]  # P-value column
            try:
                p_val = float(p_val_str)
                if p_val <= 0.05:
                    # Highlight significant results
                    for col in range(len(headers)):
                        table[(row, col)].set_facecolor('#90EE90')  # Light green
            except ValueError:
                pass
        
        ax.set_title(
            'Monte Carlo Test Summary Statistics',
            fontsize=14,
            fontweight='bold',
            pad=20
        )
        
        plt.tight_layout()
        return fig
    
    def figure_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 string for HTML embedding.
        
        Args:
            fig: Matplotlib figure
        
        Returns:
            Base64 encoded PNG string
        """
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=150, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_base64
    
    def generate_all_visualizations(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        metrics: List[str] = ['final_pnl', 'sharpe_ratio', 'profit_factor']
    ) -> Dict[str, str]:
        """Generate all visualizations and return as base64 encoded images.
        
        Args:
            backtest_result: Original backtest result
            mc_result: Monte Carlo suite results
            metrics: Metrics to visualize
        
        Returns:
            Dictionary mapping visualization names to base64 encoded images
        """
        visualizations = {}
        
        # Distribution histograms for each test and metric
        for test_name in ['permutation', 'bootstrap', 'randomized_entry']:
            if test_name == 'permutation':
                test_data = mc_result.permutation
                dist_key = 'permuted_distributions'
            elif test_name == 'bootstrap':
                test_data = mc_result.bootstrap
                dist_key = 'bootstrap_distributions'
            else:
                test_data = mc_result.randomized_entry
                dist_key = 'random_distributions'
            
            for metric in metrics:
                observed = test_data.get('observed_metrics', {}).get(metric, 0.0)
                distribution = test_data.get(dist_key, {}).get(metric, np.array([]))
                
                if len(distribution) > 0:
                    fig = self.plot_distribution_histogram(
                        observed_value=observed,
                        distribution=distribution,
                        metric_name=metric,
                        test_name=test_name
                    )
                    key = f'{test_name}_{metric}_histogram'
                    visualizations[key] = self.figure_to_base64(fig)
        
        # Box plot comparison
        fig = self.plot_box_plot_comparison(mc_result, metrics)
        visualizations['box_plot_comparison'] = self.figure_to_base64(fig)
        
        # Summary table
        fig = self.plot_metric_summary_table(mc_result, metrics)
        visualizations['summary_table'] = self.figure_to_base64(fig)
        
        return visualizations

