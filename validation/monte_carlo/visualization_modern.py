"""Modern Monte Carlo visualization module - Industry Standard Alternatives to Histograms.

Industry-standard visualizations based on:
- QuantConnect: Violin plots, CDF curves, summary cards
- Backtrader: Box plots, percentile bands
- Academic: ECDF plots, confidence intervals
- Professional: Clean, scannable cards with key metrics

Replaces histograms with:
1. Violin plots (distribution shape + quartiles)
2. CDF plots (percentile visualization - most intuitive)
3. Summary cards (key metrics at a glance)
4. Comparison tables (enhanced with color coding)
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
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


class ModernMonteCarloVisualizer:
    """Modern visualizations replacing histograms with industry-standard alternatives."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        """Initialize visualizer."""
        plt.style.use(style)
        self.colors = {
            'observed': '#2E86AB',      # Blue - observed value
            'baseline': '#2E86AB',       # Blue - baseline
            'random': '#BC4749',         # Red - random/MC
            'success': '#6A994E',        # Green - pass
            'warning': '#F18F01',        # Orange - warning
            'fail': '#BC4749',           # Red - fail
            'background': '#F5F5F5',
            'grid': '#E0E0E0',
            'text': '#333333'
        }
    
    def plot_violin_comparison(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str,
        p_value: float,
        percentile: float
    ) -> Figure:
        """Plot violin plot showing distribution shape + quartiles.
        
        Industry standard: Violin plots show distribution shape better than histograms.
        Shows: median, quartiles, distribution density, observed value.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric
            test_name: Name of MC test
            p_value: P-value
            percentile: Percentile rank
        
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Create data for violin plot
        data = pd.DataFrame({
            'value': finite_dist,
            'type': 'Monte Carlo'
        })
        
        # Plot violin
        parts = ax.violinplot(
            [finite_dist],
            positions=[0],
            widths=0.6,
            showmeans=True,
            showmedians=True,
            showextrema=True
        )
        
        # Style violin
        for pc in parts['bodies']:
            pc.set_facecolor(self.colors['background'])
            pc.set_alpha(0.7)
            pc.set_edgecolor(self.colors['text'])
        
        parts['cmeans'].set_color(self.colors['random'])
        parts['cmeans'].set_linewidth(2)
        parts['cmedians'].set_color(self.colors['random'])
        parts['cmedians'].set_linewidth(2)
        
        # Add observed value line
        ax.axvline(
            observed_value,
            color=self.colors['observed'],
            linewidth=3,
            linestyle='--',
            label=f'Observed: {observed_value:.2f}',
            zorder=10
        )
        
        # Add percentile bands
        percentiles = [5, 25, 50, 75, 95]
        percentile_values = np.percentile(finite_dist, percentiles)
        
        for i, (pct, val) in enumerate(zip(percentiles, percentile_values)):
            color = self.colors['warning'] if i < 2 else self.colors['success'] if i > 2 else self.colors['text']
            ax.axvline(
                val,
                color=color,
                linewidth=1,
                linestyle=':',
                alpha=0.5,
                label=f'{pct}th percentile' if i in [0, 2, 4] else None
            )
        
        # Formatting
        test_display = test_name.replace('_', ' ').title()
        metric_display = metric_name.replace('_', ' ').title()
        
        # Determine status (using ASCII to avoid font issues)
        status = "[PASS]" if p_value <= 0.05 else "[FAIL]" if p_value >= 0.95 else "[WARN]"
        status_color = self.colors['success'] if p_value <= 0.05 else self.colors['fail'] if p_value >= 0.95 else self.colors['warning']
        
        ax.set_title(
            f'{test_display}: {metric_display}\n'
            f'P-value: {p_value:.4f} | Percentile: {percentile:.1f}% | {status}',
            fontsize=12,
            fontweight='bold',
            color=status_color,
            pad=15
        )
        ax.set_ylabel(metric_display, fontsize=11)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        return fig
    
    def plot_cdf_with_percentile(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str,
        p_value: float,
        percentile: float
    ) -> Figure:
        """Plot Cumulative Distribution Function (CDF) - Most intuitive for percentile visualization.
        
        Industry standard: CDF plots show percentile directly - easier to interpret than histograms.
        Shows: What % of simulations performed worse than observed.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric
            test_name: Name of MC test
            p_value: P-value
            percentile: Percentile rank
        
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Sort for CDF
        sorted_dist = np.sort(finite_dist)
        n = len(sorted_dist)
        cdf_y = np.arange(1, n + 1) / n * 100  # Percentile
        
        # Plot CDF
        ax.plot(
            sorted_dist,
            cdf_y,
            color=self.colors['random'],
            linewidth=2.5,
            label='Monte Carlo CDF',
            zorder=5
        )
        
        # Fill area under curve
        ax.fill_between(
            sorted_dist,
            cdf_y,
            alpha=0.3,
            color=self.colors['random']
        )
        
        # Add observed value with percentile line
        ax.axvline(
            observed_value,
            color=self.colors['observed'],
            linewidth=3,
            linestyle='--',
            label=f'Observed: {observed_value:.2f}',
            zorder=10
        )
        
        # Draw horizontal line to CDF
        ax.axhline(
            percentile,
            color=self.colors['observed'],
            linewidth=2,
            linestyle=':',
            alpha=0.7,
            zorder=9
        )
        
        # Add percentile annotation
        ax.annotate(
            f'{percentile:.1f}th percentile',
            xy=(observed_value, percentile),
            xytext=(10, 10),
            textcoords='offset points',
            fontsize=10,
            fontweight='bold',
            color=self.colors['observed'],
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color=self.colors['observed'])
        )
        
        # Add reference lines for key percentiles
        for ref_pct in [5, 25, 50, 75, 95]:
            ref_val = np.percentile(finite_dist, ref_pct)
            ax.axvline(
                ref_val,
                color=self.colors['grid'],
                linewidth=1,
                linestyle=':',
                alpha=0.5
            )
        
        # Formatting
        test_display = test_name.replace('_', ' ').title()
        metric_display = metric_name.replace('_', ' ').title()
        
        # Determine status (using ASCII to avoid font issues)
        status = "[PASS]" if p_value <= 0.05 else "[FAIL]" if p_value >= 0.95 else "[WARN]"
        status_color = self.colors['success'] if p_value <= 0.05 else self.colors['fail'] if p_value >= 0.95 else self.colors['warning']
        
        ax.set_title(
            f'{test_display}: {metric_display} - Cumulative Distribution\n'
            f'P-value: {p_value:.4f} | Percentile: {percentile:.1f}% | {status}',
            fontsize=12,
            fontweight='bold',
            color=status_color,
            pad=15
        )
        ax.set_xlabel(metric_display, fontsize=11)
        ax.set_ylabel('Percentile (%)', fontsize=11)
        ax.set_ylim(0, 100)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def plot_summary_card(
        self,
        observed_value: float,
        distribution: np.ndarray,
        metric_name: str,
        test_name: str,
        p_value: float,
        percentile: float
    ) -> Figure:
        """Plot summary card - Clean, scannable visualization.
        
        Industry standard: Summary cards show key metrics at a glance.
        Better for reports than histograms - easier to scan.
        
        Args:
            observed_value: Observed metric value
            distribution: MC distribution values
            metric_name: Name of metric
            test_name: Name of MC test
            p_value: P-value
            percentile: Percentile rank
        
        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.axis('off')
        
        # Remove infinite and NaN values
        finite_dist = distribution[np.isfinite(distribution)]
        if len(finite_dist) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # Calculate statistics
        mean_val = np.mean(finite_dist)
        median_val = np.median(finite_dist)
        std_val = np.std(finite_dist, ddof=1)
        min_val = np.min(finite_dist)
        max_val = np.max(finite_dist)
        p5 = np.percentile(finite_dist, 5)
        p95 = np.percentile(finite_dist, 95)
        
        # Determine status (using ASCII to avoid font issues)
        if p_value <= 0.05:
            status = "[PASS]"
            status_color = self.colors['success']
            interpretation = "Strategy significantly outperforms null hypothesis"
        elif p_value >= 0.95:
            status = "[FAIL]"
            status_color = self.colors['fail']
            interpretation = "Strategy significantly underperforms null hypothesis"
        else:
            status = "[WARN]"
            status_color = self.colors['warning']
            interpretation = "Strategy performance is marginal"
        
        # Formatting
        test_display = test_name.replace('_', ' ').title()
        metric_display = metric_name.replace('_', ' ').title()
        
        # Create card content
        y_start = 0.95
        line_height = 0.12
        
        # Title
        ax.text(
            0.5, y_start,
            f'{test_display}: {metric_display}',
            ha='center',
            va='top',
            fontsize=14,
            fontweight='bold',
            transform=ax.transAxes
        )
        
        y_pos = y_start - line_height * 1.5
        
        # Status
        ax.text(
            0.5, y_pos,
            status,
            ha='center',
            va='top',
            fontsize=16,
            fontweight='bold',
            color=status_color,
            transform=ax.transAxes
        )
        
        y_pos -= line_height * 1.2
        
        # Key metrics
        metrics_text = [
            f'Observed Value: {observed_value:.2f}',
            f'P-value: {p_value:.4f}',
            f'Percentile: {percentile:.1f}%',
            '',
            f'MC Mean: {mean_val:.2f}',
            f'MC Median: {median_val:.2f}',
            f'MC Std Dev: {std_val:.2f}',
            '',
            f'MC Range: [{min_val:.2f}, {max_val:.2f}]',
            f'MC 5th-95th: [{p5:.2f}, {p95:.2f}]'
        ]
        
        for line in metrics_text:
            if line:
                ax.text(
                    0.5, y_pos,
                    line,
                    ha='center',
                    va='top',
                    fontsize=10,
                    transform=ax.transAxes
                )
            y_pos -= line_height
        
        # Interpretation
        ax.text(
            0.5, 0.05,
            interpretation,
            ha='center',
            va='bottom',
            fontsize=9,
            style='italic',
            color=self.colors['text'],
            transform=ax.transAxes,
            wrap=True
        )
        
        # Add border
        rect = plt.Rectangle(
            (0.02, 0.02), 0.96, 0.96,
            fill=False,
            edgecolor=status_color,
            linewidth=2,
            transform=ax.transAxes
        )
        ax.add_patch(rect)
        
        plt.tight_layout()
        return fig
    
    def figure_to_base64(self, fig: Figure) -> str:
        """Convert matplotlib figure to base64 encoded string."""
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)
        return img_base64
    
    def generate_all_visualizations(
        self,
        backtest_result: BacktestResult,
        mc_result: MonteCarloSuiteResult,
        metrics: List[str] = ['final_pnl', 'sharpe_ratio', 'profit_factor'],
        visualization_type: str = 'cdf'  # 'cdf', 'violin', or 'card'
    ) -> Dict[str, str]:
        """Generate all visualizations using modern alternatives to histograms.
        
        Args:
            backtest_result: Original backtest result
            mc_result: Monte Carlo suite results
            metrics: Metrics to visualize
            visualization_type: Type of visualization ('cdf', 'violin', or 'card')
        
        Returns:
            Dictionary mapping visualization names to base64 encoded images
        """
        visualizations = {}
        
        # Map test names to data
        test_map = {
            'permutation': ('permutation', 'permuted_distributions'),
            'bootstrap': ('bootstrap', 'bootstrap_distributions'),
            'randomized_entry': ('randomized_entry', 'random_distributions')
        }
        
        # Generate visualizations for each test and metric
        for test_name, (test_key, dist_key) in test_map.items():
            test_data = getattr(mc_result, test_key, None)
            if test_data is None:
                continue
            
            # Handle dict or object access
            if isinstance(test_data, dict):
                observed_metrics = test_data.get('observed_metrics', {})
                distributions = test_data.get(dist_key, {})
                p_values = test_data.get('p_values', {})
                percentiles = test_data.get('percentiles', {})
            else:
                observed_metrics = getattr(test_data, 'observed_metrics', {})
                distributions = getattr(test_data, dist_key, {})
                p_values = getattr(test_data, 'p_values', {})
                percentiles = getattr(test_data, 'percentiles', {})
            
            for metric in metrics:
                observed = observed_metrics.get(metric, 0.0) if isinstance(observed_metrics, dict) else getattr(observed_metrics, metric, 0.0)
                distribution = distributions.get(metric, np.array([])) if isinstance(distributions, dict) else getattr(distributions, metric, np.array([]))
                p_value = p_values.get(metric, 1.0) if isinstance(p_values, dict) else getattr(p_values, metric, 1.0)
                percentile = percentiles.get(metric, 0.0) if isinstance(percentiles, dict) else getattr(percentiles, metric, 0.0)
                
                if len(distribution) > 0:
                    # Convert to numpy array if needed
                    if not isinstance(distribution, np.ndarray):
                        distribution = np.array(distribution)
                    
                    # Generate visualization based on type
                    if visualization_type == 'cdf':
                        fig = self.plot_cdf_with_percentile(
                            observed_value=observed,
                            distribution=distribution,
                            metric_name=metric,
                            test_name=test_name,
                            p_value=p_value,
                            percentile=percentile
                        )
                        key_suffix = 'cdf'
                    elif visualization_type == 'violin':
                        fig = self.plot_violin_comparison(
                            observed_value=observed,
                            distribution=distribution,
                            metric_name=metric,
                            test_name=test_name,
                            p_value=p_value,
                            percentile=percentile
                        )
                        key_suffix = 'violin'
                    else:  # card
                        fig = self.plot_summary_card(
                            observed_value=observed,
                            distribution=distribution,
                            metric_name=metric,
                            test_name=test_name,
                            p_value=p_value,
                            percentile=percentile
                        )
                        key_suffix = 'card'
                    
                    key = f'{test_name}_{metric}_{key_suffix}'
                    visualizations[key] = self.figure_to_base64(fig)
        
        return visualizations

