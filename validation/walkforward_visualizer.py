"""Walk-Forward Analysis Visualizations - Industry Standard Dashboard.

Implements the "Quant Dashboard" approach with:
- Panel 1: Equity Curve (Log Scale) with IS/OOS separation
- Panel 2: Walk-Forward Efficiency (WFE) Waterfall Chart

Based on industry standards from top-tier quant funds.

FIXED: All Timestamp arithmetic issues completely resolved.
"""

from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from datetime import datetime

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

from validation.walkforward import WalkForwardResult, WalkForwardStep
from engine.backtest_engine import BacktestResult, Trade


class WalkForwardVisualizer:
    """Generate industry-standard visualizations for walk-forward analysis."""
    
    def __init__(self):
        """Initialize visualizer."""
        if not PLOTLY_AVAILABLE:
            raise ImportError("Plotly is required for walk-forward visualizations. Install with: pip install plotly")
        
        self.colors = {
            'train': '#4CAF50',      # Green for training periods
            'test': '#2196F3',       # Blue for test periods
            'excluded': '#999999',   # Grey for excluded periods
            'wfe_good': '#4CAF50',   # Green for WFE >= 60%
            'wfe_warning': '#FFC107', # Yellow for 40-60%
            'wfe_bad': '#f44336',    # Red for < 40%
            'threshold': '#666666',  # Grey for threshold lines
        }
    
    def _to_pydatetime(self, ts) -> datetime:
        """Safely convert any timestamp-like object to Python datetime."""
        if ts is None:
            raise ValueError("Timestamp cannot be None")
        if isinstance(ts, datetime):
            return ts
        if isinstance(ts, pd.Timestamp):
            return ts.to_pydatetime()
        # Handle any other type by converting through pandas
        return pd.to_datetime(ts).to_pydatetime()
    
    def reconstruct_time_indexed_equity(
        self,
        result: BacktestResult,
        period_start,
        period_end
    ) -> Tuple[List[datetime], List[float]]:
        """
        Reconstruct time-indexed equity curve from trades.
        
        Returns lists instead of Series to avoid any pandas type issues.
        
        Args:
            result: BacktestResult with trades
            period_start: Start of the period (any timestamp-like object)
            period_end: End of the period (any timestamp-like object)
        
        Returns:
            Tuple of (timestamps as Python datetime, equity values as float)
        """
        # Convert to Python datetime immediately - this is the critical fix
        try:
            period_start_dt = self._to_pydatetime(period_start)
            period_end_dt = self._to_pydatetime(period_end)
        except Exception as e:
            # If conversion fails, provide detailed error
            raise ValueError(f"Failed to convert timestamps: start={period_start} ({type(period_start)}), end={period_end} ({type(period_end)}). Error: {e}")
        
        if not result.trades or len(result.trades) == 0:
            # No trades - return flat line at initial capital
            return (
                [period_start_dt, period_end_dt],
                [float(result.initial_capital), float(result.initial_capital)]
            )
        
        # Build equity curve from trades using only Python types
        timestamps = [period_start_dt]
        equity_values = [float(result.initial_capital)]
        
        current_equity = float(result.initial_capital)
        
        for trade in result.trades:
            # Update equity after each trade
            pnl = float(trade.pnl_after_costs)
            current_equity = current_equity + pnl  # Explicit float addition
            equity_values.append(current_equity)
            
            # Convert exit_time to Python datetime
            try:
                exit_dt = self._to_pydatetime(trade.exit_time)
                timestamps.append(exit_dt)
            except Exception as e:
                raise ValueError(f"Failed to convert trade exit_time: {trade.exit_time} ({type(trade.exit_time)}). Error: {e}")
        
        # Add final point at period end (only if different from last trade)
        # Compare datetime objects directly
        if period_end_dt > timestamps[-1]:
            timestamps.append(period_end_dt)
            equity_values.append(current_equity)
        
        return timestamps, equity_values
    
    def plot_equity_curve_panel(
        self,
        wf_result: WalkForwardResult,
        initial_capital: float = 10000.0
    ) -> go.Figure:
        """
        Panel 1: Simple Equity Curve (Log Scale) with IS/OOS separation.
        
        Clean, simple visualization showing:
        - Log scale for percentage changes
        - Visual separation of training (IS) vs test (OOS) periods
        - Excluded periods shown in grey/dashed
        - Basic tooltips
        
        Args:
            wf_result: WalkForwardResult with all steps
            initial_capital: Starting capital
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        cumulative_equity = float(initial_capital)
        
        # Process first step's training period (this is the base)
        if wf_result.steps:
            first_step = wf_result.steps[0]
            
            try:
                train_timestamps, train_values = self.reconstruct_time_indexed_equity(
                    first_step.train_result,
                    first_step.train_start,
                    first_step.train_end
                )
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct training equity for step {first_step.step_number}: {e}")
            
            # Normalize to start from initial capital (pure Python float arithmetic)
            train_start_value = float(train_values[0])
            train_offset = float(initial_capital) - train_start_value
            train_normalized = [float(v) + train_offset for v in train_values]
            
            # Add first training period (simple)
            fig.add_trace(go.Scatter(
                x=train_timestamps,
                y=train_normalized,
                mode='lines',
                name='Training (IS)',
                line=dict(color=self.colors['train'], width=2),
                hovertemplate='<b>Training Period</b><br>' +
                             'Date: %{x}<br>' +
                             'Equity: $%{y:,.2f}<extra></extra>'
            ))
            
            cumulative_equity = float(train_normalized[-1])
        
        # Process each test period
        for step in wf_result.steps:
            # Test period (OOS)
            try:
                test_timestamps, test_values = self.reconstruct_time_indexed_equity(
                    step.test_result,
                    step.test_start,
                    step.test_end
                )
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct test equity for step {step.step_number}: {e}")
            
            # Normalize to cumulative equity (pure Python float arithmetic)
            test_start_value = float(test_values[0])
            test_offset = float(cumulative_equity) - test_start_value
            test_normalized = [float(v) + test_offset for v in test_values]
            
            # Determine line style based on exclusion
            line_style = 'dash' if step.excluded_from_stats else 'solid'
            line_color = self.colors['excluded'] if step.excluded_from_stats else self.colors['test']
            line_width = 2 if step.excluded_from_stats else 3
            
            test_label = f"Step {step.step_number} OOS"
            if step.excluded_from_stats:
                test_label += " [EXCLUDED]"
            
            # Add test period (simple)
            test_pf = float(step.test_metrics.get('pf', 0))
            fig.add_trace(go.Scatter(
                x=test_timestamps,
                y=test_normalized,
                mode='lines',
                name=test_label,
                line=dict(color=line_color, width=line_width, dash=line_style),
                hovertemplate=f'<b>{test_label}</b><br>' +
                             f'Date: %{{x}}<br>' +
                             f'Equity: $%{{y:,.2f}}<br>' +
                             f'PF: {test_pf:.2f}<extra></extra>'
            ))
            
            # Add simple PF annotation at end of test period
            if not step.excluded_from_stats and test_pf > 0:
                fig.add_annotation(
                    x=test_timestamps[-1],
                    y=test_normalized[-1],
                    text=f"PF: {test_pf:.2f}",
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor=line_color,
                    bgcolor="white",
                    bordercolor=line_color,
                    borderwidth=1,
                    font=dict(size=10, color=line_color)
                )
            
            cumulative_equity = float(test_normalized[-1])
        
        # Add vertical lines to separate periods
        for step in wf_result.steps:
            try:
                test_start_dt = self._to_pydatetime(step.test_start)
                test_start_str = test_start_dt.isoformat()
                
                # Vertical line for equity chart
                fig.add_shape(
                    type="line",
                    x0=test_start_str,
                    x1=test_start_str,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="gray", width=1, dash="dot"),
                    opacity=0.5
                )
                
                # Annotation
                fig.add_annotation(
                    x=test_start_str,
                    y=1,
                    yref="paper",
                    text=f"Step {step.step_number}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="gray")
                )
            except Exception as e:
                print(f"Warning: Could not add vline for step {step.step_number}: {e}")
        
        # Update layout with log scale
        fig.update_layout(
            title={
                'text': 'Walk-Forward Equity Curve (Log Scale)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Date',
            yaxis_title='Equity ($)',
            yaxis_type='log',  # Log scale for percentage changes
            hovermode='x unified',
            height=500,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            template='plotly_white'
        )
        
        return fig
    
    def plot_drawdown_panel(
        self,
        wf_result: WalkForwardResult,
        initial_capital: float = 10000.0
    ) -> go.Figure:
        """
        Panel: Drawdown Visualization (Separate Chart).
        
        Shows drawdown percentage over time as a separate chart.
        
        Args:
            wf_result: WalkForwardResult with all steps
            initial_capital: Starting capital
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        cumulative_equity = float(initial_capital)
        all_timestamps = []
        all_equity_values = []
        
        # Process first step's training period
        if wf_result.steps:
            first_step = wf_result.steps[0]
            try:
                train_timestamps, train_values = self.reconstruct_time_indexed_equity(
                    first_step.train_result,
                    first_step.train_start,
                    first_step.train_end
                )
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct training equity for step {first_step.step_number}: {e}")
            
            train_start_value = float(train_values[0])
            train_offset = float(initial_capital) - train_start_value
            train_normalized = [float(v) + train_offset for v in train_values]
            
            all_timestamps.extend(train_timestamps)
            all_equity_values.extend(train_normalized)
            cumulative_equity = float(train_normalized[-1])
        
        # Process each test period
        for step in wf_result.steps:
            try:
                test_timestamps, test_values = self.reconstruct_time_indexed_equity(
                    step.test_result,
                    step.test_start,
                    step.test_end
                )
            except Exception as e:
                raise RuntimeError(f"Failed to reconstruct test equity for step {step.step_number}: {e}")
            
            test_start_value = float(test_values[0])
            test_offset = float(cumulative_equity) - test_start_value
            test_normalized = [float(v) + test_offset for v in test_values]
            
            all_timestamps.extend(test_timestamps)
            all_equity_values.extend(test_normalized)
            cumulative_equity = float(test_normalized[-1])
        
        # Calculate drawdown using industry standard formula
        # Industry Standard Formula: Drawdown % = (Peak - Current) / Peak * 100
        # Example: Peak = $10,000, Current = $9,200 → Drawdown = (10000-9200)/10000*100 = 8%
        if len(all_equity_values) > 1:
            equity_array = np.array(all_equity_values, dtype=float)
            
            # Calculate running peak (maximum equity seen so far)
            running_max = np.maximum.accumulate(equity_array)
            
            # Industry standard formula: (Peak - Current) / Peak * 100
            # This gives positive percentages (0% = at peak, 8% = 8% below peak)
            # Avoid division by zero
            with np.errstate(divide='ignore', invalid='ignore'):
                drawdown_pct = ((running_max - equity_array) / running_max) * 100.0
                # Replace any inf/nan with 0
                drawdown_pct = np.nan_to_num(drawdown_pct, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Ensure values are between 0% and 100% (drawdown can't exceed 100%)
            drawdown_pct = np.clip(drawdown_pct, 0.0, 100.0)
            
            # Convert to list for Plotly (ensures proper formatting)
            drawdown_pct_list = [float(x) for x in drawdown_pct]
            equity_list = [float(x) for x in equity_array]
            peak_list = [float(x) for x in running_max]
            
            # Prepare hover text with pre-formatted values to avoid template parsing issues
            hover_texts = []
            for i in range(len(equity_list)):
                equity_str = f"${equity_list[i]:,.2f}"
                peak_str = f"${peak_list[i]:,.2f}"
                dd_str = f"{drawdown_pct_list[i]:.2f}%"
                hover_texts.append(
                    f'<b>Drawdown</b><br>' +
                    f'Date: {all_timestamps[i].strftime("%B %d, %Y")}<br>' +
                    f'Current Equity: {equity_str}<br>' +
                    f'Peak Equity: {peak_str}<br>' +
                    f'Drawdown: <b>{dd_str}</b><br>' +
                    f'<i>Percentage decline from peak</i>'
                )
            
            # Industry standard display: Show as positive percentages with inverted y-axis
            # This way drawdown visually "goes down" on the chart
            # 0% at top = at peak, higher % going down = deeper drawdown
            
            # Add drawdown trace with enhanced tooltip showing actual values
            fig.add_trace(go.Scatter(
                x=all_timestamps,
                y=drawdown_pct_list,
                mode='lines',
                name='Drawdown',
                line=dict(color='#f44336', width=2),
                fill='tozeroy',
                fillcolor='rgba(244, 67, 54, 0.3)',
                hovertext=hover_texts,
                hovertemplate='%{hovertext}<extra></extra>'
            ))
        
        # Add vertical lines to separate periods
        for step in wf_result.steps:
            try:
                test_start_dt = self._to_pydatetime(step.test_start)
                test_start_str = test_start_dt.isoformat()
                
                fig.add_shape(
                    type="line",
                    x0=test_start_str,
                    x1=test_start_str,
                    y0=0,
                    y1=1,
                    yref="paper",
                    line=dict(color="gray", width=1, dash="dot"),
                    opacity=0.5
                )
                
                fig.add_annotation(
                    x=test_start_str,
                    y=1,
                    yref="paper",
                    text=f"Step {step.step_number}",
                    showarrow=False,
                    yanchor="bottom",
                    font=dict(size=10, color="gray")
                )
            except Exception as e:
                print(f"Warning: Could not add vline for step {step.step_number}: {e}")
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Drawdown Analysis',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            yaxis=dict(
                # Industry standard: Inverted y-axis so drawdown visually goes "down"
                # 0% at top = at peak, higher % going down = deeper drawdown
                autorange='reversed',
                range=[None, 0]  # Start from top (0%) and go down
            ),
            hovermode='x unified',
            height=400,
            showlegend=False,
            template='plotly_white'
        )
        
        return fig
    
    def plot_wfe_waterfall(
        self,
        wf_result: WalkForwardResult
    ) -> go.Figure:
        """
        Panel 2: Walk-Forward Efficiency (WFE) Waterfall Chart.
        
        Shows WFE for each OOS period with:
        - Color coding: Green (≥60%), Yellow (40-60%), Red (<40%)
        - Threshold line at 60%
        - Value labels on bars
        - Excluded periods shown in grey
        
        Args:
            wf_result: WalkForwardResult with all steps
        
        Returns:
            Plotly Figure
        """
        fig = go.Figure()
        
        # Calculate WFE for each step
        periods = []
        wfe_values = []
        colors = []
        labels = []
        
        for step in wf_result.steps:
            train_pf = float(step.train_metrics.get('pf', 0))
            test_pf = float(step.test_metrics.get('pf', 0))
            
            if train_pf > 0:
                wfe = test_pf / train_pf
            else:
                wfe = 0.0
            
            # Determine color based on WFE and exclusion status
            if step.excluded_from_stats:
                color = self.colors['excluded']
                label = f"Step {step.step_number} [EXCLUDED]"
            elif wfe >= 0.6:
                color = self.colors['wfe_good']
                label = f"Step {step.step_number}"
            elif wfe >= 0.4:
                color = self.colors['wfe_warning']
                label = f"Step {step.step_number}"
            else:
                color = self.colors['wfe_bad']
                label = f"Step {step.step_number}"
            
            periods.append(f"Step {step.step_number}")
            wfe_values.append(wfe)
            colors.append(color)
            labels.append(label)
        
        # Create bar chart with custom hover data
        hover_texts = []
        for step, wfe in zip(wf_result.steps, wfe_values):
            train_pf = float(step.train_metrics.get('pf', 0))
            test_pf = float(step.test_metrics.get('pf', 0))
            hover_texts.append(
                f'<b>{step.step_number}</b><br>' +
                f'WFE: {wfe:.1%}<br>' +
                f'Test PF: {test_pf:.2f}<br>' +
                f'Train PF: {train_pf:.2f}'
            )
        
        fig.add_trace(go.Bar(
            x=periods,
            y=wfe_values,
            marker_color=colors,
            text=[f"{wfe:.1%}" for wfe in wfe_values],
            textposition='outside',
            hovertemplate='%{customdata}<extra></extra>',
            customdata=hover_texts,
            name='WFE'
        ))
        
        # Add threshold line at 60%
        fig.add_hline(
            y=0.6,
            line_dash="dash",
            line_color=self.colors['threshold'],
            annotation_text="Target (60%)",
            annotation_position="right"
        )
        
        # Calculate average WFE (only from included periods)
        included_wfe = [wfe for wfe, step in zip(wfe_values, wf_result.steps) 
                       if not step.excluded_from_stats]
        if included_wfe:
            avg_wfe = float(np.mean(included_wfe))
            fig.add_hline(
                y=avg_wfe,
                line_dash="dot",
                line_color="blue",
                annotation_text=f"Avg: {avg_wfe:.1%}",
                annotation_position="left"
            )
        
        # Update layout
        fig.update_layout(
            title={
                'text': 'Walk-Forward Efficiency (WFE)',
                'x': 0.5,
                'xanchor': 'center'
            },
            xaxis_title='Period',
            yaxis_title='WFE (%)',
            yaxis=dict(
                tickformat='.0%',
                range=[0, max(1.2, max(wfe_values) * 1.1) if wfe_values else 1.2]
            ),
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        return fig
    
    def generate_dashboard_html(
        self,
        wf_result: WalkForwardResult,
        initial_capital: float = 10000.0,
        show_drawdown: bool = True,
        show_performance_bands: bool = True,
        comparison_results: Optional[List[WalkForwardResult]] = None
    ) -> str:
        """
        Generate HTML for all panels (separate charts).
        
        Returns HTML with embedded Plotly charts, each in its own section.
        
        Args:
            wf_result: WalkForwardResult with all steps
            initial_capital: Starting capital
            show_drawdown: Whether to show drawdown visualization
            show_performance_bands: Whether to show ±1 std dev performance bands
            comparison_results: Optional list of WalkForwardResult for comparison
        
        Returns:
            HTML string with embedded charts
        """
        try:
            # Generate Panel 1: Simple Equity Curve
            equity_fig = self.plot_equity_curve_panel(wf_result, initial_capital)
            equity_html = equity_fig.to_html(include_plotlyjs=False, div_id='wf-equity-chart', full_html=False)
            
            # Generate Panel 2: Drawdown (separate chart)
            drawdown_html = ""
            if show_drawdown:
                drawdown_fig = self.plot_drawdown_panel(wf_result, initial_capital)
                drawdown_html = drawdown_fig.to_html(include_plotlyjs=False, div_id='wf-drawdown-chart', full_html=False)
            
            # Generate Panel 3: WFE Waterfall
            wfe_fig = self.plot_wfe_waterfall(wf_result)
            wfe_html = wfe_fig.to_html(include_plotlyjs=False, div_id='wf-wfe-chart', full_html=False)
            
            # Combine into dashboard layout with separate charts
            dashboard_html = f"""
            <div style="margin: 30px 0; padding: 20px; background: #f9f9f9; border-radius: 5px;">
                <h3 style="margin-top: 0;">Walk-Forward Dashboard</h3>
                
                <div style="margin: 20px 0; background: white; padding: 15px; border-radius: 5px;">
                    <h4>Panel 1: Equity Curve (Log Scale)</h4>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Cumulative equity across all periods. Training periods (IS) in green, 
                        test periods (OOS) in blue. Excluded periods shown in grey/dashed.
                        Log scale shows percentage changes accurately. At a glance, you can easily see 
                        PF continuity, consistency, and smoothness.
                    </p>
                    {equity_html}
                </div>
"""
            
            if show_drawdown and drawdown_html:
                dashboard_html += f"""
                <div style="margin: 20px 0; background: white; padding: 15px; border-radius: 5px;">
                    <h4>Panel 2: Drawdown Analysis</h4>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        Drawdown percentage over time. Shows periods of equity decline from peak values.
                        Lower drawdown indicates better risk management. Hover to see exact drawdown values.
                    </p>
                    {drawdown_html}
                </div>
"""
            
            dashboard_html += f"""
                <div style="margin: 20px 0; background: white; padding: 15px; border-radius: 5px;">
                    <h4>Panel {3 if show_drawdown else 2}: Walk-Forward Efficiency (WFE)</h4>
                    <p style="color: #666; font-size: 0.9em; margin-bottom: 10px;">
                        WFE = Test PF / Train PF. Measures performance retention from training to test.
                        <strong>Target: ≥60%</strong> (green), <strong>Warning: 40-60%</strong> (yellow), 
                        <strong>Fail: &lt;40%</strong> (red). Excluded periods shown in grey.
                    </p>
                    {wfe_html}
                </div>
            </div>
            """
            
            return dashboard_html
            
        except Exception as e:
            # Return error message in HTML format
            return f"""
            <div style="margin: 30px 0; padding: 20px; background: #fff3cd; border: 1px solid #ffc107; border-radius: 5px;">
                <h3 style="margin-top: 0; color: #856404;">Walk-Forward Visualization Error</h3>
                <p style="color: #856404;">Could not generate walk-forward visualizations: {str(e)}</p>
                <p style="color: #856404; font-size: 0.9em;">Check that all timestamps in the walk-forward results are valid.</p>
            </div>
            """