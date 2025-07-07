"""
Visualization module for creating charts and plots
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime

class ChartGenerator:
    """Class for generating various charts and visualizations"""
    
    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Default chart dimensions
        self.default_width = config.CHART_WIDTH if hasattr(config, 'CHART_WIDTH') else 12
        self.default_height = config.CHART_HEIGHT if hasattr(config, 'CHART_HEIGHT') else 8
    
    def create_performance_chart(self, results: Dict[str, Any]) -> go.Figure:
        """Create performance comparison chart"""
        fig = go.Figure()
        
        # Add traces for each strategy
        for strategy_name, strategy_results in results.items():
            if isinstance(strategy_results, dict) and 'dates' in strategy_results:
                dates = strategy_results['dates']
                
                if 'total_values' in strategy_results:
                    values = strategy_results['total_values']
                elif 'portfolio_values' in strategy_results:
                    values = strategy_results['portfolio_values']
                else:
                    continue
                
                # Normalize to start at 100
                normalized_values = np.array(values) / values[0] * 100
                
                fig.add_trace(go.Scatter(
                    x=dates,
                    y=normalized_values,
                    mode='lines',
                    name=strategy_name.replace('_', ' ').title(),
                    line=dict(width=2)
                ))
        
        fig.update_layout(
            title='Portfolio Performance Comparison',
            xaxis_title='Date',
            yaxis_title='Normalized Value (Base = 100)',
            width=self.default_width,
            height=self.default_height,
            hovermode='x unified',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )
        
        return fig
    
    def create_correlation_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create correlation heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title='Correlation Matrix',
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def create_drawdown_chart(self, returns: pd.Series, title: str = "Drawdown Analysis") -> go.Figure:
        """Create drawdown chart"""
        # Calculate cumulative returns and drawdowns
        cumulative_returns = (1 + returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Cumulative Returns', 'Drawdown'),
            vertical_spacing=0.1
        )
        
        # Cumulative returns
        fig.add_trace(
            go.Scatter(
                x=cumulative_returns.index,
                y=cumulative_returns.values,
                mode='lines',
                name='Cumulative Returns',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Drawdown
        fig.add_trace(
            go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                mode='lines',
                name='Drawdown',
                fill='tonexty',
                line=dict(color='red', width=1),
                fillcolor='rgba(255,0,0,0.3)'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=title,
            width=self.default_width,
            height=self.default_height,
            showlegend=False
        )
        
        return fig
    
    def create_return_distribution_chart(self, returns: pd.Series, 
                                       title: str = "Return Distribution") -> go.Figure:
        """Create return distribution chart"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Histogram', 'Q-Q Plot'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Histogram
        fig.add_trace(
            go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='Returns',
                opacity=0.7
            ),
            row=1, col=1
        )
        
        # Q-Q plot
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(returns)))
        sample_quantiles = np.sort(returns.values)
        
        fig.add_trace(
            go.Scatter(
                x=theoretical_quantiles,
                y=sample_quantiles,
                mode='markers',
                name='Q-Q Plot',
                marker=dict(size=4)
            ),
            row=1, col=2
        )
        
        # Add diagonal line for Q-Q plot
        min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
        max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
        fig.add_trace(
            go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                name='Normal Distribution',
                line=dict(dash='dash', color='red')
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title=title,
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def create_rolling_metrics_chart(self, data: pd.DataFrame, 
                                   metrics: List[str]) -> go.Figure:
        """Create rolling metrics chart"""
        fig = make_subplots(
            rows=len(metrics), cols=1,
            shared_xaxes=True,
            subplot_titles=metrics,
            vertical_spacing=0.1
        )
        
        for i, metric in enumerate(metrics):
            if metric in data.columns:
                fig.add_trace(
                    go.Scatter(
                        x=data.index,
                        y=data[metric],
                        mode='lines',
                        name=metric,
                        line=dict(width=2)
                    ),
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title='Rolling Metrics Analysis',
            width=self.default_width,
            height=self.default_height * len(metrics) / 2,
            showlegend=False
        )
        
        return fig
    
    def create_hedge_effectiveness_chart(self, hedge_effectiveness: Dict[str, Any]) -> go.Figure:
        """Create hedge effectiveness visualization"""
        # Extract effectiveness metrics
        strategies = []
        effectiveness_values = []
        
        for strategy, metrics in hedge_effectiveness.items():
            if isinstance(metrics, dict) and 'hedge_effectiveness' in metrics:
                strategies.append(strategy.replace('_', ' ').title())
                effectiveness_values.append(metrics['hedge_effectiveness'])
        
        # Create bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=strategies,
                y=effectiveness_values,
                marker_color='lightblue',
                text=[f'{val:.2%}' for val in effectiveness_values],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            title='Hedge Effectiveness by Strategy',
            xaxis_title='Strategy',
            yaxis_title='Hedge Effectiveness',
            width=self.default_width,
            height=self.default_height,
            yaxis=dict(tickformat='.1%')
        )
        
        return fig
    
    def create_risk_metrics_chart(self, risk_metrics: Dict[str, Any]) -> go.Figure:
        """Create risk metrics comparison chart"""
        strategies = list(risk_metrics.keys())
        metrics = ['volatility', 'max_drawdown', 'var_95', 'expected_shortfall']
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[m.replace('_', ' ').title() for m in metrics],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
        
        for i, metric in enumerate(metrics):
            values = []
            for strategy in strategies:
                if isinstance(risk_metrics[strategy], dict):
                    if metric in risk_metrics[strategy]:
                        values.append(risk_metrics[strategy][metric])
                    elif 'tail_risk' in risk_metrics[strategy] and metric in risk_metrics[strategy]['tail_risk']:
                        values.append(risk_metrics[strategy]['tail_risk'][metric])
                    else:
                        values.append(0)
                else:
                    values.append(0)
            
            row, col = positions[i]
            fig.add_trace(
                go.Bar(
                    x=strategies,
                    y=values,
                    name=metric,
                    showlegend=False
                ),
                row=row, col=col
            )
        
        fig.update_layout(
            title='Risk Metrics Comparison',
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def create_fbx_analysis_chart(self, fbx_data: pd.DataFrame) -> go.Figure:
        """Create FBX analysis chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            subplot_titles=('FBX Index', 'Daily Returns', 'Rolling Volatility'),
            vertical_spacing=0.1
        )
        
        # FBX Index
        fig.add_trace(
            go.Scatter(
                x=fbx_data.index,
                y=fbx_data['FBX'],
                mode='lines',
                name='FBX Index',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Daily Returns
        if 'Returns' in fbx_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=fbx_data.index,
                    y=fbx_data['Returns'],
                    mode='lines',
                    name='Daily Returns',
                    line=dict(color='green', width=1)
                ),
                row=2, col=1
            )
        
        # Rolling Volatility
        if 'Vol_20' in fbx_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=fbx_data.index,
                    y=fbx_data['Vol_20'],
                    mode='lines',
                    name='20-Day Volatility',
                    line=dict(color='red', width=2)
                ),
                row=3, col=1
            )
        
        fig.update_layout(
            title='FBX Index Analysis',
            width=self.default_width,
            height=self.default_height,
            showlegend=False
        )
        
        return fig
    
    def create_portfolio_composition_chart(self, portfolio_breakdown: Dict[str, float]) -> go.Figure:
        """Create portfolio composition pie chart"""
        labels = list(portfolio_breakdown.keys())
        values = list(portfolio_breakdown.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            textinfo='label+percent',
            textposition='outside'
        )])
        
        fig.update_layout(
            title='Portfolio Composition',
            width=self.default_width,
            height=self.default_height
        )
        
        return fig
    
    def create_scenario_analysis_chart(self, scenario_results: Dict[str, Any]) -> go.Figure:
        """Create scenario analysis chart"""
        scenarios = []
        strategy_names = []
        returns = []
        
        for strategy, strategy_scenarios in scenario_results.items():
            if isinstance(strategy_scenarios, dict):
                for scenario, metrics in strategy_scenarios.items():
                    if isinstance(metrics, dict) and 'avg_return' in metrics:
                        scenarios.append(scenario)
                        strategy_names.append(strategy)
                        returns.append(metrics['avg_return'])
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame({
            'Scenario': scenarios,
            'Strategy': strategy_names,
            'Average_Return': returns
        })
        
        fig = px.bar(df, x='Scenario', y='Average_Return', color='Strategy',
                     title='Scenario Analysis - Average Returns',
                     barmode='group')
        
        fig.update_layout(
            width=self.default_width,
            height=self.default_height,
            xaxis_title='Market Scenario',
            yaxis_title='Average Return',
            yaxis=dict(tickformat='.2%')
        )
        
        return fig
    
    def save_chart(self, fig: go.Figure, filename: str, format: str = 'png'):
        """Save chart to file"""
        try:
            filepath = f"charts/{filename}.{format}"
            
            if format == 'html':
                fig.write_html(filepath)
            elif format == 'png':
                fig.write_image(filepath)
            elif format == 'pdf':
                fig.write_image(filepath)
            
            self.logger.info(f"Chart saved: {filepath}")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving chart: {e}")
            return None
