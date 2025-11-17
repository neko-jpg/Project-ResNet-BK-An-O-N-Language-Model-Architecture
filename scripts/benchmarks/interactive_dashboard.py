"""
Interactive Dashboard for Mamba vs ResNet-BK Comparison

Creates a web-based visualization with zoom, filter, and comparison tools.
Implements task 20.2 from mamba-killer-ultra-scale spec.

Requirements: 8.23, 8.24

Features:
- Interactive plots with zoom and pan
- Filter by metric category
- One-click comparison between models
- Real-time data loading
- Export functionality

Usage:
    # Start dashboard server
    python scripts/interactive_dashboard.py
    
    # Specify port
    python scripts/interactive_dashboard.py --port 8050
    
    # Load specific results directory
    python scripts/interactive_dashboard.py --results_dir results/killer_graphs
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional
import webbrowser
from threading import Timer

try:
    import dash
    from dash import dcc, html, Input, Output, State
    import plotly.graph_objects as go
    import plotly.express as px
    import pandas as pd
    DASH_AVAILABLE = True
except ImportError:
    DASH_AVAILABLE = False
    print("Warning: dash and plotly not installed. Install with: pip install dash plotly")

import numpy as np


class InteractiveDashboard:
    """Interactive dashboard for model comparison."""
    
    def __init__(self, results_dir: str = 'results/killer_graphs'):
        """
        Initialize dashboard.
        
        Args:
            results_dir: directory containing results JSON files
        """
        self.results_dir = Path(results_dir)
        self.app = None
        self.data = {}
        
        # Load data
        self.load_data()
    
    def load_data(self):
        """Load all results from JSON files."""
        print(f"Loading data from {self.results_dir}...")
        
        # Load stability data
        stability_json = self.results_dir / 'stability_graph.json'
        if stability_json.exists():
            with open(stability_json, 'r') as f:
                self.data['stability'] = json.load(f)
            print(f"  ✓ Loaded stability data")
        
        # Load quantization data
        quant_json = self.results_dir / 'quantization_graph.json'
        if quant_json.exists():
            with open(quant_json, 'r') as f:
                self.data['quantization'] = json.load(f)
            print(f"  ✓ Loaded quantization data")
        
        # Load efficiency data
        eff_json = self.results_dir / 'efficiency_graph.json'
        if eff_json.exists():
            with open(eff_json, 'r') as f:
                self.data['efficiency'] = json.load(f)
            print(f"  ✓ Loaded efficiency data")
        
        # Load comparison table
        table_csv = self.results_dir / 'comparison_table.csv'
        if table_csv.exists():
            self.data['comparison_table'] = pd.read_csv(table_csv)
            print(f"  ✓ Loaded comparison table")
        
        if not self.data:
            print("  Warning: No data files found. Dashboard will show empty plots.")
    
    def create_stability_plot(self) -> go.Figure:
        """Create interactive stability plot."""
        fig = go.Figure()
        
        if 'stability' not in self.data:
            fig.add_annotation(
                text="No stability data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        stability = self.data['stability']
        
        # Plot ResNet-BK
        rb_data = stability.get('resnetbk', {})
        for seq_len, metrics in rb_data.items():
            if seq_len >= 1000000:
                label = f"{int(seq_len)//1000000}M"
            elif seq_len >= 1000:
                label = f"{int(seq_len)//1000}k"
            else:
                label = str(seq_len)
            
            fig.add_trace(go.Scatter(
                x=[seq_len],
                y=[metrics.get('final_loss', 0)],
                mode='markers+lines',
                name=f'ResNet-BK N={label}',
                marker=dict(size=10, color='#2E86AB'),
                hovertemplate=f'<b>ResNet-BK N={label}</b><br>' +
                             'Loss: %{y:.4f}<br>' +
                             f'Diverged: {metrics.get("diverged", False)}<br>' +
                             f'NaN count: {metrics.get("num_nan", 0)}<extra></extra>'
            ))
        
        # Plot Mamba
        mb_data = stability.get('mamba', {})
        for seq_len, metrics in mb_data.items():
            if seq_len >= 1000000:
                label = f"{int(seq_len)//1000000}M"
            elif seq_len >= 1000:
                label = f"{int(seq_len)//1000}k"
            else:
                label = str(seq_len)
            
            color = '#F18F01' if metrics.get('diverged', False) else '#A23B72'
            
            fig.add_trace(go.Scatter(
                x=[seq_len],
                y=[metrics.get('final_loss', 0)],
                mode='markers+lines',
                name=f'Mamba N={label}',
                marker=dict(size=10, color=color, symbol='square'),
                hovertemplate=f'<b>Mamba N={label}</b><br>' +
                             'Loss: %{y:.4f}<br>' +
                             f'Diverged: {metrics.get("diverged", False)}<br>' +
                             f'NaN count: {metrics.get("num_nan", 0)}<extra></extra>'
            ))
        
        fig.update_layout(
            title='Long-Context Stability Comparison',
            xaxis_title='Sequence Length',
            yaxis_title='Final Loss',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_quantization_plot(self) -> go.Figure:
        """Create interactive quantization plot."""
        fig = go.Figure()
        
        if 'quantization' not in self.data:
            fig.add_annotation(
                text="No quantization data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        quant = self.data['quantization']
        
        # Prepare data
        bit_widths = [32, 16, 8, 4, 2]
        bit_labels = ['FP32', 'FP16', 'INT8', 'INT4', 'INT2']
        
        rb_ppls = []
        mb_ppls = []
        
        rb_data = quant.get('resnetbk', {})
        mb_data = quant.get('mamba', {})
        
        for bits in bit_widths:
            rb_ppls.append(rb_data.get(str(bits), {}).get('perplexity', None))
            mb_ppls.append(mb_data.get(str(bits), {}).get('perplexity', None))
        
        # Plot ResNet-BK
        fig.add_trace(go.Scatter(
            x=bit_labels,
            y=rb_ppls,
            mode='markers+lines',
            name='ResNet-BK',
            marker=dict(size=12, color='#2E86AB'),
            line=dict(width=3),
            hovertemplate='<b>ResNet-BK</b><br>' +
                         'Precision: %{x}<br>' +
                         'PPL: %{y:.2f}<extra></extra>'
        ))
        
        # Plot Mamba
        fig.add_trace(go.Scatter(
            x=bit_labels,
            y=mb_ppls,
            mode='markers+lines',
            name='Mamba',
            marker=dict(size=12, color='#A23B72', symbol='square'),
            line=dict(width=3),
            hovertemplate='<b>Mamba</b><br>' +
                         'Precision: %{x}<br>' +
                         'PPL: %{y:.2f}<extra></extra>'
        ))
        
        # Add deployment threshold
        fig.add_hline(
            y=100,
            line_dash="dash",
            line_color="#F18F01",
            annotation_text="Deployment Threshold (PPL < 100)",
            annotation_position="right"
        )
        
        fig.update_layout(
            title='Quantization Robustness Comparison',
            xaxis_title='Quantization Precision',
            yaxis_title='Perplexity (lower is better)',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_efficiency_plot(self) -> go.Figure:
        """Create interactive efficiency plot."""
        fig = go.Figure()
        
        if 'efficiency' not in self.data:
            fig.add_annotation(
                text="No efficiency data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return fig
        
        eff = self.data['efficiency']
        
        # Get ResNet-BK data
        rb_stats = eff.get('resnetbk', {})
        rb_configs = rb_stats.get('num_configs', 0)
        
        # Get Mamba data
        mb_stats = eff.get('mamba', {})
        mb_configs = mb_stats.get('num_configs', 0)
        
        # For now, show summary statistics
        # In a full implementation, would load individual config data
        
        fig.add_trace(go.Scatter(
            x=[rb_stats.get('mean_flops', 0) / 1e9],
            y=[rb_stats.get('mean_ppl', 0)],
            mode='markers',
            name='ResNet-BK (mean)',
            marker=dict(size=20, color='#2E86AB'),
            hovertemplate='<b>ResNet-BK</b><br>' +
                         'Mean FLOPs: %{x:.2f} G<br>' +
                         'Mean PPL: %{y:.2f}<extra></extra>'
        ))
        
        fig.add_trace(go.Scatter(
            x=[mb_stats.get('mean_flops', 0) / 1e9],
            y=[mb_stats.get('mean_ppl', 0)],
            mode='markers',
            name='Mamba (mean)',
            marker=dict(size=20, color='#A23B72', symbol='square'),
            hovertemplate='<b>Mamba</b><br>' +
                         'Mean FLOPs: %{x:.2f} G<br>' +
                         'Mean PPL: %{y:.2f}<extra></extra>'
        ))
        
        fig.update_layout(
            title='Dynamic Efficiency Comparison',
            xaxis_title='Average FLOPs per Token (GFLOPs)',
            yaxis_title='Perplexity (lower is better)',
            hovermode='closest',
            template='plotly_white',
            height=600
        )
        
        return fig
    
    def create_comparison_table(self) -> html.Div:
        """Create interactive comparison table."""
        if 'comparison_table' not in self.data:
            return html.Div("No comparison table data available")
        
        df = self.data['comparison_table']
        
        # Create table
        table = html.Table([
            html.Thead(
                html.Tr([html.Th(col) for col in df.columns])
            ),
            html.Tbody([
                html.Tr([
                    html.Td(df.iloc[i][col]) for col in df.columns
                ]) for i in range(len(df))
            ])
        ], className='comparison-table')
        
        return table
    
    def create_app(self):
        """Create Dash application."""
        if not DASH_AVAILABLE:
            raise ImportError("dash and plotly are required. Install with: pip install dash plotly")
        
        self.app = dash.Dash(__name__)
        
        # Define layout
        self.app.layout = html.Div([
            html.H1("Mamba-Killer Interactive Dashboard", 
                   style={'textAlign': 'center', 'color': '#2E86AB'}),
            
            html.Div([
                html.P("Compare ResNet-BK and Mamba across three critical dimensions:"),
                html.Ul([
                    html.Li("Long-Context Stability (up to 1M tokens)"),
                    html.Li("Quantization Robustness (FP32 to INT2)"),
                    html.Li("Dynamic Compute Efficiency (PPL vs FLOPs)")
                ])
            ], style={'margin': '20px', 'padding': '20px', 'backgroundColor': '#f0f0f0'}),
            
            # Tabs for different views
            dcc.Tabs(id='tabs', value='stability', children=[
                dcc.Tab(label='Long-Context Stability', value='stability'),
                dcc.Tab(label='Quantization Robustness', value='quantization'),
                dcc.Tab(label='Dynamic Efficiency', value='efficiency'),
                dcc.Tab(label='Comparison Table', value='table'),
            ]),
            
            html.Div(id='tab-content', style={'padding': '20px'}),
            
            # Footer
            html.Div([
                html.Hr(),
                html.P("Interactive Dashboard for Mamba-Killer Ultra-Scale ResNet-BK",
                      style={'textAlign': 'center', 'color': '#666'})
            ])
        ])
        
        # Define callbacks
        @self.app.callback(
            Output('tab-content', 'children'),
            Input('tabs', 'value')
        )
        def render_content(tab):
            if tab == 'stability':
                return dcc.Graph(
                    id='stability-graph',
                    figure=self.create_stability_plot(),
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            elif tab == 'quantization':
                return dcc.Graph(
                    id='quantization-graph',
                    figure=self.create_quantization_plot(),
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            elif tab == 'efficiency':
                return dcc.Graph(
                    id='efficiency-graph',
                    figure=self.create_efficiency_plot(),
                    config={'displayModeBar': True, 'displaylogo': False}
                )
            elif tab == 'table':
                return self.create_comparison_table()
        
        return self.app
    
    def run(self, port: int = 8050, debug: bool = False, open_browser: bool = True):
        """
        Run dashboard server.
        
        Args:
            port: port number
            debug: enable debug mode
            open_browser: automatically open browser
        """
        if self.app is None:
            self.create_app()
        
        # Open browser after short delay
        if open_browser:
            Timer(1.5, lambda: webbrowser.open(f'http://localhost:{port}')).start()
        
        print(f"\n{'='*80}")
        print(f"Starting Interactive Dashboard")
        print(f"{'='*80}")
        print(f"Dashboard URL: http://localhost:{port}")
        print(f"Press Ctrl+C to stop")
        print(f"{'='*80}\n")
        
        self.app.run_server(debug=debug, port=port, host='0.0.0.0')


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Interactive Dashboard for Mamba vs ResNet-BK Comparison"
    )
    
    parser.add_argument(
        '--results_dir',
        type=str,
        default='results/killer_graphs',
        help='Directory containing results JSON files'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8050,
        help='Port number for dashboard server (default: 8050)'
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )
    parser.add_argument(
        '--no_browser',
        action='store_true',
        help='Do not automatically open browser'
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    if not DASH_AVAILABLE:
        print("Error: dash and plotly are required for the interactive dashboard.")
        print("Install with: pip install dash plotly")
        return 1
    
    # Create dashboard
    dashboard = InteractiveDashboard(args.results_dir)
    
    # Run server
    try:
        dashboard.run(
            port=args.port,
            debug=args.debug,
            open_browser=not args.no_browser
        )
    except KeyboardInterrupt:
        print("\n\nDashboard stopped by user")
    except Exception as e:
        print(f"\nError running dashboard: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())
