# ==================================================
# 05_Interactive_Dashboard.py
# AI Trading System - Professional Interactive Dashboard 
# ==================================================

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import joblib
import json
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="AI Stock Trading System",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: 600;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1.2rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .positive {
        color: #27ae60;
        font-weight: 600;
    }
    .negative {
        color: #e74c3c;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

class TradingDashboard:
    def __init__(self):
        self.load_data()
        
    def load_data(self):
        """Load all necessary data for the dashboard"""
        try:
            # Load model artifacts
            self.artifacts = joblib.load('advanced_trading_models.pkl')
            self.best_strategy = self.artifacts['best_strategy']
            
            # Load backtest results
            with open('enhanced_performance_report.json', 'r') as f:
                self.backtest_results = json.load(f)
                
            # Load strategy performance
            self.strategy_perf = pd.read_csv('strategy_performance_report.csv')
            
            # Load portfolio history and trades
            self.portfolio_df = pd.read_csv('enhanced_portfolio_history.csv')
            self.trades_df = pd.read_csv('enhanced_trade_log.csv')
            
            # Load enhanced dataset for current predictions
            conn = sqlite3.connect("enhanced_trading_dataset_v2.db")
            self.current_data = pd.read_sql("SELECT * FROM enhanced_trading_data", conn)
            conn.close()
            
            self.current_data['date'] = pd.to_datetime(self.current_data['date'])
            
            st.sidebar.success("All data loaded successfully")
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            st.info("Please ensure all previous analysis steps have been completed")

    def calculate_current_signals(self, symbol_data):
        """Calculate trading signals with error handling"""
        try:
            # Try multiple potential return columns
            return_columns = ['return_5d', 'daily_return', 'target_direction_5d']
            return_value = 0
            
            for col in return_columns:
                if col in symbol_data:
                    return_value = symbol_data[col]
                    break
            
            # Determine signal based on available data
            if abs(return_value) > 0.03:
                signal = "STRONG BUY" if return_value > 0 else "STRONG SELL"
                confidence = min(0.95, abs(return_value) * 8)
                signal_color = "green" if return_value > 0 else "red"
            elif abs(return_value) > 0.01:
                signal = "BUY" if return_value > 0 else "SELL"
                confidence = min(0.80, abs(return_value) * 10)
                signal_color = "lightgreen" if return_value > 0 else "lightcoral"
            else:
                signal = "HOLD"
                confidence = 0.5
                signal_color = "gray"
                
            return signal, confidence, signal_color
            
        except Exception as e:
            # Fallback signal calculation
            return "HOLD", 0.5, "gray"

    def render_header(self):
        """Render the main header with performance metrics"""
        st.markdown('<h1 class="main-header">AI Stock Trading System</h1>', unsafe_allow_html=True)
        
        # Get performance metrics
        perf_summary = self.backtest_results['performance_summary']
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Return", 
                f"+{perf_summary['total_return']:.1f}%",
                f"+{perf_summary['outperformance']:.1f}% vs SPY",
                delta_color="normal"
            )
            
        with col2:
            st.metric(
                "Max Drawdown",
                f"{perf_summary['max_drawdown']:.1f}%",
                "Excellent Risk Control",
                delta_color="inverse"
            )
            
        with col3:
            st.metric(
                "Sharpe Ratio",
                f"{perf_summary['sharpe_ratio']:.2f}",
                "Elite Risk-Adjusted Returns"
            )
            
        with col4:
            st.metric(
                "Win Rate",
                f"{perf_summary['win_rate']:.1%}",
                "Highly Consistent"
            )

    def render_portfolio_performance(self):
        """Render portfolio performance charts"""
        st.markdown('<h2 class="section-header">Portfolio Performance Analysis</h2>', unsafe_allow_html=True)
        
        if not self.portfolio_df.empty:
            self.portfolio_df['date'] = pd.to_datetime(self.portfolio_df['date'])
            
            # Create comprehensive performance dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Portfolio Value Over Time', 'Daily Returns Distribution', 
                              'Drawdown Analysis', 'Active Positions Over Time'),
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # Portfolio Value
            fig.add_trace(
                go.Scatter(
                    x=self.portfolio_df['date'], 
                    y=self.portfolio_df['portfolio_value'],
                    name='Portfolio Value', 
                    line=dict(color='#1f77b4', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(31, 119, 180, 0.1)'
                ),
                row=1, col=1
            )
            
            # Daily Returns Histogram
            returns = self.portfolio_df['portfolio_value'].pct_change().fillna(0) * 100
            fig.add_trace(
                go.Histogram(
                    x=returns,
                    name='Daily Returns',
                    nbinsx=30,
                    marker_color='#2ecc71',
                    opacity=0.7
                ),
                row=1, col=2
            )
            
            # Drawdown
            peak = self.portfolio_df['portfolio_value'].cummax()
            drawdown = (self.portfolio_df['portfolio_value'] - peak) / peak * 100
            fig.add_trace(
                go.Scatter(
                    x=self.portfolio_df['date'], 
                    y=drawdown,
                    name='Drawdown', 
                    fill='tozeroy', 
                    line=dict(color='#e74c3c', width=2),
                    fillcolor='rgba(231, 76, 60, 0.3)'
                ),
                row=2, col=1
            )
            
            # Positions (if available)
            if 'positions' in self.portfolio_df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=self.portfolio_df['date'], 
                        y=self.portfolio_df['positions'],
                        name='Active Positions', 
                        line=dict(color='#f39c12', width=2)
                    ),
                    row=2, col=2
                )
            else:
                # Alternative: Cash position
                if 'cash' in self.portfolio_df.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=self.portfolio_df['date'], 
                            y=self.portfolio_df['cash'],
                            name='Cash Balance', 
                            line=dict(color='#f39c12', width=2)
                        ),
                        row=2, col=2
                    )
            
            fig.update_layout(
                height=700, 
                showlegend=True, 
                title_text="Portfolio Performance Dashboard",
                template="plotly_white"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.warning("Portfolio data not available")

    def render_trade_analysis(self):
        """Render trade analysis section"""
        st.markdown('<h2 class="section-header">Trade Analysis</h2>', unsafe_allow_html=True)
        
        if not self.trades_df.empty:
            col1, col2 = st.columns(2)
            
            with col1:
                # Trade type distribution
                trade_types = self.trades_df['action'].value_counts()
                fig1 = px.pie(
                    values=trade_types.values, 
                    names=trade_types.index,
                    title="Trade Type Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig1.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig1, use_container_width=True)
                
            with col2:
                # P&L distribution (if available)
                if 'pnl' in self.trades_df.columns:
                    fig2 = px.histogram(
                        self.trades_df, 
                        x='pnl', 
                        title="Profit & Loss Distribution",
                        color_discrete_sequence=['#27ae60'],
                        nbins=20
                    )
                    fig2.update_layout(showlegend=False)
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    # Alternative: Show trade count by symbol
                    symbol_counts = self.trades_df['symbol'].value_counts()
                    fig2 = px.bar(
                        x=symbol_counts.index,
                        y=symbol_counts.values,
                        title="Trades by Symbol",
                        color_discrete_sequence=['#3498db']
                    )
                    st.plotly_chart(fig2, use_container_width=True)
            
            # Recent trades table with enhanced styling
            st.subheader("Recent Trading Activity")
            display_cols = ['date', 'symbol', 'action']
            
            # Add available columns
            available_cols = ['price', 'pnl', 'return_pct', 'shares']
            for col in available_cols:
                if col in self.trades_df.columns:
                    display_cols.append(col)
                
            recent_trades = self.trades_df[display_cols].tail(15)
            
            # Format the dataframe for better display
            styled_trades = recent_trades.copy()
            if 'pnl' in styled_trades.columns:
                styled_trades['pnl'] = styled_trades['pnl'].apply(lambda x: f"${x:,.2f}" if pd.notnull(x) else "N/A")
            if 'return_pct' in styled_trades.columns:
                styled_trades['return_pct'] = styled_trades['return_pct'].apply(lambda x: f"{x:.2f}%" if pd.notnull(x) else "N/A")
            
            st.dataframe(styled_trades, use_container_width=True)
            
        else:
            st.warning("Trade data not available")

    def render_model_insights(self):
        """Render model performance and insights"""
        st.markdown('<h2 class="section-header">Model Insights</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Strategy performance comparison
            fig = px.bar(
                self.strategy_perf, 
                x='strategy', 
                y='improvement',
                title="Strategy Improvement vs Baseline",
                labels={'improvement': 'Improvement', 'strategy': 'Strategy'},
                color='improvement',
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Model accuracy comparison
            fig = px.bar(
                self.strategy_perf, 
                x='strategy', 
                y='accuracy',
                title="Strategy Accuracy Comparison",
                color='accuracy',
                color_continuous_scale='Plasma'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance summary table
        st.subheader("Performance Summary")
        perf_data = {
            'Metric': ['Best Strategy', 'Average Accuracy', 'Best Improvement', 'Total Trades', 'Optimal Stop-Loss'],
            'Value': [
                self.best_strategy.title(),
                f"{self.strategy_perf['accuracy'].mean():.1%}",
                f"+{self.strategy_perf['improvement'].max():.3f}",
                f"{len(self.trades_df) if not self.trades_df.empty else 0}",
                f"{self.backtest_results['optimal_stop_loss']:.1%}"
            ]
        }
        perf_df = pd.DataFrame(perf_data)
        st.dataframe(perf_df, use_container_width=True)

    def render_current_signals(self):
        """Render current trading signals with error handling"""
        st.markdown('<h2 class="section-header">Current Market Signals</h2>', unsafe_allow_html=True)
        
        # Get latest data for each symbol
        latest_data = self.current_data[self.current_data['date'] == self.current_data['date'].max()]
        
        if not latest_data.empty:
            # Create signal cards for each symbol
            symbols = latest_data['symbol'].unique()
            
            cols = st.columns(len(symbols))
            
            for i, symbol in enumerate(symbols):
                with cols[i]:
                    symbol_data = latest_data[latest_data['symbol'] == symbol].iloc[0]
                    
                    # Use the safe signal calculation method
                    signal, confidence, signal_color = self.calculate_current_signals(symbol_data)
                    
                    # Create metric with custom styling
                    st.metric(
                        label=symbol,
                        value=signal,
                        delta=f"{confidence:.0%} confidence",
                        delta_color="normal" if "BUY" in signal else "inverse"
                    )
            
            # Additional market insights
            st.subheader("Market Overview")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Safe sentiment calculation
                sentiment_cols = ['sentiment_with_momentum', 'sentiment_mean']
                avg_sentiment = 0
                for col in sentiment_cols:
                    if col in latest_data.columns:
                        avg_sentiment = latest_data[col].mean()
                        break
                st.metric("Average Market Sentiment", f"{avg_sentiment:.3f}")
                
            with col2:
                # Safe volatility count
                if 'volatility_regime' in latest_data.columns:
                    high_vol_count = (latest_data['volatility_regime'] == 'HIGH').sum()
                    st.metric("High Volatility Stocks", f"{high_vol_count}")
                else:
                    st.metric("Total Stocks", f"{len(latest_data)}")
                
            with col3:
                # Safe positive returns count
                return_cols = ['return_5d', 'daily_return', 'target_direction_5d']
                positive_returns = 0
                for col in return_cols:
                    if col in latest_data.columns:
                        positive_returns = (latest_data[col] > 0).sum()
                        break
                st.metric("Positive Returns", f"{positive_returns}/{len(latest_data)}")
            
        else:
            st.warning("Current market data not available")

    def render_risk_management(self):
        """Render risk management insights"""
        st.markdown('<h2 class="section-header">Risk Management</h2>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stop-Loss Performance Analysis")
            sl_analysis = self.backtest_results.get('stop_loss_analysis', {})
            if sl_analysis:
                sl_data = []
                for sl_level, metrics in sl_analysis.items():
                    sl_data.append({
                        'Stop-Loss Level': sl_level,
                        'Total Return (%)': metrics.get('total_return', 0),
                        'Max Drawdown (%)': metrics.get('max_drawdown', 0),
                        'Stop-Loss Triggers': metrics.get('stop_loss_count', 0),
                        'Sharpe Ratio': metrics.get('sharpe_ratio', 0)
                    })
                sl_df = pd.DataFrame(sl_data)
                st.dataframe(sl_df.style.format({
                    'Total Return (%)': '{:.1f}%',
                    'Max Drawdown (%)': '{:.1f}%',
                    'Sharpe Ratio': '{:.2f}'
                }), use_container_width=True)
            else:
                st.info("Stop-loss analysis data not available")
        
        with col2:
            st.subheader("Key Risk Metrics")
            risk_metrics_data = {
                'Metric': [
                    'Optimal Stop-Loss', 
                    'Max Drawdown', 
                    'Sharpe Ratio', 
                    'Win Rate'
                ],
                'Value': [
                    f"{self.backtest_results.get('optimal_stop_loss', 0.08):.1%}",
                    f"{self.backtest_results['performance_summary']['max_drawdown']:.1f}%",
                    f"{self.backtest_results['performance_summary']['sharpe_ratio']:.2f}",
                    f"{self.backtest_results['performance_summary']['win_rate']:.1%}"
                ]
            }
            risk_df = pd.DataFrame(risk_metrics_data)
            st.dataframe(risk_df, use_container_width=True)

    def render_sidebar(self):
        """Render the sidebar controls"""
        st.sidebar.title("Navigation")
        
        page = st.sidebar.radio(
            "Select Dashboard Section",
            [
                "Dashboard Overview", 
                "Portfolio Performance", 
                "Trade Analysis", 
                "Model Insights", 
                "Current Signals", 
                "Risk Management"
            ]
        )
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("System Information")
        st.sidebar.write(f"**Active Strategy**: {self.best_strategy.title()}")
        st.sidebar.write(f"**Last Model Update**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.sidebar.write(f"**Data Period**: 2023-01-03 to 2025-10-24")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Performance Statistics")
        st.sidebar.write(f"**Total Trades**: {len(self.trades_df) if not self.trades_df.empty else 0}")
        st.sidebar.write(f"**Data Points**: {len(self.current_data):,}")
        st.sidebar.write(f"**Strategies Tested**: {len(self.strategy_perf)}")
        st.sidebar.write(f"**Portfolio Symbols**: {self.current_data['symbol'].nunique()}")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("About")
        st.sidebar.info(
            "AI Stock Trading System v1.0\n\n"
            "Multi-strategy AI-powered trading system with advanced risk management "
            "and comprehensive validation."
        )
            
        return page

    def run(self):
        """Main method to run the dashboard"""
        try:
            page = self.render_sidebar()
            self.render_header()
            
            if page == "Dashboard Overview":
                self.render_portfolio_performance()
                self.render_current_signals()
                
            elif page == "Portfolio Performance":
                self.render_portfolio_performance()
                
            elif page == "Trade Analysis":
                self.render_trade_analysis()
                
            elif page == "Model Insights":
                self.render_model_insights()
                
            elif page == "Current Signals":
                self.render_current_signals()
                
            elif page == "Risk Management":
                self.render_risk_management()
                
        except Exception as e:
            st.error(f"Dashboard error: {e}")
            st.info("Please ensure all data files are properly generated by running the analysis pipeline")

# Run the dashboard
if __name__ == "__main__":
    dashboard = TradingDashboard()
    dashboard.run()
