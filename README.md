
# Intelligent-Portfolio-Management-System

## Live Dashboard
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://intelligent-portfolio-management-system-j2bwr5rbyj6lyqxkl6gove.streamlit.app/)

## Project Overview
A comprehensive machine learning-powered stock trading system that implements advanced algorithmic strategies with professional risk management. This system combines quantitative analysis with AI-driven decision making to deliver consistent trading performance.

## Technical Approach

### Machine Learning Framework
- **Ensemble Modeling**: Combined HistGradientBoosting and RandomForest classifiers
- **Multi-Strategy Architecture**: Three distinct trading approaches tested and optimized
- **Feature Engineering**: 25+ technical indicators including momentum, volatility, and volume-based features
- **Market Regime Detection**: Automated classification of high/medium/low volatility periods

### Trading Methodology
- **Confidence-Based Execution**: Trades only when model confidence exceeds optimized thresholds
- **Dynamic Position Sizing**: 8-15% allocation based on signal strength
- **Stop-Loss Implementation**: 8% automatic risk management
- **Transaction Cost Modeling**: Realistic commission and slippage accounting

### Validation Framework
- **Walk-Forward Testing**: Robust time-series validation across multiple periods
- **Regime-Specific Analysis**: Performance evaluation under different market conditions
- **Statistical Significance**: Comprehensive metrics including Sharpe ratio and maximum drawdown

## Key Results and Performance Metrics

### Performance Highlights
- **Total Return**: +87.9% (Significantly outperforming SPY benchmark)
- **Risk Management**: Maximum drawdown limited to -2.3%
- **Consistency**: 88.7% win rate across all trades
- **Efficiency**: Sharpe ratio of 3.07 indicating superior risk-adjusted returns
- **Market Outperformance**: +64.4% vs SPY benchmark

## Technical Architecture

### Data Processing Pipeline
- Historical price and sentiment data collection and cleaning from Alpha Vantage API
- Technical indicator calculation (moving averages, RSI, momentum oscillators)
- Feature selection using correlation analysis
- Market regime classification

### Model Development
- Target variable engineering for multiple trading strategies
- Class imbalance handling using SMOTE
- Hyperparameter optimization through cross-validation
- Ensemble model selection based on walk-forward performance

### Backtesting Engine
- Realistic trading simulation with transaction costs
- Stop-loss implementation and position sizing
- Portfolio-level risk management
- Comprehensive performance reporting

## Technology Stack

- **Programming**: Python 3.8+
- **Machine Learning**: Scikit-learn, Imbalanced-learn
- **Data Processing**: Pandas, NumPy, SQLite
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Dashboard**: Streamlit
- **Model Persistence**: Joblib

## Project Structure

```
Intelligent-Portfolio-Management-System/
├── notebooks/          # Analysis pipeline
├── src/               # Source code
├── data/              # Database files
├── models/            # Trained ML models
├── results/           # Performance reports
├── streamlit_app.py   # Web dashboard
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Getting Started

### Prerequisites
- Python 3.8 or higher
- Jupyter Notebook environment
- Required packages in requirements.txt

### Installation
```bash
git clone https://github.com/yourusername/Intelligent-Portfolio-Management-System.git
cd Intelligent-Portfolio-Management-System
pip install -r requirements.txt
```

### Execution Pipeline
1. Run feature engineering notebook
2. Execute model training and selection
3. Perform backtesting analysis
4. Conduct validation testing
5. Launch Streamlit dashboard

## Web Application Features

The interactive dashboard provides:
- Real-time portfolio performance visualization
- Trade analysis and statistics
- Model performance metrics
- Risk management insights
- Current market signals

## Validation and Robustness

The system has been rigorously tested through:
- Multiple market regime analysis
- Walk-forward validation across different time periods
- Symbol-level performance consistency checks
- Statistical significance testing of results

*For detailed implementation and performance analysis, visit the live dashboard or explore the source code in the repository.*
