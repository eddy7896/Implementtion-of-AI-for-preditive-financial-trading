# 📈 Stock Predictor with Advanced Technical Indicators

A professional-grade stock price prediction system using LSTM neural networks and comprehensive technical analysis. Features a user-friendly GUI with 29+ technical indicators and advanced mathematical formulas.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)

## 🌟 Key Features

### 📊 **Professional Technical Analysis**
- **29+ Technical Indicators** with proper mathematical formulas
- **6 Categories**: Moving Averages, Momentum, Trend, Volatility, Volume, Oscillators
- **Smart Defaults**: Pre-selected balanced indicator combinations
- **Selective Calculation**: Performance-optimized to calculate only selected indicators

### 🤖 **Advanced AI Prediction**
- **LSTM Neural Network** for time series forecasting
- **7-Day Price Predictions** with confidence intervals
- **Multiple Model Support**: LSTM with fallback linear regression
- **Feature Engineering**: Combines price data with technical indicators

### 🖥️ **Professional GUI Interface**
- **User-Friendly Design** with organized categories and controls
- **Real-Time Visualization**: Interactive charts and candlestick plots
- **Progress Tracking**: Live updates during prediction process
- **Export Capabilities**: Save results, charts, and console logs

### 📈 **Comprehensive Analysis**
- **Technical Analysis Summary**: RSI, MACD, Bollinger Bands analysis
- **Trend Identification**: Multiple timeframe trend analysis
- **Volatility Measurement**: ATR, Bollinger Band width
- **Volume Analysis**: OBV, VWAP, accumulation/distribution

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+** (3.9 or 3.10 recommended)
- **Windows, macOS, or Linux**
- **Internet connection** (for downloading stock data)

### Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/your-username/stock-predictor.git
cd stock-predictor
```

#### 2. Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. Install Required Dependencies
```bash
# Install all requirements from requirements.txt
pip install -r requirements.txt
```

**Manual Installation (Alternative):**
```bash
# Core libraries
pip install pandas numpy matplotlib scikit-learn
pip install yfinance ta tensorflow keras
pip install seaborn plotly  # Optional: for enhanced plotting

# For development/testing
pip install pytest jupyter  # Optional: for development
```

#### 4. Verify Installation
```bash
# Test the system
python test_comprehensive_indicators.py
```

### 🎯 Quick Launch

#### Start the GUI Application
```bash
python stock_predictor_gui.py
```

#### Command Line Usage
```bash
python stock_predictor_lstm.py
```

## 📖 Usage Guide

### GUI Application Walkthrough

1. **Launch the Application**
   ```bash
   python stock_predictor_gui.py
   ```

2. **Configure Stock Settings**
   - Enter stock ticker (e.g., AAPL, MSFT, GOOGL)
   - Set historical data range
   - Specify prediction start date

3. **Select Technical Indicators**
   - Choose from 29+ professional indicators
   - Use "Reset Defaults" for balanced selection
   - Or "Select All" for comprehensive analysis

4. **Configure Model Parameters**
   - Lookback days: 30-120 (default: 60)
   - Training epochs: 20-100 (default: 50)

5. **Run Prediction**
   - Click "Start Prediction"
   - Monitor progress in real-time
   - View results in multiple tabs

### Example: Predicting Apple Stock
```python
from stock_predictor_lstm import StockPredictor

# Initialize with selected indicators
predictor = StockPredictor(
    ticker="AAPL", 
    lookback_days=60,
    selected_indicators=['sma_20', 'rsi', 'macd', 'bb_upper', 'bb_lower']
)

# Fetch and analyze data
predictor.fetch_data("2020-01-01", "2025-08-01")
predictor.calculate_technical_indicators()
predictor.prepare_lstm_data()

# Train model and predict
predictor.build_lstm_model()
predictor.train_model(epochs=50)
dates, prices = predictor.predict_next_week("2025-08-01")
```

## 📊 Technical Indicators Available

### 📈 **Moving Averages (6 indicators)**
| Indicator | Period | Description |
|-----------|--------|-------------|
| SMA 5-day | 5 | Short-term trend (1 week) |
| SMA 10-day | 10 | Medium-short trend (2 weeks) |
| SMA 20-day | 20 | Medium-term trend (1 month) |
| SMA 50-day | 50 | Long-term trend (2.5 months) |
| EMA 12-day | 12 | Fast exponential moving average |
| EMA 26-day | 26 | Slow exponential moving average |

### ⚡ **Momentum Indicators (5 indicators)**
| Indicator | Period | Range | Description |
|-----------|--------|-------|-------------|
| RSI | 14 | 0-100 | Relative Strength Index |
| Stochastic %K | 14 | 0-100 | Fast stochastic oscillator |
| Stochastic %D | 3 | 0-100 | Slow stochastic (smoothed) |
| Williams %R | 14 | -100 to 0 | Williams Percent Range |
| Rate of Change | 12 | % | Percentage price change |

### 📊 **Trend Indicators (6 indicators)**
| Indicator | Parameters | Description |
|-----------|------------|-------------|
| MACD Line | 12,26 | Moving Average Convergence Divergence |
| MACD Signal | 9 | Signal line for MACD crossovers |
| MACD Histogram | - | Difference between MACD and Signal |
| ADX | 14 | Average Directional Index (trend strength) |
| Aroon Up | 14 | Measures time since highest high |
| Aroon Down | 14 | Measures time since lowest low |

### 🌊 **Volatility Indicators (5 indicators)**
| Indicator | Parameters | Description |
|-----------|------------|-------------|
| Bollinger Upper | 20,2 | Upper volatility band |
| Bollinger Middle | 20 | Middle band (SMA) |
| Bollinger Lower | 20,2 | Lower volatility band |
| Bollinger Width | - | Band width (volatility measure) |
| ATR | 14 | Average True Range |

### 📦 **Volume Indicators (4 indicators)**
| Indicator | Description |
|-----------|-------------|
| Volume SMA | 20-period volume moving average |
| OBV | On-Balance Volume (cumulative volume flow) |
| VWAP | Volume Weighted Average Price |
| A/D Line | Accumulation/Distribution Line |

### 💹 **Additional Indicators (3 indicators)**
| Indicator | Description |
|-----------|-------------|
| High-Low % | Daily trading range percentage |
| Open-Close % | Daily price change percentage |
| CCI | Commodity Channel Index (cyclical trends) |

## 🏗️ System Architecture

```
┌─────────────────────┐    ┌──────────────────────┐    ┌─────────────────────┐
│   GUI Interface     │    │  StockPredictor      │    │ TechnicalIndicators │
│                     │───▶│    (Coordinator)     │───▶│   (Calculations)    │
│ • 29 Checkboxes     │    │ • Selection Mapping  │    │ • Math Formulas     │
│ • Smart Defaults    │    │ • Data Preparation   │    │ • Vectorized Ops    │
│ • Category Groups   │    │ • LSTM Integration   │    │ • Professional Impl │
└─────────────────────┘    └──────────────────────┘    └─────────────────────┘
```

### Core Components

1. **`technical_indicators.py`** - Mathematical calculation engine
2. **`stock_predictor_lstm.py`** - Coordinator and LSTM model
3. **`stock_predictor_gui.py`** - Professional GUI interface
4. **`test_comprehensive_indicators.py`** - Complete test suite

## 📝 Example Output

### Weekly Prediction Results
```
WEEKLY STOCK PRICE PREDICTIONS
============================================================
Stock: AAPL
Historical data ends: 2025-08-01
Last known price: $227.52

7-Day Forecast:
------------------------------------------------------------
Day  Date         Day Name   Price        Change   Change%
------------------------------------------------------------
1    2025-08-04   Mon        $229.15     +$1.63   +0.72%
2    2025-08-05   Tue        $231.28     +$3.76   +1.65%
3    2025-08-06   Wed        $228.94     +$1.42   +0.62%
4    2025-08-07   Thu        $230.67     +$3.15   +1.38%
5    2025-08-08   Fri        $232.11     +$4.59   +2.02%

WEEK SUMMARY:
Expected High:  $232.11
Expected Low:   $228.94
Week End Price: $232.11
Expected Weekly Return: +2.02%
Weekly Trend: BULLISH (Upward)
```

### Technical Analysis Summary
```
TECHNICAL ANALYSIS SUMMARY for AAPL
==================================================
Date: 2025-08-01
Current Price: $227.52

RSI: 58.43 (Neutral)
MA Trend: Short-term Uptrend
SMA20: $225.18, SMA50: $220.45
MACD: Bullish (MACD: 2.1847, Signal: 1.9203)
Bollinger Bands: Within bands (neutral)
```

## 🧪 Testing

### Run Complete Test Suite
```bash
# Test all indicators and integration
python test_comprehensive_indicators.py

# Test selective indicators
python test_selective_indicators.py

# Test specific functionality
pytest tests/ -v
```

### Expected Test Results
```
COMPREHENSIVE TECHNICAL INDICATORS TEST SUITE
============================================================
[SUCCESS] Technical Indicators Module: 28/28 working
[SUCCESS] StockPredictor Integration: Completed  
[SUCCESS] Indicator Descriptions: Completed

All tests completed successfully!
The technical indicators system is ready for use!
```

## 📁 Project Structure

```
stock-predictor/
├── README.md                              # This file
├── requirements.txt                       # Python dependencies
├── stock_predictor_gui.py                # Main GUI application
├── stock_predictor_lstm.py               # LSTM model and coordinator
├── technical_indicators.py               # Mathematical calculation engine
├── prediction_config.py                  # Configuration settings
├── test_comprehensive_indicators.py      # Complete test suite
├── test_selective_indicators.py          # Indicator selection tests
├── TECHNICAL_INDICATORS_ARCHITECTURE.md  # Detailed architecture docs
├── TECHNICAL_INDICATORS_SUMMARY.md       # Project summary
└── demo_weekly_prediction.py            # Demo script
```

## 🔧 Configuration

### Model Parameters (in GUI or `prediction_config.py`)
```python
MODEL_CONFIG = {
    'lookback_days': 60,        # Historical days for prediction
    'prediction_days': 7,       # Days to predict (weekly)
    'epochs': 50,              # LSTM training epochs
    'batch_size': 32,          # Training batch size
    'test_size': 0.2          # Train/test split ratio
}
```

### Stock Configuration
```python
STOCK_CONFIGS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    }
}
```

## 🚨 Troubleshooting

### Common Issues and Solutions

#### 1. **TensorFlow Installation Issues**
```bash
# For CPU-only version
pip install tensorflow-cpu

# For GPU support (if available)
pip install tensorflow-gpu

# Alternative: Use conda
conda install tensorflow
```

#### 2. **yfinance Data Download Errors**
```bash
# Update yfinance to latest version
pip install --upgrade yfinance

# Alternative: Use different data source
# The system includes fallback data handling
```

#### 3. **GUI Not Starting (tkinter issues)**
```bash
# On Ubuntu/Debian
sudo apt-get install python3-tk

# On macOS with Homebrew
brew install python-tk

# Windows: tkinter included with Python
```

#### 4. **Memory Issues with Large Datasets**
- Reduce the number of selected indicators
- Decrease lookback_days (30-60 recommended)
- Use shorter historical data periods

### Performance Optimization Tips

1. **For Faster Calculations:**
   - Select fewer indicators (10-15 optimal)
   - Use shorter lookback periods (30-60 days)
   - Reduce training epochs (20-30 for testing)

2. **For Better Accuracy:**
   - Use more historical data (3+ years)
   - Include comprehensive indicator sets
   - Increase training epochs (50-100)

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. **Fork the repository**
2. **Create a feature branch** (`git checkout -b feature/new-indicator`)
3. **Make your changes** with proper documentation
4. **Add tests** for new functionality
5. **Submit a pull request**

### Adding New Technical Indicators

1. **Implement in `technical_indicators.py`:**
   ```python
   def new_indicator(self, period=14):
       """
       New Indicator Description
       
       Formula: [Mathematical formula here]
       """
       # Implementation
       return calculated_values
   ```

2. **Add to GUI selection** in `stock_predictor_gui.py`
3. **Add to calculation mapping** in `stock_predictor_lstm.py`
4. **Update tests** and documentation

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. **Do not use for actual trading without proper risk management.** Past performance does not guarantee future results. The authors are not responsible for any financial losses.

## 🙏 Acknowledgments

- **yfinance** - For providing free stock market data
- **TensorFlow/Keras** - For deep learning capabilities  
- **pandas/numpy** - For data manipulation and mathematical operations
- **matplotlib** - For visualization capabilities
- **TA-Lib community** - For technical analysis inspiration

## 📞 Support

For issues, questions, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/your-username/stock-predictor/issues)
- **Documentation**: See `TECHNICAL_INDICATORS_ARCHITECTURE.md` for detailed system docs
- **Email**: [your-email@domain.com]

---

**Built with ❤️ for the trading and machine learning community**

*Happy Trading! 📈*