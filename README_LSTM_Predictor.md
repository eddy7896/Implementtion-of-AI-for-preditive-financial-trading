# Stock Price Prediction with LSTM and Technical Indicators

This implementation provides a comprehensive stock price prediction system that uses **LSTM neural networks** and **technical indicators** to predict stock prices for the next 7 trading days from a given historical data end date.

## 🎯 Key Features

- **LSTM Neural Network**: Deep learning model for time series prediction
- **26+ Technical Indicators**: RSI, MACD, Bollinger Bands, Moving Averages, Stochastic Oscillator, etc.
- **Weekly Predictions**: Predicts exactly 7 trading days (excluding weekends)
- **Comprehensive Visualization**: Charts showing historical data, indicators, and predictions
- **Model Evaluation**: Performance metrics including RMSE, MAE, and MAPE
- **Fallback Method**: Linear regression backup when LSTM is unavailable

## 📁 Files Overview

- `stock_predictor_lstm.py` - Main prediction system with StockPredictor class
- `demo_weekly_prediction.py` - Demonstration script showing complete workflow
- `test_weekly_prediction.py` - Comprehensive testing script
- `prediction_config.py` - Configuration settings and stock parameters

## 🚀 Quick Start

### Basic Usage Example

```python
from stock_predictor_lstm import StockPredictor

# Initialize predictor
predictor = StockPredictor("AAPL", lookback_days=60, prediction_days=7)

# Fetch historical data
predictor.fetch_data("2020-01-01", "2025-08-01")

# Calculate technical indicators
predictor.calculate_technical_indicators()

# Prepare and train model
predictor.prepare_lstm_data()
predictor.build_lstm_model()
predictor.train_model(epochs=50)

# Make predictions for next week
dates, prices = predictor.predict_next_week("2025-08-01")

# Create visualization
predictor.plot_predictions(dates, prices)
```

### Run Complete Demo

```bash
python demo_weekly_prediction.py
```

### Run Tests

```bash
python test_weekly_prediction.py
```

## 📊 Technical Indicators Used

The system calculates 26+ technical indicators:

### Price-Based Indicators
- **SMA** (Simple Moving Averages): 5, 10, 20, 50-day periods
- **EMA** (Exponential Moving Averages): 12, 26-day periods
- **Bollinger Bands**: Upper, Middle, Lower bands + Band Width
- **Price Ratios**: High-Low percentage, Open-Close percentage

### Momentum Indicators
- **RSI** (Relative Strength Index): 14-day period
- **Stochastic Oscillator**: %K and %D lines
- **MACD**: Main line, Signal line, Histogram

### Volume Indicators
- **OBV** (On-Balance Volume)
- **Volume SMA**: 20-day average volume
- **Volume Analysis**: Volume vs price correlation

### Volatility Indicators
- **ATR** (Average True Range): 14-day period
- **Bollinger Band Width**: Volatility measure

## 🧠 LSTM Model Architecture

```
Model: Sequential
┌─────────────────────────────────┐
│ LSTM Layer 1 (50 neurons)      │ ← return_sequences=True
├─────────────────────────────────┤
│ Dropout (0.2)                  │
├─────────────────────────────────┤
│ LSTM Layer 2 (50 neurons)      │ ← return_sequences=True
├─────────────────────────────────┤
│ Dropout (0.2)                  │
├─────────────────────────────────┤
│ LSTM Layer 3 (50 neurons)      │ ← return_sequences=False
├─────────────────────────────────┤
│ Dropout (0.2)                  │
├─────────────────────────────────┤
│ Dense Layer (25 neurons)       │
├─────────────────────────────────┤
│ Output Layer (1 neuron)        │ ← Price prediction
└─────────────────────────────────┘

Input Shape: (batch_size, 60, 26)
- 60 days of historical data
- 26 technical indicator features
```

## 📈 Prediction Process

### Step 1: Data Preparation
1. Fetch historical stock data from Yahoo Finance
2. Calculate 26+ technical indicators
3. Clean and normalize data
4. Create 60-day sliding windows

### Step 2: Model Training
1. Split data into train/validation/test sets
2. Scale features using MinMaxScaler
3. Train LSTM model with early stopping
4. Evaluate performance on test set

### Step 3: Prediction Generation
1. Use last 60 days of data as input
2. Generate predictions iteratively for 7 days
3. Account for feature interdependencies
4. Convert scaled predictions back to actual prices

### Step 4: Visualization
1. Plot historical prices with technical indicators
2. Display 7-day predictions with confidence intervals
3. Show performance metrics and analysis

## 📅 Example Scenario

**Input**: Historical data ends on `2025-08-01` (Friday)
**Output**: Predictions for the following week:

```
Historical Data End: 2025-08-01 (Friday)
Last Known Price: $225.50

7-Day Forecast:
+----------------------------------------------------------+
| Day |   Date     | Day Name | Predicted Price | Change% |
+----------------------------------------------------------+
|  1  | 2025-08-04 | Monday   |     $227.80     |  +1.02% |
|  2  | 2025-08-05 | Tuesday  |     $229.15     |  +1.62% |
|  3  | 2025-08-06 | Wednesday|     $228.90     |  +1.51% |
|  4  | 2025-08-07 | Thursday |     $230.25     |  +2.11% |
|  5  | 2025-08-08 | Friday   |     $231.40     |  +2.62% |
+----------------------------------------------------------+

Week Summary:
Expected High:  $231.40
Expected Low:   $227.80
Expected Weekly Return: +2.62%
Weekly Trend: 📈 Bullish (Upward)
```

## 🎛️ Configuration Options

### Model Parameters
```python
MODEL_CONFIG = {
    'lookback_days': 60,        # Historical days for prediction
    'prediction_days': 7,       # Days to predict (weekly)
    'epochs': 50,              # Training epochs
    'batch_size': 32,          # Training batch size
    'validation_split': 0.1,   # Validation data percentage
    'test_size': 0.2          # Test data percentage
}
```

### Supported Stocks
- **AAPL** - Apple Inc.
- **MSFT** - Microsoft Corporation
- **GOOGL** - Alphabet Inc.
- **TSLA** - Tesla Inc.
- **NVDA** - NVIDIA Corporation
- Any stock symbol available on Yahoo Finance

## 📊 Performance Metrics

The system provides comprehensive performance evaluation:

- **RMSE** (Root Mean Square Error): Overall prediction accuracy
- **MAE** (Mean Absolute Error): Average prediction error
- **MAPE** (Mean Absolute Percentage Error): Percentage-based accuracy
- **Directional Accuracy**: Percentage of correct trend predictions

## 🔧 Dependencies

```bash
pip install yfinance pandas numpy matplotlib scikit-learn tensorflow ta
```

## 🚀 Advanced Usage

### Custom Stock Analysis
```python
# Analyze any stock with custom parameters
predictor = StockPredictor(
    ticker="YOUR_STOCK", 
    lookback_days=90,    # Longer history
    prediction_days=14   # Two-week prediction
)
```

### Batch Processing
```python
# Analyze multiple stocks
stocks = ["AAPL", "MSFT", "GOOGL", "TSLA"]
for ticker in stocks:
    predictor = StockPredictor(ticker)
    # ... run prediction process
```

### Custom Indicators
```python
# Add your own technical indicators
def custom_indicator(df):
    return df['Close'].rolling(10).std()

# Integrate into the prediction pipeline
predictor.df['Custom_Indicator'] = custom_indicator(predictor.df)
predictor.features.append('Custom_Indicator')
```

## ⚠️ Important Notes

### Market Considerations
- Predictions are for **trading days only** (excludes weekends/holidays)
- Model works best with liquid, actively traded stocks
- External factors (news, earnings) can affect predictions
- Past performance doesn't guarantee future results

### Technical Limitations
- Requires significant historical data (3+ years recommended)
- GPU recommended for faster LSTM training
- Internet connection required for data fetching
- Model retraining needed for different time periods

### Risk Disclaimer
This tool is for **educational and research purposes only**. Stock market predictions are inherently uncertain and should not be used as the sole basis for investment decisions. Always conduct your own research and consider consulting financial professionals.

## 🔄 Model Updates

### Retraining Schedule
- **Daily**: Update predictions with latest data
- **Weekly**: Retrain model with new data
- **Monthly**: Evaluate and optimize model parameters
- **Quarterly**: Review and update technical indicators

### Performance Monitoring
```python
# Monitor prediction accuracy
predictor.evaluate_model()

# Track prediction vs actual results
actual_prices = get_actual_prices(prediction_dates)
accuracy = calculate_accuracy(predicted_prices, actual_prices)
```

## 📞 Support

For questions, improvements, or bug reports:
1. Check existing documentation
2. Review example scripts
3. Test with different parameters
4. Consider market conditions and external factors

## 🎯 Future Enhancements

- **Ensemble Methods**: Combine LSTM with other ML models
- **Sentiment Analysis**: Incorporate news and social media sentiment
- **Multi-timeframe**: Support for different prediction horizons
- **Real-time Predictions**: Live market data integration
- **Advanced Visualization**: Interactive charts and dashboards
- **Risk Metrics**: Value at Risk (VaR) and other risk measures