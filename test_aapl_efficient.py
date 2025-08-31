import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("Starting AAPL Technical Indicators Test (Efficient Version)")
print("=" * 60)

# ---------------------------
# Test Parameters (Automated)
# ---------------------------
ticker = "AAPL"
start_date = "2005-01-01"
end_date = "2025-08-22"
indicators = [1, 2, 3, 4]  # All indicators: RSI, Bollinger Bands, MACD, SMA

print(f"Ticker: {ticker}")
print(f"Start Date: {start_date}")
print(f"End Date: {end_date}")
print(f"Testing all technical indicators: RSI, Bollinger Bands, MACD, SMA")
print()

# ---------------------------
# Fetch Stock Data
# ---------------------------
print("Fetching stock data...")
try:
    df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    df.dropna(inplace=True)
    print(f"Successfully fetched {len(df)} trading days of data")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Price range: ${float(df['Close'].min()):.2f} - ${float(df['Close'].max()):.2f}")
    print()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit(1)

# ---------------------------
# Technical Indicators Calculation
# ---------------------------
print("Calculating technical indicators...")

try:
    # RSI
    if 1 in indicators:
        print("- Calculating RSI (14-period)")
        df["RSI"] = ta.momentum.rsi(df["Close"], window=14)

    # Bollinger Bands
    if 2 in indicators:
        print("- Calculating Bollinger Bands (20-period, 2 std dev)")
        df["BBL"] = ta.volatility.bollinger_lband(df["Close"], window=20, window_dev=2)
        df["BBM"] = ta.volatility.bollinger_mavg(df["Close"], window=20)
        df["BBU"] = ta.volatility.bollinger_hband(df["Close"], window=20, window_dev=2)

    # MACD
    if 3 in indicators:
        print("- Calculating MACD (12, 26, 9)")
        df["MACD"] = ta.trend.macd(df["Close"], window_fast=12, window_slow=26)
        df["MACD_signal"] = ta.trend.macd_signal(df["Close"], window_fast=12, window_slow=26, window_sign=9)
        df["MACD_diff"] = ta.trend.macd_diff(df["Close"], window_fast=12, window_slow=26, window_sign=9)

    # Simple Moving Averages
    if 4 in indicators:
        print("- Calculating SMA (50-day & 200-day)")
        df["SMA50"] = ta.trend.sma_indicator(df["Close"], window=50)
        df["SMA200"] = ta.trend.sma_indicator(df["Close"], window=200)

    # Remove NaN values after calculations
    df.dropna(inplace=True)
    print(f"After calculating indicators: {len(df)} rows remaining")
    print()

except Exception as e:
    print(f"Error calculating technical indicators: {e}")
    exit(1)

# ---------------------------
# Latest Technical Indicator Values
# ---------------------------
print("Latest Technical Indicator Values:")
print("-" * 40)
latest_row = df.iloc[-1]
print(f"Date: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Close Price: ${latest_row['Close']:.2f}")

if 1 in indicators and 'RSI' in df.columns:
    rsi_val = latest_row['RSI']
    print(f"RSI: {rsi_val:.2f}")
    if rsi_val > 70:
        print("  -> Overbought signal")
    elif rsi_val < 30:
        print("  -> Oversold signal")
    else:
        print("  -> Neutral")

if 2 in indicators and 'BBU' in df.columns:
    bb_upper = latest_row['BBU']
    bb_middle = latest_row['BBM']
    bb_lower = latest_row['BBL']
    current_price = latest_row['Close']
    print(f"Bollinger Bands - Upper: ${bb_upper:.2f}, Middle: ${bb_middle:.2f}, Lower: ${bb_lower:.2f}")
    if current_price > bb_upper:
        print("  -> Price above upper band (potential sell signal)")
    elif current_price < bb_lower:
        print("  -> Price below lower band (potential buy signal)")
    else:
        print("  -> Price within bands (neutral)")

if 3 in indicators and 'MACD' in df.columns:
    macd_val = latest_row['MACD']
    macd_signal_val = latest_row['MACD_signal']
    macd_hist = latest_row['MACD_diff']
    print(f"MACD: {macd_val:.4f}, Signal: {macd_signal_val:.4f}, Histogram: {macd_hist:.4f}")
    if macd_val > macd_signal_val:
        print("  -> Bullish (MACD above signal)")
    else:
        print("  -> Bearish (MACD below signal)")

if 4 in indicators and 'SMA50' in df.columns:
    sma50 = latest_row['SMA50']
    sma200 = latest_row['SMA200']
    current_price = latest_row['Close']
    print(f"SMA50: ${sma50:.2f}, SMA200: ${sma200:.2f}")
    if current_price > sma50 and sma50 > sma200:
        print("  -> Strong uptrend (golden cross pattern)")
    elif current_price < sma50 and sma50 < sma200:
        print("  -> Strong downtrend (death cross pattern)")
    elif current_price > sma50:
        print("  -> Short-term uptrend")
    elif current_price < sma50:
        print("  -> Short-term downtrend")
    else:
        print("  -> Mixed signals")

print()

# ---------------------------
# Statistical Summary (Recent 252 days)
# ---------------------------
print("Performance Summary (Last 252 Trading Days):")
print("-" * 45)
recent_data = df.tail(min(252, len(df)))

current_price = recent_data['Close'].iloc[-1]
year_high = recent_data['Close'].max()
year_low = recent_data['Close'].min()
year_avg = recent_data['Close'].mean()
volatility = recent_data['Close'].std()

print(f"Current Price: ${current_price:.2f}")
print(f"52-Week High: ${year_high:.2f}")
print(f"52-Week Low: ${year_low:.2f}")
print(f"52-Week Average: ${year_avg:.2f}")
print(f"Price Volatility (std): ${volatility:.2f}")

# Calculate returns
daily_returns = recent_data['Close'].pct_change().dropna()
cumulative_return = (current_price / recent_data['Close'].iloc[0] - 1) * 100
avg_daily_return = daily_returns.mean() * 100
volatility_pct = daily_returns.std() * 100 * np.sqrt(252)  # Annualized volatility

print(f"1-Year Return: {cumulative_return:.2f}%")
print(f"Average Daily Return: {avg_daily_return:.4f}%")
print(f"Annualized Volatility: {volatility_pct:.2f}%")

if 1 in indicators and 'RSI' in recent_data.columns:
    rsi_data = recent_data['RSI'].dropna()
    avg_rsi = rsi_data.mean()
    overbought_days = (rsi_data > 70).sum()
    oversold_days = (rsi_data < 30).sum()
    print(f"\nRSI Analysis:")
    print(f"  Average RSI: {avg_rsi:.2f}")
    print(f"  Overbought days (>70): {overbought_days}")
    print(f"  Oversold days (<30): {oversold_days}")

print()

# ---------------------------
# Simple Prediction Model
# ---------------------------
print("Creating Price Prediction Model:")
print("-" * 35)

# Prepare features
features = []
if 1 in indicators and 'RSI' in df.columns:
    features.append("RSI")
if 2 in indicators and 'BBU' in df.columns:
    features += ["BBL", "BBM", "BBU"]
if 3 in indicators and 'MACD' in df.columns:
    features += ["MACD", "MACD_signal", "MACD_diff"]
if 4 in indicators and 'SMA50' in df.columns:
    features += ["SMA50", "SMA200"]

if len(features) > 0:
    # Create target variables (future prices)
    df_model = df.copy()
    df_model["Close_Future_5"] = df_model["Close"].shift(-5)
    df_model["Close_Future_10"] = df_model["Close"].shift(-10)
    df_model["Close_Future_20"] = df_model["Close"].shift(-20)
    
    # Remove rows with NaN in target variables
    df_model = df_model.dropna()
    
    if len(df_model) > 100:  # Ensure we have enough data
        X = df_model[features]
        
        # Train models for different time horizons
        models = {}
        predictions = {}
        
        for days in [5, 10, 20]:
            y = df_model[f"Close_Future_{days}"]
            model = LinearRegression().fit(X, y)
            models[days] = model
            
            # Make prediction using latest data
            last_features = df.iloc[-1][features].values.reshape(1, -1)
            pred = model.predict(last_features)[0]
            predictions[days] = pred
        
        print(f"Features used: {', '.join(features)}")
        print(f"Training data points: {len(df_model)}")
        print(f"\nPrice Predictions:")
        print(f"Current Price: ${current_price:.2f}")
        
        for days in [5, 10, 20]:
            pred_price = predictions[days]
            change = pred_price - current_price
            change_pct = (change / current_price) * 100
            print(f"+{days:2d} days: ${pred_price:.2f} ({change:+.2f}, {change_pct:+.2f}%)")
        
        # Calculate model accuracy on recent data (last 100 points)
        recent_test = df_model.tail(100)
        if len(recent_test) > 0:
            X_test = recent_test[features]
            actual_5 = recent_test["Close_Future_5"]
            pred_5 = models[5].predict(X_test)
            
            mse = np.mean((actual_5 - pred_5) ** 2)
            rmse = np.sqrt(mse)
            mae = np.mean(np.abs(actual_5 - pred_5))
            
            print(f"\nModel Performance (5-day predictions, last 100 samples):")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
    else:
        print("Insufficient data for reliable predictions")
else:
    print("No features available for prediction model")

print()

# ---------------------------
# Create Simple Visualization
# ---------------------------
print("Creating visualization of recent data...")

# Use only last 500 days for visualization to avoid memory issues
recent_viz_data = df.tail(500)

plt.figure(figsize=(15, 10))

# Price chart with moving averages
plt.subplot(2, 2, 1)
plt.plot(recent_viz_data.index, recent_viz_data["Close"], label="Close Price", linewidth=1)

if 4 in indicators and 'SMA50' in recent_viz_data.columns:
    plt.plot(recent_viz_data.index, recent_viz_data["SMA50"], label="SMA 50", alpha=0.8)
    plt.plot(recent_viz_data.index, recent_viz_data["SMA200"], label="SMA 200", alpha=0.8)

if 2 in indicators and 'BBU' in recent_viz_data.columns:
    plt.fill_between(recent_viz_data.index, recent_viz_data["BBL"], recent_viz_data["BBU"], 
                     alpha=0.2, color='gray', label='Bollinger Bands')

plt.title(f"{ticker} Price Chart (Last 500 Days)")
plt.legend()
plt.grid(True, alpha=0.3)

# RSI
if 1 in indicators and 'RSI' in recent_viz_data.columns:
    plt.subplot(2, 2, 2)
    plt.plot(recent_viz_data.index, recent_viz_data["RSI"], color='purple', linewidth=1)
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7)
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7)
    plt.fill_between(recent_viz_data.index, 30, 70, alpha=0.1, color='gray')
    plt.title("RSI (14-period)")
    plt.ylim(0, 100)
    plt.grid(True, alpha=0.3)

# MACD
if 3 in indicators and 'MACD' in recent_viz_data.columns:
    plt.subplot(2, 2, 3)
    plt.plot(recent_viz_data.index, recent_viz_data["MACD"], label="MACD", linewidth=1)
    plt.plot(recent_viz_data.index, recent_viz_data["MACD_signal"], label="Signal", linewidth=1)
    plt.bar(recent_viz_data.index, recent_viz_data["MACD_diff"], alpha=0.3, label="Histogram")
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title("MACD Indicator")
    plt.legend()
    plt.grid(True, alpha=0.3)

# Volume
plt.subplot(2, 2, 4)
plt.bar(recent_viz_data.index, recent_viz_data["Volume"], alpha=0.6, width=1)
plt.title("Trading Volume")
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('aapl_technical_analysis.png', dpi=100, bbox_inches='tight')
print("Chart saved as 'aapl_technical_analysis.png'")

print()
print("=" * 60)
print("AAPL Technical Analysis Test Completed Successfully!")
print("=" * 60)

# Summary of key findings
print("\nKEY FINDINGS SUMMARY:")
print("-" * 20)
print(f"• Total trading days analyzed: {len(df)}")
print(f"• Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
print(f"• Current price: ${current_price:.2f}")
print(f"• All-time high in dataset: ${df['Close'].max():.2f}")
print(f"• All-time low in dataset: ${df['Close'].min():.2f}")

if 1 in indicators and 'RSI' in df.columns:
    current_rsi = df['RSI'].iloc[-1]
    print(f"• Current RSI: {current_rsi:.1f} ({'Overbought' if current_rsi > 70 else 'Oversold' if current_rsi < 30 else 'Neutral'})")

if 4 in indicators and 'SMA50' in df.columns:
    sma50_current = df['SMA50'].iloc[-1]
    sma200_current = df['SMA200'].iloc[-1]
    trend = "Bullish" if current_price > sma50_current > sma200_current else "Bearish" if current_price < sma50_current < sma200_current else "Mixed"
    print(f"• Moving average trend: {trend}")

print(f"• Technical indicators calculated: {len(indicators)} (RSI, Bollinger Bands, MACD, SMA)")
print("\nTest completed without errors!")