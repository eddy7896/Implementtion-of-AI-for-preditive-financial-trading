import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ta
from datetime import timedelta
import warnings
warnings.filterwarnings('ignore')

print("Starting AAPL Technical Indicators Test")
print("=" * 50)

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
    df = yf.download(ticker, start=start_date, end=end_date)
    
    # Flatten multi-level columns if they exist
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Ensure we have the expected column names
    expected_columns = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    for col in expected_columns:
        if col not in df.columns:
            print(f"Warning: Column {col} not found in data")
    
    df.dropna(inplace=True)
    print(f"Successfully fetched {len(df)} trading days of data")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Price range: ${float(df['Close'].min()):.2f} - ${float(df['Close'].max()):.2f}")
    print(f"Columns: {list(df.columns)}")
    print()
except Exception as e:
    print(f"Error fetching data: {e}")
    exit(1)

# ---------------------------
# Technical Indicators (using ta)
# ---------------------------
print("Calculating technical indicators...")

# RSI
if 1 in indicators:
    print("- Calculating RSI (14-period)")
    close_series = df["Close"].squeeze()
    df["RSI"] = ta.momentum.RSIIndicator(close=close_series, window=14).rsi()

# Bollinger Bands
if 2 in indicators:
    print("- Calculating Bollinger Bands (20-period, 2 std dev)")
    close_series = df["Close"].squeeze()
    bb = ta.volatility.BollingerBands(close=close_series, window=20, window_dev=2)
    df["BBL"] = bb.bollinger_lband()
    df["BBM"] = bb.bollinger_mavg()
    df["BBU"] = bb.bollinger_hband()

# MACD
if 3 in indicators:
    print("- Calculating MACD (12, 26, 9)")
    close_series = df["Close"].squeeze()
    macd = ta.trend.MACD(close=close_series)
    df["MACD"] = macd.macd()
    df["MACD_signal"] = macd.macd_signal()
    df["MACD_diff"] = macd.macd_diff()

# Simple Moving Averages
if 4 in indicators:
    print("- Calculating SMA (50-day & 200-day)")
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

df.dropna(inplace=True)
print(f"After calculating indicators: {len(df)} rows remaining")
print()

# ---------------------------
# Feature Selection
# ---------------------------
features = []
if 1 in indicators:
    features.append("RSI")
if 2 in indicators:
    features += ["BBL", "BBM", "BBU"]
if 3 in indicators:
    features += ["MACD", "MACD_signal", "MACD_diff"]
if 4 in indicators:
    features += ["SMA50", "SMA200"]

print(f"Features selected: {features}")
print()

# ---------------------------
# Display Latest Technical Indicator Values
# ---------------------------
print("Latest Technical Indicator Values:")
print("-" * 40)
latest_row = df.iloc[-1]
print(f"Date: {df.index[-1].strftime('%Y-%m-%d')}")
print(f"Close Price: ${latest_row['Close']:.2f}")

if 1 in indicators:
    print(f"RSI: {latest_row['RSI']:.2f}")

if 2 in indicators:
    print(f"Bollinger Bands - Upper: ${latest_row['BBU']:.2f}, Middle: ${latest_row['BBM']:.2f}, Lower: ${latest_row['BBL']:.2f}")

if 3 in indicators:
    print(f"MACD: {latest_row['MACD']:.4f}, Signal: {latest_row['MACD_signal']:.4f}, Histogram: {latest_row['MACD_diff']:.4f}")

if 4 in indicators:
    print(f"SMA50: ${latest_row['SMA50']:.2f}, SMA200: ${latest_row['SMA200']:.2f}")

print()

# ---------------------------
# Technical Analysis Summary
# ---------------------------
print("Technical Analysis Summary:")
print("-" * 40)

if 1 in indicators:
    rsi_current = latest_row['RSI']
    if rsi_current > 70:
        rsi_signal = "Overbought"
    elif rsi_current < 30:
        rsi_signal = "Oversold"
    else:
        rsi_signal = "Neutral"
    print(f"RSI Signal: {rsi_signal} (RSI: {rsi_current:.2f})")

if 2 in indicators:
    current_price = latest_row['Close']
    bb_upper = latest_row['BBU']
    bb_lower = latest_row['BBL']
    if current_price > bb_upper:
        bb_signal = "Above upper band (potential sell)"
    elif current_price < bb_lower:
        bb_signal = "Below lower band (potential buy)"
    else:
        bb_signal = "Within bands (neutral)"
    print(f"Bollinger Bands Signal: {bb_signal}")

if 3 in indicators:
    macd_line = latest_row['MACD']
    macd_signal_line = latest_row['MACD_signal']
    if macd_line > macd_signal_line:
        macd_signal = "Bullish (MACD above signal)"
    else:
        macd_signal = "Bearish (MACD below signal)"
    print(f"MACD Signal: {macd_signal}")

if 4 in indicators:
    current_price = latest_row['Close']
    sma50 = latest_row['SMA50']
    sma200 = latest_row['SMA200']
    if current_price > sma50 > sma200:
        sma_signal = "Strong uptrend (above both SMAs, golden cross)"
    elif current_price > sma50 and current_price > sma200:
        sma_signal = "Uptrend (above both SMAs)"
    elif current_price < sma50 < sma200:
        sma_signal = "Strong downtrend (below both SMAs, death cross)"
    elif current_price < sma50 and current_price < sma200:
        sma_signal = "Downtrend (below both SMAs)"
    else:
        sma_signal = "Mixed signals"
    print(f"SMA Signal: {sma_signal}")

print()

# ---------------------------
# Create Future Targets for Prediction
# ---------------------------
print("Creating prediction model...")
df["Close_Future_5"] = df["Close"].shift(-5)
df["Close_Future_10"] = df["Close"].shift(-10)
df["Close_Future_20"] = df["Close"].shift(-20)

df_model = df.dropna()

if len(df_model) < 50:
    print("Warning: Insufficient data for reliable predictions")
else:
    X = df_model[features]
    y5 = df_model["Close_Future_5"]
    y10 = df_model["Close_Future_10"]
    y20 = df_model["Close_Future_20"]

    # Train models
    model5 = LinearRegression().fit(X, y5)
    model10 = LinearRegression().fit(X, y10)
    model20 = LinearRegression().fit(X, y20)

    # Last available feature row
    last_features = df.iloc[-1][features].values.reshape(1, -1)
    last_date = df.index[-1]

    predictions = [
        model5.predict(last_features)[0],
        model10.predict(last_features)[0],
        model20.predict(last_features)[0],
    ]

    # Generate future business dates (skip weekends/holidays)
    try:
        future_dates = pd.bdate_range(start=last_date, periods=21)[[4, 9, 19]]  # +5, +10, +20 days
    except:
        # If business date range fails, use simple date arithmetic
        future_dates = [last_date + timedelta(days=5), last_date + timedelta(days=10), last_date + timedelta(days=20)]

    print("Price Predictions (Linear Regression):")
    print(f"Current Price: ${df.iloc[-1]['Close']:.2f}")
    print(f"+5  days prediction: ${predictions[0]:.2f}")
    print(f"+10 days prediction: ${predictions[1]:.2f}")
    print(f"+20 days prediction: ${predictions[2]:.2f}")
    print()

# ---------------------------
# Statistical Summary
# ---------------------------
print("Statistical Summary (last 252 trading days - 1 year):")
print("-" * 50)
recent_data = df.tail(min(252, len(df)))

print(f"Price Statistics:")
print(f"  Current: ${recent_data['Close'].iloc[-1]:.2f}")
print(f"  52-week High: ${recent_data['Close'].max():.2f}")
print(f"  52-week Low: ${recent_data['Close'].min():.2f}")
print(f"  Average: ${recent_data['Close'].mean():.2f}")
print(f"  Volatility (std): ${recent_data['Close'].std():.2f}")

if 1 in indicators:
    print(f"RSI Statistics:")
    print(f"  Current: {recent_data['RSI'].iloc[-1]:.2f}")
    print(f"  Average: {recent_data['RSI'].mean():.2f}")
    print(f"  Times Overbought (>70): {(recent_data['RSI'] > 70).sum()}")
    print(f"  Times Oversold (<30): {(recent_data['RSI'] < 30).sum()}")

print()

# ---------------------------
# Visualization
# ---------------------------
print("Creating visualization...")
plt.figure(figsize=(16, 12))

# Main price chart
plt.subplot(3, 1, 1)
plt.plot(df.index, df["Close"], label="Close Price", linewidth=1)

if 2 in indicators:
    plt.plot(df.index, df["BBU"], label="Bollinger Upper", linestyle="--", alpha=0.6, color='red')
    plt.plot(df.index, df["BBL"], label="Bollinger Lower", linestyle="--", alpha=0.6, color='red')
    plt.fill_between(df.index, df["BBL"], df["BBU"], alpha=0.1, color='gray')

if 4 in indicators:
    plt.plot(df.index, df["SMA50"], label="SMA 50", color='orange')
    plt.plot(df.index, df["SMA200"], label="SMA 200", color='purple')

# Plot predictions if available
if 'predictions' in locals():
    plt.scatter(future_dates, predictions, color="red", s=50, label="Predictions (+5,+10,+20 days)", zorder=5)
    for d, p in zip(future_dates, predictions):
        plt.annotate(f"${p:.2f}", (d, p), xytext=(5, 5), textcoords='offset points', 
                    fontsize=8, color="red", fontweight='bold')

plt.title(f"{ticker} Stock Price with Technical Indicators", fontsize=14, fontweight='bold')
plt.legend()
plt.ylabel("Price ($)")
plt.grid(True, alpha=0.3)

# RSI subplot
if 1 in indicators:
    plt.subplot(3, 1, 2)
    plt.plot(df.index, df["RSI"], label="RSI (14)", color='purple')
    plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
    plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
    plt.axhline(y=50, color='b', linestyle='-', alpha=0.5, label='Neutral (50)')
    plt.fill_between(df.index, 30, 70, alpha=0.1, color='gray')
    plt.title("RSI Indicator")
    plt.ylabel("RSI")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.ylim(0, 100)

# MACD subplot
if 3 in indicators:
    plt.subplot(3, 1, 3)
    plt.plot(df.index, df["MACD"], label="MACD", color='blue')
    plt.plot(df.index, df["MACD_signal"], label="Signal", color='red')
    plt.bar(df.index, df["MACD_diff"], label="Histogram", alpha=0.3, color='green')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.title("MACD Indicator")
    plt.ylabel("MACD")
    plt.xlabel("Date")
    plt.legend()
    plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("=" * 50)
print("AAPL Technical Analysis Test Completed Successfully!")
print("=" * 50)