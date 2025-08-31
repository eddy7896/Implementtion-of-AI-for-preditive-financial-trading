import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import ta
from datetime import timedelta

# ---------------------------
# User Inputs
# ---------------------------
ticker = input("Enter stock ticker symbol (e.g., AAPL): ").upper()
start_date = input("Enter start date (YYYY-MM-DD): ")
end_date = input("Enter end date (YYYY-MM-DD): ")

print("\nChoose Technical Indicators to calculate:")
print("1. RSI")
print("2. Bollinger Bands")
print("3. MACD")
print("4. SMA (50-day & 200-day)")
print("Enter numbers separated by commas (e.g., 1,3,4): ")
choice = input("Your choice: ")

indicators = [int(x.strip()) for x in choice.split(",")]

# ---------------------------
# Fetch Stock Data
# ---------------------------
df = yf.download(ticker, start=start_date, end=end_date)
df.dropna(inplace=True)

# ---------------------------
# Technical Indicators (using ta)
# ---------------------------
if 1 in indicators:
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=14).rsi().squeeze()

if 2 in indicators:
    bb = ta.volatility.BollingerBands(df["Close"], window=20, window_dev=2)
    df["BBL"] = bb.bollinger_lband().squeeze()
    df["BBM"] = bb.bollinger_mavg().squeeze()
    df["BBU"] = bb.bollinger_hband().squeeze()

if 3 in indicators:
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd().squeeze()
    df["MACD_signal"] = macd.macd_signal().squeeze()
    df["MACD_diff"] = macd.macd_diff().squeeze()

if 4 in indicators:
    df["SMA50"] = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

df.dropna(inplace=True)

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

# ---------------------------
# Create Future Targets
# ---------------------------
df["Close_Future_5"] = df["Close"].shift(-5)
df["Close_Future_10"] = df["Close"].shift(-10)
df["Close_Future_20"] = df["Close"].shift(-20)

df.dropna(inplace=True)

X = df[features]
y5 = df["Close_Future_5"]
y10 = df["Close_Future_10"]
y20 = df["Close_Future_20"]

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
future_dates = pd.bdate_range(start=last_date, periods=21)[[4, 9, 19]]  # +5, +10, +20 days

# ---------------------------
# Visualization
# ---------------------------
plt.figure(figsize=(14, 8))
plt.plot(df.index, df["Close"], label="Historical Close Price")

if 2 in indicators:
    plt.plot(df.index, df["BBU"], label="Bollinger Upper", linestyle="--", alpha=0.6)
    plt.plot(df.index, df["BBL"], label="Bollinger Lower", linestyle="--", alpha=0.6)
if 4 in indicators:
    plt.plot(df.index, df["SMA50"], label="SMA 50")
    plt.plot(df.index, df["SMA200"], label="SMA 200")

# Plot predictions
plt.scatter(future_dates, predictions, color="red", label="Predicted Close Price (+5,+10,+20 days)")
for d, p in zip(future_dates, predictions):
    plt.text(d, p, f"{p:.2f}", fontsize=9, ha="left", color="red")

plt.title(f"{ticker} Stock with Indicators and Predictions")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")
plt.grid(True)
plt.show()
