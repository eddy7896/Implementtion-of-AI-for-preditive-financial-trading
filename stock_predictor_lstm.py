import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from technical_indicators import TechnicalIndicators
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Deep Learning Libraries
try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    print("TensorFlow not available. Installing...")
    import subprocess
    import sys
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
        from tensorflow import keras
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.optimizers import Adam
        TENSORFLOW_AVAILABLE = True
        print("TensorFlow installed successfully!")
    except Exception as e:
        print(f"Could not install TensorFlow: {e}")
        print("Will use alternative prediction method...")
        TENSORFLOW_AVAILABLE = False

class StockPredictor:
    def __init__(self, ticker, lookback_days=60, prediction_days=7, selected_indicators=None):
        """
        Initialize Stock Predictor with LSTM and Technical Indicators
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL')
            lookback_days: Number of historical days to use for prediction
            prediction_days: Number of days to predict (default: 7 for weekly)
            selected_indicators: List of selected indicator keys from GUI (optional)
        """
        self.ticker = ticker
        self.lookback_days = lookback_days
        self.prediction_days = prediction_days
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.df = None
        self.features = []
        self.selected_indicators = selected_indicators or []
        
        # Mapping from GUI checkbox keys to DataFrame column names
        self.indicator_mapping = {
            # Moving Averages
            'sma_5': 'SMA_5',
            'sma_10': 'SMA_10',
            'sma_20': 'SMA_20',
            'sma_50': 'SMA_50',
            'ema_12': 'EMA_12',
            'ema_26': 'EMA_26',
            
            # Momentum Indicators
            'rsi': 'RSI',
            'stoch_k': 'Stoch_k',
            'stoch_d': 'Stoch_d',
            'roc': 'ROC',
            'williams_r': 'Williams_R',
            
            # Trend Indicators
            'macd': 'MACD',
            'macd_signal': 'MACD_signal',
            'macd_diff': 'MACD_diff',
            'adx': 'ADX',
            'aroon_up': 'Aroon_Up',
            'aroon_down': 'Aroon_Down',
            
            # Volatility Indicators
            'bb_upper': 'BB_upper',
            'bb_middle': 'BB_middle',
            'bb_lower': 'BB_lower',
            'bb_width': 'BB_width',
            'atr': 'ATR',
            
            # Volume Indicators
            'volume_sma': 'Volume_SMA',
            'obv': 'OBV',
            'vwap': 'VWAP',
            'ad_line': 'AD_Line',
            
            # Price-based Indicators
            'high_low_pct': 'High_Low_Pct',
            'open_close_pct': 'Open_Close_Pct',
            
            # Oscillators
            'cci': 'CCI',
        }
        
    def fetch_data(self, start_date, end_date):
        """Fetch stock data and calculate technical indicators"""
        print(f"Fetching data for {self.ticker} from {start_date} to {end_date}...")
        
        try:
            # Download stock data
            self.df = yf.download(self.ticker, start=start_date, end=end_date, progress=False)
            
            # Flatten multi-level columns if they exist
            if isinstance(self.df.columns, pd.MultiIndex):
                self.df.columns = self.df.columns.droplevel(1)
            
            self.df.dropna(inplace=True)
            print(f"Successfully fetched {len(self.df)} trading days")
            
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def calculate_technical_indicators(self):
        """
        =======================================================================
        CALCULATE TECHNICAL INDICATORS FOR LSTM PREDICTION MODEL
        =======================================================================
        
        This method is the core of the technical analysis system. It takes the
        user-selected indicators from the GUI and calculates them using proper
        mathematical formulas, then prepares the data for LSTM model training.
        
        Process Flow:
        1. Initialize TechnicalIndicators class with OHLCV data
        2. Map GUI selections to calculation functions
        3. Calculate only selected indicators (performance optimization)
        4. Add calculated values to the main DataFrame
        5. Clean data and prepare feature list for LSTM
        6. Validate data sufficiency for model training
        
        This selective calculation approach:
        - Improves performance by only calculating needed indicators
        - Reduces memory usage
        - Allows users to customize their analysis
        - Provides detailed logging for debugging
        """
        print("Calculating selected technical indicators using advanced formulas...")
        
        try:
            # ===================================================================
            # STEP 1: INITIALIZE TECHNICAL INDICATORS CALCULATOR
            # ===================================================================
            # Create an instance of our comprehensive TechnicalIndicators class
            # This class contains all 29+ indicators with proper mathematical formulas
            tech_indicators = TechnicalIndicators(self.df)
            
            # ===================================================================
            # STEP 2: DEFINE BASE FEATURES FOR LSTM MODEL
            # ===================================================================
            # Always include basic OHLCV data as these are fundamental
            # These form the foundation for all technical analysis
            base_features = ['Open', 'High', 'Low', 'Close', 'Volume']
            
            # Track which indicators were successfully calculated
            calculated_indicators = []
            
            print(f"Selected indicators: {self.selected_indicators}")
            
            # ===================================================================
            # STEP 3: INDICATOR CALCULATION MAPPING SYSTEM
            # ===================================================================
            # This dictionary maps GUI selection keys to actual calculation methods
            # Using lambda functions for lazy evaluation - only calculated when needed
            # This design pattern allows for:
            # - Efficient memory usage (only selected indicators calculated)
            # - Easy maintenance (add new indicators by extending this dict)
            # - Clear separation between GUI and calculation logic
            
            indicator_calculations = {
                # ===============================================================
                # MOVING AVERAGES - TREND IDENTIFICATION
                # ===============================================================
                # Simple Moving Averages for different timeframes
                'sma_5': lambda: tech_indicators.sma(5),     # Short-term trend (1 week)
                'sma_10': lambda: tech_indicators.sma(10),   # Medium-short trend (2 weeks)
                'sma_20': lambda: tech_indicators.sma(20),   # Medium trend (1 month)
                'sma_50': lambda: tech_indicators.sma(50),   # Long-term trend (2.5 months)
                
                # Exponential Moving Averages for responsive trend analysis
                'ema_12': lambda: tech_indicators.ema(12),   # Fast EMA for MACD
                'ema_26': lambda: tech_indicators.ema(26),   # Slow EMA for MACD
                
                # ===============================================================
                # MOMENTUM INDICATORS - PRICE VELOCITY AND STRENGTH
                # ===============================================================
                'rsi': lambda: tech_indicators.rsi(14),                    # Relative Strength Index
                'stoch_k': lambda: tech_indicators.stochastic()[0],       # Fast Stochastic
                'stoch_d': lambda: tech_indicators.stochastic()[1],       # Slow Stochastic
                'roc': lambda: tech_indicators.roc(12),                   # Rate of Change
                'williams_r': lambda: tech_indicators.williams_r(14),     # Williams %R
                
                # ===============================================================
                # TREND INDICATORS - DIRECTION AND MOMENTUM ANALYSIS
                # ===============================================================
                'macd': lambda: tech_indicators.macd()[0],           # MACD Line
                'macd_signal': lambda: tech_indicators.macd()[1],   # MACD Signal Line
                'macd_diff': lambda: tech_indicators.macd()[2],     # MACD Histogram
                'adx': lambda: tech_indicators.adx(14),             # Average Directional Index
                'aroon_up': lambda: tech_indicators.aroon()[0],     # Aroon Up
                'aroon_down': lambda: tech_indicators.aroon()[1],   # Aroon Down
                
                # ===============================================================
                # VOLATILITY INDICATORS - PRICE MOVEMENT RANGE
                # ===============================================================
                'bb_upper': lambda: tech_indicators.bollinger_bands()[0],    # Bollinger Upper Band
                'bb_middle': lambda: tech_indicators.bollinger_bands()[1],   # Bollinger Middle Band
                'bb_lower': lambda: tech_indicators.bollinger_bands()[2],    # Bollinger Lower Band
                'bb_width': lambda: tech_indicators.bollinger_band_width(),  # Bollinger Band Width
                'atr': lambda: tech_indicators.atr(14),                      # Average True Range
                
                # ===============================================================
                # VOLUME INDICATORS - MONEY FLOW AND PARTICIPATION
                # ===============================================================
                'volume_sma': lambda: tech_indicators.volume_sma(20),              # Volume Moving Average
                'obv': lambda: tech_indicators.obv(),                             # On-Balance Volume
                'vwap': lambda: tech_indicators.vwap(),                           # Volume Weighted Average Price
                'ad_line': lambda: tech_indicators.accumulation_distribution(),   # A/D Line
                
                # ===============================================================
                # PRICE-BASED INDICATORS - INTRADAY ANALYSIS
                # ===============================================================
                'high_low_pct': lambda: tech_indicators.high_low_percentage(),     # Daily Range %
                'open_close_pct': lambda: tech_indicators.open_close_percentage(), # Daily Change %
                
                # ===============================================================
                # OSCILLATORS - CYCLICAL ANALYSIS
                # ===============================================================
                'cci': lambda: tech_indicators.cci(20),    # Commodity Channel Index
            }
            
            # ===================================================================
            # STEP 4: SELECTIVE INDICATOR CALCULATION
            # ===================================================================
            # Process each indicator selected by the user through the GUI
            # This selective approach improves performance significantly
            
            for indicator_key in self.selected_indicators:
                if indicator_key in indicator_calculations:
                    try:
                        # Get the standardized column name for the indicator
                        # This mapping ensures consistent naming across the system
                        column_name = self.indicator_mapping.get(indicator_key, indicator_key.upper())
                        
                        # Execute the calculation function (lambda is evaluated here)
                        # This is where the actual mathematical calculation occurs
                        indicator_values = indicator_calculations[indicator_key]()
                        
                        # Add the calculated values to our main DataFrame
                        # This integrates the indicator with the price data
                        self.df[column_name] = indicator_values
                        calculated_indicators.append(column_name)
                        
                        # Log successful calculation with description
                        print(f"[OK] Calculated {column_name} ({tech_indicators.get_indicator_description(indicator_key)})")
                        
                    except Exception as e:
                        # Handle individual indicator calculation errors gracefully
                        # This ensures one failed indicator doesn't break the entire process
                        print(f"[WARNING] Failed to calculate {indicator_key}: {str(e)}")
                        continue
                else:
                    # Alert about unknown indicator keys (helps with debugging)
                    print(f"[WARNING] Unknown indicator key: {indicator_key}")
            
            # ===================================================================
            # STEP 5: PREPARE FEATURE LIST FOR LSTM MODEL
            # ===================================================================
            # Combine base OHLCV data with calculated technical indicators
            # This creates the complete feature set for machine learning
            self.features = base_features + calculated_indicators
            
            # ===================================================================
            # STEP 6: DATA CLEANING AND VALIDATION
            # ===================================================================
            # Remove NaN values that result from indicator calculations
            # Some indicators need initial periods to calculate (e.g., 20-day SMA needs 20 days)
            initial_rows = len(self.df)
            self.df.dropna(inplace=True)
            final_rows = len(self.df)
            
            # Report data loss from indicator calculations
            if initial_rows > final_rows:
                print(f"Removed {initial_rows - final_rows} rows with NaN values from indicator calculations")
            
            # ===================================================================
            # STEP 7: FINAL SUMMARY AND VALIDATION
            # ===================================================================
            print(f"Selected technical indicators calculated: {calculated_indicators}")
            print(f"Total features: {len(self.features)}")
            print(f"Final rows after cleaning: {len(self.df)}")
            
            # Validate sufficient data for LSTM training
            # LSTM needs lookback_days + buffer for proper training/testing split
            min_required_rows = self.lookback_days + 50
            if len(self.df) < min_required_rows:
                print(f"Warning: Limited data available ({len(self.df)} rows). Consider using fewer indicators or longer historical period.")
            
            return True
            
        except Exception as e:
            # Handle any unexpected errors in the calculation process
            print(f"Error calculating technical indicators: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def prepare_lstm_data(self, test_size=0.2):
        """Prepare data for LSTM training"""
        print("Preparing LSTM data...")
        
        # Select features
        feature_data = self.df[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(feature_data)
        
        # Create sequences for LSTM
        X, y = [], []
        
        for i in range(self.lookback_days, len(scaled_data)):
            X.append(scaled_data[i-self.lookback_days:i])
            # Target is the next day's close price (index 3 in features)
            y.append(scaled_data[i, 3])  # Close price index
        
        X, y = np.array(X), np.array(y)
        
        # Split into train and test
        split_idx = int(len(X) * (1 - test_size))
        
        self.X_train = X[:split_idx]
        self.X_test = X[split_idx:]
        self.y_train = y[:split_idx]
        self.y_test = y[split_idx:]
        
        print(f"Training data shape: {self.X_train.shape}")
        print(f"Testing data shape: {self.X_test.shape}")
        
        return True
    
    def build_lstm_model(self):
        """Build and compile LSTM model"""
        if not TENSORFLOW_AVAILABLE:
            print("TensorFlow not available. Cannot build LSTM model.")
            return False
            
        print("Building LSTM model...")
        
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.lookback_days, len(self.features))),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        print("LSTM model built successfully!")
        return True
    
    def train_model(self, epochs=100, batch_size=32, validation_split=0.1):
        """Train the LSTM model"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("Cannot train model - TensorFlow not available or model not built")
            return False
            
        print(f"Training LSTM model for {epochs} epochs...")
        
        try:
            history = self.model.fit(
                self.X_train, self.y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=validation_split,
                verbose=1,
                shuffle=False
            )
            
            print("Model training completed!")
            return True
            
        except Exception as e:
            print(f"Error training model: {e}")
            return False
    
    def predict_next_week(self, end_date_str):
        """
        Predict stock prices for the next 7 days from the given end date
        
        Args:
            end_date_str: End date of historical data (e.g., '2025-08-01')
        """
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("Cannot make predictions - using fallback method")
            return self._fallback_prediction(end_date_str)
        
        print(f"Predicting next 7 days from {end_date_str}...")
        
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            
            # Get the last lookback_days of data
            end_idx = self.df.index.get_indexer([end_date], method='nearest')[0]
            start_idx = max(0, end_idx - self.lookback_days + 1)
            
            last_sequence = self.df.iloc[start_idx:end_idx + 1][self.features].values
            last_sequence_scaled = self.scaler.transform(last_sequence)
            
            # Make predictions for next 7 days
            predictions = []
            current_sequence = last_sequence_scaled[-self.lookback_days:].reshape(1, self.lookback_days, len(self.features))
            
            for day in range(self.prediction_days):
                # Predict next day
                next_pred = self.model.predict(current_sequence, verbose=0)[0][0]
                predictions.append(next_pred)
                
                # Update sequence for next prediction
                # Create a new row with the predicted close price
                new_row = current_sequence[0, -1, :].copy()
                new_row[3] = next_pred  # Update close price
                
                # Shift sequence and add new prediction
                current_sequence = np.roll(current_sequence, -1, axis=1)
                current_sequence[0, -1, :] = new_row
            
            # Inverse transform predictions (only close price)
            predictions_array = np.zeros((len(predictions), len(self.features)))
            predictions_array[:, 3] = predictions  # Close price column
            
            predictions_inverse = self.scaler.inverse_transform(predictions_array)
            predicted_prices = predictions_inverse[:, 3]
            
            # Generate prediction dates
            prediction_dates = []
            current_date = end_date
            for i in range(self.prediction_days):
                current_date += timedelta(days=1)
                # Skip weekends for stock market
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                prediction_dates.append(current_date)
            
            return prediction_dates, predicted_prices
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None, None
    
    def _fallback_prediction(self, end_date_str):
        """Fallback prediction method using linear regression"""
        from sklearn.linear_model import LinearRegression
        
        print("Using fallback linear regression prediction...")
        
        try:
            end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
            end_idx = self.df.index.get_indexer([end_date], method='nearest')[0]
            
            # Use last 30 days for linear regression
            recent_data = self.df.iloc[max(0, end_idx-29):end_idx+1]
            
            # Prepare features and target
            X = np.arange(len(recent_data)).reshape(-1, 1)
            y = recent_data['Close'].values
            
            # Train model
            model = LinearRegression().fit(X, y)
            
            # Predict next 7 days
            future_X = np.arange(len(recent_data), len(recent_data) + self.prediction_days).reshape(-1, 1)
            predictions = model.predict(future_X)
            
            # Generate prediction dates
            prediction_dates = []
            current_date = end_date
            for i in range(self.prediction_days):
                current_date += timedelta(days=1)
                while current_date.weekday() >= 5:
                    current_date += timedelta(days=1)
                prediction_dates.append(current_date)
            
            return prediction_dates, predictions
            
        except Exception as e:
            print(f"Error in fallback prediction: {e}")
            return None, None
    
    def evaluate_model(self):
        """Evaluate model performance"""
        if not TENSORFLOW_AVAILABLE or self.model is None:
            print("Cannot evaluate - model not available")
            return
            
        print("Evaluating model performance...")
        
        try:
            # Make predictions on test set
            y_pred = self.model.predict(self.X_test, verbose=0)
            
            # Inverse transform for actual values
            y_test_inverse = np.zeros((len(self.y_test), len(self.features)))
            y_test_inverse[:, 3] = self.y_test
            y_test_actual = self.scaler.inverse_transform(y_test_inverse)[:, 3]
            
            y_pred_inverse = np.zeros((len(y_pred), len(self.features)))
            y_pred_inverse[:, 3] = y_pred.flatten()
            y_pred_actual = self.scaler.inverse_transform(y_pred_inverse)[:, 3]
            
            # Calculate metrics
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            
            print(f"Model Performance Metrics:")
            print(f"  RMSE: ${rmse:.2f}")
            print(f"  MAE: ${mae:.2f}")
            print(f"  MSE: {mse:.2f}")
            
            # Calculate percentage accuracy
            mape = np.mean(np.abs((y_test_actual - y_pred_actual) / y_test_actual)) * 100
            print(f"  MAPE: {mape:.2f}%")
            
        except Exception as e:
            print(f"Error evaluating model: {e}")
    
    def plot_predictions(self, prediction_dates, predicted_prices, days_to_show=30):
        """Plot historical data with predictions"""
        print("Creating prediction visualization...")
        
        plt.figure(figsize=(15, 10))
        
        # Get recent historical data
        recent_data = self.df.tail(days_to_show)
        
        # Plot historical data
        plt.subplot(2, 1, 1)
        plt.plot(recent_data.index, recent_data['Close'], label='Historical Close Price', linewidth=2, color='blue')
        
        # Plot technical indicators
        if 'SMA_20' in recent_data.columns:
            plt.plot(recent_data.index, recent_data['SMA_20'], label='SMA 20', alpha=0.7, color='orange')
        if 'BB_upper' in recent_data.columns:
            plt.fill_between(recent_data.index, recent_data['BB_lower'], recent_data['BB_upper'], 
                           alpha=0.2, color='gray', label='Bollinger Bands')
        
        # Plot predictions
        if prediction_dates and predicted_prices is not None:
            plt.plot(prediction_dates, predicted_prices, 'ro-', label='7-Day Predictions', 
                    linewidth=2, markersize=8)
            
            # Add prediction values as annotations
            for date, price in zip(prediction_dates, predicted_prices):
                plt.annotate(f'${price:.2f}', (date, price), xytext=(5, 5), 
                           textcoords='offset points', fontsize=8, 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
        
        plt.title(f'{self.ticker} Stock Price Prediction - Next 7 Days', fontsize=14, fontweight='bold')
        plt.xlabel('Date')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        # Plot RSI if available
        if 'RSI' in recent_data.columns:
            plt.subplot(2, 1, 2)
            plt.plot(recent_data.index, recent_data['RSI'], label='RSI', color='purple')
            plt.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
            plt.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
            plt.fill_between(recent_data.index, 30, 70, alpha=0.1, color='gray')
            plt.title('RSI Indicator')
            plt.ylabel('RSI')
            plt.xlabel('Date')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.xticks(rotation=45)
            plt.ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f'{self.ticker}_lstm_prediction.png', dpi=150, bbox_inches='tight')
        print(f"Chart saved as '{self.ticker}_lstm_prediction.png'")
        plt.show()
    
    def get_technical_analysis_summary(self):
        """Get current technical analysis summary"""
        if self.df is None or len(self.df) == 0:
            return
            
        print(f"\nTechnical Analysis Summary for {self.ticker}")
        print("=" * 50)
        
        latest = self.df.iloc[-1]
        current_price = latest['Close']
        
        print(f"Current Price: ${current_price:.2f}")
        print(f"Date: {self.df.index[-1].strftime('%Y-%m-%d')}")
        
        # RSI Analysis
        if 'RSI' in self.df.columns:
            rsi = latest['RSI']
            if rsi > 70:
                rsi_signal = "Overbought"
            elif rsi < 30:
                rsi_signal = "Oversold"
            else:
                rsi_signal = "Neutral"
            print(f"RSI: {rsi:.2f} ({rsi_signal})")
        
        # Moving Average Analysis
        if 'SMA_20' in self.df.columns and 'SMA_50' in self.df.columns:
            sma20 = latest['SMA_20']
            sma50 = latest['SMA_50']
            
            if current_price > sma20 > sma50:
                ma_trend = "Strong Uptrend"
            elif current_price < sma20 < sma50:
                ma_trend = "Strong Downtrend"
            elif current_price > sma20:
                ma_trend = "Short-term Uptrend"
            else:
                ma_trend = "Short-term Downtrend"
                
            print(f"MA Trend: {ma_trend}")
            print(f"SMA20: ${sma20:.2f}, SMA50: ${sma50:.2f}")
        
        # MACD Analysis
        if 'MACD' in self.df.columns:
            macd = latest['MACD']
            macd_signal = latest['MACD_signal']
            macd_trend = "Bullish" if macd > macd_signal else "Bearish"
            print(f"MACD: {macd_trend} (MACD: {macd:.4f}, Signal: {macd_signal:.4f})")


def main():
    """Main function to run stock prediction"""
    print("Stock Predictor with LSTM and Technical Indicators")
    print("=" * 55)
    
    # Configuration
    TICKER = "AAPL"  # Change this to any stock symbol
    START_DATE = "2020-01-01"
    END_DATE = "2025-08-01"  # Historical data end date
    PREDICTION_DATE = END_DATE  # Predict from this date
    
    # Initialize predictor with all indicators for standalone testing
    all_indicators = [
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'stoch_k', 'stoch_d', 'atr', 'volume_sma', 'obv',
        'high_low_pct', 'open_close_pct'
    ]
    predictor = StockPredictor(TICKER, lookback_days=60, prediction_days=7, selected_indicators=all_indicators)
    
    # Step 1: Fetch data
    if not predictor.fetch_data(START_DATE, END_DATE):
        return
    
    # Step 2: Calculate technical indicators
    if not predictor.calculate_technical_indicators():
        return
    
    # Step 3: Show technical analysis
    predictor.get_technical_analysis_summary()
    
    # Step 4: Prepare data for LSTM
    if not predictor.prepare_lstm_data():
        return
    
    # Step 5: Build and train model (if TensorFlow is available)
    if TENSORFLOW_AVAILABLE:
        if predictor.build_lstm_model():
            predictor.train_model(epochs=50, batch_size=32)
            predictor.evaluate_model()
    
    # Step 6: Make predictions
    prediction_dates, predicted_prices = predictor.predict_next_week(PREDICTION_DATE)
    
    if prediction_dates and predicted_prices is not None:
        print(f"\n7-Day Price Predictions starting from {PREDICTION_DATE}:")
        print("-" * 50)
        current_price = predictor.df['Close'].iloc[-1]
        print(f"Current Price (as of {PREDICTION_DATE}): ${current_price:.2f}")
        
        for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
            change = price - current_price
            change_pct = (change / current_price) * 100
            print(f"Day {i+1} ({date.strftime('%Y-%m-%d')}): ${price:.2f} "
                  f"({change:+.2f}, {change_pct:+.2f}%)")
        
        # Step 7: Create visualization
        predictor.plot_predictions(prediction_dates, predicted_prices, days_to_show=50)
        
        print(f"\nPrediction Summary:")
        print(f"Week High: ${max(predicted_prices):.2f}")
        print(f"Week Low: ${min(predicted_prices):.2f}")
        print(f"Week End: ${predicted_prices[-1]:.2f}")
        week_return = ((predicted_prices[-1] - current_price) / current_price) * 100
        print(f"Expected Week Return: {week_return:+.2f}%")
    
    else:
        print("Failed to generate predictions")

if __name__ == "__main__":
    main()