#!/usr/bin/env python3
"""
Demonstration Script: Weekly Stock Price Prediction
Shows exactly how to predict 7 days from historical data end date

Example: If historical data ends on 2025-08-01, predict 2025-08-04 to 2025-08-08
"""

import sys
from datetime import datetime, timedelta
from stock_predictor_lstm import StockPredictor

def demonstrate_weekly_prediction():
    """
    Demonstrate weekly stock prediction exactly as requested:
    - Historical data ends on 2025-08-01 (Friday)
    - Predict next 7 trading days: 2025-08-04 to 2025-08-08 (Mon-Fri)
    """
    
    print("WEEKLY STOCK PRICE PREDICTION DEMONSTRATION")
    print("=" * 60)
    print("Scenario: Predict stock prices for the week following historical data")
    print("Historical Data End Date: 2025-08-01 (Friday)")
    print("Prediction Target: Next 7 trading days (2025-08-04 to 2025-08-08)")
    print()
    
    # Configuration
    TICKER = "AAPL"  # Apple Inc.
    HISTORICAL_END_DATE = "2025-08-01"  # End of historical data
    START_DATE = "2020-01-01"  # Start of historical data (5+ years for training)
    
    print(f"Stock Ticker: {TICKER}")
    print(f"Training Data Period: {START_DATE} to {HISTORICAL_END_DATE}")
    print()
    
    # Initialize the predictor
    predictor = StockPredictor(
        ticker=TICKER, 
        lookback_days=60,      # Use 60 days of history for each prediction
        prediction_days=7      # Predict 7 days ahead
    )
    
    # Step 1: Fetch historical stock data
    print("STEP 1: Fetching Historical Stock Data")
    print("-" * 40)
    success = predictor.fetch_data(START_DATE, HISTORICAL_END_DATE)
    if not success:
        print("[ERROR] Failed to fetch stock data")
        return
    
    print(f"[SUCCESS] Successfully fetched {len(predictor.df)} trading days of data")
    print(f"Date range: {predictor.df.index[0].strftime('%Y-%m-%d')} to {predictor.df.index[-1].strftime('%Y-%m-%d')}")
    print(f"Last close price: ${predictor.df['Close'].iloc[-1]:.2f}")
    print()
    
    # Step 2: Calculate technical indicators
    print("STEP 2: Calculating Technical Indicators")
    print("-" * 40)
    success = predictor.calculate_technical_indicators()
    if not success:
        print("[ERROR] Failed to calculate technical indicators")
        return
    
    indicators_calculated = len(predictor.features)
    print(f"[SUCCESS] Calculated {indicators_calculated} technical indicators:")
    print("   - Moving Averages (SMA 5,10,20,50 & EMA 12,26)")
    print("   - RSI (Relative Strength Index)")
    print("   - MACD (Moving Average Convergence Divergence)")
    print("   - Bollinger Bands")
    print("   - Stochastic Oscillator")
    print("   - ATR (Average True Range)")
    print("   - Volume indicators (OBV, Volume SMA)")
    print("   - Price-based features")
    print()
    
    # Show current technical analysis
    predictor.get_technical_analysis_summary()
    
    # Step 3: Prepare data for LSTM training
    print("\nSTEP 3: Preparing LSTM Training Data")
    print("-" * 40)
    success = predictor.prepare_lstm_data(test_size=0.2)
    if not success:
        print("[ERROR] Failed to prepare training data")
        return
    
    print(f"[SUCCESS] Training data prepared")
    print(f"   Training samples: {len(predictor.X_train)}")
    print(f"   Testing samples: {len(predictor.X_test)}")
    print(f"   Features per sample: {len(predictor.features)}")
    print(f"   Lookback window: {predictor.lookback_days} days")
    print()
    
    # Step 4: Build and train LSTM model
    print("STEP 4: Building and Training LSTM Model")
    print("-" * 40)
    
    try:
        if predictor.build_lstm_model():
            print("[SUCCESS] LSTM model architecture built:")
            print("   - 3 LSTM layers (50 neurons each)")
            print("   - Dropout layers (0.2) for regularization")
            print("   - Dense output layer")
            print("   - Adam optimizer with MSE loss")
            print()
            
            print("Training model (this may take a few minutes)...")
            if predictor.train_model(epochs=30, batch_size=32, validation_split=0.1):
                print("[SUCCESS] Model training completed successfully")
                predictor.evaluate_model()
            else:
                print("[WARNING] LSTM training failed, will use linear regression fallback")
        else:
            print("[WARNING] LSTM not available, will use linear regression fallback")
    except Exception as e:
        print(f"[WARNING] Error with LSTM: {str(e)[:100]}...")
        print("Will use linear regression fallback method")
    
    print()
    
    # Step 5: Generate predictions for the next week
    print("STEP 5: Generating Weekly Predictions")
    print("-" * 40)
    print(f"Predicting 7 trading days starting from {HISTORICAL_END_DATE}")
    
    prediction_dates, predicted_prices = predictor.predict_next_week(HISTORICAL_END_DATE)
    
    if prediction_dates and predicted_prices is not None:
        print("[SUCCESS] Weekly predictions generated successfully!")
        print()
        
        # Display the predictions
        display_predictions(predictor, prediction_dates, predicted_prices, HISTORICAL_END_DATE)
        
        # Step 6: Create visualization
        print("STEP 6: Creating Visualization")
        print("-" * 40)
        predictor.plot_predictions(prediction_dates, predicted_prices, days_to_show=60)
        print("[SUCCESS] Prediction chart saved and displayed")
        
        return True
    else:
        print("[ERROR] Failed to generate predictions")
        return False

def display_predictions(predictor, prediction_dates, predicted_prices, end_date):
    """Display the prediction results in a formatted table"""
    
    last_known_price = predictor.df['Close'].iloc[-1]
    
    print("WEEKLY STOCK PRICE PREDICTIONS")
    print("=" * 60)
    print(f"Stock: {predictor.ticker}")
    print(f"Historical data ends: {end_date}")
    print(f"Last known price: ${last_known_price:.2f}")
    print()
    
    print("7-Day Forecast:")
    print("+" + "-"*58 + "+")
    print("| Day |   Date     | Day Name | Predicted Price | Change  | Change% |")
    print("+" + "-"*58 + "+")
    
    for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
        day_name = date.strftime('%A')[:3]  # Mon, Tue, etc.
        date_str = date.strftime('%Y-%m-%d')
        change = price - last_known_price
        change_pct = (change / last_known_price) * 100
        
        print(f"| {i+1:2d}  | {date_str} | {day_name:8s} | ${price:10.2f}   | {change:+6.2f} | {change_pct:+6.2f}% |")
    
    print("+" + "-"*58 + "+")
    
    # Summary statistics
    week_high = max(predicted_prices)
    week_low = min(predicted_prices)
    week_end = predicted_prices[-1]
    week_volatility = ((week_high - week_low) / last_known_price) * 100
    
    print()
    print("WEEK SUMMARY:")
    print(f"Expected High:  ${week_high:.2f}")
    print(f"Expected Low:   ${week_low:.2f}")
    print(f"Week End Price: ${week_end:.2f}")
    print(f"Week Volatility: {week_volatility:.2f}%")
    
    weekly_return = ((week_end - last_known_price) / last_known_price) * 100
    print(f"Expected Weekly Return: {weekly_return:+.2f}%")
    
    # Trend direction
    if week_end > last_known_price:
        trend_desc = "BULLISH (Upward)"
    elif week_end < last_known_price:
        trend_desc = "BEARISH (Downward)"
    else:
        trend_desc = "NEUTRAL (Sideways)"
    
    print(f"Weekly Trend: {trend_desc}")
    
    print()
    print("PREDICTION METHODOLOGY:")
    print("- Uses LSTM neural network trained on 5+ years of historical data")
    print("- Incorporates 26+ technical indicators (RSI, MACD, Bollinger Bands, etc.)")
    print("- 60-day lookback window for pattern recognition")
    print("- Accounts for price, volume, and volatility patterns")
    print("- Predictions are for trading days only (excludes weekends)")

def show_example_scenarios():
    """Show different example scenarios"""
    print("\nEXAMPLE USAGE SCENARIOS:")
    print("=" * 40)
    
    scenarios = [
        ("AAPL", "2025-08-01", "Apple Inc. - Tech giant, stable growth"),
        ("TSLA", "2025-08-01", "Tesla Inc. - EV leader, high volatility"),
        ("MSFT", "2025-08-01", "Microsoft - Cloud services, steady performer"),
        ("NVDA", "2025-08-01", "NVIDIA - AI/GPU leader, growth stock")
    ]
    
    for ticker, end_date, description in scenarios:
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        next_monday = end_dt + timedelta(days=(7 - end_dt.weekday()) % 7 + 1)
        next_friday = next_monday + timedelta(days=4)
        
        print(f"{ticker:4s}: {description}")
        print(f"      Historical data: 2020-01-01 to {end_date}")
        print(f"      Predict week: {next_monday.strftime('%Y-%m-%d')} to {next_friday.strftime('%Y-%m-%d')}")
        print()

def main():
    """Main demonstration function"""
    print("=" * 70)
    print("STOCK PREDICTION USING LSTM AND TECHNICAL INDICATORS")
    print("Weekly Forecasting Demonstration")
    print("=" * 70)
    print()
    
    # Show example scenarios
    show_example_scenarios()
    
    # Run the main demonstration
    print("Running demonstration with AAPL...")
    print()
    
    success = demonstrate_weekly_prediction()
    
    if success:
        print("\n" + "=" * 70)
        print("[SUCCESS] DEMONSTRATION COMPLETED SUCCESSFULLY!")
        print("=" * 70)
        print("\nKey Features Demonstrated:")
        print("• Historical data collection and processing")
        print("• Technical indicator calculation (26+ indicators)")
        print("• LSTM neural network training") 
        print("• 7-day stock price prediction")
        print("• Comprehensive visualization")
        print("• Performance metrics and analysis")
        print()
        print("Files generated:")
        print("- AAPL_lstm_prediction.png - Prediction chart")
        print()
        print("To test other stocks, modify the TICKER variable in the script.")
        
    else:
        print("\n[ERROR] Demonstration failed. Check error messages above.")

if __name__ == "__main__":
    main()