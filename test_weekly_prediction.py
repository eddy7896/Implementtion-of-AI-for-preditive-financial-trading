#!/usr/bin/env python3
"""
Weekly Stock Prediction Test Script
Demonstrates 7-day stock price prediction using LSTM and technical indicators
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from stock_predictor_lstm import StockPredictor
import pandas as pd
from datetime import datetime, timedelta

def test_weekly_prediction(ticker="AAPL", end_date="2025-08-01"):
    """
    Test weekly prediction functionality
    
    Args:
        ticker: Stock symbol to test
        end_date: End date for historical data (format: YYYY-MM-DD)
    """
    print(f"Testing Weekly Stock Prediction for {ticker}")
    print("=" * 60)
    
    # Calculate start date (5 years of data for better training)
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    start_dt = end_dt - timedelta(days=5*365)
    start_date = start_dt.strftime('%Y-%m-%d')
    
    print(f"Historical Data Period: {start_date} to {end_date}")
    print(f"Prediction Period: 7 days starting from {end_date}")
    print()
    
    # Initialize predictor
    predictor = StockPredictor(ticker, lookback_days=60, prediction_days=7)
    
    # Step 1: Fetch and prepare data
    print("Step 1: Fetching stock data...")
    if not predictor.fetch_data(start_date, end_date):
        print("❌ Failed to fetch data")
        return False
    print("✅ Data fetched successfully")
    
    # Step 2: Calculate technical indicators
    print("\nStep 2: Calculating technical indicators...")
    if not predictor.calculate_technical_indicators():
        print("❌ Failed to calculate technical indicators")
        return False
    print("✅ Technical indicators calculated")
    
    # Display current technical analysis
    predictor.get_technical_analysis_summary()
    
    # Step 3: Prepare LSTM data
    print("\nStep 3: Preparing LSTM training data...")
    if not predictor.prepare_lstm_data():
        print("❌ Failed to prepare LSTM data")
        return False
    print("✅ LSTM data prepared")
    
    # Step 4: Build and train model
    print("\nStep 4: Building and training LSTM model...")
    try:
        if predictor.build_lstm_model():
            print("✅ LSTM model built")
            
            # Train with fewer epochs for faster testing
            if predictor.train_model(epochs=30, batch_size=32):
                print("✅ Model training completed")
                
                # Evaluate model
                predictor.evaluate_model()
            else:
                print("⚠️ Model training failed, using fallback method")
        else:
            print("⚠️ LSTM not available, using fallback method")
    except Exception as e:
        print(f"⚠️ Error with LSTM: {e}, using fallback method")
    
    # Step 5: Make predictions
    print(f"\nStep 5: Predicting next 7 days from {end_date}...")
    prediction_dates, predicted_prices = predictor.predict_next_week(end_date)
    
    if prediction_dates and predicted_prices is not None:
        print("✅ Predictions generated successfully")
        
        # Display predictions
        print("\n" + "="*60)
        print("WEEKLY STOCK PRICE PREDICTIONS")
        print("="*60)
        
        # Get the last known price
        last_price = predictor.df['Close'].iloc[-1]
        print(f"Last Known Price ({end_date}): ${last_price:.2f}")
        print()
        
        print("7-Day Forecast:")
        print("-" * 40)
        total_change = 0
        
        for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
            day_change = price - last_price
            day_change_pct = (day_change / last_price) * 100
            total_change += day_change_pct
            
            # Determine market day name
            day_name = date.strftime('%A')
            date_str = date.strftime('%Y-%m-%d')
            
            print(f"Day {i+1} ({day_name}, {date_str}): ${price:.2f} ({day_change_pct:+.2f}%)")
        
        print("-" * 40)
        print(f"Week High: ${max(predicted_prices):.2f}")
        print(f"Week Low:  ${min(predicted_prices):.2f}")
        print(f"Week End:  ${predicted_prices[-1]:.2f}")
        
        week_return = ((predicted_prices[-1] - last_price) / last_price) * 100
        print(f"Expected Weekly Return: {week_return:+.2f}%")
        
        # Trend analysis
        if predicted_prices[-1] > predicted_prices[0]:
            trend = "📈 Upward"
        elif predicted_prices[-1] < predicted_prices[0]:
            trend = "📉 Downward"
        else:
            trend = "➡️ Sideways"
        print(f"Weekly Trend: {trend}")
        
        # Step 6: Create visualization
        print(f"\nStep 6: Creating visualization...")
        predictor.plot_predictions(prediction_dates, predicted_prices, days_to_show=60)
        print("✅ Chart created and saved")
        
        return True
    else:
        print("❌ Failed to generate predictions")
        return False

def test_multiple_scenarios():
    """Test different scenarios"""
    print("Testing Multiple Prediction Scenarios")
    print("=" * 50)
    
    # Test scenarios
    scenarios = [
        ("AAPL", "2025-08-01"),  # Apple - tech stock
        ("MSFT", "2025-08-01"),  # Microsoft - another tech stock
        ("TSLA", "2025-08-01"),  # Tesla - volatile stock
    ]
    
    for ticker, end_date in scenarios:
        print(f"\n{'='*20} Testing {ticker} {'='*20}")
        try:
            success = test_weekly_prediction(ticker, end_date)
            if success:
                print(f"✅ {ticker} prediction completed successfully")
            else:
                print(f"❌ {ticker} prediction failed")
        except Exception as e:
            print(f"❌ Error testing {ticker}: {e}")
        print("\n" + "-"*50)

def main():
    """Main function"""
    print("Weekly Stock Prediction Test Suite")
    print("=" * 50)
    
    # Test single stock prediction
    print("Testing single stock prediction...")
    success = test_weekly_prediction("AAPL", "2025-08-01")
    
    if success:
        print("\n✅ Weekly prediction test completed successfully!")
        
        # Optionally test multiple scenarios
        test_more = input("\nTest additional stocks? (y/n): ").lower().strip()
        if test_more == 'y':
            test_multiple_scenarios()
    else:
        print("\n❌ Weekly prediction test failed!")

if __name__ == "__main__":
    main()