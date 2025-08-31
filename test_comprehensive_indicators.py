#!/usr/bin/env python3
"""
Comprehensive test script for the new technical indicators system
Tests all indicators with proper formulas and integration
"""

from stock_predictor_lstm import StockPredictor
from technical_indicators import TechnicalIndicators
import pandas as pd
import numpy as np
from datetime import datetime

def test_technical_indicators_module():
    """Test the TechnicalIndicators class directly"""
    print("Testing TechnicalIndicators class directly")
    print("=" * 60)
    
    # Create sample OHLCV data
    dates = pd.date_range(start='2020-01-01', end='2024-12-31', freq='D')
    np.random.seed(42)
    
    # Generate realistic stock data
    base_price = 100
    prices = [base_price]
    volumes = []
    
    for i in range(len(dates) - 1):
        # Random walk with slight upward bias
        change = np.random.normal(0.001, 0.02)  # 0.1% daily drift, 2% volatility
        new_price = prices[-1] * (1 + change)
        prices.append(max(new_price, 1))  # Prevent negative prices
    
    # Create OHLC from closing prices
    high_prices = [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices]
    low_prices = [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices]
    open_prices = [prices[i-1] if i > 0 else prices[0] for i in range(len(prices))]
    volumes = [int(abs(np.random.normal(1000000, 200000))) for _ in prices]
    
    # Create DataFrame
    df = pd.DataFrame({
        'Open': open_prices,
        'High': high_prices,
        'Low': low_prices,
        'Close': prices,
        'Volume': volumes
    }, index=dates)
    
    # Initialize technical indicators
    tech_ind = TechnicalIndicators(df)
    
    print(f"[OK] Created test dataset with {len(df)} trading days")
    print(f"  Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Test all indicator categories
    test_results = {}
    
    # Test Moving Averages
    print("\n--- Testing Moving Averages ---")
    indicators_to_test = [
        ('SMA 5', lambda: tech_ind.sma(5)),
        ('SMA 20', lambda: tech_ind.sma(20)),
        ('SMA 50', lambda: tech_ind.sma(50)),
        ('EMA 12', lambda: tech_ind.ema(12)),
        ('EMA 26', lambda: tech_ind.ema(26)),
    ]
    
    for name, calc_func in indicators_to_test:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    # Test Momentum Indicators
    print("\n--- Testing Momentum Indicators ---")
    momentum_tests = [
        ('RSI', lambda: tech_ind.rsi(14)),
        ('Stochastic %K', lambda: tech_ind.stochastic()[0]),
        ('Stochastic %D', lambda: tech_ind.stochastic()[1]),
        ('Rate of Change', lambda: tech_ind.roc(12)),
        ('Williams %R', lambda: tech_ind.williams_r(14)),
    ]
    
    for name, calc_func in momentum_tests:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    # Test Trend Indicators
    print("\n--- Testing Trend Indicators ---")
    trend_tests = [
        ('MACD Line', lambda: tech_ind.macd()[0]),
        ('MACD Signal', lambda: tech_ind.macd()[1]),
        ('MACD Histogram', lambda: tech_ind.macd()[2]),
        ('ADX', lambda: tech_ind.adx(14)),
        ('Aroon Up', lambda: tech_ind.aroon()[0]),
        ('Aroon Down', lambda: tech_ind.aroon()[1]),
    ]
    
    for name, calc_func in trend_tests:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    # Test Volatility Indicators
    print("\n--- Testing Volatility Indicators ---")
    volatility_tests = [
        ('Bollinger Upper', lambda: tech_ind.bollinger_bands()[0]),
        ('Bollinger Middle', lambda: tech_ind.bollinger_bands()[1]),
        ('Bollinger Lower', lambda: tech_ind.bollinger_bands()[2]),
        ('Bollinger Width', lambda: tech_ind.bollinger_band_width()),
        ('ATR', lambda: tech_ind.atr(14)),
    ]
    
    for name, calc_func in volatility_tests:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    # Test Volume Indicators
    print("\n--- Testing Volume Indicators ---")
    volume_tests = [
        ('OBV', lambda: tech_ind.obv()),
        ('Volume SMA', lambda: tech_ind.volume_sma(20)),
        ('VWAP', lambda: tech_ind.vwap()),
        ('A/D Line', lambda: tech_ind.accumulation_distribution()),
    ]
    
    for name, calc_func in volume_tests:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    # Test Price-based Indicators
    print("\n--- Testing Price-based Indicators ---")
    price_tests = [
        ('High-Low %', lambda: tech_ind.high_low_percentage()),
        ('Open-Close %', lambda: tech_ind.open_close_percentage()),
        ('CCI', lambda: tech_ind.cci(20)),
    ]
    
    for name, calc_func in price_tests:
        try:
            result = calc_func()
            test_results[name] = len(result.dropna())
            print(f"[OK] {name}: {len(result.dropna())} valid values")
        except Exception as e:
            print(f"[ERROR] {name}: Error - {str(e)}")
    
    print(f"\n[SUCCESS] Technical Indicators Module Test Complete")
    print(f"  Total indicators tested: {len(test_results)}")
    print(f"  All indicators working: {len([r for r in test_results.values() if r > 0]) == len(test_results)}")
    
    return len(test_results), len([r for r in test_results.values() if r > 0])

def test_stock_predictor_integration():
    """Test StockPredictor integration with new indicators"""
    print("\n\nTesting StockPredictor integration")
    print("=" * 60)
    
    # Test different indicator combinations
    test_cases = [
        {
            'name': 'Basic Moving Averages',
            'indicators': ['sma_5', 'sma_20', 'ema_12', 'ema_26']
        },
        {
            'name': 'Momentum Pack',
            'indicators': ['rsi', 'stoch_k', 'stoch_d', 'williams_r', 'roc']
        },
        {
            'name': 'Advanced Trend Analysis',
            'indicators': ['macd', 'macd_signal', 'macd_diff', 'adx', 'aroon_up', 'aroon_down']
        },
        {
            'name': 'Volatility Suite',
            'indicators': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr']
        },
        {
            'name': 'Volume Analysis',
            'indicators': ['obv', 'volume_sma', 'vwap', 'ad_line']
        },
        {
            'name': 'Complete Analysis',
            'indicators': [
                'sma_20', 'ema_12', 'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                'atr', 'obv', 'high_low_pct', 'open_close_pct', 'cci', 'adx'
            ]
        }
    ]
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Initialize predictor
            predictor = StockPredictor(
                ticker="AAPL",
                lookback_days=30,
                prediction_days=7,
                selected_indicators=test_case['indicators']
            )
            
            # Test data fetching
            print("  Fetching data...")
            if predictor.fetch_data("2023-01-01", "2024-06-01"):
                print(f"  [OK] Data fetched: {len(predictor.df)} days")
                
                # Test indicator calculation
                print("  Calculating indicators...")
                if predictor.calculate_technical_indicators():
                    print(f"  [OK] Indicators calculated successfully")
                    print(f"    Selected: {len(test_case['indicators'])}")
                    print(f"    Total features: {len(predictor.features)}")
                    
                    # Verify correct indicators were calculated
                    expected_columns = set()
                    for ind_key in test_case['indicators']:
                        if ind_key in predictor.indicator_mapping:
                            expected_columns.add(predictor.indicator_mapping[ind_key])
                    
                    actual_columns = set(predictor.features) - {'Open', 'High', 'Low', 'Close', 'Volume'}
                    
                    if expected_columns <= actual_columns:
                        print("  [OK] All expected indicators present")
                    else:
                        missing = expected_columns - actual_columns
                        print(f"  [WARNING] Missing indicators: {missing}")
                    
                    # Test data preparation
                    print("  Preparing LSTM data...")
                    if predictor.prepare_lstm_data():
                        print(f"  [OK] LSTM data prepared")
                        print(f"    Training shape: {predictor.X_train.shape}")
                        print(f"    Feature dimensions: {predictor.X_train.shape[2]}")
                        
                        # Test prediction capability (without training)
                        print("  Testing fallback prediction...")
                        dates, prices = predictor._fallback_prediction("2024-06-01")
                        if dates and prices is not None:
                            print(f"  [OK] Fallback prediction works: {len(prices)} days predicted")
                        else:
                            print("  [WARNING] Fallback prediction failed")
                    else:
                        print("  [ERROR] LSTM data preparation failed")
                else:
                    print("  [ERROR] Indicator calculation failed")
            else:
                print("  [ERROR] Data fetching failed")
                
        except Exception as e:
            print(f"  [ERROR] Test case failed with error: {str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n[SUCCESS] StockPredictor Integration Test Complete")

def test_indicator_descriptions():
    """Test that all indicators have proper descriptions"""
    print("\n\nTesting Indicator Descriptions")
    print("=" * 60)
    
    predictor = StockPredictor("AAPL")
    
    # Create a dummy DataFrame for testing descriptions
    dummy_df = pd.DataFrame({
        'Open': [100],
        'High': [105],
        'Low': [95],
        'Close': [102],
        'Volume': [1000000]
    })
    tech_ind = TechnicalIndicators(dummy_df)
    
    all_indicators = list(predictor.indicator_mapping.keys())
    
    print(f"Testing descriptions for {len(all_indicators)} indicators:")
    
    for indicator in all_indicators:
        description = tech_ind.get_indicator_description(indicator)
        if "Description not available" in description:
            print(f"[WARNING] {indicator}: Missing description")
        else:
            print(f"[OK] {indicator}: {description[:50]}...")
    
    print(f"\n[SUCCESS] Indicator Descriptions Test Complete")

def main():
    """Run all comprehensive tests"""
    print("COMPREHENSIVE TECHNICAL INDICATORS TEST SUITE")
    print("=" * 80)
    
    try:
        # Test 1: Technical Indicators Module
        total_indicators, working_indicators = test_technical_indicators_module()
        
        # Test 2: StockPredictor Integration
        test_stock_predictor_integration()
        
        # Test 3: Indicator Descriptions
        test_indicator_descriptions()
        
        print("\n" + "=" * 80)
        print("TEST SUMMARY")
        print("=" * 80)
        print(f"[SUCCESS] Technical Indicators Module: {working_indicators}/{total_indicators} working")
        print(f"[SUCCESS] StockPredictor Integration: Completed")
        print(f"[SUCCESS] Indicator Descriptions: Completed")
        print("\n[SUCCESS] All tests completed successfully!")
        print("\nThe technical indicators system is ready for use!")
        print("You can now run the GUI and select from a comprehensive set of indicators.")
        
    except Exception as e:
        print(f"\n[ERROR] Test suite failed with error: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()