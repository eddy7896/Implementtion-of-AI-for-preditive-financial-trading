#!/usr/bin/env python3
"""
Test script to verify that the StockPredictor works with selective technical indicators
"""

from stock_predictor_lstm import StockPredictor
import pandas as pd

def test_selective_indicators():
    """Test StockPredictor with different indicator combinations"""
    
    print("Testing StockPredictor with selective technical indicators")
    print("=" * 60)
    
    # Test configurations
    test_cases = [
        {
            'name': 'Basic SMA indicators only',
            'indicators': ['sma_5', 'sma_10', 'sma_20']
        },
        {
            'name': 'RSI and MACD only',
            'indicators': ['rsi', 'macd', 'macd_signal', 'macd_diff']
        },
        {
            'name': 'Bollinger Bands and EMA',
            'indicators': ['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'ema_12', 'ema_26']
        },
        {
            'name': 'Volume indicators',
            'indicators': ['volume_sma', 'obv']
        },
        {
            'name': 'Price percentage indicators',
            'indicators': ['high_low_pct', 'open_close_pct']
        },
        {
            'name': 'All indicators',
            'indicators': [
                'sma_5', 'sma_10', 'sma_20', 'sma_50',
                'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_diff',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'stoch_k', 'stoch_d', 'atr', 'volume_sma', 'obv',
                'high_low_pct', 'open_close_pct'
            ]
        }
    ]
    
    # Test each configuration
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Initialize predictor with selected indicators
            predictor = StockPredictor(
                ticker="AAPL", 
                lookback_days=30, 
                prediction_days=7,
                selected_indicators=test_case['indicators']
            )
            
            # Test data fetching
            print("Fetching data...")
            if predictor.fetch_data("2024-01-01", "2024-06-01"):
                print(f"[OK] Data fetched successfully: {len(predictor.df)} days")
                
                # Test indicator calculation
                print("Calculating indicators...")
                if predictor.calculate_technical_indicators():
                    print(f"[OK] Indicators calculated successfully")
                    print(f"  Selected indicators: {len(test_case['indicators'])}")
                    print(f"  Total features: {len(predictor.features)}")
                    print(f"  Feature columns: {predictor.features}")
                    
                    # Verify that only requested indicators are calculated
                    expected_indicators = set()
                    for indicator in test_case['indicators']:
                        if indicator in predictor.indicator_mapping:
                            expected_indicators.add(predictor.indicator_mapping[indicator])
                    
                    calculated_indicators = set(predictor.features) - {'Open', 'High', 'Low', 'Close', 'Volume'}
                    
                    if calculated_indicators == expected_indicators:
                        print("[OK] Only selected indicators were calculated")
                    else:
                        print(f"[WARNING] Indicator mismatch!")
                        print(f"  Expected: {expected_indicators}")
                        print(f"  Calculated: {calculated_indicators}")
                    
                    # Test data preparation
                    print("Preparing LSTM data...")
                    if predictor.prepare_lstm_data():
                        print(f"[OK] LSTM data prepared successfully")
                        print(f"  Training samples: {predictor.X_train.shape}")
                        print(f"  Feature dimensions: {predictor.X_train.shape[2]}")
                    else:
                        print("[ERROR] Failed to prepare LSTM data")
                else:
                    print("[ERROR] Failed to calculate indicators")
            else:
                print("[ERROR] Failed to fetch data")
                
        except Exception as e:
            print(f"[ERROR] Test failed with error: {str(e)}")
        
        print()  # Add spacing between test cases

def test_mapping_consistency():
    """Test that GUI checkbox keys map correctly to DataFrame columns"""
    
    print("\nTesting GUI checkbox to DataFrame column mapping")
    print("=" * 50)
    
    predictor = StockPredictor("AAPL")
    
    # All GUI checkbox keys
    gui_keys = [
        'sma_5', 'sma_10', 'sma_20', 'sma_50',
        'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'stoch_k', 'stoch_d', 'atr', 'volume_sma', 'obv',
        'high_low_pct', 'open_close_pct'
    ]
    
    print("Checking mapping consistency...")
    all_mapped = True
    
    for key in gui_keys:
        if key in predictor.indicator_mapping:
            column_name = predictor.indicator_mapping[key]
            print(f"[OK] {key} -> {column_name}")
        else:
            print(f"[ERROR] {key} -> NOT MAPPED")
            all_mapped = False
    
    if all_mapped:
        print("\n[OK] All GUI checkbox keys are properly mapped!")
    else:
        print("\n[ERROR] Some GUI checkbox keys are missing from mapping!")

if __name__ == "__main__":
    # Test selective indicators
    test_selective_indicators()
    
    # Test mapping consistency
    test_mapping_consistency()
    
    print("\n" + "=" * 60)
    print("Testing completed! You can now run the GUI and test with different")
    print("indicator combinations. The selected indicators will be logged in")
    print("the GUI and only those indicators will be calculated and used.")
    print("=" * 60)