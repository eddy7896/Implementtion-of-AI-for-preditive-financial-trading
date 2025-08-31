"""
Configuration file for stock prediction
Modify these settings to test different scenarios
"""

# Stock configuration
STOCK_CONFIGS = {
    'AAPL': {
        'name': 'Apple Inc.',
        'sector': 'Technology',
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    },
    'MSFT': {
        'name': 'Microsoft Corporation',
        'sector': 'Technology', 
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    },
    'GOOGL': {
        'name': 'Alphabet Inc.',
        'sector': 'Technology',
        'start_date': '2020-01-01', 
        'end_date': '2025-08-01'
    },
    'TSLA': {
        'name': 'Tesla Inc.',
        'sector': 'Automotive',
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    },
    'NVDA': {
        'name': 'NVIDIA Corporation',
        'sector': 'Technology',
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    }
}

# Model configuration
MODEL_CONFIG = {
    'lookback_days': 60,        # Days of historical data to use for prediction
    'prediction_days': 7,       # Number of days to predict (weekly)
    'epochs': 50,              # Training epochs for LSTM
    'batch_size': 32,          # Batch size for training
    'validation_split': 0.1,   # Validation data split
    'test_size': 0.2          # Test data split
}

# Technical indicators configuration
TECHNICAL_INDICATORS = {
    'sma_periods': [5, 10, 20, 50],    # Simple moving average periods
    'ema_periods': [12, 26],           # Exponential moving average periods  
    'rsi_period': 14,                  # RSI calculation period
    'macd_fast': 12,                   # MACD fast period
    'macd_slow': 26,                   # MACD slow period
    'macd_signal': 9,                  # MACD signal period
    'bollinger_period': 20,            # Bollinger bands period
    'bollinger_std': 2,                # Bollinger bands standard deviation
    'stoch_period': 14,                # Stochastic oscillator period
    'atr_period': 14,                  # Average True Range period
    'volume_sma_period': 20            # Volume SMA period
}

# Visualization configuration
PLOT_CONFIG = {
    'figure_size': (15, 10),
    'dpi': 150,
    'days_to_show': 60,        # Days of historical data to show in plots
    'save_plots': True,        # Whether to save plots to files
    'show_plots': True         # Whether to display plots
}

# Example prediction scenarios
PREDICTION_SCENARIOS = [
    {
        'name': 'Apple August 2025',
        'ticker': 'AAPL',
        'end_date': '2025-08-01',
        'description': 'Predict AAPL for the week of Aug 4-8, 2025'
    },
    {
        'name': 'Tesla August 2025', 
        'ticker': 'TSLA',
        'end_date': '2025-08-01',
        'description': 'Predict TSLA for the week of Aug 4-8, 2025'
    },
    {
        'name': 'Microsoft August 2025',
        'ticker': 'MSFT', 
        'end_date': '2025-08-01',
        'description': 'Predict MSFT for the week of Aug 4-8, 2025'
    }
]

def get_stock_config(ticker):
    """Get configuration for a specific stock ticker"""
    return STOCK_CONFIGS.get(ticker.upper(), {
        'name': f'{ticker.upper()} Stock',
        'sector': 'Unknown',
        'start_date': '2020-01-01',
        'end_date': '2025-08-01'
    })

def get_prediction_dates(end_date_str):
    """
    Generate the next 7 business days after the end date
    
    Args:
        end_date_str: End date in YYYY-MM-DD format
        
    Returns:
        List of date strings for the prediction period
    """
    from datetime import datetime, timedelta
    
    end_date = datetime.strptime(end_date_str, '%Y-%m-%d')
    prediction_dates = []
    
    current_date = end_date
    days_added = 0
    
    while days_added < 7:
        current_date += timedelta(days=1)
        # Skip weekends (Saturday=5, Sunday=6)
        if current_date.weekday() < 5:
            prediction_dates.append(current_date.strftime('%Y-%m-%d'))
            days_added += 1
    
    return prediction_dates

def print_scenario_info(scenario):
    """Print information about a prediction scenario"""
    print(f"Scenario: {scenario['name']}")
    print(f"Ticker: {scenario['ticker']}")
    print(f"End Date: {scenario['end_date']}")
    print(f"Description: {scenario['description']}")
    
    # Get prediction dates
    pred_dates = get_prediction_dates(scenario['end_date'])
    print(f"Prediction Dates: {pred_dates[0]} to {pred_dates[-1]}")
    print()