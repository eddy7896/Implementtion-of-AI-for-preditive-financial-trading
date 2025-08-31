#!/usr/bin/env python3
"""
=============================================================================
COMPREHENSIVE TECHNICAL INDICATORS MODULE
=============================================================================

This module provides a complete collection of professional-grade technical
indicators for stock market analysis. Each indicator is implemented with
proper mathematical formulas, comprehensive documentation, and optimized
calculations using pandas and numpy.

Key Features:
- 29+ technical indicators across all major categories
- Mathematically accurate implementations
- Professional documentation with formulas
- Optimized for performance with vectorized operations
- Easy integration with stock prediction systems

Categories Covered:
- Trend Indicators (SMA, EMA, MACD, ADX)
- Momentum Indicators (RSI, Stochastic, Williams %R)
- Volatility Indicators (Bollinger Bands, ATR, Keltner Channels)
- Volume Indicators (OBV, VWAP, A/D Line)
- Price-based Indicators (High-Low %, Open-Close %)
- Oscillators (CCI, Aroon)

=============================================================================
"""

import pandas as pd
import numpy as np
from typing import Union, Tuple, Optional

class TechnicalIndicators:
    """
    ==========================================================================
    COMPREHENSIVE TECHNICAL INDICATORS CLASS
    ==========================================================================
    
    This class provides a complete suite of technical indicators for financial
    market analysis. Each method implements a specific indicator with proper
    mathematical formulas and professional-grade calculations.
    
    The class is designed to work with OHLCV (Open, High, Low, Close, Volume)
    data and provides both individual indicator calculations and bulk processing
    capabilities.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        ======================================================================
        INITIALIZATION - PREPARE DATA FOR INDICATOR CALCULATIONS
        ======================================================================
        
        Sets up the technical indicators calculator with stock price data.
        Extracts individual price series for efficient calculations.
        
        Args:
            df: DataFrame with OHLCV data containing columns:
                - 'Open': Opening prices
                - 'High': High prices
                - 'Low': Low prices
                - 'Close': Closing prices
                - 'Volume': Trading volume
                
        The initialization creates individual series references for:
        - Fast access to price data
        - Vectorized mathematical operations
        - Memory-efficient calculations
        """
        # Store a copy of the original DataFrame to avoid modifying source data
        self.df = df.copy()
        
        # Extract individual price series for efficient calculations
        # These series will be used repeatedly in indicator calculations
        self.close = df['Close']    # Most commonly used for trend analysis
        self.high = df['High']      # Used for volatility and momentum calculations
        self.low = df['Low']        # Used for support/resistance analysis
        self.open = df['Open']      # Used for gap analysis and price change calculations
        self.volume = df['Volume']  # Used for volume-based indicators
    
    # ==========================================================================
    # TREND INDICATORS - IDENTIFY MARKET DIRECTION AND MOMENTUM
    # ==========================================================================
    """
    Trend indicators help identify the overall direction of price movement
    and the strength of that trend. They are fundamental to technical analysis
    and form the backbone of many trading strategies.
    
    Key Trend Indicators Implemented:
    - Simple Moving Average (SMA): Basic trend identification
    - Exponential Moving Average (EMA): Responsive trend analysis
    - MACD: Trend changes and momentum shifts
    - DEMA/TEMA: Advanced smoothed trend indicators
    """
    
    def sma(self, period: int = 20, column: str = 'Close') -> pd.Series:
        """
        ======================================================================
        SIMPLE MOVING AVERAGE (SMA) - BASIC TREND IDENTIFICATION
        ======================================================================
        
        The Simple Moving Average is the most fundamental trend indicator.
        It smooths price data by calculating the arithmetic mean of prices
        over a specified number of periods.
        
        Mathematical Formula:
        SMA = (P1 + P2 + P3 + ... + Pn) / n
        
        Where:
        - P = Price at each period
        - n = Number of periods
        
        Usage in Trading:
        - Price above SMA = Uptrend
        - Price below SMA = Downtrend
        - SMA slope indicates trend strength
        - Multiple SMAs create support/resistance levels
        
        Args:
            period: Number of periods for calculation (typical: 5, 10, 20, 50, 200)
            column: Column to calculate SMA for (default: 'Close')
            
        Returns:
            pd.Series: SMA values with same index as input data
        """
        # Use pandas rolling window for efficient calculation
        # min_periods=1 ensures we get values from the first data point
        return self.df[column].rolling(window=period, min_periods=1).mean()
    
    def ema(self, period: int = 12, column: str = 'Close') -> pd.Series:
        """
        ======================================================================
        EXPONENTIAL MOVING AVERAGE (EMA) - RESPONSIVE TREND ANALYSIS
        ======================================================================
        
        The Exponential Moving Average gives more weight to recent prices,
        making it more responsive to price changes than the Simple Moving Average.
        This responsiveness makes it valuable for identifying trend changes early.
        
        Mathematical Formulas:
        1. Smoothing Factor: α = 2 / (period + 1)
        2. EMA Today = (Price Today × α) + (EMA Yesterday × (1 - α))
        
        Key Properties:
        - More weight on recent prices
        - Faster response to price changes
        - Less lag than SMA
        - Better for volatile markets
        
        Trading Applications:
        - Early trend change detection
        - Dynamic support/resistance levels
        - Crossover strategies with other EMAs
        - MACD calculation component
        
        Args:
            period: Number of periods for calculation (typical: 12, 26, 50)
            column: Column to calculate EMA for (default: 'Close')
            
        Returns:
            pd.Series: EMA values with exponential weighting
        """
        # Pandas ewm() implements exponential weighted moving average
        # adjust=False uses recursive formula for consistent results
        return self.df[column].ewm(span=period, adjust=False).mean()
    
    def macd(self, fast_period: int = 12, slow_period: int = 26, signal_period: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        ======================================================================
        MACD - MOVING AVERAGE CONVERGENCE DIVERGENCE
        ======================================================================
        
        MACD is one of the most popular trend-following momentum indicators.
        It shows the relationship between two moving averages of a security's price.
        
        Mathematical Formulas:
        1. MACD Line = EMA(12) - EMA(26)
        2. Signal Line = EMA(MACD Line, 9)
        3. MACD Histogram = MACD Line - Signal Line
        
        Component Analysis:
        - MACD Line: Difference between fast and slow EMA
        - Signal Line: Smoothed version of MACD line
        - Histogram: Difference between MACD and Signal lines
        
        Trading Signals:
        - MACD above Signal Line = Bullish momentum
        - MACD below Signal Line = Bearish momentum
        - MACD crossing above zero = Uptrend confirmation
        - MACD crossing below zero = Downtrend confirmation
        - Histogram expanding = Momentum increasing
        - Histogram contracting = Momentum decreasing
        
        Args:
            fast_period: Fast EMA period (standard: 12)
            slow_period: Slow EMA period (standard: 26)
            signal_period: Signal line EMA period (standard: 9)
            
        Returns:
            Tuple containing:
            - MACD line (fast EMA - slow EMA)
            - Signal line (EMA of MACD line)
            - Histogram (MACD line - Signal line)
        """
        # Calculate the fast and slow EMAs
        ema_fast = self.ema(fast_period)
        ema_slow = self.ema(slow_period)
        
        # MACD line is the difference between fast and slow EMAs
        macd_line = ema_fast - ema_slow
        
        # Signal line is EMA of the MACD line
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        
        # Histogram shows the difference between MACD and Signal lines
        histogram = macd_line - signal_line
        
        return macd_line, signal_line, histogram
    
    def dema(self, period: int = 14) -> pd.Series:
        """
        Double Exponential Moving Average (DEMA)
        
        Formula: DEMA = 2 × EMA(period) - EMA(EMA(period))
        
        Args:
            period: Number of periods for calculation
            
        Returns:
            pd.Series: DEMA values
        """
        ema1 = self.ema(period)
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        return 2 * ema1 - ema2
    
    def tema(self, period: int = 14) -> pd.Series:
        """
        Triple Exponential Moving Average (TEMA)
        
        Formula: TEMA = 3×EMA1 - 3×EMA2 + EMA3
        Where: EMA1 = EMA(price), EMA2 = EMA(EMA1), EMA3 = EMA(EMA2)
        
        Args:
            period: Number of periods for calculation
            
        Returns:
            pd.Series: TEMA values
        """
        ema1 = self.ema(period)
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        return 3 * ema1 - 3 * ema2 + ema3
    
    # =================================================================
    # MOMENTUM INDICATORS (RSI, Stochastic, ROC, etc.)
    # =================================================================
    
    def rsi(self, period: int = 14) -> pd.Series:
        """
        Relative Strength Index (RSI)
        
        Formulas:
        - RS = Average Gain / Average Loss
        - RSI = 100 - (100 / (1 + RS))
        - Average Gain = SMA of gains over period
        - Average Loss = SMA of losses over period
        
        Args:
            period: Number of periods for calculation (default: 14)
            
        Returns:
            pd.Series: RSI values (0-100)
        """
        delta = self.close.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        avg_gain = gains.ewm(alpha=1/period, adjust=False).mean()
        avg_loss = losses.ewm(alpha=1/period, adjust=False).mean()
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def stochastic(self, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """
        Stochastic Oscillator (%K and %D)
        
        Formulas:
        - %K = 100 × (Current Close - Lowest Low) / (Highest High - Lowest Low)
        - %D = SMA(%K, d_period)
        
        Args:
            k_period: Period for %K calculation (default: 14)
            d_period: Period for %D smoothing (default: 3)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (%K values, %D values)
        """
        lowest_low = self.low.rolling(window=k_period).min()
        highest_high = self.high.rolling(window=k_period).max()
        
        k_percent = 100 * (self.close - lowest_low) / (highest_high - lowest_low)
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return k_percent, d_percent
    
    def roc(self, period: int = 12) -> pd.Series:
        """
        Rate of Change (ROC)
        
        Formula: ROC = ((Current Price - Price n periods ago) / Price n periods ago) × 100
        
        Args:
            period: Number of periods to look back (default: 12)
            
        Returns:
            pd.Series: ROC values in percentage
        """
        return ((self.close - self.close.shift(period)) / self.close.shift(period)) * 100
    
    def momentum(self, period: int = 10) -> pd.Series:
        """
        Momentum Indicator
        
        Formula: Momentum = Current Price - Price n periods ago
        
        Args:
            period: Number of periods to look back (default: 10)
            
        Returns:
            pd.Series: Momentum values
        """
        return self.close - self.close.shift(period)
    
    def williams_r(self, period: int = 14) -> pd.Series:
        """
        Williams %R
        
        Formula: %R = -100 × (Highest High - Current Close) / (Highest High - Lowest Low)
        
        Args:
            period: Number of periods for calculation (default: 14)
            
        Returns:
            pd.Series: Williams %R values (-100 to 0)
        """
        highest_high = self.high.rolling(window=period).max()
        lowest_low = self.low.rolling(window=period).min()
        
        williams_r = -100 * (highest_high - self.close) / (highest_high - lowest_low)
        
        return williams_r
    
    # =================================================================
    # VOLATILITY INDICATORS (Bollinger Bands, ATR, etc.)
    # =================================================================
    
    def bollinger_bands(self, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Bollinger Bands
        
        Formulas:
        - Middle Band = SMA(period)
        - Upper Band = Middle Band + (std_dev × Standard Deviation)
        - Lower Band = Middle Band - (std_dev × Standard Deviation)
        
        Args:
            period: Number of periods for SMA calculation (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Upper band, Middle band, Lower band)
        """
        middle_band = self.sma(period)
        std = self.close.rolling(window=period).std()
        
        upper_band = middle_band + (std_dev * std)
        lower_band = middle_band - (std_dev * std)
        
        return upper_band, middle_band, lower_band
    
    def bollinger_band_width(self, period: int = 20, std_dev: float = 2) -> pd.Series:
        """
        Bollinger Band Width
        
        Formula: BB Width = (Upper Band - Lower Band) / Middle Band
        
        Args:
            period: Number of periods for BB calculation (default: 20)
            std_dev: Number of standard deviations (default: 2)
            
        Returns:
            pd.Series: Bollinger Band Width values
        """
        upper, middle, lower = self.bollinger_bands(period, std_dev)
        return (upper - lower) / middle
    
    def atr(self, period: int = 14) -> pd.Series:
        """
        Average True Range (ATR)
        
        Formulas:
        - True Range = max[(High - Low), abs(High - Previous Close), abs(Low - Previous Close)]
        - ATR = EMA(True Range, period)
        
        Args:
            period: Number of periods for calculation (default: 14)
            
        Returns:
            pd.Series: ATR values
        """
        prev_close = self.close.shift(1)
        
        tr1 = self.high - self.low
        tr2 = abs(self.high - prev_close)
        tr3 = abs(self.low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.ewm(span=period, adjust=False).mean()
        
        return atr
    
    def keltner_channels(self, period: int = 20, multiplier: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Keltner Channels
        
        Formulas:
        - Middle Line = EMA(period)
        - Upper Channel = Middle Line + (multiplier × ATR)
        - Lower Channel = Middle Line - (multiplier × ATR)
        
        Args:
            period: Number of periods for EMA and ATR (default: 20)
            multiplier: ATR multiplier (default: 2)
            
        Returns:
            Tuple[pd.Series, pd.Series, pd.Series]: (Upper channel, Middle line, Lower channel)
        """
        middle_line = self.ema(period)
        atr_values = self.atr(period)
        
        upper_channel = middle_line + (multiplier * atr_values)
        lower_channel = middle_line - (multiplier * atr_values)
        
        return upper_channel, middle_line, lower_channel
    
    # =================================================================
    # VOLUME INDICATORS
    # =================================================================
    
    def obv(self) -> pd.Series:
        """
        On-Balance Volume (OBV)
        
        Formula: 
        - If Close > Previous Close: OBV = Previous OBV + Volume
        - If Close < Previous Close: OBV = Previous OBV - Volume
        - If Close = Previous Close: OBV = Previous OBV
        
        Returns:
            pd.Series: OBV values
        """
        price_change = self.close.diff()
        volume_direction = pd.Series(index=self.df.index, dtype=float)
        
        volume_direction[price_change > 0] = self.volume[price_change > 0]
        volume_direction[price_change < 0] = -self.volume[price_change < 0]
        volume_direction[price_change == 0] = 0
        
        obv = volume_direction.cumsum()
        
        return obv
    
    def volume_sma(self, period: int = 20) -> pd.Series:
        """
        Volume Simple Moving Average
        
        Formula: Volume SMA = Sum of Volume over n periods / n
        
        Args:
            period: Number of periods for calculation (default: 20)
            
        Returns:
            pd.Series: Volume SMA values
        """
        return self.volume.rolling(window=period).mean()
    
    def vwap(self) -> pd.Series:
        """
        Volume Weighted Average Price (VWAP)
        
        Formula: VWAP = Σ(Typical Price × Volume) / Σ(Volume)
        Where Typical Price = (High + Low + Close) / 3
        
        Returns:
            pd.Series: VWAP values
        """
        typical_price = (self.high + self.low + self.close) / 3
        vwap = (typical_price * self.volume).cumsum() / self.volume.cumsum()
        
        return vwap
    
    def accumulation_distribution(self) -> pd.Series:
        """
        Accumulation/Distribution Line (A/D Line)
        
        Formulas:
        - Money Flow Multiplier = ((Close - Low) - (High - Close)) / (High - Low)
        - Money Flow Volume = Money Flow Multiplier × Volume
        - A/D Line = Previous A/D Line + Current Money Flow Volume
        
        Returns:
            pd.Series: A/D Line values
        """
        money_flow_multiplier = ((self.close - self.low) - (self.high - self.close)) / (self.high - self.low)
        money_flow_multiplier = money_flow_multiplier.fillna(0)  # Handle division by zero
        
        money_flow_volume = money_flow_multiplier * self.volume
        ad_line = money_flow_volume.cumsum()
        
        return ad_line
    
    # =================================================================
    # PRICE-BASED INDICATORS
    # =================================================================
    
    def high_low_percentage(self) -> pd.Series:
        """
        High-Low Percentage
        
        Formula: (High - Low) / Close
        
        Returns:
            pd.Series: High-Low percentage values
        """
        return (self.high - self.low) / self.close
    
    def open_close_percentage(self) -> pd.Series:
        """
        Open-Close Percentage
        
        Formula: (Close - Open) / Open
        
        Returns:
            pd.Series: Open-Close percentage values
        """
        return (self.close - self.open) / self.open
    
    def typical_price(self) -> pd.Series:
        """
        Typical Price
        
        Formula: (High + Low + Close) / 3
        
        Returns:
            pd.Series: Typical price values
        """
        return (self.high + self.low + self.close) / 3
    
    def median_price(self) -> pd.Series:
        """
        Median Price
        
        Formula: (High + Low) / 2
        
        Returns:
            pd.Series: Median price values
        """
        return (self.high + self.low) / 2
    
    def weighted_close(self) -> pd.Series:
        """
        Weighted Close Price
        
        Formula: (High + Low + 2×Close) / 4
        
        Returns:
            pd.Series: Weighted close values
        """
        return (self.high + self.low + 2 * self.close) / 4
    
    # =================================================================
    # OSCILLATORS AND OTHER INDICATORS
    # =================================================================
    
    def cci(self, period: int = 20) -> pd.Series:
        """
        Commodity Channel Index (CCI)
        
        Formulas:
        - Typical Price = (High + Low + Close) / 3
        - SMA = Simple Moving Average of Typical Price
        - Mean Deviation = Average of absolute deviations from SMA
        - CCI = (Typical Price - SMA) / (0.015 × Mean Deviation)
        
        Args:
            period: Number of periods for calculation (default: 20)
            
        Returns:
            pd.Series: CCI values
        """
        typical_price = self.typical_price()
        sma_tp = typical_price.rolling(window=period).mean()
        mean_deviation = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - np.mean(x))), raw=True
        )
        
        cci = (typical_price - sma_tp) / (0.015 * mean_deviation)
        
        return cci
    
    def aroon(self, period: int = 14) -> Tuple[pd.Series, pd.Series]:
        """
        Aroon Indicator (Aroon Up and Aroon Down)
        
        Formulas:
        - Aroon Up = ((period - periods since highest high) / period) × 100
        - Aroon Down = ((period - periods since lowest low) / period) × 100
        
        Args:
            period: Number of periods for calculation (default: 14)
            
        Returns:
            Tuple[pd.Series, pd.Series]: (Aroon Up, Aroon Down)
        """
        def periods_since_extreme(series, func):
            rolling_window = series.rolling(window=period + 1)
            if func == 'max':
                extreme_idx = rolling_window.apply(lambda x: x.argmax(), raw=True)
            else:  # min
                extreme_idx = rolling_window.apply(lambda x: x.argmin(), raw=True)
            return period - extreme_idx
        
        aroon_up = (periods_since_extreme(self.high, 'max') / period) * 100
        aroon_down = (periods_since_extreme(self.low, 'min') / period) * 100
        
        return aroon_up, aroon_down
    
    def adx(self, period: int = 14) -> pd.Series:
        """
        Average Directional Index (ADX)
        
        Complex calculation involving True Range, Directional Movement, and smoothing.
        Measures the strength of a trend (0-100).
        
        Args:
            period: Number of periods for calculation (default: 14)
            
        Returns:
            pd.Series: ADX values
        """
        # Calculate True Range
        atr_values = self.atr(period)
        
        # Calculate Directional Movement
        dm_plus = pd.Series(index=self.df.index, dtype=float)
        dm_minus = pd.Series(index=self.df.index, dtype=float)
        
        high_diff = self.high - self.high.shift(1)
        low_diff = self.low.shift(1) - self.low
        
        dm_plus[(high_diff > low_diff) & (high_diff > 0)] = high_diff
        dm_plus[(high_diff <= low_diff) | (high_diff <= 0)] = 0
        
        dm_minus[(low_diff > high_diff) & (low_diff > 0)] = low_diff
        dm_minus[(low_diff <= high_diff) | (low_diff <= 0)] = 0
        
        # Smooth the values
        dm_plus_smooth = dm_plus.ewm(span=period).mean()
        dm_minus_smooth = dm_minus.ewm(span=period).mean()
        
        # Calculate Directional Indicators
        di_plus = 100 * (dm_plus_smooth / atr_values)
        di_minus = 100 * (dm_minus_smooth / atr_values)
        
        # Calculate DX
        dx = 100 * abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.ewm(span=period).mean()
        
        return adx
    
    # =================================================================
    # UTILITY METHODS
    # =================================================================
    
    def calculate_all_indicators(self, selected_indicators: list = None) -> pd.DataFrame:
        """
        Calculate all or selected technical indicators and add them to the DataFrame
        
        Args:
            selected_indicators: List of indicator keys to calculate (if None, calculates all)
            
        Returns:
            pd.DataFrame: DataFrame with added indicator columns
        """
        result_df = self.df.copy()
        
        # Define all available indicators
        all_indicators = {
            'sma_5': lambda: self.sma(5),
            'sma_10': lambda: self.sma(10),
            'sma_20': lambda: self.sma(20),
            'sma_50': lambda: self.sma(50),
            'ema_12': lambda: self.ema(12),
            'ema_26': lambda: self.ema(26),
            'rsi': lambda: self.rsi(14),
            'macd': lambda: self.macd()[0],
            'macd_signal': lambda: self.macd()[1],
            'macd_diff': lambda: self.macd()[2],
            'bb_upper': lambda: self.bollinger_bands()[0],
            'bb_middle': lambda: self.bollinger_bands()[1],
            'bb_lower': lambda: self.bollinger_bands()[2],
            'bb_width': lambda: self.bollinger_band_width(),
            'stoch_k': lambda: self.stochastic()[0],
            'stoch_d': lambda: self.stochastic()[1],
            'atr': lambda: self.atr(14),
            'volume_sma': lambda: self.volume_sma(20),
            'obv': lambda: self.obv(),
            'high_low_pct': lambda: self.high_low_percentage(),
            'open_close_pct': lambda: self.open_close_percentage(),
            'roc': lambda: self.roc(12),
            'williams_r': lambda: self.williams_r(14),
            'cci': lambda: self.cci(20),
            'aroon_up': lambda: self.aroon()[0],
            'aroon_down': lambda: self.aroon()[1],
            'adx': lambda: self.adx(14),
            'vwap': lambda: self.vwap(),
            'ad_line': lambda: self.accumulation_distribution(),
        }
        
        # Column name mapping
        column_mapping = {
            'sma_5': 'SMA_5',
            'sma_10': 'SMA_10',
            'sma_20': 'SMA_20',
            'sma_50': 'SMA_50',
            'ema_12': 'EMA_12',
            'ema_26': 'EMA_26',
            'rsi': 'RSI',
            'macd': 'MACD',
            'macd_signal': 'MACD_signal',
            'macd_diff': 'MACD_diff',
            'bb_upper': 'BB_upper',
            'bb_middle': 'BB_middle',
            'bb_lower': 'BB_lower',
            'bb_width': 'BB_width',
            'stoch_k': 'Stoch_k',
            'stoch_d': 'Stoch_d',
            'atr': 'ATR',
            'volume_sma': 'Volume_SMA',
            'obv': 'OBV',
            'high_low_pct': 'High_Low_Pct',
            'open_close_pct': 'Open_Close_Pct',
            'roc': 'ROC',
            'williams_r': 'Williams_R',
            'cci': 'CCI',
            'aroon_up': 'Aroon_Up',
            'aroon_down': 'Aroon_Down',
            'adx': 'ADX',
            'vwap': 'VWAP',
            'ad_line': 'AD_Line',
        }
        
        # Calculate selected indicators
        indicators_to_calculate = selected_indicators if selected_indicators else list(all_indicators.keys())
        
        for indicator_key in indicators_to_calculate:
            if indicator_key in all_indicators:
                try:
                    column_name = column_mapping.get(indicator_key, indicator_key.upper())
                    result_df[column_name] = all_indicators[indicator_key]()
                    print(f"Calculated: {column_name}")
                except Exception as e:
                    print(f"Error calculating {indicator_key}: {str(e)}")
        
        return result_df
    
    def get_indicator_description(self, indicator_key: str) -> str:
        """
        Get description of a specific indicator
        
        Args:
            indicator_key: Key of the indicator
            
        Returns:
            str: Description of the indicator
        """
        descriptions = {
            'sma_5': 'Simple Moving Average (5-period) - Average of closing prices over 5 periods',
            'sma_10': 'Simple Moving Average (10-period) - Average of closing prices over 10 periods',
            'sma_20': 'Simple Moving Average (20-period) - Average of closing prices over 20 periods',
            'sma_50': 'Simple Moving Average (50-period) - Average of closing prices over 50 periods',
            'ema_12': 'Exponential Moving Average (12-period) - Weighted average giving more weight to recent prices',
            'ema_26': 'Exponential Moving Average (26-period) - Weighted average giving more weight to recent prices',
            'rsi': 'Relative Strength Index - Momentum oscillator (0-100) measuring speed and change of price movements',
            'macd': 'MACD Line - Moving Average Convergence Divergence, difference between 12-day and 26-day EMA',
            'macd_signal': 'MACD Signal Line - 9-day EMA of MACD line',
            'macd_diff': 'MACD Histogram - Difference between MACD line and signal line',
            'bb_upper': 'Bollinger Bands Upper - Upper band (SMA + 2×standard deviation)',
            'bb_middle': 'Bollinger Bands Middle - Middle band (20-period SMA)',
            'bb_lower': 'Bollinger Bands Lower - Lower band (SMA - 2×standard deviation)',
            'bb_width': 'Bollinger Bands Width - Measure of volatility (upper-lower)/middle',
            'stoch_k': 'Stochastic %K - Fast stochastic oscillator showing current price relative to high-low range',
            'stoch_d': 'Stochastic %D - Slow stochastic, 3-period SMA of %K',
            'atr': 'Average True Range - Volatility indicator measuring the degree of price movement',
            'volume_sma': 'Volume SMA - 20-period simple moving average of trading volume',
            'obv': 'On-Balance Volume - Cumulative volume based on price direction',
            'high_low_pct': 'High-Low Percentage - (High-Low)/Close, measures intraday volatility',
            'open_close_pct': 'Open-Close Percentage - (Close-Open)/Open, measures daily price change',
            'roc': 'Rate of Change - Percentage change in price over specified period',
            'williams_r': 'Williams %R - Momentum indicator similar to stochastic oscillator',
            'cci': 'Commodity Channel Index - Oscillator identifying cyclical trends',
            'aroon_up': 'Aroon Up - Measures time since highest high within period',
            'aroon_down': 'Aroon Down - Measures time since lowest low within period',
            'adx': 'Average Directional Index - Measures strength of trend regardless of direction',
            'vwap': 'Volume Weighted Average Price - Average price weighted by volume',
            'ad_line': 'Accumulation/Distribution Line - Volume-based indicator of money flow',
        }
        
        return descriptions.get(indicator_key, 'Description not available')