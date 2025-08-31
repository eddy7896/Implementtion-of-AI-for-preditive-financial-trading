#!/usr/bin/env python3
"""
Stock Price Prediction GUI Application
Interactive interface for LSTM-based stock price forecasting with technical indicators
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import threading
import sys
import os
import io
from datetime import datetime, timedelta

# Configure matplotlib for GUI use
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# Import our stock predictor
from stock_predictor_lstm import StockPredictor

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction with LSTM & Technical Indicators")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize variables
        self.predictor = None
        self.prediction_thread = None
        self.is_predicting = False
        
        # Track matplotlib figures for proper cleanup
        self.current_prediction_fig = None
        self.current_chart_fig = None
        self.prediction_canvas = None
        self.chart_canvas = None
        
        # Console redirection
        self.console_buffer = io.StringIO()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
        # Create the main interface
        self.create_widgets()
        
        # Center the window
        self.center_window()
    
    def center_window(self):
        """Center the window on the screen"""
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')
    
    def create_widgets(self):
        """Create all GUI widgets with scrollable container"""
        
        # Create main scrollable container
        self.create_scrollable_container()
        
        # Title
        title_label = ttk.Label(self.scrollable_frame, text="Stock Price Prediction System",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # Left panel for inputs
        self.create_input_panel(self.scrollable_frame)
        
        # Right panel for results and visualization
        self.create_results_panel(self.scrollable_frame)
        
        # Bottom panel for progress and controls
        self.create_control_panel(self.scrollable_frame)
    
    def create_scrollable_container(self):
        """Create a scrollable container for the entire GUI"""
        
        # Configure root window
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        
        # Create main container frame
        container_frame = ttk.Frame(self.root)
        container_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        container_frame.columnconfigure(0, weight=1)
        container_frame.rowconfigure(0, weight=1)
        
        # Create canvas and scrollbar
        self.main_canvas = tk.Canvas(container_frame, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=self.main_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(container_frame, orient="horizontal", command=self.main_canvas.xview)
        
        # Create scrollable frame
        self.scrollable_frame = ttk.Frame(self.main_canvas, padding="10")
        
        # Configure scrollable frame
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        
        # Configure canvas
        self.main_canvas.configure(
            yscrollcommand=self.v_scrollbar.set,
            xscrollcommand=self.h_scrollbar.set
        )
        
        # Create window in canvas
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        
        # Grid the canvas and scrollbars
        self.main_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # Configure column weights for responsiveness
        self.scrollable_frame.columnconfigure(1, weight=1)
        
        # Bind mouse wheel to canvas
        self.bind_mousewheel()
        
        # Bind canvas resize event
        self.main_canvas.bind('<Configure>', self.on_canvas_configure)
    
    def bind_mousewheel(self):
        """Bind mouse wheel scrolling to the canvas"""
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        
        def _on_shift_mousewheel(event):
            self.main_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        
        # Bind mouse wheel events
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.main_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
        
        # For Linux
        self.main_canvas.bind_all("<Button-4>", lambda e: self.main_canvas.yview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Button-5>", lambda e: self.main_canvas.yview_scroll(1, "units"))
        self.main_canvas.bind_all("<Shift-Button-4>", lambda e: self.main_canvas.xview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Shift-Button-5>", lambda e: self.main_canvas.xview_scroll(1, "units"))
    
    def on_canvas_configure(self, event):
        """Handle canvas resize events"""
        # Update the scrollable frame width to match canvas width if content is smaller
        canvas_width = event.width
        frame_width = self.scrollable_frame.winfo_reqwidth()
        
        if frame_width < canvas_width:
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)
        else:
            self.main_canvas.itemconfig(self.canvas_window, width=frame_width)
    
    def create_input_panel(self, parent):
        """Create input panel with stock parameters and technical indicators"""
        
        # Input frame
        input_frame = ttk.LabelFrame(parent, text="Stock Configuration", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))
        
        # Stock Ticker
        ttk.Label(input_frame, text="Stock Ticker:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ticker_var = tk.StringVar(value="AAPL")
        ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var, width=15)
        ticker_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), pady=2)
        
        # Historical Start Date
        ttk.Label(input_frame, text="Historical Start Date:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.start_date_var = tk.StringVar(value="2020-01-01")
        start_entry = ttk.Entry(input_frame, textvariable=self.start_date_var, width=15)
        start_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(row=1, column=2, sticky=tk.W, padx=(5, 0))
        
        # Historical End Date
        ttk.Label(input_frame, text="Historical End Date:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.end_date_var = tk.StringVar(value="2025-08-01")
        end_entry = ttk.Entry(input_frame, textvariable=self.end_date_var, width=15)
        end_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(row=2, column=2, sticky=tk.W, padx=(5, 0))
        
        # Prediction Start Date
        ttk.Label(input_frame, text="Prediction Start Date:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.pred_date_var = tk.StringVar(value="2025-08-01")
        pred_entry = ttk.Entry(input_frame, textvariable=self.pred_date_var, width=15)
        pred_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(Start of prediction)", font=('Arial', 8)).grid(row=3, column=2, sticky=tk.W, padx=(5, 0))
        
        # Separator
        ttk.Separator(input_frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # ===================================================================
        # TECHNICAL INDICATORS SELECTION INTERFACE
        # ===================================================================
        """
        This section creates the comprehensive technical indicators selection
        interface. It provides users with 29+ professional-grade indicators
        organized by category for easy selection and analysis customization.
        
        The interface design includes:
        - Scrollable container for the full indicator list
        - Organized categories for better user experience
        - Smart default selections for balanced analysis
        - Individual checkbox controls for each indicator
        - Bulk selection controls (Select All, Deselect All, Reset)
        """
        
        # Main section label for technical indicators
        ttk.Label(input_frame, text="Technical Indicators:", font=('Arial', 10, 'bold')).grid(row=5, column=0, columnspan=3, sticky=tk.W, pady=(5, 2))
        
        # ===================================================================
        # SCROLLABLE CONTAINER SETUP FOR INDICATORS LIST
        # ===================================================================
        """
        Create a scrollable container to accommodate the full list of 29+
        technical indicators. This design ensures the interface remains
        usable even with many options while maintaining clean layout.
        """
        
        # Create canvas for scrollable area (fixed height to prevent GUI expansion)
        canvas = tk.Canvas(input_frame, height=200)
        
        # Create vertical scrollbar linked to canvas
        scrollbar = ttk.Scrollbar(input_frame, orient="vertical", command=canvas.yview)
        
        # Create frame that will contain all indicator checkboxes
        scrollable_frame = ttk.Frame(canvas)
        
        # Configure scrollable area to update when content size changes
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        # Create scrollable window within canvas
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Position canvas and scrollbar in the grid layout
        canvas.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar.grid(row=6, column=2, sticky=(tk.N, tk.S))
        
        # ===================================================================
        # COMPREHENSIVE TECHNICAL INDICATORS DEFINITION
        # ===================================================================
        """
        This is the master list of all available technical indicators.
        Each indicator is defined with:
        - key: Internal identifier for system integration
        - label: User-friendly display name
        - default: Whether indicator is selected by default
        
        The indicators are organized by category for better understanding:
        - Moving Averages: Trend identification and smoothing
        - Momentum: Price velocity and strength measurement
        - Trend: Direction and momentum analysis
        - Volatility: Price movement range analysis
        - Volume: Money flow and market participation
        - Price-based: Intraday pattern analysis
        - Oscillators: Cyclical pattern identification
        """
        
        # Dictionary to store checkbox variables for each indicator
        # This allows programmatic access to user selections
        self.indicator_vars = {}
        
        # Master list of all available technical indicators
        indicators = [
            # ===============================================================
            # MOVING AVERAGES - TREND IDENTIFICATION AND SMOOTHING
            # ===============================================================
            # Simple Moving Averages for different timeframes
            ("sma_5", "SMA 5-day", True),      # Very short-term trend (1 week)
            ("sma_10", "SMA 10-day", True),    # Short-term trend (2 weeks)
            ("sma_20", "SMA 20-day", True),    # Medium-term trend (1 month)
            ("sma_50", "SMA 50-day", True),    # Long-term trend (2.5 months)
            
            # Exponential Moving Averages for responsive analysis
            ("ema_12", "EMA 12-day", True),    # Fast EMA component for MACD
            ("ema_26", "EMA 26-day", True),    # Slow EMA component for MACD
            
            # ===============================================================
            # MOMENTUM INDICATORS - PRICE VELOCITY AND STRENGTH
            # ===============================================================
            ("rsi", "RSI (14-period)", True),           # Relative Strength Index (most popular momentum indicator)
            ("stoch_k", "Stochastic %K", True),         # Fast stochastic oscillator
            ("stoch_d", "Stochastic %D", True),         # Slow stochastic (smoothed %K)
            ("roc", "Rate of Change", False),           # Percentage change over period
            ("williams_r", "Williams %R", False),       # Alternative to stochastic oscillator
            
            # ===============================================================
            # TREND INDICATORS - DIRECTION AND MOMENTUM ANALYSIS
            # ===============================================================
            ("macd", "MACD", True),                     # MACD main line (most popular trend indicator)
            ("macd_signal", "MACD Signal", True),       # MACD signal line for crossover signals
            ("macd_diff", "MACD Histogram", True),      # MACD histogram for momentum analysis
            ("adx", "Average Directional Index", False), # Trend strength measurement
            ("aroon_up", "Aroon Up", False),            # Uptrend strength indicator
            ("aroon_down", "Aroon Down", False),        # Downtrend strength indicator
            
            # ===============================================================
            # VOLATILITY INDICATORS - PRICE MOVEMENT RANGE ANALYSIS
            # ===============================================================
            ("bb_upper", "Bollinger Bands Upper", True),    # Upper volatility band
            ("bb_middle", "Bollinger Bands Middle", True),  # Middle band (20-day SMA)
            ("bb_lower", "Bollinger Bands Lower", True),    # Lower volatility band
            ("bb_width", "Bollinger Bands Width", True),    # Volatility measurement
            ("atr", "Average True Range", True),            # True volatility measurement
            
            # ===============================================================
            # VOLUME INDICATORS - MONEY FLOW AND MARKET PARTICIPATION
            # ===============================================================
            ("volume_sma", "Volume SMA", True),                         # Volume moving average
            ("obv", "On-Balance Volume", True),                         # Cumulative volume flow
            ("vwap", "Volume Weighted Average Price", False),           # Price weighted by volume
            ("ad_line", "Accumulation/Distribution Line", False),       # Money flow indicator
            
            # ===============================================================
            # PRICE-BASED INDICATORS - INTRADAY PATTERN ANALYSIS
            # ===============================================================
            ("high_low_pct", "High-Low Percentage", True),      # Daily trading range
            ("open_close_pct", "Open-Close Percentage", True),  # Daily price change
            
            # ===============================================================
            # OSCILLATORS - CYCLICAL PATTERN IDENTIFICATION
            # ===============================================================
            ("cci", "Commodity Channel Index", False),     # Cyclical trend identifier
        ]
        
        # ===================================================================
        # CREATE INDIVIDUAL CHECKBOX CONTROLS
        # ===================================================================
        """
        Generate checkbox controls for each technical indicator.
        Each checkbox is linked to a BooleanVar that tracks its selection state.
        These states are later read to determine which indicators to calculate.
        """
        
        row = 0
        for key, label, default in indicators:
            # Create Boolean variable to track checkbox state
            var = tk.BooleanVar(value=default)
            
            # Store variable reference for later access
            self.indicator_vars[key] = var
            
            # Create checkbox with user-friendly label
            cb = ttk.Checkbutton(scrollable_frame, text=label, variable=var)
            
            # Position checkbox in scrollable grid
            cb.grid(row=row, column=0, sticky=tk.W, pady=1)
            row += 1
        
        # Select/Deselect All buttons
        button_frame = ttk.Frame(input_frame)
        button_frame.grid(row=7, column=0, columnspan=3, pady=5)
        
        ttk.Button(button_frame, text="Select All", command=self.select_all_indicators).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Deselect All", command=self.deselect_all_indicators).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="Reset Defaults", command=self.reset_default_indicators).pack(side=tk.LEFT, padx=2)
        
        # Model Parameters
        ttk.Separator(input_frame, orient='horizontal').grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        ttk.Label(input_frame, text="Model Parameters:", font=('Arial', 10, 'bold')).grid(row=9, column=0, columnspan=3, sticky=tk.W, pady=(5, 2))
        
        # Lookback Days
        ttk.Label(input_frame, text="Lookback Days:").grid(row=10, column=0, sticky=tk.W, pady=2)
        self.lookback_var = tk.StringVar(value="60")
        ttk.Entry(input_frame, textvariable=self.lookback_var, width=8).grid(row=10, column=1, sticky=tk.W, pady=2)
        
        # Training Epochs
        ttk.Label(input_frame, text="Training Epochs:").grid(row=11, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="50")
        ttk.Entry(input_frame, textvariable=self.epochs_var, width=8).grid(row=11, column=1, sticky=tk.W, pady=2)
        
        # Configure column weights
        input_frame.columnconfigure(1, weight=1)
    
    def create_results_panel(self, parent):
        """Create results panel with tabs for different views"""
        
        # Results frame
        results_frame = ttk.LabelFrame(parent, text="Results & Visualization", padding="5")
        results_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Prediction Results Tab
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Predictions")
        
        # Results text area
        self.results_text = ScrolledText(self.results_tab, wrap=tk.WORD, width=50, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.results_tab.rowconfigure(0, weight=1)
        self.results_tab.columnconfigure(0, weight=1)
        
        # Stock Prediction Chart Tab
        self.prediction_chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_chart_tab, text="Prediction Chart")
        
        # Prediction chart frame
        self.prediction_chart_frame = ttk.Frame(self.prediction_chart_tab)
        self.prediction_chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.prediction_chart_tab.rowconfigure(0, weight=1)
        self.prediction_chart_tab.columnconfigure(0, weight=1)
        
        # Technical Analysis Charts Tab
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="Technical Charts")
        
        # Chart frame
        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.chart_tab.rowconfigure(0, weight=1)
        self.chart_tab.columnconfigure(0, weight=1)
        
        # Technical Analysis Tab
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Technical Analysis")
        
        self.analysis_text = ScrolledText(self.analysis_tab, wrap=tk.WORD, width=50, height=20)
        self.analysis_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.analysis_tab.rowconfigure(0, weight=1)
        self.analysis_tab.columnconfigure(0, weight=1)
        
        # Console Output Tab
        self.console_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.console_tab, text="Console Logs")
        
        self.console_text = ScrolledText(self.console_tab, wrap=tk.WORD, width=50, height=20,
                                        bg='black', fg='lightgreen', insertbackground='lightgreen',
                                        font=('Consolas', 9))
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.console_tab.rowconfigure(0, weight=1)
        self.console_tab.columnconfigure(0, weight=1)
        
        # Add console control buttons
        console_controls = ttk.Frame(self.console_tab)
        console_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        
        ttk.Button(console_controls, text="Clear Console",
                  command=self.clear_console).pack(side=tk.LEFT, padx=2)
        ttk.Button(console_controls, text="Save Console Log",
                  command=self.save_console_log).pack(side=tk.LEFT, padx=2)
        
        # Set up console redirection
        self.setup_console_redirection()
    
    def create_control_panel(self, parent):
        """Create control panel with progress bar and action buttons"""
        
        # Control frame
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        control_frame.columnconfigure(1, weight=1)
        
        # Progress bar
        ttk.Label(control_frame, text="Progress:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
        
        # Progress label
        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        
        # Action buttons
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        
        self.predict_btn = ttk.Button(button_frame, text="Start Prediction", 
                                     command=self.start_prediction, style="Accent.TButton")
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", 
                                  command=self.stop_prediction, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Results",
                  command=self.clear_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear All",
                  command=self.clear_all, style="Accent.TButton").pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Save Results",
                  command=self.save_results).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Load Example",
                  command=self.load_example).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Help",
                  command=self.show_help).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Quit",
                  command=self.quit_application).pack(side=tk.LEFT, padx=5)
    
    def select_all_indicators(self):
        """Select all technical indicators"""
        for var in self.indicator_vars.values():
            var.set(True)
    
    def deselect_all_indicators(self):
        """Deselect all technical indicators"""
        for var in self.indicator_vars.values():
            var.set(False)
    
    def reset_default_indicators(self):
        """
        =======================================================================
        RESET TO SMART DEFAULT INDICATOR SELECTION
        =======================================================================
        
        This method restores a carefully curated set of default technical
        indicators that provide balanced coverage across all analysis categories.
        
        The default selection includes:
        - Core trend indicators (moving averages, MACD)
        - Essential momentum indicators (RSI, Stochastic)
        - Key volatility measures (Bollinger Bands, ATR)
        - Important volume indicators (OBV, Volume SMA)
        - Basic price pattern indicators
        
        This balanced approach provides:
        - Comprehensive market analysis without overwhelming complexity
        - Proven indicator combinations used by professional traders
        - Good performance balance (not too many indicators to slow calculation)
        - Coverage of all major technical analysis categories
        """
        
        # Carefully selected core default indicators for balanced analysis
        # These 20 indicators cover all major categories while maintaining performance
        defaults = [
            # ===============================================================
            # ESSENTIAL MOVING AVERAGES - TREND FOUNDATION
            # ===============================================================
            "sma_5", "sma_10", "sma_20", "sma_50",  # Multi-timeframe trend analysis
            "ema_12", "ema_26",                      # Components for MACD calculation
            
            # ===============================================================
            # KEY MOMENTUM INDICATORS - MARKET STRENGTH
            # ===============================================================
            "rsi",                                   # Most popular momentum oscillator
            "stoch_k", "stoch_d",                   # Stochastic oscillator pair
            
            # ===============================================================
            # CORE TREND ANALYSIS - DIRECTION AND MOMENTUM
            # ===============================================================
            "macd", "macd_signal", "macd_diff",     # Complete MACD system
            
            # ===============================================================
            # ESSENTIAL VOLATILITY ANALYSIS - PRICE MOVEMENT RANGE
            # ===============================================================
            "bb_upper", "bb_middle", "bb_lower",    # Complete Bollinger Bands system
            "bb_width",                             # Volatility measurement
            "atr",                                  # True range volatility
            
            # ===============================================================
            # IMPORTANT VOLUME ANALYSIS - MONEY FLOW
            # ===============================================================
            "volume_sma",                           # Volume trend analysis
            "obv",                                  # On-balance volume flow
            
            # ===============================================================
            # BASIC PRICE PATTERNS - INTRADAY ANALYSIS
            # ===============================================================
            "high_low_pct",                         # Daily trading range
            "open_close_pct"                        # Daily price change
        ]
        
        # Apply the default selection to all indicator checkboxes
        # Set each indicator's checkbox based on whether it's in the defaults list
        for key, var in self.indicator_vars.items():
            var.set(key in defaults)
    
    def validate_inputs(self):
        """Validate user inputs"""
        
        # Check ticker
        ticker = self.ticker_var.get().strip().upper()
        if not ticker:
            raise ValueError("Please enter a stock ticker symbol")
        
        # Check dates
        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            pred_date = datetime.strptime(self.pred_date_var.get(), "%Y-%m-%d")
        except ValueError:
            raise ValueError("Please enter dates in YYYY-MM-DD format")
        
        if start_date >= end_date:
            raise ValueError("Historical start date must be before end date")
        
        if pred_date < end_date:
            raise ValueError("Prediction start date should be on or after historical end date")
        
        # Check model parameters
        try:
            lookback = int(self.lookback_var.get())
            epochs = int(self.epochs_var.get())
            if lookback < 1 or lookback > 365:
                raise ValueError("Lookback days must be between 1 and 365")
            if epochs < 1 or epochs > 1000:
                raise ValueError("Training epochs must be between 1 and 1000")
        except ValueError as e:
            if "invalid literal" in str(e):
                raise ValueError("Lookback days and epochs must be valid integers")
            else:
                raise e
        
        # Check at least one indicator is selected
        selected_indicators = [key for key, var in self.indicator_vars.items() if var.get()]
        if not selected_indicators:
            raise ValueError("Please select at least one technical indicator")
        
        return ticker, start_date, end_date, pred_date, lookback, epochs, selected_indicators
    
    def update_progress(self, value, text):
        """Update progress bar and label (thread-safe)"""
        def _update_progress():
            try:
                self.progress_var.set(value)
                self.progress_label.config(text=text)
                self.root.update_idletasks()
            except Exception as e:
                print(f"Error updating progress: {str(e)}")
        
        # Schedule update on main thread if called from worker thread
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, _update_progress)
        else:
            _update_progress()
    
    def log_message(self, message, tab="results"):
        """Add message to appropriate text widget (thread-safe)"""
        def _update_log():
            try:
                timestamp = datetime.now().strftime("%H:%M:%S")
                formatted_message = f"[{timestamp}] {message}\n"
                
                if tab == "results":
                    self.results_text.insert(tk.END, formatted_message)
                    self.results_text.see(tk.END)
                elif tab == "analysis":
                    self.analysis_text.insert(tk.END, formatted_message)
                    self.analysis_text.see(tk.END)
                
                self.root.update_idletasks()
            except Exception as e:
                print(f"Error updating log: {str(e)}")
        
        # Schedule update on main thread if called from worker thread
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, _update_log)
        else:
            _update_log()
    
    def start_prediction(self):
        """Start the prediction process in a separate thread"""
        
        try:
            # Validate inputs
            ticker, start_date, end_date, pred_date, lookback, epochs, selected_indicators = self.validate_inputs()
            
            # Update UI state
            self.is_predicting = True
            self.predict_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            
            # Clear previous results
            self.clear_results()
            
            # Start prediction in separate thread
            self.prediction_thread = threading.Thread(
                target=self.run_prediction,
                args=(ticker, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"), 
                      pred_date.strftime("%Y-%m-%d"), lookback, epochs, selected_indicators)
            )
            self.prediction_thread.daemon = True
            self.prediction_thread.start()
            
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))
            
    def stop_prediction(self):
        """Stop the prediction process"""
        self.is_predicting = False
        self.update_progress(0, "Stopping...")
        
        # Reset UI state
        self.predict_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        
        self.log_message("Prediction stopped by user")
        self.update_progress(0, "Ready")
    
    def run_prediction(self, ticker, start_date, end_date, pred_date, lookback, epochs, selected_indicators):
        """Run the prediction process"""
        
        try:
            self.log_message(f"Starting prediction for {ticker}")
            self.log_message(f"Historical data: {start_date} to {end_date}")
            self.log_message(f"Prediction from: {pred_date}")
            
            # Step 1: Initialize predictor
            self.update_progress(5, "Initializing predictor...")
            self.log_message(f"Selected indicators: {', '.join(selected_indicators)}")
            self.predictor = StockPredictor(ticker, lookback_days=lookback, prediction_days=7, selected_indicators=selected_indicators)
            
            if not self.is_predicting:
                return
            
            # Step 2: Fetch data
            self.update_progress(10, "Fetching stock data...")
            self.log_message("Fetching historical stock data...")
            
            if not self.predictor.fetch_data(start_date, end_date):
                self.log_message("ERROR: Failed to fetch stock data")
                return
            
            data_points = len(self.predictor.df)
            self.log_message(f"Successfully fetched {data_points} trading days")
            
            if not self.is_predicting:
                return
            
            # Step 3: Calculate technical indicators
            self.update_progress(25, "Calculating technical indicators...")
            self.log_message("Calculating technical indicators...")
            
            if not self.predictor.calculate_technical_indicators():
                self.log_message("ERROR: Failed to calculate technical indicators")
                return
            
            indicators_calculated = len(self.predictor.features)
            self.log_message(f"Calculated {indicators_calculated} technical indicators")
            
            # Display current technical analysis (thread-safe)
            self.root.after(0, self.display_technical_analysis)
            
            if not self.is_predicting:
                return
            
            # Step 4: Prepare LSTM data
            self.update_progress(40, "Preparing training data...")
            self.log_message("Preparing LSTM training data...")
            
            if not self.predictor.prepare_lstm_data():
                self.log_message("ERROR: Failed to prepare training data")
                return
            
            self.log_message(f"Training samples: {len(self.predictor.X_train)}")
            self.log_message(f"Testing samples: {len(self.predictor.X_test)}")
            
            if not self.is_predicting:
                return
            
            # Step 5: Build and train model
            self.update_progress(50, "Building LSTM model...")
            self.log_message("Building LSTM model...")
            
            try:
                if self.predictor.build_lstm_model():
                    self.log_message("LSTM model built successfully")
                    
                    # Train model
                    self.update_progress(60, f"Training model ({epochs} epochs)...")
                    self.log_message(f"Training LSTM model for {epochs} epochs...")
                    
                    if self.predictor.train_model(epochs=epochs, batch_size=32):
                        self.log_message("Model training completed successfully")
                        
                        # Evaluate model
                        self.update_progress(80, "Evaluating model...")
                        self.predictor.evaluate_model()
                    else:
                        self.log_message("WARNING: LSTM training failed, using fallback method")
                else:
                    self.log_message("WARNING: LSTM not available, using fallback method")
            except Exception as e:
                self.log_message(f"WARNING: LSTM error: {str(e)}, using fallback method")
            
            if not self.is_predicting:
                return
            
            # Step 6: Generate predictions
            self.update_progress(90, "Generating predictions...")
            self.log_message(f"Predicting next 7 days from {pred_date}...")
            
            prediction_dates, predicted_prices = self.predictor.predict_next_week(pred_date)
            
            if prediction_dates and predicted_prices is not None:
                self.log_message("Weekly predictions generated successfully!")
                self.root.after(0, lambda: self.display_predictions(prediction_dates, predicted_prices, pred_date))
                
                # Create visualizations
                self.update_progress(95, "Creating visualizations...")
                try:
                    self.root.after(0, lambda: self.create_prediction_chart(prediction_dates, predicted_prices))
                    self.root.after(0, lambda: self.create_charts(prediction_dates, predicted_prices))
                    
                    self.update_progress(100, "Complete!")
                    self.log_message("Prediction process completed successfully!")
                except Exception as viz_error:
                    self.log_message(f"Warning: Visualization error: {str(viz_error)}")
                    self.update_progress(100, "Complete (charts failed)")
                
            else:
                self.log_message("ERROR: Failed to generate predictions")
                
        except Exception as e:
            self.log_message(f"ERROR: {str(e)}")
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{str(e)}")
        
        finally:
            # Reset UI state in thread-safe manner
            self.is_predicting = False
            self.root.after(0, self._reset_ui_state)
    
    def display_technical_analysis(self):
        """Display current technical analysis"""
        
        if self.predictor is None or self.predictor.df is None:
            return
        
        try:
            latest = self.predictor.df.iloc[-1]
            current_price = latest['Close']
            date_str = self.predictor.df.index[-1].strftime('%Y-%m-%d')
            
            analysis_text = f"\nTECHNICAL ANALYSIS SUMMARY for {self.predictor.ticker}\n"
            analysis_text += "=" * 50 + "\n"
            analysis_text += f"Date: {date_str}\n"
            analysis_text += f"Current Price: ${current_price:.2f}\n\n"
            
            # RSI Analysis
            if 'RSI' in self.predictor.df.columns:
                rsi = latest['RSI']
                if rsi > 70:
                    rsi_signal = "Overbought"
                elif rsi < 30:
                    rsi_signal = "Oversold"
                else:
                    rsi_signal = "Neutral"
                analysis_text += f"RSI: {rsi:.2f} ({rsi_signal})\n"
            
            # Moving Average Analysis
            if 'SMA_20' in self.predictor.df.columns and 'SMA_50' in self.predictor.df.columns:
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
                    
                analysis_text += f"MA Trend: {ma_trend}\n"
                analysis_text += f"SMA20: ${sma20:.2f}, SMA50: ${sma50:.2f}\n"
            
            # MACD Analysis
            if 'MACD' in self.predictor.df.columns:
                macd = latest['MACD']
                macd_signal = latest['MACD_signal']
                macd_trend = "Bullish" if macd > macd_signal else "Bearish"
                analysis_text += f"MACD: {macd_trend} (MACD: {macd:.4f}, Signal: {macd_signal:.4f})\n"
            
            # Bollinger Bands
            if 'BB_upper' in self.predictor.df.columns:
                bb_upper = latest['BB_upper']
                bb_lower = latest['BB_lower']
                if current_price > bb_upper:
                    bb_signal = "Above upper band (potential sell signal)"
                elif current_price < bb_lower:
                    bb_signal = "Below lower band (potential buy signal)"
                else:
                    bb_signal = "Within bands (neutral)"
                analysis_text += f"Bollinger Bands: {bb_signal}\n"
            
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, analysis_text)
            
        except Exception as e:
            self.log_message(f"Error in technical analysis: {str(e)}", "analysis")
    
    def display_predictions(self, prediction_dates, predicted_prices, pred_date):
        """Display prediction results"""
        
        if self.predictor is None:
            return
        
        last_known_price = self.predictor.df['Close'].iloc[-1]
        
        results_text = f"\nWEEKLY STOCK PRICE PREDICTIONS\n"
        results_text += "=" * 60 + "\n"
        results_text += f"Stock: {self.predictor.ticker}\n"
        results_text += f"Historical data ends: {pred_date}\n"
        results_text += f"Last known price: ${last_known_price:.2f}\n\n"
        
        results_text += "7-Day Forecast:\n"
        results_text += "-" * 60 + "\n"
        results_text += f"{'Day':<4} {'Date':<12} {'Day Name':<10} {'Price':<12} {'Change':<8} {'Change%':<8}\n"
        results_text += "-" * 60 + "\n"
        
        for i, (date, price) in enumerate(zip(prediction_dates, predicted_prices)):
            day_name = date.strftime('%A')[:3]
            date_str = date.strftime('%Y-%m-%d')
            change = price - last_known_price
            change_pct = (change / last_known_price) * 100
            
            results_text += f"{i+1:<4} {date_str:<12} {day_name:<10} ${price:<11.2f} {change:<+7.2f} {change_pct:<+7.2f}%\n"
        
        results_text += "-" * 60 + "\n"
        
        # Summary statistics
        week_high = max(predicted_prices)
        week_low = min(predicted_prices)
        week_end = predicted_prices[-1]
        
        results_text += f"\nWEEK SUMMARY:\n"
        results_text += f"Expected High:  ${week_high:.2f}\n"
        results_text += f"Expected Low:   ${week_low:.2f}\n"
        results_text += f"Week End Price: ${week_end:.2f}\n"
        
        weekly_return = ((week_end - last_known_price) / last_known_price) * 100
        results_text += f"Expected Weekly Return: {weekly_return:+.2f}%\n"
        
        # Trend direction
        if week_end > last_known_price:
            trend_desc = "BULLISH (Upward)"
        elif week_end < last_known_price:
            trend_desc = "BEARISH (Downward)"
        else:
            trend_desc = "NEUTRAL (Sideways)"
        
        results_text += f"Weekly Trend: {trend_desc}\n"
        
        self.log_message(results_text)
    
    def create_prediction_chart(self, prediction_dates, predicted_prices):
        """Create focused prediction chart with candlestick pattern showing only current price and predictions"""
        
        if self.predictor is None:
            return
        
        try:
            # Properly cleanup previous prediction chart
            self.cleanup_prediction_chart()
            
            # Check if we still have valid data after cleanup
            if not self.is_predicting or self.predictor is None:
                return
            
            # Create matplotlib figure for prediction - focused view
            fig, ax = plt.subplots(1, 1, figsize=(14, 8))
            fig.suptitle(f'{self.predictor.ticker} - 7-Day Price Prediction', fontsize=16, fontweight='bold')
            
            if prediction_dates and predicted_prices is not None:
                # Get current price (last known price)
                current_price = self.predictor.df['Close'].iloc[-1]
                current_date = self.predictor.df.index[-1]
                
                # Create combined data: current price + predictions
                combined_dates = [current_date] + list(prediction_dates)
                combined_prices = [current_price] + list(predicted_prices)
                
                # Create candlestick-like data for predictions
                # For predictions, we'll create synthetic OHLC data
                ohlc_data = []
                
                for i, (date, price) in enumerate(zip(combined_dates, combined_prices)):
                    if i == 0:  # Current price
                        # Use actual OHLC from the last trading day
                        last_day = self.predictor.df.iloc[-1]
                        ohlc_data.append({
                            'Date': date,
                            'Open': last_day['Open'],
                            'High': last_day['High'],
                            'Low': last_day['Low'],
                            'Close': last_day['Close'],
                            'Type': 'Current'
                        })
                    else:  # Predictions
                        # Create synthetic OHLC around predicted price
                        volatility = abs(price - combined_prices[i-1]) * 0.3
                        open_price = combined_prices[i-1] + (price - combined_prices[i-1]) * 0.2
                        high_price = max(open_price, price) + volatility
                        low_price = min(open_price, price) - volatility
                        
                        ohlc_data.append({
                            'Date': date,
                            'Open': open_price,
                            'High': high_price,
                            'Low': low_price,
                            'Close': price,
                            'Type': 'Prediction'
                        })
                
                # Plot candlestick-style chart
                bar_width = 0.6
                for i, candle in enumerate(ohlc_data):
                    date = candle['Date']
                    open_price = candle['Open']
                    high_price = candle['High']
                    low_price = candle['Low']
                    close_price = candle['Close']
                    candle_type = candle['Type']
                    
                    # Determine color (green if close > open, red if close < open)
                    if candle_type == 'Current':
                        color = 'blue'
                        alpha = 1.0
                        edge_color = 'darkblue'
                    else:
                        color = 'green' if close_price >= open_price else 'red'
                        alpha = 0.8
                        edge_color = 'darkgreen' if close_price >= open_price else 'darkred'
                    
                    # Draw the candle body
                    body_height = abs(close_price - open_price)
                    body_bottom = min(open_price, close_price)
                    
                    # Candle body (rectangle)
                    ax.bar(date, body_height, bottom=body_bottom, width=bar_width,
                          color=color, alpha=alpha, edgecolor=edge_color, linewidth=1)
                    
                    # High-low line (wick)
                    ax.plot([date, date], [low_price, high_price], color=edge_color, linewidth=2)
                    
                    # Add price labels on predictions
                    if candle_type == 'Prediction':
                        ax.annotate(f'${close_price:.2f}',
                                  (date, high_price),
                                  xytext=(0, 15),
                                  textcoords='offset points',
                                  fontsize=10, fontweight='bold',
                                  ha='center',
                                  bbox=dict(boxstyle='round,pad=0.3',
                                           facecolor=color, alpha=0.7, edgecolor=edge_color))
                    
                    # Day labels
                    if candle_type == 'Current':
                        day_label = 'Current'
                    else:
                        day_num = i  # 1, 2, 3... for prediction days
                        day_label = f'Day {day_num}'
                    
                    ax.annotate(day_label,
                              (date, low_price),
                              xytext=(0, -25),
                              textcoords='offset points',
                              fontsize=9, ha='center',
                              bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgray', alpha=0.7))
                
                # Add trend line connecting close prices
                ax.plot(combined_dates, combined_prices, 'k--', alpha=0.5, linewidth=1, label='Price Trend')
                
                # Highlight prediction period with background
                if len(prediction_dates) > 1:
                    ax.axvspan(prediction_dates[0], prediction_dates[-1],
                             alpha=0.05, color='yellow', label='7-Day Prediction Period')
                
                # Calculate price range for better visualization
                all_highs = [candle['High'] for candle in ohlc_data]
                all_lows = [candle['Low'] for candle in ohlc_data]
                price_range = max(all_highs) - min(all_lows)
                margin = price_range * 0.1
                
                ax.set_ylim(min(all_lows) - margin, max(all_highs) + margin)
                
                # Calculate and show prediction statistics
                week_return = ((predicted_prices[-1] - current_price) / current_price) * 100
                week_high = max(predicted_prices)
                week_low = min(predicted_prices)
                
                # Add statistics box
                stats_text = f'Current: ${current_price:.2f}\n'
                stats_text += f'Week End: ${predicted_prices[-1]:.2f}\n'
                stats_text += f'Expected Return: {week_return:+.1f}%\n'
                stats_text += f'Week Range: ${week_low:.2f} - ${week_high:.2f}'
                
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                       verticalalignment='top', fontsize=10,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
            
            # Styling
            ax.set_title('7-Day Price Prediction (Candlestick View)', fontsize=14, pad=20)
            ax.set_xlabel('Trading Days', fontsize=12)
            ax.set_ylabel('Price ($)', fontsize=12)
            ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
            
            # Format x-axis
            ax.tick_params(axis='x', rotation=45, labelsize=10)
            ax.tick_params(axis='y', labelsize=10)
            
            # Add legend
            ax.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)
            
            plt.tight_layout()
            
            # Embed chart in tkinter
            self.prediction_canvas = FigureCanvasTkAgg(fig, self.prediction_chart_frame)
            self.prediction_canvas.draw()
            self.prediction_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_prediction_fig = fig
            
            # Add toolbar for chart interaction
            toolbar_frame = ttk.Frame(self.prediction_chart_frame)
            toolbar_frame.pack(fill=tk.X, pady=5)
            
            ttk.Button(toolbar_frame, text="Save Prediction Chart",
                      command=lambda: self.save_prediction_chart()).pack(side=tk.LEFT, padx=2)
            ttk.Button(toolbar_frame, text="Zoom to Fit",
                      command=lambda: self.zoom_prediction_chart()).pack(side=tk.LEFT, padx=2)
            ttk.Button(toolbar_frame, text="Focus Predictions",
                      command=lambda: self.focus_predictions_safe(prediction_dates, predicted_prices)).pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            self.log_message(f"Error creating prediction chart: {str(e)}")

    def zoom_prediction_chart(self):
        """Zoom prediction chart to fit"""
        try:
            if self.current_prediction_fig and self.prediction_canvas:
                for ax in self.current_prediction_fig.get_axes():
                    ax.autoscale()
                self.prediction_canvas.draw()
        except Exception as e:
            self.log_message(f"Error zooming prediction chart: {str(e)}")
    
    def focus_predictions_safe(self, prediction_dates, predicted_prices):
        """Focus chart view on just the prediction period (thread-safe)"""
        try:
            if (self.current_prediction_fig and self.prediction_canvas and
                prediction_dates and predicted_prices):
                
                ax = self.current_prediction_fig.get_axes()[0]
                
                # Set x-axis to show only prediction period with small margin
                margin = (prediction_dates[-1] - prediction_dates[0]) * 0.1
                ax.set_xlim(prediction_dates[0] - margin, prediction_dates[-1] + margin)
                
                # Set y-axis to show prediction price range with margin
                price_range = max(predicted_prices) - min(predicted_prices)
                margin_y = max(price_range * 0.2, 5)  # At least $5 margin
                ax.set_ylim(min(predicted_prices) - margin_y, max(predicted_prices) + margin_y)
                
                self.prediction_canvas.draw()
        except Exception as e:
            self.log_message(f"Error focusing predictions: {str(e)}")

    def create_charts(self, prediction_dates, predicted_prices):
        """Create and display charts"""
        
        if self.predictor is None:
            return
        
        try:
            # Properly cleanup previous charts
            self.cleanup_main_charts()
            
            # Check if we still have valid data after cleanup
            if not self.is_predicting or self.predictor is None:
                return
            
            # Create matplotlib figure
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f'{self.predictor.ticker} Stock Analysis & Predictions', fontsize=14, fontweight='bold')
            
            # Get recent data for visualization
            recent_data = self.predictor.df.tail(60)
            
            # Chart 1: Price with predictions
            ax1.plot(recent_data.index, recent_data['Close'], label='Historical Price', linewidth=2, color='blue')
            
            if 'SMA_20' in recent_data.columns:
                ax1.plot(recent_data.index, recent_data['SMA_20'], label='SMA 20', alpha=0.7, color='orange')
            
            if 'BB_upper' in recent_data.columns:
                ax1.fill_between(recent_data.index, recent_data['BB_lower'], recent_data['BB_upper'],
                               alpha=0.2, color='gray', label='Bollinger Bands')
            
            # Plot predictions
            if prediction_dates and predicted_prices is not None:
                ax1.plot(prediction_dates, predicted_prices, 'ro-', label='7-Day Predictions',
                        linewidth=2, markersize=6)
                
                # Add prediction values as annotations
                for date, price in zip(prediction_dates, predicted_prices):
                    ax1.annotate(f'${price:.1f}', (date, price), xytext=(5, 5),
                               textcoords='offset points', fontsize=8,
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7))
            
            ax1.set_title('Price Chart with Predictions')
            ax1.set_ylabel('Price ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.tick_params(axis='x', rotation=45)
            
            # Chart 2: RSI
            if 'RSI' in recent_data.columns:
                ax2.plot(recent_data.index, recent_data['RSI'], color='purple', linewidth=1)
                ax2.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought (70)')
                ax2.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold (30)')
                ax2.fill_between(recent_data.index, 30, 70, alpha=0.1, color='gray')
                ax2.set_title('RSI Indicator')
                ax2.set_ylabel('RSI')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                ax2.set_ylim(0, 100)
                ax2.tick_params(axis='x', rotation=45)
            
            # Chart 3: MACD
            if 'MACD' in recent_data.columns:
                ax3.plot(recent_data.index, recent_data['MACD'], label='MACD', linewidth=1, color='blue')
                ax3.plot(recent_data.index, recent_data['MACD_signal'], label='Signal', linewidth=1, color='red')
                ax3.bar(recent_data.index, recent_data['MACD_diff'], alpha=0.3, label='Histogram', color='green')
                ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                ax3.set_title('MACD Indicator')
                ax3.set_ylabel('MACD')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                ax3.tick_params(axis='x', rotation=45)
            
            # Chart 4: Volume
            ax4.bar(recent_data.index, recent_data['Volume'], alpha=0.6, width=1, color='green')
            ax4.set_title('Trading Volume')
            ax4.set_ylabel('Volume')
            ax4.grid(True, alpha=0.3)
            ax4.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Embed chart in tkinter
            self.chart_canvas = FigureCanvasTkAgg(fig, self.chart_frame)
            self.chart_canvas.draw()
            self.chart_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            self.current_chart_fig = fig
            
            # Add toolbar for chart interaction
            toolbar = tk.Frame(self.chart_frame)
            toolbar.pack(fill=tk.X)
            
            ttk.Button(toolbar, text="Save Chart",
                      command=lambda: self.save_chart()).pack(side=tk.LEFT, padx=2)
            
        except Exception as e:
            self.log_message(f"Error creating technical charts: {str(e)}")
            # Close figure if it was created but failed to embed
            try:
                if 'fig' in locals():
                    plt.close(fig)
            except:
                pass
    
    def save_chart(self):
        """Save the current chart"""
        try:
            if self.current_chart_fig is None:
                messagebox.showwarning("No Chart", "No chart available to save")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Chart"
            )
            if filename:
                self.current_chart_fig.savefig(filename, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Chart saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save chart:\n{str(e)}")
    
    def save_prediction_chart(self):
        """Save the prediction chart"""
        try:
            if self.current_prediction_fig is None:
                messagebox.showwarning("No Chart", "No prediction chart available to save")
                return
                
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")],
                title="Save Prediction Chart",
                initialname=f"{self.predictor.ticker}_prediction_chart.png" if self.predictor else "prediction_chart.png"
            )
            if filename:
                self.current_prediction_fig.savefig(filename, dpi=150, bbox_inches='tight')
                messagebox.showinfo("Success", f"Prediction chart saved as {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save prediction chart:\n{str(e)}")

    def clear_results(self):
        """Clear all results"""
        self.results_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        
        # Properly cleanup matplotlib figures and canvases
        self.cleanup_charts()
    
    def clear_all(self):
        """Clear all data and reset the program to first run state"""
        
        # Confirm action with user
        if messagebox.askyesno("Clear All",
                              "This will reset all inputs, results, and charts to default values.\n\n"
                              "Are you sure you want to continue?"):
            
            # Stop any running prediction first
            if self.is_predicting:
                self.stop_prediction()
            
            # Reset all input fields to their default values
            self.ticker_var.set("AAPL")
            self.start_date_var.set("2020-01-01")
            self.end_date_var.set("2025-08-01")
            self.pred_date_var.set("2025-08-01")
            self.lookback_var.set("60")
            self.epochs_var.set("50")
            
            # Reset technical indicators to defaults
            self.reset_default_indicators()
            
            # Clear all results and analysis
            self.results_text.delete(1.0, tk.END)
            self.analysis_text.delete(1.0, tk.END)
            
            # Clear console
            self.clear_console()
            
            # Clean up all charts and figures
            self.cleanup_charts()
            
            # Reset predictor instance
            self.predictor = None
            
            # Reset progress bar and status
            self.progress_var.set(0)
            self.progress_label.config(text="Ready")
            
            # Reset button states
            self.predict_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
            # Reset prediction flags
            self.is_predicting = False
            self.prediction_thread = None
            
            # Switch to the first tab (Predictions)
            self.notebook.select(0)
            
            # Log the reset action
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.console_text.insert(tk.END, f"[{timestamp}] Program reset to initial state\n")
            self.console_text.insert(tk.END, f"[{timestamp}] All inputs, results, and charts cleared\n\n")
            
            # Show confirmation message
            messagebox.showinfo("Reset Complete",
                              "All data has been cleared and the program has been reset to its initial state.\n\n"
                              "Default values have been restored for all inputs.")
    
    def setup_console_redirection(self):
        """Set up console output redirection to capture print statements"""
        class ConsoleRedirector:
            def __init__(self, text_widget, original_stream, tag=None):
                self.text_widget = text_widget
                self.original_stream = original_stream
                self.tag = tag
            
            def write(self, string):
                # Write to original stream (for debugging purposes)
                if self.original_stream:
                    try:
                        self.original_stream.write(string)
                        self.original_stream.flush()
                    except:
                        pass
                
                # Write to GUI console
                if string.strip():  # Only log non-empty strings
                    self.update_console_text(string)
            
            def update_console_text(self, text):
                """Thread-safe console text update"""
                def _update():
                    try:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_text = f"[{timestamp}] {text}"
                        if not formatted_text.endswith('\n'):
                            formatted_text += '\n'
                        
                        self.text_widget.insert(tk.END, formatted_text)
                        self.text_widget.see(tk.END)
                        
                        # Limit console buffer size to prevent memory issues
                        lines = int(self.text_widget.index('end-1c').split('.')[0])
                        if lines > 1000:
                            self.text_widget.delete(1.0, "100.0")
                    except Exception as e:
                        pass  # Silently handle GUI update errors
                
                # Schedule update on main thread if needed
                try:
                    if threading.current_thread() != threading.main_thread():
                        self.text_widget.after(0, _update)
                    else:
                        _update()
                except:
                    pass
            
            def flush(self):
                if self.original_stream:
                    try:
                        self.original_stream.flush()
                    except:
                        pass
        
        # Set up redirectors
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)
        
        # Add welcome message
        self.console_text.insert(tk.END, "[CONSOLE] Console logging initialized\n")
        self.console_text.insert(tk.END, "[CONSOLE] All print statements and errors will appear here\n\n")
    
    def clear_console(self):
        """Clear the console output"""
        self.console_text.delete(1.0, tk.END)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_text.insert(tk.END, f"[{timestamp}] Console cleared\n\n")
    
    def save_console_log(self):
        """Save console log to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")],
            title="Save Console Log",
            initialname=f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        )
        if filename:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    f.write("STOCK PREDICTOR CONSOLE LOG\n")
                    f.write("=" * 50 + "\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                    f.write(self.console_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Console log saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save console log:\n{str(e)}")
    
    def restore_console_streams(self):
        """Restore original console streams"""
        try:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except:
            pass
    
    def save_results(self):
        """Save results to file"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
            title="Save Results"
        )
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write("STOCK PREDICTION RESULTS\n")
                    f.write("=" * 50 + "\n\n")
                    f.write("PREDICTIONS:\n")
                    f.write(self.results_text.get(1.0, tk.END))
                    f.write("\n\nTECHNICAL ANALYSIS:\n")
                    f.write(self.analysis_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")
    
    def load_example(self):
        """Load example configuration"""
        self.ticker_var.set("AAPL")
        self.start_date_var.set("2020-01-01")
        self.end_date_var.set("2025-08-01")
        self.pred_date_var.set("2025-08-01")
        self.lookback_var.set("60")
        self.epochs_var.set("30")
        self.reset_default_indicators()
        messagebox.showinfo("Example Loaded", "Example configuration loaded for Apple (AAPL)")
    
    def show_help(self):
        """Show help dialog"""
        help_text = """
STOCK PREDICTION HELP

INPUTS:
• Stock Ticker: Enter the stock symbol (e.g., AAPL, MSFT, GOOGL)
• Historical Start Date: Start date for training data (YYYY-MM-DD)
• Historical End Date: End date for training data (YYYY-MM-DD)  
• Prediction Start Date: Date from which to predict (typically same as end date)

TECHNICAL INDICATORS:
Select the indicators to use for prediction. More indicators generally improve accuracy but increase computation time.

MODEL PARAMETERS:
• Lookback Days: Number of historical days used for each prediction (30-120 recommended)
• Training Epochs: Number of training iterations (20-100 recommended)

TIPS:
• Use at least 2-3 years of historical data for better accuracy
• More epochs generally improve accuracy but take longer
• Start with default settings and adjust based on results
• Check the Technical Analysis tab for current market conditions

The system will:
1. Download historical stock data
2. Calculate selected technical indicators  
3. Train an LSTM neural network
4. Generate 7-day price predictions
5. Display results with charts and analysis
        """
        
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x500")
        help_window.transient(self.root)
        help_window.grab_set()
        
        text_widget = ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)
    
    def cleanup_charts(self):
        """Properly cleanup all charts and matplotlib figures"""
        self.cleanup_prediction_chart()
        self.cleanup_main_charts()
    
    def cleanup_prediction_chart(self):
        """Cleanup prediction chart canvas and figure"""
        try:
            # Close matplotlib figure if it exists
            if self.current_prediction_fig is not None:
                plt.close(self.current_prediction_fig)
                self.current_prediction_fig = None
            
            # Clear canvas reference
            if self.prediction_canvas is not None:
                try:
                    self.prediction_canvas.get_tk_widget().destroy()
                except:
                    pass
                self.prediction_canvas = None
            
            # Clear all widgets in the frame
            for widget in self.prediction_chart_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
                    
        except Exception as e:
            self.log_message(f"Warning: Error cleaning up prediction chart: {str(e)}")
    
    def cleanup_main_charts(self):
        """Cleanup main charts canvas and figure"""
        try:
            # Close matplotlib figure if it exists
            if self.current_chart_fig is not None:
                plt.close(self.current_chart_fig)
                self.current_chart_fig = None
            
            # Clear canvas reference
            if self.chart_canvas is not None:
                try:
                    self.chart_canvas.get_tk_widget().destroy()
                except:
                    pass
                self.chart_canvas = None
            
            # Clear all widgets in the frame
            for widget in self.chart_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
                    
        except Exception as e:
            self.log_message(f"Warning: Error cleaning up main charts: {str(e)}")
    
    def _reset_ui_state(self):
        """Thread-safe UI state reset"""
        try:
            self.predict_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            
            # Only reset progress if prediction didn't complete successfully
            if not hasattr(self, 'progress_var') or self.progress_var.get() < 100:
                self.update_progress(0, "Ready")
        except Exception as e:
            print(f"Error resetting UI state: {str(e)}")
    
    def quit_application(self):
        """Quit the application with proper cleanup"""
        try:
            if self.is_predicting:
                if messagebox.askokcancel("Quit",
                                        "Prediction is currently running. Do you want to quit anyway?\n\n"
                                        "This will stop the prediction and close the application."):
                    self.stop_prediction()
                    self._perform_quit()
            else:
                if messagebox.askokcancel("Quit", "Are you sure you want to quit the application?"):
                    self._perform_quit()
        except Exception as e:
            print(f"Error during quit: {str(e)}")
            self._perform_quit()  # Force quit if there's an error
    
    def _perform_quit(self):
        """Perform the actual quit operation with cleanup"""
        try:
            # Clean up charts and figures
            self.cleanup_charts()
            
            # Restore console streams
            self.restore_console_streams()
            
            # Stop any running threads
            self.is_predicting = False
            
            # Close the application
            self.root.quit()
            self.root.destroy()
            
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            try:
                self.root.destroy()
            except:
                pass

def main():
    """Main function to run the GUI application"""
    
    # Create the main window
    root = tk.Tk()
    
    # Set application icon and title
    root.title("Stock Price Predictor with LSTM")
    
    try:
        # Try to set a nice icon (optional)
        root.iconbitmap(default="stock_icon.ico")  # You can add an icon file
    except:
        pass  # No icon file, that's fine
    
    # Create the application
    app = StockPredictorGUI(root)
    
    # Handle window closing
    def on_closing():
        try:
            if app.is_predicting:
                if messagebox.askokcancel("Quit", "Prediction is running. Do you want to quit?"):
                    app.stop_prediction()
                    app.cleanup_charts()  # Clean up charts before closing
                    app.restore_console_streams()  # Restore console streams
                    root.destroy()
            else:
                app.cleanup_charts()  # Clean up charts before closing
                app.restore_console_streams()  # Restore console streams
                root.destroy()
        except Exception as e:
            print(f"Error during window closing: {str(e)}")
            try:
                app.restore_console_streams()
            except:
                pass
            root.destroy()  # Force close if there's an error
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    
    # Start the GUI event loop
    root.mainloop()

if __name__ == "__main__":
    main()