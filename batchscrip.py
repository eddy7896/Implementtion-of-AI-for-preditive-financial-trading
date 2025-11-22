#!/usr/bin/env python3
"""
Upgraded Stock Price Prediction GUI Application
Features added:
- Batch processing up to 50 tickers (parallel processing with ThreadPoolExecutor)
- Retry logic and robust error handling per ticker
- Per-ticker charts saved to `batch_outputs/` and a master Excel summary + per-ticker sheets
- Lightweight risk score (based on historical volatility) and model evaluation capture
- Progress updates and logging into GUI console

Original uploaded file path: /mnt/data/stock_predictor_gui.py

Note: This file is an upgraded drop-in replacement for the original GUI.
Make sure stock_predictor_lstm.py (the predictor class) is available in the same directory.
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tkinter.scrolledtext import ScrolledText
import threading
import sys
import os
import io
from datetime import datetime, timedelta
import traceback

# Parallel and utility imports
import concurrent.futures
import time
import math
import zipfile

# Configure matplotlib for GUI use
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import numpy as np

# Import our stock predictor (must exist)
from stock_predictor_lstm import StockPredictor

# ---------------------------
# Configuration defaults
# ---------------------------
MAX_BATCH_SIZE = 50
MAX_PARALLEL_WORKERS = 8  # tweak for your machine
RETRY_LIMIT = 2
OUTPUT_FOLDER = "batch_outputs"

# Ensure output folder exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

class StockPredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction with LSTM & Technical Indicators (Upgraded)")
        self.root.geometry("1250x820")
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
        self.root.update_idletasks()
        width = self.root.winfo_width()
        height = self.root.winfo_height()
        x = (self.root.winfo_screenwidth() // 2) - (width // 2)
        y = (self.root.winfo_screenheight() // 2) - (height // 2)
        self.root.geometry(f'{width}x{height}+{x}+{y}')

    def create_widgets(self):
        self.create_scrollable_container()
        title_label = ttk.Label(self.scrollable_frame, text="Stock Price Prediction System (Upgraded)",
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))

        self.create_input_panel(self.scrollable_frame)
        self.create_results_panel(self.scrollable_frame)
        self.create_control_panel(self.scrollable_frame)

    # -- reuse and adapt many helper functions from original --
    def create_scrollable_container(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        container_frame = ttk.Frame(self.root)
        container_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        container_frame.columnconfigure(0, weight=1)
        container_frame.rowconfigure(0, weight=1)
        self.main_canvas = tk.Canvas(container_frame, highlightthickness=0)
        self.v_scrollbar = ttk.Scrollbar(container_frame, orient="vertical", command=self.main_canvas.yview)
        self.h_scrollbar = ttk.Scrollbar(container_frame, orient="horizontal", command=self.main_canvas.xview)
        self.scrollable_frame = ttk.Frame(self.main_canvas, padding="10")
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.main_canvas.configure(scrollregion=self.main_canvas.bbox("all"))
        )
        self.main_canvas.configure(yscrollcommand=self.v_scrollbar.set, xscrollcommand=self.h_scrollbar.set)
        self.canvas_window = self.main_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.main_canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.v_scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))
        self.h_scrollbar.grid(row=1, column=0, sticky=(tk.W, tk.E))
        self.scrollable_frame.columnconfigure(1, weight=1)
        self.bind_mousewheel()
        self.main_canvas.bind('<Configure>', self.on_canvas_configure)

    def bind_mousewheel(self):
        def _on_mousewheel(event):
            self.main_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        def _on_shift_mousewheel(event):
            self.main_canvas.xview_scroll(int(-1 * (event.delta / 120)), "units")
        self.main_canvas.bind_all("<MouseWheel>", _on_mousewheel)
        self.main_canvas.bind_all("<Shift-MouseWheel>", _on_shift_mousewheel)
        self.main_canvas.bind_all("<Button-4>", lambda e: self.main_canvas.yview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Button-5>", lambda e: self.main_canvas.yview_scroll(1, "units"))
        self.main_canvas.bind_all("<Shift-Button-4>", lambda e: self.main_canvas.xview_scroll(-1, "units"))
        self.main_canvas.bind_all("<Shift-Button-5>", lambda e: self.main_canvas.xview_scroll(1, "units"))

    def on_canvas_configure(self, event):
        canvas_width = event.width
        frame_width = self.scrollable_frame.winfo_reqwidth()
        if frame_width < canvas_width:
            self.main_canvas.itemconfig(self.canvas_window, width=canvas_width)
        else:
            self.main_canvas.itemconfig(self.canvas_window, width=frame_width)

    def create_input_panel(self, parent):
        input_frame = ttk.LabelFrame(parent, text="Stock Configuration", padding="10")
        input_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 10))

        # Multi-ticker input
        ttk.Label(input_frame, text="Stock Tickers (comma-separated, max 50):").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.ticker_var = tk.StringVar(value="AAPL, MSFT, GOOGL")
        ticker_entry = ttk.Entry(input_frame, textvariable=self.ticker_var, width=50)
        ticker_entry.grid(row=0, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=2)

        ttk.Label(input_frame, text="Historical Start Date:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.start_date_var = tk.StringVar(value="2020-01-01")
        start_entry = ttk.Entry(input_frame, textvariable=self.start_date_var, width=15)
        start_entry.grid(row=1, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(row=1, column=2, sticky=tk.W, padx=(5, 0))

        ttk.Label(input_frame, text="Historical End Date:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.end_date_var = tk.StringVar(value=(datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d"))
        end_entry = ttk.Entry(input_frame, textvariable=self.end_date_var, width=15)
        end_entry.grid(row=2, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(YYYY-MM-DD)", font=('Arial', 8)).grid(row=2, column=2, sticky=tk.W, padx=(5, 0))

        ttk.Label(input_frame, text="Prediction Start Date:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.pred_date_var = tk.StringVar(value=self.end_date_var.get())
        pred_entry = ttk.Entry(input_frame, textvariable=self.pred_date_var, width=15)
        pred_entry.grid(row=3, column=1, sticky=(tk.W, tk.E), pady=2)
        ttk.Label(input_frame, text="(Start of prediction)", font=('Arial', 8)).grid(row=3, column=2, sticky=tk.W, padx=(5, 0))

        ttk.Separator(input_frame, orient='horizontal').grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)

        # minimal indicators selection (reuse defaults)
        self.indicator_vars = {}
        indicators = [("sma_20", "SMA 20", True), ("rsi", "RSI", True), ("macd", "MACD", True),
                      ("bb_upper", "BB Upper", True), ("obv", "OBV", True)]
        row = 5
        for key, label, default in indicators:
            var = tk.BooleanVar(value=default)
            self.indicator_vars[key] = var
            cb = ttk.Checkbutton(input_frame, text=label, variable=var)
            cb.grid(row=row, column=0, sticky=tk.W, pady=1, columnspan=3)
            row += 1

        ttk.Separator(input_frame, orient='horizontal').grid(row=row, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=8)
        row += 1

        ttk.Label(input_frame, text="Model Parameters:", font=('Arial', 10, 'bold')).grid(row=row, column=0, columnspan=3, sticky=tk.W, pady=(5, 2))
        row += 1

        ttk.Label(input_frame, text="Lookback Days:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.lookback_var = tk.StringVar(value="60")
        ttk.Entry(input_frame, textvariable=self.lookback_var, width=8).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(input_frame, text="Training Epochs:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.epochs_var = tk.StringVar(value="30")
        ttk.Entry(input_frame, textvariable=self.epochs_var, width=8).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(input_frame, text="Parallel Workers:").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.workers_var = tk.StringVar(value=str(MAX_PARALLEL_WORKERS))
        ttk.Entry(input_frame, textvariable=self.workers_var, width=8).grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        input_frame.columnconfigure(1, weight=1)

    def create_results_panel(self, parent):
        results_frame = ttk.LabelFrame(parent, text="Results & Visualization", padding="5")
        results_frame.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(10, 0))
        results_frame.rowconfigure(0, weight=1)
        results_frame.columnconfigure(0, weight=1)
        self.notebook = ttk.Notebook(results_frame)
        self.notebook.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        self.results_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.results_tab, text="Predictions")
        self.results_text = ScrolledText(self.results_tab, wrap=tk.WORD, width=60, height=20)
        self.results_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.prediction_chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.prediction_chart_tab, text="Prediction Chart")
        self.prediction_chart_frame = ttk.Frame(self.prediction_chart_tab)
        self.prediction_chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.chart_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.chart_tab, text="Technical Charts")
        self.chart_frame = ttk.Frame(self.chart_tab)
        self.chart_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.analysis_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.analysis_tab, text="Technical Analysis")
        self.analysis_text = ScrolledText(self.analysis_tab, wrap=tk.WORD, width=50, height=20)
        self.analysis_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        self.console_tab = ttk.Frame(self.notebook)
        self.notebook.add(self.console_tab, text="Console Logs")
        self.console_text = ScrolledText(self.console_tab, wrap=tk.WORD, width=50, height=20,
                                        bg='black', fg='lightgreen', insertbackground='lightgreen',
                                        font=('Consolas', 9))
        self.console_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=5)
        console_controls = ttk.Frame(self.console_tab)
        console_controls.grid(row=1, column=0, sticky=(tk.W, tk.E), padx=5, pady=5)
        ttk.Button(console_controls, text="Clear Console", command=self.clear_console).pack(side=tk.LEFT, padx=2)
        ttk.Button(console_controls, text="Save Console Log", command=self.save_console_log).pack(side=tk.LEFT, padx=2)
        self.setup_console_redirection()

    def create_control_panel(self, parent):
        control_frame = ttk.LabelFrame(parent, text="Controls", padding="10")
        control_frame.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        control_frame.columnconfigure(1, weight=1)
        ttk.Label(control_frame, text="Progress:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(control_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=(10, 10), pady=2)
        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.grid(row=0, column=2, sticky=tk.W, pady=2)
        button_frame = ttk.Frame(control_frame)
        button_frame.grid(row=1, column=0, columnspan=3, pady=(10, 0))
        self.predict_btn = ttk.Button(button_frame, text="Start Prediction", command=self.start_batch_prediction, style="Accent.TButton")
        self.predict_btn.pack(side=tk.LEFT, padx=5)
        self.batch_btn = ttk.Button(button_frame, text="Batch (up to 50) Prediction", command=self.start_batch_prediction)
        self.batch_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_prediction, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Save Results", command=self.save_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Load Example", command=self.load_example).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Help", command=self.show_help).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Quit", command=self.quit_application).pack(side=tk.LEFT, padx=5)

    # -------------------- existing helper methods (validate_inputs, logging, progress) --------------------
    def validate_inputs(self):
        tickers_raw = self.ticker_var.get().strip()
        if not tickers_raw:
            raise ValueError("Please enter at least one stock ticker")
        tickers = [t.strip().upper() for t in tickers_raw.split(",") if t.strip()]
        if len(tickers) > MAX_BATCH_SIZE:
            raise ValueError(f"Max {MAX_BATCH_SIZE} tickers allowed")

        try:
            start_date = datetime.strptime(self.start_date_var.get(), "%Y-%m-%d")
            end_date = datetime.strptime(self.end_date_var.get(), "%Y-%m-%d")
            pred_date = datetime.strptime(self.pred_date_var.get(), "%Y-%m-%d")
        except ValueError:
            raise ValueError("Please enter dates in YYYY-MM-DD format")
        if start_date >= end_date:
            raise ValueError("Historical start date must be before end date")
        if pred_date < end_date:
            # allow equal as well; user may want to predict starting from last available
            pass
        try:
            lookback = int(self.lookback_var.get())
            epochs = int(self.epochs_var.get())
            workers = int(self.workers_var.get())
            if lookback < 1 or lookback > 365:
                raise ValueError("Lookback days must be between 1 and 365")
            if epochs < 1 or epochs > 1000:
                raise ValueError("Training epochs must be between 1 and 1000")
            if workers < 1 or workers > 32:
                raise ValueError("Parallel workers must be between 1 and 32")
        except ValueError as e:
            if 'invalid literal' in str(e):
                raise ValueError("Lookback days, epochs, and parallel workers must be valid integers")
            else:
                raise e

        selected_indicators = [key for key, var in self.indicator_vars.items() if var.get()]
        if not selected_indicators:
            raise ValueError("Please select at least one technical indicator")

        return tickers, start_date, end_date, pred_date, lookback, epochs, selected_indicators, workers

    def update_progress(self, value, text):
        def _update():
            try:
                self.progress_var.set(value)
                self.progress_label.config(text=text)
                self.root.update_idletasks()
            except Exception as e:
                print(f"Error updating progress: {str(e)}")
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, _update)
        else:
            _update()

    def log_message(self, message, tab="results"):
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
        if threading.current_thread() != threading.main_thread():
            self.root.after(0, _update_log)
        else:
            _update_log()

    # -------------------- Batch orchestration (upgraded) --------------------
    def start_batch_prediction(self):
        try:
            tickers, start_date, end_date, pred_date, lookback, epochs, indicators, workers = self.validate_inputs()
            if len(tickers) == 0:
                raise ValueError("No tickers found")
            self.is_predicting = True
            self.predict_btn.config(state="disabled")
            self.batch_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.clear_results()

            # Kick off background thread which will use ThreadPoolExecutor internally
            self.prediction_thread = threading.Thread(
                target=self.run_parallel_batch,
                args=(tickers, start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"),
                      pred_date.strftime("%Y-%m-%d"), lookback, epochs, indicators, workers)
            )
            self.prediction_thread.daemon = True
            self.prediction_thread.start()

        except Exception as e:
            messagebox.showerror("Input Error", str(e))

    def stop_prediction(self):
        self.is_predicting = False
        self.update_progress(0, "Stopping...")
        self.predict_btn.config(state="normal")
        self.batch_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.log_message("Batch stopped by user")
        self.update_progress(0, "Ready")

    def run_parallel_batch(self, tickers, start_date, end_date, pred_date, lookback, epochs, indicators, workers):
        """Run batch predictions in parallel with retries and save detailed outputs."""
        start_time = time.time()
        summary_results = []
        details = {}  # ticker -> DataFrame for detailed sheet
        total = len(tickers)

        # clamp workers
        max_workers = min(max(1, int(workers)), MAX_PARALLEL_WORKERS)

        self.log_message(f"Starting parallel batch for {total} tickers with {max_workers} workers")

        def worker_task(ticker):
            attempts = 0
            last_exc = None
            while attempts <= RETRY_LIMIT and self.is_predicting:
                attempts += 1
                try:
                    self.log_message(f"[{ticker}] Attempt {attempts} - initializing predictor")
                    predictor = StockPredictor(ticker, lookback_days=lookback, prediction_days=7, selected_indicators=indicators)
                    ok = predictor.fetch_data(start_date, end_date)
                    if not ok:
                        raise RuntimeError("fetch_data failed")
                    predictor.calculate_technical_indicators()
                    predictor.prepare_lstm_data()

                    # try to build/train model (some tickers might fail; continue with fallback)
                    try:
                        predictor.build_lstm_model()
                        predictor.train_model(epochs=epochs, batch_size=32)
                    except Exception as e:
                        # log but continue - predictor may still expose fallback prediction
                        self.log_message(f"[{ticker}] Model train warning: {e}")

                    # evaluate if possible
                    eval_metrics = {}
                    try:
                        res = predictor.evaluate_model()
                        if isinstance(res, dict):
                            eval_metrics = res
                    except Exception:
                        # try to compute simple in-sample metric if predictor has y_test/y_pred
                        try:
                            y_true = getattr(predictor, 'y_test', None)
                            y_pred = getattr(predictor, 'y_pred', None)
                            if y_true is not None and y_pred is not None and len(y_true) == len(y_pred) and len(y_true)>0:
                                mse = float(np.mean((np.array(y_true) - np.array(y_pred))**2))
                                eval_metrics = {'mse': mse}
                        except Exception:
                            pass

                    prediction_dates, predicted_prices = predictor.predict_next_week(pred_date)
                    if prediction_dates is None or predicted_prices is None:
                        raise RuntimeError('Prediction generation failed')

                    # create per-ticker dataframe (actual historical + predicted appended)
                    hist_df = predictor.df.copy()
                    hist_df = hist_df.reset_index().rename(columns={'index': 'Date'})

                    preds_df = pd.DataFrame({
                        'Date': prediction_dates,   # keep as datetime objects
                        'Predicted_Close': predicted_prices
                    })


                    # Risk score: historical volatility (std of daily returns * sqrt(252))
                    try:
                        returns = hist_df['Close'].pct_change().dropna()
                        volatility = float(returns.std() * math.sqrt(252))
                        risk_score = round(volatility * 100, 3)  # percentage annualized vol
                    except Exception:
                        risk_score = None

                    # Save chart for this ticker
                    chart_path = os.path.join(OUTPUT_FOLDER, f"{ticker}_prediction.png")
                    try:
                        fig, ax = plt.subplots(1, 1, figsize=(8,4))
                        ax.plot(hist_df['Date'], hist_df['Close'], label='Historical Close')
                        ax.plot(preds_df['Date'], preds_df['Predicted_Close'], 'ro-', label='Predictions')
                        ax.set_title(f"{ticker} - Historical & Predicted Close")
                        ax.tick_params(axis='x', rotation=30)
                        ax.legend()
                        plt.tight_layout()
                        fig.savefig(chart_path, dpi=150)
                        plt.close(fig)
                    except Exception as e:
                        self.log_message(f"[{ticker}] Chart failed: {e}")

                    # Prepare summary info
                    last_actual = float(hist_df['Close'].iloc[-1])
                    pred_end = float(predicted_prices[-1])
                    ret_pct = round(((pred_end - last_actual) / last_actual) * 100, 3)

                    summary = {
                        'Ticker': ticker,
                        'Last_Actual_Price': last_actual,
                        'Pred_Week_End_Price': pred_end,
                        'Return_%': ret_pct,
                        'Pred_High': float(max(predicted_prices)),
                        'Pred_Low': float(min(predicted_prices)),
                        'Risk_Score_Annualized_%': risk_score,
                        'Chart_Path': chart_path,
                        'Prediction_Start': pred_date,
                        'Actual_End_Date': hist_df['Date'].iloc[-1]
                    }
                    summary.update(eval_metrics)

                    # combine detail dataframe (hist + predictions)
                    detail_df = hist_df.copy()
                    # ensure Date is string
                    detail_df['Date'] = detail_df['Date'].astype(str)
                    # append predictions with NaNs for non-predicted cols
                    preds_df_extended = preds_df.copy()
                    # align columns
                    for c in detail_df.columns:
                        if c not in preds_df_extended.columns:
                            preds_df_extended[c] = np.nan
                    preds_df_extended = preds_df_extended[detail_df.columns]
                    combined_detail = pd.concat([detail_df, preds_df_extended], ignore_index=True, sort=False)

                    return (ticker, summary, combined_detail)

                except Exception as e:
                    last_exc = e
                    self.log_message(f"[{ticker}] Error on attempt {attempts}: {e}")
                    time.sleep(1)
                    continue
            # final failure
            err_msg = traceback.format_exception_only(type(last_exc), last_exc) if last_exc else ['Unknown error']
            return (ticker, {'error': str(err_msg)}, None)

        # Run parallel tasks
        futures = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            for t in tickers:
                futures[executor.submit(worker_task, t)] = t

            completed = 0
            for fut in concurrent.futures.as_completed(futures):
                ticker = futures[fut]
                try:
                    res = fut.result()
                    if res is None:
                        continue
                    tkr, summary, detail_df = res
                    completed += 1
                    pct = int((completed/total)*100)
                    self.update_progress(pct, f"Completed {completed}/{total} ({tkr})")

                    if detail_df is not None:
                        details[tkr] = detail_df
                    if isinstance(summary, dict):
                        summary_results.append(summary)
                except Exception as e:
                    self.log_message(f"{ticker} future exception: {e}")

        # Write outputs: master Excel with summary sheet and per-ticker sheets; plus zipped charts
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        master_xlsx = os.path.join(OUTPUT_FOLDER, f'batch_summary_{timestamp}.xlsx')
        try:
            with pd.ExcelWriter(master_xlsx, engine='openpyxl') as writer:
                if summary_results:
                    pd.DataFrame(summary_results).to_excel(writer, sheet_name='Summary', index=False)
                for tkr, ddf in details.items():
                    # limit sheet name to 31 chars
                    sheet = tkr[:31]
                    try:
                        ddf.to_excel(writer, sheet_name=sheet, index=False)
                    except Exception:
                        # fallback: save as CSV per ticker
                        csv_path = os.path.join(OUTPUT_FOLDER, f"{tkr}_details_{timestamp}.csv")
                        ddf.to_csv(csv_path, index=False)
            self.log_message(f"Saved master workbook: {master_xlsx}")
        except Exception as e:
            self.log_message(f"Failed to write master workbook: {e}")

        # Zip charts
        zip_path = os.path.join(OUTPUT_FOLDER, f'charts_{timestamp}.zip')
        try:
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for summary in summary_results:
                    chart = summary.get('Chart_Path')
                    if chart and os.path.exists(chart):
                        zf.write(chart, arcname=os.path.basename(chart))
            self.log_message(f"Saved charts zip: {zip_path}")
        except Exception as e:
            self.log_message(f"Zipping charts failed: {e}")

        # Final status
        elapsed = time.time() - start_time
        self.log_message(f"Batch finished in {elapsed:.1f}s. Results: {len(summary_results)} succeeded, {total - len(summary_results)} failed")
        self.update_progress(100, "Batch Completed")

        # Reset UI state
        self.is_predicting = False
        self.root.after(0, self._reset_ui_state)

    # -------------------- smaller helpers and original functions (display, charts, save) --------------------
    def display_technical_analysis(self):
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
            if 'RSI' in self.predictor.df.columns:
                rsi = latest['RSI']
                if rsi > 70:
                    rsi_signal = "Overbought"
                elif rsi < 30:
                    rsi_signal = "Oversold"
                else:
                    rsi_signal = "Neutral"
                analysis_text += f"RSI: {rsi:.2f} ({rsi_signal})\n"
            self.analysis_text.delete(1.0, tk.END)
            self.analysis_text.insert(tk.END, analysis_text)
        except Exception as e:
            self.log_message(f"Error in technical analysis: {str(e)}", "analysis")

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)
        self.analysis_text.delete(1.0, tk.END)
        self.cleanup_charts()

    def clear_console(self):
        self.console_text.delete(1.0, tk.END)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console_text.insert(tk.END, f"[{timestamp}] Console cleared\n\n")

    def save_console_log(self):
        filename = filedialog.asksaveasfilename(defaultextension=".txt",
                                                filetypes=[("Text files", "*.txt"), ("Log files", "*.log"), ("All files", "*.*")],
                                                title="Save Console Log",
                                                initialname=f"console_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
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

    def save_results(self):
        filename = filedialog.asksaveasfilename(defaultextension=".xlsx",
                                                filetypes=[("Excel files", "*.xlsx"), ("CSV files", "*.csv"), ("All files", "*.*")],
                                                title="Save Results")
        if filename:
            try:
                with open(filename, 'w') as f:
                    f.write(self.results_text.get(1.0, tk.END))
                messagebox.showinfo("Success", f"Results saved to {filename}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save results:\n{str(e)}")

    def load_example(self):
        self.ticker_var.set("AAPL, MSFT, GOOGL")
        self.start_date_var.set("2020-01-01")
        self.end_date_var.set((datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'))
        self.pred_date_var.set(self.end_date_var.get())
        self.lookback_var.set("60")
        self.epochs_var.set("30")
        self.workers_var.set(str(MAX_PARALLEL_WORKERS))
        messagebox.showinfo("Example Loaded", "Example configuration loaded")

    def show_help(self):
        help_text = """
Upgraded Batch Prediction Help

• Enter up to 50 tickers separated by commas
• Pick historical date range (use at least 2 years for better accuracy)
• Set lookback, epochs, and parallel workers (1-32)
• Click 'Batch (up to 50) Prediction' to run parallel batch

Outputs are saved to the `batch_outputs/` folder: a master workbook and per-ticker charts.
"""
        help_window = tk.Toplevel(self.root)
        help_window.title("Help")
        help_window.geometry("600x400")
        help_window.transient(self.root)
        help_window.grab_set()
        text_widget = ScrolledText(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)
        ttk.Button(help_window, text="Close", command=help_window.destroy).pack(pady=10)

    def setup_console_redirection(self):
        class ConsoleRedirector:
            def __init__(self, text_widget, original_stream, tag=None):
                self.text_widget = text_widget
                self.original_stream = original_stream
                self.tag = tag
            def write(self, string):
                if self.original_stream:
                    try:
                        self.original_stream.write(string)
                        self.original_stream.flush()
                    except:
                        pass
                if string.strip():
                    self.update_console_text(string)
            def update_console_text(self, text):
                def _update():
                    try:
                        timestamp = datetime.now().strftime("%H:%M:%S")
                        formatted_text = f"[{timestamp}] {text}"
                        if not formatted_text.endswith('\n'):
                            formatted_text += '\n'
                        self.text_widget.insert(tk.END, formatted_text)
                        self.text_widget.see(tk.END)
                        lines = int(self.text_widget.index('end-1c').split('.')[0])
                        if lines > 1000:
                            self.text_widget.delete(1.0, "100.0")
                    except Exception as e:
                        pass
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
        sys.stdout = ConsoleRedirector(self.console_text, self.original_stdout)
        sys.stderr = ConsoleRedirector(self.console_text, self.original_stderr)
        self.console_text.insert(tk.END, "[CONSOLE] Console logging initialized\n")

    def cleanup_charts(self):
        try:
            if self.current_prediction_fig is not None:
                plt.close(self.current_prediction_fig)
                self.current_prediction_fig = None
            if self.prediction_canvas is not None:
                try:
                    self.prediction_canvas.get_tk_widget().destroy()
                except:
                    pass
                self.prediction_canvas = None
            if self.current_chart_fig is not None:
                plt.close(self.current_chart_fig)
                self.current_chart_fig = None
            if self.chart_canvas is not None:
                try:
                    self.chart_canvas.get_tk_widget().destroy()
                except:
                    pass
                self.chart_canvas = None
            for widget in self.prediction_chart_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
            for widget in self.chart_frame.winfo_children():
                try:
                    widget.destroy()
                except:
                    pass
        except Exception as e:
            self.log_message(f"Warning: Error cleaning up charts: {str(e)}")

    def _reset_ui_state(self):
        try:
            self.predict_btn.config(state="normal")
            self.batch_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            if not hasattr(self, 'progress_var') or self.progress_var.get() < 100:
                self.update_progress(0, "Ready")
        except Exception as e:
            print(f"Error resetting UI state: {str(e)}")

    def quit_application(self):
        try:
            if self.is_predicting:
                if messagebox.askokcancel("Quit", "Prediction is running. Do you want to quit?"):
                    self.stop_prediction()
                    self.cleanup_charts()
                    self.restore_console_streams()
                    self.root.destroy()
            else:
                if messagebox.askokcancel("Quit", "Are you sure you want to quit the application?"):
                    self.cleanup_charts()
                    self.restore_console_streams()
                    self.root.destroy()
        except Exception as e:
            print(f"Error during quit: {str(e)}")
            try:
                self.restore_console_streams()
            except:
                pass
            self.root.destroy()

    def restore_console_streams(self):
        try:
            sys.stdout = self.original_stdout
            sys.stderr = self.original_stderr
        except:
            pass

# -------------------- Main --------------------

def main():
    root = tk.Tk()
    root.title("Stock Price Predictor with LSTM - Upgraded Batch")
    try:
        root.iconbitmap(default="stock_icon.ico")
    except:
        pass
    app = StockPredictorGUI(root)
    def on_closing():
        try:
            if app.is_predicting:
                if messagebox.askokcancel("Quit", "Prediction is running. Do you want to quit?"):
                    app.stop_prediction()
                    app.cleanup_charts()
                    app.restore_console_streams()
                    root.destroy()
            else:
                app.cleanup_charts()
                app.restore_console_streams()
                root.destroy()
        except Exception as e:
            print(f"Error during window closing: {str(e)}")
            try:
                app.restore_console_streams()
            except:
                pass
            root.destroy()
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
