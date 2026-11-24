"""
Phase 2: Indian Market Feature Engineering & Target Formation
Constructs leakage-free features based on Indian Domestic Gold Market:
1. Domestic Gold Returns (₹ per 10g & GOLDBEES ETF)
2. USD/INR Currency Momentum (Core Domestic Multiplier)
3. Nifty 50 Equity Index Risk Appetite
4. Domestic Gold / Silver Ratio
5. Technical Oscillators (14-day RSI, MACD, Moving Average Ratios, Volatility Regimes)
6. US 10Y Real TIPS Rates (Global Macro Anchor)
7. Domestic Binary Directional Targets (Next-Day & 3-Day in INR)

Output: data/processed/indian_gold_features.csv
"""

import sys
import importlib
from pathlib import Path
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def get_data_ingestion_runner():
    """Dynamically import run_data_ingestion from 01_data_ingestion."""
    module = importlib.import_module("src.01_data_ingestion")
    return getattr(module, "run_data_ingestion")

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50.0)

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """Calculate MACD Line, Signal Line, and Histogram."""
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return macd_line, macd_signal, macd_hist

def engineer_indian_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Build stationary, predictive feature matrix for Indian Gold."""
    df = df_input.copy()
    
    # Primary price series in INR
    gold_inr = df["Indian_Gold_24K_10g"]
    
    # ---------------------------------------------------------
    # 1. Domestic Gold Price Momentum (INR)
    # ---------------------------------------------------------
    df["Indian_Gold_Return_1d"] = gold_inr.pct_change(1)
    df["Indian_Gold_Return_3d"] = gold_inr.pct_change(3)
    df["Indian_Gold_Return_5d"] = gold_inr.pct_change(5)
    df["Indian_Gold_Return_10d"] = gold_inr.pct_change(10)
    
    # NSE Gold BeES ETF Momentum
    if "GOLDBEES_Close" in df.columns:
        df["GOLDBEES_Return_1d"] = df["GOLDBEES_Close"].pct_change(1)
        df["GOLDBEES_Return_5d"] = df["GOLDBEES_Close"].pct_change(5)
        
    # ---------------------------------------------------------
    # 2. USD / INR Exchange Rate Dynamics (Primary Indian Driver)
    # ---------------------------------------------------------
    df["USDINR_Return_1d"] = df["USDINR_Close"].pct_change(1)
    df["USDINR_Return_5d"] = df["USDINR_Close"].pct_change(5)
    df["USDINR_Vol_10d"] = df["USDINR_Return_1d"].rolling(10).std() * np.sqrt(252)
    
    # ---------------------------------------------------------
    # 3. Nifty 50 Equity Dynamics (Indian Risk Appetite)
    # ---------------------------------------------------------
    if "NIFTY50_Close" in df.columns:
        df["NIFTY_Return_1d"] = df["NIFTY50_Close"].pct_change(1)
        df["NIFTY_Return_5d"] = df["NIFTY50_Close"].pct_change(5)
        df["Gold_to_Nifty_Ratio"] = gold_inr / df["NIFTY50_Close"]
        
    # ---------------------------------------------------------
    # 4. Domestic Gold / Silver Ratio
    # ---------------------------------------------------------
    if "Global_Silver_USD" in df.columns:
        # Silver price in INR
        silver_inr = (df["Global_Silver_USD"] / config.TROY_OUNCE_TO_GRAMS * 10.0) * df["USDINR_Close"]
        df["Gold_Silver_Ratio_INR"] = gold_inr / silver_inr
        df["GSR_Change_5d"] = df["Gold_Silver_Ratio_INR"].pct_change(5)
        
    # ---------------------------------------------------------
    # 5. Macroeconomic Drivers (FRED)
    # ---------------------------------------------------------
    if "10Y_Real_Rate" in df.columns:
        df["Real_Rate_Change_1d"] = df["10Y_Real_Rate"].diff(1)
        df["Real_Rate_Change_5d"] = df["10Y_Real_Rate"].diff(5)
        
    if "10Y_Nominal_Yield" in df.columns:
        df["Nominal_Yield_Change_5d"] = df["10Y_Nominal_Yield"].diff(5)

    # ---------------------------------------------------------
    # 6. Technical Indicators on Domestic Gold (₹)
    # ---------------------------------------------------------
    sma_5 = gold_inr.rolling(window=5).mean()
    sma_10 = gold_inr.rolling(window=10).mean()
    sma_20 = gold_inr.rolling(window=20).mean()
    sma_50 = gold_inr.rolling(window=50).mean()
    
    df["SMA_Ratio_5_20"] = sma_5 / sma_20
    df["SMA_Ratio_10_50"] = sma_10 / sma_50
    df["Price_to_SMA20"] = gold_inr / sma_20
    
    # 14-day RSI
    df["RSI_14"] = calculate_rsi(gold_inr, period=14)
    
    # MACD Indicators
    macd, signal, hist = calculate_macd(gold_inr, fast=12, slow=26, signal=9)
    df["MACD"] = macd
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist
    
    # Annualized Rolling Volatilities
    df["Indian_Gold_Vol_10d"] = df["Indian_Gold_Return_1d"].rolling(10).std() * np.sqrt(252)
    df["Indian_Gold_Vol_20d"] = df["Indian_Gold_Return_1d"].rolling(20).std() * np.sqrt(252)
    df["Vol_Ratio_10_20"] = df["Indian_Gold_Vol_10d"] / df["Indian_Gold_Vol_20d"]
    
    # Autoregressive Lags
    df["Indian_Gold_Return_Lag1"] = df["Indian_Gold_Return_1d"].shift(1)
    df["Indian_Gold_Return_Lag2"] = df["Indian_Gold_Return_1d"].shift(2)
    
    # ---------------------------------------------------------
    # 7. Domestic Targets (Zero Look-Ahead Bias)
    # ---------------------------------------------------------
    # Next-Day Direction in Indian Rupee: 1 if Close_(t+1) > Close_t, else 0
    df["Target_Indian_Gold_Next_Day"] = (gold_inr.shift(-1) > gold_inr).astype(int)
    
    # 3-Day Directional Target (Smooths 1-day noise)
    df["Target_Indian_Gold_3D"] = (gold_inr.shift(-3) > gold_inr).astype(int)
    
    # Forward Returns in INR for Backtesting
    df["Target_Return_INR_1d"] = (gold_inr.shift(-1) - gold_inr) / gold_inr
    df["Target_Return_INR_3d"] = (gold_inr.shift(-3) - gold_inr) / gold_inr
    
    # Clean up initial NaNs from rolling windows (e.g. 50-day SMA)
    # We drop rows where features are NaN, but preserve the latest rows for live inference
    feature_cols_only = [c for c in df.columns if not c.startswith("Target_")]
    df_clean = df.dropna(subset=feature_cols_only).copy()
    return df_clean

def run_feature_engineering() -> pd.DataFrame:
    """Main runner for Phase 2 Indian feature engineering."""
    print("=" * 65)
    print("PHASE 2: INDIAN FEATURE ENGINEERING & TARGET FORMATION")
    print("=" * 65)
    
    input_path = config.PROCESSED_DATA_DIR / "indian_gold_cleaned.csv"
    if not input_path.exists():
        print("Cleaned Indian dataset not found. Running Phase 1 first...")
        run_ingestion = get_data_ingestion_runner()
        df_clean = run_ingestion()
    else:
        print(f"Loading cleaned dataset from {input_path}...")
        df_clean = pd.read_csv(input_path, parse_dates=["Date"]).set_index("Date")
        
    print(f"Input records: {len(df_clean)} rows.")
    print("Engineering domestic USD/INR dynamics, Nifty 50 sentiment, and ₹/10g targets...")
    feature_matrix = engineer_indian_features(df_clean)
    
    # Save processed feature matrix
    output_path = config.PROCESSED_DATA_DIR / "indian_gold_features.csv"
    feature_matrix.to_csv(output_path)
    
    target_counts_1d = feature_matrix["Target_Indian_Gold_Next_Day"].value_counts(normalize=True) * 100
    target_counts_3d = feature_matrix["Target_Indian_Gold_3D"].value_counts(normalize=True) * 100
    
    print(f"\n[SUCCESS] Phase 2 Indian Feature Engineering Completed!")
    print(f"  - Output records: {len(feature_matrix)} Indian trading days")
    print(f"  - Total feature columns: {len(feature_matrix.columns)}")
    print(f"  - Saved to: {output_path}")
    print("\nDomestic Target Distribution (Next-Day Direction):")
    print(f"  - UP Days (1)   : {target_counts_1d.get(1, 0):.2f}%")
    print(f"  - DOWN Days (0) : {target_counts_1d.get(0, 0):.2f}%")
    print("\nDomestic Target Distribution (3-Day Horizon):")
    print(f"  - UP Periods (1)   : {target_counts_3d.get(1, 0):.2f}%")
    print(f"  - DOWN Periods (0) : {target_counts_3d.get(0, 0):.2f}%")
    print("=" * 65)
    
    return feature_matrix

if __name__ == "__main__":
    run_feature_engineering()
