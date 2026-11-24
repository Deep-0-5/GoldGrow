"""
Phase 1: Indian Market Data Ingestion Pipeline
Fetches native Indian exchange assets and macro drivers:
1. GOLDBEES.NS   - Nippon India ETF Gold BeES (NSE-listed Gold Benchmark in INR)
2. USDINR=X      - US Dollar to Indian Rupee Exchange Rate (Primary Domestic Driver)
3. ^NSEI         - Nifty 50 Index (Indian Equity Risk Sentiment)
4. GC=F & SI=F   - Global Spot Gold & Silver Benchmarks
5. FRED DFII10   - 10-Year Real TIPS Interest Rate (Global Opportunity Cost)
6. Derives Landed Indian 24K Gold Rate (₹ per 10g) with 6% Customs Duty + 3% GST

Output: data/processed/indian_gold_cleaned.csv
"""

import sys
import os
from pathlib import Path
import json
import requests
import pandas as pd
import numpy as np
import yfinance as yf

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def fetch_fred_series(series_id: str, api_key: str, start_date: str) -> pd.DataFrame:
    """Fetch official economic series from St. Louis FRED API."""
    url = "https://api.stlouisfed.org/fred/series/observations"
    params = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "observation_start": start_date
    }
    response = requests.get(url, params=params, timeout=15)
    response.raise_for_status()
    data = response.json()
    
    observations = data.get("observations", [])
    if not observations:
        raise ValueError(f"No observations found for FRED series {series_id}")
        
    df = pd.DataFrame(observations)[["date", "value"]]
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df["date"] = pd.to_datetime(df["date"])
    col_name = config.FRED_SERIES.get(series_id, series_id)
    return df.set_index("date").rename(columns={"value": col_name})

def run_data_ingestion() -> pd.DataFrame:
    """Main data ingestion runner for Indian Market assets."""
    print("=" * 65)
    print("PHASE 1: INDIAN MARKET DATA INGESTION PIPELINE")
    print("=" * 65)
    
    # 1. Download Indian market assets and global benchmarks via yfinance
    print("[1/3] Downloading Indian assets (GOLDBEES.NS, USD/INR, Nifty 50, Global Metals)...")
    tickers = list(config.INDIAN_TICKERS.values())
    raw_df = yf.download(tickers, start=config.START_DATE, progress=False)
    
    # Process Close prices
    if isinstance(raw_df.columns, pd.MultiIndex):
        close_df = raw_df["Close"]
        high_df = raw_df["High"]
        low_df = raw_df["Low"]
        open_df = raw_df["Open"]
        vol_df = raw_df["Volume"]
    else:
        close_df = raw_df
        high_df = low_df = open_df = vol_df = None

    df_indian = pd.DataFrame(index=close_df.index)
    
    # Map columns to standardized names
    df_indian["GOLDBEES_Close"] = close_df["GOLDBEES.NS"]
    df_indian["USDINR_Close"] = close_df["USDINR=X"]
    df_indian["NIFTY50_Close"] = close_df["^NSEI"]
    
    if "GC=F" in close_df.columns:
        df_indian["Global_Gold_USD"] = close_df["GC=F"]
    if "SI=F" in close_df.columns:
        df_indian["Global_Silver_USD"] = close_df["SI=F"]
        
    if high_df is not None and "GOLDBEES.NS" in high_df.columns:
        df_indian["GOLDBEES_High"] = high_df["GOLDBEES.NS"]
        df_indian["GOLDBEES_Low"] = low_df["GOLDBEES.NS"]
        df_indian["GOLDBEES_Open"] = open_df["GOLDBEES.NS"]
        df_indian["GOLDBEES_Volume"] = vol_df["GOLDBEES.NS"]
        
    df_indian.index = pd.to_datetime(df_indian.index).tz_localize(None)
    
    # 2. Fetch FRED Macro Rates (10Y Real TIPS Yield & Nominal Yield)
    print("[2/3] Fetching macroeconomic drivers from FRED API...")
    macro_dfs = []
    if config.FRED_API_KEY.strip():
        try:
            for s_id in config.FRED_SERIES.keys():
                df_f = fetch_fred_series(s_id, config.FRED_API_KEY, config.START_DATE)
                macro_dfs.append(df_f)
            combined_macro = pd.concat(macro_dfs, axis=1)
            df_merged = df_indian.join(combined_macro, how="outer")
        except Exception as e:
            print(f"  [WARNING] FRED API fetch failed ({e}). Proceeding with market feeds.")
            df_merged = df_indian
    else:
        df_merged = df_indian

    # 3. Clean, sort, and forward-fill Indian market holidays
    print("[3/3] Aligning dates to Indian exchange calendar & calculating landed 24K rate...")
    df_merged = df_merged.sort_index()
    df_merged = df_merged.dropna(how="all")
    # Forward-fill across weekend & holiday gaps
    df_merged = df_merged.ffill().bfill()
    
    # 4. Derive Official Landed Indian 24K Gold Rate (₹ per 10 grams)
    # Raw Metal Value in INR per 10g: (Spot USD / 31.1035) * 10 * USDINR
    usd_per_gram = df_merged["Global_Gold_USD"] / config.TROY_OUNCE_TO_GRAMS
    raw_inr_10g = (usd_per_gram * 10.0) * df_merged["USDINR_Close"]
    
    # Official IBJA / Retail City (Ahmedabad/Mumbai/Delhi) 24K Landed Benchmark
    # Includes Total Tariff, AIDC, Bullion Premium (15%) + 3% GST (1.1845x multiplier)
    df_merged["Indian_Gold_24K_10g"] = raw_inr_10g * config.INDIAN_RETAIL_MULTIPLIER
    df_merged["Indian_Gold_22K_10g"] = df_merged["Indian_Gold_24K_10g"] * config.PURITY_22K
    
    df_merged.index.name = "Date"
    
    # Save cleaned dataset
    output_path = config.PROCESSED_DATA_DIR / "indian_gold_cleaned.csv"
    df_merged.to_csv(output_path)
    
    # Cache raw copy
    df_merged.to_csv(config.RAW_DATA_DIR / "indian_raw_cache.csv")
    
    print(f"\n[SUCCESS] Phase 1 Indian Ingestion Completed!")
    print(f"  - Total clean Indian trading sessions: {len(df_merged)}")
    print(f"  - Date Range: {df_merged.index.min().strftime('%Y-%m-%d')} to {df_merged.index.max().strftime('%Y-%m-%d')}")
    print(f"  - Clean file saved to: {output_path}")
    print(f"  - Columns available: {list(df_merged.columns)}")
    print("=" * 65)
    
    return df_merged

if __name__ == "__main__":
    run_data_ingestion()
