"""
Indian Market Configuration
Native Asset: Gold on Indian Exchanges (NSE: GOLDBEES.NS, MCX Gold / Retail 10g)
Currency: Indian Rupee (INR - ₹)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# API Keys
FRED_API_KEY = os.getenv("FRED_API_KEY", "f783c1f8dbd709e210b6d4a563906a54")
TIINGO_API_KEY = os.getenv("TIINGO_API_KEY", "d622d7e86d2e2758c285c98cb0e0b4aa0839804d")

# Indian Market Specific Settings
MARKET_NAME = "India (NSE / MCX / Retail Bullion)"
CURRENCY = "INR"
CURRENCY_SYMBOL = "₹"
GOLD_PRIMARY_UNIT = "10 Grams"       # Standard Indian retail & MCX trade unit
TROY_OUNCE_TO_GRAMS = 31.1035

# Indian Tax, Tariff & Retail Structure (IBJA / City Benchmark Rates)
# Total Landed Retail Multiplier: Basic Customs Duty + AIDC + Bullion Premium (~15%) + 3% GST
INDIAN_CUSTOMS_DUTY_PCT = 0.15       # 15% Total Duty, AIDC Cess & Landed Premium
INDIAN_GST_PCT = 0.03                # 3.0% GST on Gold
INDIAN_RETAIL_MULTIPLIER = 1.1845    # Exact IBJA / Retail City (Ahmedabad/Mumbai/Delhi) Multiplier

# Karat Purity Standards
PURITY_24K = 0.999                   # 99.9% Pure Gold (Bullion / Coins)
PURITY_22K = 0.9167                  # 91.67% Hallmark Jewelry (916)
PURITY_18K = 0.750                   # 75.0% Diamond Jewelry

# Start Date for Time-Series History
START_DATE = "2018-01-01"

# Primary Indian Market Tickers & Proxies
INDIAN_TICKERS = {
    "GOLDBEES": "GOLDBEES.NS",       # Nippon India ETF Gold BeES (NSE-listed Gold Benchmark)
    "USD_INR": "USDINR=X",           # USD to Indian Rupee Exchange Rate (Core Domestic Multiplier)
    "NIFTY50": "^NSEI",              # Nifty 50 Index (Indian Equity Risk Sentiment)
    "GLOBAL_GOLD": "GC=F",           # International Spot Gold Continuous Futures
    "GLOBAL_SILVER": "SI=F"          # International Silver Futures (To compute Indian Gold/Silver ratio)
}

# FRED Economic Data Series
FRED_SERIES = {
    "DFII10": "10Y_Real_Rate",       # 10-Year Real TIPS Yield
    "DGS10": "10Y_Nominal_Yield",    # 10-Year Nominal Treasury Yield
}

# Indian Market Volatility Calibration
# Typical daily retail/MCX gold fluctuation is ~0.6% - 1.0% (₹600 to ₹1,400 band)
DEFAULT_DAILY_VOLATILITY_PCT = 0.0065  # ~0.65% typical intraday swing
VOLATILITY_BAND_MIN_INR = 500.0        # Minimum expected daily band (₹500)
VOLATILITY_BAND_MAX_INR = 1250.0       # Maximum expected daily band (₹1,250)

# Scenario A Risk-Reward Settings for Indian Trades (₹)
SL_PCT = 0.01                        # 1.0% Stop Loss
TP_PCT = 0.02                        # 2.0% Take Profit
MAX_HOLD_DAYS = 5                    # 5-day holding horizon
CONFIDENCE_THRESHOLD = 0.60          # Only execute trades with >=60% model confidence
