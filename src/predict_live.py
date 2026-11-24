"""
Live Prediction Module: Indian Gold Market Terminal
Focus: Real-time inference on Indian Domestic Gold (MCX / NSE Gold BeES / Retail 10g)
"""

import sys
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

def load_indian_model():
    """Load the trained Indian ensemble model artifact."""
    model_path = config.MODELS_DIR / "indian_gold_ensemble.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Run Phase 3 first.")
    artifact = joblib.load(model_path)
    return artifact["pipeline"], artifact["features"], artifact

def generate_indian_live_forecast(refresh_data: bool = False):
    """Run real-time prediction for Indian domestic gold."""
    if refresh_data:
        import importlib
        ingest_mod = importlib.import_module("src.01_data_ingestion")
        fe_mod = importlib.import_module("src.02_feature_engineering")
        print("Fetching latest Indian market data (GOLDBEES, USD/INR, Nifty 50)...")
        ingest_mod.run_data_ingestion()
        fe_mod.run_feature_engineering()

    features_path = config.PROCESSED_DATA_DIR / "indian_gold_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Features matrix not found at {features_path}.")
        
    df = pd.read_csv(features_path, parse_dates=["Date"]).set_index("Date")
    latest_row = df.iloc[-1]
    latest_date = df.index[-1].strftime("%b %d, %Y")
    
    current_24k_10g = float(latest_row["Indian_Gold_24K_10g"])
    current_22k_10g = float(latest_row["Indian_Gold_22K_10g"])
    current_goldbees = float(latest_row.get("GOLDBEES_Close", current_24k_10g / 750.0))
    usdinr_rate = float(latest_row["USDINR_Close"])
    nifty_close = float(latest_row.get("NIFTY50_Close", 25000))
    
    # Load model
    pipeline, feature_names, metadata = load_indian_model()
    
    # Prepare feature row
    X_latest = pd.DataFrame([latest_row[feature_names]], columns=feature_names)
    probs = pipeline.predict_proba(X_latest)[0]
    prob_down = probs[0] * 100
    prob_up = probs[1] * 100
    
    is_up = prob_up >= prob_down
    direction = "UP 📈" if is_up else "DOWN 📉"
    confidence = prob_up if is_up else prob_down
    
    # Signal Logic
    if prob_up >= 60.0:
        signal = "STRONG BUY (LONG MCX / GOLDBEES)"
        recommendation = "Enter Long on MCX Gold Mini or accumulate NSE Gold BeES. Target +2%, Stop Loss -1%."
    elif prob_down >= 60.0:
        signal = "STRONG SELL (SHORT MCX / HEDGE)"
        recommendation = "Hedge physical inventory or Short MCX Gold Futures. Target -2%, Stop Loss +1%."
    else:
        signal = "NEUTRAL CHOP (NO TRADE ⏸️)"
        recommendation = "Low conviction. Avoid intraday noise; wait for high-probability domestic breakout."
        
    # Calculate 1-Day (Tomorrow T+1) Expected Price & Tightened Range
    raw_daily_vol_pct = float(latest_row["Indian_Gold_Vol_10d"]) / np.sqrt(252) if "Indian_Gold_Vol_10d" in latest_row else 0.012
    # Dampen outlier multi-day volatility spikes into realistic single-day Indian range (₹500 to ₹1,250 band)
    calibrated_vol_inr = float(np.clip(current_24k_10g * (raw_daily_vol_pct * 0.32), config.VOLATILITY_BAND_MIN_INR, config.VOLATILITY_BAND_MAX_INR))
    calibrated_vol_pct = calibrated_vol_inr / current_24k_10g

    conviction_weight = (confidence - 50.0) / 50.0
    expected_1d_pct = conviction_weight * (calibrated_vol_pct * 0.65)
    if not is_up:
        expected_1d_pct = -abs(expected_1d_pct)
    else:
        expected_1d_pct = abs(expected_1d_pct)
        
    expected_24k_10g = current_24k_10g * (1.0 + expected_1d_pct)
    expected_22k_10g = current_22k_10g * (1.0 + expected_1d_pct)
    diff_24k = expected_24k_10g - current_24k_10g
    pct_diff_1d = (diff_24k / current_24k_10g) * 100
    expected_high_24k = current_24k_10g + calibrated_vol_inr
    expected_low_24k = current_24k_10g - calibrated_vol_inr
    expected_goldbees = current_goldbees * (1.0 + expected_1d_pct)

    print("\n" + "=" * 68)
    print("        GOLDGROW: INDIAN GOLD FORECAST DASHBOARD (₹)")
    print("=" * 68)
    print(f"  Current Trading Date   : {latest_date}")
    print(f"  Current 24K Pure (10g) : ₹{int(current_24k_10g):,}")
    print(f"  Current 22K Hall (10g) : ₹{int(current_22k_10g):,}")
    print(f"  Current 24K (Per 1g)   : ₹{current_24k_10g/10.0:,.1f}")
    print(f"  NSE Gold BeES ETF Close: ₹{current_goldbees:.2f}")
    print(f"  USD / INR Exchange Rate: ₹{usdinr_rate:.2f} per $1 USD")
    print(f"  Nifty 50 Index Close   : {nifty_close:,.2f}")
    print("-" * 68)
    print("  >>> EXPECTED NEXT 1-DAY (TOMORROW T+1) PRICE PROJECTION <<<")
    print(f"  Expected 24K (10g) T+1 : ₹{int(expected_24k_10g):,} ({'+' if diff_24k >= 0 else ''}₹{int(diff_24k):,}, {'+' if pct_diff_1d >= 0 else ''}{pct_diff_1d:.2f}%)")
    print(f"  Expected 22K (10g) T+1 : ₹{int(expected_22k_10g):,} ({'+' if diff_24k >= 0 else ''}₹{int(diff_24k*config.PURITY_22K):,})")
    print(f"  Expected 24K (1g) T+1  : ₹{expected_24k_10g/10.0:,.1f}")
    print(f"  Expected Gold BeES T+1 : ₹{expected_goldbees:.2f} ({'+' if diff_24k >= 0 else ''}₹{expected_goldbees - current_goldbees:.2f})")
    print(f"  Tomorrow's Range (24K) : ₹{int(expected_low_24k):,} — ₹{int(expected_high_24k):,} (Band: ±₹{int(expected_high_24k - current_24k_10g):,})")
    print("-" * 68)
    print(f"  PREDICTED DIRECTION    : {direction}")
    print(f"  Model Confidence (UP)  : {prob_up:.2f}%")
    print(f"  Model Confidence (DOWN): {prob_down:.2f}%")
    print(f"  TRADE EXECUTION SIGNAL : {signal}")
    print(f"  RECOMMENDATION         : {recommendation}")
    print("-" * 68)
    print("  DOMESTIC & MACRO DRIVERS TODAY:")
    print(f"    • USD/INR 1-Day Change        : {latest_row['USDINR_Return_1d']*100:+.2f}%")
    print(f"    • Nifty 50 1-Day Return       : {latest_row['NIFTY_Return_1d']*100:+.2f}%")
    print(f"    • 14-Day Domestic Gold RSI    : {latest_row['RSI_14']:.2f}")
    print(f"    • Domestic 10-Day Volatility  : {latest_row['Indian_Gold_Vol_10d']*100:.2f}%")
    print(f"    • US 10Y Real TIPS Yield      : {latest_row.get('10Y_Real_Rate', 2.88):.2f}%")
    print(f"    • Effective Import Duty + GST : ~9.18% (6% Customs Duty + 3% GST)")
    print("=" * 68 + "\n")
    
    return {
        "date": latest_date,
        "rate_24k_10g": int(current_24k_10g),
        "rate_22k_10g": int(current_22k_10g),
        "goldbees": round(current_goldbees, 2),
        "usdinr": round(usdinr_rate, 2),
        "prob_up": round(prob_up, 1),
        "prob_down": round(prob_down, 1),
        "signal": signal,
        "recommendation": recommendation
    }

if __name__ == "__main__":
    generate_indian_live_forecast(refresh_data=False)
