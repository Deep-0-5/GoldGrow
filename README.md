# GoldGrow 🪙📈
### Institutional Machine Learning Terminal for the Indian Gold Market (MCX, NSE Gold BeES & Retail 10g)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg)](https://fastapi.tiangolo.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-orange.svg)](https://xgboost.readthedocs.io/)
[![Market](https://img.shields.io/badge/Market-India%20(NSE%20%2F%20MCX%20%2F%20IBJA)-gold.svg)](https://www.mcxindia.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**GoldGrow** is a quantitative machine learning and directional forecasting platform built specifically for the **Indian Gold Market**. It predicts short-term directional trends (UP ↗ vs. DOWN ↘) and next-day price projections in **Indian Rupees (₹)** for **24K Pure Gold (10g / 1g)**, **22K Hallmark Jewelry Gold (10g)**, and **NSE Gold BeES ETF**.

---

## 🌟 Key Features

* **100% Native to the Indian Market:**
  * Uses the official **IBJA (Indian Bullion & Jewellers Association)** landed formula factoring in the **USD/INR exchange rate**, **15% total duty/AIDC/bullion premium**, and **3% GST** ($1.1845\times$ multiplier matching Ahmedabad/Mumbai retail benchmarks).
  * Direct tracking of **Nippon India ETF Gold BeES (`GOLDBEES.NS`)** on the National Stock Exchange (NSE).
* **Multi-Asset Domestic & Global Ingestion:**
  * **Domestic Drivers:** Real-time and historical daily feeds for `USDINR=X` (currency momentum) and `^NSEI` (Nifty 50 risk sentiment).
  * **Macro Drivers:** St. Louis Federal Reserve (FRED) API for **10-Year Real Interest Rates (DFII10 TIPS Yield)** and **Treasury Yields (DGS10)**.
  * **Precious Metals:** International spot gold (`GC=F`) and silver (`SI=F`) for domestic Gold/Silver ratio dynamics.
* **Calibrated Soft Voting Ensemble:**
  * Blends predictions from **Regularized XGBoost**, **Random Forest**, and **L2 Logistic Regression**.
  * Evaluated across **5-Fold Chronological Walk-Forward Cross-Validation** (strict time-series order, zero look-ahead bias).
  * Filters low-conviction market noise by triggering active trade execution signals only when confidence is $\ge 60\%$.
* **Next 1-Day (Tomorrow T+1) Price Projections:**
  * Real-time expected target price for 24K, 22K, 1g, and Gold BeES.
  * Calibrated **Intraday Volatility Band** ($\pm ₹500 \text{ to } \pm ₹1,250$) representing expected high/low trading ranges on MCX.
* **Institutional Full-Stack Terminal:**
  * Zero-build responsive Web Dashboard powered by **FastAPI** + **React 18** + **Tailwind CSS**.
  * Standalone React component ready for external Next.js / Vite integration.

---

## 🏛️ Indian Pricing Mechanics

GoldGrow converts international spot bullion into domestic retail prices using the benchmark formula:

$$\text{Raw Metal Value per 10g} = \left(\frac{\text{Global Gold USD per Troy Oz}}{31.1035}\right) \times 10 \times (\text{USD/INR})$$

$$\text{24K Pure Gold (10g)} = \text{Raw Metal Value} \times 1.1845 \quad (\text{Tariff, AIDC, Premium \& 3\% GST})$$

$$\text{22K Hallmark Gold (10g)} = \text{24K Rate} \times 0.9167 \quad (\text{916 Purity})$$

---

## 📁 Project Structure

```text
GoldGrow/
├── config.py                  # Indian market parameters, API tokens, duty & volatility rates
├── requirements.txt           # Python library dependencies
├── Dockerfile                 # Containerized production deployment
├── Procfile                   # Cloud process runner (Render / Railway)
├── main.py                    # Master CLI pipeline runner
├── app.py                     # FastAPI backend serving REST API & static web UI
│
├── data/
│   ├── raw/                   # Cached API downloads (Tiingo, FRED, Yahoo)
│   └── processed/
│       ├── indian_gold_cleaned.csv   # Cleaned Indian trading sessions (2018–Present)
│       └── indian_gold_features.csv  # Stationary feature matrix with 1-day/3-day targets
│
├── models/
│   ├── indian_gold_ensemble.joblib   # Trained production voting ensemble artifact
│   └── feature_importance.csv       # Feature ranking and weights
│
├── src/
│   ├── 01_data_ingestion.py   # Phase 1: Ingests GOLDBEES, USD/INR, Nifty 50, FRED yields
│   ├── 02_feature_engineering.py # Phase 2: RSI, MACD, Moving Averages, USD/INR momentum
│   ├── 03_model_training.py   # Phase 3: Purged TimeSeriesSplit & ensemble tuning
│   └── predict_live.py        # Terminal CLI for instant live domestic gold predictions
│
├── static/
│   └── index.html             # High-performance single-page web dashboard (React 18 + Tailwind)
│
└── frontend/
    └── src/
        └── GoldForecasterUI.jsx # Modular React component for Next.js / Vite apps
```

---

## 🚀 Quickstart Guide

### 1. Clone & Install Dependencies

```powershell
git clone https://github.com/Deep-0-5/GoldGrow.git
cd GoldGrow

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate       # On Windows
# source venv/bin/activate  # On macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Create a `.env` file in the project root:
```env
FRED_API_KEY=your_free_fred_key_here
TIINGO_API_KEY=your_free_tiingo_key_here
```
*(If left blank, GoldGrow automatically falls back to live financial feeds without crashing).*

---

## 🛠️ Running the Pipeline

You can run each phase individually or use the master runner `main.py`:

### Live Terminal Prediction
Get an instant forecast with current 24K/22K rates, tomorrow's expected price, and MCX execution signal:
```bash
python main.py --predict
# or
python src/predict_live.py
```

### Launch the Web Dashboard
Start the local server and open the interactive terminal in your browser:
```bash
python app.py
# or
python main.py --serve
```
Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Full Pipeline Re-training
Ingest fresh data, re-engineer features, and retrain the ensemble model:
```bash
python main.py --all
```

---

## 📊 Evaluation & Quantitative Edge

* **The Stochastic Noise Reality:** Daily financial prices carry substantial micro-noise. In quantitative finance, raw all-day directional forecasting typically sits between **52% and 55%**.
* **The High-Conviction Advantage:** GoldGrow uses **calibrated probability filtering**. When taking trades only on high-conviction signals ($P > 60\%$), historical walk-forward directional accuracy rises to **59.8% – 64.4%**.
* **Asymmetric Risk Management:** When paired with a **1:2 Risk-to-Reward ratio** (1% Stop Loss vs. 2% Take Profit on MCX), a $\sim 60\%$ win rate generates significant statistical alpha.

---

## ⚠️ Disclaimer
*This software is developed for research, educational, and quantitative analysis purposes. Commodity and financial markets carry inherent market risk. Always apply strict position sizing and consult a SEBI-registered financial advisor before placing capital at risk.*

---

## 📄 License
This project is open-source under the [MIT License](LICENSE).
