"""
Phase 3: Indian Gold Market Ensemble Model Training & Walk-Forward CV
Focus: Directional Accuracy & Precision on Indian Domestic Gold Trades
Features:
- Indian Features: USD/INR, Nifty 50, Gold BeES, Gold/Silver Ratio in INR, Domestic RSI & Volatility
- Soft Voting Ensemble combining:
    1. Tuned XGBoost Classifier
    2. Regularized Random Forest Classifier
    3. Regularized Logistic Regression (L2)
- Evaluated on Indian Market sessions across 5 Chronological Walk-Forward Folds
- Exports production pipeline to models/indian_gold_ensemble.joblib
"""

import sys
from pathlib import Path
import json
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, confusion_matrix
import xgboost as xgb

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

# Curated High-Signal Features for Indian Gold Market
INDIAN_CURATED_FEATURES = [
    "USDINR_Return_1d",          # Daily change in USD/INR (Core Domestic Multiplier)
    "USDINR_Return_5d",          # Weekly trend in USD/INR
    "Indian_Gold_Return_Lag1",   # Yesterday's domestic gold return
    "Indian_Gold_Return_5d",     # 5-day domestic gold trend
    "Indian_Gold_Return_10d",    # 10-day domestic gold momentum
    "GOLDBEES_Return_1d",        # 1-day return of NSE Gold BeES ETF
    "NIFTY_Return_1d",           # Indian equity risk appetite (Nifty 50)
    "Gold_to_Nifty_Ratio",       # Indian Gold vs Equity ratio
    "Gold_Silver_Ratio_INR",     # Domestic precious metals spread
    "GSR_Change_5d",             # Weekly change in domestic Gold/Silver ratio
    "Real_Rate_Change_1d",       # US 10Y Real TIPS Yield daily change
    "RSI_14",                    # 14-day domestic RSI oscillator
    "Indian_Gold_Vol_10d",       # 10-day annualized domestic volatility
    "SMA_Ratio_5_20",            # 5-day vs 20-day trend ratio
]

def evaluate_predictions(y_true, y_pred, y_probs, model_name="Model"):
    """Compute detailed accuracy and precision metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec_up = precision_score(y_true, y_pred, zero_division=0)
    prec_down = precision_score(1 - y_true, 1 - y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_probs)
    
    # Confidence threshold metrics (>55% and >60%)
    high_conf_55 = (y_probs >= 0.55) | (y_probs <= 0.45)
    acc_55 = accuracy_score(y_true[high_conf_55], (y_probs[high_conf_55] >= 0.5).astype(int)) if high_conf_55.sum() > 0 else 0
    cov_55 = (high_conf_55.sum() / len(y_true)) * 100
    
    high_conf_60 = (y_probs >= 0.60) | (y_probs <= 0.40)
    acc_60 = accuracy_score(y_true[high_conf_60], (y_probs[high_conf_60] >= 0.5).astype(int)) if high_conf_60.sum() > 0 else 0
    cov_60 = (high_conf_60.sum() / len(y_true)) * 100

    return {
        "Model": model_name,
        "Accuracy_All": acc * 100,
        "Precision_UP": prec_up * 100,
        "Precision_DOWN": prec_down * 100,
        "ROC_AUC": roc_auc,
        "Acc_Conf_55": acc_55 * 100,
        "Coverage_55": cov_55,
        "Acc_Conf_60": acc_60 * 100,
        "Coverage_60": cov_60,
    }

def create_indian_ensemble_pipeline():
    """Create calibrated Soft Voting Ensemble for Indian Market."""
    clf_lr = LogisticRegression(max_iter=1000, C=0.08, random_state=42)
    clf_rf = RandomForestClassifier(n_estimators=160, max_depth=3, min_samples_leaf=20, random_state=42, n_jobs=-1)
    clf_xgb = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=2,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.6,
        reg_lambda=1.8,
        eval_metric="logloss",
        random_state=42
    )
    
    voting_ensemble = VotingClassifier(
        estimators=[
            ("lr", clf_lr),
            ("rf", clf_rf),
            ("xgb", clf_xgb)
        ],
        voting="soft",
        weights=[1.0, 1.2, 1.5]
    )
    
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("ensemble", voting_ensemble)
    ])
    
    return pipeline

def benchmark_indian_horizons(df: pd.DataFrame, n_splits: int = 5):
    """Benchmark 1-Day vs 3-Day Horizon on Indian domestic gold."""
    print(f"\nBenchmarking Indian 1-Day vs 3-Day Horizon via {n_splits}-Fold Walk-Forward Cross-Validation...")
    
    X = df[INDIAN_CURATED_FEATURES].copy()
    targets = {
        "Indian 1-Day Horizon (Target_Next_Day)": df["Target_Indian_Gold_Next_Day"].copy(),
        "Indian 3-Day Horizon (Target_3D)": df["Target_Indian_Gold_3D"].copy()
    }
    
    results = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for target_label, y in targets.items():
        fold_metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            pipeline = create_indian_ensemble_pipeline()
            pipeline.fit(X_train, y_train)
            
            probs = pipeline.predict_proba(X_test)[:, 1]
            preds = (probs >= 0.50).astype(int)
            
            m = evaluate_predictions(y_test, preds, probs, model_name=target_label)
            fold_metrics.append(m)
            
        df_fm = pd.DataFrame(fold_metrics)
        results.append({
            "Horizon Target": target_label,
            "CV_Accuracy_All (%)": df_fm["Accuracy_All"].mean(),
            "Precision_UP (%)": df_fm["Precision_UP"].mean(),
            "Precision_DOWN (%)": df_fm["Precision_DOWN"].mean(),
            "ROC_AUC": df_fm["ROC_AUC"].mean(),
            "Acc_Conf_55% (%)": df_fm["Acc_Conf_55"].mean(),
            "Coverage_55% (%)": df_fm["Coverage_55"].mean(),
            "Acc_Conf_60% (%)": df_fm["Acc_Conf_60"].mean(),
            "Coverage_60% (%)": df_fm["Coverage_60"].mean(),
        })
        
    return pd.DataFrame(results)

def train_and_export_indian_ensemble(df: pd.DataFrame):
    """Train production Ensemble on Indian domestic gold history and export."""
    print("\nTraining final production Ensemble on Indian 3-day gold trend...")
    X = df[INDIAN_CURATED_FEATURES].copy()
    y = df["Target_Indian_Gold_3D"].copy()
    
    # 80/20 chronological holdout test
    split_idx = int(len(X) * 0.80)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    pipeline = create_indian_ensemble_pipeline()
    pipeline.fit(X_train, y_train)
    
    probs = pipeline.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.50).astype(int)
    
    test_metrics = evaluate_predictions(y_test, preds, probs, model_name="Indian 3D Ensemble Holdout")
    
    # Retrain on 100% of data for live inference
    final_pipeline = create_indian_ensemble_pipeline()
    final_pipeline.fit(X, y)
    
    # Save the pipeline artifact
    artifact_path = config.MODELS_DIR / "indian_gold_ensemble.joblib"
    artifact_metadata = {
        "pipeline": final_pipeline,
        "features": INDIAN_CURATED_FEATURES,
        "target": "Target_Indian_Gold_3D",
        "market": "Indian Domestic (NSE/MCX/Retail)",
        "last_trained_date": df.index.max().strftime("%Y-%m-%d"),
        "total_training_records": len(df)
    }
    joblib.dump(artifact_metadata, artifact_path)
    print(f"Indian production model successfully exported to: {artifact_path}")
    
    return test_metrics

def run_model_training():
    """Main runner for Phase 3 Indian model training."""
    print("=" * 65)
    print("PHASE 3: INDIAN GOLD ENSEMBLE TRAINING & HORIZON BENCHMARK")
    print("=" * 65)
    
    features_path = config.PROCESSED_DATA_DIR / "indian_gold_features.csv"
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}. Run Phase 2 first.")
        
    df = pd.read_csv(features_path, parse_dates=["Date"]).set_index("Date")
    print(f"Indian Dataset Loaded: {len(df)} trading sessions from {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    print(f"Curated Domestic Features: {len(INDIAN_CURATED_FEATURES)} features.")
    
    # 1. Compare 1-Day vs 3-Day Horizon
    horizon_comparison = benchmark_indian_horizons(df, n_splits=5)
    print("\n" + "="*65)
    print("INDIAN HORIZON BENCHMARK (5-FOLD WALK-FORWARD CV):")
    print("="*65)
    print(horizon_comparison.to_string(index=False))
    
    # 2. Train and Export Production Ensemble
    holdout_metrics = train_and_export_indian_ensemble(df)
    print("\n" + "="*65)
    print("UNSEEN HOLDOUT TEST SET PERFORMANCE (Most Recent 20% Period):")
    print("="*65)
    for k, v in holdout_metrics.items():
        if isinstance(v, float):
            print(f"  - {k:<20}: {v:.2f}")
        else:
            print(f"  - {k:<20}: {v}")
    print("=" * 65)
    
    return horizon_comparison, holdout_metrics

if __name__ == "__main__":
    run_model_training()
