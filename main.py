"""
GoldGrow: Master Pipeline Orchestrator
CLI Usage:
    python main.py --ingest    # Phase 1: Ingest live data from Tiingo & FRED
    python main.py --features  # Phase 2: Compute technical & macro features
    python main.py --train     # Phase 3: Train high-accuracy ensemble model
    python main.py --predict   # Live inference on current market close
    python main.py --all       # End-to-end pipeline run
"""

import sys
import argparse
import importlib

def main():
    parser = argparse.ArgumentParser(description="GoldGrow Machine Learning Pipeline")
    parser.add_argument("--ingest", action="store_true", help="Run Phase 1: Data Ingestion (Tiingo + FRED)")
    parser.add_argument("--features", action="store_true", help="Run Phase 2: Feature Engineering")
    parser.add_argument("--train", action="store_true", help="Run Phase 3: Ensemble Model Training")
    parser.add_argument("--predict", action="store_true", help="Run Live Forecast on latest market data")
    parser.add_argument("--serve", action="store_true", help="Launch interactive Web UI dashboard on http://localhost:8000")
    parser.add_argument("--all", action="store_true", help="Run complete pipeline end-to-end")
    
    args = parser.parse_args()
    
    # Default behavior if no flag passed
    if not any([args.ingest, args.features, args.train, args.predict, args.serve, args.all]):
        print("No flag specified. Defaulting to live forecast: python main.py --predict")
        args.predict = True

    if args.all or args.ingest:
        ingest_mod = importlib.import_module("src.01_data_ingestion")
        ingest_mod.run_data_ingestion()
        
    if args.all or args.features:
        fe_mod = importlib.import_module("src.02_feature_engineering")
        fe_mod.run_feature_engineering()
        
    if args.all or args.train:
        train_mod = importlib.import_module("src.03_model_training")
        train_mod.run_model_training()
        
    if args.all or args.predict:
        predict_mod = importlib.import_module("src.predict_live")
        predict_mod.generate_live_forecast(refresh_data=False)

    if args.serve:
        import uvicorn
        print("Starting AURUM Forecaster Web Dashboard on http://localhost:8000...")
        uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

if __name__ == "__main__":
    main()
