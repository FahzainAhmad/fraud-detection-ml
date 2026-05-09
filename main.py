"""
main.py
=======
LD7187: Machine Learning on Cloud — Group Project
Fraud Detection Pipeline

Main entry file. Runs the full pipeline in this order:
    1. Load data
    2. EDA (Task 2)
    3. Pre-processing (Task 3)
    4. Model training (Task 4)
    5. Evaluation (Task 5)

Usage
-----
    python main.py

Outputs
-------
    All plots and the deployment output are saved to ./outputs/
"""

import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")

warnings.filterwarnings("ignore")
np.random.seed(42)

from config import OUTPUT_DIR
from data_loader import load_data
from eda import run_eda
from preprocessing import preprocess
from models import train_all_models
from evaluation import evaluate_all

import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def main():
    print("=" * 55)
    print("  LD7187 — Fraud Detection Pipeline")
    print("=" * 55)

    # Step 1: load data
    df = load_data()

    # Step 2: run EDA
    run_eda(df)

    # Step 3: preprocess data
    X_train, y_train, X_test, y_test, feature_names = preprocess(df)

    # XGBoost uses this ratio in scale_pos_weight
    # it is taken from the original data before SMOTE
    df_temp = df.copy()
    fraud_ratio = (df_temp["Class"] == 0).sum() / (df_temp["Class"] == 1).sum()

    # Step 4: train models
    results = train_all_models(X_train, y_train, fraud_ratio)

    # Step 5: evaluate models
    metrics_df, thresh_df = evaluate_all(results, X_test, y_test)

    # Done
    print("\n" + "=" * 55)
    print("  PIPELINE COMPLETE")
    print("=" * 55)
    print(f"\nAll outputs saved to ./{OUTPUT_DIR}/")
    print("\nFiles generated:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
