"""
preprocessing.py
================
Task 3: Data preprocessing.
Handles feature engineering, scaling, train/test split, and SMOTE,
then makes a quick SMOTE vs no-SMOTE comparison chart.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score, precision_score, roc_auc_score
from imblearn.over_sampling import SMOTE

from config import (
    OUTPUT_DIR, RANDOM_STATE, TEST_SIZE,
    SCALE_COLS, PEAK_HOUR_CUTOFF
)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def preprocess(df: pd.DataFrame):
    """
    Full preprocessing pipeline.

    Steps
    -----
    1. Feature engineering (log_amount, hour, peak_hour)
    2. Drop original Time and Amount columns
    3. Stratified train/test split
    4. RobustScaler on amount-derived features
    5. Apply SMOTE on training set only (partial oversample, not full balance)
    6. SMOTE vs no-SMOTE comparison plot

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset from data_loader.

    Returns
    -------
    X_train_res, y_train_res : training data after SMOTE
    X_test_scaled, y_test    : scaled test data (no SMOTE on test set)
    feature_names            : list of feature column names
    """
    print("\n=== Task 3: Data Pre-processing ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Feature engineering
    # log_amount helps pull in the long right tail from big amount values
    df = df.copy()
    df["log_amount"] = np.log1p(df["Amount"])

    # hour converts the running Time counter into hour of day (0-24)
    df["hour"] = (df["Time"] % 86400) / 3600

    # peak_hour marks transactions from midnight to early morning
    df["peak_hour"] = (df["hour"] <= PEAK_HOUR_CUTOFF).astype(int)

    # Drop Time and Amount because we replaced them with engineered features
    df_model = df.drop(columns=["Time", "Amount", "hour"])

    X = df_model.drop(columns=["Class"])
    y = df_model["Class"]

    print(f"Features after engineering : {X.shape[1]}")
    print(f"  Added : log_amount, peak_hour")
    print(f"  Dropped : Time, Amount")

    # Train/test split
    # stratify=y keeps the tiny fraud ratio similar in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain size : {X_train.shape[0]}  |  Test size : {X_test.shape[0]}")
    print(f"Train fraud: {y_train.sum()}        |  Test fraud: {y_test.sum()}")

    # RobustScaler
    # Fit only on training data so no test information leaks in.
    # RobustScaler uses median and IQR, so it handles big outliers better.
    scaler = RobustScaler()
    X_train_scaled = X_train.copy()
    X_test_scaled  = X_test.copy()
    X_train_scaled[SCALE_COLS] = scaler.fit_transform(X_train[SCALE_COLS])
    X_test_scaled[SCALE_COLS]  = scaler.transform(X_test[SCALE_COLS])

    # SMOTE
    # Use it on training data only.
    # Using it on test data would leak information and fake better scores.
    # sampling_strategy=0.1 caps fraud at 10% of majority count (faster than full balance).
    smote = SMOTE(random_state=RANDOM_STATE, sampling_strategy=0.1)
    X_train_res, y_train_res = smote.fit_resample(X_train_scaled, y_train)

    print(f"\nAfter SMOTE:")
    print(f"  Train size : {X_train_res.shape[0]}")
    print(f"  Fraud      : {y_train_res.sum():,}")
    print(f"  Legitimate : {(y_train_res == 0).sum():,}")

    # SMOTE comparison plot
    _plot_smote_comparison(X_train_scaled, y_train,
                           X_train_res, y_train_res,
                           X_test_scaled, y_test)

    return X_train_res, y_train_res, X_test_scaled, y_test, list(X.columns)


# SMOTE comparison helpers

def _quick_rf(X_tr, y_tr, X_te, y_te):
    """Train a quick Random Forest and return main metrics."""
    m = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE
    )
    m.fit(X_tr, y_tr)
    y_p = m.predict(X_te)
    return {
        "F1":        round(f1_score(y_te, y_p), 4),
        "Recall":    round(recall_score(y_te, y_p), 4),
        "Precision": round(precision_score(y_te, y_p), 4),
        "ROC-AUC":   round(roc_auc_score(y_te, m.predict_proba(X_te)[:, 1]), 4)
    }


def _plot_smote_comparison(X_tr, y_tr, X_tr_res, y_tr_res, X_te, y_te):
    """Bar chart comparing Random Forest with and without SMOTE."""
    print("\nRunning SMOTE comparison experiment...")
    no_smote   = _quick_rf(X_tr,     y_tr,     X_te, y_te)
    with_smote = _quick_rf(X_tr_res, y_tr_res, X_te, y_te)

    print(f"  Without SMOTE : {no_smote}")
    print(f"  With SMOTE    : {with_smote}")

    cmp_df = pd.DataFrame({"Without SMOTE": no_smote, "With SMOTE": with_smote})
    fig, ax = plt.subplots(figsize=(8, 5))
    cmp_df.T.plot(kind="bar", ax=ax, edgecolor="black", rot=0,
                  color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"])
    ax.set_title("Impact of SMOTE on Random Forest Performance", fontsize=13)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    for container in ax.containers:
        ax.bar_label(container, fmt="%.3f", fontsize=7, padding=2)
    _save(fig, "06_smote_vs_nosmote.png")
