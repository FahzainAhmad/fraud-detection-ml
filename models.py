"""
models.py
=========
Task 4: Model training and selection.
Trains four models with RandomizedSearchCV tuning,
then saves per-model CV plots plus one combined comparison plot.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

from config import (
    OUTPUT_DIR, RANDOM_STATE, CV_FOLDS,
    LR_N_ITER, RF_N_ITER, XGB_N_ITER, NN_N_ITER,
    PALETTE
)


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def train_all_models(X_train, y_train, fraud_ratio: float) -> dict:
    """
    Train Logistic Regression, Random Forest, XGBoost, and Neural Network
    using RandomizedSearchCV with StratifiedKFold cross-validation.

    Parameters
    ----------
    X_train      : SMOTE-resampled training features
    y_train      : SMOTE-resampled training labels
    fraud_ratio  : ratio of legitimate to fraud (used for XGBoost scale_pos_weight)

    Returns
    -------
    dict : {model_name: fitted_best_estimator}
    """
    print("\n=== Task 4: Model Training ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results      = {}
    cv_scores_all = {}

    results["Logistic Regression"], cv_scores_all["Logistic Regression"] = \
        _train_logistic_regression(X_train, y_train, cv)

    results["Random Forest"], cv_scores_all["Random Forest"] = \
        _train_random_forest(X_train, y_train, cv)

    results["XGBoost"], cv_scores_all["XGBoost"] = \
        _train_xgboost(X_train, y_train, cv, fraud_ratio)

    results["Neural Network"], cv_scores_all["Neural Network"] = \
        _train_neural_network(X_train, y_train, cv)

    _plot_cv_comparison(cv_scores_all)

    return results


# Individual model training helpers

def _plot_cv_scores(name: str, scores: np.ndarray):
    """Bar chart of PR-AUC by fold for one model."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, CV_FOLDS + 1), scores,
           color=PALETTE[name], edgecolor="black", alpha=0.85)
    ax.axhline(scores.mean(), color="red", linestyle="--",
               label=f"Mean = {scores.mean():.4f}")
    ax.set_title(f"Cross-Validation PR-AUC — {name}")
    ax.set_xlabel("Fold")
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend()
    safe = name.replace(" ", "_").lower()
    _save(fig, f"cv_{safe}.png")


def _train_logistic_regression(X_train, y_train, cv):
    print("\n  Training Logistic Regression...")
    search = RandomizedSearchCV(
        LogisticRegression(
            class_weight="balanced", random_state=RANDOM_STATE, max_iter=2000
        ),
        param_distributions={"C": [0.001, 0.01, 0.1, 1, 10]},
        n_iter=LR_N_ITER, cv=cv,
        scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_val_score(
        best, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1
    )
    _plot_cv_scores("Logistic Regression", scores)
    print(f"    Best C        : {search.best_params_['C']}")
    print(f"    CV PR-AUC     : {scores.mean():.4f} +/- {scores.std():.4f}")
    return best, scores


def _train_random_forest(X_train, y_train, cv):
    print("\n  Training Random Forest...")
    param_grid = {
        "n_estimators":    [100, 200],
        "max_depth":       [10, 20],
        "min_samples_split": [2, 5],
        "class_weight":    ["balanced"]
    }
    search = RandomizedSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE),
        param_grid, n_iter=RF_N_ITER, cv=cv,
        scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_val_score(
        best, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1
    )
    _plot_cv_scores("Random Forest", scores)
    print(f"    Best params   : {search.best_params_}")
    print(f"    CV PR-AUC     : {scores.mean():.4f} +/- {scores.std():.4f}")
    return best, scores


def _train_xgboost(X_train, y_train, cv, fraud_ratio):
    print("\n  Training XGBoost...")
    param_grid = {
        "n_estimators":    [100, 200, 300],
        "max_depth":       [3, 5, 7],
        "learning_rate":   [0.05, 0.1, 0.2],
        "subsample":       [0.8, 1.0],
        "colsample_bytree":[0.8, 1.0]
    }
    search = RandomizedSearchCV(
        xgb.XGBClassifier(
            scale_pos_weight=fraud_ratio,
            use_label_encoder=False,
            eval_metric="aucpr",
            random_state=RANDOM_STATE
        ),
        param_grid, n_iter=XGB_N_ITER, cv=cv,
        scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_val_score(
        best, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1
    )
    _plot_cv_scores("XGBoost", scores)
    print(f"    Best params   : {search.best_params_}")
    print(f"    CV PR-AUC     : {scores.mean():.4f} +/- {scores.std():.4f}")
    return best, scores


def _train_neural_network(X_train, y_train, cv):
    print("\n  Training Neural Network...")
    param_grid = {
        "hidden_layer_sizes": [(64, 32), (128, 64), (128, 64, 32)],
        "activation":         ["relu", "tanh"],
        "alpha":              [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01]
    }
    search = RandomizedSearchCV(
        MLPClassifier(max_iter=300, early_stopping=True, random_state=RANDOM_STATE),
        param_grid, n_iter=NN_N_ITER, cv=cv,
        scoring="average_precision", n_jobs=-1, random_state=RANDOM_STATE
    )
    search.fit(X_train, y_train)
    best = search.best_estimator_

    scores = cross_val_score(
        best, X_train, y_train,
        cv=cv, scoring="average_precision", n_jobs=-1
    )
    _plot_cv_scores("Neural Network", scores)
    print(f"    Best params   : {search.best_params_}")
    print(f"    CV PR-AUC     : {scores.mean():.4f} +/- {scores.std():.4f}")
    return best, scores


def _plot_cv_comparison(cv_scores_all: dict):
    """Grouped bar chart to compare PR-AUC across all models by fold."""
    cv_df = pd.DataFrame(cv_scores_all)
    cv_df.index = [f"Fold {i+1}" for i in range(CV_FOLDS)]

    fig, ax = plt.subplots(figsize=(9, 5))
    cv_df.plot(kind="bar", ax=ax, edgecolor="black", rot=0,
               color=list(PALETTE.values()))
    ax.set_title("Cross-Validation PR-AUC per Fold — All Models", fontsize=13)
    ax.set_ylabel("PR-AUC")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    _save(fig, "07_cv_comparison_all_models.png")
