"""
evaluation.py
=============
Task 5: Model evaluation and plots.
Creates confusion matrices, ROC/PR curves, threshold plots,
feature importance, and SHAP visuals for trained models.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap

from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)

from config import OUTPUT_DIR, PALETTE, SHAP_SAMPLE


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def evaluate_all(results: dict, X_test, y_test):
    """
    Full evaluation pipeline for all trained models.

    Parameters
    ----------
    results : dict
        {model_name: fitted_estimator} from models.train_all_models()
    X_test  : scaled test features
    y_test  : test labels

    Returns
    -------
    metrics_df  : pd.DataFrame — summary metrics for all models
    thresh_df   : pd.DataFrame — threshold tuning results
    """
    print("\n=== Task 5: Model Evaluation ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    metrics_rows   = []
    threshold_rows = []

    for name, model in results.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred  = model.predict(X_test)

        # Outputs for this model
        _print_report(name, y_test, y_pred)
        _plot_confusion_matrix(name, y_test, y_pred)
        thresh_row = _plot_threshold_tuning(name, y_test, y_pred, y_proba)

        metrics_rows.append({
            "Model":     name,
            "Precision": round(precision_score(y_test, y_pred), 4),
            "Recall":    round(recall_score(y_test, y_pred), 4),
            "F1 Score":  round(f1_score(y_test, y_pred), 4),
            "ROC-AUC":   round(roc_auc_score(y_test, y_proba), 4),
            "PR-AUC":    round(average_precision_score(y_test, y_proba), 4)
        })
        threshold_rows.append(thresh_row)

    # Comparison plots across all models
    _plot_roc_curves(results, X_test, y_test)
    _plot_pr_curves(results, X_test, y_test)
    _plot_feature_importance(results["Random Forest"], X_test)
    _plot_shap(results["XGBoost"], X_test)

    # Final summary tables
    metrics_df = pd.DataFrame(metrics_rows)
    thresh_df  = pd.DataFrame(threshold_rows)

    print("\n=== Summary Metrics ===")
    print(metrics_df.to_string(index=False))
    print("\n=== Threshold Tuning Results ===")
    print(thresh_df.to_string(index=False))

    return metrics_df, thresh_df


# Per-model helper functions

def _print_report(name, y_test, y_pred):
    print(f"\n--- {name} ---")
    print(classification_report(y_test, y_pred,
                                 target_names=["Legitimate", "Fraud"]))


def _plot_confusion_matrix(name, y_test, y_pred):
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Pred Legit", "Pred Fraud"],
                yticklabels=["True Legit", "True Fraud"])
    ax.set_title(f"Confusion Matrix — {name}")
    safe = name.replace(" ", "_").lower()
    _save(fig, f"08_confusion_{safe}.png")


def _plot_threshold_tuning(name, y_test, y_pred, y_proba) -> dict:
    """
    Plot F1, Precision, Recall vs classification threshold.
    Finds the threshold where F1 is highest.
    """
    prec_arr, rec_arr, thresholds = precision_recall_curve(y_test, y_proba)
    f1_arr = np.where(
        (prec_arr + rec_arr) == 0, 0,
        2 * prec_arr * rec_arr / (prec_arr + rec_arr)
    )
    best_idx    = np.argmax(f1_arr[:-1])
    best_thresh = thresholds[best_idx]
    best_f1     = f1_arr[best_idx]
    default_f1  = f1_score(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(thresholds, f1_arr[:-1],   color=PALETTE[name], label="F1 Score")
    ax.plot(thresholds, prec_arr[:-1], color="green",  linestyle="--", label="Precision")
    ax.plot(thresholds, rec_arr[:-1],  color="orange", linestyle="--", label="Recall")
    ax.axvline(best_thresh, color="red", linestyle=":",
               label=f"Optimal threshold = {best_thresh:.3f}")
    ax.set_title(f"Threshold Tuning — {name}")
    ax.set_xlabel("Classification Threshold")
    ax.set_ylabel("Score")
    ax.legend()
    safe = name.replace(" ", "_").lower()
    _save(fig, f"09_threshold_{safe}.png")

    return {
        "Model":             name,
        "Default Threshold": 0.50,
        "Optimal Threshold": round(float(best_thresh), 4),
        "F1 @ Default":      round(default_f1, 4),
        "F1 @ Optimal":      round(best_f1, 4)
    }


# Combined comparison plots

def _plot_roc_curves(results, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in results.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})", color=PALETTE[name])
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Models")
    ax.legend()
    _save(fig, "10_roc_curves.png")


def _plot_pr_curves(results, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, model in results.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        ap = average_precision_score(y_test, y_proba)
        ax.plot(rec, prec, label=f"{name} (AP = {ap:.3f})", color=PALETTE[name])
    ax.axhline(y_test.mean(), color="gray", linestyle="--",
               label=f"Random baseline ({y_test.mean():.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — All Models")
    ax.legend()
    _save(fig, "11_pr_curves.png")


def _plot_feature_importance(rf_model, X_test):
    importances = pd.Series(
        rf_model.feature_importances_, index=X_test.columns
    ).sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 5))
    importances.plot(kind="bar", ax=ax, color="#55A868", edgecolor="black")
    ax.set_title("Top 15 Feature Importances — Random Forest")
    ax.set_ylabel("Importance Score")
    plt.xticks(rotation=45, ha="right")
    _save(fig, "12_feature_importance_rf.png")


def _plot_shap(xgb_model, X_test):
    print("\n  Generating SHAP values for XGBoost (this may take a minute)...")
    sample      = X_test.iloc[:SHAP_SAMPLE]
    explainer   = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(sample)

    # Bar chart: average absolute SHAP value per feature
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance — XGBoost")
    _save(fig, "13_shap_bar.png")

    # Beeswarm: shows direction and size of each feature effect
    fig = plt.figure(figsize=(9, 6))
    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Beeswarm Plot — XGBoost")
    _save(fig, "14_shap_beeswarm.png")
