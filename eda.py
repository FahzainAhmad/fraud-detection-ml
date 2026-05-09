"""
eda.py
======
Task 2: Exploratory Data Analysis.
Builds EDA plots and saves them in the outputs folder.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from config import OUTPUT_DIR, PALETTE


def _save(fig, name: str):
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


def run_eda(df: pd.DataFrame):
    """
    Run full EDA on the raw dataframe.
    Produces 5 plots saved to OUTPUT_DIR.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset as returned by data_loader.load_data().
    """
    print("\n=== Task 2: Exploratory Data Analysis ===")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    _plot_class_distribution(df)
    _plot_amount_by_class(df)
    _plot_correlation_heatmap(df)
    _plot_feature_histograms(df)
    _plot_transaction_timing(df)

    print("EDA complete.\n")


# Individual plot helpers

def _plot_class_distribution(df):
    """Bar chart to show how unbalanced the classes are."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["Class"].value_counts()
    bars = ax.bar(["Legitimate (0)", "Fraud (1)"], counts.values,
                  color=["#4C72B0", "#DD8452"], edgecolor="black")
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1000,
                f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("Class Distribution — Severe Imbalance", fontsize=13)
    ax.set_ylabel("Number of Transactions")
    ax.set_yscale("log")
    _save(fig, "01_class_distribution.png")


def _plot_amount_by_class(df):
    """Two boxplots of amount by class (raw and log scale)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, log_scale, sfx in zip(axes, [False, True], ["Raw", "Log Scale"]):
        data = [df[df["Class"] == 0]["Amount"], df[df["Class"] == 1]["Amount"]]
        ax.boxplot(data, labels=["Legitimate", "Fraud"], showfliers=False)
        ax.set_title(f"Amount by Class — {sfx}")
        ax.set_ylabel("Amount (USD)")
        if log_scale:
            ax.set_yscale("log")
    fig.suptitle("Transaction Amount Distribution", fontsize=13)
    plt.tight_layout()
    _save(fig, "02_amount_by_class.png")

    print("Amount statistics by class:")
    print(df.groupby("Class")["Amount"].describe())


def _plot_correlation_heatmap(df):
    """Heatmap of the 10 features most linked to Class."""
    corr_series = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top_feats = corr_series.index[:10].tolist() + ["Class"]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[top_feats].corr(), annot=True, fmt=".2f",
                cmap="coolwarm", ax=ax, linewidths=0.5)
    ax.set_title("Correlation Heatmap — Top 10 Features vs Class")
    _save(fig, "03_correlation_heatmap.png")

    print("\nTop features correlated with Class:")
    print(corr_series.head(10))


def _plot_feature_histograms(df):
    """Overlay histograms for top 6 features, split by class."""
    corr_series = df.corr()["Class"].drop("Class").abs().sort_values(ascending=False)
    top6 = corr_series.index[:6].tolist()

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    for ax, feat in zip(axes.flatten(), top6):
        ax.hist(df[df["Class"] == 0][feat], bins=60, alpha=0.6,
                label="Legitimate", color="#4C72B0", density=True)
        ax.hist(df[df["Class"] == 1][feat], bins=60, alpha=0.6,
                label="Fraud", color="#DD8452", density=True)
        ax.set_title(feat)
        ax.legend(fontsize=8)
    fig.suptitle("Feature Distributions — Legitimate vs Fraud", fontsize=13)
    plt.tight_layout()
    _save(fig, "04_feature_histograms.png")


def _plot_transaction_timing(df):
    """Histograms by hour of day for each class."""
    hour = (df["Time"] % 86400) / 3600   # turn seconds into hour-of-day values

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, cls, label, color in zip(
        axes, [0, 1], ["Legitimate", "Fraud"], ["#4C72B0", "#DD8452"]
    ):
        ax.hist(hour[df["Class"] == cls], bins=24,
                color=color, edgecolor="black", alpha=0.85)
        ax.set_title(f"{label} Transactions by Hour of Day")
        ax.set_xlabel("Hour (0–24)")
        ax.set_ylabel("Count")
    fig.suptitle("Transaction Timing Pattern", fontsize=13)
    plt.tight_layout()
    _save(fig, "05_transaction_timing.png")
