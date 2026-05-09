"""
data_loader.py
==============
Loads `creditcard.csv` and prints a quick summary.
"""

import pandas as pd
from config import DATA_PATH


def load_data() -> pd.DataFrame:
    """
    Load the ULB credit card fraud dataset.

    Returns
    -------
    pd.DataFrame
        Raw data with all original columns unchanged.
    """
    print("\n=== Loading Data ===")
    df = pd.read_csv(DATA_PATH)

    print(f"Shape          : {df.shape}")
    print(f"Missing values : {df.isnull().sum().sum()}")
    print(f"Duplicates     : {df.duplicated().sum()}")
    print(f"\nClass distribution:\n{df['Class'].value_counts()}")

    fraud_rate = df["Class"].mean() * 100
    print(f"\nFraud rate     : {fraud_rate:.4f}%")
    print(f"  → {df['Class'].sum():,} fraudulent transactions out of {len(df):,} total")

    return df
