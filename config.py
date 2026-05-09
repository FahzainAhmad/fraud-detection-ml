"""
config.py
=========
Project settings for fraud detection.
I keep the main knobs here so they are easy to change.
"""

# Where files live
DATA_PATH  = "creditcard.csv"   # main dataset file
OUTPUT_DIR = "outputs"          # folder for charts and run results

# Fixed seed so results stay repeatable
RANDOM_STATE = 42

# Train/test split size
TEST_SIZE = 0.20       # keep 20% for testing, 80% for training

# Cross-validation setup
CV_FOLDS = 5           # number of folds used for validation

# How many random search tries per model
LR_N_ITER  = 5         # Logistic Regression
RF_N_ITER  = 6         # Random Forest
XGB_N_ITER = 10        # XGBoost
NN_N_ITER  = 8         # Neural Network

# How many rows to use for SHAP plots
SHAP_SAMPLE = 500

# Colors used in charts
PALETTE = {
    "Logistic Regression": "#4C72B0",
    "Random Forest":       "#55A868",
    "XGBoost":             "#DD8452",
    "Neural Network":      "#C44E52"
}

# Columns that need scaling
SCALE_COLS = ["log_amount"]

# Peak-hour cutoff
PEAK_HOUR_CUTOFF = 5   # from 12 AM to 5 AM counts as peak hours