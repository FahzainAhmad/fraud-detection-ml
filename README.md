# Fraud Detection Project (LD7187)

## Description
This project builds a full machine learning pipeline to spot fraud in credit card transactions.

In simple terms, it:
- reads the transaction data
- explores patterns in normal vs fraud transactions
- prepares the data for machine learning
- trains multiple models
- compares model performance with useful charts and metrics

The goal is to find fraud as accurately as possible, especially in a highly unbalanced dataset where fraud cases are very rare.

## What This Project Does
- Loads the `creditcard.csv` dataset
- Runs EDA (charts and quick insights)
- Creates extra features from raw columns
- Handles class imbalance using SMOTE
- Trains 4 models:
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Neural Network (MLP)
- Evaluates each model using:
  - confusion matrix
  - ROC and PR curves
  - threshold tuning
  - feature importance and SHAP plots

## Project Structure
```
fraud-detection-ml/
├── main.py
├── config.py
├── data_loader.py
├── eda.py
├── preprocessing.py
├── models.py
├── evaluation.py
├── colab_deploy.ipynb
├── requirements.txt
├── creditcard.csv
└── outputs/                # created after first run
```

## File Description
- `main.py` - Runs the full pipeline from start to finish
- `config.py` - Stores settings like paths, split size, CV folds, and tuning limits
- `data_loader.py` - Loads the dataset and prints basic dataset stats
- `eda.py` - Creates EDA charts (class balance, amount, heatmap, feature plots, timing)
- `preprocessing.py` - Feature engineering, train/test split, scaling, and SMOTE
- `models.py` - Trains and tunes all models with cross-validation
- `evaluation.py` - Final model evaluation charts and reports
- `colab_deploy.ipynb` - Run the same project in Google Colab
- `requirements.txt` - Python libraries needed for this project
- `outputs/` - Saved charts and result files

## Dataset
Source: [Kaggle ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Quick facts:
- 284,807 total transactions
- 492 fraud transactions
- fraud rate is around 0.17%

Columns include `Time`, `V1` to `V28`, `Amount`, and `Class`.

## Setup
Install dependencies:

```bash
pip install -r requirements.txt
```

## How To Run (Local)
Run the full project:

```bash
python main.py
```

After it finishes, check the `outputs/` folder for all generated charts.

## Main Outputs
You will get charts for:
- class imbalance
- amount comparison
- correlation heatmap
- feature distributions
- transaction timing
- SMOTE impact
- cross-validation comparison
- confusion matrices
- threshold tuning
- ROC and PR curves
- feature importance
- SHAP plots

## Run on Google Colab
Use `colab_deploy.ipynb` if you want to run this in the cloud.

Basic steps:
1. Push this repo to GitHub
2. Open `colab_deploy.ipynb` in Google Colab
3. Set your repo URL in the notebook
4. Run cells from top to bottom

## Notes
- This project is for academic learning and model comparison.
- Because fraud cases are rare, metrics like Recall, PR-AUC, and F1 are important.
