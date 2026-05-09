# Fraud Detection — LD7187 Machine Learning on Cloud

Group project for module LD7187 at Northumbria University.  
Builds and evaluates a machine learning pipeline to detect fraudulent credit card transactions.

---

## Project Structure

```
fraud_detection/
├── main.py             # Start here
├── config.py           # Project settings (paths, tuning values, colors)
├── data_loader.py      # Load the dataset and print a quick check
├── eda.py              # Task 2: EDA charts
├── preprocessing.py    # Task 3: prep data, make features, run SMOTE
├── models.py           # Task 4: train models and compare CV scores
├── evaluation.py       # Task 5: metrics, plots, and SHAP
├── azure_deploy.py     # Azure ML deployment script
├── requirements.txt    # Python packages needed
└── outputs/            # Generated plots (created on first run)
```

---

## Dataset

ULB Credit Card Fraud Detection dataset from Kaggle.  
Download `creditcard.csv` and place it in this folder before running.

- Link: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions | 492 fraud cases (0.17% fraud rate)
- Features: Time, V1–V28 (PCA), Amount, Class

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Usage

```bash
python main.py
```

All output plots are saved to the `outputs/` folder.

---

## Models Trained

| Model | Technique |
|-------|-----------|
| Logistic Regression | Baseline linear model, class_weight=balanced |
| Random Forest | Ensemble, RandomizedSearchCV tuning |
| XGBoost | Gradient boosting, scale_pos_weight for imbalance |
| Neural Network | MLPClassifier, early stopping |

---

## Outputs Generated

| File | Description |
|------|-------------|
| 01_class_distribution.png | Class imbalance bar chart |
| 02_amount_by_class.png | Transaction amount boxplots |
| 03_correlation_heatmap.png | Top 10 feature correlations |
| 04_feature_histograms.png | Feature distributions by class |
| 05_transaction_timing.png | Fraud timing by hour of day |
| 06_smote_vs_nosmote.png | SMOTE impact comparison |
| 07_cv_comparison_all_models.png | CV PR-AUC across all models |
| cv_*.png | Per-model cross-validation scores |
| 08_confusion_*.png | Confusion matrix per model |
| 09_threshold_*.png | Threshold tuning per model |
| 10_roc_curves.png | ROC curves — all models |
| 11_pr_curves.png | Precision-Recall curves — all models |
| 12_feature_importance_rf.png | Random Forest feature importance |
| 13_shap_bar.png | SHAP bar chart — XGBoost |
| 14_shap_beeswarm.png | SHAP beeswarm plot — XGBoost |

---

## Cloud Deployment (Google Colab — Free)

See `colab_deploy.ipynb` for cloud execution on Google Colab (GCP infrastructure).

Steps:
1. Upload all `.py` files to a GitHub repo
2. Open `colab_deploy.ipynb` in Google Colab (colab.research.google.com)
3. Update the `GITHUB_REPO` variable in Cell 2 with your repo URL
4. Run all cells top to bottom
5. Upload `creditcard.csv` when prompted in Cell 3

No account needed beyond a free Google account. No cost.
