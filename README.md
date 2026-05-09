# Fraud Detection Project (LD7187)

## Description
This project tries to detect fake credit card transactions (fraud) using machine learning.

The dataset has a big challenge: fraud cases are very few compared to normal transactions.
So this project is built to handle that imbalance and still find fraud as well as possible.

In plain words, the project:
- loads transaction data
- studies patterns with charts
- prepares the data for model training
- trains multiple models
- compares the models and shows which one performs better

## Why This Project Matters
In real life, missing a fraud transaction can cost money.
At the same time, flagging too many normal transactions as fraud is also a problem.

So we do not look at only one metric.
We check precision, recall, F1, ROC-AUC, and PR-AUC to get a fair view.

## What Happens Step by Step
1. Read `creditcard.csv`
2. Print basic data checks (missing values, duplicates, fraud rate)
3. Create EDA plots to understand the data
4. Build extra features from existing columns
5. Split data into train and test
6. Scale important columns
7. Apply SMOTE on training data only
8. Train 4 models with hyperparameter tuning
9. Evaluate all models with charts and metric tables
10. Save all outputs in `outputs/`

## Project Structure
```text
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
└── outputs/                 # created after first run
```

## File-by-File Guide (Use + Description)

### `main.py`
**What it is:** Main runner script.  
**What it does:** Runs the full pipeline in order (load -> EDA -> preprocess -> train -> evaluate).  
**When to use:** Use this file when you want to run everything locally in one command.  
**Output:** Prints progress in terminal and saves all charts/results to `outputs/`.

### `config.py`
**What it is:** Central settings file.  
**What it does:** Stores paths, random seed, split ratio, cross-validation folds, tuning iteration counts, plot colors, and feature settings.  
**When to use:** Edit this file when you want to change project behavior without touching core logic.

### `data_loader.py`
**What it is:** Data loading module.  
**What it does:** Reads `creditcard.csv` and shows quick checks (shape, missing values, duplicates, class counts, fraud rate).  
**When to use:** Used automatically by `main.py`, but useful if you only want to test data loading.

### `eda.py`
**What it is:** Exploratory Data Analysis module.  
**What it does:** Creates charts to understand the dataset:
- class imbalance chart
- amount by class
- top feature correlation heatmap
- feature histograms
- transaction hour pattern  
**When to use:** Use this when you want to inspect data patterns before model training.

### `preprocessing.py`
**What it is:** Data preparation module.  
**What it does:**
- creates `log_amount`
- creates hour-based features
- does train/test split with stratification
- scales selected columns with `RobustScaler`
- applies SMOTE to training data
- creates a quick SMOTE vs no-SMOTE comparison chart  
**When to use:** Used before training models to make data model-ready.

### `models.py`
**What it is:** Model training module.  
**What it does:** Trains and tunes 4 models using `RandomizedSearchCV` and stratified cross-validation:
- Logistic Regression
- Random Forest
- XGBoost
- Neural Network (MLP)  
It also saves CV score plots for each model and one combined comparison plot.  
**When to use:** Use when you want trained models with tuned settings.

### `evaluation.py`
**What it is:** Model evaluation module.  
**What it does:** For each trained model, it generates:
- classification report
- confusion matrix
- threshold tuning chart
- ROC and PR curves  
It also creates:
- Random Forest feature importance chart
- SHAP plots for XGBoost  
**When to use:** Use this to compare models and choose the best one for fraud detection.

### `colab_deploy.ipynb`
**What it is:** Notebook to run the project on Google Colab.  
**What it does:** Installs dependencies, clones your repo, runs the full pipeline, and displays/downloads outputs.  
**When to use:** Use this when you want cloud execution or do not want to run locally.

### `requirements.txt`
**What it is:** Dependency list.  
**What it does:** Lists Python packages needed to run this project.  
**When to use:** Run install command before first run.

### `creditcard.csv`
**What it is:** Input dataset file.  
**What it does:** Provides the transaction records used by the full pipeline.  
**Important:** Keep this file in the project root so scripts can find it.

### `outputs/`
**What it is:** Generated output folder.  
**What it does:** Stores all created charts, plots, and result visuals.  
**When to use:** Open this folder after pipeline run to review results.

## Dataset Details
Source: [Kaggle ULB Credit Card Fraud Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

Quick facts:
- Total transactions: 284,807
- Fraud transactions: 492
- Fraud rate: about 0.17% (very low, very imbalanced)

Main columns:
- `Time`
- `V1` to `V28` (transformed features)
- `Amount`
- `Class` (0 = normal, 1 = fraud)

## Local Setup
Install all required packages:

```bash
pip install -r requirements.txt
```

## How To Run Locally
Run this command in the project folder:

```bash
python main.py
```

What you will see:
- task-by-task progress logs in terminal
- generated chart files inside `outputs/`
- final model metric tables printed at the end

## Output Files You Get
The pipeline generates files like:
- class distribution plot
- amount comparison plot
- correlation heatmap
- feature distribution charts
- transaction timing plot
- SMOTE impact chart
- cross-validation score charts
- confusion matrices per model
- threshold tuning charts per model
- ROC and PR curve comparison
- feature importance chart
- SHAP bar and beeswarm plots

## How To Run in Google Colab
Use `colab_deploy.ipynb`.

Simple flow:
1. Push this project to GitHub
2. Open the notebook in Colab
3. Set your GitHub repo URL in the notebook
4. Run cells from top to bottom
5. View or download outputs from the notebook

## Tips for Better Results
- Always check `Recall` and `PR-AUC` closely for fraud problems
- Do not apply SMOTE on test data
- Keep random seed fixed for reproducible comparison
- If needed, tune threshold after training instead of always using 0.50

## Final Note
This project is mainly for learning and comparison.
It shows a full end-to-end fraud detection workflow in a clear and practical way.
