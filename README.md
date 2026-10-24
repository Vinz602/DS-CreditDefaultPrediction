# DS-FraudDetection: Credit Default Prediction

Predicting whether a borrower will experience serious delinquency (90+ days past due) within two years, using the Kaggle **[Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)** dataset and a gradient-boosted tree classifier.

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-HistGradientBoosting-orange.svg)

---

## Overview

Lenders need to estimate the probability that an applicant will default. The target is rare (about 1 in 15 borrowers), so this is a **highly imbalanced binary classification** problem: a model that always predicts "no default" would be 93.3% accurate while catching zero defaulters. For that reason the project evaluates with **ROC-AUC** and **PR-AUC** rather than accuracy.

The notebook covers the full workflow: data loading, cleaning, exploratory analysis, feature engineering, training, evaluation, and generation of a Kaggle submission file.

## Dataset

| Property | Value |
|---|---|
| Source | Kaggle competition `GiveMeSomeCredit` |
| Train rows | 150,000 |
| Test rows | 101,503 (unlabeled) |
| Target | `SeriousDlqin2yrs` (1 = 90+ days past due within 2 years) |
| Positive class | 10,026 defaults (**6.68%**, imbalance ratio about 14 : 1) |
| Baseline accuracy (always predict 0) | 93.32% |

**Features (10 raw inputs):**

| Feature | Description |
|---|---|
| `RevolvingUtilizationOfUnsecuredLines` | Credit card / personal line balances divided by credit limits |
| `age` | Borrower age in years |
| `NumberOfTime30-59DaysPastDueNotWorse` | Times 30-59 days past due in the last 2 years |
| `NumberOfTime60-89DaysPastDueNotWorse` | Times 60-89 days past due in the last 2 years |
| `NumberOfTimes90DaysLate` | Times 90+ days past due |
| `DebtRatio` | Monthly debt payments divided by monthly gross income |
| `MonthlyIncome` | Monthly income (about 20% missing) |
| `NumberOfOpenCreditLinesAndLoans` | Open loans and lines of credit |
| `NumberRealEstateLoansOrLines` | Mortgage and real estate loans |
| `NumberOfDependents` | Dependents excluding the borrower (about 2.6% missing) |

## Approach

### 1. Cleaning & feature engineering

| Issue found in EDA | Treatment |
|---|---|
| Late-payment counts contain the sentinel values `96` and `98` (not real counts) | Replaced with `NaN` |
| `age` contains 0 | Values under 18 set to `NaN` |
| `RevolvingUtilizationOfUnsecuredLines` has extreme values (max about 50,708) although it should be a ratio | Clipped to `[0, 2]` |
| `NumberOfDependents` missing | Filled with `0` |
| `MonthlyIncome` missing | Kept as `NaN`, plus a new binary feature **`income_missing`** (missingness can itself be informative) |

Cleaning is wrapped in a single `make_features(df)` function so train and test data receive identical treatment.

### 2. Split
Stratified 80/20 train/validation split (`random_state=42`), giving `[120000, 11]` for training and `[30000, 11]` for validation, with the default rate preserved in both.

### 3. Model
`HistGradientBoostingClassifier` (scikit-learn), chosen because it handles `NaN` natively (no imputation needed) and works well on tabular data:

```python
HistGradientBoostingClassifier(
    learning_rate=0.05, max_iter=500, max_leaf_nodes=15,
    min_samples_leaf=50, l2_regularization=1.0,
    early_stopping=True, validation_fraction=0.15,
    n_iter_no_change=30, random_state=42)
```

### 4. Final prediction
The model is refit on all 150,000 labeled rows, then used to predict default probabilities for the 101,503 test rows, written to `submission.csv` (`Id`, `Probability`).

## Results

Held-out validation set (30,000 rows, 2,005 defaults):

| Metric | Value |
|---|---|
| ROC-AUC | **0.8681** |
| PR-AUC (average precision) | **0.4066** |

Classification report at the default 0.5 threshold:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (no default) | 0.94 | 0.99 | 0.97 | 27,995 |
| 1 (default) | 0.61 | 0.19 | 0.29 | 2,005 |

**Interpretation:** the model ranks borrowers well (ROC-AUC 0.87), but at a 0.5 probability cutoff it flags only 19% of true defaulters. PR-AUC (0.41, versus a no-skill baseline of about 0.067) is the more honest summary under this imbalance. Because the model outputs probabilities, the threshold can be lowered to trade precision for recall depending on the relative cost of a missed default versus a rejected good borrower.

## Repository Structure

```
DS-FraudDetection/
├── FraudDetection.ipynb   # end-to-end notebook
└── README.md
```

## Getting Started

### Prerequisites
- Python 3.10+
- A Kaggle account with the **Give Me Some Credit** competition rules accepted (required to download the data)

### Installation

```bash
git clone https://github.com/Vinz602/DS-FraudDetection.git
cd DS-FraudDetection

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install kagglehub pandas numpy matplotlib seaborn scikit-learn xlrd jupyter
```

`xlrd` is needed to read the competition's `Data Dictionary.xls`.

### Run

```bash
jupyter notebook FraudDetection.ipynb
```

The first cell calls `kagglehub.login()`, which prompts for your Kaggle credentials. The data is then downloaded automatically via `kagglehub.competition_download('GiveMeSomeCredit')`. Run the cells top to bottom to reproduce the results and generate `submission.csv`.

## Possible Improvements

- Tune the decision threshold (e.g. maximize F1 or fix a target recall using the precision-recall curve)
- Cross-validation instead of a single hold-out split
- Hyperparameter search and comparison against XGBoost / LightGBM / logistic regression
- Probability calibration (isotonic or Platt) if scores are used as actual default probabilities
- Feature interactions such as total number of late payments, or income-per-dependent
- Model explainability with SHAP

## Acknowledgements

Dataset: [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit), Kaggle.
