# Credit Card Fraud Detection

Machine learning pipeline to detect fraudulent credit card transactions on
the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Problem

Identify the small fraction of credit-card transactions that are
fraudulent. The dataset contains 284,807 transactions with only ~0.17%
fraud — an extreme class-imbalance problem.

## Planned approach

- Exploratory data analysis on class balance and transaction amounts.
- Stratified train/test split.
- SMOTE oversampling for the minority (fraud) class.
- Train Logistic Regression and Random Forest classifiers.
- Evaluate with precision, recall, F1, confusion matrix, and ROC-AUC.

## Setup

```bash
pip install -r requirements.txt
```

Download `creditcard.csv` from Kaggle and place it in `data/`.

## License

MIT — see [LICENSE](LICENSE).
