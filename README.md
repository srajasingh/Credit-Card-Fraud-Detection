# Credit Card Fraud Detection

A complete machine learning pipeline that flags fraudulent credit card
transactions on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
(284,807 transactions, 0.17% fraud).

Built with **scikit-learn**, **imbalanced-learn (SMOTE)**, **pandas**, and **seaborn**.

---

## Why this project is interesting

Credit-card fraud is a textbook **extreme class imbalance** problem: only
**492 of 284,807** transactions are fraudulent (0.17%). A model that always
predicts "genuine" still scores 99.83% accuracy and catches zero fraud, so
the project is built around the metrics that actually matter in production:
**recall**, **precision**, **F1**, and **ROC-AUC**.

---

## Results

Evaluated on a held-out test set of **56,962 transactions** (98 fraud).
SMOTE was applied to the training set only — the test set keeps the real
production distribution.

| Model               | Precision (fraud) | Recall (fraud) | F1 (fraud) | ROC-AUC   |
| ------------------- | ----------------: | -------------: | ---------: | --------: |
| Logistic Regression |             0.058 |      **0.918** |      0.109 |    0.9698 |
| Random Forest       |         **0.845** |          0.837 |  **0.841** | **0.9731** |

**Reading the results:**

- **Logistic Regression** catches 90/98 frauds (91.8% recall) but raises
  1,461 false alarms — sensible if every alert is sent to a cheap human
  review queue.
- **Random Forest** catches 82/98 frauds (83.7% recall) with only 15 false
  alarms. 84% of its alerts are real fraud — far more deployable.
- The right operating point depends on the **cost of a missed fraud vs the
  cost of investigating a false positive**. Tune the decision threshold on
  `predict_proba` to slide along the precision-recall curve.

---

## Visualisations

| | |
| :---: | :---: |
| ![Class distribution](images/01_class_distribution.png) | ![Amount distribution](images/02_amount_distribution.png) |
| Extreme class imbalance | Transaction amount distribution |
| ![Amount by class](images/03_amount_by_class.png) | ![Correlation heatmap](images/04_correlation_heatmap.png) |
| Amount by class | Feature correlation heatmap |
| ![CM — Logistic Regression](images/05_cm_logistic_regression.png) | ![CM — Random Forest](images/05_cm_random_forest.png) |
| Confusion matrix — Logistic Regression | Confusion matrix — Random Forest |
| ![ROC curves](images/06_roc_curves.png) | |
| ROC curves for both models | |

---

## Pipeline

1. **Load** `creditcard.csv` (284,807 × 31).
2. **EDA** — class distribution, transaction-amount distribution, amount
   by class, correlation heatmap.
3. **Feature prep** — `Time` and `Amount` standardised with
   `StandardScaler`. V1–V28 are already PCA components and left alone.
4. **Stratified 80/20 split** — `stratify=y` is critical because of the
   0.17% positive rate.
5. **SMOTE oversampling** — applied **only to the training set** (never
   the test set) to avoid information leakage.
6. **Models** — Logistic Regression and Random Forest.
7. **Evaluation** — full classification report, confusion matrix, ROC-AUC,
   ROC curves.

---

## Project structure

```
Credit-Card-Fraud-Detection/
├── data/
│   └── creditcard.csv          # download from Kaggle (gitignored)
├── notebooks/
│   └── fraud_detection.ipynb   # narrated walkthrough
├── images/                     # generated plots
├── fraud_detection.py          # runnable end-to-end pipeline
├── requirements.txt
├── LICENSE
└── README.md
```

---

## Setup

```bash
git clone https://github.com/<your-username>/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
pip install -r requirements.txt
```

Download the dataset from
[Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) and put
`creditcard.csv` into `data/`.

## Run

```bash
python fraud_detection.py
```

Or open the notebook:

```bash
jupyter notebook notebooks/fraud_detection.ipynb
```

All plots are written to `images/`.

---

## Business takeaways

- Fraud is ~0.17% of transactions — **accuracy is the wrong metric**.
- **Recall** is the business priority: every missed fraud is a direct loss.
- SMOTE lets the model learn the minority class without discarding genuine
  data the way undersampling does.
- Random Forest gives the best precision/recall trade-off in this setup
  and is the better production candidate.

## Next steps

- Threshold tuning with a precision-recall curve.
- Gradient boosting (XGBoost / LightGBM) — usually wins on tabular data.
- SHAP feature attributions for model explainability.
- Concept-drift monitoring once deployed.

---

## License

[MIT](LICENSE)

## Author

**Sraja Singh** — <singhsraja2@gmail.com>
