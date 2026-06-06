"""
Credit Card Fraud Detection
---------------------------
End-to-end pipeline:
  1. Load data
  2. EDA + visualisations
  3. Train/test split (stratified)
  4. Handle class imbalance with SMOTE
  5. Train Logistic Regression and Random Forest
  6. Evaluate with classification report, confusion matrix, ROC-AUC
  7. Save all plots to ./images/
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)

from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = Path(os.environ.get("DATA_PATH", "data/creditcard.csv"))
IMG_DIR = Path("images")
IMG_DIR.mkdir(parents=True, exist_ok=True)
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# 1. Load
# ---------------------------------------------------------------------------
def load_data(path: Path) -> pd.DataFrame:
    print(f"[load] reading {path}")
    df = pd.read_csv(path)
    print(f"[load] shape={df.shape}")
    print(f"[load] missing values: {int(df.isnull().sum().sum())}")
    return df


# ---------------------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------------------
def run_eda(df: pd.DataFrame) -> None:
    counts = df["Class"].value_counts()
    print(f"[eda] class counts:\n{counts}")
    print(f"[eda] fraud rate: {counts.get(1, 0) / len(df):.4%}")

    # Class distribution
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Class", data=df, palette=["#2ecc71", "#e74c3c"])
    plt.title("Class Distribution (0 = Genuine, 1 = Fraud)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "01_class_distribution.png", dpi=130)
    plt.close()

    # Amount distribution
    plt.figure(figsize=(8, 4))
    sns.histplot(df["Amount"], bins=80, color="#3498db")
    plt.title("Transaction Amount Distribution")
    plt.xlim(0, df["Amount"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(IMG_DIR / "02_amount_distribution.png", dpi=130)
    plt.close()

    # Amount by class
    plt.figure(figsize=(7, 4))
    sns.boxplot(x="Class", y="Amount", data=df, palette=["#2ecc71", "#e74c3c"])
    plt.title("Transaction Amount by Class")
    plt.ylim(0, df["Amount"].quantile(0.99))
    plt.tight_layout()
    plt.savefig(IMG_DIR / "03_amount_by_class.png", dpi=130)
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 9))
    sns.heatmap(df.corr(), cmap="coolwarm", center=0, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "04_correlation_heatmap.png", dpi=130)
    plt.close()

    print(f"[eda] plots saved to {IMG_DIR}/")


# ---------------------------------------------------------------------------
# 3. Prep
# ---------------------------------------------------------------------------
def prepare(df: pd.DataFrame):
    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    # Scale Time and Amount (V1..V28 are already PCA-scaled)
    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[prep] train={X_train.shape}  test={X_test.shape}")

    smote = SMOTE(random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    print(f"[prep] after SMOTE: {np.bincount(y_res)}")
    return X_res, X_test, y_res, y_test


# ---------------------------------------------------------------------------
# 4. Train + evaluate
# ---------------------------------------------------------------------------
def evaluate(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)

    print(f"\n===== {name} =====")
    print(classification_report(y_test, pred, digits=4))
    print(f"ROC-AUC: {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Fraud"],
        yticklabels=["Genuine", "Fraud"],
    )
    plt.title(f"Confusion Matrix — {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(IMG_DIR / f"05_cm_{fname}.png", dpi=130)
    plt.close()

    return {"name": name, "auc": auc, "prob": prob}


def plot_roc(results, y_test) -> None:
    plt.figure(figsize=(7, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["prob"])
        plt.plot(fpr, tpr, label=f"{r['name']} (AUC={r['auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.4)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMG_DIR / "06_roc_curves.png", dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data(DATA_PATH)
    run_eda(df)
    X_res, X_test, y_res, y_test = prepare(df)

    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, n_jobs=-1)
    lr.fit(X_res, y_res)
    lr_res = evaluate("Logistic Regression", lr, X_test, y_test)

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_res, y_res)
    rf_res = evaluate("Random Forest", rf, X_test, y_test)

    plot_roc([lr_res, rf_res], y_test)
    print(f"\n[done] all plots written to {IMG_DIR.resolve()}")


if __name__ == "__main__":
    main()
