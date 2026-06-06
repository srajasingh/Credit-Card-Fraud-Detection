"""
Streamlit app — Credit Card Fraud Detection
============================================
Live demo: trains a Random Forest on the Kaggle creditcard.csv dataset
and lets users interact with the model in the browser.

Run locally:
    streamlit run app.py
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Credit Card Fraud Detector",
    page_icon="🛡️",
    layout="wide",
)

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
st.title("🛡️ Credit Card Fraud Detection")
st.markdown(
    "An end-to-end ML pipeline using **SMOTE + Random Forest** on the "
    "[Kaggle Credit Card Fraud dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)."
)
st.divider()

# ─────────────────────────────────────────────
# Sidebar — upload + settings
# ─────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")
    uploaded_file = st.file_uploader(
        "Upload `creditcard.csv`", type=["csv"],
        help="Download from Kaggle and upload here."
    )
    st.markdown("---")
    model_choice = st.selectbox(
        "Model", ["Random Forest", "Logistic Regression", "Both"]
    )
    test_size = st.slider("Test set size", 0.1, 0.4, 0.2, 0.05)
    n_estimators = st.slider("RF: n_estimators", 50, 300, 100, 50)
    st.markdown("---")
    st.caption("Built by Sraja Singh · [GitHub](https://github.com/srajasingh/Credit-Card-Fraud-Detection)")

# ─────────────────────────────────────────────
# Helper: load + cache
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(file) -> pd.DataFrame:
    return pd.read_csv(file)


@st.cache_resource(show_spinner="Training model (this takes ~30s)…")
def train_pipeline(csv_bytes, test_sz, n_est, choice):
    df = pd.read_csv(io.BytesIO(csv_bytes))

    X = df.drop("Class", axis=1).copy()
    y = df["Class"].copy()

    scaler = StandardScaler()
    X[["Time", "Amount"]] = scaler.fit_transform(X[["Time", "Amount"]])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_sz, random_state=42, stratify=y
    )

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    models = {}

    if choice in ["Random Forest", "Both"]:
        rf = RandomForestClassifier(n_estimators=n_est, random_state=42, n_jobs=-1)
        rf.fit(X_res, y_res)
        models["Random Forest"] = rf

    if choice in ["Logistic Regression", "Both"]:
        lr = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        lr.fit(X_res, y_res)
        models["Logistic Regression"] = lr

    return models, X_test, y_test, scaler, df


# ─────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────
if uploaded_file is None:
    st.info("👈 Upload `creditcard.csv` from the sidebar to get started.", icon="📂")

    st.subheader("About this project")
    col1, col2, col3 = st.columns(3)
    col1.metric("Transactions", "284,807")
    col2.metric("Fraud Rate", "0.17%")
    col3.metric("Best ROC-AUC", "0.973")

    st.markdown("""
    #### Pipeline
    1. Load Kaggle credit card dataset (284,807 transactions)  
    2. Exploratory Data Analysis  
    3. Stratified 80/20 train/test split  
    4. SMOTE oversampling on training set only  
    5. Train Random Forest / Logistic Regression  
    6. Evaluate with precision, recall, F1, ROC-AUC  
    7. Interactive single-transaction prediction  
    """)
    st.stop()

# ── Load & display basic info ──────────────────
df = load_data(uploaded_file)

st.subheader("📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)
fraud_count = df["Class"].sum()
genuine_count = len(df) - fraud_count
c1.metric("Total Transactions", f"{len(df):,}")
c2.metric("Genuine", f"{genuine_count:,}")
c3.metric("Fraud", f"{int(fraud_count):,}")
c4.metric("Fraud Rate", f"{fraud_count/len(df):.3%}")

with st.expander("Preview raw data"):
    st.dataframe(df.head(20), use_container_width=True)

st.divider()

# ── EDA ────────────────────────────────────────
st.subheader("🔍 Exploratory Data Analysis")

tab1, tab2, tab3 = st.tabs(["Class Distribution", "Amount Distribution", "Correlation Heatmap"])

with tab1:
    fig, ax = plt.subplots(figsize=(5, 3))
    sns.countplot(x="Class", data=df, palette=["#2ecc71", "#e74c3c"], ax=ax)
    ax.set_xticklabels(["Genuine (0)", "Fraud (1)"])
    ax.set_title("Class Distribution")
    st.pyplot(fig, use_container_width=False)

with tab2:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    sns.histplot(df["Amount"], bins=80, color="#3498db", ax=axes[0])
    axes[0].set_xlim(0, df["Amount"].quantile(0.99))
    axes[0].set_title("Transaction Amount (all)")
    sns.boxplot(x="Class", y="Amount", data=df, palette=["#2ecc71", "#e74c3c"], ax=axes[1])
    axes[1].set_ylim(0, df["Amount"].quantile(0.99))
    axes[1].set_title("Amount by Class")
    axes[1].set_xticklabels(["Genuine", "Fraud"])
    st.pyplot(fig, use_container_width=True)

with tab3:
    st.caption("Computing correlation on a sample of 5,000 rows for speed.")
    sample = df.sample(5000, random_state=42)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(sample.corr(), cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("Feature Correlation Heatmap")
    st.pyplot(fig, use_container_width=True)

st.divider()

# ── Train ──────────────────────────────────────
st.subheader("🤖 Model Training & Evaluation")

with st.spinner("Training model…"):
    uploaded_file.seek(0)
    csv_bytes = uploaded_file.read()
    models, X_test, y_test, scaler, _ = train_pipeline(
        csv_bytes, test_size, n_estimators, model_choice
    )

st.success(f"✅ Model(s) trained! Test set: {len(X_test):,} transactions ({int(y_test.sum())} fraud)")

# ── Metrics per model ──────────────────────────
roc_results = []

for name, model in models.items():
    st.markdown(f"#### {name}")
    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, prob)
    roc_results.append((name, prob, auc))

    report = classification_report(y_test, pred, output_dict=True)
    fraud_metrics = report.get("1", {})

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("ROC-AUC", f"{auc:.4f}")
    m2.metric("Precision (fraud)", f"{fraud_metrics.get('precision', 0):.3f}")
    m3.metric("Recall (fraud)", f"{fraud_metrics.get('recall', 0):.3f}")
    m4.metric("F1 (fraud)", f"{fraud_metrics.get('f1-score', 0):.3f}")

    col_cm, col_rpt = st.columns([1, 2])
    with col_cm:
        cm = confusion_matrix(y_test, pred)
        fig, ax = plt.subplots(figsize=(4, 3))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Genuine", "Fraud"],
                    yticklabels=["Genuine", "Fraud"])
        ax.set_title(f"Confusion Matrix")
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        st.pyplot(fig)
    with col_rpt:
        st.text(classification_report(y_test, pred, digits=4))

# ── ROC curves ────────────────────────────────
if roc_results:
    st.markdown("#### ROC Curves")
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, prob, auc in roc_results:
        fpr, tpr, _ = roc_curve(y_test, prob)
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(loc="lower right")
    st.pyplot(fig, use_container_width=False)

st.divider()

# ── Live Prediction ────────────────────────────
st.subheader("🔮 Live Single Transaction Prediction")
st.caption("Adjust the sliders to simulate a transaction and get a fraud probability.")

# Pick the best model (RF preferred)
pred_model_name = "Random Forest" if "Random Forest" in models else list(models.keys())[0]
pred_model = models[pred_model_name]
st.info(f"Using: **{pred_model_name}**")

# Let user pick a real transaction as starting point
sample_idx = st.number_input(
    "Start from a real transaction (row index)", 0, len(df) - 1, 0, 1
)
sample_row = df.drop("Class", axis=1).iloc[sample_idx].copy()
actual_label = int(df["Class"].iloc[sample_idx])
st.caption(f"Actual label for row {sample_idx}: {'🚨 FRAUD' if actual_label == 1 else '✅ Genuine'}")

# Scale Time & Amount
sample_row[["Time", "Amount"]] = scaler.transform(
    pd.DataFrame([sample_row[["Time", "Amount"]].values], columns=["Time", "Amount"])
)[0]

# Let user tweak Amount & Time only (V features are PCA components)
col_a, col_b = st.columns(2)
with col_a:
    amount = st.slider("Amount (original $)", 0.0, 5000.0, float(df["Amount"].iloc[sample_idx]), 1.0)
with col_b:
    time_val = st.slider("Time (seconds since first txn)", 0, 172800, int(df["Time"].iloc[sample_idx]), 100)

# Rebuild scaled input
input_df = df.drop("Class", axis=1).iloc[[sample_idx]].copy()
input_df["Amount"] = amount
input_df["Time"] = time_val
input_df[["Time", "Amount"]] = scaler.transform(input_df[["Time", "Amount"]])

proba = pred_model.predict_proba(input_df)[0][1]
prediction = pred_model.predict(input_df)[0]

col_res1, col_res2 = st.columns(2)
col_res1.metric("Fraud Probability", f"{proba:.2%}")
if prediction == 1:
    col_res2.error("🚨 PREDICTION: FRAUD")
else:
    col_res2.success("✅ PREDICTION: Genuine")

st.progress(float(proba), text=f"Fraud score: {proba:.2%}")
