"""
Streamlit app: Bank Marketing - Term Deposit Subscription Classifier
--------------------------------------------------------------------
Loads the five pre-trained models from model/ and evaluates them on a
user-uploaded test CSV (no re-training happens in the app).
"""

import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)

MODEL_DIR = "model"
TARGET = "y"
MODELS = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Naive Bayes (Gaussian)": "naive_bayes",
    "Support Vector Machine": "svm",
}

st.set_page_config(page_title="Bank Marketing Classifier", page_icon="🏦",
                   layout="wide")


@st.cache_resource(show_spinner="Loading model...")
def load_model(file_name: str):
    return joblib.load(os.path.join(MODEL_DIR, f"{file_name}.joblib"))


def compute_metrics(y_true, y_pred, y_prob) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_prob),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
st.sidebar.title("⚙️ Controls")
model_label = st.sidebar.selectbox("Select a classification model",
                                   list(MODELS.keys()))
uploaded = st.sidebar.file_uploader("Upload test data (CSV)", type=["csv"],
                                    help="Must contain all 19 input features "
                                         "and the target column 'y'.")
use_sample = st.sidebar.checkbox("Use bundled test_data.csv instead",
                                 value=uploaded is None)
show_all = st.sidebar.checkbox("Compare all five models on this data",
                               value=False)

st.sidebar.markdown("---")
st.sidebar.caption("BITS Pilani WILP · M.Tech (AIML) · Machine Learning · "
                   "Assignment 2")

# ----------------------------------------------------------------------------
# Main page
# ----------------------------------------------------------------------------
st.title("🏦 Bank Marketing: Term Deposit Subscription Prediction")
st.markdown(
    "Predict whether a bank client will **subscribe to a term deposit** "
    "(`y = yes/no`) after a direct-marketing phone campaign. "
    "Dataset: [UCI Bank Marketing (bank-additional-full)]"
    "(https://archive.ics.uci.edu/dataset/222/bank+marketing) - "
    "41,188 records, 19 input features after dropping the leaky `duration` column."
)

with st.expander("ℹ️ About the five models and the training setup"):
    st.markdown("""
* **Split:** 80 / 20 stratified train-test split (`random_state=42`).
* **Pre-processing** (inside every saved pipeline): `StandardScaler` on the
  10 numeric columns and `OneHotEncoder` on the 9 categorical columns.
* **Class imbalance:** only ~11 % of clients subscribe, so Logistic Regression,
  Decision Tree and SVM use `class_weight="balanced"`.
* Models are **loaded from disk** - nothing is re-trained here.
""")

# ---- load data --------------------------------------------------------------
df = None
if uploaded is not None and not use_sample:
    try:
        df = pd.read_csv(uploaded)
        if df.shape[1] == 1:               # user uploaded a ';' separated file
            uploaded.seek(0)
            df = pd.read_csv(uploaded, sep=";")
        st.success(f"Uploaded file loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the CSV: {e}")
elif os.path.exists("test_data.csv"):
    df = pd.read_csv("test_data.csv")
    st.info(f"Using bundled **test_data.csv** ({df.shape[0]} rows). "
            "Upload your own file from the sidebar to override.")
else:
    st.warning("Please upload a test CSV from the sidebar.")

if df is not None:
    # ---- validate ----------------------------------------------------------
    model = load_model(MODELS[model_label])
    expected = list(model.named_steps["prep"].feature_names_in_)

    if TARGET not in df.columns:
        st.error(f"The uploaded file has no target column named `{TARGET}`.")
        st.stop()
    missing = [c for c in expected if c not in df.columns]
    if missing:
        st.error(f"Missing input feature columns: {missing}")
        st.stop()
    if "duration" in df.columns:
        st.caption("Column `duration` found - it is ignored (not used by the models).")

    X = df[expected]
    y_true = df[TARGET]
    if y_true.dtype == object or str(y_true.dtype) == "str":
        y_true = (y_true.astype(str).str.lower() == "yes").astype(int)
    y_true = y_true.astype(int)

    with st.expander("👀 Preview of the uploaded data"):
        st.dataframe(df.head(10), use_container_width=True)
        st.write("Target distribution:", y_true.value_counts().rename(
            {0: "no (0)", 1: "yes (1)"}).to_dict())

    # ---- predict with the selected model --------------------------------------
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    metrics = compute_metrics(y_true, y_pred, y_prob)

    st.subheader(f"📊 Evaluation metrics - {model_label}")
    cols = st.columns(6)
    for col, (name, value) in zip(cols, metrics.items()):
        col.metric(name, f"{value:.4f}")

    left, right = st.columns(2)

    with left:
        st.subheader("Confusion matrix")
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=["no", "yes"], yticklabels=["no", "yes"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        st.pyplot(fig, use_container_width=False)

    with right:
        st.subheader("ROC curve")
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        fig2, ax2 = plt.subplots(figsize=(4.5, 3.8))
        ax2.plot(fpr, tpr, label=f"AUC = {metrics['AUC']:.3f}")
        ax2.plot([0, 1], [0, 1], "k--", lw=0.8)
        ax2.set_xlabel("False positive rate")
        ax2.set_ylabel("True positive rate")
        ax2.legend(loc="lower right")
        st.pyplot(fig2, use_container_width=False)

    st.subheader("Classification report")
    report = classification_report(y_true, y_pred, target_names=["no", "yes"],
                                   output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).T.style.format("{:.4f}"),
                 use_container_width=True)

    # ---- optional: compare all models ----------------------------------------
    if show_all:
        st.subheader("🏁 All five models on this test data")
        rows = {}
        progress = st.progress(0.0)
        for i, (label, key) in enumerate(MODELS.items()):
            m = load_model(key)
            rows[label] = compute_metrics(y_true, m.predict(X),
                                          m.predict_proba(X)[:, 1])
            progress.progress((i + 1) / len(MODELS))
        progress.empty()
        comp = pd.DataFrame(rows).T
        st.dataframe(comp.style.format("{:.4f}").highlight_max(axis=0,
                     color="#c6efce"), use_container_width=True)

        fig3, ax3 = plt.subplots(figsize=(9, 3.5))
        comp.plot(kind="bar", ax=ax3, width=0.8)
        ax3.set_ylim(0, 1)
        ax3.set_ylabel("Score")
        ax3.legend(ncol=6, fontsize=8, loc="upper center",
                   bbox_to_anchor=(0.5, -0.25))
        plt.xticks(rotation=0, fontsize=8)
        st.pyplot(fig3)

    # ---- download predictions --------------------------------------------------
    out = df.copy()
    out["predicted_y"] = y_pred
    out["probability_yes"] = y_prob.round(4)
    st.download_button("⬇️ Download predictions as CSV",
                       out.to_csv(index=False).encode(),
                       file_name=f"predictions_{MODELS[model_label]}.csv",
                       mime="text/csv")
