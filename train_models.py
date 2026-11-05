"""
train_models.py
---------------
Trains five classification models on the UCI Bank Marketing dataset
(bank-additional-full.csv) and saves:

    model/<model_name>.joblib   -> trained sklearn Pipeline (preprocessing + classifier)
    model/metrics.json          -> evaluation metrics of every model on the hold-out test set
    test_data.csv               -> the hold-out test split (features + target) used by the app

Run:  python train_models.py
"""

import json
import os
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV

RANDOM_STATE = 42
DATA_PATH = os.path.join("data", "bank-additional-full.csv")
MODEL_DIR = "model"
TEST_PATH = "test_data.csv"
TARGET = "y"

# "duration" is dropped on purpose: it is only known AFTER the call ends and
# therefore leaks the outcome (see the UCI dataset notes).
DROP_COLS = ["duration"]


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, sep=";")
    df = df.drop(columns=DROP_COLS)
    df[TARGET] = (df[TARGET] == "yes").astype(int)   # yes -> 1, no -> 0
    return df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    cat_cols = X.select_dtypes(exclude=np.number).columns.tolist()
    num_cols = X.select_dtypes(include=np.number).columns.tolist()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )


def get_models() -> dict:
    """The five classifiers required by the assignment."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "decision_tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced",
            random_state=RANDOM_STATE),
        "knn": KNeighborsClassifier(n_neighbors=15, weights="distance"),
        "naive_bayes": GaussianNB(var_smoothing=1e-3),
        "svm": CalibratedClassifierCV(
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced",
                cache_size=1000, random_state=RANDOM_STATE),
            method="sigmoid", cv=3, ensemble=False),
    }


def evaluate(model, X, y) -> dict:
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": round(accuracy_score(y, y_pred), 4),
        "AUC": round(roc_auc_score(y, y_prob), 4),
        "Precision": round(precision_score(y, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y, y_pred), 4),
        "F1": round(f1_score(y, y_pred), 4),
        "MCC": round(matthews_corrcoef(y, y_pred), 4),
    }


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_dataset(DATA_PATH)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    print(f"Dataset: {X.shape[0]} rows, {X.shape[1]} features, "
          f"positive rate = {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE)

    # Save the hold-out split so the Streamlit app can be tested on unseen data
    test_df = X_test.copy()
    test_df[TARGET] = y_test.values
    test_df.to_csv(TEST_PATH, index=False)
    print(f"Saved {len(test_df)} test rows -> {TEST_PATH}")

    results = {}
    for name, clf in get_models().items():
        t0 = time.time()
        pipe = Pipeline([("prep", build_preprocessor(X)), ("clf", clf)])
        pipe.fit(X_train, y_train)
        results[name] = evaluate(pipe, X_test, y_test)
        joblib.dump(pipe, os.path.join(MODEL_DIR, f"{name}.joblib"))
        print(f"{name:20s} {results[name]}  ({time.time() - t0:.1f}s)")

    with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nComparison table (markdown):")
    table = pd.DataFrame(results).T
    print(table.to_markdown())


if __name__ == "__main__":
    main()
