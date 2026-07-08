# -*- coding: utf-8 -*-
"""
Reproduces the notebook's pipeline (feature engineering + final LR/KNN models)
and exports everything the Streamlit dashboard needs:
  - dashboard/models/lr_pipeline.joblib
  - dashboard/models/knn_pipeline.joblib
  - dashboard/data/passengers_scored.parquet
  - dashboard/data/service_cols.json (list of the 14 service rating columns)

Run once from the dashboard/scripts/ folder (train.csv/test.csv must be here):
    python train_and_export.py
"""
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE.parent / "models"
DATA_DIR = HERE.parent / "data"
MODELS_DIR.mkdir(exist_ok=True, parents=True)
DATA_DIR.mkdir(exist_ok=True, parents=True)

SERVICE_COLS = [
    "Inflight wifi service", "Departure/Arrival time convenient",
    "Ease of Online booking", "Gate location", "Food and drink",
    "Online boarding", "Seat comfort", "Inflight entertainment",
    "On-board service", "Leg room service", "Baggage handling",
    "Checkin service", "Inflight service", "Cleanliness",
]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(df["Departure Delay in Minutes"])
    df["loyal_business"] = (
        (df["Customer Type"] == "Loyal Customer") & (df["Type of Travel"] == "Business travel")
    ).astype(int)
    df["mean_service_score"] = df[SERVICE_COLS].mean(axis=1)
    bins = [0, 18, 30, 45, 60, 100]
    labels = ["minor", "young_adult", "adult", "mid_career", "senior"]
    df["age_group"] = pd.cut(df["Age"], bins=bins, labels=labels)
    return df


def main():
    train = pd.read_csv(HERE / "train.csv").drop(columns=["Unnamed: 0"])
    test = pd.read_csv(HERE / "test.csv").drop(columns=["Unnamed: 0"])

    train_fe = add_features(train)
    test_fe = add_features(test)

    feature_cols = [c for c in train_fe.columns if c not in ["id", "satisfaction"]]
    cat_cols = train_fe[feature_cols].select_dtypes(include=["object", "category"]).columns.tolist()
    num_cols = [c for c in feature_cols if c not in cat_cols]

    X = train_fe[feature_cols]
    y = train_fe["satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1})
    X_test = test_fe[feature_cols]
    y_test = test_fe["satisfaction"].map({"neutral or dissatisfied": 0, "satisfied": 1})

    preprocess = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), cat_cols),
    ])

    lr_final = Pipeline([
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE, C=10)),
    ])
    knn_final = Pipeline([
        ("preprocess", preprocess),
        ("clf", KNeighborsClassifier(n_neighbors=11, weights="distance", n_jobs=-1)),
    ])

    print("Training final Logistic Regression...")
    lr_final.fit(X, y)
    print("Training final KNN...")
    knn_final.fit(X, y)

    joblib.dump(lr_final, MODELS_DIR / "lr_pipeline.joblib")
    joblib.dump(knn_final, MODELS_DIR / "knn_pipeline.joblib")
    print("Models exported to", MODELS_DIR)

    # ---- Build the scored dataset used by the dashboard (train + test) ----
    def score(df_fe, split_name):
        X_ = df_fe[feature_cols]
        out = df_fe.copy()
        out["dataset"] = split_name
        out["lr_proba"] = lr_final.predict_proba(X_)[:, 1]
        out["knn_proba"] = knn_final.predict_proba(X_)[:, 1]
        out["lr_pred"] = (out["lr_proba"] >= 0.5).map({True: "satisfied", False: "neutral or dissatisfied"})
        out["knn_pred"] = (out["knn_proba"] >= 0.5).map({True: "satisfied", False: "neutral or dissatisfied"})
        return out

    scored = pd.concat([score(train_fe, "train"), score(test_fe, "test")], ignore_index=True)
    scored["age_group"] = scored["age_group"].astype(str)

    keep_cols = [
        "id", "dataset", "Gender", "Customer Type", "Age", "age_group", "Type of Travel", "Class",
        "Flight Distance", "loyal_business", "mean_service_score",
        "Departure Delay in Minutes", "Arrival Delay in Minutes",
        "satisfaction", "lr_proba", "knn_proba", "lr_pred", "knn_pred",
    ] + SERVICE_COLS
    scored = scored[keep_cols]
    scored.to_parquet(DATA_DIR / "passengers_scored.parquet", index=False)
    print(f"Scored dataset exported: {scored.shape} -> {DATA_DIR / 'passengers_scored.parquet'}")

    with open(DATA_DIR / "service_cols.json", "w") as f:
        json.dump(SERVICE_COLS, f)

    # ---- Metrics summary for the dashboard (avoids recomputation at runtime) ----
    from sklearn.metrics import (
        accuracy_score, recall_score, precision_score, f1_score, roc_auc_score,
        roc_curve, confusion_matrix,
    )
    metrics = {}
    for name, proba_col, pred_col in [("Régression Logistique", "lr_proba", "lr_pred"),
                                       ("KNN", "knn_proba", "knn_pred")]:
        test_rows = scored[scored["dataset"] == "test"]
        y_true = (test_rows["satisfaction"] == "satisfied").astype(int)
        y_pred = (test_rows[proba_col] >= 0.5).astype(int)
        fpr, tpr, _ = roc_curve(y_true, test_rows[proba_col])
        # downsample to ~150 points for a lightweight, fast-loading JSON
        if len(fpr) > 150:
            idx = np.linspace(0, len(fpr) - 1, 150).astype(int)
            fpr, tpr = fpr[idx], tpr[idx]
        cm = confusion_matrix(y_true, y_pred).tolist()
        metrics[name] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "recall_class0": recall_score(y_true, y_pred, pos_label=0),
            "precision_class0": precision_score(y_true, y_pred, pos_label=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro"),
            "roc_auc": roc_auc_score(y_true, test_rows[proba_col]),
            "fpr": fpr.tolist(),
            "tpr": tpr.tolist(),
            "confusion_matrix": cm,
        }
    with open(DATA_DIR / "metrics.json", "w") as f:
        json.dump(metrics, f)
    print("Metrics exported to", DATA_DIR / "metrics.json")

    # LR coefficients for interpretation page
    feature_names = lr_final.named_steps["preprocess"].get_feature_names_out().tolist()
    coefs = lr_final.named_steps["clf"].coef_[0].tolist()
    with open(DATA_DIR / "lr_coefficients.json", "w") as f:
        json.dump({"feature_names": feature_names, "coefficients": coefs}, f)
    print("LR coefficients exported.")


if __name__ == "__main__":
    main()
