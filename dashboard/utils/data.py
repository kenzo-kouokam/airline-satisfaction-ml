# -*- coding: utf-8 -*-
"""Cached data & model loaders shared across all dashboard pages."""
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


@st.cache_data(show_spinner="Chargement des données passagers...")
def load_scored_data() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "passengers_scored.parquet")
    df["age_group"] = pd.Categorical(
        df["age_group"], categories=["minor", "young_adult", "adult", "mid_career", "senior"], ordered=True
    )
    return df


@st.cache_data
def load_service_cols() -> list[str]:
    with open(DATA_DIR / "service_cols.json") as f:
        return json.load(f)


@st.cache_data
def load_metrics() -> dict:
    with open(DATA_DIR / "metrics.json") as f:
        return json.load(f)


@st.cache_data
def load_lr_coefficients() -> pd.DataFrame:
    with open(DATA_DIR / "lr_coefficients.json") as f:
        raw = json.load(f)
    return pd.DataFrame({"feature": raw["feature_names"], "coefficient": raw["coefficients"]})


@st.cache_resource(show_spinner="Chargement des modèles...")
def load_models():
    lr = joblib.load(MODELS_DIR / "lr_pipeline.joblib")
    knn = joblib.load(MODELS_DIR / "knn_pipeline.joblib")
    return {"Régression Logistique": lr, "KNN": knn}


def compute_live_metrics(df: pd.DataFrame, model_choice: str) -> dict | None:
    """Recomputes accuracy/recall/precision/F1/ROC AUC on the TEST rows of `df`
    (whatever the current sidebar filters produced), for the chosen model.
    Returns None if there are no test rows in the current selection."""
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score

    test_rows = df[df["dataset"] == "test"]
    if len(test_rows) < 10 or test_rows["satisfaction"].nunique() < 2:
        return None

    proba_col = "knn_proba" if model_choice == "KNN" else "lr_proba"
    y_true = (test_rows["satisfaction"] == "satisfied").astype(int)
    y_pred = (test_rows[proba_col] >= 0.5).astype(int)
    return {
        "n": len(test_rows),
        "accuracy": accuracy_score(y_true, y_pred),
        "recall_class0": recall_score(y_true, y_pred, pos_label=0),
        "precision_class0": precision_score(y_true, y_pred, pos_label=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "roc_auc": roc_auc_score(y_true, test_rows[proba_col]),
    }


AGE_GROUP_LABELS = {
    "minor": "Mineur (< 18)",
    "young_adult": "Jeune adulte (18-30)",
    "adult": "Adulte (30-45)",
    "mid_career": "Milieu de carrière (45-60)",
    "senior": "Senior (60+)",
}
