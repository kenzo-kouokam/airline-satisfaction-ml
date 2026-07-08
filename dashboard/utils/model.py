# -*- coding: utf-8 -*-
"""Builds a single-passenger feature row (same engineering as training) and scores it."""
import pandas as pd

AGE_BINS = [0, 18, 30, 45, 60, 100]
AGE_LABELS = ["minor", "young_adult", "adult", "mid_career", "senior"]


def build_input_row(inputs: dict, service_cols: list[str]) -> pd.DataFrame:
    row = dict(inputs)
    row["loyal_business"] = int(
        row["Customer Type"] == "Loyal Customer" and row["Type of Travel"] == "Business travel"
    )
    row["mean_service_score"] = sum(row[c] for c in service_cols) / len(service_cols)
    age_group = pd.cut([row["Age"]], bins=AGE_BINS, labels=AGE_LABELS)[0]
    row["age_group"] = age_group
    return pd.DataFrame([row])


def predict(pipeline, input_row: pd.DataFrame) -> float:
    """Returns P(satisfied) for a single-row dataframe."""
    return float(pipeline.predict_proba(input_row)[:, 1][0])


def lr_local_contributions(lr_pipeline, input_row: pd.DataFrame, top_n: int = 8) -> pd.DataFrame:
    """Per-feature contribution (coefficient * standardized value) for THIS passenger,
    using the logistic regression's linear decision function."""
    preprocess = lr_pipeline.named_steps["preprocess"]
    clf = lr_pipeline.named_steps["clf"]
    X_transformed = preprocess.transform(input_row)
    if hasattr(X_transformed, "toarray"):
        X_transformed = X_transformed.toarray()
    contributions = X_transformed[0] * clf.coef_[0]
    feature_names = preprocess.get_feature_names_out()
    contrib_df = pd.DataFrame({"feature": feature_names, "contribution": contributions})
    contrib_df["abs"] = contrib_df["contribution"].abs()
    contrib_df = contrib_df[contrib_df["abs"] > 1e-6].sort_values("abs", ascending=False).head(top_n)
    return contrib_df.drop(columns="abs").sort_values("contribution")
