"""CRM-specific training and open-opportunity scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.data import Dataset


def priority_group(probability: float) -> str:
    """Convert a win probability into a presentation-friendly priority label."""
    if not 0 <= probability <= 1:
        raise ValueError("probability must be between 0 and 1")
    if probability >= 0.70:
        return "High"
    if probability >= 0.40:
        return "Medium"
    return "Low"


def _display_value(value: object) -> str:
    if pd.isna(value):
        return "Missing"
    if isinstance(value, (float, np.floating)):
        return f"{value:.2f}"
    return str(value)


def _linear_factor_labels(model: Pipeline, dataset: Dataset, rows: pd.DataFrame) -> list[list[str]]:
    """Return a human-readable label for every transformed feature in every row."""
    prep = model.named_steps["prep"]
    encoder = prep.named_transformers_["cat"].named_steps['encode']
    metadata: list[tuple[str, object | None]] = []
    for column, categories in zip(dataset.categorical, encoder.categories_):
        metadata.extend((column, category) for category in categories)
    metadata.extend((column, None) for column in dataset.numeric)

    labels = []
    for _, row in rows.iterrows():
        labels.append([
            f"{column}={_display_value(category if category is not None else row[column])}"
            for column, category in metadata
        ])
    return labels


def _linear_decision_factors(
    model: Pipeline,
    dataset: Dataset,
    rows: pd.DataFrame,
    top_n: int,
) -> pd.DataFrame:
    """Explain a binary linear classifier with exact feature log-odds contributions."""
    clf = model.named_steps["clf"]
    if not hasattr(clf, "coef_") or np.asarray(clf.coef_).shape[0] != 1:
        raise ValueError("decision-factor explanations require a fitted binary linear classifier")
    if top_n < 1:
        raise ValueError("top_n_factors must be at least 1")

    transformed = np.asarray(model.named_steps["prep"].transform(rows))
    contributions = transformed * np.asarray(clf.coef_).reshape(1, -1)
    labels = _linear_factor_labels(model, dataset, rows)
    explained_rows = []
    for values, row_labels in zip(contributions, labels):
        positive = np.flatnonzero(values > 0)
        negative = np.flatnonzero(values < 0)
        positive = positive[np.argsort(values[positive])[::-1]][:top_n]
        negative = negative[np.argsort(values[negative])][:top_n]
        explained_rows.append({
            'intercept_log_odds': float(clf.intercept_[0]),
            'sum_feature_log_odds': float(values.sum()),
            'model_log_odds': float(clf.intercept_[0]+values.sum()),
            "positive_factors": "; ".join(
                f"{row_labels[index]} ({values[index]:+.3f})" for index in positive
            ),
            "negative_factors": "; ".join(
                f"{row_labels[index]} ({values[index]:+.3f})" for index in negative
            ),
        })
    return pd.DataFrame(explained_rows, index=rows.index)


def score_open_deals(
    model: Pipeline,
    dataset: Dataset,
    *,
    include_explanations: bool = False,
    top_n_factors: int = 3,
) -> pd.DataFrame:
    """Score Engaging CRM deals and optionally explain a linear model's decision."""
    if dataset.name != "crm":
        raise ValueError("open-deal scoring is defined only for the CRM dataset")

    open_deals = dataset.extra["scorable_open_deals"]
    if open_deals.empty:
        columns=['opportunity_id','deal_stage','engage_date','account','product','sales_agent',
                 'win_probability','loss_probability','predicted_outcome','priority']
        if include_explanations:
            columns+=['positive_factors','negative_factors','intercept_log_odds','sum_feature_log_odds','model_log_odds']
        return pd.DataFrame(columns=columns)
    features = open_deals[dataset.feature_columns]
    probabilities = model.predict_proba(features)[:, 1]
    scored = open_deals[["opportunity_id", "deal_stage", "engage_date", "account", "product", "sales_agent"]].copy()
    scored["win_probability"] = probabilities
    scored["loss_probability"] = 1.0 - probabilities
    scored["predicted_outcome"] = np.where(probabilities >= 0.5, "Won", "Lost")
    scored["priority"] = [priority_group(float(value)) for value in probabilities]
    if include_explanations:
        explanations = _linear_decision_factors(model, dataset, features, top_n_factors)
        scored = pd.concat([scored, explanations], axis=1)
    return scored.sort_values("win_probability", ascending=False, kind="stable").reset_index(drop=True)
