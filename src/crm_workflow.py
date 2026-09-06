"""CRM-specific training and open-opportunity scoring helpers."""

from __future__ import annotations

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


def score_open_deals(model: Pipeline, dataset: Dataset) -> pd.DataFrame:
    """Score CRM deals that have entered Engaging and retain IDs for reporting."""
    if dataset.name != "crm":
        raise ValueError("open-deal scoring is defined only for the CRM dataset")

    open_deals = dataset.extra["scorable_open_deals"]
    probabilities = model.predict_proba(open_deals[dataset.feature_columns])[:, 1]
    scored = open_deals[["opportunity_id", "deal_stage", "engage_date", "account", "product", "sales_agent"]].copy()
    scored["win_probability"] = probabilities
    scored["priority"] = [priority_group(float(value)) for value in probabilities]
    return scored.sort_values("win_probability", ascending=False, kind="stable").reset_index(drop=True)
