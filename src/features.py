"""Preprocessing shared by every model.

One-hot encodes the dataset's categorical columns; median-imputes and
(optionally) standardizes the numeric ones. Fitted inside each model's
``Pipeline``, so it only ever sees the training fold.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import Dataset


def make_preprocessor(dataset: Dataset, scale_numeric: bool) -> ColumnTransformer:
    """Build the column transformer for ``dataset``.

    Args:
        scale_numeric: standardize numeric columns. Needed for Logistic
            Regression and the neural models; pointless for tree ensembles.
    """
    numeric_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), dataset.categorical),
            ("num", Pipeline(numeric_steps), dataset.numeric),
        ],
        verbose_feature_names_out=False,
    )
