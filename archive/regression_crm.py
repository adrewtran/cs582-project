"""ARCHIVED (CRM only). Regression variant: predict ``close_value`` for Won deals.

Result: close_value = sales_price x iid noise; nothing beats predicting the list
price (R2 0.985). Kept for the paper's leakage-audit appendix.

    python -m archive.regression_crm

Same split seed and preprocessing as :mod:`src.baselines`. Includes a naive
"predict the product's list price" baseline because, on this dataset, that single
feature explains nearly all of the variance -- any model has to be judged against
it, not against the mean.

"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.baselines import RANDOM_STATE, REPORTS_DIR as _REPORTS, TEST_SIZE
from src.data import Dataset
from src.datasets import load
from src.features import make_preprocessor

REPORTS_DIR = _REPORTS / "crm"
FIGURES_DIR = REPORTS_DIR / "figures"

TARGET = "close_value"
METRIC_COLUMNS = ["mae", "rmse", "mape", "r2"]


def won_deals(dataset: Dataset) -> pd.DataFrame:
    return dataset.frame[dataset.frame["is_won"] == 1].reset_index(drop=True)


def split(won: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold-out split stratified by product so every price tier appears in both folds."""
    return train_test_split(
        won[DATASET.feature_columns], won[TARGET], test_size=TEST_SIZE,
        stratify=won["product"], random_state=RANDOM_STATE,
    )


def make_models() -> dict[str, Pipeline]:
    return {
        "ridge_log_target": Pipeline(
            [
                ("prep", make_preprocessor(DATASET, scale_numeric=True)),
                (
                    "reg",
                    TransformedTargetRegressor(
                        regressor=Ridge(alpha=1.0), func=np.log1p, inverse_func=np.expm1
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("prep", make_preprocessor(DATASET, scale_numeric=False)),
                (
                    "reg",
                    RandomForestRegressor(
                        n_estimators=500, min_samples_leaf=5, n_jobs=-1, random_state=RANDOM_STATE
                    ),
                ),
            ]
        ),
    }


def evaluate(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


def save_pred_vs_actual(preds: dict[str, np.ndarray], y_test: pd.Series) -> Path:
    fig, axes = plt.subplots(1, len(preds), figsize=(4.5 * len(preds), 4.5), sharey=True)
    for ax, (name, y_pred) in zip(np.atleast_1d(axes), preds.items()):
        ax.scatter(y_test, y_pred, s=6, alpha=0.4)
        lim = [0, max(y_test.max(), y_pred.max()) * 1.05]
        ax.plot(lim, lim, "--", color="grey", linewidth=1)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_title(name.replace("_", " ").title())
        ax.set_xlabel("actual close_value")
    np.atleast_1d(axes)[0].set_ylabel("predicted close_value")
    path = FIGURES_DIR / "regression_pred_vs_actual.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


DATASET: Dataset | None = None


def run(dataset: Dataset | None = None) -> pd.DataFrame:
    global DATASET
    DATASET = dataset = dataset or load("crm")
    X_train, X_test, y_train, y_test = split(won_deals(dataset))
    REPORTS_DIR.mkdir(exist_ok=True); FIGURES_DIR.mkdir(exist_ok=True)

    rows, preds = [], {}
    # Naive baselines the learned models must beat.
    preds["naive_mean"] = np.full(len(y_test), y_train.mean())
    preds["naive_list_price"] = X_test["sales_price"].to_numpy(dtype=float)
    for name, model in make_models().items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
    for name, y_pred in preds.items():
        rows.append({"model": name, **evaluate(y_test, y_pred)})

    save_pred_vs_actual({k: v for k, v in preds.items() if not k.startswith("naive_mean")}, y_test)
    table = pd.DataFrame(rows).set_index("model")[METRIC_COLUMNS]
    table.to_csv(REPORTS_DIR / "regression_metrics.csv")
    return table


def main(argv: list[str] | None = None) -> int:
    argparse.ArgumentParser(description=__doc__.splitlines()[0]).parse_args(argv)
    table = run()
    print(table.round(4).to_string())
    print(f"\nWrote reports to {REPORTS_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
