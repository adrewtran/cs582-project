"""Train and evaluate the baseline classifiers: Logistic Regression and Random Forest.

Both share the preprocessing from :mod:`src.features` and the fixed split below,
so their numbers -- and those of the MLP / TabNet models that come later -- are
directly comparable.

    python -m src.baselines leads
    python -m src.baselines bank --split temporal
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, accuracy_score, average_precision_score,
    balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

from src.data import Dataset
from src.datasets import LOADERS, load
from src.features import make_preprocessor

RANDOM_STATE = 42
DEFAULT_DATASET = "crm"
TEST_SIZE = 0.2
CV_FOLDS = 5

REPORTS_DIR = Path(__file__).resolve().parents[1] / "reports"

METRIC_COLUMNS = [
    "accuracy", "balanced_accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc",
    "cv_roc_auc_mean", "cv_roc_auc_std",
]


def make_models(dataset: Dataset) -> dict[str, Pipeline]:
    """The two baselines, each wrapped with its own preprocessing."""
    return {
        "dummy_majority": Pipeline([
            ("prep", make_preprocessor(dataset, scale_numeric=False)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ]),
        "logistic_regression": Pipeline([
            ("prep", make_preprocessor(dataset, scale_numeric=True)),
            ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE)),
        ]),
        "random_forest": Pipeline([
            ("prep", make_preprocessor(dataset, scale_numeric=False)),
            ("clf", RandomForestClassifier(
                n_estimators=500, min_samples_leaf=2, class_weight="balanced",
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
    }


def split(dataset: Dataset, how: str = "random") -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Hold-out split. Same seed everywhere so every model sees the same rows.

    ``random``: stratified 80/20. ``temporal``: the last 20% of rows by original
    order become the test set -- only meaningful when the loader kept the rows in
    date order (``report["row_order_is_temporal"]``).
    """
    X, y = dataset.features(), dataset.labels()
    if how == "random":
        return train_test_split(X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    if how == "temporal":
        if not dataset.report.get("row_order_is_temporal"):
            raise ValueError(f"{dataset.name}: rows are not in time order; a temporal split is meaningless")
        cut = int(len(X) * (1 - TEST_SIZE))
        return X.iloc[:cut], X.iloc[cut:], y.iloc[:cut], y.iloc[cut:]
    raise ValueError(f"unknown split {how!r}")


def evaluate(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    """The proposal's metric set, plus PR-AUC and balanced accuracy for imbalanced sets."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "pr_auc": average_precision_score(y_test, y_prob),
    }


def cross_validate_auc(model: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=-1)
    return float(scores.mean()), float(scores.std())


def feature_importance(name: str, model: Pipeline) -> pd.Series | None:
    """Per-feature contribution after one-hot encoding, largest first.

    Random Forest: impurity importances. Logistic Regression: absolute
    standardized coefficients (comparable only because numerics were scaled).
    """
    names = model.named_steps["prep"].get_feature_names_out()
    clf = model.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        values = clf.feature_importances_
    elif hasattr(clf, "coef_"):
        values = np.abs(clf.coef_.ravel())
    else:
        return None
    return pd.Series(values, index=names, name=name).sort_values(ascending=False)


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------


def _title(name: str) -> str:
    return name.replace("_", " ").title()


def save_confusion_matrix(out_dir: Path, name: str, model, X_test, y_test) -> Path:
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay.from_estimator(model, X_test, y_test, colorbar=False, ax=ax)
    ax.set_title(_title(name))
    path = out_dir / "figures" / f"confusion_matrix_{name}.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def save_roc_curves(out_dir: Path, models: dict[str, Pipeline], X_test, y_test) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    for name, model in models.items():
        RocCurveDisplay.from_estimator(model, X_test, y_test, name=_title(name), ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", linewidth=1)
    ax.set_title(f"ROC curves (hold-out) - {out_dir.relative_to(REPORTS_DIR)}")
    path = out_dir / "figures" / "roc_curves.png"
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    return path


def save_feature_importance(out_dir: Path, importance: pd.Series, top_n: int = 20) -> None:
    name = importance.name
    importance.rename("importance").to_csv(out_dir / f"feature_importance_{name}.csv", index_label="feature")
    top = importance.head(top_n)[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.3 * len(top) + 1))
    ax.barh(top.index, top.values)
    ax.set_title(f"Top {len(top)} features - {_title(name)}")
    fig.tight_layout(); fig.savefig(out_dir / "figures" / f"feature_importance_{name}.png", dpi=150); plt.close(fig)


def run(dataset: Dataset, how: str = "random") -> pd.DataFrame:
    """Fit both baselines on ``dataset``, write every report artefact, return the metrics table."""
    out_dir = REPORTS_DIR / dataset.name / ("temporal" if how == "temporal" else "")
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    X_train, X_test, y_train, y_test = split(dataset, how)
    print(f"split={how}: train={len(X_train)} (pos {y_train.mean():.3f})  test={len(X_test)} (pos {y_test.mean():.3f})")

    models = make_models(dataset)
    rows = []
    for name, model in models.items():
        cv_mean, cv_std = cross_validate_auc(model, X_train, y_train)
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        metrics.update(cv_roc_auc_mean=cv_mean, cv_roc_auc_std=cv_std)
        rows.append({"model": name, **metrics})

        importance = feature_importance(name, model)
        save_confusion_matrix(out_dir, name, model, X_test, y_test)
        if importance is not None:
            save_feature_importance(out_dir, importance)

        print(f"\n== {name}")
        print("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
        print(f"  confusion matrix [[TN FP] [FN TP]]: {confusion_matrix(y_test, model.predict(X_test)).tolist()}")
        if importance is not None:
            print("  top features: " + ", ".join(importance.head(6).index))

    save_roc_curves(out_dir, models, X_test, y_test)
    table = pd.DataFrame(rows).set_index("model")[METRIC_COLUMNS]
    table.to_csv(out_dir / "baseline_metrics.csv")
    return table


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dataset", nargs="?", default=DEFAULT_DATASET, choices=sorted(LOADERS))
    parser.add_argument("--split", default="random", choices=["random", "temporal"])
    args = parser.parse_args(argv)

    dataset = load(args.dataset)
    print(f"{dataset.name}: {dataset.report['rows']} rows, positive rate {dataset.report['positive_rate']}, "
          f"{len(dataset.categorical)} categorical + {len(dataset.numeric)} numeric features")
    table = run(dataset, args.split)
    print("\n" + table.round(4).to_string())
    print(f"\nWrote reports to {REPORTS_DIR / dataset.name}{'/temporal' if args.split == 'temporal' else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
