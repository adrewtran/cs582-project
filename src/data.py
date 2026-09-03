"""Shared dataset container and cleaning helpers.

Every loader in :mod:`src.datasets` returns a :class:`Dataset`. Everything
downstream -- preprocessing, baselines, the advanced models, explainability --
works from that object and never touches a raw file.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"


@dataclass
class Dataset:
    """A cleaned, labeled table plus the column roles a model needs.

    Attributes:
        name: short id, also the sub-directory under ``data/`` and ``reports/``.
        frame: one row per labeled example; contains the target and every column
            in ``categorical`` / ``numeric``. May carry extra columns for EDA --
            ``features()`` never returns those.
        target: name of the binary 0/1 label column.
        categorical: columns to one-hot encode. Missing values are already a
            literal ``"Missing"`` category where missingness is informative.
        numeric: columns to impute (median) and optionally scale.
        report: counts collected while cleaning, for the write-up.
        extra: anything else worth keeping (e.g. the CRM's open deals).
    """

    name: str
    frame: pd.DataFrame
    target: str
    categorical: list[str]
    numeric: list[str]
    report: dict[str, object] = field(default_factory=dict)
    extra: dict[str, object] = field(default_factory=dict)

    @property
    def feature_columns(self) -> list[str]:
        return self.categorical + self.numeric

    def features(self) -> pd.DataFrame:
        return self.frame[self.feature_columns]

    def labels(self) -> pd.Series:
        return self.frame[self.target].astype(int)

    def __post_init__(self) -> None:
        missing = [c for c in self.feature_columns + [self.target] if c not in self.frame]
        if missing:
            raise ValueError(f"{self.name}: columns missing from frame: {missing}")
        overlap = set(self.categorical) & set(self.numeric)
        if overlap:
            raise ValueError(f"{self.name}: columns listed as both categorical and numeric: {overlap}")
        bad = set(self.frame[self.target].dropna().unique()) - {0, 1}
        if bad:
            raise ValueError(f"{self.name}: target must be 0/1, found {bad}")


# --------------------------------------------------------------------------------------
# Helpers shared by the loaders
# --------------------------------------------------------------------------------------


def is_text(series: pd.Series) -> bool:
    return series.dtype == object or isinstance(series.dtype, pd.StringDtype)


def strip_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from every text column."""
    out = frame.copy()
    for column in out.columns:
        if is_text(out[column]):
            # map, not .str.strip(): a mixed object column (numbers + blank strings)
            # would have its numbers turned into NaN by the .str accessor.
            out[column] = out[column].map(lambda v: v.strip() if isinstance(v, str) else v)
    return out


def blank_to_na(frame: pd.DataFrame, placeholders: tuple[str, ...] = ()) -> pd.DataFrame:
    """Treat empty strings -- and any listed placeholder tokens -- as missing."""
    out = frame.replace(r"^\s*$", np.nan, regex=True)
    if placeholders:
        out = out.replace(list(placeholders), np.nan)
    return out


def collapse_rare(series: pd.Series, top_k: int, other: str = "Other") -> pd.Series:
    """Keep the ``top_k`` most frequent levels, fold the rest into ``other``."""
    keep = series.value_counts().head(top_k).index
    return series.where(series.isin(keep) | series.isna(), other)


def fill_missing_category(frame: pd.DataFrame, columns: list[str], token: str = "Missing") -> pd.DataFrame:
    """Make missingness an explicit category on the given columns."""
    out = frame.copy()
    for column in columns:
        out[column] = out[column].astype("object").where(out[column].notna(), token).astype(str)
    return out


def yes_no_to_int(series: pd.Series) -> pd.Series:
    return series.map({"Yes": 1, "No": 0, "yes": 1, "no": 0}).astype("Int64")


def basic_report(frame: pd.DataFrame, target: str) -> dict[str, object]:
    positives = int(frame[target].sum())
    return {
        "rows": len(frame),
        "positives": positives,
        "negatives": len(frame) - positives,
        "positive_rate": round(positives / len(frame), 4) if len(frame) else float("nan"),
    }
