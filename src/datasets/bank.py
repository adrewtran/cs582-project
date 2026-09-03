"""UCI Bank Marketing (bank-additional-full) -- the secondary dataset.

41,188 telemarketing contacts, target ``y`` (11.3% positive). Rows are in date
order (May 2008 -> Nov 2010), which the temporal split in :mod:`src.baselines`
relies on -- do not shuffle here.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, Dataset, basic_report, strip_strings

BANK_FILE = DATA_DIR / "bank" / "bank-additional" / "bank-additional-full.csv"
TARGET = "subscribed"

#: Call duration is unknown before the call and near-determines the outcome
#: (AUC 0.95 with, 0.80 without). The dataset's own README says to drop it.
LEAKAGE_COLUMNS = ("duration",)

#: ``pdays = 999`` is the sentinel for "never contacted in a previous campaign".
PDAYS_NEVER = 999

CATEGORICAL = ["job", "marital", "education", "default", "housing", "loan", "contact", "month", "day_of_week", "poutcome"]
NUMERIC = [
    "age", "campaign", "pdays", "previous", "never_contacted",
    "emp.var.rate", "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed",
]


def load_raw(path: Path = BANK_FILE) -> pd.DataFrame:
    return strip_strings(pd.read_csv(path, sep=";"))


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = raw.drop(columns=list(LEAKAGE_COLUMNS)).copy()
    never = frame["pdays"] == PDAYS_NEVER
    frame["never_contacted"] = never.astype(int)
    frame["pdays"] = frame["pdays"].where(~never, np.nan)
    frame[TARGET] = (frame.pop("y") == "yes").astype(int)
    report = {
        "dropped_leakage": list(LEAKAGE_COLUMNS),
        "never_contacted_rows": int(never.sum()),
        "unknown_counts": {c: int((frame[c] == "unknown").sum()) for c in CATEGORICAL if (frame[c] == "unknown").any()},
    }
    return frame, report


def build(path: Path = BANK_FILE) -> Dataset:
    frame, report = clean(load_raw(path))
    report = basic_report(frame, TARGET) | report | {"row_order_is_temporal": True}
    return Dataset("bank", frame, TARGET, CATEGORICAL, NUMERIC, report)
