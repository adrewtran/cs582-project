"""X Education Lead Scoring -- the primary dataset.

9,240 leads, target ``Converted`` (38.5% positive). See PLAN.md "Cleaning sheets"
for the reasoning behind every rule below.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import (
    DATA_DIR, Dataset, basic_report, blank_to_na, collapse_rare, fill_missing_category,
    is_text, strip_strings, yes_no_to_int,
)

LEADS_FILE = DATA_DIR / "leads" / "Leads X Education.csv"
TARGET = "converted"

#: Web-form placeholder that means "left blank".
PLACEHOLDERS = ("Select",)

ID_COLUMNS = ("Prospect ID", "Lead Number")

#: Sales-team annotations written *after* the lead was contacted. Each one
#: near-determines the outcome (e.g. Tags "Will revert after reading the email"
#: converts 97%, "Ringing" 3%). Never features.
LEAKAGE_COLUMNS = (
    "Tags",
    "Lead Quality",
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
    "Last Notable Activity",
)

#: Looks like a rep's assessment (Potential Lead converts 79%, Student of
#: SomeSchool 4%). Kept by default; flip ``include_lead_profile`` to compare.
BORDERLINE_COLUMNS = ("Lead Profile",)

#: A binary flag with fewer minority rows than this carries nothing learnable.
MIN_MINORITY_ROWS = 10

#: Long-tailed categoricals folded to top-k + Other before one-hot.
TOP_K = {"Country": 5, "Lead Source": 8}

NUMERIC = ["TotalVisits", "Total Time Spent on Website", "Page Views Per Visit"]


def load_raw(path: Path = LEADS_FILE) -> pd.DataFrame:
    return blank_to_na(strip_strings(pd.read_csv(path)), PLACEHOLDERS)


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("/", "_")


def clean(raw: pd.DataFrame, include_lead_profile: bool = True) -> tuple[pd.DataFrame, list[str], list[str], dict]:
    frame = raw.drop(columns=list(ID_COLUMNS) + list(LEAKAGE_COLUMNS))
    if not include_lead_profile:
        frame = frame.drop(columns=list(BORDERLINE_COLUMNS))

    # Constant columns carry nothing; drop them before anything else.
    constant = [c for c in frame.columns if frame[c].nunique(dropna=True) <= 1]
    frame = frame.drop(columns=constant)

    # Yes/No flags -> 0/1, dropping the ones that are effectively constant.
    yes_no = [c for c in frame.columns if is_text(frame[c]) and set(frame[c].dropna().unique()) <= {"Yes", "No"}]
    near_constant = [c for c in yes_no if frame[c].value_counts().min() < MIN_MINORITY_ROWS]
    frame = frame.drop(columns=near_constant)
    binary = [c for c in yes_no if c not in near_constant]
    for column in binary:
        frame[column] = yes_no_to_int(frame[column])

    for column, k in TOP_K.items():
        frame[column] = collapse_rare(frame[column], k)

    categorical = [c for c in frame.columns if is_text(frame[c]) and c != "Converted"]
    missing_before = frame[categorical].isna().mean().round(3)
    frame = fill_missing_category(frame, categorical)

    frame = frame.rename(columns={"Converted": TARGET})
    frame = frame.rename(columns={c: _snake(c) for c in frame.columns})
    categorical = [_snake(c) for c in categorical]
    numeric = [_snake(c) for c in NUMERIC + binary]

    report = {
        "dropped_constant": constant,
        "dropped_near_constant_flags": near_constant,
        "dropped_leakage": list(LEAKAGE_COLUMNS),
        "lead_profile_included": include_lead_profile,
        "categorical_missing_rate": {k: float(v) for k, v in missing_before.items() if v > 0},
        "numeric_missing_rows": {c: int(frame[_snake(c)].isna().sum()) for c in NUMERIC},
    }
    return frame, categorical, numeric, report


def build(path: Path = LEADS_FILE, include_lead_profile: bool = True) -> Dataset:
    frame, categorical, numeric, report = clean(load_raw(path), include_lead_profile)
    report = basic_report(frame, TARGET) | report
    return Dataset("leads", frame, TARGET, categorical, numeric, report)
