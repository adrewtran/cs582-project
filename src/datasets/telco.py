"""IBM Telco Customer Churn -- optional third dataset.

7,043 customers, target ``Churn Value`` (26.5% positive). Framing is churn rather
than sales; it exists as a third point for the cross-dataset explainability
comparison.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data import DATA_DIR, Dataset, basic_report, blank_to_na, strip_strings

TELCO_FILE = DATA_DIR / "telco" / "Telco_customer_churn.xlsx"
TARGET = "churn"

#: IBM's own churn model output, lifetime value computed with hindsight, the
#: reason given on leaving, and a duplicate of the target.
LEAKAGE_COLUMNS = ("Churn Score", "CLTV", "Churn Reason", "Churn Label")

#: Identifiers and geography (every row is California).
DROP_COLUMNS = ("CustomerID", "Count", "Country", "State", "City", "Zip Code", "Lat Long", "Latitude", "Longitude")

CATEGORICAL = [
    "Gender", "Senior Citizen", "Partner", "Dependents", "Phone Service", "Multiple Lines",
    "Internet Service", "Online Security", "Online Backup", "Device Protection", "Tech Support",
    "Streaming TV", "Streaming Movies", "Contract", "Paperless Billing", "Payment Method",
]
NUMERIC = ["Tenure Months", "Monthly Charges", "Total Charges"]


def _snake(name: str) -> str:
    return name.strip().lower().replace(" ", "_")


def load_raw(path: Path = TELCO_FILE) -> pd.DataFrame:
    return blank_to_na(strip_strings(pd.read_excel(path)))


def clean(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    frame = raw.drop(columns=list(LEAKAGE_COLUMNS) + list(DROP_COLUMNS)).copy()
    frame["Total Charges"] = pd.to_numeric(frame["Total Charges"], errors="coerce")
    frame = frame.rename(columns={"Churn Value": TARGET})
    frame = frame.rename(columns={c: _snake(c) for c in frame.columns})
    report = {
        "dropped_leakage": list(LEAKAGE_COLUMNS),
        "dropped_id_geo": list(DROP_COLUMNS),
        "total_charges_blank": int(frame["total_charges"].isna().sum()),
    }
    return frame, report


def build(path: Path = TELCO_FILE) -> Dataset:
    frame, report = clean(load_raw(path))
    report = basic_report(frame, TARGET) | report
    return Dataset("telco", frame, TARGET, [_snake(c) for c in CATEGORICAL], [_snake(c) for c in NUMERIC], report)
