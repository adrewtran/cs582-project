"""Maven CRM Sales Opportunities -- retired as the modeling dataset.

Kept for the paper's leakage-audit appendix: every attribute is independent of
the outcome (leakage-free ROC-AUC ~0.5), while ``close_value`` is 0 for every
Lost deal and > 0 for every Won deal, i.e. it is the label in disguise.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.data import DATA_DIR, Dataset, basic_report, blank_to_na, strip_strings

CRM_DIR = DATA_DIR / "crm"
DATE_FORMAT = "%m/%d/%y"
TARGET = "is_won"
CLOSED_STAGES = ("Won", "Lost")

#: Known only after a deal closes -- never model features.
LEAKAGE_COLUMNS = ("close_date", "close_value", "deal_stage")

#: ``sales_pipeline.product`` lacks the space that ``products.product`` uses.
PRODUCT_NAME_FIXES = {"GTXPro": "GTX Pro"}
SECTOR_FIXES = {"technolgy": "technology"}

CATEGORICAL = ["product", "series", "sector", "office_location", "regional_office", "manager", "sales_agent"]
NUMERIC = [
    "sales_price", "revenue", "employees", "revenue_per_employee", "year_established",
    "account_age_at_engage", "engage_year", "engage_month", "engage_quarter", "engage_dayofweek",
    "is_subsidiary",
]


def load_raw(data_dir: Path = CRM_DIR) -> dict[str, pd.DataFrame]:
    names = ["sales_pipeline", "accounts", "products", "sales_teams"]
    return {n: blank_to_na(strip_strings(pd.read_csv(data_dir / f"{n}.csv"))) for n in names}


def clean_pipeline(pipeline: pd.DataFrame) -> pd.DataFrame:
    out = pipeline.copy()
    out["product"] = out["product"].replace(PRODUCT_NAME_FIXES)
    for column in ("engage_date", "close_date"):
        out[column] = pd.to_datetime(out[column], format=DATE_FORMAT, errors="coerce")
    return out


def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    out = accounts.copy()
    out["sector"] = out["sector"].replace(SECTOR_FIXES)
    out["is_subsidiary"] = out["subsidiary_of"].notna().astype(int)
    out["revenue_per_employee"] = (out["revenue"] * 1_000_000 / out["employees"]).replace([np.inf, -np.inf], np.nan)
    return out


def join_tables(pipeline, accounts, products, sales_teams) -> pd.DataFrame:
    products = products.assign(product=products["product"].replace(PRODUCT_NAME_FIXES))
    joined = (
        pipeline.merge(products, on="product", how="left", validate="many_to_one")
        .merge(accounts, on="account", how="left", validate="many_to_one")
        .merge(sales_teams, on="sales_agent", how="left", validate="many_to_one")
    )
    if len(joined) != len(pipeline):
        raise AssertionError("join changed the row count; check for duplicate keys")
    return joined


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Calendar and account-age features from ``engage_date`` only -- never ``close_date``."""
    out = frame.copy()
    engage = out["engage_date"]
    out["engage_year"] = engage.dt.year
    out["engage_month"] = engage.dt.month
    out["engage_quarter"] = engage.dt.quarter
    out["engage_dayofweek"] = engage.dt.dayofweek
    out["account_age_at_engage"] = out["engage_year"] - out["year_established"]
    return out


def build(data_dir: Path = CRM_DIR) -> Dataset:
    raw = load_raw(data_dir)
    joined = join_tables(clean_pipeline(raw["sales_pipeline"]), clean_accounts(raw["accounts"]),
                         raw["products"], raw["sales_teams"])
    joined = add_derived_features(joined)
    closed = joined["deal_stage"].isin(CLOSED_STAGES)
    joined[TARGET] = np.where(closed, (joined["deal_stage"] == "Won").astype(int), np.nan)

    labeled = joined[closed].reset_index(drop=True)
    labeled[TARGET] = labeled[TARGET].astype(int)
    open_deals = joined[~closed].drop(columns=TARGET).reset_index(drop=True)

    report = basic_report(labeled, TARGET) | {
        "raw_rows": {k: len(v) for k, v in raw.items()},
        "open_rows": len(open_deals),
        "unmatched_product": int(joined["series"].isna().sum()),
        "unmatched_account_in_labeled": int(labeled["sector"].isna().sum()),
    }
    return Dataset("crm", labeled, TARGET, CATEGORICAL, NUMERIC, report, {"open_deals": open_deals})
