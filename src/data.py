"""Load, clean, and join the CRM Sales Opportunities tables.

The four raw CSVs in ``data/`` are read-only inputs. This module turns them into
one analysis table and splits off the rows that carry a Won/Lost label from the
deals that are still open.

Run as a script to write the processed tables and print a cleaning report::

    python -m src.data
    python -m src.data --data-dir data --out-dir data/processed
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DATE_FORMAT = "%m/%d/%y"
TARGET = "is_won"
POSITIVE_STAGE = "Won"
CLOSED_STAGES = ("Won", "Lost")

#: Known only after a deal closes -- never use these as model features.
LEAKAGE_COLUMNS = ("close_date", "close_value", "deal_stage", TARGET)

#: ``sales_pipeline.product`` is written without the space that ``products.product``
#: uses. Left unfixed, roughly one in seven product joins drops silently.
PRODUCT_NAME_FIXES = {"GTXPro": "GTX Pro"}

#: Misspellings in ``accounts.sector``.
SECTOR_FIXES = {"technolgy": "technology"}

CATEGORICAL_FEATURES = [
    "product",
    "series",
    "sector",
    "office_location",
    "regional_office",
    "manager",
    "sales_agent",
]

NUMERIC_FEATURES = [
    "sales_price",
    "revenue",
    "employees",
    "revenue_per_employee",
    "year_established",
    "account_age_at_engage",
    "engage_year",
    "engage_month",
    "engage_quarter",
    "engage_dayofweek",
    "is_subsidiary",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


@dataclass
class Dataset:
    """The joined CRM data, split by whether the outcome is known.

    Attributes:
        labeled: Won/Lost rows, with ``is_won`` set. Use these for train/test.
        open_deals: Engaging/Prospecting rows. No label -- scoring targets only.
        report: Counts collected while cleaning, for the write-up.
    """

    labeled: pd.DataFrame
    open_deals: pd.DataFrame
    report: dict[str, object]

    def features(self) -> pd.DataFrame:
        """Leakage-free feature columns of the labeled rows."""
        return self.labeled[FEATURE_COLUMNS]

    def target(self) -> pd.Series:
        """Binary label: 1 for Won, 0 for Lost."""
        return self.labeled[TARGET]


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------


def _strip_strings(frame: pd.DataFrame) -> pd.DataFrame:
    """Trim surrounding whitespace from every text column."""
    out = frame.copy()
    for column in out.columns:
        if out[column].dtype == object or isinstance(out[column].dtype, pd.StringDtype):
            out[column] = out[column].str.strip()
    return out


def _blank_to_na(frame: pd.DataFrame) -> pd.DataFrame:
    """Treat empty strings as missing values."""
    return frame.replace(r"^\s*$", np.nan, regex=True)


def load_raw(data_dir: Path | str = DATA_DIR) -> dict[str, pd.DataFrame]:
    """Read the four raw CSVs with whitespace trimmed and blanks as NaN."""
    data_dir = Path(data_dir)
    names = ["sales_pipeline", "accounts", "products", "sales_teams"]
    return {
        name: _blank_to_na(_strip_strings(pd.read_csv(data_dir / f"{name}.csv")))
        for name in names
    }


# --------------------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------------------


def clean_pipeline(pipeline: pd.DataFrame) -> pd.DataFrame:
    """Normalize product keys and parse the two date columns."""
    out = pipeline.copy()
    out["product"] = out["product"].replace(PRODUCT_NAME_FIXES)
    for column in ("engage_date", "close_date"):
        out[column] = pd.to_datetime(out[column], format=DATE_FORMAT, errors="coerce")
    return out


def clean_accounts(accounts: pd.DataFrame) -> pd.DataFrame:
    """Fix sector spellings and derive account-level attributes."""
    out = accounts.copy()
    out["sector"] = out["sector"].replace(SECTOR_FIXES)
    out["is_subsidiary"] = out["subsidiary_of"].notna().astype(int)
    # Revenue is in millions of USD; scale to dollars per head so the ratio reads sanely.
    out["revenue_per_employee"] = (out["revenue"] * 1_000_000 / out["employees"]).replace(
        [np.inf, -np.inf], np.nan
    )
    return out


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """Apply the same product-name normalization used on the pipeline table."""
    out = products.copy()
    out["product"] = out["product"].replace(PRODUCT_NAME_FIXES)
    return out


def clean_sales_teams(sales_teams: pd.DataFrame) -> pd.DataFrame:
    """No fixes needed today; kept so every table goes through one path."""
    return sales_teams.copy()


# --------------------------------------------------------------------------------------
# Joining and feature derivation
# --------------------------------------------------------------------------------------


def join_tables(
    pipeline: pd.DataFrame,
    accounts: pd.DataFrame,
    products: pd.DataFrame,
    sales_teams: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join the dimension tables onto the opportunity fact table."""
    joined = (
        pipeline.merge(products, on="product", how="left", validate="many_to_one")
        .merge(accounts, on="account", how="left", validate="many_to_one")
        .merge(sales_teams, on="sales_agent", how="left", validate="many_to_one")
    )
    if len(joined) != len(pipeline):
        raise AssertionError("join changed the row count; check for duplicate keys")
    return joined


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add calendar and account-age features from ``engage_date`` only.

    ``close_date`` is deliberately untouched: any duration that depends on it is
    unknown at prediction time.
    """
    out = frame.copy()
    engage = out["engage_date"]
    out["engage_year"] = engage.dt.year
    out["engage_month"] = engage.dt.month
    out["engage_quarter"] = engage.dt.quarter
    out["engage_dayofweek"] = engage.dt.dayofweek
    out["account_age_at_engage"] = out["engage_year"] - out["year_established"]
    return out


def add_label(frame: pd.DataFrame) -> pd.DataFrame:
    """Set ``is_won`` for closed deals; leave it missing for open ones."""
    out = frame.copy()
    is_closed = out["deal_stage"].isin(CLOSED_STAGES)
    out[TARGET] = np.where(is_closed, (out["deal_stage"] == POSITIVE_STAGE).astype("Int64"), pd.NA)
    out[TARGET] = out[TARGET].astype("Int64")
    return out


# --------------------------------------------------------------------------------------
# Pipeline entry point
# --------------------------------------------------------------------------------------


def build_dataset(data_dir: Path | str = DATA_DIR) -> Dataset:
    """Load, clean, join, and split the CRM tables into a modeling dataset."""
    raw = load_raw(data_dir)

    pipeline = clean_pipeline(raw["sales_pipeline"])
    accounts = clean_accounts(raw["accounts"])
    products = clean_products(raw["products"])
    sales_teams = clean_sales_teams(raw["sales_teams"])

    joined = join_tables(pipeline, accounts, products, sales_teams)
    joined = add_derived_features(joined)
    joined = add_label(joined)

    labeled = joined[joined["deal_stage"].isin(CLOSED_STAGES)].reset_index(drop=True)
    open_deals = joined[~joined["deal_stage"].isin(CLOSED_STAGES)].reset_index(drop=True)

    report = _build_report(raw, joined, labeled, open_deals)
    return Dataset(labeled=labeled, open_deals=open_deals, report=report)


def _build_report(
    raw: dict[str, pd.DataFrame],
    joined: pd.DataFrame,
    labeled: pd.DataFrame,
    open_deals: pd.DataFrame,
) -> dict[str, object]:
    """Collect the counts worth quoting in the paper and worth watching in CI."""
    won = int((labeled[TARGET] == 1).sum())
    missing = labeled[FEATURE_COLUMNS].isna().sum()
    return {
        "raw_rows": {name: len(frame) for name, frame in raw.items()},
        "joined_rows": len(joined),
        "labeled_rows": len(labeled),
        "open_rows": len(open_deals),
        "won": won,
        "lost": len(labeled) - won,
        "won_rate": round(won / len(labeled), 4) if len(labeled) else float("nan"),
        "unmatched_product": int(joined["series"].isna().sum()),
        "unmatched_account": int(joined["sector"].isna().sum()),
        "unmatched_agent": int(joined["regional_office"].isna().sum()),
        "labeled_missing_features": {k: int(v) for k, v in missing.items() if v},
    }


def _format_report(report: dict[str, object]) -> str:
    lines = [
        "Raw rows:            " + ", ".join(f"{k}={v}" for k, v in report["raw_rows"].items()),
        f"Joined rows:         {report['joined_rows']}",
        f"Labeled (Won/Lost):  {report['labeled_rows']}"
        f"  (won={report['won']}, lost={report['lost']}, won_rate={report['won_rate']})",
        f"Open (excluded):     {report['open_rows']}",
        f"Unmatched product:   {report['unmatched_product']}",
        f"Unmatched account:   {report['unmatched_account']}",
        f"Unmatched agent:     {report['unmatched_agent']}",
    ]
    missing = report["labeled_missing_features"]
    lines.append(
        "Missing features:    "
        + (", ".join(f"{k}={v}" for k, v in missing.items()) if missing else "none")
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR, help="directory of raw CSVs")
    parser.add_argument(
        "--out-dir", type=Path, default=PROCESSED_DIR, help="where to write processed CSVs"
    )
    args = parser.parse_args(argv)

    dataset = build_dataset(args.data_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    dataset.labeled.to_csv(args.out_dir / "labeled.csv", index=False)
    dataset.open_deals.to_csv(args.out_dir / "open_deals.csv", index=False)

    print(_format_report(dataset.report))
    print(f"\nWrote {args.out_dir / 'labeled.csv'} and {args.out_dir / 'open_deals.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
