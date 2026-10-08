"""Extend the real product catalog with synthetic products for TabNet stress tests.

Reads data/crm/products.csv (read-only) and writes data/crm_synthetic/products.csv.
Synthetic rows are flagged so they are never mixed into final results.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "crm" / "products.csv"
OUT = ROOT / "data" / "crm_synthetic" / "products.csv"

# Extra tiers inside existing series plus a new GTS series (price spread covers ~25 to ~52k).
NEW_PRODUCTS = [
    ("GTX Lite", "GTX", 250),
    ("GTX Ultra", "GTX", 9800),
    ("MG Lite", "MG", 25),
    ("MG Elite", "MG", 7200),
    ("GTK 250", "GTK", 13400),
    ("GTK 1000", "GTK", 52000),
    ("GTS Starter", "GTS", 1500),
    ("GTS Standard", "GTS", 3200),
    ("GTS Enterprise", "GTS", 15000),
]


def main() -> None:
    real = pd.read_csv(SRC).assign(synthetic=False)
    new = pd.DataFrame(NEW_PRODUCTS, columns=["product", "series", "sales_price"]).assign(synthetic=True)
    out = pd.concat([real, new], ignore_index=True)
    assert out["product"].is_unique, "duplicate product names"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
