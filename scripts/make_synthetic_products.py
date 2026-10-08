"""Extend the real product catalog with synthetic products for TabNet stress tests.

Reads data/crm/products.csv (read-only) and writes data/crm_synthetic/products.csv.
Synthetic rows are flagged so they are never mixed into final results.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "crm" / "products.csv"
OUT = ROOT / "data" / "crm_synthetic" / "products.csv"

# Target industry per product, using the account `sector` labels (raw spelling, e.g. "technolgy").
# Every one of the 10 sectors gets at least one product so product x sector effects can be planted.
INDUSTRY = {
    "GTX Basic": "retail", "GTX Plus Basic": "services", "GTX Pro": "marketing",
    "GTX Plus Pro": "finance", "MG Special": "employment", "MG Advanced": "entertainment",
    "GTK 500": "telecommunications", "GTX Lite": "retail", "GTX Ultra": "technolgy",
    "MG Lite": "employment", "MG Elite": "software", "GTK 250": "telecommunications",
    "GTK 1000": "medical", "GTS Starter": "services", "GTS Standard": "marketing",
    "GTS Enterprise": "finance",
}

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
    out["industry"] = out["product"].map(INDUSTRY)
    assert out["industry"].notna().all(), "product missing an industry"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
